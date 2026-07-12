#!/usr/bin/env python3
"""RunPod を REST + SSH(paramiko) で駆動するドライバ。seg コンテナ内で実行する想定。
state/鍵は /work/runpod/.rp/ に保存（bind mount でホストへ永続）。

  up [--gpu G] [--disk N] [--image IMG]   : 鍵生成→Pod作成→SSH疎通→nvidia-smi
  status                                   : Pod 状態
  run "CMD"                                : SSH でコマンド実行（出力ストリーム）
  put LOCAL REMOTE                         : SFTP アップロード
  get REMOTE LOCAL                         : SFTP ダウンロード
  term                                     : Pod 削除（課金停止）
"""
import os, sys, json, time, urllib.request, urllib.error

RP = "/work/runpod/.rp"
KEY_PRIV = RP + "/id_rsa"
STATE = RP + "/state.json"
API = "https://rest.runpod.io/v1"


def H():
    k = os.environ["RKEY"]
    return {"Authorization": "Bearer " + k, "Content-Type": "application/json"}


def api(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, headers=H(), method=method)
    try:
        r = urllib.request.urlopen(req, timeout=60)
        t = r.read().decode()
        return r.status, (json.loads(t) if t.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:500]


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def save_state(s):
    os.makedirs(RP, exist_ok=True)
    json.dump(s, open(STATE, "w"))


def ensure_key():
    import paramiko
    os.makedirs(RP, exist_ok=True)
    if not os.path.exists(KEY_PRIV):
        k = paramiko.RSAKey.generate(2048)
        k.write_private_key_file(KEY_PRIV)
        open(KEY_PRIV + ".pub", "w").write("ssh-rsa " + k.get_base64() + " seg")
    return open(KEY_PRIV + ".pub").read().strip()


def find_ssh(pod):
    ip = pod.get("publicIp") or ""
    pm = pod.get("portMappings") or {}
    port = None
    if isinstance(pm, dict):
        port = pm.get("22") or pm.get(22)
    elif isinstance(pm, list):
        for x in pm:
            if isinstance(x, dict) and str(x.get("privatePort")) == "22":
                port = x.get("publicPort"); ip = x.get("ip", ip) or ip
    return ip, port


def establish(pid):
    for i in range(40):
        s2, p = api("GET", "/pods/" + pid)
        if isinstance(p, dict):
            ip, port = find_ssh(p)
            print(f"  [{i}] status={p.get('desiredStatus')} ip={ip} port={port}")
            if ip and port:
                st = load_state(); st.update({"pod_id": pid, "ssh_host": ip, "ssh_port": port}); save_state(st)
                try:
                    cli = ssh_connect()
                    _, out, _ = cli.exec_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true")
                    print("SSH OK:", out.read().decode().strip()); cli.close()
                    print("POD READY pod_id=" + pid); return True
                except Exception as e:
                    print("  ssh not ready:", repr(e)[:120])
        time.sleep(10)
    print("timed out; pod_id=" + pid); return False


def cmd_wait(args):
    establish(load_state()["pod_id"])


def ssh_connect():
    import paramiko
    s = load_state()
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(s["ssh_host"], port=int(s["ssh_port"]), username="root",
                key_filename=KEY_PRIV, timeout=30, banner_timeout=30, auth_timeout=30)
    return cli


def cmd_up(args):
    import paramiko
    pub = ensure_key()
    gpu = _arg(args, "--gpu", "NVIDIA GeForce RTX 4090")
    disk = int(_arg(args, "--disk", "80"))
    image = _arg(args, "--image", "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
    body = {"name": "seg-anno", "imageName": image, "gpuTypeIds": [gpu], "gpuCount": 1,
            "containerDiskInGb": disk, "volumeInGb": 0, "ports": ["22/tcp"],
            "env": {"PUBLIC_KEY": pub}}
    st, pod = api("POST", "/pods", body)
    print("create:", st, pod if isinstance(pod, str) else pod.get("id"))
    if st not in (200, 201):
        sys.exit(1)
    pid = pod["id"]
    save_state({"pod_id": pid})
    establish(pid)


def cmd_run(args):
    cli = ssh_connect()
    chan = cli.get_transport().open_session()
    chan.get_pty()
    chan.exec_command(args[0])
    import select
    while True:
        if chan.recv_ready():
            sys.stdout.write(chan.recv(4096).decode(errors="replace")); sys.stdout.flush()
        if chan.exit_status_ready() and not chan.recv_ready():
            break
        time.sleep(0.1)
    print("\n[exit]", chan.recv_exit_status())
    cli.close()


def cmd_put(args):
    cli = ssh_connect(); sf = cli.open_sftp()
    local, remote = args[0], args[1]
    t0 = time.time(); last = [0]
    def cb(done, total):
        if done - last[0] > 50_000_000:
            last[0] = done; print(f"  {done/1e6:.0f}/{total/1e6:.0f} MB")
    sf.put(local, remote, callback=cb)
    print(f"put done {time.time()-t0:.0f}s -> {remote}")
    cli.close()


def cmd_get(args):
    cli = ssh_connect(); sf = cli.open_sftp()
    sf.get(args[0], args[1]); print("got", args[1]); cli.close()


def cmd_status(args):
    s = load_state()
    st, p = api("GET", "/pods/" + s.get("pod_id", "none"))
    print(st, json.dumps(p, indent=2)[:800] if isinstance(p, dict) else p)


def cmd_term(args):
    s = load_state()
    pid = s.get("pod_id")
    if not pid:
        print("no pod"); return
    print("terminate:", *api("DELETE", "/pods/" + pid))


def _arg(args, flag, default):
    return args[args.index(flag) + 1] if flag in args else default


if __name__ == "__main__":
    cmds = {"up": cmd_up, "wait": cmd_wait, "run": cmd_run, "put": cmd_put, "get": cmd_get,
            "status": cmd_status, "term": cmd_term}
    cmds[sys.argv[1]](sys.argv[2:])
