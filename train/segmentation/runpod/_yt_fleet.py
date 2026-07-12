#!/usr/bin/env python3
"""YouTube 収集を N pod で並列実行するオーケストレータ（seg コンテナ内で実行）。
parts fleet と違い hub 不要 — 各 pod が独立に CC動画をDLし frames+parts を emit する。
クエリは collect_youtube.py 側で --num-shards/--shard によりラウンドロビン分割。

  yt_fleet create N            : 新規 N pod 作成（SSH 準備まで）
  yt_fleet dispatch [maxvid maxfr] : scripts/queries/.env 転送 → setup → run_youtube 起動
  yt_fleet poll                : 進捗（done フラグ / emit frames 数）
  yt_fleet fetch               : youtube_out_<shard>.tar.gz を /work/runpod/youtube_fleet/ に回収
  yt_fleet term                : 全 pod 削除（課金停止）

env: RKEY=RunPod APIキー, HF_TOKEN=HuggingFace（ゲート付き SAM3 用）
"""
import os, sys, json, time, io, urllib.request, urllib.error
import paramiko

RP = "/work/runpod/.rp"
KEY = RP + "/id_rsa"
FLEET = RP + "/yt_fleet.json"
API = "https://rest.runpod.io/v1"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
GPU = "NVIDIA GeForce RTX 4090"
HERE = os.path.dirname(os.path.abspath(__file__))           # /work/runpod
WORKROOT = os.path.dirname(HERE)                            # /work


def H():
    return {"Authorization": "Bearer " + os.environ["RKEY"], "Content-Type": "application/json"}


def api(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    try:
        r = urllib.request.urlopen(urllib.request.Request(API + path, data=data, headers=H(), method=method), timeout=60)
        t = r.read().decode()
        return r.status, (json.loads(t) if t.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


def pub():
    return open(KEY + ".pub").read().strip()


def find_ssh(p):
    ip = p.get("publicIp") or ""
    pm = p.get("portMappings") or {}
    port = pm.get("22") if isinstance(pm, dict) else None
    return ip, port


def sshc(host, port):
    c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=int(port), username="root", key_filename=KEY,
              timeout=30, banner_timeout=30, auth_timeout=30)
    return c


def run(c, cmd):
    ch = c.get_transport().open_session(); ch.exec_command(cmd)
    out = b""
    while True:
        if ch.recv_ready(): out += ch.recv(8192)
        if ch.exit_status_ready() and not ch.recv_ready(): break
        time.sleep(0.05)
    return ch.recv_exit_status(), out.decode(errors="replace")


def cmd_create(args):
    n = int(args[0])
    pods = []
    for k in range(n):
        st, p = api("POST", "/pods", {"name": f"yt-fleet-{k}", "imageName": IMAGE,
                    "gpuTypeIds": [GPU], "gpuCount": 1, "containerDiskInGb": 60,
                    "volumeInGb": 0, "ports": ["22/tcp"], "env": {"PUBLIC_KEY": pub()}})
        if st not in (200, 201):
            print("create fail", st, p); continue
        pods.append({"id": p["id"], "shard": k, "num_shards": n})
        print("created", p["id"], "shard", k)
    json.dump(pods, open(FLEET, "w"))
    for it in range(40):
        ready = 0
        for pd in pods:
            s, p = api("GET", "/pods/" + pd["id"])
            if isinstance(p, dict):
                ip, port = find_ssh(p)
                if ip and port:
                    pd["ip"], pd["port"] = ip, port; ready += 1
        json.dump(pods, open(FLEET, "w"))
        print(f"  ready {ready}/{len(pods)}")
        if ready == len(pods):
            break
        time.sleep(10)
    for pd in pods:
        try:
            c = sshc(pd["ip"], pd["port"]); rc, o = run(c, "echo ok"); c.close()
            pd["ssh"] = (o.strip() == "ok")
        except Exception as e:
            pd["ssh"] = False; print("ssh fail", pd["id"], repr(e)[:80])
    json.dump(pods, open(FLEET, "w"))
    print("ssh ok:", sum(1 for p in pods if p.get("ssh")), "/", len(pods))


def cmd_dispatch(args):
    maxvid = args[0] if len(args) > 0 else "150"   # 10台 x 150 = 最大 ~1500 動画
    maxfr = args[1] if len(args) > 1 else "2000"   # 10台 x 2000 = 最大 ~20000 枚（dedup で実数以下）
    pods = json.load(open(FLEET))
    hf = os.environ.get("HF_TOKEN", "")
    files = [(WORKROOT + "/collect_youtube.py", "/workspace/collect_youtube.py"),
             (HERE + "/youtube_queries.txt", "/workspace/youtube_queries.txt"),
             (HERE + "/setup_youtube.sh", "/workspace/setup_youtube.sh"),
             (HERE + "/run_youtube.sh", "/workspace/run_youtube.sh")]
    for pd in pods:
        if not pd.get("ssh"):
            continue
        c = sshc(pd["ip"], pd["port"]); sf = c.open_sftp()
        run(c, "mkdir -p /workspace")
        for loc, rem in files:
            sf.put(loc, rem)
        sf.putfo(io.BytesIO((f"HF_TOKEN={hf}\n").encode()), "/workspace/.env")
        prep = ("#!/usr/bin/env bash\ncd /workspace\nset -a; . ./.env; set +a\n"
                "bash setup_youtube.sh > setup.log 2>&1\n"
                f"bash run_youtube.sh {pd['shard']} {pd['num_shards']} {maxvid} {maxfr}\n")
        sf.putfo(io.BytesIO(prep.encode()), "/workspace/prep.sh")
        sf.close()
        run(c, "cd /workspace && setsid bash prep.sh > prep.log 2>&1 < /dev/null & echo launched")
        c.close()
        print("dispatched", pd["id"], "shard", pd["shard"], "/", pd["num_shards"])


def cmd_poll(args):
    pods = [p for p in json.load(open(FLEET)) if p.get("ssh")]
    done = 0
    for pd in pods:
        try:
            c = sshc(pd["ip"], pd["port"])
            rc, o = run(c, "cd /workspace && (test -f done_%d && echo DONE || echo run); "
                           "ls out/frames 2>/dev/null | wc -l; tail -1 proc_%d.log 2>/dev/null | tr -d '\\r'"
                           % (pd["shard"], pd["shard"]))
            c.close()
            L = o.strip().split("\n")
            st = L[0] if L else "?"
            frames = L[1] if len(L) > 1 else "?"
            tail = L[2] if len(L) > 2 else ""
            if st == "DONE":
                done += 1
            print(f"shard {pd['shard']:2d} [{pd['id']}]: {st} frames={frames}  {tail[:70]}")
        except Exception as e:
            print(pd["id"], "poll err", repr(e)[:80])
    print(f"DONE {done}/{len(pods)}")


def cmd_fetch(args):
    out = "/work/runpod/youtube_fleet"
    os.makedirs(out, exist_ok=True)
    pods = [p for p in json.load(open(FLEET)) if p.get("ssh")]
    n = 0
    for pd in pods:
        try:
            c = sshc(pd["ip"], pd["port"]); sf = c.open_sftp()
            fn = "youtube_out_%d.tar.gz" % pd["shard"]
            try:
                sf.stat("/workspace/" + fn)
                sf.get("/workspace/" + fn, out + "/" + fn); n += 1
                print("fetched", fn)
            except FileNotFoundError:
                print("not ready", pd["id"], "shard", pd["shard"])
            sf.close(); c.close()
        except Exception as e:
            print(pd["id"], "fetch err", repr(e)[:80])
    print(f"fetched {n} tarballs -> {out}  (次: python merge_youtube_fleet.py)")


def cmd_term(args):
    for pd in json.load(open(FLEET)):
        print("term", pd["id"], *api("DELETE", "/pods/" + pd["id"]))


if __name__ == "__main__":
    {"create": cmd_create, "dispatch": cmd_dispatch, "poll": cmd_poll,
     "fetch": cmd_fetch, "term": cmd_term}[sys.argv[1]](sys.argv[2:])
