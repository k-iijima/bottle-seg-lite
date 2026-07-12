#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""マスクリファイン(refine_masks.py)を N-pod H100 フリートで分散実行するオーケストレータ。
_attr_fleet.py と同じ運用（hub 経由の時差 scp / nohup 起動 / heal 再配分）。

  python _refine_fleet.py create N [WORKERS] : 新規 N pod 作成（pod あたり WORKERS シャード、既定2）
  python _refine_fleet.py seed               : refine_inputs.tar を pod 0 へ SFTP 転送
  python _refine_fleet.py dispatch [EXTRA]   : スクリプト転送→データ取得→setup→worker起動
  python _refine_fleet.py poll               : 進捗
  python _refine_fleet.py heal               : 未完シャードを再割当て
  python _refine_fleet.py fetch              : refined_*.json を runpod/refined/ へ回収
  python _refine_fleet.py term               : fleet pod を全削除
"""
import io, json, os, sys, time, urllib.error, urllib.request
from pathlib import Path
import paramiko

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
RP = HERE / ".rp"
KEY = str(RP / "id_rsa")
FLEET = RP / "refine_fleet.json"
TAR = HERE / "refine_inputs.tar"
API = "https://rest.runpod.io/v1"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
# H100 優先、容量エラー時のフォールバック順
GPUS = ["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe",
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"]
DISK_GB = 50
DATA_OK = "test -f /workspace/pet_bottle/annotations/instances_all_sam3full.json"


def envval(name):
    v = os.environ.get(name)
    if v:
        return v
    for p in [HERE, *HERE.parents]:
        f = p / ".env"
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith(name + "="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit(f"{name} が .env にも環境変数にもない")


def H():
    return {"Authorization": "Bearer " + envval("RUNPOD_KEY"), "Content-Type": "application/json"}


def api(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    try:
        r = urllib.request.urlopen(urllib.request.Request(API + path, data=data, headers=H(), method=method), timeout=60)
        t = r.read().decode()
        return r.status, (json.loads(t) if t.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


def gql(query, variables=None):
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request("https://api.runpod.io/graphql?api_key=" + envval("RUNPOD_KEY"),
                                 data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        r = urllib.request.urlopen(req, timeout=60)
        return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


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


def load_fleet():
    return json.load(open(FLEET))


def save_fleet(pods):
    json.dump(pods, open(FLEET, "w"))


def cmd_create(args):
    n = int(args[0])
    workers = int(args[1]) if len(args) > 1 else 2
    pub = open(KEY + ".pub").read().strip()
    pods = load_fleet() if FLEET.exists() else []
    next_shard = max((s for p in pods for s in p["shards"]), default=-1) + 1
    MUT = """
mutation($in: PodFindAndDeployOnDemandInput) {
  podFindAndDeployOnDemand(input: $in) { id costPerHr machine { gpuDisplayName } }
}"""
    for k in range(n):
        shards = [next_shard + k * workers + j for j in range(workers)]
        pid = None
        for attempt in range(6):
            gpu = GPUS[min(attempt // 2, len(GPUS) - 1)]
            cloud = "SECURE" if attempt % 2 == 0 else "ALL"
            st, r = gql(MUT, {"in": {
                "cloudType": cloud, "gpuCount": 1, "gpuTypeId": gpu,
                "name": f"refine-fleet-{len(pods)+1}", "imageName": IMAGE,
                "containerDiskInGb": DISK_GB, "volumeInGb": 0, "ports": "22/tcp",
                "env": [{"key": "PUBLIC_KEY", "value": pub}]}})
            pod = r.get("data", {}).get("podFindAndDeployOnDemand") if isinstance(r, dict) else None
            if st == 200 and pod:
                pid = pod["id"]
                print("created", pid, "shards", shards,
                      f"${pod.get('costPerHr')}/h ({gpu}, {cloud})")
                break
            print(f"create retry {attempt+1} ({gpu}, {cloud})", st, str(r)[:150]); time.sleep(5)
        if not pid:
            print("create fail shards", shards); continue
        pods.append({"id": pid, "shards": shards})
    save_fleet(pods)
    for it in range(40):
        ready = 0
        for pd in pods:
            if pd.get("ip") and pd.get("port"):
                ready += 1; continue
            s, p = api("GET", "/pods/" + pd["id"])
            if isinstance(p, dict):
                ip, port = find_ssh(p)
                if ip and port:
                    pd["ip"], pd["port"] = ip, port; ready += 1
        save_fleet(pods)
        print(f"  ready {ready}/{len(pods)}")
        if ready == len(pods):
            break
        time.sleep(10)
    for pd in pods:
        if pd.get("ssh"):
            continue
        try:
            c = sshc(pd["ip"], pd["port"]); rc, o = run(c, "echo ok"); c.close()
            pd["ssh"] = (o.strip() == "ok")
        except Exception as e:
            pd["ssh"] = False; print("ssh fail", pd["id"], repr(e)[:80])
    save_fleet(pods)
    print("ssh ok:", sum(1 for p in pods if p.get("ssh")), "/", len(pods))


def cmd_seed(args):
    pods = load_fleet()
    hub = pods[0]
    c = sshc(hub["ip"], hub["port"]); sf = c.open_sftp()
    t0 = time.time(); last = [0]
    def cb(done, total):
        if done - last[0] > 200_000_000:
            last[0] = done
            print(f"  {done/1e6:.0f}/{total/1e6:.0f} MB  {done/1e6/(time.time()-t0):.1f} MB/s")
    sf.put(str(TAR), "/workspace/refine_inputs.tar", callback=cb)
    sf.close()
    rc, o = run(c, "cd /workspace && tar xf refine_inputs.tar && mv -f runpod/setup_refine.sh . 2>/dev/null; "
                   + DATA_OK + " && echo EXTRACT_OK")
    c.close()
    print(f"seed done {time.time()-t0:.0f}s ->", hub["id"], o.strip())


def total_shards(pods):
    return max(s for p in pods for s in p["shards"]) + 1


def shard_cmd(total, i, extra):
    return (f"python refine_masks.py --data-root pet_bottle "
            f"--num-shards {total} --shard {i} --emit refined_{i}.json {extra}".strip())


def cmd_dispatch(args):
    extra = " ".join(args)
    pods = load_fleet()
    total = total_shards(pods)
    hub = pods[0]
    hf = envval("HF_TOKEN")
    files = [(str(HERE.parent / "refine_masks.py"), "/workspace/refine_masks.py"),
             (str(HERE / "setup_refine.sh"), "/workspace/setup_refine.sh"),
             (KEY, "/root/.ssh/id_rsa")]
    for k, pd in enumerate(pods):
        if not pd.get("ssh"):
            print("skip (no ssh)", pd["id"]); continue
        c = sshc(pd["ip"], pd["port"]); sf = c.open_sftp()
        run(c, "mkdir -p /root/.ssh /workspace")
        for loc, rem in files:
            sf.put(loc, rem)
        sf.putfo(io.BytesIO(f"HF_TOKEN={hf}\n".encode()), "/workspace/.env")
        pull = "" if k == 0 else f"""
if ! {DATA_OK}; then
  sleep {k * 20}
  for try in 1 2 3 4 5; do
    scp -i /root/.ssh/id_rsa -P {hub['port']} -o StrictHostKeyChecking=no \\
        root@{hub['ip']}:/workspace/refine_inputs.tar /workspace/refine_inputs.tar && break
    rm -f /workspace/refine_inputs.tar; sleep $((try * 30))
  done
  tar xf /workspace/refine_inputs.tar -C /workspace || true
fi
"""
        launches = "\n".join(
            f"nohup bash -c '{shard_cmd(total, i, extra)} > w_{i}.log 2>&1' >/dev/null 2>&1 &"
            for i in pd["shards"])
        prep = f"""#!/usr/bin/env bash
cd /workspace
chmod 600 /root/.ssh/id_rsa
export HF_TOKEN={hf}
export HF_HUB_ENABLE_HF_TRANSFER=1
{pull}
{DATA_OK} || {{ echo NO_DATA >> prep.log; exit 1; }}
bash setup_refine.sh > setup.log 2>&1
pkill -f refine_masks.py 2>/dev/null; sleep 2
rm -f {' '.join(f'refined_{i}.json w_{i}.log' for i in pd['shards'])}
{launches}
echo PREP_DONE >> prep.log
"""
        sf.putfo(io.BytesIO(prep.replace("\r\n", "\n").encode()), "/workspace/prep.sh")
        sf.close()
        run(c, "cd /workspace && nohup bash prep.sh > prep.out 2>&1 </dev/null & echo launched")
        c.close()
        print("dispatched", pd["id"], "shards", pd["shards"])


def cmd_poll(args):
    pods = load_fleet()
    total = total_shards(pods)
    done_total = 0
    for pd in pods:
        if not pd.get("ssh"):
            print(pd["id"], "no ssh"); continue
        try:
            c = sshc(pd["ip"], pd["port"])
            rc, o = run(c, "cd /workspace && ls refined_*.json 2>/dev/null | wc -l; "
                           "tail -1 prep.log 2>/dev/null; "
                           "for f in $(ls -t w_*.log 2>/dev/null | head -2); do "
                           "echo -n \"$f \"; grep -oE '[0-9]+/[0-9]+ imgs.*img/s' $f | tail -1; done")
            c.close()
            lines = [l for l in o.strip().split("\n") if l.strip()]
            done = int(lines[0]) if lines and lines[0].strip().isdigit() else 0
            done_total += done
            print(f"{pd['id']} shards={pd['shards']}: done={done} | " + " | ".join(lines[1:]))
        except Exception as e:
            print(pd["id"], "poll err", repr(e)[:80])
    print(f"TOTAL done shards: {done_total}/{total}")


def cmd_heal(args):
    pods = load_fleet()
    total = total_shards(pods)
    extra = " ".join(args)
    with_data, missing = [], []
    for pd in pods:
        try:
            c = sshc(pd["ip"], pd["port"])
            rc, o = run(c, DATA_OK + " && echo Y || echo N")
            has_data = "Y" in o
            if has_data:
                with_data.append(pd)
            for i in pd["shards"]:
                rc, o = run(c, f"test -f /workspace/refined_{i}.json && echo DONE || "
                               f"(pgrep -f '[r]efine_masks.py.*--shard {i} ' >/dev/null && echo RUN || echo MISS)")
                st = o.strip().split()[-1] if o.strip() else "?"
                if st == "MISS":
                    missing.append(i)
                print(pd["id"], "shard", i, st, "" if has_data else "(no data)")
            c.close()
        except Exception as e:
            print(pd["id"], "dead:", repr(e)[:80])
            missing.extend(pd["shards"])
    if not missing:
        print("nothing to heal"); return
    if not with_data:
        print("no pods with data!"); return
    for k, i in enumerate(missing):
        pd = with_data[k % len(with_data)]
        c = sshc(pd["ip"], pd["port"])
        cmdl = shard_cmd(total, i, extra)
        run(c, "cd /workspace && nohup bash -c '"
               f"while [ $(pgrep -cf \"[r]efine_masks.py\") -ge 3 ]; do sleep 60; done; "
               f"{cmdl} > w_{i}.log 2>&1' >/dev/null 2>&1 & echo queued")
        c.close()
        if i not in pd["shards"]:
            pd["shards"].append(i)
        print("healed: shard", i, "->", pd["id"])
    save_fleet(pods)


def cmd_fetch(args):
    out = HERE / "refined"; out.mkdir(exist_ok=True)
    n = 0
    for pd in load_fleet():
        if not pd.get("ssh"):
            continue
        try:
            c = sshc(pd["ip"], pd["port"]); sf = c.open_sftp()
            for fn in sf.listdir("/workspace"):
                if fn.startswith("refined_") and fn.endswith(".json"):
                    sf.get("/workspace/" + fn, str(out / fn)); n += 1
            sf.close(); c.close()
        except Exception as e:
            print(pd["id"], "fetch err", repr(e)[:80])
    print(f"fetched {n} refined files -> {out}")


def cmd_term(args):
    for pd in load_fleet():
        print("term", pd["id"], *api("DELETE", "/pods/" + pd["id"]))


def cmd_status(args):
    st, r = api("GET", "/pods")
    pods = r if isinstance(r, list) else r.get("pods", r)
    print(json.dumps(pods, indent=1)[:2000] if not isinstance(pods, list) else
          "\n".join(f"{p['id']} {p.get('name')} {p.get('desiredStatus')} ${p.get('costPerHr')}/h" for p in pods) or "no pods")


if __name__ == "__main__":
    {"create": cmd_create, "seed": cmd_seed, "dispatch": cmd_dispatch, "poll": cmd_poll,
     "heal": cmd_heal, "fetch": cmd_fetch, "term": cmd_term, "status": cmd_status}[sys.argv[1]](sys.argv[2:])
