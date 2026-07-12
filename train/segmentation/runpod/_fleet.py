#!/usr/bin/env python3
"""8-pod フリートで segment_parts を分散実行するオーケストレータ（seg コンテナ内で実行）。
hub = 既存 pod（state.json, 全データ所持）。新規 N pod は hub から pod 間コピーでデータ取得。
全 32 シャードを pod ごとに 4 シャードずつ割当（pod0=hub:0-3, pod_k:4k..4k+3）。

  fleet create N         : 新規 N pod 作成（SSH 準備まで）
  fleet dispatch         : 各 pod に scripts/鍵/.env 転送→hubからデータpull→setup→4ワーカー起動
  fleet poll             : 進捗（done parts / 32）
  fleet fetch            : 全 pod の parts_*.json を /work/runpod/parts/ に回収
  fleet term             : 新規 pod を全削除（hub は別途 _drive term）
"""
import os, sys, json, time, io, urllib.request, urllib.error
import paramiko

RP = "/work/runpod/.rp"
KEY = RP + "/id_rsa"
STATE = RP + "/state.json"          # hub
FLEET = RP + "/fleet.json"
API = "https://rest.runpod.io/v1"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
GPU = "NVIDIA GeForce RTX 4090"
TOTAL_SHARDS = 90                    # 30 fleet pods x WPP=3 = 90（hub はデータ供給専用、非worker）
WPP = 3                              # workers per pod（1台3プロセス並列）
PART_MIN = 64


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


def hub_info():
    s = json.load(open(STATE))
    return s["ssh_host"], int(s["ssh_port"])


def cmd_create(args):
    n = int(args[0])
    pods = []
    for k in range(n):
        st, p = api("POST", "/pods", {"name": f"seg-fleet-{k+1}", "imageName": IMAGE,
                    "gpuTypeIds": [GPU], "gpuCount": 1, "containerDiskInGb": 80,
                    "volumeInGb": 0, "ports": ["22/tcp"], "env": {"PUBLIC_KEY": pub()}})
        if st not in (200, 201):
            print("create fail", st, p); continue
        # hub は非worker（データ供給専用）。fleet pod_k が shards [k*WPP .. k*WPP+WPP-1] を担当
        pods.append({"id": p["id"], "start": k * WPP})
        print("created", p["id"], "start_shard", k * WPP)
    json.dump(pods, open(FLEET, "w"))
    # poll ssh ready
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
    # verify ssh
    for pd in pods:
        try:
            c = sshc(pd["ip"], pd["port"]); rc, o = run(c, "echo ok"); c.close()
            pd["ssh"] = (o.strip() == "ok")
        except Exception as e:
            pd["ssh"] = False; print("ssh fail", pd["id"], repr(e)[:80])
    json.dump(pods, open(FLEET, "w"))
    print("ssh ok:", sum(1 for p in pods if p.get("ssh")), "/", len(pods))


def cmd_dispatch(args):
    hip, hport = hub_info()
    pods = json.load(open(FLEET))
    hf = os.environ.get("HF_TOKEN", "")
    files = [("segment_parts.py", "/workspace/segment_parts.py"),
             ("merge_parts.py", "/workspace/merge_parts.py"),
             ("runpod/setup_parts.sh", "/workspace/setup_parts.sh"),
             ("runpod/run_shards.sh", "/workspace/run_shards.sh"),
             (KEY, "/root/.ssh/id_rsa")]
    for pd in pods:
        if not pd.get("ssh"):
            continue
        c = sshc(pd["ip"], pd["port"]); sf = c.open_sftp()
        run(c, "mkdir -p /root/.ssh /workspace")
        for loc, rem in files:
            sf.put(loc, rem)
        sf.putfo(io.BytesIO((f"HF_TOKEN={hf}\n").encode()), "/workspace/.env")
        prep = (
            "#!/usr/bin/env bash\ncd /workspace\nchmod 600 /root/.ssh/id_rsa\n"
            f"ssh -i /root/.ssh/id_rsa -p {hport} -o StrictHostKeyChecking=no root@{hip} "
            "'tar cf - -C /workspace pet_bottle' | tar xf - -C /workspace\n"
            "bash setup_parts.sh > setup.log 2>&1\n"
            f"bash run_shards.sh {TOTAL_SHARDS} {PART_MIN} {pd['start']} {WPP}\n"
            "echo PREP_DONE >> prep.log\n")
        sf.putfo(io.BytesIO(prep.encode()), "/workspace/prep.sh")
        sf.close()
        run(c, "cd /workspace && setsid bash prep.sh > prep.log 2>&1 < /dev/null & echo launched")
        c.close()
        print("dispatched", pd["id"], "shards", pd["start"], "-", pd["start"] + WPP - 1)


def cmd_poll(args):
    hip, hport = hub_info()
    targets = [("hub", hip, hport)] + [(p["id"], p["ip"], p["port"]) for p in json.load(open(FLEET)) if p.get("ssh")]
    total_done = 0
    for name, ip, port in targets:
        try:
            c = sshc(ip, port)
            rc, o = run(c, "cd /workspace && ls parts_*.json 2>/dev/null | wc -l; tail -1 prep.log 2>/dev/null; tr '\\r' '\\n' < $(ls -t w_*.log 2>/dev/null|head -1) 2>/dev/null | grep -oE '[0-9]+/[0-9]+ \\[' | tail -1")
            c.close()
            lines = o.strip().split("\n")
            done = lines[0] if lines else "?"
            total_done += int(done) if done.isdigit() else 0
            print(f"{name}: parts={done} {' '.join(lines[1:])}")
        except Exception as e:
            print(name, "poll err", repr(e)[:80])
    print(f"TOTAL parts: {total_done}/{TOTAL_SHARDS}")


def cmd_fetch(args):
    os.makedirs("/work/runpod/parts", exist_ok=True)
    hip, hport = hub_info()
    targets = [("hub", hip, hport)] + [(p["id"], p["ip"], p["port"]) for p in json.load(open(FLEET)) if p.get("ssh")]
    n = 0
    for name, ip, port in targets:
        try:
            c = sshc(ip, port); sf = c.open_sftp()
            for fn in sf.listdir("/workspace"):
                if fn.startswith("parts_") and fn.endswith(".json"):
                    sf.get("/workspace/" + fn, "/work/runpod/parts/" + fn); n += 1
            sf.close(); c.close()
        except Exception as e:
            print(name, "fetch err", repr(e)[:80])
    print(f"fetched {n} parts files -> /work/runpod/parts/")


def cmd_heal(args):
    hip, hport = hub_info()
    pods = json.load(open(FLEET))
    for pd in pods:
        if not pd.get("ssh"):
            continue
        c = sshc(pd["ip"], pd["port"])
        rc, o = run(c, "test -f /workspace/pet_bottle/annotations/instances_all_sam3merge.json && echo OK || echo NO")
        if "NO" in o:
            print(pd["id"], "re-pulling data from hub ...")
            run(c, f"cd /workspace && ssh -i /root/.ssh/id_rsa -p {hport} -o StrictHostKeyChecking=no "
                   f"root@{hip} 'tar cf - -C /workspace pet_bottle' | tar xf - -C /workspace")
        miss = []
        for s in range(pd["start"], pd["start"] + WPP):
            rc, o = run(c, f"test -f /workspace/parts_{s}.json && echo Y || echo N")
            if "N" in o:
                miss.append(s)
        for s in miss:
            run(c, f"cd /workspace && pkill -f 'shard {s} ' 2>/dev/null; setsid bash -c "
                   f"'python segment_parts.py --data-root pet_bottle --part-min {PART_MIN} "
                   f"--num-shards {TOTAL_SHARDS} --shard {s} --emit-parts parts_{s}.json > w_{s}.log 2>&1' </dev/null &")
        print(pd["id"], "relaunched missing shards", miss)
        c.close()


def cmd_term(args):
    for pd in json.load(open(FLEET)):
        print("term", pd["id"], *api("DELETE", "/pods/" + pd["id"]))


if __name__ == "__main__":
    {"create": cmd_create, "dispatch": cmd_dispatch, "poll": cmd_poll,
     "fetch": cmd_fetch, "heal": cmd_heal, "term": cmd_term}[sys.argv[1]](sys.argv[2:])
