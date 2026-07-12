#!/usr/bin/env python3
"""yt_fleet の特定 shard の pod だけを作り直す（他 pod には触れない）。
403 等で停滞した1台を新 IP のpodへ置き換える用途。

  python _yt_recreate.py <shard> [max_videos=150] [max_frames=2000]
env: RKEY, HF_TOKEN
"""
import io, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _yt_fleet as F

shard = int(sys.argv[1])
maxvid = sys.argv[2] if len(sys.argv) > 2 else "150"
maxfr = sys.argv[3] if len(sys.argv) > 3 else "2000"

fleet = json.load(open(F.FLEET))
entry = next((p for p in fleet if p["shard"] == shard), None)
num_shards = entry["num_shards"] if entry else 10

# 1) 旧pod削除
if entry:
    print("delete old", entry["id"], *F.api("DELETE", "/pods/" + entry["id"]))

# 2) 新pod作成
st, p = F.api("POST", "/pods", {"name": f"yt-fleet-{shard}r", "imageName": F.IMAGE,
              "gpuTypeIds": [F.GPU], "gpuCount": 1, "containerDiskInGb": 60,
              "volumeInGb": 0, "ports": ["22/tcp"], "env": {"PUBLIC_KEY": F.pub()}})
if st not in (200, 201):
    print("create fail", st, p); sys.exit(1)
new = {"id": p["id"], "shard": shard, "num_shards": num_shards}
print("created", new["id"], "shard", shard)

# 3) SSH 待ち
for it in range(40):
    s, pd = F.api("GET", "/pods/" + new["id"])
    if isinstance(pd, dict):
        ip, port = F.find_ssh(pd)
        if ip and port:
            new["ip"], new["port"] = ip, port
            try:
                c = F.sshc(ip, port); rc, o = F.run(c, "echo ok"); c.close()
                if o.strip() == "ok":
                    new["ssh"] = True; print("  ssh ready", ip, port); break
            except Exception as e:
                print("  ssh wait", repr(e)[:60])
    time.sleep(10)
if not new.get("ssh"):
    print("ssh not ready; aborting"); sys.exit(1)

# 4) この pod にだけ dispatch
hf = os.environ.get("HF_TOKEN", "")
files = [(F.WORKROOT + "/collect_youtube.py", "/workspace/collect_youtube.py"),
         (F.HERE + "/youtube_queries.txt", "/workspace/youtube_queries.txt"),
         (F.HERE + "/setup_youtube.sh", "/workspace/setup_youtube.sh"),
         (F.HERE + "/run_youtube.sh", "/workspace/run_youtube.sh")]
c = F.sshc(new["ip"], new["port"]); sf = c.open_sftp()
F.run(c, "mkdir -p /workspace")
for loc, rem in files:
    sf.put(loc, rem)
sf.putfo(io.BytesIO((f"HF_TOKEN={hf}\n").encode()), "/workspace/.env")
prep = ("#!/usr/bin/env bash\ncd /workspace\nset -a; . ./.env; set +a\n"
        "bash setup_youtube.sh > setup.log 2>&1\n"
        f"bash run_youtube.sh {shard} {num_shards} {maxvid} {maxfr}\n")
sf.putfo(io.BytesIO(prep.encode()), "/workspace/prep.sh")
sf.close()
F.run(c, "cd /workspace && setsid bash prep.sh > prep.log 2>&1 < /dev/null & echo launched")
c.close()
print("dispatched", new["id"], "shard", shard, "/", num_shards)

# 5) fleet.json 更新
fleet = [p for p in fleet if p["shard"] != shard] + [new]
fleet.sort(key=lambda x: x["shard"])
json.dump(fleet, open(F.FLEET, "w"))
print("fleet.json updated")
