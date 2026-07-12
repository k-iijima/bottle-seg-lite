#!/usr/bin/env python3
"""完了済み idle pod を再活用する。既存クエリの深掘り(playlist 51件目以降)を
download-archive で取得し、全動画を無制限で再処理（parts/frames/tar を上書き＝包括版に）。

  python _yt_extend.py <pod_id> [spq=150] [maxvid=300] [maxfr=0]
env: 不要（pod 側 /workspace/.env の HF_TOKEN を使用）
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _yt_fleet as F

pod_id = sys.argv[1]
spq = sys.argv[2] if len(sys.argv) > 2 else "150"
maxvid = sys.argv[3] if len(sys.argv) > 3 else "300"
maxfr = sys.argv[4] if len(sys.argv) > 4 else "0"

pods = json.load(open(F.FLEET))
p = [x for x in pods if x["id"] == pod_id][0]
shard, n = p["shard"], p["num_shards"]

c = F.sshc(p["ip"], p["port"])
# 最新スクリプトを再送（SPQ 引数対応版）
sf = c.open_sftp()
sf.put(F.WORKROOT + "/collect_youtube.py", "/workspace/collect_youtube.py")
sf.put(F.HERE + "/run_youtube.sh", "/workspace/run_youtube.sh")
sf.put(F.HERE + "/youtube_queries.txt", "/workspace/youtube_queries.txt")
sf.close()


def run(cmd):
    ch = c.get_transport().open_session(); ch.exec_command(cmd)
    o = b""
    while True:
        if ch.recv_ready(): o += ch.recv(8192)
        if ch.exit_status_ready() and not ch.recv_ready(): break
        time.sleep(0.05)
    return o.decode(errors="replace").strip()


# done 解除 → nohup で run_youtube.sh を深掘りパラメータ付きで再起動（stdout 切り離し）
run("cd /workspace && rm -f done_%d" % shard)
launch = ("cd /workspace && nohup bash -c 'set -a; . ./.env; set +a; "
          "bash run_youtube.sh %d %d %s %s %s' > extend_%d.log 2>&1 &"
          % (shard, n, maxvid, maxfr, spq, shard))
run(launch)
time.sleep(8)
alive = run('ps aux | grep -c "[c]ollect_youtube.py download"')
print(f"pod {pod_id} shard {shard}: extend launched (spq={spq} maxvid={maxvid} maxfr={maxfr}) dl_alive={alive}")
c.close()
