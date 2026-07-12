#!/usr/bin/env python3
"""遅いDL等で詰まった pod を、今ある動画のまま SAM3 処理フェーズへ強制移行させる。
DLループ/yt-dlp を止め、.part を掃除し、process→tar→done を detached 起動する。

  python _yt_force_process.py <pod_id> [max_frames=2000]
env: 不要（pod 側 /workspace/.env の HF_TOKEN を使用）
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _yt_fleet as F

pod_id = sys.argv[1]
maxfr = sys.argv[2] if len(sys.argv) > 2 else "2000"
pods = json.load(open(F.FLEET))
p = [x for x in pods if x["id"] == pod_id][0]
shard = p["shard"]

remote = (
    "cd /workspace; "
    "pkill -9 -f run_youtube; pkill -9 -f 'collect_youtube.py download'; pkill -9 -f yt-dlp; sleep 2; "
    "find runpod/youtube/videos -name '*.part' -delete; rm -f done_%d; " % shard +
    "nohup bash -c '"
    "cd /workspace; set -a; . ./.env; set +a; "
    f"python collect_youtube.py process --emit-dir out --shard {shard} "
    f"--max-frames {maxfr} --fps 0.5 --bottle-score 0.45 --dedup-thresh 0.88 > proc_{shard}.log 2>&1; "
    f"cd out && tar czf /workspace/youtube_out_{shard}.tar.gz parts_{shard}.json frames; "
    f"echo DONE > /workspace/done_{shard}"
    "' >/dev/null 2>&1 &"
)

c = F.sshc(p["ip"], p["port"])
F.run(c, remote)
import time
time.sleep(8)
alive = F.run(c, 'ps aux | grep -c "[c]ollect_youtube.py process"')
print(f"pod {pod_id} shard {shard}: force-process launched (maxfr={maxfr}) proc_alive={alive[1].strip()}")
c.close()
