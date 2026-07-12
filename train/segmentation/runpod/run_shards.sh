#!/usr/bin/env bash
# Pod 上で segment_parts を N ワーカー並列起動（setsid で完全デタッチ）。
# 使い方: bash run_shards.sh [N=4] [part_min=96]
cd /workspace
N=${1:-4}        # 総シャード数
PMIN=${2:-96}
START=${3:-0}    # このPodが担当する開始シャード
COUNT=${4:-$N}   # このPodが担当するシャード数（=同時ワーカー数）
pkill -f segment_parts.py 2>/dev/null || true
sleep 3
for ((j=0;j<COUNT;j++)); do
  i=$((START+j))
  setsid bash -c "python segment_parts.py --data-root pet_bottle --part-min ${PMIN} --num-shards ${N} --shard ${i} --emit-parts parts_${i}.json > w_${i}.log 2>&1" </dev/null &
done
sleep 6
echo "launched shards ${START}..$((START+COUNT-1)) of ${N} (part_min=${PMIN})"
echo "running procs: $(pgrep -fc 'segment_parts.py')"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
