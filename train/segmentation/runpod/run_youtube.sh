#!/usr/bin/env bash
# Pod 上で「CC動画DL → フレーム抽出 → SAM3 bottle判定 → CLIP埋め込み付きで emit → tar」。
# 使い方: HF_TOKEN=hf_xxx bash run_youtube.sh <shard> <num_shards> [max_videos] [max_frames] [extra args...]
set -e
cd /workspace
# MAXVID/MAXFR は <=0 で無制限。SPQ=search-per-query(クエリ毎の取得上限=ページ深さ)。
# 近接特徴量は dedup(--dedup-thresh) でスキップされるので実フレーム数は上限以下になる。
SHARD=${1:-0}; N=${2:-1}; MAXVID=${3:-150}; MAXFR=${4:-2000}; SPQ=${5:-50}
shift $(( $# < 5 ? $# : 5 )) || true
: "${HF_TOKEN:?HF_TOKEN を環境変数で渡してください}"

python collect_youtube.py download \
  --queries-file youtube_queries.txt --shard "$SHARD" --num-shards "$N" \
  --max-videos "$MAXVID" --search-per-query "$SPQ" --max-filesize-mb 250 > dl_${SHARD}.log 2>&1

python collect_youtube.py process \
  --emit-dir out --shard "$SHARD" --max-frames "$MAXFR" \
  --fps 0.5 --bottle-score 0.45 --dedup-thresh 0.88 "$@" > proc_${SHARD}.log 2>&1

cd out && tar czf /workspace/youtube_out_${SHARD}.tar.gz parts_${SHARD}.json frames
echo DONE > /workspace/done_${SHARD}
echo "[run_youtube] shard ${SHARD} done -> youtube_out_${SHARD}.tar.gz"
