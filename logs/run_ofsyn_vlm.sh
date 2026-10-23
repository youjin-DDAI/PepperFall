#!/bin/bash
cd "$(dirname "$0")/.."
PY=${PY:-python}
for k in 0 1 2; do
  $PY data/preprocess/vlm_tokens.py --ext OF-Syn --mean-only --shard $k --num-shards 3 --device cuda:$k > logs/ofsyn_vlm_shard$k.log 2>&1 &
done
wait
$PY data/preprocess/vlm_tokens.py --ext OF-Syn --merge-only > logs/ofsyn_vlm_merge.log 2>&1 || exit 1
# Motion-removed input (last frame repeated): test only, single shard -> auto merge
$PY data/preprocess/vlm_tokens.py --ext OF-Syn --mean-only --split test --frames last --device cuda:0 > logs/ofsyn_vlm_last.log 2>&1 &
$PY data/preprocess/vlm_tokens.py --ext GMDCSA24 --mean-only --split test --frames last --device cuda:1 > logs/gmd_vlm_last.log 2>&1 &
$PY data/preprocess/vlm_tokens.py --ext CAUCAFall --mean-only --split test --frames last --device cuda:2 > logs/cauc_vlm_last.log 2>&1 &
wait
echo ALL_DONE
