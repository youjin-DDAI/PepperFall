#!/bin/bash
# Full OF-Syn patch tokens (for cross-attention). 6 shards, 2 sequential per GPU -> merge
cd "$(dirname "$0")/.."
PY=${PY:-python}
for k in 0 1 2; do
  ($PY data/preprocess/vlm_tokens.py --ext OF-Syn --shard $k --num-shards 6 --device cuda:$k > logs/ofsyn_tok_shard$k.log 2>&1
   $PY data/preprocess/vlm_tokens.py --ext OF-Syn --shard $((k+3)) --num-shards 6 --device cuda:$k > logs/ofsyn_tok_shard$((k+3)).log 2>&1) &
done
wait
$PY data/preprocess/vlm_tokens.py --ext OF-Syn --merge-only > logs/ofsyn_tok_merge.log 2>&1 && echo TOKENS_DONE || echo TOKENS_FAILED
