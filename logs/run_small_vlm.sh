#!/bin/bash
# Smaller-VLM rerun: tokmean features with Qwen2.5-VL-3B (GPU 2) and Qwen3-VL-2B (GPU 0), then
# multiclass concat / multiclass tokmean / binary-event concat on OF-Syn, seeds 0-2 -> summary
cd "$(dirname "$0")/.."
PY=${PY:-python}

chain() {  # $1 model, $2 tag, $3 gpu
  for e in CAUCAFall OF-Syn; do
    $PY data/preprocess/vlm_tokens.py --ext $e --split test $([ $e = OF-Syn ] && echo val train) --mean-only \
      --model $1 --tag $2 --batch 16 --device cuda:$3 > logs/feat_$2_$e.log 2>&1 || { echo "FEAT_FAIL $2 $e"; return; }
  done
  $PY data/preprocess/vlm_tokens.py --split test --mean-only --model $1 --tag $2 --batch 16 --device cuda:$3 \
    > logs/feat_$2_FallVision.log 2>&1 && $PY data/preprocess/vlm_tokens.py --split test --merge-only --tag $2 \
    >> logs/feat_$2_FallVision.log 2>&1 || { echo "FEAT_FAIL $2 FallVision"; return; }
  echo "FEAT_DONE $2"
  for k in 0 1 2; do
    ($PY train/train_event.py --vlm-tag $2 --label multiclass --mode skel_tokmean --run A --seed $k --epochs 30 --device cuda:$3 > logs/small_$2_mc_skel_s$k.log 2>&1
     $PY train/train_event.py --vlm-tag $2 --label multiclass --mode tokmean --seed $k --epochs 30 --device cuda:$3 > logs/small_$2_mc_tok_s$k.log 2>&1
     $PY train/train_event.py --vlm-tag $2 --mode skel_tokmean --run A --seed $k --epochs 30 --device cuda:$3 > logs/small_$2_ev_skel_s$k.log 2>&1) &
  done
  wait
  echo "TRAIN_DONE $2"
}

chain Qwen/Qwen2.5-VL-3B-Instruct q25-3b 2 &
chain Qwen/Qwen3-VL-2B-Instruct q3-2b 0 &
wait
echo SMALL_VLM_DONE
