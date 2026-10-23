#!/bin/bash
# OmniFall 16-action multiclass training (OF-Syn): ST-GCN A/B, VLM tokmean MLP, ST-GCN + VLM tokmean, seed 0-2 -> summary
cd "$(dirname "$0")/.."
PY=${PY:-python}
for k in 0 1 2; do
  ($PY train/train_event.py --label multiclass --mode stgcn --run A --seed $k --epochs 30 --device cuda:$k > logs/mc_stgcn_A_s$k.log 2>&1
   $PY train/train_event.py --label multiclass --mode stgcn --run B --seed $k --epochs 30 --device cuda:$k > logs/mc_stgcn_B_s$k.log 2>&1
   $PY train/train_event.py --label multiclass --mode tokmean --seed $k --epochs 30 --device cuda:$k > logs/mc_tokmean_s$k.log 2>&1
   $PY train/train_event.py --label multiclass --mode skel_tokmean --run A --seed $k --epochs 30 --device cuda:$k > logs/mc_skel_tokmean_s$k.log 2>&1) &
done
wait
echo MC_TRAIN_DONE
until grep -qE "XATTN_(DONE|ABORT)" logs/run_xattn_event.log; do sleep 60; done
$PY eval/summarize_event.py > logs/summarize_event.log 2>&1
echo MC_DONE
