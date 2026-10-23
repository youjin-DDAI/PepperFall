#!/bin/bash
# Control: train on OF-Syn (+FallVision) with video labels (ST-GCN, Run A/B, seed 0-2). Starts after VLM extraction; regenerates summary at end
cd "$(dirname "$0")/.."
PY=${PY:-python}
until grep -q ALL_DONE logs/run_ofsyn_vlm.log; do sleep 30; done
for k in 0 1 2; do
  (for run in A B; do
     for tr in "OF-Syn" "OF-Syn FallVision"; do
       $PY train/train_event.py --label video --train $tr --run $run --seed $k --epochs 30 --device cuda:$k \
         > "logs/videolabel_${tr// /+}_run${run}_s$k.log" 2>&1
     done
   done) &
done
wait
until grep -q EVENT_VLM_DONE logs/run_event_vlm.log; do sleep 30; done
$PY eval/summarize_event.py > logs/summarize_event.log 2>&1
echo VIDEOLABEL_DONE
