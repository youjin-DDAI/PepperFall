#!/bin/bash
# After OF-Syn VLM feature extraction (run_ofsyn_vlm.sh): 18 event-label VLM / fusion runs -> summary
cd "$(dirname "$0")/.."
PY=${PY:-python}
until grep -q ALL_DONE logs/run_ofsyn_vlm.log; do sleep 30; done
for k in 0 1 2; do
  (for mode in hidden tokmean skel_tokmean; do
     for tr in "OF-Syn" "OF-Syn FallVision"; do
       $PY train/train_event.py --mode $mode --train $tr --run A --seed $k --epochs 30 --device cuda:$k \
         > "logs/event_${mode}_${tr// /+}_s$k.log" 2>&1
     done
   done) &
done
wait
$PY eval/summarize_event.py > logs/summarize_event.log 2>&1
echo EVENT_VLM_DONE
