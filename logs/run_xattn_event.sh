#!/bin/bash
# After token extraction: event-label cross-attention, 2 variants x 3 seeds (OF-Syn+FallVision, Run A) -> summary
cd "$(dirname "$0")/.."
PY=${PY:-python}
for k in 0 1 2; do
  (for mode in skel_xattn skel_xattn_sm; do
     $PY train/train_event.py --mode $mode --train OF-Syn FallVision --run A --seed $k --epochs 30 --device cuda:$k \
       > "logs/event_${mode}_OF-Syn+FallVision_s$k.log" 2>&1
   done) &
done
wait
$PY eval/summarize_event.py > logs/summarize_event.log 2>&1
echo XATTN_DONE
