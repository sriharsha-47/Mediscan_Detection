#!/usr/bin/env bash
# Reproduces every table and figure. Env knobs: SEEDS, EPOCHS, RES, QUICK=1 (smoke test), TUNE_N
set -euo pipefail
SEEDS=${SEEDS:-"0 1 2"}; EPOCHS=${EPOCHS:-20}; RES=${RES:-224}; TUNE_N=${TUNE_N:-12}
Q=""; if [ "${QUICK:-0}" = "1" ]; then Q="--quick"; TUNE_N=3; fi
PY=${PY:-python}
C="--seeds $SEEDS --epochs $EPOCHS --res $RES $Q"

echo "== PART 1 =="
$PY audit.py
$PY split.py
$PY shortcut.py
$PY baselines.py
$PY arch_report.py
for m in resnet18 mobilenetv3; do $PY train.py --model $m --tag $m $C; done
$PY train.py --model arogyanet        --tag arogyanet        $C
$PY train.py --model arogyanet_nores  --tag arogyanet_nores  $C
$PY train.py --model arogyanet_nose   --tag arogyanet_nose   $C
$PY train.py --model arogyanet_meta   --tag arogyanet_meta   $C
$PY train.py --model arogyanet --tag arogyanet_res96     --seeds $SEEDS --epochs $EPOCHS --res 96 $Q
$PY train.py --model arogyanet --tag arogyanet_aug_none  --aug none  $C
$PY train.py --model arogyanet --tag arogyanet_aug_naive --aug naive $C
$PY aggregate.py

echo "== PART 2 =="
$PY tune.py --n $TUNE_N --res $RES $Q
$PY train.py --model arogyanet --tag arogyanet_tuned       --hparams results/best_hparams.json $C
$PY train.py --model arogyanet --tag arogyanet_tuned_cw    --hparams results/best_hparams.json --pos_weight auto $C
$PY train.py --model arogyanet --tag arogyanet_tuned_clahe --hparams results/best_hparams.json --preproc clahe $C
$PY threshold_cost.py
$PY ceo_numbers.py
$PY cv_eval.py --epochs $EPOCHS --res $RES $Q
$PY ensemble.py
$PY embeddings.py
$PY make_final.py
$PY reject_test.py
$PY aggregate.py
# The external dataset is touched exactly once, here:
$PY external_eval.py
$PY build_report.py
echo "DONE. See results/, figures/, reports/"
