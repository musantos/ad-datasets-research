#!/usr/bin/env bash
# Seed sweep for the N=48 / stride-8 perception point.
#
# Fixed epochs, NO early-stop (--patience 0): every seed is compared at equal
# compute, and best.pt per seed captures the true peak within the budget --
# avoiding the run-1 trap where a late recovery (peak @ e18, after a dip) would
# be cut by early-stop. One ledger row per seed -> ranking is a sort, not a
# grep across 10 logs. Checkpoints are NEVER pruned: the full distribution
# (mean/std over seeds) is the citable result, not the single best.
#
# ~25 epochs x ~150 s x 10 seeds ~= 10 h wall -- an overnight job.
set -u  # NOT -e: a single failed seed must not abort the whole sweep.

CONTAINER=gpu_env
SRC=/workspace/src
TRAINER=$SRC/perception/models/train_vision_model.py
CACHE=/workspace/datasets/waymo/perception_cache
EPOCHS=25
SEEDS="1 2 3 4 5 6 7 8 9 10"
CSV=runs/sweep_n48_s8.csv
PREFIX=faster_rcnn_n48_s8_seed

dex() { docker exec -i "$CONTAINER" sh -c "$1"; }

# Fresh ledger for this sweep; any prior CSV is kept as a timestamped backup
# (non-destructive -- matches the list/confirm/execute rule for data).
dex "if [ -f $CSV ]; then mv $CSV ${CSV}.bak.\$(date +%Y%m%d_%H%M%S); fi"

for S in $SEEDS; do
  RID=${PREFIX}${S}
  echo "=================== seed $S  ($RID) ==================="
  dex "rm -rf runs/$RID"   # clean dir so train_log.jsonl is not appended to
  docker exec -i "$CONTAINER" env PYTHONPATH=$SRC python "$TRAINER" \
    --model faster_rcnn --cache-root "$CACHE" \
    --epochs $EPOCHS --batch-size 2 --warmup-iters 500 \
    --eval-ap-every 1 --patience 0 \
    --seed "$S" --run-id "$RID" --sweep-csv "$CSV" \
    || echo "[sweep] seed $S FAILED (rc $?) -- continuing"
done

echo
echo "=================== RANKING (best AP, desc) ==================="
# header, then rows sorted by column 3 (best_ap) descending.
# Crashed seeds leave no row (visible as a gap); a completed-but-no-eval run
# would show 'nan' and sink to the bottom.
dex "head -1 $CSV; tail -n +2 $CSV | sort -t, -k3,3 -gr"
echo
echo "CSV: $CSV   |   checkpoints: runs/${PREFIX}<N>/best.pt (all kept)"
