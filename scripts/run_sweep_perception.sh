#!/usr/bin/env bash
# Seed sweep for a perception point -- arm-parametrized, protocol-overridable.
#
# ONE script for every arm of the specialist-vs-generalist study. The arm is an
# argument; the protocol is a set of defaults you CAN override for exploration.
#
#   ./run_sweep_perception.sh <model> [lr]
#
# Arm (positional):
#   ./run_sweep_perception.sh faster_rcnn           # from-scratch (LR defaults in trainer)
#   ./run_sweep_perception.sh faster_rcnn_pt 1e-3   # pretrained  (LR required by trainer)
#
# Protocol knobs (env vars, defaults = the locked study protocol). Override by
# prefixing the call -- the override then shows up in the command and the log:
#   SEEDS   (default "1 2 3 4 5 6 7 8 9 10")  space-separated; one seed is fine
#   EPOCHS  (default 25)
#   PATIENCE(default 0 = no early-stop)        evaluations, not epochs
#   MIN_DELTA (default 1e-3)                    min AP gain counted as improvement
#   POINT   (default n48_s8)                    tags the run-id prefix AND the ledger
#
# Quick 1-vs-1 overnight comparison (direction, NOT verdict -- see warning):
#   SEEDS=1 EPOCHS=10 POINT=quick ./run_sweep_perception.sh faster_rcnn_pt 1e-3
#   SEEDS=1 EPOCHS=10 POINT=quick ./run_sweep_perception.sh faster_rcnn
# POINT=quick keeps those runs in their own ledger/prefix, away from the real
# 10-seed sweep. Give BOTH arms the SAME overrides or the comparison is void.
#
# Why env-vars and not just hard-coding: the protocol is the DEFAULT, so a bare
# call always reproduces the study; an override is explicit and auditable. Only
# the model + its LR define an arm; prefix and ledger are DERIVED from model and
# POINT, so arms and exploratory runs never cross-contaminate.
#
# best.pt per seed captures the true peak within the budget; checkpoints are
# NEVER pruned -- the full distribution (mean/std over seeds) is the citable
# result, not the single best.  Full sweep ~25 ep x ~130-150 s x 10 ~= 9-10 h.
set -u  # NOT -e: a single failed seed must not abort the whole sweep.

# --- arm (the only positional parameters) ----------------------------------
MODEL=${1:?usage: run_sweep_perception.sh <model> [lr]   e.g. faster_rcnn_pt 1e-3}
LR=${2:-}                                # empty -> let the trainer pick its default

# --- protocol knobs (env override; defaults = locked study protocol) -------
: "${SEEDS:=1 2 3 4 5 6 7 8 9 10}"
: "${EPOCHS:=25}"
: "${PATIENCE:=0}"
: "${MIN_DELTA:=1e-3}"
: "${POINT:=n48_s8}"

# --- fixed infrastructure (not knobs) --------------------------------------
CONTAINER=gpu_env
SRC=/workspace/src
TRAINER=$SRC/perception/models/train_vision_model.py
CACHE=/workspace/datasets/waymo/perception_cache

# --- derived (never cross arms / runs) -------------------------------------
PREFIX=${MODEL}_${POINT}_seed
CSV=runs/sweep_${MODEL}_${POINT}.csv
LR_ARG=""; [ -n "$LR" ] && LR_ARG="--lr $LR"

dex() { docker exec -i "$CONTAINER" sh -c "$1"; }

echo "[sweep] model=$MODEL lr=${LR:-<trainer-default>} point=$POINT epochs=$EPOCHS "\
"patience=$PATIENCE seeds=[$SEEDS]"
echo "[sweep] ledger=$CSV  prefix=${PREFIX}<N>"

# Loud warning when the protocol is overridden: such runs are exploratory, both
# arms must share the override, and the project invariant stands -- N=10 seeds
# before any verdict (scratch std ~0.0048; a <0.005-0.01 gap can be seed noise).
WARN=""
[ "$EPOCHS"   != 25 ] && WARN="$WARN epochs=$EPOCHS"
[ "$PATIENCE" != 0  ] && WARN="$WARN patience=$PATIENCE"
NSEEDS=$(set -- $SEEDS; echo $#)
[ "$NSEEDS"   -lt 10 ] && WARN="$WARN seeds=$NSEEDS(<10)"
[ -n "$WARN" ] && echo "[sweep] !! PROTOCOL OVERRIDE:$WARN -- exploratory only; "\
"keep BOTH arms identical; N=10 seeds before any verdict."

# Fresh ledger for this (arm, point); any prior CSV kept as a timestamped backup
# (non-destructive -- matches the list/confirm/execute rule for data).
dex "if [ -f $CSV ]; then mv $CSV ${CSV}.bak.\$(date +%Y%m%d_%H%M%S); fi"

for S in $SEEDS; do
  RID=${PREFIX}${S}
  echo "=================== seed $S  ($RID) ==================="
  dex "rm -rf runs/$RID"   # clean dir so train_log.jsonl is not appended to
  docker exec -i "$CONTAINER" env PYTHONPATH=$SRC python "$TRAINER" \
    --model "$MODEL" --cache-root "$CACHE" \
    --epochs "$EPOCHS" --batch-size 2 --warmup-iters 500 $LR_ARG \
    --eval-ap-every 1 --patience "$PATIENCE" --min-delta "$MIN_DELTA" \
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