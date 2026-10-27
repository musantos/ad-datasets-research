#!/usr/bin/env bash
# plot_comparison.sh -- post-run plotting orchestrator (decoupled from training).
#
# The training pipeline (run_perception_pipeline.sh) is opportunistic: it packs
# whatever arms fit on each GPU and only writes DATA (ledgers, train_log.jsonl,
# predictions.json). It does NOT decide comparisons -- a single invocation is
# pinned to one GPU and cannot see the other's experiment, so any comparison it
# draws is accidental. THIS script is where a comparison is chosen, after the
# fact, across whatever experiment folders hold the arms -- regardless of which
# GPU trained them.
#
# It is a thin fan-out: given one arm list, it calls the four plot scripts
# (scaling + curves + PR + confusion), which derive name/colour/marker from the
# model key via model_style.py. No GPU, no training, no inference.
#
# Arms are model-level: "exp_id:model" (e.g. cross_g0:retinanet). Scaling reads
# them directly (N from STEPS); the per-point panels (curves/PR/confusion) append
# _n<N>_<tag> per step automatically and write into OUT/n<N>_<tag>/.
#
# Overrides (optional, positional to ARMS, space-separated):
#   NAMES="RetinaNet SSD"        -> --arm-name per arm
#   COLORS="blue orange"         -> --arm-color per arm  (e.g. the clean 2-arm azul×laranja)
#   LINESTYLES="solid dashed"    -> --arm-linestyle per arm
#
# Usage:
#   ARMS="cross_g0:retinanet cross_g1:retinanet_pt cross_g1:ssd cross_g0:ssd_pt" \
#   STEPS="6 24 96" TAG=s8 OUT=/workspace/runs/plots_cross \
#     ./plot_comparison.sh
#
#   # just the 2-arm pair with the strong blue×orange contrast:
#   ARMS="cross_g0:retinanet cross_g1:retinanet_pt" COLORS="blue orange" \
#   STEPS="6 24 96" OUT=/workspace/runs/plots_pair ./plot_comparison.sh
set -u

# ---- parameters (env-overridable, same idiom as the training pipeline) ------
: "${ARMS:?set ARMS=\"exp_id:model ...\" (space-separated, model-level)}"
: "${RUNS_BASE:=/workspace/runs}"
: "${STEPS:=6 24 96}"
: "${TAG:=s8}"
: "${OUT:=$RUNS_BASE/plots_comparison}"

# stage toggles
: "${DO_SCALING:=1}"
: "${DO_CURVES:=1}"
: "${DO_PR:=1}"
: "${DO_CONFUSION:=1}"

# per-point knobs
: "${IOU:=0.5}"
: "${SCORE_THRESH:=0.5}"
: "${FORMATS:=png,svg}"
: "${DARK:=0}"

# optional per-arm overrides (positional to ARMS)
: "${NAMES:=}"
: "${COLORS:=}"
: "${LINESTYLES:=}"

# ---- script locations (this dir) -------------------------------------------
HERE="$(cd "$(dirname "$0")" && pwd)"
PLOT_SCALING=$HERE/plot_scaling.py
PLOT_CURVES=$HERE/plot_curves.py
PLOT_PR=$HERE/plot_pr.py
PLOT_CONFUSION=$HERE/plot_confusion.py

# If PYTHONPATH is needed for eval_ap's sibling imports, inherit it from the env.
PY="${PYTHON:-python}"
DARK_ARG=""; [ "$DARK" = 1 ] && DARK_ARG="--dark"

# ---- build the shared flag lists -------------------------------------------
read -r -a ARM_ARR <<< "$ARMS"

# override flags, expanded positionally; absent lists contribute nothing.
build_overrides() {  # emits --arm-name/.../--arm-color flags on stdout
  local -a names cols lss; read -r -a names <<< "$NAMES"
  read -r -a cols <<< "$COLORS"; read -r -a lss <<< "$LINESTYLES"
  local i
  for i in "${!names[@]}"; do printf ' --arm-name %q' "${names[$i]}"; done
  for i in "${!cols[@]}";  do printf ' --arm-color %q' "${cols[$i]}"; done
  for i in "${!lss[@]}";   do printf ' --arm-linestyle %q' "${lss[$i]}"; done
}
OVERRIDES="$(build_overrides)"

# model-level --arm flags (scaling)
SCALE_ARMS=(); for a in "${ARM_ARR[@]}"; do SCALE_ARMS+=( --arm "$a" ); done

mkdir -p "$OUT"
echo "[cmp] arms=[$ARMS] steps=[$STEPS] tag=$TAG -> $OUT"
echo "[cmp] stages: scaling=$DO_SCALING curves=$DO_CURVES pr=$DO_PR confusion=$DO_CONFUSION"
[ -n "$OVERRIDES" ] && echo "[cmp] overrides:$OVERRIDES"

# ---- scaling (one figure over all N) ---------------------------------------
if [ "$DO_SCALING" = 1 ]; then
  echo "[cmp] scaling -> $OUT"
  # shellcheck disable=SC2086
  "$PY" "$PLOT_SCALING" --runs-base "$RUNS_BASE" --steps "$STEPS" --tag "$TAG" \
    --out-dir "$OUT" "${SCALE_ARMS[@]}" $OVERRIDES $DARK_ARG \
    || echo "[cmp] scaling FAILED"
fi

# ---- per-point panels (curves/PR/confusion), one sub-dir per step ----------
for N in $STEPS; do
  POINT="n${N}_${TAG}"
  PT_ARMS=(); for a in "${ARM_ARR[@]}"; do PT_ARMS+=( --arm "${a}_${POINT}" ); done
  POUT="$OUT/$POINT"; mkdir -p "$POUT"

  if [ "$DO_CURVES" = 1 ]; then
    echo "[cmp] curves $POINT -> $POUT"
    # shellcheck disable=SC2086
    "$PY" "$PLOT_CURVES" --runs-base "$RUNS_BASE" "${PT_ARMS[@]}" \
      --out-dir "$POUT" --formats "$FORMATS" $OVERRIDES $DARK_ARG \
      || echo "[cmp] curves FAILED $POINT"
  fi
  if [ "$DO_PR" = 1 ]; then
    echo "[cmp] pr $POINT -> $POUT"
    # shellcheck disable=SC2086
    "$PY" "$PLOT_PR" --runs-base "$RUNS_BASE" "${PT_ARMS[@]}" \
      --iou "$IOU" --out-dir "$POUT" --formats "$FORMATS" $OVERRIDES $DARK_ARG \
      || echo "[cmp] pr FAILED $POINT"
  fi
  if [ "$DO_CONFUSION" = 1 ]; then
    echo "[cmp] confusion $POINT -> $POUT"
    # shellcheck disable=SC2086
    "$PY" "$PLOT_CONFUSION" --runs-base "$RUNS_BASE" "${PT_ARMS[@]}" \
      --iou "$IOU" --score-thresh "$SCORE_THRESH" --out-dir "$POUT" \
      --formats "$FORMATS" $OVERRIDES $DARK_ARG \
      || echo "[cmp] confusion FAILED $POINT"
  fi
done

echo "[cmp] done -> $OUT"
