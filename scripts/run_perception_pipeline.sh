#!/usr/bin/env bash
# run_perception_pipeline.sh  (v3 -- full auto: train + inference + plots + video)
# Data-scaling ladder orchestrator for the 2D mono-camera perception A/B study
# (pretrained+FrozenBN vs from-scratch+GN). ONE self-contained file.
#
# Per STEP N in STEPS (N = segment count = ladder x-axis):
#   1. derive --limit-train from the LIVE train manifest (first N segments ==
#      first K frames exactly: 0 empty records + segment-contiguous order);
#   2. TRAIN both arms (SEEDS seeds, EPOCHS epochs, snapshots at SAVE_EPOCHS),
#      COCO-AP on the FIXED full val every epoch (val is never limited);
#   3. INFER per seed -> infer_val/predictions.json (lean, for PR/confusion);
#   4. PLOTS per step (curves + PR + confusion) -> PLOTS_DIR/<point>/;
#   5. VIDEO per step: pick the MEDIAN seed of each arm automatically, dump PNGs
#      for those two, render the A-vs-B comparison -> VIDEO_DIR/<point>/;
#   6. ntfy status (host-side) with ETA, and a ntfy ping on any seed failure.
#   At the end: AP x N scaling plot -> PLOTS_DIR/.
# Stages 3-5 are NON-FATAL: a plot/video error never costs the training.
#
# MODES:
#   full ladder :  ./run_perception_pipeline.sh
#   single step :  POINT=n6_s8 LIMIT_TRAIN=150 ./run_perception_pipeline.sh --point faster_rcnn_pt 1e-3
#   smoke test  :  SMOKE=1 ./run_perception_pipeline.sh   (1 seed, 3 ep, step 6)
#
# EVERYTHING below is an env-overridable parameter; defaults = the locked protocol.
set -u  # NOT -e: one failed seed/step must not abort the whole ladder.

# ---- experiment parameters -------------------------------------------------
: "${STEPS:=6 12 24 48 96}"               # ladder steps = segment counts (x-axis)
: "${SEEDS:=1 2 3 4 5 6 7 8 9 10}"
: "${EPOCHS:=32}"
: "${PATIENCE:=0}"                        # 0 = no early-stop (ladder epoch parity)
: "${MIN_DELTA:=1e-3}"
: "${WARMUP:=500}"                        # LR warmup iters; trainer clamps to <=1 epoch
: "${SAVE_EPOCHS:=8,16,32}"               # weights-only snapshots (comma, 1-based)
: "${SCORE_THRESH:=0.05}"                 # inference score thresh (same both arms)
: "${STRIDE_TAG:=s8}"                     # LABEL only; real stride is set at extraction
: "${ARMS_SPEC:=pretrained:faster_rcnn_pt:1e-3 from-scratch:faster_rcnn:}"

# ---- stage toggles ---------------------------------------------------------
: "${DO_INFER:=1}"                        # predictions.json per seed (PR/confusion)
: "${DO_PLOTS:=1}"                        # curves + PR + confusion per step, + scaling
: "${DO_VIDEO:=1}"                        # median-seed A-vs-B comparison video per step

# ---- video parameters ------------------------------------------------------
: "${DUMP_SCORE_THRESH:=0.05}"            # dump keeps >= this; video filters higher
: "${VIDEO_SCORE_THRESH:=0.5}"            # display threshold in the video
: "${VIDEO_LAYOUT:=3up}"                  # 3up | dual-overlay
: "${VIDEO_FPS:=5}"

# ---- output directories (container paths) ----------------------------------
: "${PLOTS_DIR:=/workspace/runs/plots}"
: "${VIDEO_DIR:=/workspace/runs/videos}"

# ---- notifications (host-side; empty topic = silent) -----------------------
: "${NTFY_TOPIC:=}"
: "${NTFY_URL:=https://ntfy.sh}"

# ---- rough ETA calibration (measured n48: ~124-146 s/ep over 587 it) -------
: "${S_PER_IT:=0.23}"
: "${OVERHEAD:=1.1}"

# ---- smoke overrides -------------------------------------------------------
: "${SMOKE:=0}"
if [ "$SMOKE" = 1 ]; then
  STEPS=6; SEEDS=1; EPOCHS=3; SAVE_EPOCHS="2,3"
  echo "[pipe] SMOKE mode: step 6, 1 seed, 3 epochs, both arms + infer + plots + video."
fi

read -r -a ARMS <<< "$ARMS_SPEC"

# ---- fixed infrastructure --------------------------------------------------
CONTAINER=gpu_env
SRC=/workspace/src
DET=$SRC/perception/tools/det2d
TRAINER=$SRC/perception/models/train_vision_model.py
INFER=$DET/inference_frcnn.py
DUMP=$DET/dump_predictions.py
VIDEO=$DET/compare_video.py
PLOT_CURVES=$DET/plot_curves.py
PLOT_PR=$DET/plot_pr.py
PLOT_CONFUSION=$DET/plot_confusion.py
PLOT_SCALING=$DET/plot_scaling.py          # may not exist yet (guarded)
CACHE=/workspace/datasets/waymo/perception_cache
MANIFEST=$CACHE/train/manifest.json

# ---- helpers ---------------------------------------------------------------
dex() { docker exec -i "$CONTAINER" sh -c "$1"; }

ntfy() {  # title, message, [priority]
  [ -z "$NTFY_TOPIC" ] && return 0
  curl -s --max-time 10 \
    -H "Title: $1" -H "Priority: ${3:-default}" -H "Tags: robot" \
    -d "$2" "$NTFY_URL/$NTFY_TOPIC" >/dev/null 2>&1 || true
}

fmt_hms() { awk -v s="$1" 'BEGIN{printf "%dh %02dm", int(s/3600), int((s%3600)/60)}'; }

# --limit-train for step N from the LIVE manifest, or "SKIP:<available>".
limit_for_step() {
  docker exec -i "$CONTAINER" python3 -I -c '
import json, sys
n = int(sys.argv[1]); path = sys.argv[2]
d = json.load(open(path))
order = []
for e in d:
    if not order or order[-1] != e["segment"]:
        order.append(e["segment"])
if n > len(order):
    print("SKIP:%d" % len(order))
else:
    sub = set(order[:n])
    print(sum(1 for e in d if e["segment"] in sub))
' "$1" "$MANIFEST"
}

# Seed whose best_ap is closest to the arm mean (ties -> lowest seed). "" if none.
median_seed() {
  docker exec -i "$CONTAINER" python3 -I -c '
import csv, sys
model, point = sys.argv[1], sys.argv[2]
path = "runs/sweep_%s_%s.csv" % (model, point)
rows = []
try:
    for r in csv.DictReader(open(path)):
        v = r.get("best_ap", "").strip()
        if v and v.lower() != "nan":
            rows.append((int(r["seed"]), float(v)))
except FileNotFoundError:
    pass
if not rows:
    print("")
else:
    mean = sum(a for _, a in rows) / len(rows)
    print(min(rows, key=lambda t: (abs(t[1] - mean), t[0]))[0])
' "$1" "$2"
}

est_step_secs() {  # limit_train -> est seconds for ALL arms, all seeds
  local nseeds narms
  nseeds=$(set -- $SEEDS; echo $#); narms=${#ARMS[@]}
  awk -v l="$1" -v e="$EPOCHS" -v n="$nseeds" -v a="$narms" -v s="$S_PER_IT" -v o="$OVERHEAD" \
    'BEGIN{ printf "%d", int(l/2)*e*n*a*s*o }'
}

# ---- core: one step, one arm (the folded sweep) ----------------------------
sweep_point() {  # label model lr point limit_train
  local ALABEL=$1 MODEL=$2 LR=$3 POINT=$4 LIMIT=$5
  local PREFIX=${MODEL}_${POINT}_seed
  local CSV=runs/sweep_${MODEL}_${POINT}.csv
  local LR_ARG="";    [ -n "$LR" ]    && LR_ARG="--lr $LR"
  local LIMIT_ARG=""; [ -n "$LIMIT" ] && LIMIT_ARG="--limit-train $LIMIT"
  local SAVE_ARG="";  [ -n "$SAVE_EPOCHS" ] && SAVE_ARG="--save-epochs $SAVE_EPOCHS"

  echo "[step] arm=$ALABEL model=$MODEL lr=${LR:-<default>} point=$POINT "\
"limit_train=${LIMIT:-<full>} epochs=$EPOCHS seeds=[$SEEDS] save=[$SAVE_EPOCHS]"
  dex "if [ -f $CSV ]; then mv $CSV ${CSV}.bak.\$(date +%Y%m%d_%H%M%S); fi"

  for S in $SEEDS; do
    local RID=${PREFIX}${S}
    echo "  --- seed $S ($RID) ---"
    dex "rm -rf runs/$RID"
    if ! docker exec -i "$CONTAINER" env PYTHONPATH=$SRC python "$TRAINER" \
         --model "$MODEL" --cache-root "$CACHE" \
         --epochs "$EPOCHS" --batch-size 2 --warmup-iters "$WARMUP" $LR_ARG \
         --eval-ap-every 1 --patience "$PATIENCE" --min-delta "$MIN_DELTA" \
         $LIMIT_ARG $SAVE_ARG \
         --seed "$S" --run-id "$RID" --sweep-csv "$CSV"; then
      echo "  [step] $RID FAILED -- continuing"
      ntfy "FAIL $RID" "training seed failed at $POINT" high
    fi
  done
}

# ---- core: inference pass (predictions.json per seed) ----------------------
infer_point() {  # model point
  local MODEL=$1 POINT=$2 PREFIX=${1}_${2}_seed
  for S in $SEEDS; do
    local RID=${PREFIX}${S}
    dex "[ -f runs/$RID/best.pt ]" || { echo "  [infer] no best.pt $RID -- skip"; continue; }
    echo "  [infer] $RID"
    docker exec -i "$CONTAINER" env PYTHONPATH=$SRC python "$INFER" \
      --ckpt "runs/$RID/best.pt" --cache-root "$CACHE" --split val \
      --score-thresh "$SCORE_THRESH" --quiet \
      || { echo "  [infer] $RID FAILED -- continuing"; ntfy "FAIL infer $RID" "$POINT" default; }
  done
}

# ---- plots per step: curves (from logs) + PR + confusion (from predictions) -
plot_step() {  # point
  local POINT=$1 OUT=$PLOTS_DIR/$POINT
  dex "mkdir -p $OUT"
  local CURVE_ARMS=() PRED_ARMS=()
  for A in "${ARMS[@]}"; do
    IFS=: read -r L M _ <<< "$A"
    CURVE_ARMS+=( --arm "$L=${M}_${POINT}_seed*" )
    PRED_ARMS+=( --arm "$L=/workspace/runs/${M}_${POINT}_seed*/infer_val/predictions.json" )
  done
  echo "  [plot] curves/PR/confusion -> $OUT"
  docker exec -i "$CONTAINER" env PYTHONPATH=$SRC python "$PLOT_CURVES" \
    --runs-root /workspace/runs "${CURVE_ARMS[@]}" --out-dir "$OUT" \
    || echo "  [plot] curves FAILED $POINT"
  docker exec -i "$CONTAINER" env PYTHONPATH=$SRC python "$PLOT_PR" \
    "${PRED_ARMS[@]}" --iou 0.5 --out-dir "$OUT" \
    || echo "  [plot] pr FAILED $POINT"
  docker exec -i "$CONTAINER" env PYTHONPATH=$SRC python "$PLOT_CONFUSION" \
    "${PRED_ARMS[@]}" --iou 0.5 --score-thresh 0.5 --out-dir "$OUT" \
    || echo "  [plot] confusion FAILED $POINT"
}

# ---- video per step: median seed of each arm, dump PNGs, render A-vs-B ------
video_step() {  # point
  local POINT=$1
  if [ "${#ARMS[@]}" -ne 2 ]; then echo "  [video] needs exactly 2 arms -- skip"; return 0; fi
  local LA MA LB MB SA SB RA RB OUT
  IFS=: read -r LA MA _ <<< "${ARMS[0]}"
  IFS=: read -r LB MB _ <<< "${ARMS[1]}"
  SA=$(median_seed "$MA" "$POINT"); SB=$(median_seed "$MB" "$POINT")
  if [ -z "$SA" ] || [ -z "$SB" ]; then echo "  [video] no median seed -- skip"; return 0; fi
  RA=${MA}_${POINT}_seed${SA}; RB=${MB}_${POINT}_seed${SB}
  echo "  [video] median seeds: $LA=seed$SA  $LB=seed$SB"
  for RID in "$RA" "$RB"; do
    dex "[ -f runs/$RID/best.pt ]" || { echo "  [video] no best.pt $RID -- skip"; return 0; }
    docker exec -i "$CONTAINER" env PYTHONPATH=$SRC python "$DUMP" \
      --ckpt "runs/$RID/best.pt" --cache-root "$CACHE" --split val \
      --dump-score-thresh "$DUMP_SCORE_THRESH" --out-dir "runs/$RID/dump_val" \
      || { echo "  [video] dump $RID FAILED -- skip"; return 0; }
  done
  OUT=$VIDEO_DIR/$POINT; dex "mkdir -p $OUT"
  docker exec -i "$CONTAINER" env PYTHONPATH=$SRC python "$VIDEO" \
    --json-a "runs/$RA/dump_val/predictions.json" \
    --json-b "runs/$RB/dump_val/predictions.json" \
    --label-a "$LA" --label-b "$LB" \
    --layout "$VIDEO_LAYOUT" --score-thresh "$VIDEO_SCORE_THRESH" --fps "$VIDEO_FPS" \
    --out-dir "$OUT" \
    || echo "  [video] compare_video FAILED $POINT -- continuing"
}

# ---- single-step mode (exploration) ----------------------------------------
if [ "${1:-}" = "--point" ]; then
  shift
  MODEL=${1:?usage: --point <model> [lr]}; LR=${2:-}
  : "${POINT:=n48_${STRIDE_TAG}}"
  sweep_point "manual" "$MODEL" "$LR" "$POINT" "${LIMIT_TRAIN:-}"
  [ "$DO_INFER" = 1 ] && infer_point "$MODEL" "$POINT"
  exit 0
fi

# ---- full ladder -----------------------------------------------------------
NSEEDS=$(set -- $SEEDS; echo $#)
echo "[pipe] ladder steps=[$STEPS] arms=${#ARMS[@]} seeds=$NSEEDS epochs=$EPOCHS"
echo "[pipe] infer=$DO_INFER plots=$DO_PLOTS video=$DO_VIDEO plots_dir=$PLOTS_DIR videos_dir=$VIDEO_DIR ntfy=${NTFY_TOPIC:-<off>}"
dex "mkdir -p $PLOTS_DIR $VIDEO_DIR runs"

declare -A LIMIT_OF
TOTAL_SECS=0; PLAN=""
for N in $STEPS; do
  L=$(limit_for_step "$N")
  if [ -z "$L" ]; then echo "[pipe] !! cannot read manifest for n$N -- aborting"; exit 1; fi
  case "$L" in
    SKIP:*) avail=${L#SKIP:}
            echo "[pipe] n$N: only $avail segments extracted -> SKIP"
            ntfy "ladder: n$N skipped" "only $avail segments extracted" ;;
    *)      LIMIT_OF[$N]=$L
            secs=$(est_step_secs "$L"); TOTAL_SECS=$((TOTAL_SECS + secs))
            PLAN="$PLAN n$N(limit=$L,~$(fmt_hms "$secs"))" ;;
  esac
done
echo "[pipe] plan:$PLAN"
echo "[pipe] rough ETA (${#ARMS[@]} arms, $NSEEDS seeds): $(fmt_hms "$TOTAL_SECS")"
ntfy "ladder START" "steps:$PLAN
total ~$(fmt_hms "$TOTAL_SECS")" high

START=$(date +%s); DONE_SECS=0
for N in $STEPS; do
  [ -z "${LIMIT_OF[$N]:-}" ] && continue
  L=${LIMIT_OF[$N]}; POINT=n${N}_${STRIDE_TAG}
  pt_est=$(est_step_secs "$L")
  echo "======================================================================"
  echo "[pipe] STEP n$N (limit_train=$L, point=$POINT)  est ~$(fmt_hms "$pt_est")"
  ntfy "n$N START" "limit=$L, est ~$(fmt_hms "$pt_est")"
  t0=$(date +%s)

  for A in "${ARMS[@]}"; do
    IFS=: read -r ALABEL AMODEL ALR <<< "$A"
    sweep_point "$ALABEL" "$AMODEL" "$ALR" "$POINT" "$L"
    [ "$DO_INFER" = 1 ] && infer_point "$AMODEL" "$POINT"
  done
  [ "$DO_PLOTS" = 1 ] && plot_step "$POINT"
  [ "$DO_VIDEO" = 1 ] && video_step "$POINT"

  t1=$(date +%s); el=$((t1 - t0)); DONE_SECS=$((DONE_SECS + el))
  rem=$((TOTAL_SECS - DONE_SECS)); [ "$rem" -lt 0 ] && rem=0
  echo "[pipe] step n$N done in $(fmt_hms "$el"); remaining ~$(fmt_hms "$rem")"
  ntfy "n$N DONE" "took $(fmt_hms "$el"); remaining ~$(fmt_hms "$rem")"
done

# ---- scaling plot (guarded) ------------------------------------------------
echo "[pipe] ladder training complete in $(fmt_hms "$(( $(date +%s) - START ))")"
if [ "$DO_PLOTS" = 1 ] && dex "[ -f $PLOT_SCALING ]"; then
  SCALE_ARMS=()
  for A in "${ARMS[@]}"; do IFS=: read -r L M _ <<< "$A"; SCALE_ARMS+=( --arm "$L=$M" ); done
  echo "[pipe] AP x N scaling plot -> $PLOTS_DIR"
  docker exec -i "$CONTAINER" python "$PLOT_SCALING" \
    --runs-root /workspace/runs --steps "$STEPS" --tag "$STRIDE_TAG" \
    --out-dir "$PLOTS_DIR" "${SCALE_ARMS[@]}" \
    || echo "[pipe] plot_scaling FAILED -- ledgers in runs/sweep_*.csv"
else
  echo "[pipe] (plot_scaling.py not found or plots off -- data in runs/sweep_*_n<N>_${STRIDE_TAG}.csv)"
fi

ntfy "ladder COMPLETE" "total $(fmt_hms "$(( $(date +%s) - START ))")" high
echo "[pipe] DONE. plots in $PLOTS_DIR/<point>/ ; videos in $VIDEO_DIR/<point>/ ; ledgers runs/sweep_*.csv"