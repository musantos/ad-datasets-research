#!/usr/bin/env bash
# run_perception_pipeline.sh  (v5 -- v4 + per-experiment output subfolder + AMP
# passthrough)
# Data-scaling ladder orchestrator for the 2D mono-camera perception A/B study
# (pretrained+FrozenBN vs from-scratch+GN). ONE self-contained file.
#
# Per STEP N in STEPS (N = segment count = ladder x-axis):
#   1. derive --limit-train from the LIVE train manifest (first N segments ==
#      first K frames exactly: 0 empty records + segment-contiguous order);
#   2. TRAIN every arm (SEEDS seeds, EPOCHS epochs, snapshots at SAVE_EPOCHS),
#      COCO-AP on the FIXED full val every epoch (val is never limited);
#   3. INFER per seed -> infer_val/predictions.json (lean, for PR/confusion);
#   4. PLOTS per step (curves + PR + confusion) -> PLOTS_DIR/<point>/;
#   5. VIDEO per step: pick the MEDIAN seed of each arm automatically, dump PNGs
#      for those two, render the A-vs-B comparison -> VIDEO_DIR/<point>/;
#   6. ntfy status (host-side) with a banded ETA, and a ntfy ping on any failure.
#   At the end: AP x N scaling plot -> PLOTS_DIR/.
# Stages 3-5 are NON-FATAL: a plot/video error never costs the training.
#
# v5 CHANGES (vs v4):
#   * EXPERIMENT SUBFOLDER. Everything for one campaign lands under
#     runs/<EXP>/ instead of flat in runs/: per-seed run dirs, sweep ledgers,
#     telemetry, plots/ and videos/. Set EXP to name the campaign; it defaults
#     to a timestamp. No more 100 loose folders in runs/.
#   * AMP PASSTHROUGH. AMP=1 adds --amp (+ --amp-dtype $AMP_DTYPE) to every
#     training call, so an AMP A/B is just two runs with different EXP + AMP.
#     AMP breaks FP32 parity -- run it as its OWN campaign, never mixed into a
#     baseline ladder.
#   * LR_SCALE + GRAD_CLIP PASSTHROUGH. LR_SCALE multiplies each arm's base LR
#     in the trainer (base = --lr if given, else the per-model default), for
#     LR-stability ablations (e.g. default vs default x2.5). GRAD_CLIP sets the
#     trainer's gradient-clip norm (0 disables). Both default to the locked
#     protocol (LR_SCALE=1, GRAD_CLIP=10.0), so an un-set run is unchanged.
#
# ETA MODEL (v4): the per-step cost splits into a TRAIN part that scales with
# (limit_train x epochs) and a fixed TAIL part (infer + plots + video) that does
# NOT scale with limit. The two are timed separately each step. The next step's
# forecast is reported as a BAND:
#   best  = predicted_train(next) + observed_tail           (additive overhead)
#   worst = predicted_train(next) x observed_train_multiplier + observed_tail
#
# HW TELEMETRY (v4, opt-in HW_MONITOR=1): a background sampler logs RAM/SWAP/
# VRAM/GPU-util/GPU-temp/per-core-CPU once per second to TELEMETRY_CSV for the
# whole run, started at ladder start and stopped via trap at exit.
#   v5.1: MULTI-GPU. NUM_GPUS GPUs (default 2) are logged, one 4-column block per
#   GPU tagged with its PCI slot (e.g. GPU_Temp_g1_05:00.0), flattened to a single
#   row per sample. v4 assumed one GPU and would split a 2-GPU sample across two
#   physical lines, corrupting the CSV. The live layout is validated against the
#   pinned NUM_GPUS/GPU_BUS_IDS and telemetry REFUSES to start on a mismatch.
#
# MODES:
#   full ladder :  ./run_perception_pipeline.sh
#   single step :  POINT=n6_s8 LIMIT_TRAIN=150 ./run_perception_pipeline.sh --point faster_rcnn_pt 1e-3
#   smoke test  :  SMOKE=1 ./run_perception_pipeline.sh   (1 seed, 3 ep, step 6)
#   with HW log :  HW_MONITOR=1 ./run_perception_pipeline.sh
#   named + AMP :  EXP=amp_n6_bf16 AMP=1 STEPS=6 SEEDS="1 2 3" ./run_perception_pipeline.sh
#   LR ablation :  EXP=lr_abl_x2p5 LR_SCALE=2.5 STEPS="6 24" SEEDS="1 2 3" EPOCHS=12 ./run_perception_pipeline.sh
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
: "${SCORE_THRESH:=0.05}"                 # inference score thresh (same every arm)
: "${STRIDE_TAG:=s8}"                     # LABEL only; real stride is set at extraction
: "${BATCH_SIZE:=2}"                      # train batch size (2 = locked ladder protocol)
: "${EVAL_EVERY:=1}"                      # COCO-AP every K epochs (>1 trims eval wall in sweeps)
: "${LR_SCALE:=1}"                        # multiply each arm's base LR in the trainer (LR-stability ablation); 1 = off
: "${GRAD_CLIP:=10.0}"                    # max grad-norm forwarded to the trainer (0 disables clipping)
# Arm spec: "label:model:lr[:display]" tokens, whitespace-separated.
# The optional 4th field is a cosmetic name; it must NOT contain spaces -- the
# tokens are whitespace-separated, so a space in the display field is parsed as
# a new (malformed) arm. Use hyphens. If omitted it falls back to
# "<model> (<label>)". The rest of the pipeline keys off label/model only.
# NOTE: an arm with an EMPTY lr field (e.g. "base:retinanet::RetinaNet-base")
# now takes the trainer's PER-MODEL default LR, not a single global default --
# so from-scratch RetinaNet gets 1.25e-3, not the Faster R-CNN 2.5e-3 that used
# to drive it to NaN. Pass an explicit lr only to override that default.
: "${ARMS_SPEC:=pretrained:faster_rcnn_pt:1e-3:FRCNN-pretrained from-scratch:faster_rcnn::FRCNN-base}"

# ---- experiment folder + AMP (v5) ------------------------------------------
# EXP names the campaign; all outputs live under runs/$EXP/. Keep it stable for
# one campaign and distinct across campaigns (e.g. amp_n6_fp32 vs amp_n6_bf16).
: "${EXP:=exp_$(date +%Y%m%d_%H%M%S)}"
RUNS_REL="runs/$EXP"                      # container-relative (cwd = /workspace)
RUNS_ABS="/workspace/$RUNS_REL"           # absolute, for tools taking --runs-root
# AMP: off by default (FP32 baseline discipline). AMP=1 -> mixed precision.
: "${AMP:=0}"
: "${AMP_DTYPE:=bfloat16}"                # bfloat16 (Blackwell-preferred) | float16
AMP_ARG=""
[ "$AMP" = 1 ] && AMP_ARG="--amp --amp-dtype $AMP_DTYPE"

# ---- GPU pinning (dual-GPU: one job per card) ------------------------------
# One gpu_env container (started --gpus all) sees both cards; jobs are separated
# by PROCESS, not container. CUDA_DEV sets CUDA_VISIBLE_DEVICES INSIDE the
# container for this invocation's GPU-bound stages (train/infer/dump), so two
# parallel invocations (two docker exec into the same gpu_env) pin to different
# cards. Empty (default) = unset = current behavior (PyTorch -> cuda:0).
# CUDA_DEVICE_ORDER=PCI_BUS_ID is pinned alongside so the CUDA index == the
# nvidia-smi/telemetry bus index (CUDA defaults to fastest-first, which with two
# identical cards need NOT match the bus order -- without this you could pin to a
# card and read the OTHER card's temperature). So: CUDA_DEV=0 -> g0_01:00.0
# (display), CUDA_DEV=1 -> g1_05:00.0. Host-shell CUDA_VISIBLE_DEVICES does NOT
# reach docker exec -- this does.
: "${CUDA_DEV:=}"
DEV_ENV=""
[ -n "$CUDA_DEV" ] && DEV_ENV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CUDA_DEV"

# ---- stage toggles ---------------------------------------------------------
: "${DO_INFER:=1}"                        # predictions.json per seed (PR/confusion)
: "${DO_PLOTS:=1}"                        # curves + PR + confusion per step, + scaling
: "${DO_VIDEO:=1}"                        # median-seed A-vs-B comparison video per step
: "${VAL_LIMIT:=}"                        # cap val frames for infer/dump (smoke only); empty = FULL val

# ---- video parameters ------------------------------------------------------
: "${DUMP_SCORE_THRESH:=0.05}"            # dump keeps >= this; video filters higher
: "${VIDEO_SCORE_THRESH:=0.5}"            # display threshold in the video
: "${VIDEO_LAYOUT:=3up}"                  # 3up | dual-overlay
: "${VIDEO_FPS:=5}"

# ---- output directories (container paths; all under the experiment folder) -
: "${PLOTS_DIR:=$RUNS_ABS/plots}"
: "${VIDEO_DIR:=$RUNS_ABS/videos}"

# ---- notifications (host-side; empty topic = silent) -----------------------
: "${NTFY_TOPIC:=}"
: "${NTFY_URL:=https://ntfy.sh}"
: "${DATE_FMT:=%Y-%m-%d %H:%M}"           # datetime stamp used in every notification
: "${END_FMT:=%m-%d %H:%M}"               # predicted-end stamp (carries date to survive day rollover)
: "${NOTIFY_EACH:=1}"                     # 1 = counted "Exp i/TOTAL" ping (ntfy+stdout) at each run start; 0 = quiet
: "${NTFY_LOG:=$RUNS_REL/ntfy_events.log}" # the 3rd log: local catalog of EVERY ntfy event (host-side, under EXP)

# ---- HW telemetry (opt-in) -------------------------------------------------
: "${HW_MONITOR:=0}"                      # 1 = sample host HW to TELEMETRY_CSV for the whole run
: "${TELEMETRY_CSV:=$RUNS_REL/telemetry_$(date +%Y%m%d_%H%M%S).csv}"  # host path, under EXP
: "${HW_INTERVAL:=1}"                     # sampling period hint (seconds)
# Expected GPU layout, PINNED to this box. Telemetry validates the live
# nvidia-smi layout against these at start and REFUSES to run on a mismatch
# (wrong count or wrong bus id) -- so another box can't silently log days of
# mislabeled columns. On different hardware, override BOTH to match (or leave
# HW_MONITOR=0). GPU_BUS_IDS are short PCI slots in nvidia-smi index order.
: "${NUM_GPUS:=2}"                        # expected GPU count
: "${GPU_BUS_IDS:=01:00.0 05:00.0}"       # expected slots, index order (g0=display, g1=free)

# ---- ETA calibration -------------------------------------------------------
# training: measured n48/n96 ~255-262 s/ep over ~1170 it -> ~0.22 s/it.
: "${S_PER_IT:=0.22}"
: "${OVERHEAD:=1.1}"
# non-training stages, seconds per val frame (rough a-priori; self-corrected at runtime):
: "${INFER_SPF:=0.35}"   # inference pass, per seed per arm
: "${DUMP_SPF:=0.80}"    # dump PNGs for the 2 median seeds (video input)
: "${ENCODE_SPF:=0.05}"  # video encode
: "${PLOT_FIXED:=60}"    # curves+PR+confusion per step (~fixed)

# ---- smoke overrides -------------------------------------------------------
: "${SMOKE:=0}"
if [ "$SMOKE" = 1 ]; then
  STEPS=6; SEEDS=1; EPOCHS=3; SAVE_EPOCHS="2,3"
  : "${VAL_LIMIT:=16}"   # cap val too, so infer/dump/video stay fast in the smoke
  echo "[pipe] SMOKE mode: step 6, 1 seed, 3 epochs, val=$VAL_LIMIT frames, full chain."
fi

read -r -a ARMS <<< "$ARMS_SPEC"
VAL_ARG=""; [ -n "$VAL_LIMIT" ] && VAL_ARG="--limit $VAL_LIMIT"

# ---- fixed infrastructure --------------------------------------------------
CONTAINER=gpu_env
SRC=/workspace/src
DET=$SRC/perception/tools/det2d
TRAINER=$SRC/perception/models/train_vision_model.py
INFER=$DET/infer_det2d.py
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
  # Catalog EVERY event locally first, timestamped, with the message's newlines
  # flattened to spaces so each event is ONE greppable line. Runs even when
  # NTFY_TOPIC is empty, so the campaign always has a full event log on disk
  # regardless of push delivery (the "3rd log" alongside the two stdout logs).
  if [ -n "${NTFY_LOG:-}" ]; then
    mkdir -p "$(dirname "$NTFY_LOG")" 2>/dev/null || true
    printf '%s\t[%s]\t%s\t%s\n' "$(now_stamp)" "${3:-default}" "$1" \
      "$(printf '%s' "$2" | tr '\n' ' ')" >> "$NTFY_LOG" 2>/dev/null || true
  fi
  [ -z "$NTFY_TOPIC" ] && return 0
  curl -s --max-time 10 \
    -H "Title: $1" -H "Priority: ${3:-default}" -H "Tags: robot" \
    -d "$2" "$NTFY_URL/$NTFY_TOPIC" >/dev/null 2>&1 || true
}

fmt_hms() { awk -v s="$1" 'BEGIN{printf "%dh %02dm", int(s/3600), int((s%3600)/60)}'; }

now_stamp()  { date "+$DATE_FMT"; }                         # notification datetime
# Absolute wall-clock stamp for "now + <secs>" (date handles day/month/year rollover).
end_stamp()  { date -d "@$(( $(date +%s) + ${1:-0} ))" "+$END_FMT"; }

# Display name for an arm token "label:model:lr[:display]".
arm_display() {  # label model display
  if [ -n "${3:-}" ]; then echo "$3"; else echo "$2 ($1)"; fi
}

# ---- HW telemetry: background sampler (host) -------------------------------
# One row per HW_INTERVAL: date,time,RAM/SWAP(MB),VRAM(MB),GPU util/temp, per-core CPU.
# nvidia-smi on the host sees the whole GPU, container processes included, so GPU
# util and VRAM are the real values during training inside gpu_env.
HW_PID=""
GPU_SLOT_TAGS=()   # gIDX_SLOT tags, set by hw_resolve_gpus after validation

# Validate the LIVE GPU layout against the pinned expectation (NUM_GPUS +
# GPU_BUS_IDS, in nvidia-smi index/bus order) and set GPU_SLOT_TAGS. Returns 1
# on any mismatch so hw_start can refuse to run -- better than logging days of
# mislabeled columns on an unexpected box.
hw_resolve_gpus() {
  local idx bus i actual_slots=() exp_slots=()
  read -r -a exp_slots <<< "$GPU_BUS_IDS"
  while IFS=, read -r idx bus; do
    idx=$(echo "$idx" | tr -d ' '); bus=$(echo "$bus" | tr -d ' ')
    [ -z "$idx" ] && continue
    actual_slots+=( "${bus#*:}" )        # 00000000:05:00.0 -> 05:00.0
  done < <(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader,nounits 2>/dev/null)

  if [ "${#actual_slots[@]}" -eq 0 ]; then
    echo "[hw] ERROR: nvidia-smi reported no GPUs (expected NUM_GPUS=$NUM_GPUS)."; return 1
  fi
  if [ "${#exp_slots[@]}" -ne "$NUM_GPUS" ]; then
    echo "[hw] ERROR: GPU_BUS_IDS has ${#exp_slots[@]} entries but NUM_GPUS=$NUM_GPUS -- fix the params."; return 1
  fi
  if [ "${#actual_slots[@]}" -ne "$NUM_GPUS" ]; then
    echo "[hw] ERROR: NUM_GPUS=$NUM_GPUS but nvidia-smi sees ${#actual_slots[@]} GPU(s): ${actual_slots[*]}"
    echo "[hw] different box? override NUM_GPUS + GPU_BUS_IDS to match, or run with HW_MONITOR=0."; return 1
  fi
  GPU_SLOT_TAGS=()
  for ((i=0; i<NUM_GPUS; i++)); do
    if [ "${actual_slots[$i]}" != "${exp_slots[$i]}" ]; then
      echo "[hw] ERROR: GPU $i bus mismatch -- expected ${exp_slots[$i]}, found ${actual_slots[$i]}"
      echo "[hw] live: ${actual_slots[*]} | expected: ${exp_slots[*]}"
      echo "[hw] different box? override NUM_GPUS + GPU_BUS_IDS to match, or run with HW_MONITOR=0."; return 1
    fi
    GPU_SLOT_TAGS+=( "g${i}_${actual_slots[$i]}" )
  done
  echo "[hw] GPU layout OK (${NUM_GPUS}): ${GPU_SLOT_TAGS[*]}"
  return 0
}

hw_monitor_loop() {  # csv_path interval
  local csv=$1 interval=${2:-1} ncores i header g gzero=""
  ncores=$(nproc)
  # Columns come from the validated GPU_SLOT_TAGS: one 4-metric block per GPU,
  # tagged by PCI slot so a column is never ambiguous about which card it is.
  header="Data,Hora,RAM_Total(MB),RAM_Usada(MB),RAM_Livre(MB),SWAP_Total(MB),SWAP_Usada(MB),SWAP_Livre(MB)"
  for g in "${GPU_SLOT_TAGS[@]}"; do
    header="$header,VRAM_Total_${g}(MB),VRAM_Usada_${g}(MB),GPU_Util_${g}(%),GPU_Temp_${g}(C)"
    gzero="$gzero,0,0,0,0"
  done
  gzero=${gzero#,}                       # drop leading comma -> matches GPU block count
  for ((i=0; i<ncores; i++)); do header="$header,CPU_$i(%)"; done
  echo "$header" > "$csv"
  while :; do
    local ram swap gpu cpu_line
    ram=$(free -m | awk '/^Mem:/{print $2","$3","$4}')
    swap=$(free -m | awk '/^Swap:/{print $2","$3","$4}')
    # One row per GPU from nvidia-smi (index/bus order); flatten to a single CSV
    # line (paste -sd,) so a 2-GPU box can't split one sample across two rows.
    gpu=$(nvidia-smi --query-gpu=memory.total,memory.used,utilization.gpu,temperature.gpu \
          --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | paste -sd, -)
    [ -z "$gpu" ] && gpu="$gzero"
    # mpstat's own 1 s sample sets the loop cadence. Key busy% by core id (last
    # value wins) and emit core 0..max exactly once, so an extra mpstat report
    # block (some builds append an "Average:" sample) can't duplicate columns --
    # the v4 CSV came out with 2x the CPU columns from exactly that.
    cpu_line=$(mpstat -P ALL "$interval" 1 | awk '
      /^[0-9]/ && $2 ~ /^[0-9]+$/ { busy[$2+0]=100-$NF; if ($2+0>mx) mx=$2+0 }
      END { for (c=0; c<=mx; c++) printf "%s%.1f", (c?",":""), busy[c] }')
    echo "$(date +%Y-%m-%d),$(date +%H:%M:%S),$ram,$swap,$gpu,$cpu_line" >> "$csv"
  done
}

hw_start() {
  [ "$HW_MONITOR" = 1 ] || return 0
  command -v mpstat >/dev/null 2>&1 || { echo "[hw] mpstat missing (install sysstat) -- telemetry OFF"; return 0; }
  # Fail-early guard: pinned GPU layout must match the live box, or refuse to run.
  if ! hw_resolve_gpus; then
    echo "[hw] ABORT: GPU layout does not match the pinned expectation (see above)."
    ntfy "ladder ABORT [$EXP]" "$(now_stamp)
GPU layout mismatch -- telemetry guard refused to start" high
    exit 1
  fi
  mkdir -p "$(dirname "$TELEMETRY_CSV")" 2>/dev/null || true
  # The sampler runs host-side. If the EXP dir was created by the container
  # (root) the host user can't write here -- test once and degrade gracefully
  # instead of erroring every second for the whole run.
  if ! ( : > "$TELEMETRY_CSV" ) 2>/dev/null; then
    echo "[hw] cannot write $TELEMETRY_CSV (dir owned by container/root?) -- telemetry OFF"
    return 0
  fi
  hw_monitor_loop "$TELEMETRY_CSV" "$HW_INTERVAL" &
  HW_PID=$!
  echo "[hw] telemetry ON -> $TELEMETRY_CSV (pid $HW_PID)"
}

hw_stop() {
  [ -n "$HW_PID" ] || return 0
  kill "$HW_PID" 2>/dev/null; wait "$HW_PID" 2>/dev/null
  echo "[hw] telemetry stopped -> $TELEMETRY_CSV"
  HW_PID=""
}
trap hw_stop EXIT INT TERM

# Number of val frames actually processed by infer/dump (respects VAL_LIMIT).
val_frames() {
  local full
  full=$(docker exec -i "$CONTAINER" python3 -I -c '
import json, sys
try: print(len(json.load(open(sys.argv[1]))))
except Exception: print(0)
' "$CACHE/val/manifest.json")
  [ -z "$full" ] && full=0
  if [ -n "$VAL_LIMIT" ] && [ "$full" -gt 0 ] && [ "$VAL_LIMIT" -lt "$full" ]; then
    echo "$VAL_LIMIT"; else echo "$full"; fi
}

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
# Diverged seeds (diverged==1 in the ledger) are EXCLUDED -- a run that hit a
# non-finite step must never be the representative seed for the comparison video.
# argv: model point runs_rel  (ledger lives at <runs_rel>/sweep_<model>_<point>.csv)
median_seed() {
  docker exec -i "$CONTAINER" python3 -I -c '
import csv, sys
model, point, runs_rel = sys.argv[1], sys.argv[2], sys.argv[3]
path = "%s/sweep_%s_%s.csv" % (runs_rel, model, point)
rows = []
try:
    for r in csv.DictReader(open(path)):
        if r.get("diverged", "0").strip() == "1":
            continue
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
' "$1" "$2" "$3"
}

# ---- ETA: split predictions (train part scales, tail part is fixed) --------
# est_train_secs <limit_train> : predicted TRAIN seconds for the whole step
#   (all seeds, all arms). Scales with iters/epoch x epochs.
est_train_secs() {
  local nseeds narms
  nseeds=$(set -- $SEEDS; echo $#); narms=${#ARMS[@]}
  awk -v l="$1" -v e="$EPOCHS" -v n="$nseeds" -v a="$narms" \
      -v sit="$S_PER_IT" -v o="$OVERHEAD" -v bs="$BATCH_SIZE" \
    'BEGIN{ printf "%d", int(l/bs)*e*n*a*sit*o }'  # int(l/bs) = iters/epoch (s_per_it calibrated at bs=2; rough for bs>2)
}

# est_tail_secs : predicted fixed TAIL seconds (infer + plots + video) for the step.
#   Independent of limit_train; depends on val frame count and stage toggles.
est_tail_secs() {
  local nseeds narms
  nseeds=$(set -- $SEEDS; echo $#); narms=${#ARMS[@]}
  awk -v n="$nseeds" -v a="$narms" -v nv="${NVAL:-0}" \
      -v di="$DO_INFER" -v dp="$DO_PLOTS" -v dv="$DO_VIDEO" \
      -v ispf="$INFER_SPF" -v dspf="$DUMP_SPF" -v espf="$ENCODE_SPF" -v pf="$PLOT_FIXED" \
    'BEGIN{
       infer = (di=="1" ? n*a*nv*ispf : 0);
       plots = (dp=="1" ? pf : 0);
       video = (dv=="1" ? (2*nv*dspf + nv*espf) : 0);
       printf "%d", infer+plots+video;
     }'
}

# ---- core: one step, one arm (the folded sweep) ----------------------------
sweep_point() {  # label model lr point limit_train [display]
  local ALABEL=$1 MODEL=$2 LR=$3 POINT=$4 LIMIT=$5 DISPLAY=${6:-$2}
  local PREFIX=${MODEL}_${POINT}_seed
  local CSV=$RUNS_REL/sweep_${MODEL}_${POINT}.csv
  local LR_ARG="";    [ -n "$LR" ]    && LR_ARG="--lr $LR"
  local LIMIT_ARG=""; [ -n "$LIMIT" ] && LIMIT_ARG="--limit-train $LIMIT"
  local SAVE_ARG="";  [ -n "$SAVE_EPOCHS" ] && SAVE_ARG="--save-epochs $SAVE_EPOCHS"

  echo "[step] arm=$ALABEL model=$MODEL lr=${LR:-<model-default>} scale=$LR_SCALE "\
"grad_clip=$GRAD_CLIP point=$POINT limit_train=${LIMIT:-<full>} epochs=$EPOCHS "\
"seeds=[$SEEDS] save=[$SAVE_EPOCHS] amp=$AMP"
  dex "if [ -f $CSV ]; then mv $CSV ${CSV}.bak.\$(date +%Y%m%d_%H%M%S); fi"

  # Arm/step "Started" banner (e.g. "RetinaNet n96_s8 started").
  if [ "${NOTIFY_EACH:-0}" = 1 ]; then
    ntfy "$DISPLAY $POINT started" "$(now_stamp)
arm=$ALABEL model=$MODEL seeds=[$SEEDS]"
  fi

  for S in $SEEDS; do
    local RID=${PREFIX}${S}
    # Global campaign counter: one line per individual training as it launches,
    # so progress is "Exp i/TOTAL", not guesswork. TOTAL_EXP/EXP_N are set before
    # the first sweep_point call in both ladder and --point modes.
    if [ "${NOTIFY_EACH:-0}" = 1 ] && [ "${TOTAL_EXP:-0}" -gt 0 ]; then
      EXP_N=$(( ${EXP_N:-0} + 1 ))
      echo "  [exp] Exp $EXP_N/$TOTAL_EXP: $DISPLAY $POINT seed $S started"
      ntfy "Exp $EXP_N/$TOTAL_EXP started" "$(now_stamp)
$DISPLAY $POINT seed $S"
    fi
    echo "  --- seed $S ($RID) ---"
    dex "rm -rf $RUNS_REL/$RID"
    if ! docker exec -i "$CONTAINER" env $DEV_ENV PYTHONPATH=$SRC python "$TRAINER" \
         --model "$MODEL" --cache-root "$CACHE" \
         --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --warmup-iters "$WARMUP" \
         --lr-scale "$LR_SCALE" --grad-clip-norm "$GRAD_CLIP" $LR_ARG \
         --eval-ap-every "$EVAL_EVERY" --patience "$PATIENCE" --min-delta "$MIN_DELTA" \
         $LIMIT_ARG $SAVE_ARG $AMP_ARG \
         --runs-root "$RUNS_REL" \
         --seed "$S" --run-id "$RID" --sweep-csv "$CSV"; then
      echo "  [step] $RID FAILED -- continuing"
      ntfy "FAIL $RID" "$(now_stamp)
training seed failed at $POINT" high
    fi
  done
}

# ---- core: inference pass (predictions.json per seed) ----------------------
infer_point() {  # model point
  local MODEL=$1 POINT=$2 PREFIX=${1}_${2}_seed
  for S in $SEEDS; do
    local RID=${PREFIX}${S}
    dex "[ -f $RUNS_REL/$RID/best.pt ]" || { echo "  [infer] no best.pt $RID -- skip"; continue; }
    echo "  [infer] $RID"
    docker exec -i "$CONTAINER" env $DEV_ENV PYTHONPATH=$SRC python "$INFER" \
      --ckpt "$RUNS_REL/$RID/best.pt" --cache-root "$CACHE" --split val \
      --score-thresh "$SCORE_THRESH" --quiet $VAL_ARG \
      || { echo "  [infer] $RID FAILED -- continuing"; ntfy "FAIL infer $RID" "$(now_stamp)
$POINT"; }
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
    PRED_ARMS+=( --arm "$L=$RUNS_ABS/${M}_${POINT}_seed*/infer_val/predictions.json" )
  done
  echo "  [plot] curves/PR/confusion -> $OUT"
  docker exec -i "$CONTAINER" env PYTHONPATH=$SRC python "$PLOT_CURVES" \
    --runs-root "$RUNS_ABS" "${CURVE_ARMS[@]}" --out-dir "$OUT" \
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
  SA=$(median_seed "$MA" "$POINT" "$RUNS_REL"); SB=$(median_seed "$MB" "$POINT" "$RUNS_REL")
  if [ -z "$SA" ] || [ -z "$SB" ]; then echo "  [video] no median seed -- skip"; return 0; fi
  RA=${MA}_${POINT}_seed${SA}; RB=${MB}_${POINT}_seed${SB}
  echo "  [video] median seeds: $LA=seed$SA  $LB=seed$SB"
  for RID in "$RA" "$RB"; do
    dex "[ -f $RUNS_REL/$RID/best.pt ]" || { echo "  [video] no best.pt $RID -- skip"; return 0; }
    docker exec -i "$CONTAINER" env $DEV_ENV PYTHONPATH=$SRC python "$DUMP" \
      --ckpt "$RUNS_REL/$RID/best.pt" --cache-root "$CACHE" --split val \
      --dump-score-thresh "$DUMP_SCORE_THRESH" $VAL_ARG --out-dir "$RUNS_REL/$RID/dump_val" \
      || { echo "  [video] dump $RID FAILED -- skip"; return 0; }
  done
  OUT=$VIDEO_DIR/$POINT; dex "mkdir -p $OUT"
  docker exec -i "$CONTAINER" env PYTHONPATH=$SRC python "$VIDEO" \
    --json-a "$RUNS_REL/$RA/dump_val/predictions.json" \
    --json-b "$RUNS_REL/$RB/dump_val/predictions.json" \
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
  mkdir -p "$RUNS_REL" 2>/dev/null || true   # host-side first (see note below)
  dex "mkdir -p $RUNS_ABS"
  hw_start
  NSEEDS=$(set -- $SEEDS; echo $#); TOTAL_EXP=$NSEEDS; EXP_N=0
  sweep_point "manual" "$MODEL" "$LR" "$POINT" "${LIMIT_TRAIN:-}" "$MODEL"
  [ "$DO_INFER" = 1 ] && infer_point "$MODEL" "$POINT"
  exit 0
fi

# ---- full ladder -----------------------------------------------------------
NSEEDS=$(set -- $SEEDS; echo $#)
echo "[pipe] EXP=$EXP -> $RUNS_ABS"
echo "[pipe] ladder steps=[$STEPS] arms=${#ARMS[@]} seeds=$NSEEDS epochs=$EPOCHS amp=$AMP${AMP:+ dtype=$AMP_DTYPE} lr_scale=$LR_SCALE grad_clip=$GRAD_CLIP"
echo "[pipe] infer=$DO_INFER plots=$DO_PLOTS video=$DO_VIDEO plots_dir=$PLOTS_DIR videos_dir=$VIDEO_DIR ntfy=${NTFY_TOPIC:-<off>} hw=${HW_MONITOR}"
# Create the EXP dir HOST-side FIRST so it is owned by the host user: the
# container (root) can still write run/plot/video subdirs into a host-owned dir,
# but the host-side HW sampler can only write telemetry if the dir is NOT
# root-owned. Making it container-first locks the sampler out.
mkdir -p "$RUNS_REL" 2>/dev/null || true
dex "mkdir -p $PLOTS_DIR $VIDEO_DIR $RUNS_ABS"

hw_start

NVAL=$(val_frames)
echo "[pipe] val frames used by infer/dump = $NVAL"

# Plan: per step, predict train + tail separately; keep both for the band math.
declare -A LIMIT_OF TRAIN_EST_OF TAIL_EST_OF
TOTAL_PRED=0; PLAN=""
for N in $STEPS; do
  L=$(limit_for_step "$N")
  if [ -z "$L" ]; then echo "[pipe] !! cannot read manifest for n$N -- aborting"; exit 1; fi
  case "$L" in
    SKIP:*) avail=${L#SKIP:}
            echo "[pipe] n$N: only $avail segments extracted -> SKIP"
            ntfy "ladder: n$N skipped" "$(now_stamp)
only $avail segments extracted" ;;
    *)      LIMIT_OF[$N]=$L
            tr_secs=$(est_train_secs "$L"); tl_secs=$(est_tail_secs)
            TRAIN_EST_OF[$N]=$tr_secs; TAIL_EST_OF[$N]=$tl_secs
            step_secs=$((tr_secs + tl_secs)); TOTAL_PRED=$((TOTAL_PRED + step_secs))
            PLAN="$PLAN n$N(limit=$L,~$(fmt_hms "$step_secs"))" ;;
  esac
done
echo "[pipe] plan:$PLAN"
echo "[pipe] a-priori ETA (${#ARMS[@]} arms, $NSEEDS seeds, all stages): $(fmt_hms "$TOTAL_PRED")"

# Total individual trainings across the campaign = arms x seeds x RUNNABLE steps
# (SKIPped steps excluded). EXP_N counts up as each seed launches (in sweep_point),
# so every run start reports "i/TOTAL" and the step-DONE ping reports how many ran.
RUNNABLE_STEPS=0
for N in $STEPS; do [ -n "${LIMIT_OF[$N]:-}" ] && RUNNABLE_STEPS=$((RUNNABLE_STEPS + 1)); done
TOTAL_EXP=$(( ${#ARMS[@]} * NSEEDS * RUNNABLE_STEPS ))
EXP_N=0
echo "[pipe] total experiments = ${#ARMS[@]} arms x $NSEEDS seeds x $RUNNABLE_STEPS steps = $TOTAL_EXP"

ntfy "ladder START [$EXP]" "$(now_stamp)
$TOTAL_EXP experiments (${#ARMS[@]} arms x $NSEEDS seeds x $RUNNABLE_STEPS steps)
steps:$PLAN
total ~$(fmt_hms "$TOTAL_PRED") -> ends ~$(end_stamp "$TOTAL_PRED")" high

# Runtime calibration state (observed, learned from completed steps).
START=$(date +%s)
# train multiplier = observed_train / predicted_train (accumulated); 1.0 until measured.
OBS_TRAIN=0; PRED_TRAIN_DONE=0
# tail: keep the last observed fixed cost (per step) to feed the "best case".
LAST_TAIL=""
STEPS_ARR=($STEPS)

for idx in "${!STEPS_ARR[@]}"; do
  N=${STEPS_ARR[$idx]}
  [ -z "${LIMIT_OF[$N]:-}" ] && continue
  L=${LIMIT_OF[$N]}; POINT=n${N}_${STRIDE_TAG}
  pt_train=${TRAIN_EST_OF[$N]}; pt_tail=${TAIL_EST_OF[$N]}; pt_est=$((pt_train + pt_tail))

  echo "======================================================================"
  echo "[pipe] STEP n$N (limit_train=$L, point=$POINT)  est ~$(fmt_hms "$pt_est")"
  # START notification carries the ORIGINAL (a-priori) estimate for this step.
  ntfy "n$N START" "$(now_stamp)
limit=$L, est ~$(fmt_hms "$pt_est")"
  t0=$(date +%s)

  # --- train phase (timed on its own, for the train multiplier) ---
  for A in "${ARMS[@]}"; do
    IFS=: read -r ALABEL AMODEL ALR ADISP <<< "$A"
    sweep_point "$ALABEL" "$AMODEL" "$ALR" "$POINT" "$L"
  done
  t_train=$(date +%s); train_el=$((t_train - t0))

  # --- tail phase (infer + plots + video), timed on its own ---
  for A in "${ARMS[@]}"; do
    IFS=: read -r ALABEL AMODEL ALR ADISP <<< "$A"
    [ "$DO_INFER" = 1 ] && infer_point "$AMODEL" "$POINT"
  done
  [ "$DO_PLOTS" = 1 ] && plot_step "$POINT"
  [ "$DO_VIDEO" = 1 ] && video_step "$POINT"
  t1=$(date +%s); tail_el=$((t1 - t_train)); el=$((t1 - t0))

  # --- update calibration ---
  OBS_TRAIN=$((OBS_TRAIN + train_el)); PRED_TRAIN_DONE=$((PRED_TRAIN_DONE + pt_train))
  LAST_TAIL=$tail_el
  # train multiplier from all training seen so far (bounded >=1 handled by data).
  kf=$(awk -v o="$OBS_TRAIN" -v p="$PRED_TRAIN_DONE" 'BEGIN{printf "%.2f", (p>0? o/p : 1)}')

  # --- forecast the NEXT runnable step as a band ---
  next_line=""
  for ((j=idx+1; j<${#STEPS_ARR[@]}; j++)); do
    NN=${STEPS_ARR[$j]}
    [ -z "${LIMIT_OF[$NN]:-}" ] && continue
    n_train=${TRAIN_EST_OF[$NN]}
    # best  = predicted_train + observed_tail        (overhead is additive/fixed)
    # worst = predicted_train x multiplier + observed_tail
    best=$(awk  -v t="$n_train" -v tail="$LAST_TAIL" 'BEGIN{printf "%d", t+tail}')
    worst=$(awk -v t="$n_train" -v k="$kf" -v tail="$LAST_TAIL" 'BEGIN{printf "%d", t*k+tail}')
    next_line="next n$NN -- best ~$(fmt_hms "$best") * worst ~$(fmt_hms "$worst")
ends ~$(end_stamp "$best") (best) -> $(end_stamp "$worst") (worst)"
    break
  done
  [ -z "$next_line" ] && next_line="(last step)"

  # --- per-arm wall-clock breakdown for the DONE message ---
  # (train+tail are measured together per step; we report the step total per arm
  #  proportionally is misleading, so we report the two phase times instead.)
  echo "[pipe] step n$N done in $(fmt_hms "$el") (train $(fmt_hms "$train_el"), tail $(fmt_hms "$tail_el")); calib x$kf"
  ntfy "n$N DONE" "$(now_stamp) (took $(fmt_hms "$el"))
$EXP_N/$TOTAL_EXP runs launched
train $(fmt_hms "$train_el") * tail $(fmt_hms "$tail_el")
$next_line"
done

# ---- scaling plot (guarded) ------------------------------------------------
echo "[pipe] ladder training complete in $(fmt_hms "$(( $(date +%s) - START ))")"
if [ "$DO_PLOTS" = 1 ] && dex "[ -f $PLOT_SCALING ]"; then
  SCALE_ARMS=()
  for A in "${ARMS[@]}"; do IFS=: read -r L M _ <<< "$A"; SCALE_ARMS+=( --arm "$L=$M" ); done
  echo "[pipe] AP x N scaling plot -> $PLOTS_DIR"
  docker exec -i "$CONTAINER" python "$PLOT_SCALING" \
    --runs-root "$RUNS_ABS" --steps "$STEPS" --tag "$STRIDE_TAG" \
    --out-dir "$PLOTS_DIR" "${SCALE_ARMS[@]}" \
    || echo "[pipe] plot_scaling FAILED -- ledgers in $RUNS_REL/sweep_*.csv"
else
  echo "[pipe] (plot_scaling.py not found or plots off -- data in $RUNS_REL/sweep_*_n<N>_${STRIDE_TAG}.csv)"
fi

ntfy "ladder COMPLETE [$EXP]" "$(now_stamp)
total $(fmt_hms "$(( $(date +%s) - START ))")" high
echo "[pipe] DONE. plots in $PLOTS_DIR/<point>/ ; videos in $VIDEO_DIR/<point>/ ; ledgers $RUNS_REL/sweep_*.csv"
[ "$HW_MONITOR" = 1 ] && echo "[pipe] telemetry: $TELEMETRY_CSV"