#!/usr/bin/env bash
# Orchestrates the full pipeline across both containers, from the HOST:
#   GPU (train -> infer)  ->  CPU/TF (official metrics)  ->  consolidation
#
# Design (c): the HOST is the single source of truth for the run-id. It is
# generated here and passed --run-id to ALL stages. run_grid_gpu.py accepts
# --run-id verbatim, so there is NO stdout/file capture to parse.
#
# --cleanup HAZARD: run_grid_validate.py runs validate_motion_official with
# --cleanup ALWAYS (per combo). That deletes predictions AND the .infer_done
# sentinel right after each combo's metrics CSV is written. The metrics CSV
# survives, so consolidation is unaffected. BUT re-running STAGE=all/gpu on an
# already-validated RID makes the GPU phase re-INFER (no .infer_done -> not
# skipped). To re-report an existing run, use STAGE=consolidate, never all.
#
# EXIT-CODE NOTE: both grids ABSORB per-combo failures (record returncode, exit
# 0). `set -e` therefore catches grid/container-level failures (bad --model,
# container down, OOM, failing docker exec), NOT a single failed combo. See the
# hard-stop note at the tail of this file.
#
# FIRST RUN: always dry-run first ->  DRY_RUN=1 ./run_pipeline.sh

set -euo pipefail

# --- config (override via env) ----------------------------------------------
MODEL="${MODEL:-social}"          # MODELS_CFG entry: vectorized|social|map|...
SEEDS="${SEEDS:-0-1}"             # '0-7' (range) or '0,1,2' (list)
STAGE="${STAGE:-all}"             # all | gpu | validate | consolidate
GPU="${GPU_CONTAINER:-gpu_env}"   # PyTorch/CUDA container
CPU="${CPU_CONTAINER:-cpu_env}"   # TF/metrics container
WORKDIR="${WORKDIR:-/workspace}"  # shared mount; CWD so `-m src.motion.*` resolves
SCRIPTS="${SCRIPTS:-scripts}"     # dir (relative to WORKDIR) holding the runners
DRY="${DRY_RUN:-0}"               # 1 -> echo the chain, execute nothing

# --- stages that target existing artifacts REQUIRE an explicit RUN_ID --------
if [[ "$STAGE" != "all" && "$STAGE" != "gpu" && -z "${RUN_ID:-}" ]]; then
  echo "[err] STAGE=$STAGE targets existing artifacts; set RUN_ID=<existing>." >&2
  exit 2
fi

# --- run-id: host-generated, same convention as make_run_id() ---------------
# <model>_<YYYY-MM-DD>_<hex4>. Pass RUN_ID=<existing> to RESUME/target a run.
HEX4="$(python3 -c 'import secrets;print(secrets.token_hex(2))' 2>/dev/null \
        || printf '%04x' $((RANDOM % 65536)))"
RID="${RUN_ID:-${MODEL}_$(date +%F)_${HEX4}}"
echo "[run-id] ${RID}   (model=${MODEL} seeds=${SEEDS} stage=${STAGE} dry=${DRY})"

# --- helper: run one stage in a container (or echo it in dry-run) ------------
run() {
  local container="$1"; shift
  if [[ "$DRY" == "1" ]]; then
    echo "docker exec -w ${WORKDIR} ${container} $*"
  else
    docker exec -w "${WORKDIR}" "${container}" "$@"
  fi
}

# --- stages -----------------------------------------------------------------
if [[ "$STAGE" == "all" || "$STAGE" == "gpu" ]]; then
  run "$GPU" python3 "$SCRIPTS/run_grid_gpu.py"      --model "$MODEL" --run-id "$RID" --seeds "$SEEDS"
fi
if [[ "$STAGE" == "all" || "$STAGE" == "validate" ]]; then
  run "$CPU" python3 "$SCRIPTS/run_grid_validate.py" --model "$MODEL" --run-id "$RID" --seeds "$SEEDS"
fi
if [[ "$STAGE" == "all" || "$STAGE" == "consolidate" ]]; then
  run "$GPU" python3 "$SCRIPTS/consolidate_seeds.py" --model "$MODEL" --run-id "$RID" --seeds "$SEEDS"
  run "$GPU" python3 "$SCRIPTS/consolidate_gpu.py"                    --run-id "$RID"
  run "$GPU" python3 "$SCRIPTS/report.py"                             --run-id "$RID"
fi

echo "[done] stage=${STAGE} complete for run-id ${RID}"

# --- OPTIONAL hard-stop on any failed combo (not enabled) -------------------
# If you want the chain to abort when a combo failed (instead of consolidating
# partial results), the clean fix is one line in run_grid_gpu.py: after the
# combo loop, exit non-zero if any phase_index returncode != 0. Then `set -e`
# here catches it at the gpu stage. Do that as its own diff -- the host stays a
# dumb driver.