#!/usr/bin/env python3
"""
GENERIC grid, GPU phase (train -> infer). Run IN THE GPU CONTAINER. The metrics
phase (validate_motion_official) runs separately in the CPU/TF container -- see
run_grid_validate.py.

This is the UNIFIER of the old run_grid_gpu.py (item4) and run_grid_gpu_vectorized.py
(V0). The method becomes an ARGUMENT (--model), not a new file per step. All the
difference between methods lives in the MODELS_CFG REGISTRY below; the machinery
(sentinels, gpu-loggers, phase-index) is identical to the previous versions.

  python3 run_grid_gpu.py --model vectorized      # V0: {agent} x {raw,std} x 8  = 16
  python3 run_grid_gpu.py --model social          # V1: {agent} x {raw,std} x 8  = 16
  python3 run_grid_gpu.py --model multimodal --model2 sequential   # item4 (see note)
  python3 run_grid_gpu.py --model social --dry-run  # prints the commands, does not run

Design (unchanged vs the per-method versions):
  * RESUMABLE / IDEMPOTENT via .train_done / .infer_done sentinels.
  * GPU LOGGERS attached (state + procs) and killed on exit.
  * PHASE INDEX per config (time-join with the gpu_logs).

This script does NOT touch the science: it only calls the train_*/run_inference_*
modules with the same args you would type by hand.

--- THE GOTCHA the registry solves ---------------------------------------------
'social' is the only case where the MODULE NAME differs from the FOLDER PREFIX:
  module  = src.motion.train_social / run_inference_social
  folder  = vectorized_social_<run_tag>   (checkpoints, predictions, logs, metrics)
On top of that, social does NOT accept --agent-centric (agent arm is implicit) and
REQUIRES --n-neighbors. That is why 'module', 'folder', 'agent_centric' and 'extra_*'
are separate fields in the registry.
"""
import os
import sys
import time
import csv
import signal
import secrets
import argparse
import subprocess
from datetime import datetime, date

# --- MODEL REGISTRY ----------------------------------------------------------
# module        : module suffix -> src.motion.train_<module> / run_inference_<module>
# folder        : folder/CSV prefix -> <folder>_<run_tag> (checkpoints/pred/logs/metrics)
# arms          : arms to sweep (sdc|agent)
# variants      : raw and/or std (std appends --standardize and suffixes _std)
# agent_centric : if True, appends --agent-centric when arm=='agent'
#                 (False for social: agent arm is IMPLICIT, the script rejects the flag)
# extra_train   : extra fixed flags at training  (e.g.: social's --n-neighbors)
# extra_infer   : extra fixed flags at inference (idem; MUST match training)
MODELS_CFG = {
    "multimodal": dict(module="multimodal", folder="multimodal",
                       arms=["sdc", "agent"], variants=["raw"],
                       agent_centric=True, extra_train=[], extra_infer=[]),
    "sequential": dict(module="sequential", folder="sequential",
                       arms=["sdc", "agent"], variants=["raw"],
                       agent_centric=True, extra_train=[], extra_infer=[]),
    "vectorized": dict(module="vectorized", folder="vectorized",
                       arms=["agent"], variants=["raw", "std"],
                       agent_centric=True, extra_train=[], extra_infer=[]),
    "social":     dict(module="social", folder="vectorized_social",
                       arms=["agent"], variants=["raw", "std"],
                       agent_centric=False,
                       extra_train=["--n-neighbors", "16"],
                       extra_infer=["--n-neighbors", "16"]),
    # V2 (map-aware, raw-only). module 'map' -> train_map / run_inference_map;
    # folder 'vectorized_social_map'. agent_centric=False (arm implicit, no
    # --agent-centric flag). M/Np frozen a priori (calibrated on cache_val_map)
    # and MUST match the constants in train_map/run_inference_map.
    "map":        dict(module="map", folder="vectorized_social_map",
                       arms=["agent"], variants=["raw"],
                       agent_centric=False,
                       extra_train=["--n-neighbors", "16",
                                    "--n-map-polylines", "128",
                                    "--n-points-per-polyline", "20"],
                       extra_infer=["--n-neighbors", "16",
                                    "--n-map-polylines", "128",
                                    "--n-points-per-polyline", "20"]),
}

CLS_WEIGHT_DEFAULT = 20          # used if --cls-weights is not passed

# --- paths (match train_*/run_inference_*) -----------------------------------
# Env-override, default = today's value (same pattern as run_grid_validate's
# METRICS_DIR). Without env AND without --run-id these stay at today's roots
# (backward-compat). apply_run_id() nests them under runs/<run-id> at runtime.
CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "/workspace/experiments/checkpoints")
PRED_ROOT = os.environ.get("PRED_ROOT", "/workspace/datasets/waymo/predictions")
GPULOG_DIR = os.environ.get("GPULOG_DIR", "/workspace/experiments/gpu_logs")
NVIDIA_SMI_INTERVAL = 1          # seconds, matches your manual -l 5

PYEXE = sys.executable or "python3"

_loggers = []
_logfiles = []


def ts():
    """Local wall-clock string matching nvidia-smi's format prefix."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def make_run_id(model):
    """Grid run-id: <model>_<YYYY-MM-DD>_<hex4>. hex4 disambiguates same-day runs
    of the same method. Generated ONCE per grid by this runner (leaves never
    invent one)."""
    return f"{model}_{date.today():%Y-%m-%d}_{secrets.token_hex(2)}"


def apply_run_id(run_id):
    """Nest the write-roots under runs/<run_id> and export them so the
    train_*/run_inference_* children inherit the SAME location (subprocess env
    inheritance). One mechanism: this runner builds sentinels/loggers from the
    nested globals, the children write artifacts from the nested env -> both
    co-located. The run-id NESTS under each root; it never moves the root
    (roots live on different disks). *_CACHE and STATS_PATH stay OUTSIDE the
    run-id (input / shared) and are not touched here."""
    global CHECKPOINT_ROOT, PRED_ROOT, GPULOG_DIR
    CHECKPOINT_ROOT = os.path.join(CHECKPOINT_ROOT, "runs", run_id)
    PRED_ROOT = os.path.join(PRED_ROOT, "runs", run_id)
    GPULOG_DIR = os.path.join(GPULOG_DIR, "runs", run_id)
    os.environ["CHECKPOINT_ROOT"] = CHECKPOINT_ROOT
    os.environ["PRED_ROOT"] = PRED_ROOT
    os.environ["GPULOG_DIR"] = GPULOG_DIR
    # LOG_ROOT is written by the train_* children, not by this runner, so it is
    # not a global here; still export it (nested under its own default root) so
    # the per-epoch CSVs land inside the run too.
    log_root = os.environ.get("LOG_ROOT", "/workspace/experiments/logs")
    os.environ["LOG_ROOT"] = os.path.join(log_root, "runs", run_id)


def write_run_txt(run_id, cfg, cls_weights, seeds, model):
    """One run.txt per grid, written at the start under GPULOG_DIR (experiments
    side, guaranteed writable by the GPU runner). Minimal provenance; per-seed
    timestamps already live in logs/*.csv and phase_index_*.csv, not duplicated."""
    path = os.path.join(GPULOG_DIR, "run.txt")
    with open(path, "w") as f:
        f.write(f"run_id      : {run_id}\n")
        f.write(f"model       : {model}  (folder={cfg['folder']})\n")
        f.write(f"opened_at   : {ts()}\n")
        f.write(f"cls_weights : {cls_weights}\n")
        f.write(f"seeds       : {seeds[0]}..{seeds[-1]}\n")
        f.write(f"arms        : {cfg['arms']}\n")
        f.write(f"variants    : {cfg['variants']}\n")
        f.write(f"extra_train : {' '.join(cfg['extra_train'])}\n")
        f.write(f"argv        : {' '.join(sys.argv)}\n")
    print(f"[run.txt] {path}")


def start_gpu_loggers():
    """Launch the two nvidia-smi loggers you used, in the background."""
    os.makedirs(GPULOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    state_path = os.path.join(GPULOG_DIR, f"gpulog_state_{stamp}.csv")
    procs_path = os.path.join(GPULOG_DIR, f"gpulog_procs_{stamp}.csv")

    state_cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,"
        "power.draw,temperature.gpu,pstate",
        "--format=csv", "-l", str(NVIDIA_SMI_INTERVAL),
    ]
    procs_cmd = [
        "nvidia-smi",
        "--query-compute-apps=timestamp,pid,process_name,used_memory",
        "--format=csv", "-l", str(NVIDIA_SMI_INTERVAL),
    ]

    for cmd, path in [(state_cmd, state_path), (procs_cmd, procs_path)]:
        try:
            fh = open(path, "w")
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.DEVNULL)
            _loggers.append(p)
            _logfiles.append(fh)
            print(f"[gpu-logger] {path}  (pid {p.pid})")
        except FileNotFoundError:
            print("[gpu-logger] WARNING: nvidia-smi not found; skipping GPU logging.")
            return None, None
    return state_path, procs_path


def stop_gpu_loggers():
    for p in _loggers:
        try:
            p.terminate()
            p.wait(timeout=10)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    for fh in _logfiles:
        try:
            fh.close()
        except Exception:
            pass


def run_tag(cls, arm, seed, variant):
    # The WEIGHT goes into the run_tag (cls<w>) -> different weights NEVER collide in
    # the same folder. That is what makes the weight sweep safe (no silent overwrite).
    t = f"cls{cls:g}_{arm}_seed{seed}"
    return f"{t}_std" if variant == "std" else t


def train_sentinel(cfg, cls, arm, seed, variant):
    d = os.path.join(CHECKPOINT_ROOT, f"{cfg['folder']}_{run_tag(cls, arm, seed, variant)}")
    return os.path.join(d, ".train_done")


def infer_sentinel(cfg, cls, arm, seed, variant):
    d = os.path.join(PRED_ROOT, f"{cfg['folder']}_{run_tag(cls, arm, seed, variant)}")
    return os.path.join(d, ".infer_done")


def build_train_cmd(cfg, cls, arm, seed, variant):
    cmd = [PYEXE, "-m", f"src.motion.train_{cfg['module']}",
           "--cls-weight", str(cls), "--seed", str(seed)]
    if cfg["agent_centric"] and arm == "agent":
        cmd.append("--agent-centric")
    if variant == "std":
        cmd.append("--standardize")
    cmd += cfg["extra_train"]
    return cmd


def build_infer_cmd(cfg, cls, arm, seed, variant):
    cmd = [PYEXE, "-m", f"src.motion.run_inference_{cfg['module']}",
           "--tag", f"cls{cls:g}", "--seed", str(seed)]
    if cfg["agent_centric"] and arm == "agent":
        cmd.append("--agent-centric")
    if variant == "std":
        cmd.append("--standardize")
    cmd += cfg["extra_infer"]
    return cmd


def run_phase(index_writer, index_fh, folder, cls, arm, variant, seed, phase, cmd, sentinel):
    """Run one train/infer phase, logging its time window to the phase index."""
    label = f"{folder}/cls{cls:g}/{arm}/{variant}/seed{seed}/{phase}"
    if os.path.exists(sentinel):
        print(f"[skip] {label} (sentinel present)")
        return True

    print(f"\n{'='*78}\n[run ] {label}\n"
          f"       {' '.join(cmd)}\n{'='*78}")
    start_iso, start_epoch = ts(), time.time()
    proc = subprocess.run(cmd)
    end_iso, end_epoch = ts(), time.time()

    index_writer.writerow([
        folder, cls, arm, variant, seed, phase, start_iso, end_iso,
        f"{start_epoch:.3f}", f"{end_epoch:.3f}",
        f"{end_epoch - start_epoch:.1f}", proc.returncode,
    ])
    index_fh.flush()

    if proc.returncode != 0:
        print(f"[FAIL] {label} "
              f"exited {proc.returncode}; NOT writing sentinel (will retry on relaunch).")
        return False

    os.makedirs(os.path.dirname(sentinel), exist_ok=True)
    with open(sentinel, "w") as f:
        f.write(f"done {end_iso}\n")
    return True


def iter_combos(cfg, cls_weights, seeds):
    for cls in cls_weights:
        for arm in cfg["arms"]:
            for variant in cfg["variants"]:
                for seed in seeds:
                    yield cls, arm, variant, seed


def dry_run(cfg, cls_weights, seeds):
    print(f"[DRY-RUN] model={cfg['folder']} (module={cfg['module']})  "
          f"cls_weights={cls_weights} arms={cfg['arms']} "
          f"variants={cfg['variants']} seeds={list(seeds)}")
    n = 0
    for cls, arm, variant, seed in iter_combos(cfg, cls_weights, seeds):
        n += 1
        print(f"\n# combo {n}: {cfg['folder']} / cls{cls:g} / {arm} / {variant} / seed{seed}")
        print("  train:", " ".join(build_train_cmd(cfg, cls, arm, seed, variant)))
        print("  infer:", " ".join(build_infer_cmd(cfg, cls, arm, seed, variant)))
        print(f"  ckpt : {CHECKPOINT_ROOT}/{cfg['folder']}_{run_tag(cls, arm, seed, variant)}/")
        print(f"  pred : {PRED_ROOT}/{cfg['folder']}_{run_tag(cls, arm, seed, variant)}/")
    print(f"\n[DRY-RUN] {n} combos ({n} train + {n} infer). Nothing was executed.")


def main():
    ap = argparse.ArgumentParser(description="Generic GPU grid (train->infer).")
    ap.add_argument("--model", required=True, choices=sorted(MODELS_CFG),
                    help="method to run (MODELS_CFG registry entry).")
    ap.add_argument("--cls-weights", default=str(CLS_WEIGHT_DEFAULT),
                    help="weight(s) of the classification term. A single value "
                         f"(default {CLS_WEIGHT_DEFAULT}) or list '1,20,50' to sweep. "
                         "The weight goes into the run_tag (cls<w>), so weights do not collide.")
    ap.add_argument("--seeds", default="0-7",
                    help="seeds: '0-7' (range) or '0,1,2' (list). Default 0-7.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and paths without executing (for review).")
    ap.add_argument("--run-id", default=None,
                    help="grid run-id for isolation. Passed -> used verbatim "
                         "(resume). Absent + real run -> auto-generated "
                         "(<model>_<date>_<hex4>). Absent + --dry-run -> today's "
                         "roots (backward-compat), no runs/<id> nesting.")
    args = ap.parse_args()

    cfg = MODELS_CFG[args.model]

    # weight: accepts '20' (a single value) or '1,20,50' (list). g-format drops trailing zeros.
    cls_weights = [float(w) for w in args.cls_weights.split(",") if w.strip() != ""]

    if "-" in args.seeds:
        a, b = args.seeds.split("-")
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    if args.dry_run:
        # Nest only when a run-id is EXPLICIT; without one, preview today's roots
        # (backward-compat proof, smoke §9.2). Real runs auto-generate below.
        if args.run_id:
            apply_run_id(args.run_id)
        dry_run(cfg, cls_weights, seeds)
        return

    # Real run: resolve the run-id (generate if absent) and nest BEFORE anything
    # writes. Children inherit the nested roots via subprocess env inheritance.
    run_id = args.run_id or make_run_id(args.model)
    apply_run_id(run_id)
    print(f"[run-id] {run_id}")
    print(f"         checkpoints -> {CHECKPOINT_ROOT}")
    print(f"         predictions -> {PRED_ROOT}")
    print(f"         gpu_logs    -> {GPULOG_DIR}")

    os.makedirs(GPULOG_DIR, exist_ok=True)
    write_run_txt(run_id, cfg, cls_weights, seeds, args.model)
    state_path, procs_path = start_gpu_loggers()

    def _cleanup(signum=None, frame=None):
        stop_gpu_loggers()
        if signum is not None:
            sys.exit(130)
    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    index_path = os.path.join(GPULOG_DIR, f"phase_index_{stamp}.csv")
    index_fh = open(index_path, "w", newline="")
    index_fh.write(f"# model={cfg['folder']} cls_weights={cls_weights} "
                   f"seeds={seeds[0]}..{seeds[-1]}\n")
    index_fh.write(f"# gpu_state={os.path.basename(state_path) if state_path else 'NA'} "
                   f"gpu_procs={os.path.basename(procs_path) if procs_path else 'NA'}\n")
    index_writer = csv.writer(index_fh)
    index_writer.writerow([
        "model", "cls_weight", "arm", "variant", "seed", "phase",
        "start_iso", "end_iso", "start_epoch", "end_epoch",
        "duration_s", "returncode",
    ])
    index_fh.flush()
    print(f"[phase-index] {index_path}")

    combos = list(iter_combos(cfg, cls_weights, seeds))
    total = len(combos)
    done = 0
    n_fail = 0
    grid_start = time.time()

    try:
        for cls, arm, variant, seed in combos:
            done += 1
            print(f"\n########## combo {done}/{total}: "
                  f"{cfg['folder']} / cls{cls:g} / {arm} / {variant} / seed{seed} ##########")

            ok = run_phase(index_writer, index_fh, cfg["folder"], cls, arm, variant, seed,
                           "train", build_train_cmd(cfg, cls, arm, seed, variant),
                           train_sentinel(cfg, cls, arm, seed, variant))
            if not ok:
                print("       -> skipping inference for this combo (train failed).")
                n_fail += 1
                continue

            ok_infer = run_phase(index_writer, index_fh, cfg["folder"], cls, arm, variant, seed,
                                 "infer", build_infer_cmd(cfg, cls, arm, seed, variant),
                                 infer_sentinel(cfg, cls, arm, seed, variant))
            if not ok_infer:
                n_fail += 1
    finally:
        index_fh.close()
        stop_gpu_loggers()

    dur = time.time() - grid_start
    print(f"\n[DONE] GPU phase finished in {dur/60:.1f} min.")
    print(f"       Phase index: {index_path}")
    if state_path:
        print(f"       GPU state:   {state_path}")
        print(f"       GPU procs:   {procs_path}")
    print("\nNext, in the METRICS container (CPU/TF):")
    print(f"       python3 run_grid_validate.py --model {args.model} --run-id {run_id}")

    # Guard 1 (hard-stop by returncode): if any combo's train/infer subprocess
    # exited != 0, exit 1 so the orchestrator's `set -e` aborts at the GPU stage.
    # Printed AFTER the summary so the operator keeps the phase-index path and the
    # next-step hint. Catches a REAL crash (rc!=0); it cannot catch a rc=0
    # sentinel trap (0 .npy with rc=0) -- Guard 2 in consolidate_seeds.py does.
    if n_fail > 0:
        print(f"[GUARD] {n_fail} combo(s) failed (rc!=0); exiting 1.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()