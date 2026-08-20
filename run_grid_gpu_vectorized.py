#!/usr/bin/env python3
"""
V0 grid (vectorized), GPU phase (train -> infer), run IN THE GPU
CONTAINER. The metrics phase (validate_motion_official) runs separately in the
CPU/TF container -- see run_grid_validate.py.

V0 (Fase 2, vetorizado): {vectorized} x {agent} x seeds 0..7, cls20.
That is 1 config x 8 seeds = 8 train runs + 8 inference runs.

Design:
  * RESUMABLE / IDEMPOTENT. Each (model, arm, seed) writes a sentinel on clean
    completion (.train_done / .infer_done). A combo is skipped only if its
    sentinel exists, so a crash mid-run re-runs that combo (a half-written
    best.pth is NOT mistaken for a finished run). Safe to Ctrl-C and relaunch.
  * GPU LOGGER ATTACHED. Two nvidia-smi loggers (state + procs) start in the
    background here and are killed on exit, so you never launch them by hand.
  * PHASE INDEX. Every train/infer phase is logged with start/end timestamps to
    a third CSV, so the GPU logs can be attributed PER CONFIG (join on time),
    instead of one undifferentiated blob. Same wall clock as nvidia-smi.

This script does NOT touch the science: it only calls the existing train_*/
run_inference_* modules with the same args you would type by hand.
"""
import os
import sys
import time
import csv
import signal
import subprocess
from datetime import datetime

# --- grid definition (opcao C: cls fixed at 20) ------------------------------
MODELS = ["vectorized"]
ARMS = ["agent"]          # "agent" adds --agent-centric
SEEDS = list(range(8))           # 0..7  (fixed N, decided up front)
CLS_WEIGHT = 20

# --- paths (match train_*/run_inference_*) -----------------------------------
CHECKPOINT_ROOT = "/workspace/experiments/checkpoints"
PRED_ROOT = "/workspace/datasets/waymo/predictions"
GPULOG_DIR = "/workspace/experiments/gpu_logs"
NVIDIA_SMI_INTERVAL = 5          # seconds, matches your manual -l 5

PYEXE = sys.executable or "python3"

# background logger handles, closed on exit
_loggers = []
_logfiles = []


def ts():
    """Local wall-clock string matching nvidia-smi's format prefix."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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


def run_tag(arm, seed):
    return f"cls{CLS_WEIGHT:g}_{arm}_seed{seed}"


def train_sentinel(model, arm, seed):
    d = os.path.join(CHECKPOINT_ROOT, f"{model}_{run_tag(arm, seed)}")
    return os.path.join(d, ".train_done")


def infer_sentinel(model, arm, seed):
    d = os.path.join(PRED_ROOT, f"{model}_{run_tag(arm, seed)}")
    return os.path.join(d, ".infer_done")


def build_train_cmd(model, arm, seed):
    cmd = [PYEXE, "-m", f"src.motion.train_{model}",
           "--cls-weight", str(CLS_WEIGHT), "--seed", str(seed)]
    if arm == "agent":
        cmd.append("--agent-centric")
    return cmd


def build_infer_cmd(model, arm, seed):
    cmd = [PYEXE, "-m", f"src.motion.run_inference_{model}",
           "--tag", f"cls{CLS_WEIGHT:g}", "--seed", str(seed)]
    if arm == "agent":
        cmd.append("--agent-centric")
    return cmd


def run_phase(index_writer, index_fh, model, arm, seed, phase, cmd, sentinel):
    """Run one train/infer phase, logging its time window to the phase index."""
    if os.path.exists(sentinel):
        print(f"[skip] {model}/{arm}/seed{seed}/{phase} (sentinel present)")
        return True

    print(f"\n{'='*78}\n[run ] {model}/{arm}/seed{seed}/{phase}\n"
          f"       {' '.join(cmd)}\n{'='*78}")
    start_iso, start_epoch = ts(), time.time()
    proc = subprocess.run(cmd)
    end_iso, end_epoch = ts(), time.time()

    index_writer.writerow([
        model, arm, seed, phase, start_iso, end_iso,
        f"{start_epoch:.3f}", f"{end_epoch:.3f}",
        f"{end_epoch - start_epoch:.1f}", proc.returncode,
    ])
    index_fh.flush()

    if proc.returncode != 0:
        print(f"[FAIL] {model}/{arm}/seed{seed}/{phase} "
              f"exited {proc.returncode}; NOT writing sentinel (will retry on relaunch).")
        return False

    # clean completion -> drop the sentinel so relaunch skips it
    os.makedirs(os.path.dirname(sentinel), exist_ok=True)
    with open(sentinel, "w") as f:
        f.write(f"done {end_iso}\n")
    return True


def main():
    os.makedirs(GPULOG_DIR, exist_ok=True)
    state_path, procs_path = start_gpu_loggers()

    # ensure loggers die even on Ctrl-C / kill
    def _cleanup(signum=None, frame=None):
        stop_gpu_loggers()
        if signum is not None:
            sys.exit(130)
    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    index_path = os.path.join(GPULOG_DIR, f"phase_index_{stamp}.csv")
    index_fh = open(index_path, "w", newline="")
    index_fh.write(f"# grid=itemC cls_weight={CLS_WEIGHT} seeds={SEEDS[0]}..{SEEDS[-1]}\n")
    index_fh.write(f"# gpu_state={os.path.basename(state_path) if state_path else 'NA'} "
                   f"gpu_procs={os.path.basename(procs_path) if procs_path else 'NA'}\n")
    index_writer = csv.writer(index_fh)
    index_writer.writerow([
        "model", "arm", "seed", "phase",
        "start_iso", "end_iso", "start_epoch", "end_epoch",
        "duration_s", "returncode",
    ])
    index_fh.flush()
    print(f"[phase-index] {index_path}")

    total = len(MODELS) * len(ARMS) * len(SEEDS)
    done = 0
    grid_start = time.time()

    try:
        for model in MODELS:
            for arm in ARMS:
                for seed in SEEDS:
                    done += 1
                    print(f"\n########## combo {done}/{total}: "
                          f"{model} / {arm} / seed{seed} ##########")

                    ok = run_phase(index_writer, index_fh, model, arm, seed,
                                   "train", build_train_cmd(model, arm, seed),
                                   train_sentinel(model, arm, seed))
                    if not ok:
                        print("       -> skipping inference for this combo (train failed).")
                        continue

                    run_phase(index_writer, index_fh, model, arm, seed,
                              "infer", build_infer_cmd(model, arm, seed),
                              infer_sentinel(model, arm, seed))
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
    print("       python3 run_grid_validate_vectorized.py")


if __name__ == "__main__":
    main()
