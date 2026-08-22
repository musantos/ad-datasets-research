#!/usr/bin/env python3
"""
V0 grid (vectorized), METRICS phase. Run IN THE CPU/TF (metrics) CONTAINER, AFTER
run_grid_gpu_vectorized.py has produced the 8 prediction folders in the GPU container.

For each (model, arm, seed) it calls the official evaluator on that prediction
folder:

    python3 -m src.motion.validate_motion_official --pred-dir <dir> --cleanup

RESUMABLE: skips a combo whose metrics CSV already exists (glob match), so a
crash or a partial run just resumes. Safe to relaunch.

IMPORTANT -- confirm two things match your setup before the first run:
  * MODULE PATH: validate lives at src.motion.validate_motion_official (this is
    the path that worked interactively). If it's elsewhere, fix VALIDATE_MODULE.
  * METRICS OUTPUT: this script assumes validate writes a CSV named
    metrics_<rundir>_<date>.csv. If your evaluator writes elsewhere or with a
    different name, adjust METRICS_DIR / metrics_glob() so the skip check is
    correct -- otherwise it will re-validate everything (harmless but slow) or
    wrongly skip. When unsure, run once for a single dir and check where the
    CSV lands.
"""
import os
import sys
import glob
import subprocess

MODELS = ["vectorized"]
ARMS = ["agent"]
VARIANTS = ["raw", "std"]        # "std" -> pred dir e metrics CSV com sufixo _std
SEEDS = list(range(8))
CLS_WEIGHT = 20

PRED_ROOT = "/workspace/datasets/waymo/predictions"
VALIDATE_MODULE = "src.motion.validate_motion_official"

# Where the metrics CSVs land. Adjust if your evaluator writes elsewhere.
METRICS_DIR = os.environ.get("METRICS_DIR", "/workspace/experiments/metrics")

PYEXE = sys.executable or "python3"


def run_tag(arm, seed, variant):
    t = f"cls{CLS_WEIGHT:g}_{arm}_seed{seed}"
    return f"{t}_std" if variant == "std" else t


def pred_dir(model, arm, seed, variant):
    return os.path.join(PRED_ROOT, f"{model}_{run_tag(arm, seed, variant)}")


def metrics_glob(model, arm, seed, variant):
    # matches metrics_<rundir>_<date>.csv regardless of the date suffix
    rundir = f"{model}_{run_tag(arm, seed, variant)}"
    return os.path.join(METRICS_DIR, f"metrics_{rundir}_*.csv")


def main():
    total = len(MODELS) * len(ARMS) * len(VARIANTS) * len(SEEDS)
    done = 0
    n_ok = 0
    n_skip = 0
    n_fail = 0
    missing_preds = []

    for model in MODELS:
        for arm in ARMS:
            for variant in VARIANTS:
                for seed in SEEDS:
                    done += 1
                    d = pred_dir(model, arm, seed, variant)
                    tag = f"{model}/{arm}/{variant}/seed{seed}"

                    if not os.path.isdir(d) or not glob.glob(os.path.join(d, "*.npy")):
                        print(f"[warn] {done}/{total} {tag}: no predictions at {d} "
                              f"(run the GPU phase first). Skipping.")
                        missing_preds.append(tag)
                        continue

                    if glob.glob(metrics_glob(model, arm, seed, variant)):
                        print(f"[skip] {done}/{total} {tag}: metrics CSV already present.")
                        n_skip += 1
                        continue

                    cmd = [PYEXE, "-m", VALIDATE_MODULE, "--pred-dir", d, "--cleanup"]
                    print(f"\n{'='*78}\n[val ] {done}/{total} {tag}\n       {' '.join(cmd)}\n{'='*78}")
                    rc = subprocess.run(cmd).returncode
                    if rc == 0:
                        n_ok += 1
                    else:
                        print(f"[FAIL] {tag} exited {rc}.")
                        n_fail += 1

    print(f"\n[DONE] metrics phase: {n_ok} evaluated, {n_skip} skipped, {n_fail} failed.")
    if missing_preds:
        print(f"       {len(missing_preds)} combos had no predictions: {missing_preds}")
    print(f"\nMetrics CSVs in: {METRICS_DIR}")
    print("Consolidate them into one table (model,arm,seed,breakdown,minADE,...) "
          "for the V0 (1-config) x 8-seed comparison with confidence intervals.")


if __name__ == "__main__":
    main()