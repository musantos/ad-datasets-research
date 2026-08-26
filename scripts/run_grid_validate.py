#!/usr/bin/env python3
"""
GENERIC grid, METRICS phase. Run IN THE CPU/TF (metrics) CONTAINER, AFTER
run_grid_gpu.py produced the prediction folders in the GPU container.

Unifies run_grid_validate.py (item4) and run_grid_validate_vectorized.py (V0). The
method becomes an ARGUMENT (--model); the MODELS_CFG registry (the SAME concept as
run_grid_gpu.py) defines arms/variants and the FOLDER PREFIX of each method.

  python3 run_grid_validate.py --model vectorized
  python3 run_grid_validate.py --model social

For each (arm, variant, seed) it calls the official evaluator on the predictions folder:
    python3 -m src.motion.validate_motion_official --pred-dir <dir> --cleanup

RESUMABLE: skips a combo whose metrics CSV already exists (glob), so a crash or
partial run just resumes. Safe to relaunch.

IMPORTANT -- check before the first run:
  * MODULE PATH: validate lives in src.motion.validate_motion_official.
  * METRICS OUTPUT: assumes a CSV named metrics_<rundir>_<date>.csv in METRICS_DIR.
    If the evaluator writes elsewhere/another name, adjust METRICS_DIR / metrics_glob(),
    otherwise the skip re-validates everything (harmless, slow) or skips wrongly.
"""
import os
import sys
import glob
import argparse
import subprocess

# Only the fields used in the metrics phase (folder/arms/variants). Kept in
# sync with the run_grid_gpu.py registry -- if a new method is added there,
# replicate the entry here (or, in the future, extract to a shared module).
MODELS_CFG = {
    "multimodal": dict(folder="multimodal",        arms=["sdc", "agent"], variants=["raw"]),
    "sequential": dict(folder="sequential",        arms=["sdc", "agent"], variants=["raw"]),
    "vectorized": dict(folder="vectorized",        arms=["agent"],        variants=["raw", "std"]),
    "social":     dict(folder="vectorized_social", arms=["agent"],        variants=["raw", "std"]),
    "map":        dict(folder="vectorized_social_map", arms=["agent"],    variants=["raw"]),
}

CLS_WEIGHT_DEFAULT = 20          # used if --cls-weights is not passed

PRED_ROOT = "/workspace/datasets/waymo/predictions"
VALIDATE_MODULE = "src.motion.validate_motion_official"
METRICS_DIR = os.environ.get("METRICS_DIR", "/workspace/results")

PYEXE = sys.executable or "python3"


def run_tag(cls, arm, seed, variant):
    t = f"cls{cls:g}_{arm}_seed{seed}"
    return f"{t}_std" if variant == "std" else t


def pred_dir(folder, cls, arm, seed, variant):
    return os.path.join(PRED_ROOT, f"{folder}_{run_tag(cls, arm, seed, variant)}")


def metrics_glob(folder, cls, arm, seed, variant):
    # matches metrics_<rundir>_<date>.csv regardless of the date suffix
    rundir = f"{folder}_{run_tag(cls, arm, seed, variant)}"
    return os.path.join(METRICS_DIR, f"metrics_{rundir}_*.csv")


def main():
    ap = argparse.ArgumentParser(description="Generic metrics grid (validate).")
    ap.add_argument("--model", required=True, choices=sorted(MODELS_CFG))
    ap.add_argument("--cls-weights", default=str(CLS_WEIGHT_DEFAULT),
                    help="weight(s): a single value (default) or list '1,20,50'. "
                         "MUST match the weights run in the GPU phase.")
    ap.add_argument("--seeds", default="0-7",
                    help="seeds: '0-7' (range) or '0,1,2' (list). Default 0-7.")
    args = ap.parse_args()

    cfg = MODELS_CFG[args.model]
    cls_weights = [float(w) for w in args.cls_weights.split(",") if w.strip() != ""]
    if "-" in args.seeds:
        a, b = args.seeds.split("-")
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    combos = [(cls, arm, variant, seed)
              for cls in cls_weights
              for arm in cfg["arms"]
              for variant in cfg["variants"]
              for seed in seeds]
    total = len(combos)
    done = n_ok = n_skip = n_fail = 0
    missing_preds = []

    for cls, arm, variant, seed in combos:
        done += 1
        d = pred_dir(cfg["folder"], cls, arm, seed, variant)
        tag = f"{cfg['folder']}/cls{cls:g}/{arm}/{variant}/seed{seed}"

        if not os.path.isdir(d) or not glob.glob(os.path.join(d, "*.npy")):
            print(f"[warn] {done}/{total} {tag}: no predictions at {d} "
                  f"(run the GPU phase first). Skipping.")
            missing_preds.append(tag)
            continue

        if glob.glob(metrics_glob(cfg["folder"], cls, arm, seed, variant)):
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
    print(f"Consolidate: python3 consolidate_seeds.py --model {args.model} "
          f"--cls-weights {args.cls_weights} --results-dir {METRICS_DIR}")


if __name__ == "__main__":
    main()