#!/usr/bin/env python3
"""Compare two perception campaigns (FP32 vs AMP) produced by
run_perception_pipeline.sh v5.

Each campaign lives in its own EXP folder (runs/<EXP>/) holding per-seed run
dirs (<model>_n<N>_<tag>_seed*/train_log.jsonl) and sweep ledgers
(sweep_<model>_n<N>_<tag>.csv). For every (arm, step N) present in BOTH folders
this reports:

  * train wall-clock per seed = sum of per-epoch time_s in train_log.jsonl
    (pure training time -- the only thing AMP changes; infer/plot/video tail is
    excluded by design), averaged over seeds;
  * speedup = fp32_train / amp_train;
  * best AP mean +/- sd per arm (from the sweep ledgers) and the AP delta, so
    AMP's speed gain can be read against any accuracy drift (parity check).

It then projects the train-time saving to a target step from the measured
speedup and the linear-in-(limit_train) train-cost model.

Usage:
  python compare_amp.py \
      --fp32-root runs/amp_n6_fp32 --amp-root runs/amp_n6_bf16 \
      --steps "6" --tag s8 \
      --arms pretrained=faster_rcnn_pt from-scratch=faster_rcnn \
      [--project-to 768 --project-from 96]
"""
import argparse
import csv
import glob
import json
import os
import statistics as st


def train_secs_per_seed(root, model, point):
    """Total training seconds for each seed run under `root` (sum of time_s)."""
    out = {}
    for rd in glob.glob(os.path.join(root, f"{model}_{point}_seed*")):
        log = os.path.join(rd, "train_log.jsonl")
        if not os.path.isfile(log):
            continue
        total = 0.0
        with open(log) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "time_s" in rec:
                    total += float(rec["time_s"])
        seed = os.path.basename(rd).rsplit("seed", 1)[-1]
        out[seed] = total
    return out


def best_aps(root, model, point):
    """best_ap per seed from the sweep ledger under `root`."""
    path = os.path.join(root, f"sweep_{model}_{point}.csv")
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            v = (r.get("best_ap") or "").strip()
            if v and v.lower() != "nan":
                out[r["seed"]] = float(v)
    return out


def ms(vals):
    if not vals:
        return (float("nan"), 0.0, 0)
    return (st.mean(vals), st.pstdev(vals) if len(vals) > 1 else 0.0, len(vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp32-root", required=True, help="EXP folder of the FP32 baseline.")
    ap.add_argument("--amp-root", required=True, help="EXP folder of the AMP campaign.")
    ap.add_argument("--steps", default="6", help="Space-separated N values, e.g. '6 24 96'.")
    ap.add_argument("--tag", default="s8", help="Stride tag used in point names (default s8).")
    ap.add_argument("--arms", nargs="+", default=["pretrained=faster_rcnn_pt",
                                                  "from-scratch=faster_rcnn"],
                    help="label=model tokens (the model is the run-dir / ledger prefix).")
    ap.add_argument("--project-to", type=int, default=None,
                    help="Project the train-time saving to this step N (e.g. 768).")
    ap.add_argument("--project-from", type=int, default=None,
                    help="Reference step N to project from (defaults to the largest "
                         "N measured in both campaigns).")
    args = ap.parse_args()

    steps = [int(x) for x in args.steps.split()]
    arms = [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in args.arms]

    print(f"FP32 baseline : {args.fp32_root}")
    print(f"AMP campaign  : {args.amp_root}")
    print(f"{'arm':>12} {'N':>4} | {'fp32 train/seed':>16} {'amp train/seed':>15} "
          f"{'speedup':>8} | {'AP fp32':>14} {'AP amp':>14} {'dAP':>8}")
    print("-" * 104)

    measured = {}  # (label, N) -> speedup
    for N in steps:
        point = f"n{N}_{args.tag}"
        for label, model in arms:
            t_fp = train_secs_per_seed(args.fp32_root, model, point)
            t_amp = train_secs_per_seed(args.amp_root, model, point)
            ap_fp = best_aps(args.fp32_root, model, point)
            ap_amp = best_aps(args.amp_root, model, point)
            if not t_fp or not t_amp:
                print(f"{label:>12} {N:>4} | (missing train_log in one campaign -- skip)")
                continue
            mfp, _, nfp = ms(list(t_fp.values()))
            mamp, _, namp = ms(list(t_amp.values()))
            speedup = mfp / mamp if mamp else float("nan")
            measured[(label, N)] = speedup
            apf_m, apf_s, _ = ms(list(ap_fp.values()))
            apa_m, apa_s, _ = ms(list(ap_amp.values()))
            dap = apa_m - apf_m
            print(f"{label:>12} {N:>4} | {mfp:>13.1f} s ({nfp}) {mamp:>11.1f} s ({namp}) "
                  f"{speedup:>7.2f}x | {apf_m:>7.4f}±{apf_s:5.4f} {apa_m:>7.4f}±{apa_s:5.4f} "
                  f"{dap:>+8.4f}")

    # --- projection -----------------------------------------------------------
    if args.project_to is not None and measured:
        src_N = args.project_from or max(N for (_, N) in measured)
        print()
        print(f"Projection to n{args.project_to} (train scales ~linearly with "
              f"limit_train; speedup is compute-bound ~constant):")
        for label, _ in arms:
            sp = measured.get((label, src_N))
            if sp is None:
                continue
            # per-seed fp32 time at src_N, scaled linearly to target N
            point = f"n{src_N}_{args.tag}"
            t_fp = train_secs_per_seed(args.fp32_root,
                                       dict(arms)[label], point)
            if not t_fp:
                continue
            base = st.mean(list(t_fp.values()))
            scale = args.project_to / src_N
            fp_proj = base * scale
            amp_proj = fp_proj / sp
            print(f"  {label:>12}: fp32 ~{fp_proj/3600:6.1f} h/seed -> "
                  f"amp ~{amp_proj/3600:6.1f} h/seed  (saves ~{(fp_proj-amp_proj)/3600:5.1f} h/seed "
                  f"at {sp:.2f}x, from the n{src_N} rate)")
        print("  NOTE: linear scaling + constant speedup is a first-order estimate; "
              "real curves may saturate.")


if __name__ == "__main__":
    main()
