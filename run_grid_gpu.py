#!/usr/bin/env python3
"""
GENERIC grid, GPU phase (train -> infer). Run IN THE GPU CONTAINER. The metrics
phase (validate_motion_official) runs separately in the CPU/TF container -- see
run_grid_validate.py.

Este é o UNIFICADOR dos antigos run_grid_gpu.py (item4) e run_grid_gpu_vectorized.py
(V0). O método vira ARGUMENTO (--model), não um arquivo novo por degrau. Toda a
diferença entre métodos vive no REGISTRO MODELS_CFG abaixo; a máquina (sentinelas,
gpu-loggers, phase-index) é idêntica à das versões anteriores.

  python3 run_grid_gpu.py --model vectorized      # V0: {agent} x {raw,std} x 8  = 16
  python3 run_grid_gpu.py --model social          # V1: {agent} x {raw,std} x 8  = 16
  python3 run_grid_gpu.py --model multimodal --model2 sequential   # item4 (ver nota)
  python3 run_grid_gpu.py --model social --dry-run  # imprime os comandos, não roda

Design (inalterado vs as versões por-método):
  * RESUMABLE / IDEMPOTENT via sentinelas .train_done / .infer_done.
  * GPU LOGGERS anexados (state + procs) e mortos na saída.
  * PHASE INDEX por config (join por tempo com os gpu_logs).

Este script NÃO toca a ciência: só chama os módulos train_*/run_inference_*
com os mesmos args que você digitaria à mão.

--- A PEGADINHA que o registro resolve -----------------------------------------
O 'social' é o único caso onde o NOME DO MÓDULO difere do PREFIXO DE PASTA:
  módulo  = src.motion.train_social / run_inference_social
  pasta   = vectorized_social_<run_tag>   (checkpoints, predictions, logs, metrics)
Além disso o social NÃO aceita --agent-centric (braço agente é implícito) e EXIGE
--n-neighbors. Por isso 'module', 'folder', 'agent_centric' e 'extra_*' são campos
separados no registro.
"""
import os
import sys
import time
import csv
import signal
import argparse
import subprocess
from datetime import datetime

# --- REGISTRO DE MODELOS -----------------------------------------------------
# module        : sufixo do módulo -> src.motion.train_<module> / run_inference_<module>
# folder        : prefixo de pasta/CSV -> <folder>_<run_tag> (checkpoints/pred/logs/metrics)
# arms          : braços a varrer (sdc|agent)
# variants      : raw e/ou std (std anexa --standardize e sufixa _std)
# agent_centric : se True, anexa --agent-centric quando arm=='agent'
#                 (False no social: braço agente é IMPLÍCITO, o script não aceita a flag)
# extra_train   : flags fixas extras no treino  (ex.: --n-neighbors do social)
# extra_infer   : flags fixas extras na inferência (idem; MUST match o treino)
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
}

CLS_WEIGHT_DEFAULT = 20          # usado se --cls-weights não for passado

# --- paths (match train_*/run_inference_*) -----------------------------------
CHECKPOINT_ROOT = "/workspace/experiments/checkpoints"
PRED_ROOT = "/workspace/datasets/waymo/predictions"
GPULOG_DIR = "/workspace/experiments/gpu_logs"
NVIDIA_SMI_INTERVAL = 5          # seconds, matches your manual -l 5

PYEXE = sys.executable or "python3"

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


def run_tag(cls, arm, seed, variant):
    # O PESO entra no run_tag (cls<w>) -> pesos diferentes NUNCA colidem na mesma
    # pasta. É o que torna a varredura de peso segura (sem sobrescrita silenciosa).
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
    print(f"\n[DRY-RUN] {n} combos ({n} train + {n} infer). Nada foi executado.")


def main():
    ap = argparse.ArgumentParser(description="Generic GPU grid (train->infer).")
    ap.add_argument("--model", required=True, choices=sorted(MODELS_CFG),
                    help="método a rodar (entrada do registro MODELS_CFG).")
    ap.add_argument("--cls-weights", default=str(CLS_WEIGHT_DEFAULT),
                    help="peso(s) do termo de classificação. Um valor "
                         f"(default {CLS_WEIGHT_DEFAULT}) ou lista '1,20,50' para varrer. "
                         "O peso entra no run_tag (cls<w>), então pesos não colidem.")
    ap.add_argument("--seeds", default="0-7",
                    help="seeds: '0-7' (faixa) ou '0,1,2' (lista). Default 0-7.")
    ap.add_argument("--dry-run", action="store_true",
                    help="imprime os comandos e paths sem executar (para conferência).")
    args = ap.parse_args()

    cfg = MODELS_CFG[args.model]

    # peso: aceita '20' (um valor) ou '1,20,50' (lista). g-format tira zeros à toa.
    cls_weights = [float(w) for w in args.cls_weights.split(",") if w.strip() != ""]

    if "-" in args.seeds:
        a, b = args.seeds.split("-")
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    if args.dry_run:
        dry_run(cfg, cls_weights, seeds)
        return

    os.makedirs(GPULOG_DIR, exist_ok=True)
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
                continue

            run_phase(index_writer, index_fh, cfg["folder"], cls, arm, variant, seed,
                      "infer", build_infer_cmd(cfg, cls, arm, seed, variant),
                      infer_sentinel(cfg, cls, arm, seed, variant))
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
    print(f"       python3 run_grid_validate.py --model {args.model}")


if __name__ == "__main__":
    main()