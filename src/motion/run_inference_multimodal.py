import os
import numpy as np
import torch

from src.motion.multimodal_model import MultimodalTrajectoryPredictor

# RODAR ESTE SCRIPT NO CONTAINER DE TREINO (GPU) -- e onde o PyTorch e o
# checkpoint treinado existem.

NUM_MODES = 6

# Inferencia SO sobre o cache de validacao (split oficial). O run_inference
# antigo varria o cache inteiro, que continha os dados de treino -- as
# metricas resultantes mediam memorizacao junto com generalizacao.
CACHE_DIR = "/workspace/datasets/waymo/cache_val"
PRED_DIR = "/workspace/datasets/waymo/predictions/multimodal"
CHECKPOINT_PATH = "/workspace/experiments/checkpoints/multimodal/multimodal_best.pth"


def run_inference():
    os.makedirs(PRED_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERRO] Checkpoint nao encontrado: {CHECKPOINT_PATH}")
        return

    model = MultimodalTrajectoryPredictor(
        input_steps=11, output_steps=80, num_modes=NUM_MODES
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.npy')]
    if not files:
        print(f"[ERRO] Cache de validacao vazio: {CACHE_DIR}")
        return

    print(f"INFO: gerando predicoes multimodais para ate {len(files)} cenarios...")

    n_done = 0
    n_agents = 0
    with torch.no_grad():
        for i, fname in enumerate(files):
            path = os.path.join(CACHE_DIR, fname)
            data = np.load(path, allow_pickle=True).item()

            preds = {}
            for agent in data['agents']:
                if not agent.get('is_target', False):
                    continue

                # Mesma logica de zerar frames invalidos do passado usada no
                # treino (waymo_pytorch_dataset.py).
                traj = agent['trajectory'].copy()
                mask = agent['mask']
                traj[~mask] = 0.0

                x_past = torch.tensor(traj[:11, :], dtype=torch.float32)
                x_past = x_past.unsqueeze(0).to(device)  # [1, 11, 2]

                traj_out, scores = model(x_past)
                # traj_out: [1, K, 80, 2] a 10Hz (mesma taxa do treino)
                # scores:   [1, K] em LOGITS

                # Softmax aqui, e nao no modelo: a metrica oficial espera
                # scores comparaveis entre si (usados no calculo do mAP).
                probs = torch.softmax(scores, dim=1)

                # Ordenar os modos por probabilidade decrescente. A metrica
                # oficial nao exige ordem, mas ter o modo mais provavel no
                # indice 0 facilita inspecao e eventual corte de top-k.
                order = torch.argsort(probs[0], descending=True)

                preds[agent['id']] = {
                    'trajectories': traj_out[0][order].cpu().numpy(),  # [K, 80, 2]
                    'scores': probs[0][order].cpu().numpy(),           # [K]
                }
                n_agents += 1

            if preds:
                out_path = os.path.join(PRED_DIR, fname)
                np.save(out_path, preds)
                n_done += 1

            if (i + 1) % 100 == 0:
                print(f"  ... {i+1}/{len(files)} cenarios processados")

    print(f"[SUCESSO] Predicoes salvas em {PRED_DIR}")
    print(f"          {n_done} cenarios com ao menos 1 agente-alvo | "
          f"{n_agents} trajetorias-alvo (x{NUM_MODES} modos cada).")


if __name__ == "__main__":
    run_inference()
