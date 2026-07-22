import os
import numpy as np
import torch

from src.motion.simple_model import SimpleTrajectoryPredictor

# RODAR ESTE SCRIPT NO CONTAINER DE TREINO (GPU) -- e onde o PyTorch
# e o checkpoint treinado existem.

CACHE_DIR = "/workspace/datasets/waymo/cache"
PRED_DIR = "/workspace/datasets/waymo/predictions"
CHECKPOINT_PATH = "/workspace/experiments/checkpoints/motion_model_best.pth"


def run_inference():
    os.makedirs(PRED_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleTrajectoryPredictor(input_steps=11, output_steps=80).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.npy')]
    print(f"INFO: gerando predicoes para ate {len(files)} cenarios...")

    n_done = 0
    with torch.no_grad():
        for i, fname in enumerate(files):
            path = os.path.join(CACHE_DIR, fname)
            data = np.load(path, allow_pickle=True).item()

            preds = {}
            for agent in data['agents']:
                if not agent.get('is_target', False):
                    continue

                # Mesma logica de zerar frames invalidos do passado usada
                # no treino (waymo_pytorch_dataset.py v3/v4).
                traj = agent['trajectory'].copy()
                mask = agent['mask']
                traj[~mask] = 0.0

                x_past = torch.tensor(traj[:11, :], dtype=torch.float32)
                x_past = x_past.unsqueeze(0).to(device)  # [1, 11, 2]

                pred = model(x_past)  # [1, 80, 2], a 10Hz (mesma taxa do treino)
                preds[agent['id']] = pred.squeeze(0).cpu().numpy()

            if preds:
                out_path = os.path.join(PRED_DIR, fname)
                np.save(out_path, preds)
                n_done += 1

            if (i + 1) % 100 == 0:
                print(f"  ... {i+1}/{len(files)} cenarios processados")

    print(f"[SUCESSO] Predicoes salvas em {PRED_DIR} ({n_done} cenarios com ao menos 1 agente-alvo).")


if __name__ == "__main__":
    run_inference()
