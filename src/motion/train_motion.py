import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time

from src.core.waymo_pytorch_dataset import WaymoMotionDataset
from src.motion.simple_model import SimpleTrajectoryPredictor

# Mapeamento do enum Object.Type do proto do Waymo.
# NOTA: baseado na documentacao publica do WOD -- se algum tipo aparecer
# como "desconhecido" no output, precisamos confirmar contra
# waymo_open_dataset/protos/scenario.proto (enum ObjectType) no seu container.
OBJECT_TYPE_NAMES = {
    0: "UNSET",
    1: "VEICULO",
    2: "PEDESTRE",
    3: "CICLISTA",
    4: "OUTRO",
}


def masked_mse_per_example(outputs, targets, mask):
    """
    Retorna a loss mascarada POR EXEMPLO do batch (nao a media do batch
    inteiro), para permitir agrupar depois por tipo de agente.

    outputs, targets: [batch, 80, 2]
    mask: [batch, 80]
    Retorna: tensor [batch] com a MSE mascarada de cada exemplo.
    """
    diff2 = (outputs - targets) ** 2          # [batch, 80, 2]
    mask_exp = mask.unsqueeze(-1)              # [batch, 80, 1]

    masked_diff = diff2 * mask_exp
    per_example_sum = masked_diff.sum(dim=(1, 2))                       # [batch]
    per_example_count = (mask_exp.sum(dim=(1, 2)) * 2).clamp(min=1.0)   # [batch]

    return per_example_sum / per_example_count


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir = "/workspace/datasets/waymo/cache"
    checkpoint_dir = "/workspace/experiments/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 50)
    print(f"[*] INICIANDO TREINO NA GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (sem GPU detectada)'}")
    print("=" * 50)

    try:
        full_dataset = WaymoMotionDataset(cache_dir)
        if len(full_dataset) == 0:
            print(f"[ERRO] Nenhum arquivo .npy encontrado em {cache_dir}.")
            return
    except Exception as e:
        print(f"[ERRO] Falha ao carregar dataset: {e}")
        return

    val_fraction = 0.2
    val_size = max(1, int(len(full_dataset) * val_fraction))
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    print(f"[OK] Dataset total: {len(full_dataset)} exemplos "
          f"-> treino: {train_size} | validacao: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    model = SimpleTrajectoryPredictor(input_steps=11, output_steps=80).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")

    epochs = 25
    for epoch in range(epochs):
        # --- Treino ---
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        start_time = time.time()

        for history, future_gt, future_mask, agent_type in train_loader:
            history = history.to(device)
            future_gt = future_gt.to(device)
            future_mask = future_mask.to(device)

            optimizer.zero_grad()
            outputs = model(history)

            per_ex_loss = masked_mse_per_example(outputs, future_gt, future_mask)
            loss = per_ex_loss.mean()
            loss.backward()
            optimizer.step()

            train_loss_sum += per_ex_loss.sum().item()
            train_count += per_ex_loss.numel()

        avg_train_loss = train_loss_sum / train_count

        # --- Validacao (com quebra por tipo de agente) ---
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        per_type_stats = {}  # {tipo: [soma_loss, contagem]}

        with torch.no_grad():
            for history, future_gt, future_mask, agent_type in val_loader:
                history = history.to(device)
                future_gt = future_gt.to(device)
                future_mask = future_mask.to(device)

                outputs = model(history)
                per_ex_loss = masked_mse_per_example(outputs, future_gt, future_mask)

                val_loss_sum += per_ex_loss.sum().item()
                val_count += per_ex_loss.numel()

                for t, l in zip(agent_type.tolist(), per_ex_loss.cpu().tolist()):
                    if t not in per_type_stats:
                        per_type_stats[t] = [0.0, 0]
                    per_type_stats[t][0] += l
                    per_type_stats[t][1] += 1

        avg_val_loss = val_loss_sum / val_count

        checkpoint_path = os.path.join(checkpoint_dir, f"motion_model_e{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(checkpoint_dir, "motion_model_best.pth")
            torch.save(model.state_dict(), best_path)
            marker = "  <- melhor ate agora"
        else:
            marker = ""

        duration = time.time() - start_time
        print(f"[OK] Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}{marker} | "
              f"Tempo: {duration:.2f}s")

        breakdown = []
        for t, (loss_sum, n) in sorted(per_type_stats.items()):
            name = OBJECT_TYPE_NAMES.get(t, f"TIPO_{t}")
            breakdown.append(f"{name}: {loss_sum / n:.2f} (n={n})")
        print(f"         Val por tipo -> {' | '.join(breakdown)}")

    print("\n[SUCESSO] Treinamento finalizado.")
    print(f"Melhor checkpoint: motion_model_best.pth (Val Loss: {best_val_loss:.4f})")


if __name__ == "__main__":
    train()
