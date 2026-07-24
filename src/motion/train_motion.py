import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
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

    # SPLIT OFICIAL DO WAYMO -- nao ha mais random_split.
    # Treino e validacao vem de splits DIFERENTES do dataset (pastas
    # scenario/training e scenario/validation), entao nao existe
    # possibilidade de vazamento: sao cenarios distintos. Isso tambem
    # torna as metricas comparaveis com a literatura, que reporta sobre
    # o split 'validation' (o 'testing' nao tem futuro anotado).
    #
    # Sao exatamente os mesmos caches usados pelo train_multimodal.py --
    # e o que garante que a comparacao entre os dois modelos isole a
    # multimodalidade como unica variavel.
    train_cache = "/workspace/datasets/waymo/cache_train"
    val_cache = "/workspace/datasets/waymo/cache_val"

    # Diretorio novo: preserva os checkpoints antigos (split caseiro,
    # com contaminacao) em checkpoints/baseline_3shards/ para efeito de
    # historico, sem misturar com os resultados validos.
    checkpoint_dir = "/workspace/experiments/checkpoints/baseline_oficial"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 50)
    print(f"[*] INICIANDO TREINO NA GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (sem GPU detectada)'}")
    print("=" * 50)

    try:
        train_dataset = WaymoMotionDataset(train_cache)
        val_dataset = WaymoMotionDataset(val_cache)
    except Exception as e:
        print(f"[ERRO] Falha ao carregar dataset: {e}")
        return

    if len(train_dataset) == 0:
        print(f"[ERRO] Cache de treino vazio: {train_cache}")
        print("       Se voce ainda nao renomeou o cache antigo:")
        print("       mv datasets/waymo/cache datasets/waymo/cache_train")
        return

    if len(val_dataset) == 0:
        print(f"[ERRO] Cache de validacao vazio: {val_cache}")
        print("       Rode antes, no container de METRICAS:")
        print("       python3 -m src.core.waymo_preprocessor --split validation --shards 0,1,2")
        return

    print(f"[OK] Treino (split oficial 'training'):      {len(train_dataset)} exemplos")
    print(f"[OK] Validacao (split oficial 'validation'): {len(val_dataset)} exemplos")

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
    print(f"Salvo em: {checkpoint_dir}")


if __name__ == "__main__":
    train()
