import os
import time
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.core.waymo_pytorch_dataset import WaymoMotionDataset
from src.motion.multimodal_model import MultimodalTrajectoryPredictor

# Mapeamento do enum Object.Type do proto do Waymo (mesmo do baseline).
OBJECT_TYPE_NAMES = {
    0: "UNSET",
    1: "VEICULO",
    2: "PEDESTRE",
    3: "CICLISTA",
    4: "OUTRO",
}

NUM_MODES = 6          # casa com max_predictions=6 da config oficial
EPOCHS = 25
BATCH_SIZE = 64
LR = 1e-3

# SPLIT OFICIAL DO WAYMO -- nao ha random_split.
# Treino e validacao vem de shards DIFERENTES de splits DIFERENTES, entao
# nao existe possibilidade de vazamento: sao cenarios distintos. Isso
# tambem torna os numeros comparaveis com a literatura, que reporta sobre
# o split 'validation'.
TRAIN_CACHE = "/workspace/datasets/waymo/cache_train"
VAL_CACHE = "/workspace/datasets/waymo/cache_val"

# Raiz dos checkpoints. A pasta final inclui o cls_weight usado, para que
# rodadas da varredura nao se sobrescrevam.
CHECKPOINT_ROOT = "/workspace/experiments/checkpoints"


def masked_mse_per_mode(outputs, targets, mask):
    """
    Erro quadratico medio mascarado de CADA modo, por exemplo do batch.

    outputs: [batch, K, 80, 2]  -- as K hipoteses
    targets: [batch, 80, 2]     -- o ground truth (unico)
    mask:    [batch, 80]        -- 1 onde o frame futuro e valido

    Retorna: [batch, K]

    A mascara e essencial: agentes que saem de visibilidade tem frames
    invalidos zerados no dataset, e sem mascarar o modelo seria punido
    por nao prever zeros.
    """
    targets = targets.unsqueeze(1)                 # [B, 1, 80, 2]
    mask_exp = mask.unsqueeze(1).unsqueeze(-1)     # [B, 1, 80, 1]

    diff2 = (outputs - targets) ** 2               # [B, K, 80, 2]
    masked = diff2 * mask_exp

    per_mode_sum = masked.sum(dim=(2, 3))                        # [B, K]
    denom = (mask.sum(dim=1) * 2).clamp(min=1.0).unsqueeze(1)    # [B, 1]

    return per_mode_sum / denom


def wta_loss(outputs, scores, targets, mask, cls_weight):
    """
    Loss Winner-Takes-All (WTA).

    Ideia: dos K modos, so o mais proximo do ground truth recebe gradiente
    de regressao. Os outros ficam livres para cobrir hipoteses alternativas
    em vez de serem puxados para a media -- que e exatamente o que se quer
    num problema multimodal (num cruzamento, "virar" e "seguir reto" sao
    ambas corretas, e a media das duas nao e uma trajetoria plausivel).

    Em paralelo, a cabeca de score aprende via cross-entropy qual modo foi
    o vencedor. Sem esse termo o modelo teria K trajetorias mas nenhuma
    nocao de qual e a mais provavel -- e a metrica oficial (mAP) usa o score.

    NOTA SOBRE cls_weight: os dois termos vivem em escalas MUITO diferentes.
    A regressao e MSE em m^2 (dezenas a centenas); a cross-entropy com 6
    classes parte de ln(6) ~= 1.79. Com cls_weight=1.0 a classificacao vale
    ~1% da loss total e o gradiente da score_head fica afogado -- foi
    exatamente o que aconteceu na primeira rodada.

    Retorna um dicionario para nao multiplicar o numero de retornos a cada
    diagnostico novo.
    """
    per_mode = masked_mse_per_mode(outputs, targets, mask)   # [B, K]

    # argmin nao propaga gradiente (e um indice) -- o gradiente flui apenas
    # pelo valor selecionado via gather.
    best_idx = per_mode.argmin(dim=1)                        # [B]
    reg = per_mode.gather(1, best_idx.unsqueeze(1)).squeeze(1)

    cls = F.cross_entropy(scores, best_idx, reduction="none")  # [B]

    total = reg + cls_weight * cls

    # --- Diagnosticos (nao entram na loss) ---
    with torch.no_grad():
        top1_idx = scores.argmax(dim=1)
        top1 = per_mode.gather(1, top1_idx.unsqueeze(1)).squeeze(1)

        # Erro medio dos K modos = o que se obteria escolhendo ao acaso.
        # Se top1 ~= mean_all, a cabeca de score nao aprendeu nada.
        # Se top1 > mean_all, ela aprendeu algo INVERTIDO (indicio de bug).
        mean_all = per_mode.mean(dim=1)

        # Posicao do modo escolhido no ranking real de qualidade.
        # 0 = escolheu o melhor dos 6; 5 = escolheu o pior.
        # Acaso puro daria media 2.5.
        rank = (per_mode < top1.unsqueeze(1)).sum(dim=1).float()

    return {
        "total": total,        # [B] o que se otimiza
        "reg": reg,            # [B] erro do melhor modo (proxy de minADE)
        "cls": cls,            # [B] cross-entropy pura, sem peso
        "top1": top1,          # [B] erro do modo mais bem pontuado
        "mean_all": mean_all,  # [B] erro medio dos modos (baseline do acaso)
        "rank": rank,          # [B] posicao do escolhido no ranking real
        "best_idx": best_idx,  # [B] qual modo venceu (diagnostico de colapso)
    }


def train(cls_weight):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag = f"cls{cls_weight:g}"
    checkpoint_dir = os.path.join(CHECKPOINT_ROOT, f"multimodal_{tag}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 78)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (sem GPU detectada)"
    print(f"[*] TREINO MULTIMODAL (K={NUM_MODES}, cls_weight={cls_weight}) NA GPU: {gpu_name}")
    print("=" * 78)

    try:
        train_dataset = WaymoMotionDataset(TRAIN_CACHE)
        val_dataset = WaymoMotionDataset(VAL_CACHE)
    except Exception as e:
        print(f"[ERRO] Falha ao carregar dataset: {e}")
        return

    if len(train_dataset) == 0:
        print(f"[ERRO] Cache de treino vazio: {TRAIN_CACHE}")
        return
    if len(val_dataset) == 0:
        print(f"[ERRO] Cache de validacao vazio: {VAL_CACHE}")
        print("       Rode antes, no container de METRICAS:")
        print("       python3 -m src.core.waymo_preprocessor --split validation --shards 0,1,2")
        return

    print(f"[OK] Treino (split oficial 'training'):      {len(train_dataset)} exemplos")
    print(f"[OK] Validacao (split oficial 'validation'): {len(val_dataset)} exemplos")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = MultimodalTrajectoryPredictor(
        input_steps=11, output_steps=80, num_modes=NUM_MODES
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Modelo multimodal: {n_params:,} parametros")
    print(f"[OK] Checkpoints em: {checkpoint_dir}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_val_reg = float("inf")

    for epoch in range(EPOCHS):
        # ---------------- Treino ----------------
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        start_time = time.time()

        for history, future_gt, future_mask, agent_type in train_loader:
            history = history.to(device)
            future_gt = future_gt.to(device)
            future_mask = future_mask.to(device)

            optimizer.zero_grad()
            outputs, scores = model(history)

            out = wta_loss(outputs, scores, future_gt, future_mask, cls_weight)
            loss = out["total"].mean()
            loss.backward()
            optimizer.step()

            train_loss_sum += out["total"].sum().item()
            train_count += out["total"].numel()

        avg_train_loss = train_loss_sum / train_count

        # ---------------- Validacao ----------------
        model.eval()
        sums = {"reg": 0.0, "top1": 0.0, "mean_all": 0.0, "rank": 0.0, "cls": 0.0}
        val_count = 0
        per_type_stats = {}                      # {tipo: [soma_top1, contagem]}
        mode_usage = [0] * NUM_MODES             # diagnostico de colapso de modos

        with torch.no_grad():
            for history, future_gt, future_mask, agent_type in val_loader:
                history = history.to(device)
                future_gt = future_gt.to(device)
                future_mask = future_mask.to(device)

                outputs, scores = model(history)
                out = wta_loss(outputs, scores, future_gt, future_mask, cls_weight)

                for k in sums:
                    sums[k] += out[k].sum().item()
                val_count += out["reg"].numel()

                for m in out["best_idx"].cpu().tolist():
                    mode_usage[m] += 1

                for t, l in zip(agent_type.tolist(), out["top1"].cpu().tolist()):
                    if t not in per_type_stats:
                        per_type_stats[t] = [0.0, 0]
                    per_type_stats[t][0] += l
                    per_type_stats[t][1] += 1

        avg = {k: v / val_count for k, v in sums.items()}

        # Checkpoint por epoca + melhor. O criterio de "melhor" e o erro do
        # MELHOR modo, porque e o proxy direto do que a metrica oficial
        # premia -- minADE/minFDE sao best-of-6.
        torch.save(model.state_dict(),
                   os.path.join(checkpoint_dir, f"multimodal_e{epoch+1}.pth"))

        if avg["reg"] < best_val_reg:
            best_val_reg = avg["reg"]
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, "multimodal_best.pth"))
            marker = "  <- melhor ate agora"
        else:
            marker = ""

        duration = time.time() - start_time
        print(f"[OK] Epoch {epoch+1}/{EPOCHS} | "
              f"Train: {avg_train_loss:.4f} | "
              f"Val(melhor modo): {avg['reg']:.4f}{marker} | "
              f"Val(top-1): {avg['top1']:.4f} | "
              f"Tempo: {duration:.2f}s")

        breakdown = []
        for t, (loss_sum, n) in sorted(per_type_stats.items()):
            name = OBJECT_TYPE_NAMES.get(t, f"TIPO_{t}")
            breakdown.append(f"{name}: {loss_sum / n:.2f} (n={n})")
        print(f"         Val top-1 por tipo -> {' | '.join(breakdown)}")

        # --- Qualidade da cabeca de score ---
        # Se top-1 ~= acaso e rank ~= 2.5, a score_head nao aprendeu nada.
        # Se top-1 < acaso e rank < 2.5, ela esta ranqueando de verdade.
        # Se top-1 > acaso e rank > 2.5, aprendeu INVERTIDO -- suspeitar de bug.
        if avg["rank"] < 2.3:
            verdict = "ranqueando"
        elif avg["rank"] > 2.7:
            verdict = "INVERTIDO (?)"
        else:
            verdict = "~acaso"
        print(f"         Score head -> top-1: {avg['top1']:.1f} | "
              f"acaso: {avg['mean_all']:.1f} | "
              f"rank medio: {avg['rank']:.2f}/5 ({verdict}) | "
              f"CE: {avg['cls']:.3f}")

        # Se um unico modo vencer quase sempre, o modelo colapsou para um
        # unimodal disfarcado -- e o experimento perde o sentido.
        usage_pct = [100.0 * c / val_count for c in mode_usage]
        usage_str = " | ".join(f"m{i}: {p:.0f}%" for i, p in enumerate(usage_pct))
        print(f"         Uso dos modos -> {usage_str}")
        if max(usage_pct) > 90.0:
            print("         [AVISO] colapso de modos: um modo venceu >90% das vezes.")

    print(f"\n[SUCESSO] Treinamento multimodal finalizado (cls_weight={cls_weight}).")
    print(f"Melhor checkpoint: multimodal_best.pth (Val melhor modo: {best_val_reg:.4f})")
    print(f"Salvo em: {checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Treina o modelo multimodal K=6 com loss Winner-Takes-All"
    )
    parser.add_argument("--cls-weight", type=float, default=50.0,
                        help="peso do termo de classificacao na loss total")
    args = parser.parse_args()

    train(cls_weight=args.cls_weight)