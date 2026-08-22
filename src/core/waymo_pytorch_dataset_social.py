import os
import numpy as np
import torch
from torch.utils.data import Dataset

# SIBLING de waymo_pytorch_dataset_agentcentric (V1 = contexto social). O
# arquivo agent-centric NÃO é tocado; reusamos suas constantes e a rotação como
# ÚNICA fonte de verdade do layout/convenção -> impossível o frame dos vizinhos
# divergir do frame do alvo (era a falha silenciosa mais provável do V1).
from src.core.waymo_pytorch_dataset_agentcentric import (
    WaymoMotionDatasetAgentCentric,
    FS_X, FS_Y, FS_LENGTH, FS_WIDTH, FS_HEADING, FS_VX, FS_VY,
    ANCHOR_FRAME, N_PAST,
    _SCALAR_FEATURES, _ANGLE_FEATURES,
)


def _agent_past_features(full_state, mask, p0, theta0, features, heading_as_sincos):
    """Features ricas [11, n_features] de UM agente qualquer, projetadas no frame
    agente-cêntrico ANCORADO EM (p0, theta0).

    Para o ALVO, (p0,theta0) é a própria pose@10 -> reproduz o agent-centric
    byte-a-byte (o alvo cai em (0,0) e heading 0). Para um VIZINHO, usa-se o
    MESMO (p0,theta0) do alvo -> o vizinho fica posicionado/rotacionado RELATIVO
    ao alvo, que é o sinal social. A transformação é literalmente a do dataset
    agent-centric; só o anchor vem de fora.

    Invalidez: transforma primeiro, zera depois (igual ao agent-centric), pra um
    frame inválido re-centrado não vazar posição/velocidade espúria.
    """
    fs = np.asarray(full_state, dtype=np.float64)
    m = np.asarray(mask).astype(bool)

    xy = fs[:, [FS_X, FS_Y]].copy()                 # [91,2] frame SDC
    heading = fs[:, FS_HEADING].copy()              # [91]
    vel = fs[:, [FS_VX, FS_VY]].copy()              # [91,2]

    R = WaymoMotionDatasetAgentCentric._rotation_neg(theta0)
    xy = (xy - p0) @ R.T                            # translada + rotaciona
    vel = vel @ R.T                                 # só rotaciona (vetor livre)
    heading = heading - theta0                      # heading relativo

    cols = []
    for f in features:
        if f == "x":
            cols.append(xy[:, 0:1])
        elif f == "y":
            cols.append(xy[:, 1:2])
        elif f == "vx":
            cols.append(vel[:, 0:1])
        elif f == "vy":
            cols.append(vel[:, 1:2])
        elif f == "length":
            cols.append(fs[:, FS_LENGTH:FS_LENGTH + 1])
        elif f == "width":
            cols.append(fs[:, FS_WIDTH:FS_WIDTH + 1])
        elif f == "heading":
            if heading_as_sincos:
                cols.append(np.sin(heading)[:, None])
                cols.append(np.cos(heading)[:, None])
            else:
                cols.append(heading[:, None])
    feats = np.concatenate(cols, axis=1)            # [91, n_features]

    feats[~m] = 0.0
    return feats[:N_PAST, :]                         # [11, n_features]


class WaymoMotionDatasetSocial(Dataset):
    """
    Dataset do V1 (social). Estende o agent-centric expondo, além do alvo, os K
    vizinhos mais próximos da cena projetados NO FRAME DO ALVO.

    agent-centric é IMPLÍCITO e OBRIGATÓRIO: o frame social só faz sentido no
    frame agente-cêntrico do alvo. Não há braço SDC no V1 (invariante da escada).

    __getitem__ retorna uma 6-tupla (a 4-tupla do agent-centric + 2 tensores
    sociais). O alvo (x_past, y_future, future_mask, type) é BYTE-EQUIVALENTE ao
    agent-centric; se n_neighbors=0 ou a cena não tiver vizinhos válidos, o V1
    reduz à entrada do V0 (com o ramo social zerado/mascarado):

        x_past        [11, n_features]   alvo, agente-cêntrico
        neighbors     [K, 11, n_features] vizinhos no MESMO frame do alvo (pad 0)
        neighbor_mask [K]                1.0 onde o vizinho existe (0 = padding)
        y_future      [80, 2]            alvo, mesmo frame do input
        future_mask   [80]               1.0 onde o frame futuro é válido
        agent_type    escalar long

    Vizinho candidato = agente != alvo com mask[10]=True (posição "atual"
    definida, necessária pra distância e pro vetor de estado corrente). Ordena
    por (distância@10, id) e corta em K -> determinístico (seeds a priori).
    """

    def __init__(
        self,
        cache_dir,
        n_neighbors=16,
        features=("x", "y", "heading", "vx", "vy"),
        heading_as_sincos=True,
    ):
        self.cache_dir = cache_dir
        self.n_neighbors = int(n_neighbors)
        self.features = tuple(features)
        self.heading_as_sincos = heading_as_sincos

        for f in self.features:
            if f not in _SCALAR_FEATURES and f not in _ANGLE_FEATURES:
                raise ValueError(
                    f"Unknown feature '{f}'. Valid: "
                    f"{sorted(_SCALAR_FEATURES | _ANGLE_FEATURES)}"
                )

        self.n_features = sum(
            2 if (f in _ANGLE_FEATURES and self.heading_as_sincos) else 1
            for f in self.features
        )

        file_list = [f for f in os.listdir(cache_dir) if f.endswith('.npy')]
        if len(file_list) == 0:
            print(f"WARNING: No file found in {cache_dir}")

        # Index idêntico ao agent-centric: uma amostra por (cena, alvo).
        self.samples = []
        for fname in file_list:
            path = os.path.join(cache_dir, fname)
            data = np.load(path, allow_pickle=True).item()
            for agent in data['agents']:
                if agent.get('is_target', False):
                    self.samples.append((path, agent['id']))

        if len(self.samples) == 0 and len(file_list) > 0:
            print(f"WARNING: no agent with is_target=True found in {cache_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, target_id = self.samples[idx]
        data = np.load(file_path, allow_pickle=True).item()
        agents = data['agents']

        target = next((a for a in agents if a['id'] == target_id), None)
        if target is None:
            raise RuntimeError(f"Agent id={target_id} not found in {file_path}.")

        t_fs = np.asarray(target['full_state'], dtype=np.float64)
        t_mask = np.asarray(target['mask'])
        assert t_fs.shape == (91, 7), (
            f"full_state {t_fs.shape} != (91,7) em {file_path}; layout mudou."
        )
        assert t_mask.shape[0] == 91, f"mask len {t_mask.shape[0]} != 91."
        assert bool(t_mask[ANCHOR_FRAME]), (
            f"anchor frame {ANCHOR_FRAME} inválido para alvo {target_id} em "
            f"{file_path}; frame agente-cêntrico indefinido."
        )

        # Anchor do alvo (define o frame agente-cêntrico de TODOS os polylines).
        p0 = t_fs[ANCHOR_FRAME, [FS_X, FS_Y]].copy()
        theta0 = float(t_fs[ANCHOR_FRAME, FS_HEADING])

        # --- ALVO (equivalente ao agent-centric) ---
        x_past = _agent_past_features(
            t_fs, t_mask, p0, theta0, self.features, self.heading_as_sincos)

        R = WaymoMotionDatasetAgentCentric._rotation_neg(theta0)
        xy_all = (t_fs[:, [FS_X, FS_Y]].copy() - p0) @ R.T   # [91,2] frame agente
        xy_all[~t_mask.astype(bool)] = 0.0
        y_future = xy_all[N_PAST:, :]                         # [80,2]
        future_mask = t_mask[N_PAST:].astype(np.float32)      # [80]

        # --- VIZINHOS: válidos@10, K mais próximos, MESMO frame do alvo ---
        t_pos10 = t_fs[ANCHOR_FRAME, [FS_X, FS_Y]]            # frame SDC (p/ distância)
        cands = []
        for a in agents:
            if a['id'] == target_id:
                continue
            am = np.asarray(a['mask'])
            if not bool(am[ANCHOR_FRAME]):
                continue
            a_fs = np.asarray(a['full_state'], dtype=np.float64)
            dx, dy = a_fs[ANCHOR_FRAME, [FS_X, FS_Y]] - t_pos10
            dist = float(np.hypot(dx, dy))
            cands.append((dist, int(a['id']), a_fs, am))

        # Determinístico: distância crescente, empate por id.
        cands.sort(key=lambda c: (c[0], c[1]))
        cands = cands[:self.n_neighbors]

        K = self.n_neighbors
        neighbors = np.zeros((K, N_PAST, self.n_features), dtype=np.float64)
        neighbor_mask = np.zeros((K,), dtype=np.float32)
        for i, (_dist, _aid, a_fs, am) in enumerate(cands):
            neighbors[i] = _agent_past_features(
                a_fs, am, p0, theta0, self.features, self.heading_as_sincos)
            neighbor_mask[i] = 1.0

        return (
            torch.from_numpy(x_past).float(),          # [11, F]
            torch.from_numpy(neighbors).float(),        # [K, 11, F]
            torch.from_numpy(neighbor_mask).float(),    # [K]
            torch.from_numpy(y_future).float(),         # [80, 2]
            torch.from_numpy(future_mask).float(),      # [80]
            torch.tensor(int(target['type']), dtype=torch.long),
        )


if __name__ == "__main__":
    print("OK: WaymoMotionDatasetSocial class ready.")
    for feats, sc in [
        (("x", "y", "heading", "vx", "vy"), True),
        (("x", "y", "length", "width", "heading", "vx", "vy"), True),
    ]:
        n = sum(2 if (f in _ANGLE_FEATURES and sc) else 1 for f in feats)
        print(f"  features={feats} sincos={sc} -> n_features={n}")

    CACHE = os.environ.get("WAYMO_CACHE", "")
    if CACHE and os.path.isdir(CACHE):
        K = int(os.environ.get("N_NEIGHBORS", "16"))
        ds = WaymoMotionDatasetSocial(CACHE, n_neighbors=K,
                                      features=("x", "y", "heading", "vx", "vy"))
        xp, nb, nm, yf, fm, at = ds[0]
        print(f"[smoke] n_features={ds.n_features} | x_past={tuple(xp.shape)} "
              f"neighbors={tuple(nb.shape)} nmask={tuple(nm.shape)} "
              f"(vizinhos válidos={int(nm.sum())}/{K}) y_future={tuple(yf.shape)} "
              f"mask={tuple(fm.shape)} type={int(at)}")
        assert xp.shape == (11, ds.n_features)
        assert nb.shape == (K, 11, ds.n_features)
        assert nm.shape == (K,)
        assert yf.shape == (80, 2)

        # EQUIVALÊNCIA DO ALVO: x_past/y_future têm de bater byte-a-byte com o
        # dataset agent-centric (mesma amostra) -> o V1 não alterou o alvo.
        base = WaymoMotionDatasetAgentCentric(
            CACHE, agent_centric=True, features=("x", "y", "heading", "vx", "vy"))
        # alinha o índice: o agent-centric pode ter ordem de samples diferente;
        # casa pelo par (path, id).
        want = ds.samples[0]
        j = base.samples.index(want)
        bxp, byf, bfm, bat = base[j]
        assert torch.allclose(xp, bxp, atol=1e-6), "x_past divergiu do agent-centric!"
        assert torch.allclose(yf, byf, atol=1e-6), "y_future divergiu do agent-centric!"
        assert torch.allclose(fm, bfm, atol=1e-6), "future_mask divergiu!"
        print("[smoke] equivalência ALVO vs agent-centric OK (x_past, y_future, mask).")
