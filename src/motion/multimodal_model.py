import torch
import torch.nn as nn


class MultimodalTrajectoryPredictor(nn.Module):
    """
    Versao multimodal do SimpleTrajectoryPredictor.

    Diferenca conceitual em relacao ao baseline: em vez de prever UMA
    trajetoria futura, preve K trajetorias ("modos") + um score para cada
    uma. Isso existe porque a metrica oficial do Waymo Motion avalia com
    max_predictions=6 -- ou seja, o baseline unimodal estava sendo
    avaliado num regime multimodal com uma unica hipotese.

    Motivacao pratica: motion prediction e ambiguo por natureza. Na
    aproximacao de um cruzamento, "virar a esquerda" e "seguir reto" sao
    ambas corretas. Um modelo unimodal treinado com MSE aprende a MEDIA
    das duas -- uma trajetoria que nenhum agente real faria.

    O backbone (2 camadas escondidas de 256) e IDENTICO ao do baseline,
    de proposito: a unica variavel que muda entre os dois experimentos e
    a multimodalidade, para que qualquer diferenca nas metricas seja
    atribuivel a ela.

    Shapes:
        entrada:  [batch, 11, 2]
        saidas:   trajetorias [batch, K, 80, 2]
                  scores      [batch, K]   (LOGITS, nao probabilidades)
    """

    def __init__(self, input_steps=11, output_steps=80, num_modes=6, hidden_dim=256):
        super(MultimodalTrajectoryPredictor, self).__init__()

        self.input_steps = input_steps
        self.output_steps = output_steps
        self.num_modes = num_modes

        self.input_dim = input_steps * 2  # 11 frames * (x,y) = 22

        # Backbone compartilhado entre as duas cabecas.
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Cabeca 1: as K trajetorias. K * 80 * 2 = 960 valores para K=6.
        self.traj_head = nn.Linear(hidden_dim, num_modes * output_steps * 2)

        # Cabeca 2: o score de cada modo. Saida sao LOGITS -- o softmax
        # fica a cargo de quem consome (cross_entropy no treino ja aplica
        # internamente; o run_inference aplica explicitamente antes de
        # gravar, porque a metrica oficial espera scores comparaveis).
        self.score_head = nn.Linear(hidden_dim, num_modes)

    def forward(self, x):
        # x chega como [batch, 11, 2]
        batch_size = x.shape[0]

        # Achatar para [batch, 22] -- mesma logica do baseline.
        x = x.view(batch_size, -1)

        features = self.backbone(x)

        trajectories = self.traj_head(features)
        trajectories = trajectories.view(
            batch_size, self.num_modes, self.output_steps, 2
        )

        scores = self.score_head(features)  # [batch, K]

        return trajectories, scores


if __name__ == "__main__":
    model = MultimodalTrajectoryPredictor()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"OK: Modelo Multimodal carregado ({n_params:,} parametros).")

    test_input = torch.randn(2, 11, 2)
    traj, scores = model(test_input)
    print(f"Shape das trajetorias: {traj.shape}")   # esperado: [2, 6, 80, 2]
    print(f"Shape dos scores:      {scores.shape}")  # esperado: [2, 6]
