import torch
import torch.nn as nn


class SequentialTrajectoryPredictor(nn.Module):
    """
    Sequential-encoder version of MultimodalTrajectoryPredictor.

    Conceptual difference from the multimodal MLP baseline: instead of
    flattening the 11 history frames into a single vector, it reads them
    AS A SEQUENCE with a GRU. The only thing that changes between the two
    experiments is how the temporal history is aggregated
    (flatten -> recurrence); everything downstream is identical.

    Kept identical to the MLP on purpose, so any metric difference is
    attributable to the encoder and nothing else:
        - same input:   [batch, 11, n_features]
        - same outputs: trajectories [batch, K, 80, 2]
                        scores       [batch, K]  (LOGITS, not probabilities)
        - same K=6, same output horizon, same two heads.

    ITEM 4 -- rich input: the per-frame feature count is now a constructor
    argument (n_features), NOT hard-coded to 2. Read it off the dataset
    (`dataset.n_features`) and pass it in; it becomes the GRU input_size.
    n_features=2 reproduces the x,y-only baseline.

    Design notes:
        - GRU (not LSTM): 11 steps is a short sequence and the GRU has fewer
          parameters with no practical accuracy cost at this scale. A small
          Transformer encoder is a later option, but needs positional
          encoding and is finicky for sequences this short.
        - We take the LAST hidden state h_n[-1] as the history embedding,
          which is the standard, minimal choice. Attention/pooling over the
          full sequence is a possible refinement later.
        - hidden_dim=128 gives ~180k parameters (vs. the MLP's ~320k). Param
          count is NOT matched on purpose -- the comparison controls the
          input/output/loss and reports parameters in the table. If we want
          to control for capacity too, raising hidden_dim closes the gap;
          expose it as the knob it is.

    Shapes:
        input:   [batch, 11, n_features]
        outputs: trajectories [batch, K, 80, 2]
                 scores       [batch, K]
    """

    def __init__(self, input_steps=11, output_steps=80, num_modes=6,
                 hidden_dim=128, gru_layers=1, n_features=2):
        super(SequentialTrajectoryPredictor, self).__init__()

        self.input_steps = input_steps
        self.output_steps = output_steps
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim

        # Per-frame feature dimension is a knob now: 2 for x,y (baseline),
        # 6 for (x, y, heading->sin/cos, vx, vy) in item 4, etc.
        self.feature_dim = n_features

        # Encoder: reads the 11 frames as a sequence.
        self.encoder = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )

        # Head 1: the K trajectories. K * 80 * 2 = 960 values for K=6.
        self.traj_head = nn.Linear(hidden_dim, num_modes * output_steps * 2)

        # Head 2: the score of each mode. Output is LOGITS -- the softmax is
        # left to the consumer, exactly as in the MLP model (cross_entropy in
        # training applies it internally; run_inference applies it explicitly
        # before saving, because the official metric expects comparable scores).
        self.score_head = nn.Linear(hidden_dim, num_modes)

    def forward(self, x):
        # x arrives as [batch, 11, n_features] -- NO flatten, unlike the MLP.
        batch_size = x.shape[0]

        # GRU returns (output_seq, h_n). h_n is [num_layers, batch, hidden].
        _, h_n = self.encoder(x)

        # Last layer's hidden state as the history embedding: [batch, hidden].
        features = h_n[-1]

        trajectories = self.traj_head(features)
        trajectories = trajectories.view(
            batch_size, self.num_modes, self.output_steps, 2
        )

        scores = self.score_head(features)  # [batch, K]

        return trajectories, scores


if __name__ == "__main__":
    for nf in (2, 6):
        model = SequentialTrajectoryPredictor(n_features=nf)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"OK: Sequential (GRU) model loaded, n_features={nf} "
              f"({n_params:,} parameters).")

        test_input = torch.randn(2, 11, nf)
        traj, scores = model(test_input)
        print(f"   Trajectories shape: {tuple(traj.shape)}")   # [2, 6, 80, 2]
        print(f"   Scores shape:       {tuple(scores.shape)}")  # [2, 6]
