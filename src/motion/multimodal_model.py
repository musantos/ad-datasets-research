import torch
import torch.nn as nn


class MultimodalTrajectoryPredictor(nn.Module):
    """
    Multimodal version of SimpleTrajectoryPredictor.

    Conceptual difference from the baseline: instead of predicting ONE
    future trajectory, it predicts K trajectories ("modes") + a score for
    each one. This exists because the official Waymo Motion metric evaluates
    with max_predictions=6 -- i.e., the unimodal baseline was being
    evaluated in a multimodal regime with a single hypothesis.

    Practical motivation: motion prediction is ambiguous by nature. When
    approaching an intersection, "turn left" and "go straight" are both
    correct. A unimodal model trained with MSE learns the AVERAGE of the
    two -- a trajectory no real agent would ever follow.

    The backbone (2 hidden layers of 256) is IDENTICAL to the baseline's,
    on purpose: the only variable that changes between the two experiments
    is the multimodality, so that any difference in the metrics is
    attributable to it.

    ITEM 4 -- rich input: the per-frame feature count is now a constructor
    argument (n_features), NOT hard-coded to 2. Read it off the dataset
    (`dataset.n_features`) and pass it in; the flatten input_dim becomes
    input_steps * n_features. n_features=2 reproduces the x,y-only baseline.

    Shapes:
        input:   [batch, 11, n_features]
        outputs: trajectories [batch, K, 80, 2]
                 scores       [batch, K]   (LOGITS, not probabilities)
    """

    def __init__(self, input_steps=11, output_steps=80, num_modes=6,
                 hidden_dim=256, n_features=2):
        super(MultimodalTrajectoryPredictor, self).__init__()

        self.input_steps = input_steps
        self.output_steps = output_steps
        self.num_modes = num_modes
        self.n_features = n_features

        # Per-frame feature dimension is a knob now: 2 for x,y (baseline),
        # 6 for (x, y, heading->sin/cos, vx, vy) in item 4, etc.
        self.input_dim = input_steps * n_features  # 11 * n_features

        # Backbone shared between the two heads.
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Head 1: the K trajectories. K * 80 * 2 = 960 values for K=6.
        self.traj_head = nn.Linear(hidden_dim, num_modes * output_steps * 2)

        # Head 2: the score of each mode. Output is LOGITS -- the softmax
        # is left to the consumer (cross_entropy in training already applies
        # it internally; run_inference applies it explicitly before saving,
        # because the official metric expects comparable scores).
        self.score_head = nn.Linear(hidden_dim, num_modes)

    def forward(self, x):
        # x arrives as [batch, 11, n_features]
        batch_size = x.shape[0]

        # Flatten to [batch, 11 * n_features] -- same logic as the baseline,
        # the trailing feature dim just grew.
        x = x.view(batch_size, -1)

        features = self.backbone(x)

        trajectories = self.traj_head(features)
        trajectories = trajectories.view(
            batch_size, self.num_modes, self.output_steps, 2
        )

        scores = self.score_head(features)  # [batch, K]

        return trajectories, scores


if __name__ == "__main__":
    for nf in (2, 6):
        model = MultimodalTrajectoryPredictor(n_features=nf)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"OK: Multimodal model loaded, n_features={nf} "
              f"({n_params:,} parameters).")

        test_input = torch.randn(2, 11, nf)
        traj, scores = model(test_input)
        print(f"   Trajectories shape: {tuple(traj.shape)}")   # [2, 6, 80, 2]
        print(f"   Scores shape:       {tuple(scores.shape)}")  # [2, 6]
