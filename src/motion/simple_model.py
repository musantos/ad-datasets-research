import torch
import torch.nn as nn

class SimpleTrajectoryPredictor(nn.Module):
    def __init__(self, input_steps=11, output_steps=80):
        super(SimpleTrajectoryPredictor, self).__init__()

        # Input: 11 frames * 2 coordinates (x,y) = 22 numbers
        self.input_dim = input_steps * 2
        # Output: 80 frames * 2 coordinates (x,y) = 160 numbers
        self.output_dim = output_steps * 2

        # A simple network with 2 hidden layers
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, x):
        # x arrives as [batch, 11, 2]
        batch_size = x.shape[0]

        # Flatten to [batch, 22]
        x = x.view(batch_size, -1)

        # Pass through the network
        prediction = self.network(x)

        # Reshape back to [batch, 80, 2]
        return prediction.view(batch_size, 80, 2)

if __name__ == "__main__":
    model = SimpleTrajectoryPredictor()
    print("OK: Prediction model loaded.")
    # Shape test
    test_input = torch.randn(2, 11, 2)
    output = model(test_input)
    print(f"Prediction shape: {output.shape}")  # Expected: [2, 80, 2]
