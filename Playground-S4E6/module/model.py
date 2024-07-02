from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(36, 256),  # Input layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),  # Hidden 1
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),  # Hidden 2
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),  # Hidden 3
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),  # Output layer
        )
        self.double()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
