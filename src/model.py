"""Neural network model definition."""

import torch.nn as nn

from src.config import HIDDEN_SIZE, INPUT_SIZE, NUM_CLASSES


class MLP(nn.Module):
    """A simple multilayer perceptron for MNIST digit classification."""

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, images):
        """Run a forward pass for a batch of images."""
        return self.network(images)
