import torch

from src.config import INPUT_SIZE, NUM_CLASSES
from src.model import MLP


def test_mlp_forward_returns_class_logits():
    model = MLP()
    images = torch.randn(4, 1, 28, 28)

    logits = model(images)

    assert logits.shape == (4, NUM_CLASSES)


def test_mlp_accepts_configurable_layer_sizes():
    model = MLP(input_size=INPUT_SIZE, hidden_size=16, num_classes=3)
    images = torch.randn(2, 1, 28, 28)

    logits = model(images)

    assert logits.shape == (2, 3)
