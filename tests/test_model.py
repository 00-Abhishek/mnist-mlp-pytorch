"""Tests for the MLP model."""

import torch

from src.model import MLP


def test_mlp_forward_output_shape():
    """MLP should produce one logit vector per input image."""
    model = MLP()
    batch = torch.randn(4, 1, 28, 28)

    output = model(batch)

    assert output.shape == (4, 10)
