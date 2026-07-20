"""Inference helpers for trained MNIST MLP checkpoints."""

import torch

from src.config import BEST_MODEL_PATH
from src.model import MLP
from src.utils import get_device, load_checkpoint


def load_model(checkpoint_path=BEST_MODEL_PATH, device=None):
    """Load a trained MLP model from a checkpoint."""
    device = device or get_device()
    model = MLP().to(device)
    load_checkpoint(checkpoint_path, model, device)
    model.eval()
    return model


def predict(model, image, device=None) -> int:
    """Predict the digit class for a single MNIST image tensor."""
    device = device or next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        output = model(image)
        prediction = output.argmax(dim=1)

    return prediction.item()


def predict_from_checkpoint(image, checkpoint_path=BEST_MODEL_PATH) -> int:
    """Load the saved model checkpoint and predict one image."""
    device = get_device()
    model = load_model(checkpoint_path=checkpoint_path, device=device)
    return predict(model, image, device)
