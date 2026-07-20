"""Shared training, evaluation, plotting, and checkpoint helpers."""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from src.config import CHECKPOINT_DIR, IMAGE_DIR, SEED


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_project_dirs() -> None:
    """Create output directories used by the project."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(model, dataloader, device) -> float:
    """Return accuracy percentage for a model and dataloader."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def collect_predictions(model, dataloader, device):
    """Collect true and predicted labels for evaluation reports."""
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            true_labels.extend(labels.numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    return true_labels, predicted_labels


def save_checkpoint(path, model, optimizer, epoch: int, validation_accuracy: float) -> None:
    """Save model and optimizer state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_accuracy": validation_accuracy,
        },
        path,
    )


def load_checkpoint(path, model, device):
    """Load model weights from a checkpoint file."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def save_training_plots(history, image_dir=IMAGE_DIR) -> None:
    """Save training loss and accuracy plots."""
    image_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(image_dir / "training_loss.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], marker="o", label="Train Accuracy")
    plt.plot(
        epochs,
        history["validation_accuracy"],
        marker="s",
        label="Validation Accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(image_dir / "accuracy.png")
    plt.close()


def save_confusion_matrix(true_labels, predicted_labels, image_dir=IMAGE_DIR) -> None:
    """Save a confusion matrix plot."""
    image_dir.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay.from_predictions(
        true_labels,
        predicted_labels,
        cmap="Blues",
    )
    display.figure_.tight_layout()
    display.figure_.savefig(image_dir / "confusion_matrix.png")
    plt.close(display.figure_)


def print_classification_report(true_labels, predicted_labels) -> str:
    """Print and return the classification report."""
    report = classification_report(true_labels, predicted_labels)
    print("\nClassification Report")
    print("---------------------")
    print(report)
    return report
