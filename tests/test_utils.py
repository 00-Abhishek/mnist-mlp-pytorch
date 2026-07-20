import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import MLP
from src.utils import evaluate, load_checkpoint, save_checkpoint


class AlwaysZeroModel(torch.nn.Module):
    def forward(self, images):
        logits = torch.zeros(images.size(0), 10)
        logits[:, 0] = 1.0
        return logits


def test_evaluate_returns_accuracy_percentage():
    images = torch.randn(4, 1, 28, 28)
    labels = torch.tensor([0, 0, 1, 1])
    dataloader = DataLoader(TensorDataset(images, labels), batch_size=2)

    accuracy = evaluate(AlwaysZeroModel(), dataloader, torch.device("cpu"))

    assert accuracy == 50.0


def test_checkpoint_roundtrip(tmp_path):
    checkpoint_path = tmp_path / "model.pth"
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=3,
        validation_accuracy=97.5,
    )

    reloaded_model = MLP()
    checkpoint = load_checkpoint(checkpoint_path, reloaded_model, torch.device("cpu"))

    assert checkpoint["epoch"] == 3
    assert checkpoint["validation_accuracy"] == 97.5
