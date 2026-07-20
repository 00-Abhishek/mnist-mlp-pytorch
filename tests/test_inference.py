import torch

from src.inference import predict
from src.model import MLP


def test_predict_returns_integer_class():
    model = MLP()
    image = torch.randn(1, 28, 28)

    prediction = predict(model, image, device=torch.device("cpu"))

    assert isinstance(prediction, int)
    assert 0 <= prediction <= 9
