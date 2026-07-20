"""Training workflow for the MNIST MLP project."""

import torch.nn as nn
import torch.optim as optim

from src.config import BEST_MODEL_PATH, EPOCHS, LEARNING_RATE
from src.dataset import create_dataloaders
from src.inference import predict
from src.model import MLP
from src.utils import (
    collect_predictions,
    ensure_project_dirs,
    evaluate,
    get_device,
    load_checkpoint,
    print_classification_report,
    save_checkpoint,
    save_confusion_matrix,
    save_training_plots,
    set_seed,
)


def train():
    """Train the model, save artifacts, and return final metrics."""
    set_seed()
    ensure_project_dirs()

    device = get_device()
    print(f"Using Device: {device}")

    train_loader, validation_loader = create_dataloaders()
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
    }
    best_accuracy = 0.0

    print("\nStarting Training...\n")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        train_accuracy = evaluate(model, train_loader, device)
        validation_accuracy = evaluate(model, validation_loader, device)

        history["train_loss"].append(average_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_accuracy"].append(validation_accuracy)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Loss: {average_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Validation Accuracy: {validation_accuracy:.2f}%"
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            save_checkpoint(
                BEST_MODEL_PATH,
                model,
                optimizer,
                epoch + 1,
                validation_accuracy,
            )
            print("Best model saved.\n")

    print("=" * 60)
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
    print("=" * 60)

    checkpoint = load_checkpoint(BEST_MODEL_PATH, model, device)
    print("\nBest model loaded successfully!")

    true_labels, predicted_labels = collect_predictions(
        model,
        validation_loader,
        device,
    )
    save_training_plots(history)
    save_confusion_matrix(true_labels, predicted_labels)
    report = print_classification_report(true_labels, predicted_labels)

    image, label = validation_loader.dataset[0]
    prediction = predict(model, image, device)

    print("\nInference Demo")
    print("----------------")
    print(f"Actual Label     : {label}")
    print(f"Predicted Label  : {prediction}")

    return {
        "best_validation_accuracy": checkpoint["validation_accuracy"],
        "history": history,
        "classification_report": report,
        "demo_actual": label,
        "demo_prediction": prediction,
    }


if __name__ == "__main__":
    train()
