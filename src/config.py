"""Project configuration for MNIST MLP training and inference."""

from pathlib import Path


SEED = 42
BATCH_SIZE = 64
INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 128
NUM_CLASSES = 10
LEARNING_RATE = 0.001
EPOCHS = 5
VALIDATION_SPLIT = 0.2

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
IMAGE_DIR = ROOT_DIR / "images"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
