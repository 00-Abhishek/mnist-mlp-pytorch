"""Basic sanity checks for project configuration."""

from pathlib import Path

from src import config


def test_config_paths_are_path_objects() -> None:
    """Configuration path values should be pathlib.Path instances."""
    assert isinstance(config.ROOT_DIR, Path)
    assert isinstance(config.DATA_DIR, Path)
    assert isinstance(config.CHECKPOINT_DIR, Path)
    assert isinstance(config.IMAGE_DIR, Path)
    assert isinstance(config.BEST_MODEL_PATH, Path)
