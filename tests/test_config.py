from pathlib import Path

import pytest

from src.utils.config import load_config


CONFIG = Path(__file__).parents[1] / "config" / "default.yaml"


def test_loads_default_config():
    config = load_config(CONFIG)

    assert config.robot.wheel_base is None
    assert config.ekf.wheel_distance_std == 0.002
    assert config.ekf.initial_covariance == (0.01, 0.01, 0.01)
    assert config.tag_follower.target_ids == (0, 99)


def test_rejects_invalid_tuning_value(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text(
        CONFIG.read_text().replace("stop_distance: 0.20", "stop_distance: -0.20"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stop_distance"):
        load_config(path)
