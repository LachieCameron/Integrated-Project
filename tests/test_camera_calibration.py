import sys
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.sensing.camera_calibration import (
    calibrate_from_images,
    find_checkerboard,
    load_calibration,
    save_calibration,
)


def test_checkerboard_texture_has_expected_inner_corners():
    path = Path(__file__).parents[1] / "assets" / "textures" / "cal_checkerboard.png"
    image = cv2.imread(str(path))

    found, corners = find_checkerboard(image)

    assert found
    assert corners is not None
    assert corners.shape == (30, 1, 2)


def test_calibration_requires_multiple_views(tmp_path):
    path = Path(__file__).parents[1] / "assets" / "textures" / "cal_checkerboard.png"
    with np.testing.assert_raises(ValueError):
        calibrate_from_images([path])


def test_calibration_artifact_round_trip(tmp_path):
    output = tmp_path / "camera_calibration.npz"
    calibration = {
        "camera_matrix": np.eye(3),
        "distortion": np.zeros((1, 5)),
        "rms": 0.1,
        "image_width": 640,
        "image_height": 480,
        "views_used": 3,
    }

    save_calibration(calibration, output)
    camera_matrix, distortion = load_calibration(output)

    np.testing.assert_allclose(camera_matrix, np.eye(3))
    np.testing.assert_allclose(distortion, np.zeros((1, 5)))
