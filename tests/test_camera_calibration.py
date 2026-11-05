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
from src.sensing.cam import RaspiCamera


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


def test_scene_camera_can_load_saved_calibration(tmp_path):
    import mujoco

    calibration_path = tmp_path / "camera_calibration.npz"
    save_calibration(
        {
            "camera_matrix": np.array(
                [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
            ),
            "distortion": np.zeros((1, 5)),
            "rms": 0.1,
            "image_width": 640,
            "image_height": 480,
            "views_used": 3,
        },
        calibration_path,
    )
    model = mujoco.MjModel.from_xml_path(
        str(Path(__file__).parents[1] / "assets" / "scene_turtlebot3_waffle_pi.xml")
    )

    camera = RaspiCamera(model, calibration_path=calibration_path)

    assert camera.calibrated
    assert camera.camera_matrix.shape == (3, 3)


def test_marker_pose_estimation_and_world_pose_inversion():
    import mujoco

    model = mujoco.MjModel.from_xml_path(
        str(Path(__file__).parents[1] / "assets" / "scene_turtlebot3_waffle_pi.xml")
    )
    camera = RaspiCamera(model)
    marker_size = 0.2
    object_points = np.array(
        [
            [-marker_size / 2, marker_size / 2, 0.0],
            [marker_size / 2, marker_size / 2, 0.0],
            [marker_size / 2, -marker_size / 2, 0.0],
            [-marker_size / 2, -marker_size / 2, 0.0],
        ]
    )
    expected_translation = np.array([0.1, -0.05, 1.5])
    expected_rotation = np.eye(3)
    image_points, _ = cv2.projectPoints(
        object_points,
        np.zeros((3, 1)),
        expected_translation.reshape(3, 1),
        np.array(
            [
                [camera.focal_length_px, 0.0, camera.width / 2.0],
                [0.0, camera.focal_length_px, camera.height / 2.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.zeros((5, 1)),
    )

    marker_pose = camera.estimate_marker_pose(
        image_points.reshape(4, 2), marker_size
    )
    np.testing.assert_allclose(marker_pose.translation, expected_translation, atol=1e-5)
    assert marker_pose.reprojection_error_px < 1e-5

    camera_position, camera_rotation = camera.camera_pose_from_marker(
        np.array([2.0, 3.0, 0.5]),
        expected_rotation,
        marker_pose,
    )
    np.testing.assert_allclose(
        camera_position,
        np.array([1.9, 3.05, -1.0]),
        atol=1e-5,
    )
    np.testing.assert_allclose(camera_rotation, expected_rotation, atol=1e-5)


def test_marker_pose_reprojects_across_distance_range():
    import mujoco

    model = mujoco.MjModel.from_xml_path(
        str(Path(__file__).parents[1] / "assets" / "scene_turtlebot3_waffle_pi.xml")
    )
    camera = RaspiCamera(model)
    side = 0.15
    object_points = np.array(
        [
            [-side / 2, side / 2, 0.0],
            [side / 2, side / 2, 0.0],
            [side / 2, -side / 2, 0.0],
            [-side / 2, -side / 2, 0.0],
        ]
    )
    camera_matrix = np.array(
        [
            [camera.focal_length_px, 0.0, camera.width / 2.0],
            [0.0, camera.focal_length_px, camera.height / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )

    for distance in (0.4, 1.0, 2.0, 4.0):
        image_points, _ = cv2.projectPoints(
            object_points,
            np.array([[np.pi], [0.0], [0.0]]),
            np.array([[0.05], [-0.03], [distance]]),
            camera_matrix,
            np.zeros((5, 1)),
        )
        pose = camera.estimate_marker_pose(image_points.reshape(4, 2), side)
        assert pose.reprojection_error_px < 1e-4
        np.testing.assert_allclose(
            pose.translation,
            np.array([0.05, -0.03, distance]),
            atol=1e-5,
        )
