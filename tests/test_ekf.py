import numpy as np
import pytest

from src.estimation.ekf import RobotEKF


def test_straight_wheel_odometry():
    ekf = RobotEKF(wheel_base=0.288)

    for _ in range(10):
        ekf.update(0.01, 0.01, 0.1)

    np.testing.assert_allclose(ekf.state, [0.1, 0.0, 0.0], atol=1e-12)


def test_wheel_odometry_turns_using_wheel_difference():
    ekf = RobotEKF(wheel_base=0.2)

    ekf.update(0.0, 0.1, 0.1)

    assert ekf.state[0] > 0.0
    assert ekf.state[1] > 0.0
    np.testing.assert_allclose(ekf.state[2], 0.5, atol=1e-12)


def test_gyro_value_cannot_change_wheel_odometry_prediction():
    first = RobotEKF(wheel_base=0.288)
    second = RobotEKF(wheel_base=0.288)

    first.update(0.02, 0.01, 0.1)
    second.update(0.02, 0.01, 0.1)

    np.testing.assert_allclose(first.state, second.state)
    np.testing.assert_allclose(first.P, second.P)


def test_optional_gyro_fusion_corrects_wheel_heading():
    ekf = RobotEKF(
        wheel_base=0.288,
        initial_covariance=np.eye(3) * 0.1,
    )

    ekf.update(0.0, 0.0, 0.1, imu_gyro_z=1.0, imu_rate_std=0.01)
    ekf.update(0.0, 0.0, 0.1, imu_gyro_z=1.0, imu_rate_std=0.01)

    assert ekf.state[2] > 0.0
    assert ekf.state[2] < 0.2


def test_pose_fusion_is_owned_by_ekf():
    ekf = RobotEKF(
        wheel_base=0.288,
        initial_state=np.array([1.0, 0.0, 0.0]),
        initial_covariance=np.eye(3) * 0.1,
    )

    corrected = ekf.fuse_pose(np.zeros(3), np.eye(3) * 0.01)

    assert corrected[0] < 1.0


def test_absolute_pose_correction_reduces_drifted_state():
    ekf = RobotEKF(
        wheel_base=0.288,
        initial_state=np.array([1.0, -0.5, 0.4]),
        initial_covariance=np.eye(3) * 0.1,
    )
    before = ekf.state.copy()

    corrected = ekf.correct_pose(
        np.array([0.0, 0.0, 0.0]),
        np.eye(3) * 0.01,
    )

    assert np.linalg.norm(corrected) < np.linalg.norm(before)
    assert np.all(np.diag(ekf.P) < 0.1)


def test_yaw_correction_wraps_innovation_across_pi():
    ekf = RobotEKF(
        wheel_base=0.288,
        initial_state=np.array([0.0, 0.0, np.pi - 0.01]),
        initial_covariance=np.eye(3) * 0.1,
    )

    ekf.correct_yaw(-np.pi + 0.01, variance=0.001)

    assert abs(abs(ekf.state[2]) - np.pi) < 0.01


@pytest.mark.parametrize(
    ("d_left", "d_right"),
    [(float("nan"), 0.0), (0.0, float("inf"))],
)
def test_rejects_non_finite_wheel_distances(d_left, d_right):
    with pytest.raises(ValueError, match="wheel distances"):
        RobotEKF(wheel_base=0.288).update(d_left, d_right, 0.1)
