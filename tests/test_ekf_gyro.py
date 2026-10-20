import numpy as np
import pytest

from src.estimation.ekf_gyro import RobotEKFGyro


def test_gyro_rate_controls_heading_increment():
    ekf = RobotEKFGyro(wheel_base=0.2)

    state = ekf.update(0.1, 0.1, gyro_z=1.0, dt=0.1)

    np.testing.assert_allclose(state, [0.1, 0.0, 0.1])


def test_wheel_difference_is_fallback_when_gyro_is_stationary():
    ekf = RobotEKFGyro(wheel_base=0.2)

    state = ekf.update(0.0, 0.1, gyro_z=0.0, dt=0.1)

    np.testing.assert_allclose(state[2], 0.5)


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="dt"):
        RobotEKFGyro(wheel_base=0.288).update(0.0, 0.0, 0.0, 0.0)
