"""Legacy gyro-assisted wheel-odometry estimator.

This module is kept separately from :mod:`src.estimation.ekf` so the
wheel-odometry-only estimator remains the active implementation while the
gyro-assisted approach can be evaluated and fused later.
"""

import numpy as np
import mujoco

from src.kinematics.robot_parameters import parameters_from_model


class RobotEKFGyro:
    """Pose estimator using wheel distance and gyro yaw-rate increments.

    This preserves the original estimator behaviour: when the gyro rate is
    non-zero, it supplies the heading increment; otherwise wheel differential
    odometry supplies it.  It is intentionally separate from the current EKF
    until the gyro is fused with a defined measurement-noise model.
    """

    def __init__(
        self,
        model: mujoco.MjModel | None = None,
        wheel_base: float | None = None,
    ) -> None:
        if model is not None:
            wheel_base = parameters_from_model(model).wheel_base
        if wheel_base is None or wheel_base <= 0.0:
            raise ValueError("wheel_base must be positive")

        self.L = float(wheel_base)
        self.state = np.zeros(3, dtype=float)  # [x, y, yaw]
        self.P = np.eye(3, dtype=float) * 0.01

    def update(
        self,
        d_left: float,
        d_right: float,
        gyro_z: float,
        dt: float,
    ) -> np.ndarray:
        """Advance the estimate using wheel distances and gyro yaw rate."""
        values = np.asarray([d_left, d_right, gyro_z, dt], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("wheel distances, gyro rate, and dt must be finite")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        d_center = 0.5 * (d_left + d_right)
        d_theta = (
            gyro_z * dt
            if abs(gyro_z) > 0.0001
            else (d_right - d_left) / self.L
        )

        self.state[0] += d_center * np.cos(self.state[2])
        self.state[1] += d_center * np.sin(self.state[2])
        self.state[2] = (self.state[2] + d_theta) % (2.0 * np.pi)
        return self.state.copy()
