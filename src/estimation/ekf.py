"""Wheel-odometry extended Kalman filter for a differential-drive robot."""

import numpy as np


def _wrap_angle(angle: float) -> float:
    """Return an angle in the interval [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class RobotEKF:
    """Estimate planar pose using incremental wheel-encoder odometry.

    The state is ``[x, y, yaw]`` in metres and radians.  The encoder supplies
    left and right wheel *distances* for each time step; therefore the wheel
    odometry is the prediction input and is not replaced by an IMU heading
    increment.
    """

    def __init__(
        self,
        wheel_base: float = 0.287,
        wheel_distance_std: float = 0.002,
        initial_state: np.ndarray | None = None,
        initial_covariance: np.ndarray | None = None,
    ) -> None:
        if wheel_base <= 0.0:
            raise ValueError("wheel_base must be positive")
        if wheel_distance_std <= 0.0:
            raise ValueError("wheel_distance_std must be positive")

        self.L = float(wheel_base)
        self.wheel_distance_variance = float(wheel_distance_std**2)
        self.state = (
            np.zeros(3, dtype=float)
            if initial_state is None
            else np.asarray(initial_state, dtype=float).copy()
        )
        self.P = (
            np.eye(3, dtype=float) * 0.01
            if initial_covariance is None
            else np.asarray(initial_covariance, dtype=float).copy()
        )

        if self.state.shape != (3,):
            raise ValueError("initial_state must have shape (3,)")
        if self.P.shape != (3, 3):
            raise ValueError("initial_covariance must have shape (3, 3)")
        self.state[2] = _wrap_angle(float(self.state[2]))

    def predict(self, d_left: float, d_right: float) -> np.ndarray:
        """Propagate pose and covariance from one pair of wheel distances."""
        d_left = float(d_left)
        d_right = float(d_right)
        if not np.isfinite([d_left, d_right]).all():
            raise ValueError("wheel distances must be finite")

        d_center = 0.5 * (d_left + d_right)
        d_yaw = (d_right - d_left) / self.L
        yaw = float(self.state[2])
        midpoint_yaw = yaw + 0.5 * d_yaw

        self.state[0] += d_center * np.cos(midpoint_yaw)
        self.state[1] += d_center * np.sin(midpoint_yaw)
        self.state[2] = _wrap_angle(yaw + d_yaw)

        # Jacobian of the midpoint differential-drive motion model.
        F = np.eye(3)
        F[0, 2] = -d_center * np.sin(midpoint_yaw)
        F[1, 2] = d_center * np.cos(midpoint_yaw)

        # Jacobian with respect to [d_left, d_right].
        G = np.array(
            [
                [
                    0.5 * np.cos(midpoint_yaw)
                    + d_center * np.sin(midpoint_yaw) / (2.0 * self.L),
                    0.5 * np.cos(midpoint_yaw)
                    - d_center * np.sin(midpoint_yaw) / (2.0 * self.L),
                ],
                [
                    0.5 * np.sin(midpoint_yaw)
                    - d_center * np.cos(midpoint_yaw) / (2.0 * self.L),
                    0.5 * np.sin(midpoint_yaw)
                    + d_center * np.cos(midpoint_yaw) / (2.0 * self.L),
                ],
                [-1.0 / self.L, 1.0 / self.L],
            ],
            dtype=float,
        )
        wheel_noise = np.eye(2) * self.wheel_distance_variance
        self.P = F @ self.P @ F.T + G @ wheel_noise @ G.T
        self.P = 0.5 * (self.P + self.P.T)
        return self.state.copy()

    def update(self, d_left: float, d_right: float, dt: float) -> np.ndarray:
        """Advance the filter using encoder distances for one time step.

        ``dt`` is accepted to keep the estimator interface aligned with the
        simulation loop.  Wheel encoder values are already distances, so it
        does not alter the prediction.
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a positive finite value")
        return self.predict(d_left, d_right)

    def correct_pose(
        self,
        measurement: np.ndarray,
        covariance: np.ndarray,
    ) -> np.ndarray:
        """Correct the wheel prediction with an absolute ``[x, y, yaw]`` pose.

        The measurement must be expressed in the same world frame as the
        filter state.  A camera/ArUco or other landmark-localisation pipeline
        can supply this measurement; simulator ground truth must not.
        """
        measurement = np.asarray(measurement, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        if measurement.shape != (3,):
            raise ValueError("measurement must have shape (3,)")
        if covariance.shape != (3, 3):
            raise ValueError("covariance must have shape (3, 3)")
        if not np.isfinite(measurement).all() or not np.isfinite(covariance).all():
            raise ValueError("measurement and covariance must be finite")
        if not np.allclose(covariance, covariance.T):
            raise ValueError("covariance must be symmetric")

        innovation = measurement - self.state
        innovation[2] = _wrap_angle(float(innovation[2]))
        innovation_covariance = self.P + covariance

        try:
            gain = np.linalg.solve(
                innovation_covariance,
                self.P,
            ).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("measurement covariance is not positive definite") from exc

        self.state += gain @ innovation
        self.state[2] = _wrap_angle(float(self.state[2]))
        identity = np.eye(3)
        self.P = (identity - gain) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        return self.state.copy()

    def correct_yaw(self, yaw: float, variance: float) -> np.ndarray:
        """Correct heading with an absolute yaw measurement."""
        if not np.isfinite(yaw) or not np.isfinite(variance) or variance <= 0.0:
            raise ValueError("yaw and variance must be finite; variance must be positive")

        innovation = _wrap_angle(float(yaw) - float(self.state[2]))
        innovation_variance = float(self.P[2, 2] + variance)
        gain = self.P[:, 2] / innovation_variance
        self.state += gain * innovation
        self.state[2] = _wrap_angle(float(self.state[2]))
        self.P -= np.outer(gain, self.P[2, :])
        self.P = 0.5 * (self.P + self.P.T)
        return self.state.copy()
