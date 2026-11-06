"""Wheel-odometry EKF with gyro increment fusion and gated absolute-pose updates.

State: [x, y, yaw] in metres / radians, world frame.

"""

from __future__ import annotations

import math

import numpy as np
import mujoco

from src.kinematics.robot_parameters import parameters_from_model

# 99% chi-square threshold for 3 degrees of freedom.
GATE_CHI2_3DOF_99 = 11.345


def wrap_angle(angle: float) -> float:
    """Return an angle in the interval [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def camera_pose_covariance(
    range_m: float,
    xy_std_min: float = 0.01,
    xy_std_per_m2: float = 0.01,
    yaw_std_min: float = math.radians(1.0),
    yaw_std_per_m: float = math.radians(1.0),
) -> np.ndarray:
    """Return a diagonal covariance for a camera pose measurement."""
    r = max(float(range_m), 0.0)
    xy_std = xy_std_min + xy_std_per_m2 * r**2
    yaw_std = yaw_std_min + yaw_std_per_m * r
    return np.diag([xy_std**2, xy_std**2, yaw_std**2])


class RobotEKF:
    """Planar pose EKF: wheel odometry + gyro increments + absolute pose fixes."""

    def __init__(
        self,
        model: mujoco.MjModel | None = None,
        wheel_base: float | None = None,
        wheel_noise_coeff: float = 1e-4,
        wheel_noise_floor_std: float = 1e-5,
        initial_state: np.ndarray | None = None,
        initial_covariance: np.ndarray | None = None,
    ) -> None:
        """
        wheel_noise_coeff:      wheel distance variance per metre travelled (m^2 / m).
        wheel_noise_floor_std:  per-step std (m) so variance never reaches zero.
        """
        if model is not None:
            wheel_base = parameters_from_model(model).wheel_base
        if wheel_base is None or wheel_base <= 0.0:
            raise ValueError("wheel_base must be positive")
        if wheel_noise_coeff < 0.0 or wheel_noise_floor_std <= 0.0:
            raise ValueError("wheel noise parameters must be positive")

        self.L = float(wheel_base)
        self.wheel_noise_coeff = float(wheel_noise_coeff)
        self.wheel_noise_floor_var = float(wheel_noise_floor_std) ** 2

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
        self.state[2] = wrap_angle(float(self.state[2]))

        self._initial_state = self.state.copy()
        self._initial_P = self.P.copy()

        # Diagnostics from the most recent fuse_pose() call.
        self.last_nis: float | None = None
        self.last_innovation: np.ndarray | None = None
        self.last_accepted: bool | None = None
        self.n_accepted = 0
        self.n_rejected = 0

    def reset(
        self,
        state: np.ndarray | None = None,
        covariance: np.ndarray | None = None,
    ) -> None:
        """Reset to the initial (or given) estimate and clear diagnostics."""
        self.state = (
            self._initial_state.copy()
            if state is None
            else np.asarray(state, dtype=float).copy()
        )
        self.state[2] = wrap_angle(float(self.state[2]))
        self.P = (
            self._initial_P.copy()
            if covariance is None
            else np.asarray(covariance, dtype=float).copy()
        )
        self.last_nis = None
        self.last_innovation = None
        self.last_accepted = None
        self.n_accepted = 0
        self.n_rejected = 0

    @property
    def sigma(self) -> np.ndarray:
        """1-sigma standard deviations of [x, y, yaw]."""
        return np.sqrt(np.maximum(np.diag(self.P), 0.0))

    def nees(self, true_state: np.ndarray) -> float:
        """Normalised estimation error squared. EVALUATION ONLY.

        ``true_state`` must come from evaluation code (simulator ground
        truth), never from the navigation path. For a consistent filter the
        mean NEES over time/trials is close to 3 (the state dimension).
        """
        true_state = np.asarray(true_state, dtype=float)
        err = true_state - self.state
        err[2] = wrap_angle(float(err[2]))
        return float(err @ np.linalg.solve(self.P, err))

    def _wheel_variance(self, distance: float) -> float:
        return self.wheel_noise_coeff * abs(distance) + self.wheel_noise_floor_var

    # --------------------------------------------------------------- predict
    def predict(
        self,
        d_left: float,
        d_right: float,
        gyro_dyaw: float | None = None,
        gyro_dyaw_var: float | None = None,
    ) -> np.ndarray:
        """Propagate with wheel distances; optionally fuse a gyro yaw increment.

        gyro_dyaw      : yaw increment from the gyro, omega * dt (rad).
        gyro_dyaw_var  : variance of that increment (rad^2).
        """
        d_left = float(d_left)
        d_right = float(d_right)
        if not np.isfinite([d_left, d_right]).all():
            raise ValueError("wheel distances must be finite")

        var_l = self._wheel_variance(d_left)
        var_r = self._wheel_variance(d_right)

        d_center = 0.5 * (d_left + d_right)
        var_center = 0.25 * (var_l + var_r)

        dyaw = (d_right - d_left) / self.L
        var_dyaw = (var_l + var_r) / self.L**2
        # NOTE: cov(d_center, dyaw_wheel) = (var_r - var_l) / (2 L) is ignored.
        # It is zero when both wheels move equally and small otherwise.

        if gyro_dyaw is not None:
            if gyro_dyaw_var is None or not gyro_dyaw_var > 0.0:
                raise ValueError("gyro_dyaw_var must be positive with gyro_dyaw")
            if not np.isfinite(gyro_dyaw):
                raise ValueError("gyro_dyaw must be finite")
            w_wheel = 1.0 / var_dyaw
            w_gyro = 1.0 / float(gyro_dyaw_var)
            dyaw = (dyaw * w_wheel + float(gyro_dyaw) * w_gyro) / (w_wheel + w_gyro)
            var_dyaw = 1.0 / (w_wheel + w_gyro)

        yaw = float(self.state[2])
        mid = yaw + 0.5 * dyaw
        c, s = math.cos(mid), math.sin(mid)

        self.state[0] += d_center * c
        self.state[1] += d_center * s
        self.state[2] = wrap_angle(yaw + dyaw)

        # Jacobian w.r.t. the previous state.
        F = np.eye(3)
        F[0, 2] = -d_center * s
        F[1, 2] = d_center * c

        # Jacobian w.r.t. the inputs [d_center, dyaw].
        G = np.array(
            [
                [c, -0.5 * d_center * s],
                [s, 0.5 * d_center * c],
                [0.0, 1.0],
            ]
        )
        Q_in = np.diag([var_center, var_dyaw])

        self.P = F @ self.P @ F.T + G @ Q_in @ G.T
        self.P = 0.5 * (self.P + self.P.T)
        return self.state.copy()

    def update(
        self,
        d_left: float,
        d_right: float,
        dt: float,
        imu_gyro_z: float | None = None,
        imu_rate_std: float | None = None,
    ) -> np.ndarray:
        """Predict from encoder distances, fusing the gyro rate as an increment.

        imu_rate_std is the gyro rate noise std (rad/s); the increment
        variance is (imu_rate_std * dt)^2.
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a positive finite value")
        if imu_gyro_z is None:
            return self.predict(d_left, d_right)
        if imu_rate_std is None or not np.isfinite(imu_rate_std) or imu_rate_std <= 0.0:
            raise ValueError("imu_rate_std must be positive with imu_gyro_z")
        return self.predict(
            d_left,
            d_right,
            gyro_dyaw=float(imu_gyro_z) * dt,
            gyro_dyaw_var=(float(imu_rate_std) * dt) ** 2,
        )

    # ---------------------------------------------------------------- update
    def fuse_pose(
        self,
        measurement: np.ndarray,
        covariance: np.ndarray,
        gate_chi2: float | None = None,
    ) -> np.ndarray:
        """Correct with an absolute [x, y, yaw] pose (H = I), Joseph form.

        The measurement must be in the same world frame as the state and must
        come from a sensor pipeline (e.g. ArUco + known marker map), never
        from simulator ground truth.

        gate_chi2 : if given, reject the update when NIS exceeds it
                    (use GATE_CHI2_3DOF_99 for a 99% gate).

        Returns the current state estimate. NIS, innovation and accept/reject
        are stored in last_nis, last_innovation, and last_accepted.
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
        innovation[2] = wrap_angle(float(innovation[2]))
        S = self.P + covariance

        try:
            nis = float(innovation @ np.linalg.solve(S, innovation))
            gain = np.linalg.solve(S, self.P).T  # K = P S^-1 (P, S symmetric)
        except np.linalg.LinAlgError as exc:
            raise ValueError("innovation covariance is not positive definite") from exc

        self.last_nis = nis
        self.last_innovation = innovation.copy()

        if gate_chi2 is not None and nis > gate_chi2:
            self.last_accepted = False
            self.n_rejected += 1
            return self.state.copy()

        self.state += gain @ innovation
        self.state[2] = wrap_angle(float(self.state[2]))

        A = np.eye(3) - gain
        self.P = A @ self.P @ A.T + gain @ covariance @ gain.T
        self.P = 0.5 * (self.P + self.P.T)

        self.last_accepted = True
        self.n_accepted += 1
        return self.state.copy()

    def correct_pose(
        self,
        measurement: np.ndarray,
        covariance: np.ndarray,
        gate_chi2: float | None = None,
    ) -> np.ndarray:
        """Alias for :meth:`fuse_pose`."""
        return self.fuse_pose(measurement, covariance, gate_chi2)

    def correct_yaw(self, yaw: float, variance: float) -> np.ndarray:
        """Correct heading with an absolute yaw measurement (e.g. a compass)."""
        if not np.isfinite(yaw) or not np.isfinite(variance) or variance <= 0.0:
            raise ValueError("yaw and variance must be finite; variance must be positive")

        innovation = wrap_angle(float(yaw) - float(self.state[2]))
        gain = self.P[:, 2] / float(self.P[2, 2] + variance)
        self.state += gain * innovation
        self.state[2] = wrap_angle(float(self.state[2]))
        # Joseph form for H = [0, 0, 1].
        H = np.array([[0.0, 0.0, 1.0]])
        A = np.eye(3) - np.outer(gain, H)
        self.P = A @ self.P @ A.T + variance * np.outer(gain, gain)
        self.P = 0.5 * (self.P + self.P.T)
        return self.state.copy()