from __future__ import annotations

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from src.kinematics.differential_drive import DifferentialDrive, wrap_angle


def discretise(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    #Zero-order-hold discretisation via the matrix exponential
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    return Md[:n, :n], Md[:n, n:]


def controllability_rank(A: np.ndarray, B: np.ndarray, tol: float = 1e-9) -> int:
    #Rank of the controllability matrix ``[B, AB, ..., A^(n-1) B]``.
    n = A.shape[0]
    blocks = [B]
    for _ in range(n - 1):
        blocks.append(A @ blocks[-1])
    return int(np.linalg.matrix_rank(np.hstack(blocks), tol=tol))


def body_frame_error(pose: np.ndarray, ref: np.ndarray) -> np.ndarray:
    #Reference-minus-actual pose, rotated into the robot frame.
    dx = ref[0] - pose[0]
    dy = ref[1] - pose[1]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.array([c * dx + s * dy, -s * dx + c * dy, float(wrap_angle(ref[2] - pose[2]))])


class LQRTrajectoryTracker:
    #Gain-scheduled discrete-time LQR for pose tracking.

    # Bryson's-rule maxima: (ex, ey, etheta) and (dv, dw).
    DEFAULT_E_MAX = (0.10, 0.05, 0.20) # m, m, rad
    DEFAULT_DU_MAX = (0.10, 0.60)  # m/s, rad/s

    def __init__(
        self,
        dt: float,
        q_diag=None,
        r_diag=None,
        v_max: float = 0.20,
        w_max: float = 1.0,
        v_lin_min: float = 0.05,
        allow_reverse: bool = False,
    ):
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.Q = np.diag(
            1.0 / np.square(self.DEFAULT_E_MAX) if q_diag is None else np.asarray(q_diag, float)
        )
        self.R = np.diag(
            1.0 / np.square(self.DEFAULT_DU_MAX) if r_diag is None else np.asarray(r_diag, float)
        )
        self.v_max = float(v_max)
        self.w_max = float(w_max)
        self.v_lin_min = float(v_lin_min)
        self.allow_reverse = bool(allow_reverse)
        self._gain_cache: dict[tuple[float, float], np.ndarray] = {}

    # ------------------------------------------------------------------
    def gain(self, v_ref: float, w_ref: float) -> np.ndarray:
        #Return the 2x3 feedback gain ``K`` for the given reference twist.
        v_lin = max(abs(v_ref), self.v_lin_min)
        w_lin = float(np.clip(w_ref, -self.w_max, self.w_max))
        key = (round(v_lin / 0.01) * 0.01, round(w_lin / 0.05) * 0.05)
        K = self._gain_cache.get(key)
        if K is None:
            A, B = DifferentialDrive.error_dynamics(*key)
            Ad, Bd = discretise(A, B, self.dt)
            P = solve_discrete_are(Ad, Bd, self.Q, self.R)
            K = np.linalg.solve(self.R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
            self._gain_cache[key] = K
        return K

    def control(self, pose_est: np.ndarray, ref: np.ndarray, v_ref: float, w_ref: float):
        # ``pose_est`` is the EKF estimate ``[x, y, theta]``; ``ref`` is the
        # reference pose ``[x_r, y_r, theta_r]`` with twist ``(v_ref, w_ref)``.
    
        e = body_frame_error(np.asarray(pose_est, float), np.asarray(ref, float))
        du = -self.gain(v_ref, w_ref) @ e
        v = v_ref + du[0]
        w = w_ref + du[1]
        v_lo = -self.v_max if self.allow_reverse else 0.0
        v = float(np.clip(v, v_lo, self.v_max))
        w = float(np.clip(w, -self.w_max, self.w_max))
        return v, w, e
