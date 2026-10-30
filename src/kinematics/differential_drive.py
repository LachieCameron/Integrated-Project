#Differential-drive (unicycle) kinematic model for the TurtleBot3 Waffle Pi.

from __future__ import annotations

import numpy as np

from src.kinematics.robot_parameters import parameters_from_model


def wrap_angle(angle):
    #Wrap an angle (scalar or array) to [-pi, pi).
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


class DifferentialDrive:
    #Kinematics of a two-wheel differential-drive base.

    def __init__(self, wheel_base: float, wheel_radius: float, max_wheel_speed: float = 7.88):
        if wheel_base <= 0.0 or wheel_radius <= 0.0 or max_wheel_speed <= 0.0:
            raise ValueError("wheel_base, wheel_radius and max_wheel_speed must be positive")
        self.L = float(wheel_base)
        self.r = float(wheel_radius)
        self.max_wheel_speed = float(max_wheel_speed)

    @classmethod
    def from_model(cls, model) -> "DifferentialDrive":
        #Build from the loaded MuJoCo model (geometry and actuator limit).
        p = parameters_from_model(model)
        limit = float(abs(model.actuator_ctrlrange[0, 1])) if model.nu > 0 else 7.88
        return cls(p.wheel_base, p.wheel_radius, limit)

    @property
    def max_linear_speed(self) -> float:
        #Forward speed when both wheels run at the actuator limit [m/s].
        return self.r * self.max_wheel_speed

    # ------------------------------------------------------------------
    # Twist <-> wheel speeds
    # ------------------------------------------------------------------
    def twist_to_wheel_speeds(self, v: float, w: float) -> tuple[float, float]:
        #Convert (v, w) to wheel speeds [rad/s].

        left = (v - 0.5 * w * self.L) / self.r
        right = (v + 0.5 * w * self.L) / self.r
        peak = max(abs(left), abs(right))
        if peak > self.max_wheel_speed:
            scale = self.max_wheel_speed / peak
            left *= scale
            right *= scale
        return float(left), float(right)

    def wheel_speeds_to_twist(self, left: float, right: float) -> tuple[float, float]:
        #Inverse of :meth:`twist_to_wheel_speeds` (no saturation).
        v = 0.5 * self.r * (left + right)
        w = self.r * (right - left) / self.L
        return float(v), float(w)

    # ------------------------------------------------------------------
    # Motion model
    # ------------------------------------------------------------------
    def step(self, state: np.ndarray, v: float, w: float, dt: float) -> np.ndarray:
        #Exact (constant-twist) discrete update of the unicycle model.
        x, y, th = (float(s) for s in state)
        if abs(w) < 1e-9:
            x += v * dt * np.cos(th)
            y += v * dt * np.sin(th)
        else:
            x += v / w * (np.sin(th + w * dt) - np.sin(th))
            y += v / w * (-np.cos(th + w * dt) + np.cos(th))
            th += w * dt
        return np.array([x, y, float(wrap_angle(th))])

    # ------------------------------------------------------------------
    # Linearisation about a reference trajectory
    # ------------------------------------------------------------------
    @staticmethod
    def error_dynamics(v_ref: float, w_ref: float) -> tuple[np.ndarray, np.ndarray]:
        
        A = np.array(
            [
                [0.0, w_ref, 0.0],
                [-w_ref, 0.0, v_ref],
                [0.0, 0.0, 0.0],
            ]
        )
        B = np.array([[-1.0, 0.0], [0.0, 0.0], [0.0, -1.0]])
        return A, B
