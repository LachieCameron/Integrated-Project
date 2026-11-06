"""Reusable visual docking control primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.kinematics.differential_drive import wrap_angle


@dataclass(frozen=True)
class DockingConfig:
    target_width_px: float = 250.0
    center_deadband: float = 0.04
    center_release: float = 0.08
    center_filter_alpha: float = 0.35
    center_before_drive: float = 0.10
    max_linear_v: float = 0.06
    max_angular_w: float = 0.18
    linear_gain: float = 0.6
    angular_gain: float = 1.8
    heading_correction_threshold: float = math.radians(8.0)
    heading_gain: float = 1.2
    min_marker_width_px: float = 100.0
    hard_stop_distance: float = 0.14
    complete_distance: float = 0.14
    terminal_speed: float = 0.025
    center_tolerance: float = 0.12
    confirm_frames: int = 3


class VisualDockingController:
    """Final tag-visual docking controller shared by Track A entry points."""

    def __init__(self, config: DockingConfig | None = None):
        self.config = config or DockingConfig()
        self.confirm_count = 0
        self._filtered_signed_error = 0.0
        self._aligned = False
        self._straight_approach = False

    def reset(self) -> None:
        self.confirm_count = 0
        self._filtered_signed_error = 0.0
        self._aligned = False
        self._straight_approach = False

    @property
    def straight_approach(self) -> bool:
        return self._straight_approach

    def command(
        self,
        marker_width_px: float | None,
        center_error: float | None,
        signed_center_error: float | None,
        front_distance: float,
        heading: float | None = None,
        target_heading: float | None = None,
    ) -> tuple[float, float, bool]:
        """Return ``(linear_v, angular_w, docked)`` for one camera update.

        ``center_error`` is normalized to the half image width. A missing marker
        stops forward motion; callers may provide their own search rotation.
        """
        cfg = self.config
        if self._straight_approach and not np.isfinite(front_distance):
            self.confirm_count = 0
            return 0.0, 0.0, False
        if marker_width_px is None or center_error is None or signed_center_error is None:
            if self._straight_approach:
                reached = (
                    0.0 <= front_distance <= cfg.complete_distance
                )
                self.confirm_count = self.confirm_count + 1 if reached else 0
                if front_distance > cfg.hard_stop_distance:
                    return cfg.terminal_speed, 0.0, False
                return 0.0, 0.0, self.confirm_count >= cfg.confirm_frames
            self.confirm_count = 0
            return 0.0, 0.0, False

        center_error = float(center_error)
        signed_center_error = float(signed_center_error)
        alpha = float(np.clip(cfg.center_filter_alpha, 0.0, 1.0))
        self._filtered_signed_error = (
            (1.0 - alpha) * self._filtered_signed_error
            + alpha * signed_center_error
        )
        filtered_center_error = abs(self._filtered_signed_error)
        if self._aligned and filtered_center_error > cfg.center_release:
            self._aligned = False
        elif not self._aligned and filtered_center_error <= cfg.center_deadband:
            self._aligned = True

        if (
            not self._straight_approach
            and self._aligned
            and center_error <= cfg.center_before_drive
            and marker_width_px >= cfg.target_width_px
        ):
            self._straight_approach = True

        if self._straight_approach:
            angular = 0.0
            linear = cfg.terminal_speed if front_distance > cfg.hard_stop_distance else 0.0
        else:
            if self._aligned:
                angular = 0.0
            else:
                angular = float(np.clip(
                    -cfg.angular_gain * self._filtered_signed_error,
                    -cfg.max_angular_w,
                    cfg.max_angular_w,
                ))
            if center_error > cfg.center_before_drive:
                linear = 0.0
            elif marker_width_px < cfg.target_width_px:
                linear = float(np.clip(
                    cfg.linear_gain * (cfg.target_width_px - marker_width_px)
                    / cfg.target_width_px,
                    0.0,
                    cfg.max_linear_v,
                ))
            else:
                linear = cfg.terminal_speed

        if (
            not self._straight_approach
            and not self._aligned
            and filtered_center_error > cfg.center_release
            and heading is not None
            and target_heading is not None
            and abs(float(wrap_angle(target_heading - heading)))
            > cfg.heading_correction_threshold
        ):
            linear = 0.0
            angular = float(np.clip(
                cfg.heading_gain * float(wrap_angle(target_heading - heading)),
                -cfg.max_angular_w,
                cfg.max_angular_w,
            ))

        reached = (
            0.0 <= front_distance <= cfg.complete_distance
            and marker_width_px >= cfg.min_marker_width_px
            and filtered_center_error <= cfg.center_tolerance
        )
        self.confirm_count = self.confirm_count + 1 if reached else 0
        return linear, angular, self.confirm_count >= cfg.confirm_frames


def pose_command(
    pose: np.ndarray,
    target: np.ndarray,
    max_linear_v: float = 0.10,
    max_angular_w: float = 0.45,
) -> tuple[float, float, bool]:
    """Drive an estimated pose to a pre-dock pose."""
    pose = np.asarray(pose, dtype=float)
    target = np.asarray(target, dtype=float)
    dx, dy = target[:2] - pose[:2]
    distance = float(np.hypot(dx, dy))
    target_heading = math.atan2(dy, dx)
    heading_error = float(wrap_angle(target_heading - pose[2]))
    final_heading_error = float(wrap_angle(target[2] - pose[2]))

    if distance > 0.10:
        linear = float(np.clip(0.7 * distance, 0.0, max_linear_v))
        angular = float(np.clip(2.0 * heading_error, -max_angular_w, max_angular_w))
        if abs(heading_error) > math.radians(35.0):
            linear = 0.0
        return linear, angular, False
    if abs(final_heading_error) > math.radians(8.0):
        return 0.0, float(np.clip(2.0 * final_heading_error, -max_angular_w, max_angular_w)), False
    return 0.0, 0.0, True
