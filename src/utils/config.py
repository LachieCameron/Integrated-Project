"""Validated application configuration loaded from YAML."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RobotConfig:
    wheel_base: float | None
    wheel_radius_left: float | None
    wheel_radius_right: float | None
    max_wheel_speed: float | None


@dataclass(frozen=True)
class EkfConfig:
    wheel_distance_std: float
    initial_covariance: tuple[float, float, float]
    camera_measurement_covariance: tuple[float, float, float]
    gyro_enabled: bool
    gyro_rate_std: float


@dataclass(frozen=True)
class LqrConfig:
    q: tuple[float, float, float]
    r: tuple[float, float]
    max_linear_velocity: float
    max_angular_velocity: float


@dataclass(frozen=True)
class TagFollowerConfig:
    target_ids: tuple[int, ...]
    kp_linear: float
    kp_angular: float
    scan_speed: float
    target_width_pixels: float


@dataclass(frozen=True)
class ObstacleConfig:
    stop_distance: float
    slow_distance: float
    avoid_distance: float
    turn_angle_deg: float


@dataclass(frozen=True)
class CameraConfig:
    calibration_file: str | None
    marker_size_m: float
    aruco_dictionary: str
    detection_rate_hz: float


@dataclass(frozen=True)
class EvaluationConfig:
    trial_count: int
    timeout_s: float
    encoder_noise_std: float
    odometry_scale_error: float
    random_seed: int


@dataclass(frozen=True)
class AppConfig:
    robot: RobotConfig
    ekf: EkfConfig
    lqr: LqrConfig
    tag_follower: TagFollowerConfig
    obstacle_avoidance: ObstacleConfig
    camera: CameraConfig
    evaluation: EvaluationConfig


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"configuration section '{name}' must be a mapping")
    return value


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"'{name}' must be a positive number")
    return float(value)


def _optional_positive(section: dict[str, Any], name: str) -> float | None:
    value = section.get(name)
    return None if value is None else _positive(value, name)


def _vector(section: dict[str, Any], name: str, keys: tuple[str, ...]) -> tuple[float, ...]:
    value = section.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be a mapping")
    return tuple(_positive(value.get(key), f"{name}.{key}") for key in keys)


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"'{name}' must be an integer >= {minimum}")
    return value


def load_config(path: str | Path) -> AppConfig:
    """Load and validate one application configuration file."""
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")

    robot = _section(raw, "robot")
    ekf = _section(raw, "ekf")
    lqr = _section(raw, "lqr")
    tags = _section(raw, "tag_follower")
    obstacle = _section(raw, "obstacle_avoidance")
    camera = _section(raw, "camera")
    evaluation = _section(raw, "evaluation")

    target_ids = tags.get("target_ids")
    if (
        not isinstance(target_ids, list)
        or not target_ids
        or any(isinstance(value, bool) or not isinstance(value, int) for value in target_ids)
    ):
        raise ValueError("'target_ids' must be a non-empty list of integers")

    gyro_enabled = ekf.get("gyro_enabled")
    if not isinstance(gyro_enabled, bool):
        raise ValueError("'gyro_enabled' must be a boolean")

    calibration_file = camera.get("calibration_file")
    if calibration_file is not None and not isinstance(calibration_file, str):
        raise ValueError("'calibration_file' must be null or a path string")

    return AppConfig(
        robot=RobotConfig(
            wheel_base=_optional_positive(robot, "wheel_base"),
            wheel_radius_left=_optional_positive(robot, "wheel_radius_left"),
            wheel_radius_right=_optional_positive(robot, "wheel_radius_right"),
            max_wheel_speed=_optional_positive(robot, "max_wheel_speed"),
        ),
        ekf=EkfConfig(
            wheel_distance_std=_positive(ekf.get("wheel_distance_std"), "wheel_distance_std"),
            initial_covariance=_vector(ekf, "initial_covariance", ("x", "y", "yaw")),
            camera_measurement_covariance=_vector(
                ekf, "camera_measurement_covariance", ("x", "y", "yaw")
            ),
            gyro_enabled=gyro_enabled,
            gyro_rate_std=_positive(ekf.get("gyro_rate_std"), "gyro_rate_std"),
        ),
        lqr=LqrConfig(
            q=_vector(lqr, "q", ("x", "y", "yaw")),
            r=_vector(lqr, "r", ("linear_velocity", "angular_velocity")),
            max_linear_velocity=_positive(lqr.get("max_linear_velocity"), "max_linear_velocity"),
            max_angular_velocity=_positive(lqr.get("max_angular_velocity"), "max_angular_velocity"),
        ),
        tag_follower=TagFollowerConfig(
            target_ids=tuple(target_ids),
            kp_linear=_positive(tags.get("kp_linear"), "kp_linear"),
            kp_angular=_positive(tags.get("kp_angular"), "kp_angular"),
            scan_speed=_positive(tags.get("scan_speed"), "scan_speed"),
            target_width_pixels=_positive(tags.get("target_width_pixels"), "target_width_pixels"),
        ),
        obstacle_avoidance=ObstacleConfig(
            stop_distance=_positive(obstacle.get("stop_distance"), "stop_distance"),
            slow_distance=_positive(obstacle.get("slow_distance"), "slow_distance"),
            avoid_distance=_positive(obstacle.get("avoid_distance"), "avoid_distance"),
            turn_angle_deg=_positive(obstacle.get("turn_angle_deg"), "turn_angle_deg"),
        ),
        camera=CameraConfig(
            calibration_file=calibration_file,
            marker_size_m=_positive(camera.get("marker_size_m"), "marker_size_m"),
            aruco_dictionary=camera.get("aruco_dictionary"),
            detection_rate_hz=_positive(camera.get("detection_rate_hz"), "detection_rate_hz"),
        ),
        evaluation=EvaluationConfig(
            trial_count=_integer(evaluation.get("trial_count"), "trial_count", 1),
            timeout_s=_positive(evaluation.get("timeout_s"), "timeout_s"),
            encoder_noise_std=_positive(evaluation.get("encoder_noise_std"), "encoder_noise_std")
            if evaluation.get("encoder_noise_std", 0.0) > 0
            else 0.0,
            odometry_scale_error=_positive(evaluation.get("odometry_scale_error"), "odometry_scale_error")
            if evaluation.get("odometry_scale_error", 0.0) > 0
            else 0.0,
            random_seed=_integer(evaluation.get("random_seed"), "random_seed"),
        ),
    )