"""Physical parameters extracted from the loaded MuJoCo robot model."""

from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class DifferentialDriveParameters:
    wheel_base: float
    wheel_radius: float


def parameters_from_model(
    model: mujoco.MjModel,
    left_wheel_body: str = "wheel_left",
    right_wheel_body: str = "wheel_right",
) -> DifferentialDriveParameters:
    """Read differential-drive geometry from wheel body positions in the XML."""
    left_id = model.body(left_wheel_body).id
    right_id = model.body(right_wheel_body).id
    left_pos = model.body_pos[left_id]
    right_pos = model.body_pos[right_id]

    wheel_base = abs(float(left_pos[1] - right_pos[1]))
    wheel_radius = 0.5 * (abs(float(left_pos[2])) + abs(float(right_pos[2])))

    if wheel_base <= 0.0:
        raise ValueError("wheel positions in the model have zero separation")
    if wheel_radius <= 0.0:
        raise ValueError("wheel positions in the model have zero radius")

    return DifferentialDriveParameters(wheel_base, wheel_radius)

