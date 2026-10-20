from pathlib import Path

import mujoco
import numpy as np

from src.kinematics.robot_parameters import parameters_from_model
from src.estimation.ekf import RobotEKF
from src.sensing.wheel_encoder import WheelEncoder


def test_parameters_are_read_from_turtlebot_xml():
    xml_path = Path(__file__).parents[1] / "assets" / "turtlebot3_waffle_pi.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    parameters = parameters_from_model(model)

    np.testing.assert_allclose(parameters.wheel_base, 0.288)
    np.testing.assert_allclose(parameters.wheel_radius, 0.033)


def test_runtime_components_use_model_geometry():
    xml_path = Path(__file__).parents[1] / "assets" / "turtlebot3_waffle_pi.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    ekf = RobotEKF(model=model)
    encoder = WheelEncoder(model)

    assert ekf.L == 0.288
    assert encoder.radius == 0.033

