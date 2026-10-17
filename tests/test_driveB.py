import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# 1. Dynamically get the absolute path to your project root folder
current_dir = os.path.dirname(os.path.abspath(__file__))  
project_root = os.path.abspath(os.path.join(current_dir, "..")) 

import math
import numpy as np
import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer
import time

from src.sensing.IMU import IMUSensor
from src.sensing.wheel_encoder import WheelEncoder
from src.estimation.ekf import RobotEKF

path = os.path.join(project_root, "assets", "scene_turtlebot3_waffle_pi.xml")
model = mujoco.MjModel.from_xml_path(path)
data = mujoco.MjData(model)

imu = IMUSensor(model)
encoder = WheelEncoder(model)
ekf = RobotEKF()

dt = model.opt.timestep

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Simulation execution loop spinning...")
    while viewer.is_running():
        start_time = time.time()

        # 1. Controller: command wheel velocities directly (rad/s)
        data.ctrl[0] = 3.0  # Left motor
        data.ctrl[1] = 3.0  # Right motor

        # 2. Physics Step
        mujoco.mj_step(model, data)

        # 3. Read Hardware Abstraction Layer (Sensing)
        d_left, d_right, v_left, v_right = encoder.step(data)
        accel, gyro = imu.read(data)

        # 4. State Estimation
        current_pose = ekf.update(d_left, d_right, gyro[2], dt)

        # Print state telemetry cleanly
        if data.time % 0.5 < dt:
            print(f"Time: {data.time:.2f}s | Estimated Pose: X={current_pose[0]:.3f}, Y={current_pose[1]:.3f}, Heading={math.degrees(current_pose[2]):.1f}°")

        viewer.sync()

        # Enforce deterministic timing match
        elapsed = time.time() - start_time
        if elapsed < dt:
            time.sleep(dt - elapsed)