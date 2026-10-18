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
data_ghost = mujoco.MjData(model)  # Ghost data for visualization

vopt_ghost = mujoco.MjvOption()
vopt_ghost.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

pert_empty = mujoco.MjvPerturb()
catmask_dynamic = mujoco.mjtCatBit.mjCAT_DYNAMIC

imu = IMUSensor(model)
encoder = WheelEncoder(model)
ekf = RobotEKF()

dt = model.opt.timestep

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Simulation execution loop spinning...")
    while viewer.is_running():
        start_time = time.time()

        # 1. Controller: command wheel velocities directly (rad/s)
        data.ctrl[0] = -3.0  # Left motor
        data.ctrl[1] = 3.0  # Right motor

        # 2. Physics Step
        mujoco.mj_step(model, data)

        # 3. Read Hardware Abstraction Layer (Sensing)
        d_left, d_right, v_left, v_right = encoder.step(data)
        accel, gyro = imu.read(data)

        # 4. State Estimation
        current_pose = ekf.update(d_left, d_right, gyro[2], dt)
        est_x, est_y, est_theta = current_pose[0], current_pose[1], current_pose[2]


        user_scn = viewer.user_scn
        if user_scn is not None:
            # Clear previous frame's ghost data so markers don't stack infinitely
            user_scn.ngeom = 0  
            
            # Grab an empty geometry slot out of the context buffer
            geom_idx = user_scn.ngeom
            user_scn.ngeom += 1
            
            # Convert EKF heading (theta around Z) into a 3x3 rotation matrix for MuJoCo
            cos_t = np.cos(est_theta)
            sin_t = np.sin(est_theta)
            # Z-axis rotation matrix flattened in row-major layout
            r_mat = np.array([
                cos_t, -sin_t, 0.0,
                sin_t,  cos_t, 0.0,
                0.0,    0.0,   1.0
            ], dtype=np.float32)

            # Initialize a box geometry representing the Waffle Pi footprint
            mujoco.mjv_initGeom(
                user_scn.geoms[geom_idx],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.14, 0.14, 0.05],                       # Match base dimensions [x, y, z]
                pos=[est_x, est_y, 0.05],                      # EKF positions (raised slightly off floor)
                mat=r_mat,                                     # Calculated rotation matrix orientation
                rgba=[0.0, 1.0, 0.0, 0.4]                      # Semi-transparent green (R, G, B, Alpha)
            )

        # Print state telemetry cleanly
        if data.time % 0.1 < dt:
            print(f"Time: {data.time:.2f}s | Estimated Pose: X={current_pose[0]:.3f}, Y={current_pose[1]:.3f}, Heading={math.degrees(current_pose[2]):.1f}°")

        viewer.sync()

        # Enforce deterministic timing match
        elapsed = time.time() - start_time
        if elapsed < dt:
            time.sleep(dt - elapsed)