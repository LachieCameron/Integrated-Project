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
last_render_time = 0.0
render_dt = 1.0 / 90.0  # Target target visual updates at 60 Hz

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

        if data.time - last_render_time >= render_dt:
            last_render_time = data.time

            est_x, est_y, est_theta = current_pose[0], current_pose[1], current_pose[2]

            with viewer.lock():
                # 5. Synchronize your Estimate to the Ghost Container
                # Copy the physical joint states (like wheel spinning positions) so the shapes match
                data_ghost.qpos[:] = data.qpos[:]
                
                # Override the root 2D pose with your EKF's estimated values
                # In a 3D "freejoint", qpos[0:2] = X,Y and qpos[3:7] = Quaternion orientation
                data_ghost.qpos[0] = est_x
                data_ghost.qpos[1] = est_y
                
                # Convert your 1D heading angle (est_theta) to a 3D Z-axis quaternion [w, x, y, z]
                data_ghost.qpos[3] = np.cos(est_theta / 2.0)  # w
                data_ghost.qpos[4] = 0.0                      # x
                data_ghost.qpos[5] = 0.0                      # y
                data_ghost.qpos[6] = np.sin(est_theta / 2.0)  # z

                # Force MuJoCo to calculate the positions of all internal child meshes/links
                mujoco.mj_forward(model, data_ghost)

                # 6. Draw the True Skeleton Ghost to the Screen
                user_scn = viewer.user_scn
                if user_scn is not None:
                    user_scn.ngeom = 0  # Clear previous frame's ghost shapes
                    
                    # Extract all geoms from data_ghost and push them into the overlay buffer
                    mujoco.mjv_addGeoms(
                        model, 
                        data_ghost, 
                        vopt_ghost, 
                        pert_empty, 
                        catmask_dynamic, 
                        user_scn
                    )


        # Print state telemetry cleanly
        if data.time % 0.5 < dt:
            print(f"Time: {data.time:.2f}s | Estimated Pose: X={current_pose[0]:.3f}, Y={current_pose[1]:.3f}, Heading={math.degrees(current_pose[2]):.1f}°")

        viewer.sync()

        # Enforce deterministic timing match
        elapsed = time.time() - start_time
        if elapsed < dt:
            time.sleep(dt - elapsed)