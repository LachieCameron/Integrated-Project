import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

# Find where the ghost's freejoint coordinates sit inside the global qpos array
ghost_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ghost_joint")
if ghost_jnt_id == -1:
    raise RuntimeError("Could not find 'ghost_joint' in the XML file.")
g_qpos_adr = model.jnt_qposadr[ghost_jnt_id]

# Initialize software stacks
imu = IMUSensor(model)
encoder = WheelEncoder(model)
ekf = RobotEKF()

# State telemetry printing control
last_print_time = 0.0

def simulation_callback(model, data):
    """
    Native MuJoCo control hook. This function is triggered by MuJoCo's 
    core engine right before computing solver states.
    """
    global last_print_time
    dt = model.opt.timestep

    # 1. Apply Actuator Controls
    data.ctrl[0] = -3.0  # Left motor velocity
    data.ctrl[1] = 3.0   # Right motor velocity

    # 2. Read Sensors (Executed natively at every single step)
    d_left, d_right, v_left, v_right = encoder.step(data)
    accel, gyro = imu.read(data)

    # 3. State Estimation (Passing index 2 for the Z-axis yaw rate)
    current_pose = ekf.update(d_left, d_right, gyro[2], dt)
    est_x, est_y, est_theta = current_pose[0], current_pose[1], current_pose[2]

    # 4. Position the Native Visual Ghost
    data.qpos[g_qpos_adr] = est_x
    data.qpos[g_qpos_adr + 1] = est_y
    data.qpos[g_qpos_adr + 2] = 0.01  
    
    data.qpos[g_qpos_adr + 3] = np.cos(est_theta / 2.0)  # Quaternion w
    data.qpos[g_qpos_adr + 4] = 0.0                      # Quaternion x
    data.qpos[g_qpos_adr + 5] = 0.0                      # Quaternion y
    data.qpos[g_qpos_adr + 6] = np.sin(est_theta / 2.0)  # Quaternion z

    # 5. Non-blocking Terminal Logs (Every 0.5 seconds)
    if data.time - last_print_time >= 0.5:
        last_print_time = data.time
        print(f"Time: {data.time:.2f}s | EKF Pose: X={est_x:.3f}, Y={est_y:.3f}, Yaw={math.degrees(est_theta):.1f}°")

# --- REGISTER CALLBACK HOOK AND BOOT ---
print("Registering control pipeline with MuJoCo solver hooks...")
# Connects the custom python state computation to MuJoCo's internal stepping chain
mujoco.set_mjcb_control(simulation_callback)

print("Launching active blocking viewer...")
# Bypasses the passive thread limits entirely, loading via standard native bindings
mujoco.viewer.launch(model, data)

# Clean up hook when window closes to avoid lingering thread processes
mujoco.set_mjcb_control(None)
