#!/usr/bin/env python3
"""MuJoCo TurtleBot3: ArUco tag following with LiDAR obstacle avoidance."""
import os
import sys
import math
import numpy as np
import cv2
import mujoco
import mujoco.viewer

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.sensing.cam import RaspiCamera
from src.sensing.IMU import IMUSensor
from src.control.Tag_follow import DiffDriveController, TagFollower

CAM_INTERVAL = 1.0 / 30.0
STOP_DISTANCE = 0.2
CLEAR_DISTANCE = 0.4
AVOID_DISTANCE = 0.5
AVOID_TURN_RAD = math.radians(35)  # Shorter turn than previous 90 degrees
AVOID_FORWARD_V = 1.0  # m/s
AVOID_TURN_W = 0.45      # rad/s
MAX_WHEEL_SPEED = 5.0  # rad/s; verify against XML actuator limits
ANGLES = list(range(-90, 91, 15))


def angle_difference(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def main():
    xml_path = os.path.join(project_root, 'assets', 'scene_turtlebot3_waffle_pi.xml')
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f'Missing XML: {xml_path}')

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    camera = RaspiCamera(model)
    imu = IMUSensor(model)
    diff_drive = DiffDriveController()
    follower = TagFollower(target_id=0, image_width=camera.width)

    # MuJoCo rangefinders must already be defined in the XML.
    lidar_ids = []
    for angle in ANGLES:
        name = f'lidar_{angle}'
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise RuntimeError(f'Missing LiDAR sensor {name} in the XML. Add it before running.')
        if model.sensor_dim[sid] != 1:
            raise RuntimeError(f'Expected one scalar reading from {name}')
        lidar_ids.append(sid)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base')
    if base_id < 0:
        raise RuntimeError('Missing body named "base" in XML')
    if model.nu < 2:
        raise RuntimeError('Expected at least two wheel actuators')

    heading = 0.0
    last_cam_time = -CAM_INTERVAL
    desired_v, desired_w = 0.0, 0.0
    mode = 'FOLLOW'
    turn_start_heading = 0.0
    turn_direction = 1
    forward_start = np.zeros(2)
    last_message_mode = None

    def lidar_distances(data):
        values = []
        for sid in lidar_ids:
            value = float(data.sensordata[model.sensor_adr[sid]])
            # MuJoCo returns -1 for no hit. Treat as clear up to sensor range.
            values.append(value if np.isfinite(value) and value >= 0 else math.inf)
        return np.asarray(values)

    def sector_min(distances, low, high):
        return float(np.min(distances[(np.asarray(ANGLES) >= low) &
                                      (np.asarray(ANGLES) <= high)]))

    def control_loop(model, data):
        nonlocal heading, last_cam_time, desired_v, desired_w
        nonlocal mode, turn_start_heading, turn_direction, forward_start
        nonlocal last_message_mode

        _, gyro = imu.read(data)
        if abs(float(gyro[2])) > 0.001:
            heading += float(gyro[2]) * model.opt.timestep
        heading = angle_difference(heading, 0.0)

        distances = lidar_distances(data)
        front = sector_min(distances, -30, 30)
        left = sector_min(distances, 45, 90)
        right = sector_min(distances, -90, -45)
        position = np.asarray(data.xpos[base_id, :2]).copy()

        if mode == 'FOLLOW' and desired_v > 0 and front < STOP_DISTANCE:
            mode = 'AVOID_TURN'
            turn_start_heading = heading
            turn_direction = 1 if left >= right else -1

        if mode == 'AVOID_TURN':
            # IMU integration is wrapped; signed angle error handles wraparound.
            turned = turn_direction * angle_difference(heading, turn_start_heading)
            if turned >= AVOID_TURN_RAD:
                mode = 'AVOID_FORWARD'
                forward_start = position
                desired_v, desired_w = 0.0, 0.0
            else:
                desired_v, desired_w = 0.0, AVOID_TURN_W * turn_direction

        elif mode == 'AVOID_FORWARD':
            travelled = float(np.linalg.norm(position - forward_start))
            if front < STOP_DISTANCE:
                mode = 'AVOID_TURN'
                turn_start_heading = heading
                turn_direction = 1 if left >= right else -1
                desired_v, desired_w = 0.0, 0.0
            elif travelled >= AVOID_DISTANCE:
                mode = 'FOLLOW'
                desired_v, desired_w = 0.0, 0.0
            else:
                desired_v, desired_w = AVOID_FORWARD_V, 0.0

        # Keep the original tag follower running when not avoiding obstacles.
        # Camera processing is paused during avoidance to avoid advancing its scan state.
        if data.time - last_cam_time >= CAM_INTERVAL:
            last_cam_time = data.time
            frame = camera.capture_cv_frame(data)
            corners, ids, annotated = camera.detect_markers(frame)
            if mode == 'FOLLOW':
                desired_v, desired_w = follower.compute_commands(corners, ids, heading)
                # Immediate LiDAR override, including after new camera commands.
                if desired_v > 0 and front < STOP_DISTANCE:
                    mode = 'AVOID_TURN'
                    turn_start_heading = heading
                    turn_direction = 1 if left >= right else -1
                    desired_v, desired_w = 0.0, 0.0
                elif desired_v > 0 and front < CLEAR_DISTANCE:
                    desired_v = min(desired_v, AVOID_FORWARD_V)

            cv2.putText(annotated, f'Mode: {mode} / {follower.state}', (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(annotated, f'Yaw: {math.degrees(heading):.1f} deg', (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(annotated, f'Front: {front:.2f} m', (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.imshow('TurtleBot3 - Tag + LiDAR', annotated)
            cv2.waitKey(1)

        if mode != last_message_mode:
            print(f'[{data.time:.2f}s] {mode} | front={front:.2f}m')
            last_message_mode = mode

        left_wheel, right_wheel = diff_drive.twist_to_wheel_velocities(desired_v, desired_w)
        data.ctrl[0] = float(np.clip(left_wheel, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
        data.ctrl[1] = float(np.clip(right_wheel, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))

    try:
        mujoco.set_mjcb_control(control_loop)
        mujoco.viewer.launch(model, data)
    except KeyboardInterrupt:
        print('Simulation interrupted.')
    finally:
        mujoco.set_mjcb_control(None)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
