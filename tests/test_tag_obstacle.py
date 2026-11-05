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
from src.utils.config import load_config

ANGLES = list(range(-90, 91, 15))


def angle_difference(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def main():
    xml_path = os.path.join(project_root, 'assets', 'scene_turtlebot3_waffle_pi.xml')
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f'Missing XML: {xml_path}')

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    config = load_config(os.path.join(project_root, "config", "default.yaml"))
    tag_config = config.tag_follower
    obstacle_config = config.obstacle_avoidance
    cam_interval = 1.0 / config.camera.detection_rate_hz
    stop_distance = obstacle_config.stop_distance
    clear_distance = obstacle_config.slow_distance
    avoid_distance = obstacle_config.avoid_distance
    avoid_turn_rad = math.radians(obstacle_config.turn_angle_deg)
    avoid_forward_v = config.lqr.max_linear_velocity
    avoid_turn_w = config.lqr.max_angular_velocity
    max_wheel_speed = (
        config.robot.max_wheel_speed
        if config.robot.max_wheel_speed is not None
        else float(abs(model.actuator_ctrlrange[0, 1]))
    )
    camera = RaspiCamera(model)
    imu = IMUSensor(model)
    diff_drive = DiffDriveController()
    follower = TagFollower(
        target_ids=tag_config.target_ids,
        image_width=camera.width,
        Kp_linear=tag_config.kp_linear,
        Kp_angular=tag_config.kp_angular,
        scan_speed=tag_config.scan_speed,
        target_width_pixels=tag_config.target_width_pixels,
        max_linear_v=config.lqr.max_linear_velocity,
        max_angular_w=config.lqr.max_angular_velocity,
    )

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
    last_cam_time = -cam_interval
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

        # Begin the avoidance manoeuvre before the emergency stop distance.
        # The previous trigger used STOP_DISTANCE, so CLEAR_DISTANCE only
        # capped speed to the tag follower's existing 0.15 m/s maximum.
        if mode == 'FOLLOW' and desired_v > 0 and front < clear_distance:
            mode = 'AVOID_TURN'
            turn_start_heading = heading
            turn_direction = 1 if left >= right else -1

        if mode == 'AVOID_TURN':
            # IMU integration is wrapped; signed angle error handles wraparound.
            turned = turn_direction * angle_difference(heading, turn_start_heading)
            if turned >= avoid_turn_rad:
                mode = 'AVOID_FORWARD'
                forward_start = position
                desired_v, desired_w = 0.0, 0.0
            else:
                desired_v, desired_w = 0.0, avoid_turn_w * turn_direction

        elif mode == 'AVOID_FORWARD':
            travelled = float(np.linalg.norm(position - forward_start))
            if front < stop_distance:
                mode = 'AVOID_TURN'
                turn_start_heading = heading
                turn_direction = 1 if left >= right else -1
                desired_v, desired_w = 0.0, 0.0
            elif travelled >= avoid_distance:
                mode = 'FOLLOW'
                desired_v, desired_w = 0.0, 0.0
            else:
                desired_v, desired_w = avoid_forward_v, 0.0

        # Keep the original tag follower running when not avoiding obstacles.
        # Camera processing is paused during avoidance to avoid advancing its scan state.
        if data.time - last_cam_time >= cam_interval:
            last_cam_time = data.time
            frame = camera.capture_cv_frame(data)
            corners, ids, annotated = camera.detect_markers(frame)
            if mode == 'FOLLOW':
                desired_v, desired_w = follower.compute_commands(corners, ids, heading)
                # Immediate LiDAR override, including after new camera commands.
                if desired_v > 0 and front < clear_distance:
                    mode = 'AVOID_TURN'
                    turn_start_heading = heading
                    turn_direction = 1 if left >= right else -1
                    desired_v, desired_w = 0.0, 0.0

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
        data.ctrl[0] = float(np.clip(left_wheel, -max_wheel_speed, max_wheel_speed))
        data.ctrl[1] = float(np.clip(right_wheel, -max_wheel_speed, max_wheel_speed))

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
