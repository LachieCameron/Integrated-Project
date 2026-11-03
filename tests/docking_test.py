#!/usr/bin/env python3
"""MuJoCo TurtleBot3: original TagFollower + LiDAR, with ID 0 -> ID 99 handoff.

Place this file at tests/test_tag_obstacle.py.
Run from project root: PYTHONPATH=. python3 tests/test_tag_obstacle.py
"""
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
STOP_DISTANCE = 0.20
CLEAR_DISTANCE = 0.40
AVOID_DISTANCE = 0.50
AVOID_TURN_RAD = math.radians(35)
AVOID_FORWARD_V = 1.0  # Original request; wheel commands are limited below.
AVOID_TURN_W = 0.45
MAX_WHEEL_SPEED = 5.0
ANGLES = np.arange(-90, 91, 15)

COARSE_ID = 0
FINE_ID = 99
FINE_SWITCH_WIDTH = 20.0       # Pixels: tune from the on-screen ID99 width.
FINE_CONFIRM_FRAMES = 2        # Consecutive camera frames.
FINE_STOP_WIDTH = 125.0        # Pixels: tune for desired final docking distance.
DEBUG_PRINT_INTERVAL = 1.0     # Seconds between detection debug messages.


def angle_difference(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def marker_width(corners, ids, target_id):
    """Return average horizontal-edge length of a decoded ArUco marker in pixels."""
    if ids is None or corners is None:
        return 0.0
    for index, marker_id in enumerate(np.asarray(ids).reshape(-1)):
        if int(marker_id) == target_id:
            points = np.asarray(corners[index], dtype=float).reshape(4, 2)
            top = np.linalg.norm(points[1] - points[0])
            bottom = np.linalg.norm(points[2] - points[3])
            return float((top + bottom) / 2.0)
    return 0.0


def main():
    xml_path = os.path.join(project_root, 'assets', 'scene_turtlebot3_waffle_pi.xml')
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f'Missing XML: {xml_path}')

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    camera = RaspiCamera(model)
    imu = IMUSensor(model)
    diff_drive = DiffDriveController()
    follower = TagFollower(target_id=COARSE_ID, image_width=camera.width)

    lidar_ids = []
    for angle in ANGLES:
        name = f'lidar_{int(angle)}'
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0 or model.sensor_dim[sid] != 1:
            raise RuntimeError(f'Missing or invalid LiDAR sensor: {name}')
        lidar_ids.append(sid)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base')
    if base_id < 0:
        raise RuntimeError('Missing body named "base" in XML')
    left_actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'wheel_left')
    right_actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'wheel_right')
    if left_actuator < 0 or right_actuator < 0:
        raise RuntimeError('Missing wheel_left or wheel_right actuator')

    heading = 0.0
    last_sim_time = float(data.time)
    last_cam_time = -CAM_INTERVAL
    desired_v, desired_w = 0.0, 0.0
    mode = 'FOLLOW'
    turn_start_heading = 0.0
    turn_direction = 1
    forward_start = np.zeros(2)
    last_message_mode = None
    fine_detected_frames = 0
    fine_locked = False
    last_debug_time = -DEBUG_PRINT_INTERVAL

    def lidar_distances(data):
        values = []
        for sid in lidar_ids:
            value = float(data.sensordata[model.sensor_adr[sid]])
            values.append(value if np.isfinite(value) and value >= 0 else math.inf)
        return np.asarray(values)

    def sector_min(distances, low, high):
        return float(np.min(distances[(ANGLES >= low) & (ANGLES <= high)]))

    def control_loop(model, data):
        nonlocal heading, last_sim_time, last_cam_time, desired_v, desired_w
        nonlocal mode, turn_start_heading, turn_direction, forward_start
        nonlocal last_message_mode, fine_detected_frames, fine_locked, last_debug_time

        now = float(data.time)
        if now < last_sim_time - 1e-9:
            # Viewer reset: reconstruct the follower so the original scan runs again.
            nonlocal follower
            follower = TagFollower(target_id=COARSE_ID, image_width=camera.width)
            heading = 0.0
            last_cam_time = now - CAM_INTERVAL
            desired_v = desired_w = 0.0
            mode = 'FOLLOW'
            fine_detected_frames = 0
            fine_locked = False
            last_message_mode = None
            last_debug_time = now - DEBUG_PRINT_INTERVAL
        dt = max(0.0, now - last_sim_time)
        last_sim_time = now

        _, gyro = imu.read(data)
        if abs(float(gyro[2])) > 0.001:
            heading += float(gyro[2]) * dt
        heading = angle_difference(heading, 0.0)

        distances = lidar_distances(data)
        front = sector_min(distances, -30, 30)
        left = sector_min(distances, 45, 90)
        right = sector_min(distances, -90, -45)
        # Retained from original test; ground truth must be replaced for final rubric.
        position = np.asarray(data.xpos[base_id, :2]).copy()

        if mode == 'FOLLOW' and desired_v > 0 and front < STOP_DISTANCE:
            mode = 'AVOID_TURN'
            turn_start_heading = heading
            turn_direction = 1 if left >= right else -1

        if mode == 'AVOID_TURN':
            turned = turn_direction * angle_difference(heading, turn_start_heading)
            if turned >= AVOID_TURN_RAD:
                mode = 'AVOID_FORWARD'
                forward_start = position.copy()
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

        if now - last_cam_time >= CAM_INTERVAL:
            last_cam_time = now
            frame = camera.capture_cv_frame(data)
            corners, ids, annotated = camera.detect_markers(frame)
            detected_ids = ([] if ids is None else
                            [int(value) for value in np.asarray(ids).reshape(-1)])
            small_width = marker_width(corners, ids, FINE_ID)

            if not fine_locked:
                if small_width >= FINE_SWITCH_WIDTH:
                    fine_detected_frames += 1
                else:
                    fine_detected_frames = 0

                if fine_detected_frames >= FINE_CONFIRM_FRAMES:
                    fine_locked = True
                    follower.target_id = FINE_ID
                    # Original TagFollower's scan is already complete in normal use;
                    # do not replace the object or restart a full scan.
                    follower.state = 'TRACKING'
                    follower.target_width_px = FINE_STOP_WIDTH
                    print(f'[{now:.2f}s] TARGET SWITCH: ID {COARSE_ID} -> '
                          f'ID {FINE_ID} (width={small_width:.1f}px)')

            if mode == 'FOLLOW':
                desired_v, desired_w = follower.compute_commands(corners, ids, heading)
                if desired_v > 0 and front < STOP_DISTANCE:
                    mode = 'AVOID_TURN'
                    turn_start_heading = heading
                    turn_direction = 1 if left >= right else -1
                    desired_v, desired_w = 0.0, 0.0
                elif desired_v > 0 and front < CLEAR_DISTANCE:
                    desired_v = min(desired_v, AVOID_FORWARD_V)

            cv2.putText(annotated, f'Mode: {mode} / {follower.state}', (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 2)
            cv2.putText(annotated, f'Yaw: {math.degrees(heading):.1f} deg', (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
            cv2.putText(annotated, f'Front: {front:.2f} m', (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
            cv2.putText(annotated, f'Target ID: {follower.target_id}', (15, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 2)
            cv2.putText(annotated, f'Detected: {detected_ids}', (15, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 0), 1)
            cv2.putText(annotated,
                        f'ID99 width: {small_width:.1f}px  confirm: {fine_detected_frames}/{FINE_CONFIRM_FRAMES}',
                        (15, 155), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 0), 1)
            cv2.imshow('TurtleBot3 - Tag + LiDAR', annotated)
            cv2.waitKey(1)

            if now - last_debug_time >= DEBUG_PRINT_INTERVAL:
                print(f'[{now:.2f}s] IDs={detected_ids} ID99_width={small_width:.1f}px '
                      f'target={follower.target_id} mode={mode} state={follower.state}')
                last_debug_time = now

        if mode != last_message_mode:
            print(f'[{now:.2f}s] {mode} | front={front:.2f}m')
            last_message_mode = mode

        left_wheel, right_wheel = diff_drive.twist_to_wheel_velocities(desired_v, desired_w)
        data.ctrl[left_actuator] = float(np.clip(left_wheel, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
        data.ctrl[right_actuator] = float(np.clip(right_wheel, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))

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
