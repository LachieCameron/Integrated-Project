#!/usr/bin/env python3
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
from src.sensing.wheel_encoder import WheelEncoder
from src.estimation.ekf import RobotEKF
from src.control.Tag_follow import DiffDriveController, TagFollower

CAM_INTERVAL = 1.0 / 30.0
STOP_DISTANCE = 0.2
CLEAR_DISTANCE = 0.4
AVOID_DISTANCE = 0.5
AVOID_TURN_RAD = math.radians(35)
AVOID_FORWARD_V = 0.15
AVOID_TURN_W = 0.45
MAX_WHEEL_SPEED = 5.0
ANGLES = np.asarray(list(range(-90, 91, 15)))

# The robot's initial pose in the map must be KNOWN, not read from MuJoCo
# ground truth during control. Change this if your robot starts elsewhere.
INITIAL_POSE_ESTIMATE = np.array([0.0, 0.0, 0.0], dtype=float)
INITIAL_COVARIANCE = np.diag([0.15**2, 0.15**2, math.radians(15)**2])
STATUS_INTERVAL = 1.0
DOCK_TAG_ID = 99
DOCK_COMPLETE_DISTANCE = 0.18
ID99_SWITCH_MIN_WIDTH_PX = 23.0  # minimum reliable visible size; tune from camera logs
ID99_CONFIRM_FRAMES = 5
ID99_LOST_FRAMES = 5
ID99_REACQUIRE_FRAMES = 5
ID99_SEARCH_W = 0.16
ID99_CENTER_BEFORE_DRIVE = 0.10
DOCK_HARD_STOP = 0.12
# The dock can be inside CLEAR_DISTANCE; keep an emergency stop.
COARSE_STOP_WIDTH_PX = 250.0
# Keep the ORIGINAL TagFollower stop threshold (250 px), not 125 px.
DOCK_WIDTH_PX = 250.0
# Require the lower marker to be reasonably centred before latching CHARGED.
# This is a camera-only proxy, not proof of physical electrical charging.
DOCK_CENTER_TOLERANCE = 0.12  # fraction of image half-width
DOCK_CONFIRM_FRAMES = 3


def angle_difference(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def main():
    xml_path = os.path.join(project_root, 'assets', 'scene_turtlebot3_waffle_pi.xml')
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f'Missing XML: {xml_path}')

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    camera = RaspiCamera(model)
    diff_drive = DiffDriveController()
    # Follow only ID 0 at startup. After switching, follow only ID 99.
    follower = TagFollower(target_ids=(0,), image_width=camera.width,
                           target_width_pixels=250.0)
    encoder = WheelEncoder(model)
    ekf = RobotEKF(
        model=model,
        initial_state=INITIAL_POSE_ESTIMATE.copy(),
        initial_covariance=INITIAL_COVARIANCE.copy(),
    )
    lidar_ids = []
    for angle in ANGLES:
        name = f'lidar_{angle}'
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise RuntimeError(f'Missing LiDAR sensor {name} in the XML')
        if model.sensor_dim[sid] != 1:
            raise RuntimeError(f'Expected one scalar reading from {name}')
        lidar_ids.append(sid)
    if model.nu < 2:
        raise RuntimeError('Expected at least two wheel actuators')

    last_cam_time = -CAM_INTERVAL
    last_status_time = -STATUS_INTERVAL
    desired_v, desired_w = 0.0, 0.0
    mode = 'FOLLOW'
    turn_start_heading = 0.0
    turn_direction = 1
    forward_start = np.zeros(2)
    last_message_mode = None
    charged = False
    docking_phase = False
    dock_confirm_count = 0
    last_dock_debug = -1.0
    final_approach = False
    id99_seen_frames = 0
    last_switch_debug = -1.0
    id99_lost_frames = 0
    id99_reacquire_frames = 0
    last_id99_signed_error = 0.0
    last_id99_seen_time = -1.0
    coarse_ready = False

    def lidar_distances():
        values = []
        for sid in lidar_ids:
            value = float(data.sensordata[model.sensor_adr[sid]])
            values.append(value if np.isfinite(value) and value >= 0 else math.inf)
        return np.asarray(values)

    def sector_min(distances, low, high):
        return float(np.min(distances[(ANGLES >= low) & (ANGLES <= high)]))

    def control_loop(model, data):
        nonlocal last_cam_time, last_status_time, desired_v, desired_w
        nonlocal mode, turn_start_heading, turn_direction, forward_start
        nonlocal last_message_mode, charged, docking_phase, dock_confirm_count, last_dock_debug, final_approach
        nonlocal id99_seen_frames, last_switch_debug
        nonlocal id99_lost_frames, id99_reacquire_frames, last_id99_signed_error
        nonlocal last_id99_seen_time, coarse_ready

        # EKF prediction from measured wheel motion. No data.xpos or data.qpos.
        d_left, d_right, _, _ = encoder.step(data)
        pose_est = ekf.predict(d_left, d_right)
        position = pose_est[:2].copy()
        heading = float(pose_est[2])

        distances = lidar_distances()
        front = sector_min(distances, -30, 30)
        left = sector_min(distances, 45, 90)
        right = sector_min(distances, -90, -45)

        # Once docked, never restart navigation or obstacle avoidance.
        if charged:
            data.ctrl[0] = 0.0
            data.ctrl[1] = 0.0
            return

        # Preserve original reactive obstacle avoidance.
        if mode == 'FOLLOW' and not docking_phase and desired_v > 0 and front < CLEAR_DISTANCE:
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

        if data.time - last_cam_time >= CAM_INTERVAL:
            last_cam_time = float(data.time)
            frame = camera.capture_cv_frame(data)
            corners, ids, annotated = camera.detect_markers(frame)
            # Read visible marker sizes and horizontal alignment.
            visible = {}
            if ids is not None:
                for idx, tag_id in enumerate(np.asarray(ids).reshape(-1)):
                    marker = np.asarray(corners[idx]).reshape(4, 2)
                    width = float(np.linalg.norm(marker[1] - marker[0]))
                    centre = float(np.mean(marker[:, 0]))
                    centre_error = abs(centre - camera.width / 2.0) / (camera.width / 2.0)
                    signed_error = (centre - camera.width / 2.0) / (camera.width / 2.0)
                    visible[int(tag_id)] = (width, centre_error, signed_error)

            # ===== SWITCH FROM ID 0 TO ID 99 =====

            id99_visible = (
                DOCK_TAG_ID in visible
                and visible[DOCK_TAG_ID][0] >= ID99_SWITCH_MIN_WIDTH_PX
            )

            if not docking_phase and follower.state == "TRACKING":

                if id99_visible:
                    id99_seen_frames += 1
                else:
                    id99_seen_frames = 0

                if id99_seen_frames >= ID99_CONFIRM_FRAMES:

                    docking_phase = True

                    follower.target_ids = (DOCK_TAG_ID,)
                    follower.target_id = DOCK_TAG_ID
                    follower.target_width_pixels = DOCK_WIDTH_PX

                    print(
                        f"[{data.time:.2f}s] SWITCH: ID 0 -> ID 99 "
                        f"(width={visible[DOCK_TAG_ID][0]:.1f}px)"
                    )

            if docking_phase:
                if id99_visible:
                    id99_lost_frames = 0
                    id99_reacquire_frames += 1
                    last_id99_signed_error = visible[DOCK_TAG_ID][2]
                    last_id99_seen_time = float(data.time)
                else:
                    id99_lost_frames += 1
                    id99_reacquire_frames = 0

            if not docking_phase and data.time - last_switch_debug >= 2.0:
                print(f'[{data.time:.2f}s] SEARCH ID99: '
                      f'visible={DOCK_TAG_ID in visible}, '
                      f'width={visible.get(DOCK_TAG_ID, (0.0,))[0]:.1f}px, '
                      f'confirmed={id99_seen_frames}/{ID99_CONFIRM_FRAMES} '
                      f'coarse_ready={coarse_ready}')
                last_switch_debug = float(data.time)

            if mode == 'FOLLOW':
                # TagFollower handles the initial scan and ID 0 tracking.
                # Fine docking is controlled here to avoid competing searches.
                if not docking_phase:
                    desired_v, desired_w = follower.compute_commands(corners, ids, heading)
                    if coarse_ready and follower.state == 'TRACKING':
                        # Once ID 0 is close, stop and search for ID 99.
                        desired_v, desired_w = 0.0, 0.12
                    if desired_v > 0 and front < CLEAR_DISTANCE:
                        mode = 'AVOID_TURN'
                        turn_start_heading = heading
                        turn_direction = 1 if left >= right else -1
                        desired_v, desired_w = 0.0, 0.0
                else:
                    # Never allow the original follower's blind scan to drive
                    # the fine stage. Do not move forward without ID 99.
                    if not id99_visible or id99_reacquire_frames < ID99_REACQUIRE_FRAMES:
                        desired_v = 0.0
                        if id99_lost_frames <= ID99_LOST_FRAMES:
                            desired_w = 0.0
                        else:
                            # Same sign convention as TagFollower steering.
                            desired_w = (-ID99_SEARCH_W if last_id99_signed_error > 0.05
                                         else ID99_SEARCH_W)
                    else:
                        width, centre_error, signed_error = visible[DOCK_TAG_ID]
                        desired_w = float(np.clip(
                            -follower.Kp_angular * signed_error,
                            -0.35, 0.35))
                        if centre_error > ID99_CENTER_BEFORE_DRIVE:
                            desired_v = 0.0
                        elif width < DOCK_WIDTH_PX:
                            desired_v = float(np.clip(
                                follower.Kp_linear * (DOCK_WIDTH_PX - width) / DOCK_WIDTH_PX,
                                0.0, 0.08))
                        else:
                            desired_v = 0.0

                        # if width >= DOCK_WIDTH_PX and centre_error <= DOCK_CENTER_TOLERANCE:
                        #     dock_confirm_count += 1
                        # else:
                        #     dock_confirm_count = 0

                        # ----- docking logic -----
                        reached = (
                            front <= DOCK_COMPLETE_DISTANCE
                            and front >= DOCK_HARD_STOP
                            and width >= 100.0
                        )

                        aligned = centre_error <= DOCK_CENTER_TOLERANCE

                        if reached and aligned:
                            dock_confirm_count += 1
                        else:
                            dock_confirm_count = 0

                        if data.time - last_dock_debug >= 1.0:
                            print(f'[{data.time:.2f}s] ID99 width={width:.1f}px '
                                  f'centre_error={centre_error:.3f} '
                                  f'confirm={dock_confirm_count}/{DOCK_CONFIRM_FRAMES}')
                            last_dock_debug = float(data.time)
                        if dock_confirm_count >= DOCK_CONFIRM_FRAMES:
                            charged = True
                            mode = 'CHARGED'
                            desired_v, desired_w = 0.0, 0.0
                            follower.state = 'CHARGED'
                            print(f'[{data.time:.2f}s] ROBOT CHARGED - ID 99 aligned; motors stopped')
                    # Do not treat the docking station as an ordinary obstacle,
                    # but always enforce the hard emergency stop.
                    if not charged and front < DOCK_HARD_STOP:
                        desired_v = 0.0
                        desired_w = 0.0
                    final_approach = True

            sigma = np.sqrt(np.maximum(np.diag(ekf.P), 0.0))
            cv2.putText(annotated, ('ROBOT CHARGED - ID 99' if charged else f'Mode: {mode} / {follower.state} / target={99 if docking_phase else 0}'), (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(annotated, f'EKF yaw: {math.degrees(heading):.1f} deg', (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(annotated, f'Front: {front:.2f} m', (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.putText(annotated, f'EKF x,y: {position[0]:.2f}, {position[1]:.2f} m', (15, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(annotated,
                        f'1-sigma: x={sigma[0]:.3f} y={sigma[1]:.3f} yaw={math.degrees(sigma[2]):.1f}deg',
                        (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            cv2.imshow('TurtleBot3 - Tag + LiDAR + EKF', annotated)
            cv2.waitKey(1)

        if mode != last_message_mode:
            print(f'[{data.time:.2f}s] {mode} | front={front:.2f}m')
            last_message_mode = mode
        if data.time - last_status_time >= STATUS_INTERVAL:
            last_status_time = float(data.time)
            print(f'[{data.time:.1f}s] EKF pose: x={position[0]:.3f}, '
                  f'y={position[1]:.3f}, yaw={math.degrees(heading):.1f}deg; '
                  f'sigma_xy={np.sqrt(np.diag(ekf.P)[:2])}')

        if charged:
            desired_v, desired_w = 0.0, 0.0

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
        camera.renderer.close()


if __name__ == '__main__':
    main()
