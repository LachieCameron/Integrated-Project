# #!/usr/bin/env python3
# """MuJoCo TurtleBot3: ArUco visual approach with shorter LiDAR avoidance turns.

# Requires existing RaspiCamera, DiffDriveController, and lidar_-90 ... lidar_90
# sensors. Uses MuJoCo ground-truth yaw (no IMU). Adjust TAG_TARGET_WIDTH
# for final stopping distance after checking the live camera overlay.
# """
# import os
# import sys
# import math
# import numpy as np
# import cv2
# import mujoco
# import mujoco.viewer

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
# from src.sensing.cam import RaspiCamera
# from src.control.Tag_follow import DiffDriveController

# CAM_INTERVAL = 1.0 / 30.0
# ANGLES = np.arange(-90, 91, 15)
# STOP_DISTANCE = 0.32
# SLOW_DISTANCE = 0.60
# TAG_STOP_DISTANCE = 0.18  # Reduced threshold near visible docking marker; tune carefully
# HARD_STOP_DISTANCE = 0.12
# AVOID_TURN_RAD = math.radians(35)  # Previously 90 degrees
# AVOID_DISTANCE = 0.35            # Previously 0.50 metres
# AVOID_FORWARD_V = 0.09
# AVOID_TURN_W = 0.35
# SEARCH_W = 0.20
# ALIGN_W_MAX = 0.35
# ALIGN_GAIN = 0.55
# CENTER_TOL = 0.10
# APPROACH_V = 0.10
# SLOW_APPROACH_V = 0.045
# TAG_TARGET_WIDTH = 0.40  # Marker fraction of image width; larger -> closer stop
# TAG_SLOW_WIDTH = 0.22
# TAG_GRACE = 0.35
# TAG_REACQUIRE_TIMEOUT = 4.0
# MAX_WHEEL_SPEED = 5.0
# PRINT_INTERVAL = 0.5


# def angle_diff(a, b):
#     return math.atan2(math.sin(a-b), math.cos(a-b))


# def yaw_from_quat(q):
#     w, x, y, z = (float(v) for v in q)
#     return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


# def main():
#     xml_path = os.path.join(project_root, 'assets', 'scene_turtlebot3_waffle_pi.xml')
#     model = mujoco.MjModel.from_xml_path(xml_path)
#     data = mujoco.MjData(model)
#     camera = RaspiCamera(model)
#     drive = DiffDriveController()
#     base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base')
#     if base_id < 0 or model.nu < 2:
#         raise RuntimeError('Expected body "base" and two wheel actuators')
#     lidar_ids = []
#     for angle in ANGLES:
#         name = f'lidar_{angle}'
#         sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
#         if sid < 0 or model.sensor_dim[sid] != 1:
#             raise RuntimeError(f'Missing scalar LiDAR sensor: {name}')
#         lidar_ids.append(sid)

#     state = 'SEARCHING'
#     desired_v = desired_w = 0.0
#     last_cam = -CAM_INTERVAL
#     last_seen = -1e9
#     last_error = 0.0
#     tag_visible = False
#     tag_width = 0.0
#     turn_start = 0.0
#     turn_direction = 1
#     avoid_start = np.zeros(2)
#     last_print = -PRINT_INTERVAL
#     previous_state = None

#     def lidar():
#         values = np.array([float(data.sensordata[model.sensor_adr[s]]) for s in lidar_ids])
#         values[~np.isfinite(values) | (values < 0)] = np.inf
#         return values

#     def sector(values, low, high):
#         return float(np.min(values[(ANGLES >= low) & (ANGLES <= high)]))

#     def control_loop(model, data):
#         nonlocal state, desired_v, desired_w, last_cam, last_seen, last_error
#         nonlocal tag_visible, tag_width, turn_start, turn_direction, avoid_start
#         nonlocal last_print, previous_state
#         yaw = yaw_from_quat(data.xquat[base_id])
#         pos = np.array(data.xpos[base_id, :2], dtype=float)
#         ranges = lidar()
#         front = sector(ranges, -30, 30)
#         front_wide = sector(ranges, -45, 45)
#         left = sector(ranges, 45, 90)
#         right = sector(ranges, -90, -45)
#         now = float(data.time)

#         # Camera detections update at 30 Hz, independent of avoidance mode.
#         if now - last_cam >= CAM_INTERVAL or now < last_cam:
#             last_cam = now
#             frame = camera.capture_cv_frame(data)
#             corners, ids, annotated = camera.detect_markers(frame)
#             tag_visible = False
#             tag_width = 0.0
#             if ids is not None:
#                 flat_ids = np.asarray(ids).reshape(-1)
#                 matches = np.flatnonzero(flat_ids == 0)
#                 if len(matches):
#                     pts = np.asarray(corners[int(matches[0])]).reshape(4, 2)
#                     center_x = float(np.mean(pts[:, 0]))
#                     width_px = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
#                     tag_width = width_px / float(camera.width)
#                     last_error = (center_x - camera.width / 2.0) / (camera.width / 2.0)
#                     last_seen = now
#                     tag_visible = True
#                     if state in ('SEARCHING', 'TARGET_LOST'):
#                         state = 'TRACKING'
#             cv2.putText(annotated, f'Mode: {state} | Tag: {tag_visible}', (12, 28),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
#             cv2.putText(annotated, f'Yaw: {math.degrees(yaw):.1f} | Width: {tag_width:.2f}/{TAG_TARGET_WIDTH:.2f}', (12, 53),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
#             cv2.putText(annotated, f'Front: {front:.2f} | L: {left:.2f} R: {right:.2f}', (12, 78),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#             cv2.imshow('TurtleBot3 - Refined Tag + LiDAR', annotated)
#             cv2.waitKey(1)

#         recent_tag = now - last_seen <= TAG_GRACE
#         stop_threshold = TAG_STOP_DISTANCE if recent_tag else STOP_DISTANCE

#         if state == 'SEARCHING':
#             desired_v, desired_w = 0.0, SEARCH_W
#         elif state == 'TRACKING':
#             if tag_visible:
#                 if tag_width >= TAG_TARGET_WIDTH and abs(last_error) <= CENTER_TOL:
#                     state = 'ARRIVED'
#                     desired_v = desired_w = 0.0
#                 else:
#                     # Positive image error = target to camera right; positive yaw turns left.
#                     desired_w = float(np.clip(-ALIGN_GAIN * last_error, -ALIGN_W_MAX, ALIGN_W_MAX))
#                     if abs(last_error) > CENTER_TOL:
#                         desired_v = 0.0
#                     else:
#                         desired_w = 0.0
#                         desired_v = SLOW_APPROACH_V if tag_width >= TAG_SLOW_WIDTH else APPROACH_V
#             elif now - last_seen <= TAG_GRACE:
#                 desired_v, desired_w = 0.0, 0.0
#             else:
#                 state = 'TARGET_LOST'
#                 desired_v, desired_w = 0.0, 0.0
#         elif state == 'TARGET_LOST':
#             desired_v = 0.0
#             if now - last_seen > TAG_REACQUIRE_TIMEOUT:
#                 state = 'SEARCHING'
#                 desired_w = SEARCH_W
#             else:
#                 desired_w = -SEARCH_W if last_error > 0 else SEARCH_W
#         elif state == 'AVOID_TURN':
#             turned = turn_direction * angle_diff(yaw, turn_start)
#             if turned >= AVOID_TURN_RAD:
#                 state = 'AVOID_FORWARD'
#                 avoid_start = pos.copy()
#                 desired_v = desired_w = 0.0
#             else:
#                 desired_v, desired_w = 0.0, AVOID_TURN_W * turn_direction
#         elif state == 'AVOID_FORWARD':
#             distance = float(np.linalg.norm(pos - avoid_start))
#             if front < STOP_DISTANCE:
#                 state = 'AVOID_TURN'
#                 turn_start = yaw
#                 turn_direction = 1 if left >= right else -1
#                 desired_v = desired_w = 0.0
#             elif distance >= AVOID_DISTANCE:
#                 state = 'TRACKING' if now - last_seen < TAG_REACQUIRE_TIMEOUT else 'SEARCHING'
#                 desired_v = desired_w = 0.0
#             else:
#                 desired_v, desired_w = AVOID_FORWARD_V, 0.0
#         elif state == 'ARRIVED':
#             desired_v = desired_w = 0.0

#         # Collision guard applies to every state that requests forward motion.
#         if desired_v > 0:
#             if front < HARD_STOP_DISTANCE or front_wide < HARD_STOP_DISTANCE:
#                 desired_v = 0.0
#                 if state != 'ARRIVED':
#                     state = 'AVOID_TURN'
#                     turn_start = yaw
#                     turn_direction = 1 if left >= right else -1
#                     desired_w = 0.0
#             elif front < stop_threshold:
#                 desired_v = 0.0
#                 if state != 'ARRIVED':
#                     state = 'AVOID_TURN'
#                     turn_start = yaw
#                     turn_direction = 1 if left >= right else -1
#                     desired_w = 0.0
#             elif front < SLOW_DISTANCE:
#                 desired_v = min(desired_v, SLOW_APPROACH_V)

#         if state != previous_state or now - last_print >= PRINT_INTERVAL or now < last_print:
#             print(f'[{now:6.2f}] {state:14s} yaw={math.degrees(yaw):6.1f} '
#                   f'front={front:.2f} L={left:.2f} R={right:.2f} '
#                   f'tag={tag_visible} width={tag_width:.3f} v={desired_v:.2f} w={desired_w:.2f}')
#             last_print = now
#             previous_state = state

#         lw, rw = drive.twist_to_wheel_velocities(desired_v, desired_w)
#         data.ctrl[0] = float(np.clip(lw, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
#         data.ctrl[1] = float(np.clip(rw, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))

#     try:
#         mujoco.set_mjcb_control(control_loop)
#         mujoco.viewer.launch(model, data)
#     except KeyboardInterrupt:
#         print('Interrupted')
#     finally:
#         mujoco.set_mjcb_control(None)
#         cv2.destroyAllWindows()


# if __name__ == '__main__':
#     main()


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
