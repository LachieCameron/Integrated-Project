#!/usr/bin/env python3
"""Track A: encoder EKF + camera landmark correction + LiDAR + pose feedback.
Configure the marker's world coordinates and physical dimensions before use.
"""
import os
import sys
import math
import numpy as np
import cv2
import mujoco
import mujoco.viewer

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.sensing.cam import RaspiCamera
from src.control.ekf import PoseEKF, wrap
from src.control.Tag_follow import DiffDriveController, TagFollower, PoseTracker

XML = os.path.join(project_root, 'assets', 'scene_turtlebot3_waffle_pi.xml')
# REQUIRED CALIBRATION: replace with known map/marker coordinates and camera geometry.
MARKER_XY = (2.4, 2.5)  # known landmark location from scene XML
MARKER_SIZE_M = 0.20  # box half-width 0.1 m -> 0.20 m square
CAMERA_FX_PX = None  # derived from MuJoCo vertical FOV unless calibrated value supplied
CAMERA_CX_PX = None  # defaults to half image width
CAMERA_X_M = 0.0  # forward offset of camera from robot base
CAMERA_Y_M = 0.0  # left offset of camera from robot base
CAMERA_YAW_RAD = 0.0  # camera optical-axis yaw relative to robot forward
INITIAL_POSE = (0., 0., 0.)  # IMPORTANT: set to known initial spawn pose, not data.xpos
WAYPOINTS = []  # supply collision-free route waypoints for your spawn position
DOCK_STANDOFF_M = .35  # dock on the south side of marker, facing +Y
STOP_DISTANCE = .22
SLOW_DISTANCE = .48
ANGLES = list(range(-90, 91, 15))
CAM_INTERVAL = 1/30
MAX_WHEEL_SPEED = 5.0


def require_configuration():
    if MARKER_SIZE_M <= 0 or (CAMERA_FX_PX is not None and CAMERA_FX_PX <= 0):
        raise ValueError('Marker size and focal length must be positive')


def main():
    require_configuration()
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    camera = RaspiCamera(model)
    # MuJoCo cam_fovy is the vertical field of view (degrees).
    fx = (CAMERA_FX_PX if CAMERA_FX_PX is not None else
          (camera.height / 2.0) / math.tan(math.radians(float(model.cam_fovy[camera.cam_id])) / 2.0))
    cx0 = CAMERA_CX_PX if CAMERA_CX_PX is not None else camera.width / 2.0
    ekf = PoseEKF(INITIAL_POSE)
    drive = DiffDriveController()
    tracker = PoseTracker()
    sensor_ids = []
    for a in ANGLES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f'lidar_{a}')
        if sid < 0:
            raise ValueError(f'Missing lidar_{a}')
        sensor_ids.append(sid)

    # These must be JOINT names (not actuators), check the XML if necessary.
    def wheel_joint_id(actuator_index):
        if model.actuator_trntype[actuator_index] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError('Wheel actuator must use joint transmission')
        return int(model.actuator_trnid[actuator_index, 0])

    left_joint = wheel_joint_id(0)
    right_joint = wheel_joint_id(1)
    if min(left_joint, right_joint) < 0 or left_joint == right_joint:
        raise ValueError('Invalid wheel actuator/joint mapping')
    if model.jnt_type[left_joint] != mujoco.mjtJoint.mjJNT_HINGE or model.jnt_type[right_joint] != mujoco.mjtJoint.mjJNT_HINGE:
        raise ValueError('Expected hinge wheel joints')
    l_adr, r_adr = int(model.jnt_qposadr[left_joint]), int(model.jnt_qposadr[right_joint])
    last_wheels = None
    last_time = None
    last_cam_time = -CAM_INTERVAL
    target_seen_time = -1e6
    waypoint_index = 0
    finished = False
    state = 'WAYPOINT' if WAYPOINTS else 'SEARCH'
    v_cmd = w_cmd = 0.0
    last_print = -1e6

    def control(model, data):
        nonlocal last_wheels, last_time, last_cam_time, target_seen_time
        nonlocal waypoint_index, finished, state, v_cmd, w_cmd, last_print
        if last_time is not None and data.time < last_time:
            raise RuntimeError('Simulation time reset: restart the program to reset EKF')
        last_time = data.time
        wheels = np.array([data.qpos[l_adr], data.qpos[r_adr]], dtype=float)
        if last_wheels is not None:
            dl, dr = wheels-last_wheels
            ekf.predict(float(dl), float(dr))
        last_wheels = wheels.copy()

        ranges = np.array([float(data.sensordata[model.sensor_adr[sid]]) for sid in sensor_ids])
        ranges[(ranges < 0) | ~np.isfinite(ranges)] = np.inf
        front = float(np.min(ranges[4:9]))  # -30..+30 deg
        left_clear = float(np.min(ranges[9:]))
        right_clear = float(np.min(ranges[:4]))

        if data.time-last_cam_time >= CAM_INTERVAL:
            last_cam_time = data.time
            frame = camera.capture_cv_frame(data)
            corners, ids, annotated = camera.detect_markers(frame)
            if ids is not None and 0 in ids.flatten():
                i = int(np.where(ids.flatten() == 0)[0][0])
                c = corners[i].reshape(4, 2)
                width_px = (np.linalg.norm(c[1]-c[0]) + np.linalg.norm(c[2]-c[3]))/2
                if width_px > 2:
                    # Pinhole fronto-parallel approximation. For angled tags use calibrated solvePnP.
                    camera_range = fx*MARKER_SIZE_M/width_px
                    bearing_camera = -math.atan((float(np.mean(c[:, 0]))-cx0)/fx)  # image right is robot right (negative CCW)
                    # Camera offset compensated in robot base frame.
                    cx = camera_range*math.cos(bearing_camera+CAMERA_YAW_RAD)+CAMERA_X_M
                    cy = camera_range*math.sin(bearing_camera+CAMERA_YAW_RAD)+CAMERA_Y_M
                    distance = math.hypot(cx, cy)
                    bearing = math.atan2(cy, cx)
                    accepted = ekf.update_marker(distance, bearing, MARKER_XY)
                    if accepted:
                        target_seen_time = data.time
                        if state == 'SEARCH':
                            state = 'DOCK'
            cv2.putText(annotated, f'{state} | EKF: {ekf.x[0]:.2f},{ekf.x[1]:.2f},{math.degrees(ekf.x[2]):.0f}deg',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 255, 0), 2)
            cv2.putText(annotated, f'LiDAR front: {front:.2f}m', (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 255, 255), 2)
            cv2.imshow('Track A EKF / ArUco / LiDAR', annotated)
            cv2.waitKey(1)

        if finished:
            v_cmd, w_cmd = 0., 0.
        elif state == 'WAYPOINT':
            v_cmd, w_cmd, reached = tracker.command(ekf.x, WAYPOINTS[waypoint_index])
            if reached:
                waypoint_index += 1
                state = 'WAYPOINT' if waypoint_index < len(WAYPOINTS) else 'SEARCH'
        elif state == 'SEARCH':
            v_cmd, w_cmd = 0., .25
        elif state == 'DOCK':
            # Fixed docking pose SOUTH of the marker; do not move the goal with the robot.
            # Marker at (2.4, 2.5), dock at (2.4, 2.15), facing north (+Y).
            goal = (MARKER_XY[0], MARKER_XY[1] - DOCK_STANDOFF_M, math.pi / 2)
            v_cmd, w_cmd, reached = tracker.command(ekf.x, goal, final=True)
            if reached:
                finished = True
                state = 'DOCKED'
            if data.time-target_seen_time > 2.0:
                # Never keep blindly driving after losing the landmark.
                state = 'SEARCH'
                v_cmd, w_cmd = 0., 0.

        # Safety override from LiDAR. The marker may be on a wall: do not drive through it.
        if front < STOP_DISTANCE and v_cmd > 0:
            v_cmd = 0.
            # Stop rather than spin into obstacles; replanning is required for general navigation.
            w_cmd = 0.
            if state == 'WAYPOINT':
                state = 'BLOCKED'
        elif front < SLOW_DISTANCE:
            v_cmd = min(v_cmd, .04)
        if state == 'BLOCKED':
            v_cmd, w_cmd = 0., 0.
        lw, rw = drive.twist_to_wheel_velocities(v_cmd, w_cmd)
        scale = max(1.0, abs(lw)/MAX_WHEEL_SPEED, abs(rw)/MAX_WHEEL_SPEED)
        data.ctrl[0], data.ctrl[1] = lw/scale, rw/scale
        if data.time-last_print > 2:
            print(f'{data.time:.1f}s {state} EKF={np.round(ekf.x, 2)} front={front:.2f} v={v_cmd:.2f}')
            last_print = data.time

    try:
        mujoco.set_mjcb_control(control)
        mujoco.viewer.launch(model, data)
    finally:
        mujoco.set_mjcb_control(None)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
