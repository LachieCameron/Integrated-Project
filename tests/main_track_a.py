#!/usr/bin/env python3
"""Track A integrated EKF, ArUco docking, LiDAR and LQR controller."""

import argparse
import importlib.util
import math
import os
import sys
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
scripts_root = os.path.join(project_root, "scripts")
if scripts_root not in sys.path:
    sys.path.insert(0, scripts_root)

from src.control.Tag_follow import DiffDriveController, TagFollower
from src.control.docking import DockingConfig, VisualDockingController, pose_command
from src.control.lqr import LQRTrajectoryTracker
from src.estimation.ekf import GATE_CHI2_3DOF_99, RobotEKF
from src.sensing.cam import RaspiCamera
from src.sensing.wheel_encoder import WheelEncoder
from randomize_scene import randomize_scene


XML = os.path.join(project_root, "assets", "scene_turtlebot3_waffle_pi.xml")
MARKER_XY = (2.4, 2.5)
INITIAL_POSE = np.array([0.0, 0.0, 0.0])
INITIAL_COVARIANCE = np.diag([0.15**2, 0.15**2, math.radians(15)**2])
WAYPOINTS = []
PRE_DOCK_POSE = np.array([2.4, 2.15, math.pi / 2.0])
CAMERA_POSE_COVARIANCE = np.diag(
    [0.08**2, 0.08**2, math.radians(5.0)**2]
)
CAM_INTERVAL = 1.0 / 30.0
ANGLES = np.arange(-90, 91, 15)
MAX_WHEEL_SPEED = 5.0
COARSE_ID = 0
DOCK_ID = 99
DOCK_TARGET_WIDTH_PX = 250.0
DOCK_HARD_STOP = 0.14
DOCK_CENTER_BEFORE_DRIVE = 0.10
ID99_SWITCH_MIN_WIDTH_PX = 23.0
ID99_CONFIRM_FRAMES = 5
ID99_LOST_FRAMES = 5
ID99_SEARCH_W = 0.16
CLEAR_DISTANCE = 0.40
STOP_DISTANCE = 0.20
AVOID_DISTANCE = 0.50
AVOID_TURN_RAD = math.radians(35.0)
AVOID_TURN_W = 0.45
AVOID_FORWARD_V = 0.15
GHOST_BOX_SIZE = np.array([0.14, 0.14, 0.05])
GHOST_BOX_COLOR = np.array([0.0, 1.0, 0.0, 0.35], dtype=np.float32)
BASE_VISUAL_CENTER_OFFSET = np.array([-0.064, 0.0])


def load_track_helpers():
    path = os.path.join(project_root, "scripts", "run_track_a.py")
    spec = importlib.util.spec_from_file_location("run_track_a_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Track A camera helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def correct_from_camera(ekf, camera, helpers, corners, ids):
    if ids is None:
        return False
    for marker_corners, marker_id in zip(corners, np.asarray(ids).reshape(-1)):
        marker_id = int(marker_id)
        try:
            pose = helpers.robot_pose_from_wall_marker(
                camera,
                marker_id,
                marker_corners,
                helpers.marker_size_for(marker_id),
            )
            if pose is None:
                continue
            ekf.fuse_pose(
                pose,
                CAMERA_POSE_COVARIANCE,
                GATE_CHI2_3DOF_99,
            )
            return bool(ekf.last_accepted)
        except (KeyError, ValueError, np.linalg.LinAlgError) as exc:
            print(f"Camera EKF correction skipped for ID {marker_id}: {exc}")
    return False


def angle_difference(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def add_ghost_box(scene, pose):
    """Render the EKF pose as a translucent box in the MuJoCo viewer."""
    if scene.ngeom >= len(scene.geoms):
        raise RuntimeError("MuJoCo viewer scene has no room for the EKF ghost box")
    index = scene.ngeom
    scene.ngeom += 1
    yaw = float(pose[2])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.array(
        [cosine, -sine, 0.0, sine, cosine, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    offset = np.array(
        [
            cosine * BASE_VISUAL_CENTER_OFFSET[0]
            - sine * BASE_VISUAL_CENTER_OFFSET[1],
            sine * BASE_VISUAL_CENTER_OFFSET[0]
            + cosine * BASE_VISUAL_CENTER_OFFSET[1],
        ]
    )
    mujoco.mjv_initGeom(
        scene.geoms[index],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=GHOST_BOX_SIZE,
        pos=[float(pose[0] + offset[0]), float(pose[1] + offset[1]), 0.05],
        mat=rotation,
        rgba=GHOST_BOX_COLOR,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=XML)
    parser.add_argument("--random-scene", action="store_true")
    parser.add_argument("--scene-seed", type=int, default=0)
    parser.add_argument("--obstacle-count", type=int, default=3)
    parser.add_argument(
        "--docking-mode",
        choices=("pose-visual", "legacy-pixel"),
        default="pose-visual",
    )
    parser.add_argument("--no-ghost", action="store_true")
    args = parser.parse_args()

    xml_path = os.path.abspath(args.scene)
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"Missing XML: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    if args.random_scene:
        randomize_scene(model, data, args.scene_seed, args.obstacle_count)
        print(
            f"Runtime scene randomization enabled "
            f"(seed={args.scene_seed}, obstacles={args.obstacle_count})"
        )

    helpers = load_track_helpers()
    camera = RaspiCamera(model)
    encoder = WheelEncoder(model)
    ekf = RobotEKF(
        model=model,
        initial_state=INITIAL_POSE.copy(),
        initial_covariance=INITIAL_COVARIANCE.copy(),
    )
    drive = DiffDriveController()
    tracker = LQRTrajectoryTracker(dt=0.02, v_max=0.15, w_max=0.8)
    follower = TagFollower(
        target_ids=(COARSE_ID,),
        image_width=camera.width,
        target_width_pixels=DOCK_TARGET_WIDTH_PX,
    )
    visual_docking = VisualDockingController(DockingConfig(
        target_width_px=DOCK_TARGET_WIDTH_PX,
        max_linear_v=0.06,
        max_angular_w=0.18,
    ))
    if model.nu < 2:
        raise RuntimeError("Expected two wheel actuators")

    lidar_ids = []
    for angle in ANGLES:
        sensor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{angle}"
        )
        if sensor_id < 0 or model.sensor_dim[sensor_id] != 1:
            raise RuntimeError(f"Missing or invalid LiDAR sensor: lidar_{angle}")
        lidar_ids.append(sensor_id)

    last_camera_time = -CAM_INTERVAL
    last_status_time = -1.0
    last_wheels = None
    desired_v = desired_w = 0.0
    state = "WAYPOINT" if WAYPOINTS else "SEARCH"
    waypoint_index = 0
    docking_phase = False
    dock_phase = "TRACK_TAG"
    charged = False
    legacy_straight = False
    id99_seen = 0
    id99_lost = 0
    id99_last_error = 0.0
    dock_confirm = 0
    mode = "FOLLOW"
    turn_start_heading = 0.0
    turn_direction = 1.0
    forward_start = np.zeros(2)

    def lidar():
        values = []
        for sensor_id in lidar_ids:
            value = float(data.sensordata[model.sensor_adr[sensor_id]])
            values.append(value if np.isfinite(value) and value >= 0 else math.inf)
        return np.asarray(values)

    def sector_min(values, low, high):
        return float(np.min(values[(ANGLES >= low) & (ANGLES <= high)]))

    def lqr_command(target, final=False):
        target = np.asarray(target, dtype=float)
        error = target[:2] - ekf.state[:2]
        position_reached = float(np.hypot(*error)) < 0.05
        heading_reached = abs(angle_difference(target[2], ekf.state[2])) < 0.08
        if position_reached and (not final or heading_reached):
            return 0.0, 0.0, True
        v_ref = 0.0 if position_reached else 0.10
        return (*tracker.control(ekf.state, target, v_ref, 0.0)[:2], False)

    def control(model, data):
        nonlocal last_camera_time, last_status_time, last_wheels
        nonlocal desired_v, desired_w, state, waypoint_index
        nonlocal docking_phase, dock_phase, charged, legacy_straight
        nonlocal id99_seen, id99_lost, id99_last_error, dock_confirm
        nonlocal mode, turn_start_heading, turn_direction, forward_start

        d_left, d_right, _, _ = encoder.step(data)
        ekf.predict(d_left, d_right)
        position = ekf.state[:2].copy()
        heading = float(ekf.state[2])
        readings = lidar()
        front = sector_min(readings, -30, 30)
        left = sector_min(readings, 45, 90)
        right = sector_min(readings, -90, -45)

        if charged:
            data.ctrl[:2] = 0.0
            return

        if docking_phase and (visual_docking.straight_approach or legacy_straight) and \
                np.isfinite(front) and front <= DOCK_HARD_STOP:
            charged = True
            state = "CHARGED"
            desired_v = desired_w = 0.0
            print(f"[{data.time:.2f}s] DOCK RANGE REACHED; charging")
            data.ctrl[:2] = 0.0
            return

        if mode == "FOLLOW" and not docking_phase and desired_v > 0 and \
                front < CLEAR_DISTANCE:
            mode = "AVOID_TURN"
            turn_start_heading = heading
            turn_direction = 1.0 if left >= right else -1.0

        if mode == "AVOID_TURN":
            turned = turn_direction * angle_difference(heading, turn_start_heading)
            desired_v, desired_w = (0.0, AVOID_TURN_W * turn_direction)
            if turned >= AVOID_TURN_RAD:
                mode = "AVOID_FORWARD"
                forward_start = position.copy()
        elif mode == "AVOID_FORWARD":
            if front < STOP_DISTANCE:
                mode = "AVOID_TURN"
                turn_start_heading = heading
                turn_direction = 1.0 if left >= right else -1.0
                desired_v = desired_w = 0.0
            elif np.linalg.norm(position - forward_start) >= AVOID_DISTANCE:
                mode = "FOLLOW"
                desired_v = desired_w = 0.0
            else:
                desired_v, desired_w = AVOID_FORWARD_V, 0.0

        if data.time - last_camera_time >= CAM_INTERVAL:
            last_camera_time = float(data.time)
            frame = camera.capture_cv_frame(data)
            corners, ids, annotated = camera.detect_markers(frame)
            visible = {}
            if ids is not None:
                for index, raw_id in enumerate(np.asarray(ids).reshape(-1)):
                    marker = np.asarray(corners[index]).reshape(4, 2)
                    width = float((np.linalg.norm(marker[1] - marker[0]) +
                                   np.linalg.norm(marker[2] - marker[3])) / 2.0)
                    signed_error = (
                        float(np.mean(marker[:, 0])) - camera.width / 2.0
                    ) / (camera.width / 2.0)
                    visible[int(raw_id)] = (width, abs(signed_error), signed_error)

            correct_from_camera(ekf, camera, helpers, corners, ids)
            position = ekf.state[:2].copy()
            heading = float(ekf.state[2])
            coarse_visible = COARSE_ID in visible
            fine_visible = (
                DOCK_ID in visible
                and visible[DOCK_ID][0] >= ID99_SWITCH_MIN_WIDTH_PX
            )

            if not docking_phase:
                id99_seen = id99_seen + 1 if DOCK_ID in visible else 0
                if coarse_visible or id99_seen >= ID99_CONFIRM_FRAMES:
                    docking_phase = True
                    dock_phase = (
                        "PRE_DOCK"
                        if args.docking_mode == "pose-visual"
                        else "LEGACY_PIXEL"
                    )
                    follower.target_ids = (DOCK_ID,)
                    follower.target_id = DOCK_ID
                    follower.state = "TRACKING"
                    print(f"[{data.time:.2f}s] DOCK TAG -> {dock_phase}")

            if docking_phase:
                if fine_visible or coarse_visible:
                    id99_lost = 0
                    id99_last_error = visible[
                        DOCK_ID if fine_visible else COARSE_ID
                    ][2]
                else:
                    id99_lost += 1

            if mode == "FOLLOW":
                if not docking_phase:
                    desired_v, desired_w = follower.compute_commands(
                        corners, ids, heading
                    )
                elif dock_phase == "PRE_DOCK":
                    desired_v, desired_w, reached = pose_command(
                        np.r_[position, heading], PRE_DOCK_POSE
                    )
                    if reached:
                        dock_phase = "VISUAL_APPROACH"
                        visual_docking.reset()
                        print(f"[{data.time:.2f}s] PRE_DOCK reached; visual approach")
                elif dock_phase == "LEGACY_PIXEL":
                    if legacy_straight:
                        desired_v = 0.025 if front > DOCK_HARD_STOP else 0.0
                        desired_w = 0.0
                    elif fine_visible and id99_lost == 0:
                        width, center, signed = visible[DOCK_ID]
                        desired_w = 0.0 if center <= 0.04 else float(
                            np.clip(-follower.Kp_angular * signed, -0.25, 0.25)
                        )
                        desired_v = 0.0 if center > DOCK_CENTER_BEFORE_DRIVE else float(
                            np.clip(
                                follower.Kp_linear *
                                (DOCK_TARGET_WIDTH_PX - width) /
                                DOCK_TARGET_WIDTH_PX, 0.0, 0.08
                            )
                        )
                        if width >= DOCK_TARGET_WIDTH_PX and center <= 0.12:
                            legacy_straight = True
                    else:
                        desired_v = 0.0
                        desired_w = (
                            -ID99_SEARCH_W if id99_last_error > 0.05
                            else ID99_SEARCH_W
                        ) if id99_lost > ID99_LOST_FRAMES else 0.0
                elif dock_phase == "VISUAL_APPROACH":
                    if fine_visible:
                        width, center, signed = visible[DOCK_ID]
                        desired_v, desired_w, _ = visual_docking.command(
                            width, center, signed, front,
                            heading, PRE_DOCK_POSE[2],
                        )
                    elif visual_docking.straight_approach:
                        desired_v, desired_w, _ = visual_docking.command(
                            None, None, None, front
                        )
                    else:
                        desired_v = 0.0
                        desired_w = (
                            -ID99_SEARCH_W if id99_last_error > 0.05
                            else ID99_SEARCH_W
                        )

                if docking_phase and front <= DOCK_HARD_STOP:
                    desired_v = desired_w = 0.0

            cv2.putText(
                annotated,
                "CHARGED" if charged else f"{mode}/{dock_phase}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            )
            cv2.putText(
                annotated, f"Front: {front:.2f}m EKF: {position[0]:.2f},{position[1]:.2f}",
                (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,
            )
            cv2.imshow("Track A EKF / ArUco / LiDAR", annotated)
            cv2.waitKey(1)

        if docking_phase and front < DOCK_HARD_STOP:
            desired_v = desired_w = 0.0
        elif not docking_phase and front < STOP_DISTANCE and desired_v > 0:
            desired_v = desired_w = 0.0
            mode = "AVOID_TURN"
        elif front < STOP_DISTANCE + 0.18:
            desired_v = min(desired_v, 0.04)

        left_wheel, right_wheel = drive.twist_to_wheel_velocities(
            desired_v, desired_w
        )
        data.ctrl[0] = float(np.clip(left_wheel, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
        data.ctrl[1] = float(np.clip(right_wheel, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
        if data.time - last_status_time >= 1.0:
            last_status_time = float(data.time)
            print(
                f"[{data.time:.1f}s] {mode}/{dock_phase} "
                f"EKF={np.round(ekf.state, 2)} front={front:.2f}"
            )

    try:
        mujoco.set_mjcb_control(control)
        if args.no_ghost:
            mujoco.viewer.launch(model, data)
        else:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running():
                    start = time.time()
                    mujoco.mj_step(model, data)
                    viewer.user_scn.ngeom = 0
                    add_ghost_box(viewer.user_scn, ekf.state)
                    viewer.sync()
                    time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))
    finally:
        mujoco.set_mjcb_control(None)
        cv2.destroyAllWindows()
        camera.renderer.close()


if __name__ == "__main__":
    main()
