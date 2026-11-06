#!/usr/bin/env python3
"""Track A: ArUco coarse/fine docking, LiDAR avoidance, wheel-odometry EKF.

Camera-to-EKF fusion uses the calibrated wall-marker helper from Track A.
No MuJoCo ground truth is used for navigation. The CHARGED flag is a
simulated docking-success indication.
"""
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SCRIPTS_ROOT = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, SCRIPTS_ROOT)

from src.sensing.cam import RaspiCamera
from src.sensing.wheel_encoder import WheelEncoder
from src.estimation.ekf import RobotEKF
from src.control.Tag_follow import DiffDriveController, TagFollower
from src.control.docking import DockingConfig, VisualDockingController
from src.control.lqr import LQRTrajectoryTracker
from src.control.navigation import NavConfig, Navigator, State as NavState
from src.utils.config import load_config
from randomize_scene import randomize_scene


def load_track_helpers():
    path = os.path.join(PROJECT_ROOT, "scripts", "run_track_a.py")
    spec = importlib.util.spec_from_file_location("run_track_a", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Track A helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAM_INTERVAL = 1.0 / 30.0
STATUS_INTERVAL = 1.0
ANGLES = np.arange(-90, 91, 15)
STOP_DISTANCE = 0.20
CLEAR_DISTANCE = 0.40
AVOID_DISTANCE = 0.50
AVOID_TURN_RAD = math.radians(35)
AVOID_FORWARD_V = 0.15
AVOID_TURN_W = 0.45
MAX_WHEEL_SPEED = 5.0

INITIAL_POSE_ESTIMATE = np.array([0.0, 0.0, 0.0])
INITIAL_COVARIANCE = np.diag([0.15**2, 0.15**2, math.radians(15)**2])

COARSE_ID = 0
DOCK_ID = 99
COARSE_TARGET_WIDTH_PX = 250.0
DOCK_TARGET_WIDTH_PX = 250.0
ID99_SWITCH_MIN_WIDTH_PX = 23.0
ID99_CONFIRM_FRAMES = 5
ID99_LOST_FRAMES = 5
ID99_REACQUIRE_FRAMES = 5
ID99_SEARCH_W = 0.16
ID99_CENTER_BEFORE_DRIVE = 0.10
VISUAL_CENTER_DEADBAND = 0.04
VISUAL_HEADING_CORRECTION_THRESHOLD = math.radians(8.0)
DOCK_CENTER_TOLERANCE = 0.12
DOCK_COMPLETE_DISTANCE = 0.14
DOCK_HARD_STOP = 0.14
DOCK_MIN_MARKER_WIDTH_PX = 100.0
DOCK_CONFIRM_FRAMES = 3
PRE_DOCK_POSE = np.array([2.4, 2.15, math.pi / 2.0])
PRE_DOCK_POSITION_TOLERANCE = 0.10
PRE_DOCK_HEADING_TOLERANCE = math.radians(8.0)
DOCK_HEADING_TOLERANCE = math.radians(4.0)
PRE_DOCK_MAX_V = 0.10
PRE_DOCK_MAX_W = 0.45
PRE_DOCK_CONTROL_DT = 0.02

CAMERA_POSE_COVARIANCE = np.diag([0.08**2, 0.08**2, math.radians(5)**2])
POSE_GATE_CHI2 = 11.345  # 99% chi-square threshold for 3 dimensions
GHOST_BOX_SIZE = np.array([0.14, 0.14, 0.05])
GHOST_BOX_COLOR = np.array([0.0, 1.0, 0.0, 0.35], dtype=np.float32)
BASE_VISUAL_CENTER_OFFSET = np.array([-0.064, 0.0])


def angle_difference(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def correct_from_camera(ekf, camera, helpers, corners, ids):
    """Fuse a calibrated wall-marker pose into the EKF."""
    if ids is None:
        return 0
    for marker_corners, marker_id in zip(corners, np.asarray(ids).reshape(-1)):
        try:
            measurement = helpers.robot_pose_from_wall_marker(
                camera, int(marker_id), marker_corners,
                helpers.marker_size_for(int(marker_id)),
            )
            if measurement is None:
                continue
            ekf.fuse_pose(measurement, CAMERA_POSE_COVARIANCE, POSE_GATE_CHI2)
            return int(bool(ekf.last_accepted))
        except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
            print(f"Camera EKF correction skipped for ID {int(marker_id)}: {exc}")
    return 0


def add_ghost_box(scene, pose):
    """Render a translucent box at an EKF pose in the viewer user scene."""
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
    parser.add_argument(
        "--no-ghost",
        action="store_true",
        help="disable EKF ghost-box rendering",
    )
    parser.add_argument(
        "--random-scene",
        action="store_true",
        help="generate a randomized obstacle scene before starting",
    )
    parser.add_argument(
        "--scene-seed",
        type=int,
        default=0,
        help="seed for randomized obstacle placement",
    )
    parser.add_argument(
        "--obstacle-count",
        type=int,
        default=3,
        help="number of randomized obstacles",
    )
    parser.add_argument("--scene", type=str, default=os.path.join(
        PROJECT_ROOT, "assets", "scene_turtlebot3_waffle_pi.xml"
    ))
    parser.add_argument(
        "--docking-mode",
        choices=("pose-visual", "legacy-pixel"),
        default="pose-visual",
        help="pose pre-dock then visual servo, or the previous pixel-following logic",
    )
    args = parser.parse_args()
    xml_path = os.path.abspath(args.scene)
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"Missing XML: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    if args.random_scene:
        randomize_scene(model, data, args.scene_seed, args.obstacle_count)
    track_helpers = load_track_helpers()
    camera = RaspiCamera(model)
    encoder = WheelEncoder(model)
    ekf = RobotEKF(
        model=model,
        initial_state=INITIAL_POSE_ESTIMATE.copy(),
        initial_covariance=INITIAL_COVARIANCE.copy(),
    )
    app_config = load_config(os.path.join(PROJECT_ROOT, "config", "default.yaml"))
    pre_dock_config = NavConfig(
        dock_x=float(PRE_DOCK_POSE[0]),
        dock_stop_y=float(PRE_DOCK_POSE[1]),
        station_face_y=2.5,
        v_max=min(app_config.lqr.max_linear_velocity, PRE_DOCK_MAX_V),
        v_dock=min(app_config.lqr.max_linear_velocity, PRE_DOCK_MAX_V),
        w_max=min(app_config.lqr.max_angular_velocity, PRE_DOCK_MAX_W),
        stop_range=STOP_DISTANCE,
        slow_range=CLEAR_DISTANCE,
        dock_stop_range=DOCK_HARD_STOP,
        visual_docking=True,
    )
    pre_dock_tracker = LQRTrajectoryTracker(
        PRE_DOCK_CONTROL_DT,
        q_diag=app_config.lqr.q,
        r_diag=app_config.lqr.r,
        v_max=pre_dock_config.v_max,
        w_max=pre_dock_config.w_max,
    )
    pre_dock_navigator = Navigator(
        model,
        PRE_DOCK_CONTROL_DT,
        config=pre_dock_config,
        tracker=pre_dock_tracker,
    )
    diff_drive = DiffDriveController()
    follower = TagFollower(
        target_ids=(COARSE_ID,),
        image_width=camera.width,
        target_width_pixels=COARSE_TARGET_WIDTH_PX,
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
        name = f"lidar_{angle}"
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0 or model.sensor_dim[sensor_id] != 1:
            raise RuntimeError(f"Missing or invalid LiDAR sensor: {name}")
        lidar_ids.append(sensor_id)

    print("Camera EKF correction ENABLED")

    last_cam_time = -CAM_INTERVAL
    last_status_time = -STATUS_INTERVAL
    last_dock_debug = -STATUS_INTERVAL
    desired_v = desired_w = 0.0
    mode = "FOLLOW"
    turn_start_heading = 0.0
    turn_direction = 1
    forward_start = np.zeros(2)
    docking_phase = False
    dock_phase = "TRACK_TAG"
    charged = False
    id99_seen_frames = 0
    id99_lost_frames = 0
    id99_reacquire_frames = 0
    last_id99_signed_error = 0.0
    dock_confirm_count = 0
    total_camera_updates = 0
    last_message_mode = None
    legacy_straight_approach = False
    last_pre_dock_update = -PRE_DOCK_CONTROL_DT

    def lidar_distances():
        result = []
        for sensor_id in lidar_ids:
            reading = float(data.sensordata[model.sensor_adr[sensor_id]])
            result.append(reading if np.isfinite(reading) and reading >= 0 else math.inf)
        return np.asarray(result)

    def sector_min(readings, lo, hi):
        return float(np.min(readings[(ANGLES >= lo) & (ANGLES <= hi)]))

    def control_loop(model, data):
        nonlocal last_cam_time, last_status_time, last_dock_debug
        nonlocal desired_v, desired_w, mode, turn_start_heading, turn_direction
        nonlocal forward_start, docking_phase, dock_phase, charged, id99_seen_frames
        nonlocal id99_lost_frames, id99_reacquire_frames, last_id99_signed_error
        nonlocal dock_confirm_count, total_camera_updates, last_message_mode
        nonlocal legacy_straight_approach
        nonlocal last_pre_dock_update

        # Encoder-based EKF prediction; do not use simulator true pose to navigate.
        d_left, d_right, _, _ = encoder.step(data)
        ekf.predict(d_left, d_right)
        position = ekf.state[:2].copy()
        heading = float(ekf.state[2])

        readings = lidar_distances()
        front = sector_min(readings, -30, 30)
        left = sector_min(readings, 45, 90)
        right = sector_min(readings, -90, -45)

        if charged:
            data.ctrl[0] = data.ctrl[1] = 0.0
            return

        if (
            docking_phase
            and not charged
            and np.isfinite(front)
            and front <= DOCK_HARD_STOP
            and (
                legacy_straight_approach
                or visual_docking.straight_approach
            )
        ):
            charged = True
            mode = "CHARGED"
            follower.state = "CHARGED"
            desired_v = desired_w = 0.0
            print(f"[{data.time:.2f}s] DOCK RANGE REACHED; charging")
            data.ctrl[0] = data.ctrl[1] = 0.0
            return

        pre_dock_active = docking_phase and dock_phase == "PRE_DOCK"
        if (
            not pre_dock_active
            and
            mode == "FOLLOW"
            and (not docking_phase or dock_phase == "PRE_DOCK")
            and desired_v > 0
            and front < CLEAR_DISTANCE
        ):
            mode = "AVOID_TURN"
            turn_start_heading = heading
            turn_direction = 1 if left >= right else -1

        if not pre_dock_active and mode == "AVOID_TURN":
            turned = turn_direction * angle_difference(heading, turn_start_heading)
            if turned >= AVOID_TURN_RAD:
                mode = "AVOID_FORWARD"
                forward_start = position.copy()
                desired_v = desired_w = 0.0
            else:
                desired_v, desired_w = 0.0, AVOID_TURN_W * turn_direction
        elif not pre_dock_active and mode == "AVOID_FORWARD":
            travelled = float(np.linalg.norm(position - forward_start))
            if front < STOP_DISTANCE:
                mode = "AVOID_TURN"
                turn_start_heading = heading
                turn_direction = 1 if left >= right else -1
                desired_v = desired_w = 0.0
            elif travelled >= AVOID_DISTANCE:
                mode = "FOLLOW"
                desired_v = desired_w = 0.0
            else:
                desired_v, desired_w = AVOID_FORWARD_V, 0.0

        if data.time - last_cam_time >= CAM_INTERVAL:
            last_cam_time = float(data.time)
            frame = camera.capture_cv_frame(data)
            corners, ids, annotated = camera.detect_markers(frame)

            # Always populate visible tags, even when EKF landmark map is empty.
            visible = {}
            if ids is not None:
                for i, raw_id in enumerate(np.asarray(ids).reshape(-1)):
                    marker_id = int(raw_id)
                    marker = np.asarray(corners[i]).reshape(4, 2)
                    width = float(np.linalg.norm(marker[1] - marker[0]))
                    center_x = float(np.mean(marker[:, 0]))
                    signed_error = (center_x - camera.width / 2) / (camera.width / 2)
                    visible[marker_id] = (width, abs(signed_error), signed_error)

            total_camera_updates += correct_from_camera(
                ekf, camera, track_helpers, corners, ids
            )
            position = ekf.state[:2].copy()
            heading = float(ekf.state[2])

            dock_tag_visible = any(tag_id in visible for tag_id in (COARSE_ID, DOCK_ID))
            if not docking_phase and dock_tag_visible:
                id99_seen_frames = id99_seen_frames + 1 if DOCK_ID in visible else 0
                if id99_seen_frames >= ID99_CONFIRM_FRAMES or COARSE_ID in visible:
                    docking_phase = True
                    dock_phase = (
                        "PRE_DOCK"
                        if args.docking_mode == "pose-visual"
                        else "LEGACY_PIXEL"
                    )
                    if dock_phase == "LEGACY_PIXEL":
                        follower.target_ids = (DOCK_ID,)
                        follower.target_id = DOCK_ID
                        follower.target_width_pixels = DOCK_TARGET_WIDTH_PX
                        legacy_straight_approach = False
                    else:
                        mode = "FOLLOW"
                        desired_v = desired_w = 0.0
                        last_pre_dock_update = data.time - PRE_DOCK_CONTROL_DT
                    follower.state = "TRACKING"
                    print(f"[{data.time:.2f}s] DOCK TAG -> {dock_phase}")

            id99_visible = DOCK_ID in visible and visible[DOCK_ID][0] >= ID99_SWITCH_MIN_WIDTH_PX
            if docking_phase:
                if id99_visible or COARSE_ID in visible:
                    id99_lost_frames = 0
                    id99_reacquire_frames += 1
                    last_id99_signed_error = visible[DOCK_ID if id99_visible else COARSE_ID][2]
                else:
                    id99_lost_frames += 1
                    id99_reacquire_frames = 0
                    dock_confirm_count = 0

            if mode == "FOLLOW":
                if not docking_phase:
                    desired_v, desired_w = follower.compute_commands(corners, ids, heading)
                    if desired_v > 0 and front < CLEAR_DISTANCE:
                        mode = "AVOID_TURN"
                        turn_start_heading = heading
                        turn_direction = 1 if left >= right else -1
                        desired_v = desired_w = 0.0
                elif dock_phase == "LEGACY_PIXEL":
                    if legacy_straight_approach:
                        desired_v = 0.025 if front > DOCK_HARD_STOP else 0.0
                        desired_w = 0.0
                        dock_confirm_count = (
                            dock_confirm_count + 1
                            if front <= DOCK_COMPLETE_DISTANCE
                            else 0
                        )
                        if dock_confirm_count >= DOCK_CONFIRM_FRAMES:
                            charged = True
                            mode = "CHARGED"
                            follower.state = "CHARGED"
                            desired_v = desired_w = 0.0
                            print(
                                f"[{data.time:.2f}s] LEGACY PIXEL DOCK COMPLETE; "
                                "motors stopped"
                            )
                    elif not id99_visible or id99_reacquire_frames < ID99_REACQUIRE_FRAMES:
                        desired_v = 0.0
                        if id99_lost_frames <= ID99_LOST_FRAMES:
                            desired_w = 0.0
                        else:
                            desired_w = (
                                -ID99_SEARCH_W if last_id99_signed_error > 0.05
                                else ID99_SEARCH_W
                            )
                    else:
                        width, center_error, signed_error = visible[DOCK_ID]
                        desired_w = 0.0 if center_error <= VISUAL_CENTER_DEADBAND else float(np.clip(
                            -follower.Kp_angular * signed_error, -0.25, 0.25
                        ))
                        if center_error > ID99_CENTER_BEFORE_DRIVE:
                            desired_v = 0.0
                        elif width < DOCK_TARGET_WIDTH_PX:
                            desired_v = float(np.clip(
                                follower.Kp_linear
                                * (DOCK_TARGET_WIDTH_PX - width)
                                / DOCK_TARGET_WIDTH_PX,
                                0.0,
                                0.08,
                            ))
                        else:
                            desired_v = 0.0
                        if (
                            width >= DOCK_TARGET_WIDTH_PX
                            and center_error <= DOCK_CENTER_TOLERANCE
                        ):
                            legacy_straight_approach = True
                            desired_v = 0.025 if front > DOCK_HARD_STOP else 0.0

                        reached = (
                            DOCK_HARD_STOP <= front <= DOCK_COMPLETE_DISTANCE
                            and width >= DOCK_MIN_MARKER_WIDTH_PX
                        )
                        aligned = center_error <= DOCK_CENTER_TOLERANCE
                        dock_confirm_count = (
                            dock_confirm_count + 1 if reached and aligned else 0
                        )
                        if data.time - last_dock_debug >= STATUS_INTERVAL:
                            print(
                                f"[{data.time:.2f}s] LEGACY ID99 width={width:.1f}px "
                                f"center_error={center_error:.3f} "
                                f"confirm={dock_confirm_count}/{DOCK_CONFIRM_FRAMES}"
                            )
                            last_dock_debug = float(data.time)
                        if dock_confirm_count >= DOCK_CONFIRM_FRAMES:
                            charged = True
                            mode = "CHARGED"
                            follower.state = "CHARGED"
                            desired_v = desired_w = 0.0
                            print(
                                f"[{data.time:.2f}s] LEGACY PIXEL DOCK COMPLETE; "
                                "motors stopped"
                            )
                elif dock_phase == "VISUAL_APPROACH":
                    if visual_docking.straight_approach:
                        desired_v, desired_w, _ = visual_docking.command(
                            None, None, None, front
                        )
                    elif id99_visible:
                        width, center_error, signed_error = visible[DOCK_ID]
                        desired_w = (
                            0.0
                            if center_error <= VISUAL_CENTER_DEADBAND
                            else float(np.clip(
                                -follower.Kp_angular * signed_error,
                                -0.25,
                                0.25,
                            ))
                        )
                        if center_error > ID99_CENTER_BEFORE_DRIVE:
                            desired_v = 0.0
                        elif width < DOCK_TARGET_WIDTH_PX:
                            desired_v = float(np.clip(
                                follower.Kp_linear
                                * (DOCK_TARGET_WIDTH_PX - width)
                                / DOCK_TARGET_WIDTH_PX,
                                0.0,
                                0.08,
                            ))
                        else:
                            visual_docking.start_straight_approach()
                            desired_v = 0.025 if front > DOCK_HARD_STOP else 0.0
                            desired_w = 0.0
                    else:
                        desired_v = 0.0
                        desired_w = (
                            -ID99_SEARCH_W if last_id99_signed_error > 0.05
                            else ID99_SEARCH_W
                        )

                if docking_phase and not charged and front < DOCK_HARD_STOP:
                    desired_v = desired_w = 0.0

            sigma = np.sqrt(np.maximum(np.diag(ekf.P), 0.0))
            cv2.putText(annotated,
                        "DOCKED (SIMULATED)" if charged else
                        f"Mode: {mode}/{follower.state} target={DOCK_ID if docking_phase else COARSE_ID}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(annotated, f"EKF yaw: {math.degrees(heading):.1f} deg",
                        (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(annotated, f"Front: {front:.2f} m",
                        (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.putText(annotated, f"EKF x,y: {position[0]:.2f}, {position[1]:.2f} m",
                        (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(annotated,
                        f"Sigma: x={sigma[0]:.2f} y={sigma[1]:.2f} yaw={math.degrees(sigma[2]):.1f}deg",
                        (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            cv2.putText(annotated, f"Camera EKF corrections: {total_camera_updates}",
                        (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            cv2.imshow("TurtleBot3 - Tag + LiDAR + EKF", annotated)
            cv2.waitKey(1)

        if docking_phase and dock_phase == "PRE_DOCK":
            if data.time - last_pre_dock_update >= PRE_DOCK_CONTROL_DT:
                nav_output = pre_dock_navigator.update(
                    np.array([position[0], position[1], heading]),
                    readings,
                )
                desired_v, desired_w = nav_output.v, nav_output.w
                last_pre_dock_update = float(data.time)
                if nav_output.state == NavState.FAILED:
                    dock_phase = "PRE_DOCK_FAILED"
                    desired_v = desired_w = 0.0
                    reason = (
                        pre_dock_navigator.history[-1]
                        if pre_dock_navigator.history
                        else "navigation returned FAILED"
                    )
                    print(
                        f"[{data.time:.2f}s] LQR pre-dock navigation failed: "
                        f"{reason}"
                    )
                elif (
                    np.linalg.norm(position - PRE_DOCK_POSE[:2])
                    <= PRE_DOCK_POSITION_TOLERANCE
                    and abs(angle_difference(heading, PRE_DOCK_POSE[2]))
                    <= PRE_DOCK_HEADING_TOLERANCE
                ):
                    dock_phase = "VISUAL_APPROACH"
                    visual_docking.reset()
                    desired_v = desired_w = 0.0
                    print(
                        f"[{data.time:.2f}s] LQR pre-dock reached; "
                        "VISUAL_APPROACH"
                    )

        if docking_phase and not charged and front < DOCK_HARD_STOP:
            desired_v = desired_w = 0.0

            sigma = np.sqrt(np.maximum(np.diag(ekf.P), 0.0))
            cv2.putText(annotated,
                        "DOCKED (SIMULATED)" if charged else
                        f"Mode: {mode}/{follower.state} target={DOCK_ID if docking_phase else COARSE_ID}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(annotated, f"EKF yaw: {math.degrees(heading):.1f} deg",
                        (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(annotated, f"Front: {front:.2f} m",
                        (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.putText(annotated, f"EKF x,y: {position[0]:.2f}, {position[1]:.2f} m",
                        (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv2.putText(annotated,
                        f"Sigma: x={sigma[0]:.2f} y={sigma[1]:.2f} yaw={math.degrees(sigma[2]):.1f}deg",
                        (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            cv2.putText(annotated, f"Camera EKF corrections: {total_camera_updates}",
                        (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            cv2.imshow("TurtleBot3 - Tag + LiDAR + EKF", annotated)
            cv2.waitKey(1)

        if mode != last_message_mode:
            print(f"[{data.time:.2f}s] {mode} | front={front:.2f}m")
            last_message_mode = mode
        if data.time - last_status_time >= STATUS_INTERVAL:
            last_status_time = float(data.time)
            print(f"[{data.time:.1f}s] EKF x={position[0]:.3f}, y={position[1]:.3f}, "
                  f"yaw={math.degrees(heading):.1f}deg; "
                  f"sigma_xy={np.sqrt(np.maximum(np.diag(ekf.P)[:2], 0.0))}")

        left_wheel, right_wheel = diff_drive.twist_to_wheel_velocities(desired_v, desired_w)
        data.ctrl[0] = float(np.clip(left_wheel, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
        data.ctrl[1] = float(np.clip(right_wheel, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))

    def run_with_ghost():
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                start = time.time()
                mujoco.mj_step(model, data)
                viewer.user_scn.ngeom = 0
                add_ghost_box(viewer.user_scn, ekf.state)
                viewer.sync()
                time.sleep(
                    max(0.0, model.opt.timestep - (time.time() - start))
                )

    try:
        mujoco.set_mjcb_control(control_loop)
        if args.no_ghost:
            mujoco.viewer.launch(model, data)
        else:
            run_with_ghost()
    except KeyboardInterrupt:
        print("Simulation interrupted")
    finally:
        mujoco.set_mjcb_control(None)
        cv2.destroyAllWindows()
        camera.renderer.close()


if __name__ == "__main__":
    main()

# ======= original code =======
# #!/usr/bin/env python3
# """Track A: ArUco coarse/fine docking, LiDAR avoidance, wheel-odometry EKF.

# Camera-to-EKF fusion uses the calibrated wall-marker helper from Track A.
# No MuJoCo ground truth is used for navigation. The CHARGED flag is a
# simulated docking-success indication.
# """
# import argparse
# import importlib.util
# import math
# import os
# import sys
# import time

# import cv2
# import mujoco
# import mujoco.viewer
# import numpy as np

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)
# SCRIPTS_ROOT = os.path.join(PROJECT_ROOT, "scripts")
# if SCRIPTS_ROOT not in sys.path:
#     sys.path.insert(0, SCRIPTS_ROOT)

# from src.sensing.cam import RaspiCamera
# from src.sensing.wheel_encoder import WheelEncoder
# from src.estimation.ekf import RobotEKF
# from src.control.Tag_follow import DiffDriveController, TagFollower
# from src.control.docking import DockingConfig, VisualDockingController, pose_command
# from randomize_scene import randomize_scene


# def load_track_helpers():
#     path = os.path.join(PROJECT_ROOT, "scripts", "run_track_a.py")
#     spec = importlib.util.spec_from_file_location("run_track_a", path)
#     if spec is None or spec.loader is None:
#         raise RuntimeError("could not load Track A helpers")
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     return module


# CAM_INTERVAL = 1.0 / 30.0
# STATUS_INTERVAL = 1.0
# ANGLES = np.arange(-90, 91, 15)
# STOP_DISTANCE = 0.20
# CLEAR_DISTANCE = 0.40
# AVOID_DISTANCE = 0.50
# AVOID_TURN_RAD = math.radians(35)
# AVOID_FORWARD_V = 0.15
# AVOID_TURN_W = 0.45
# MAX_WHEEL_SPEED = 5.0

# INITIAL_POSE_ESTIMATE = np.array([0.0, 0.0, 0.0])
# INITIAL_COVARIANCE = np.diag([0.15**2, 0.15**2, math.radians(15)**2])

# COARSE_ID = 0
# DOCK_ID = 99
# COARSE_TARGET_WIDTH_PX = 250.0
# DOCK_TARGET_WIDTH_PX = 250.0
# ID99_SWITCH_MIN_WIDTH_PX = 23.0
# ID99_CONFIRM_FRAMES = 5
# ID99_LOST_FRAMES = 5
# ID99_REACQUIRE_FRAMES = 5
# ID99_SEARCH_W = 0.16
# ID99_CENTER_BEFORE_DRIVE = 0.10
# VISUAL_CENTER_DEADBAND = 0.04
# VISUAL_HEADING_CORRECTION_THRESHOLD = math.radians(8.0)
# DOCK_CENTER_TOLERANCE = 0.12
# DOCK_COMPLETE_DISTANCE = 0.14
# DOCK_HARD_STOP = 0.14
# DOCK_MIN_MARKER_WIDTH_PX = 100.0
# DOCK_CONFIRM_FRAMES = 3
# PRE_DOCK_POSE = np.array([2.4, 2.15, math.pi / 2.0])
# PRE_DOCK_POSITION_TOLERANCE = 0.10
# PRE_DOCK_HEADING_TOLERANCE = math.radians(8.0)
# DOCK_HEADING_TOLERANCE = math.radians(4.0)
# PRE_DOCK_MAX_V = 0.10
# PRE_DOCK_MAX_W = 0.45

# CAMERA_POSE_COVARIANCE = np.diag([0.08**2, 0.08**2, math.radians(5)**2])
# POSE_GATE_CHI2 = 11.345  # 99% chi-square threshold for 3 dimensions
# GHOST_BOX_SIZE = np.array([0.14, 0.14, 0.05])
# GHOST_BOX_COLOR = np.array([0.0, 1.0, 0.0, 0.35], dtype=np.float32)
# BASE_VISUAL_CENTER_OFFSET = np.array([-0.064, 0.0])


# def angle_difference(a, b):
#     return math.atan2(math.sin(a - b), math.cos(a - b))


# def correct_from_camera(ekf, camera, helpers, corners, ids):
#     """Fuse a calibrated wall-marker pose into the EKF."""
#     if ids is None:
#         return 0
#     for marker_corners, marker_id in zip(corners, np.asarray(ids).reshape(-1)):
#         try:
#             measurement = helpers.robot_pose_from_wall_marker(
#                 camera, int(marker_id), marker_corners,
#                 helpers.marker_size_for(int(marker_id)),
#             )
#             if measurement is None:
#                 continue
#             ekf.fuse_pose(measurement, CAMERA_POSE_COVARIANCE, POSE_GATE_CHI2)
#             return int(bool(ekf.last_accepted))
#         except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
#             print(f"Camera EKF correction skipped for ID {int(marker_id)}: {exc}")
#     return 0


# def add_ghost_box(scene, pose):
#     """Render a translucent box at an EKF pose in the viewer user scene."""
#     index = scene.ngeom
#     scene.ngeom += 1
#     yaw = float(pose[2])
#     cosine, sine = math.cos(yaw), math.sin(yaw)
#     rotation = np.array(
#         [cosine, -sine, 0.0, sine, cosine, 0.0, 0.0, 0.0, 1.0],
#         dtype=np.float32,
#     )
#     offset = np.array(
#         [
#             cosine * BASE_VISUAL_CENTER_OFFSET[0]
#             - sine * BASE_VISUAL_CENTER_OFFSET[1],
#             sine * BASE_VISUAL_CENTER_OFFSET[0]
#             + cosine * BASE_VISUAL_CENTER_OFFSET[1],
#         ]
#     )
#     mujoco.mjv_initGeom(
#         scene.geoms[index],
#         type=mujoco.mjtGeom.mjGEOM_BOX,
#         size=GHOST_BOX_SIZE,
#         pos=[float(pose[0] + offset[0]), float(pose[1] + offset[1]), 0.05],
#         mat=rotation,
#         rgba=GHOST_BOX_COLOR,
#     )


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--no-ghost",
#         action="store_true",
#         help="disable EKF ghost-box rendering",
#     )
#     parser.add_argument(
#         "--random-scene",
#         action="store_true",
#         help="generate a randomized obstacle scene before starting",
#     )
#     parser.add_argument(
#         "--scene-seed",
#         type=int,
#         default=0,
#         help="seed for randomized obstacle placement",
#     )
#     parser.add_argument(
#         "--obstacle-count",
#         type=int,
#         default=3,
#         help="number of randomized obstacles",
#     )
#     parser.add_argument("--scene", type=str, default=os.path.join(
#         PROJECT_ROOT, "assets", "scene_turtlebot3_waffle_pi.xml"
#     ))
#     parser.add_argument(
#         "--docking-mode",
#         choices=("pose-visual", "legacy-pixel"),
#         default="pose-visual",
#         help="pose pre-dock then visual servo, or the previous pixel-following logic",
#     )
#     args = parser.parse_args()
#     xml_path = os.path.abspath(args.scene)
#     if not os.path.isfile(xml_path):
#         raise FileNotFoundError(f"Missing XML: {xml_path}")
#     model = mujoco.MjModel.from_xml_path(xml_path)
#     data = mujoco.MjData(model)
#     if args.random_scene:
#         randomize_scene(model, data, args.scene_seed, args.obstacle_count)
#     track_helpers = load_track_helpers()
#     camera = RaspiCamera(model)
#     encoder = WheelEncoder(model)
#     ekf = RobotEKF(
#         model=model,
#         initial_state=INITIAL_POSE_ESTIMATE.copy(),
#         initial_covariance=INITIAL_COVARIANCE.copy(),
#     )
#     diff_drive = DiffDriveController()
#     follower = TagFollower(
#         target_ids=(COARSE_ID,),
#         image_width=camera.width,
#         target_width_pixels=COARSE_TARGET_WIDTH_PX,
#     )
#     visual_docking = VisualDockingController(DockingConfig(
#         target_width_px=DOCK_TARGET_WIDTH_PX,
#         max_linear_v=0.06,
#         max_angular_w=0.18,
#     ))
#     if model.nu < 2:
#         raise RuntimeError("Expected two wheel actuators")

#     lidar_ids = []
#     for angle in ANGLES:
#         name = f"lidar_{angle}"
#         sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
#         if sensor_id < 0 or model.sensor_dim[sensor_id] != 1:
#             raise RuntimeError(f"Missing or invalid LiDAR sensor: {name}")
#         lidar_ids.append(sensor_id)

#     print("Camera EKF correction ENABLED")

#     last_cam_time = -CAM_INTERVAL
#     last_status_time = -STATUS_INTERVAL
#     last_dock_debug = -STATUS_INTERVAL
#     desired_v = desired_w = 0.0
#     mode = "FOLLOW"
#     turn_start_heading = 0.0
#     turn_direction = 1
#     forward_start = np.zeros(2)
#     docking_phase = False
#     dock_phase = "TRACK_TAG"
#     charged = False
#     id99_seen_frames = 0
#     id99_lost_frames = 0
#     id99_reacquire_frames = 0
#     last_id99_signed_error = 0.0
#     dock_confirm_count = 0
#     total_camera_updates = 0
#     last_message_mode = None
#     legacy_straight_approach = False

#     def lidar_distances():
#         result = []
#         for sensor_id in lidar_ids:
#             reading = float(data.sensordata[model.sensor_adr[sensor_id]])
#             result.append(reading if np.isfinite(reading) and reading >= 0 else math.inf)
#         return np.asarray(result)

#     def sector_min(readings, lo, hi):
#         return float(np.min(readings[(ANGLES >= lo) & (ANGLES <= hi)]))

#     def control_loop(model, data):
#         nonlocal last_cam_time, last_status_time, last_dock_debug
#         nonlocal desired_v, desired_w, mode, turn_start_heading, turn_direction
#         nonlocal forward_start, docking_phase, dock_phase, charged, id99_seen_frames
#         nonlocal id99_lost_frames, id99_reacquire_frames, last_id99_signed_error
#         nonlocal dock_confirm_count, total_camera_updates, last_message_mode
#         nonlocal legacy_straight_approach

#         # Encoder-based EKF prediction; do not use simulator true pose to navigate.
#         d_left, d_right, _, _ = encoder.step(data)
#         ekf.predict(d_left, d_right)
#         position = ekf.state[:2].copy()
#         heading = float(ekf.state[2])

#         readings = lidar_distances()
#         front = sector_min(readings, -30, 30)
#         left = sector_min(readings, 45, 90)
#         right = sector_min(readings, -90, -45)

#         if charged:
#             data.ctrl[0] = data.ctrl[1] = 0.0
#             return

#         if (
#             docking_phase
#             and not charged
#             and np.isfinite(front)
#             and front <= DOCK_HARD_STOP
#             and (
#                 legacy_straight_approach
#                 or visual_docking.straight_approach
#             )
#         ):
#             charged = True
#             mode = "CHARGED"
#             follower.state = "CHARGED"
#             desired_v = desired_w = 0.0
#             print(f"[{data.time:.2f}s] DOCK RANGE REACHED; charging")
#             data.ctrl[0] = data.ctrl[1] = 0.0
#             return

#         if (
#             mode == "FOLLOW"
#             and (not docking_phase or dock_phase == "PRE_DOCK")
#             and desired_v > 0
#             and front < CLEAR_DISTANCE
#         ):
#             mode = "AVOID_TURN"
#             turn_start_heading = heading
#             turn_direction = 1 if left >= right else -1

#         if mode == "AVOID_TURN":
#             turned = turn_direction * angle_difference(heading, turn_start_heading)
#             if turned >= AVOID_TURN_RAD:
#                 mode = "AVOID_FORWARD"
#                 forward_start = position.copy()
#                 desired_v = desired_w = 0.0
#             else:
#                 desired_v, desired_w = 0.0, AVOID_TURN_W * turn_direction
#         elif mode == "AVOID_FORWARD":
#             travelled = float(np.linalg.norm(position - forward_start))
#             if front < STOP_DISTANCE:
#                 mode = "AVOID_TURN"
#                 turn_start_heading = heading
#                 turn_direction = 1 if left >= right else -1
#                 desired_v = desired_w = 0.0
#             elif travelled >= AVOID_DISTANCE:
#                 mode = "FOLLOW"
#                 desired_v = desired_w = 0.0
#             else:
#                 desired_v, desired_w = AVOID_FORWARD_V, 0.0

#         if data.time - last_cam_time >= CAM_INTERVAL:
#             last_cam_time = float(data.time)
#             frame = camera.capture_cv_frame(data)
#             corners, ids, annotated = camera.detect_markers(frame)

#             # Always populate visible tags, even when EKF landmark map is empty.
#             visible = {}
#             if ids is not None:
#                 for i, raw_id in enumerate(np.asarray(ids).reshape(-1)):
#                     marker_id = int(raw_id)
#                     marker = np.asarray(corners[i]).reshape(4, 2)
#                     width = float(np.linalg.norm(marker[1] - marker[0]))
#                     center_x = float(np.mean(marker[:, 0]))
#                     signed_error = (center_x - camera.width / 2) / (camera.width / 2)
#                     visible[marker_id] = (width, abs(signed_error), signed_error)

#             total_camera_updates += correct_from_camera(
#                 ekf, camera, track_helpers, corners, ids
#             )
#             position = ekf.state[:2].copy()
#             heading = float(ekf.state[2])

#             dock_tag_visible = any(tag_id in visible for tag_id in (COARSE_ID, DOCK_ID))
#             if not docking_phase and dock_tag_visible:
#                 id99_seen_frames = id99_seen_frames + 1 if DOCK_ID in visible else 0
#                 if id99_seen_frames >= ID99_CONFIRM_FRAMES or COARSE_ID in visible:
#                     docking_phase = True
#                     dock_phase = (
#                         "PRE_DOCK"
#                         if args.docking_mode == "pose-visual"
#                         else "LEGACY_PIXEL"
#                     )
#                     if dock_phase == "LEGACY_PIXEL":
#                         follower.target_ids = (DOCK_ID,)
#                         follower.target_id = DOCK_ID
#                         follower.target_width_pixels = DOCK_TARGET_WIDTH_PX
#                         legacy_straight_approach = False
#                     follower.state = "TRACKING"
#                     print(f"[{data.time:.2f}s] DOCK TAG -> {dock_phase}")

#             id99_visible = DOCK_ID in visible and visible[DOCK_ID][0] >= ID99_SWITCH_MIN_WIDTH_PX
#             if docking_phase:
#                 if id99_visible or COARSE_ID in visible:
#                     id99_lost_frames = 0
#                     id99_reacquire_frames += 1
#                     last_id99_signed_error = visible[DOCK_ID if id99_visible else COARSE_ID][2]
#                 else:
#                     id99_lost_frames += 1
#                     id99_reacquire_frames = 0
#                     dock_confirm_count = 0

#             if mode == "FOLLOW":
#                 if not docking_phase:
#                     desired_v, desired_w = follower.compute_commands(corners, ids, heading)
#                     if desired_v > 0 and front < CLEAR_DISTANCE:
#                         mode = "AVOID_TURN"
#                         turn_start_heading = heading
#                         turn_direction = 1 if left >= right else -1
#                         desired_v = desired_w = 0.0
#                 elif dock_phase == "LEGACY_PIXEL":
#                     if legacy_straight_approach:
#                         desired_v = 0.025 if front > DOCK_HARD_STOP else 0.0
#                         desired_w = 0.0
#                         dock_confirm_count = (
#                             dock_confirm_count + 1
#                             if front <= DOCK_COMPLETE_DISTANCE
#                             else 0
#                         )
#                         if dock_confirm_count >= DOCK_CONFIRM_FRAMES:
#                             charged = True
#                             mode = "CHARGED"
#                             follower.state = "CHARGED"
#                             desired_v = desired_w = 0.0
#                             print(
#                                 f"[{data.time:.2f}s] LEGACY PIXEL DOCK COMPLETE; "
#                                 "motors stopped"
#                             )
#                     elif not id99_visible or id99_reacquire_frames < ID99_REACQUIRE_FRAMES:
#                         desired_v = 0.0
#                         if id99_lost_frames <= ID99_LOST_FRAMES:
#                             desired_w = 0.0
#                         else:
#                             desired_w = (
#                                 -ID99_SEARCH_W if last_id99_signed_error > 0.05
#                                 else ID99_SEARCH_W
#                             )
#                     else:
#                         width, center_error, signed_error = visible[DOCK_ID]
#                         desired_w = 0.0 if center_error <= VISUAL_CENTER_DEADBAND else float(np.clip(
#                             -follower.Kp_angular * signed_error, -0.25, 0.25
#                         ))
#                         if center_error > ID99_CENTER_BEFORE_DRIVE:
#                             desired_v = 0.0
#                         elif width < DOCK_TARGET_WIDTH_PX:
#                             desired_v = float(np.clip(
#                                 follower.Kp_linear
#                                 * (DOCK_TARGET_WIDTH_PX - width)
#                                 / DOCK_TARGET_WIDTH_PX,
#                                 0.0,
#                                 0.08,
#                             ))
#                         else:
#                             desired_v = 0.0
#                         if (
#                             width >= DOCK_TARGET_WIDTH_PX
#                             and center_error <= DOCK_CENTER_TOLERANCE
#                         ):
#                             legacy_straight_approach = True
#                             desired_v = 0.025 if front > DOCK_HARD_STOP else 0.0

#                         reached = (
#                             DOCK_HARD_STOP <= front <= DOCK_COMPLETE_DISTANCE
#                             and width >= DOCK_MIN_MARKER_WIDTH_PX
#                         )
#                         aligned = center_error <= DOCK_CENTER_TOLERANCE
#                         dock_confirm_count = (
#                             dock_confirm_count + 1 if reached and aligned else 0
#                         )
#                         if data.time - last_dock_debug >= STATUS_INTERVAL:
#                             print(
#                                 f"[{data.time:.2f}s] LEGACY ID99 width={width:.1f}px "
#                                 f"center_error={center_error:.3f} "
#                                 f"confirm={dock_confirm_count}/{DOCK_CONFIRM_FRAMES}"
#                             )
#                             last_dock_debug = float(data.time)
#                         if dock_confirm_count >= DOCK_CONFIRM_FRAMES:
#                             charged = True
#                             mode = "CHARGED"
#                             follower.state = "CHARGED"
#                             desired_v = desired_w = 0.0
#                             print(
#                                 f"[{data.time:.2f}s] LEGACY PIXEL DOCK COMPLETE; "
#                                 "motors stopped"
#                             )
#                 elif dock_phase == "PRE_DOCK":
#                     desired_v, desired_w, pre_dock_reached = pose_command(
#                         position.tolist() + [heading],
#                         PRE_DOCK_POSE,
#                         PRE_DOCK_MAX_V,
#                         PRE_DOCK_MAX_W,
#                     )
#                     if pre_dock_reached:
#                         dock_phase = "VISUAL_APPROACH"
#                         visual_docking.reset()
#                         desired_v = desired_w = 0.0
#                         print(f"[{data.time:.2f}s] PRE_DOCK reached; VISUAL_APPROACH")
#                     if desired_v > 0.0 and front < CLEAR_DISTANCE:
#                         mode = "AVOID_TURN"
#                         turn_start_heading = heading
#                         turn_direction = 1 if left >= right else -1
#                         desired_v = desired_w = 0.0
#                 elif dock_phase == "VISUAL_APPROACH":
#                     if visual_docking.straight_approach:
#                         desired_v, desired_w, _ = visual_docking.command(
#                             None, None, None, front
#                         )
#                     elif id99_visible:
#                         width, center_error, signed_error = visible[DOCK_ID]
#                         desired_w = (
#                             0.0
#                             if center_error <= VISUAL_CENTER_DEADBAND
#                             else float(np.clip(
#                                 -follower.Kp_angular * signed_error,
#                                 -0.25,
#                                 0.25,
#                             ))
#                         )
#                         if center_error > ID99_CENTER_BEFORE_DRIVE:
#                             desired_v = 0.0
#                         elif width < DOCK_TARGET_WIDTH_PX:
#                             desired_v = float(np.clip(
#                                 follower.Kp_linear
#                                 * (DOCK_TARGET_WIDTH_PX - width)
#                                 / DOCK_TARGET_WIDTH_PX,
#                                 0.0,
#                                 0.08,
#                             ))
#                         else:
#                             visual_docking.start_straight_approach()
#                             desired_v = 0.025 if front > DOCK_HARD_STOP else 0.0
#                             desired_w = 0.0
#                     else:
#                         desired_v = 0.0
#                         desired_w = (
#                             -ID99_SEARCH_W if last_id99_signed_error > 0.05
#                             else ID99_SEARCH_W
#                         )

#                 if docking_phase and not charged and front < DOCK_HARD_STOP:
#                     desired_v = desired_w = 0.0

#             sigma = np.sqrt(np.maximum(np.diag(ekf.P), 0.0))
#             cv2.putText(annotated,
#                         "DOCKED (SIMULATED)" if charged else
#                         f"Mode: {mode}/{follower.state} target={DOCK_ID if docking_phase else COARSE_ID}",
#                         (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
#             cv2.putText(annotated, f"EKF yaw: {math.degrees(heading):.1f} deg",
#                         (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
#             cv2.putText(annotated, f"Front: {front:.2f} m",
#                         (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
#             cv2.putText(annotated, f"EKF x,y: {position[0]:.2f}, {position[1]:.2f} m",
#                         (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
#             cv2.putText(annotated,
#                         f"Sigma: x={sigma[0]:.2f} y={sigma[1]:.2f} yaw={math.degrees(sigma[2]):.1f}deg",
#                         (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
#             cv2.putText(annotated, f"Camera EKF corrections: {total_camera_updates}",
#                         (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
#             cv2.imshow("TurtleBot3 - Tag + LiDAR + EKF", annotated)
#             cv2.waitKey(1)

#         if mode != last_message_mode:
#             print(f"[{data.time:.2f}s] {mode} | front={front:.2f}m")
#             last_message_mode = mode
#         if data.time - last_status_time >= STATUS_INTERVAL:
#             last_status_time = float(data.time)
#             print(f"[{data.time:.1f}s] EKF x={position[0]:.3f}, y={position[1]:.3f}, "
#                   f"yaw={math.degrees(heading):.1f}deg; "
#                   f"sigma_xy={np.sqrt(np.maximum(np.diag(ekf.P)[:2], 0.0))}")

#         left_wheel, right_wheel = diff_drive.twist_to_wheel_velocities(desired_v, desired_w)
#         data.ctrl[0] = float(np.clip(left_wheel, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))
#         data.ctrl[1] = float(np.clip(right_wheel, -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED))

#     def run_with_ghost():
#         with mujoco.viewer.launch_passive(model, data) as viewer:
#             while viewer.is_running():
#                 start = time.time()
#                 mujoco.mj_step(model, data)
#                 viewer.user_scn.ngeom = 0
#                 add_ghost_box(viewer.user_scn, ekf.state)
#                 viewer.sync()
#                 time.sleep(
#                     max(0.0, model.opt.timestep - (time.time() - start))
#                 )

#     try:
#         mujoco.set_mjcb_control(control_loop)
#         if args.no_ghost:
#             mujoco.viewer.launch(model, data)
#         else:
#             run_with_ghost()
#     except KeyboardInterrupt:
#         print("Simulation interrupted")
#     finally:
#         mujoco.set_mjcb_control(None)
#         cv2.destroyAllWindows()
#         camera.renderer.close()


# if __name__ == "__main__":
#     main()