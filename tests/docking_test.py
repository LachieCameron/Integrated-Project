
#!/usr/bin/env python3
import os
import sys
import math
import time
import importlib.util

import cv2
import mujoco
import mujoco.viewer
import numpy as np


# ============================================================
# PROJECT IMPORTS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.sensing.cam import RaspiCamera
from src.sensing.IMU import IMUSensor
from src.control.Tag_follow import (
    DiffDriveController,
    TagFollower,
)
from src.estimation.ekf import RobotEKF
from src.sensing.wheel_encoder import WheelEncoder


def load_track_helpers():
    path = os.path.join(PROJECT_ROOT, "scripts", "run_track_a.py")
    spec = importlib.util.spec_from_file_location("run_track_a", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Track A helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================
# CONFIGURATION
# ============================================================

CAM_INTERVAL = 1.0 / 30.0
DEBUG_PRINT_INTERVAL = 1.0

COARSE_ID = 0
FINE_ID = 99

# ID 0 must reach this size before ID 99 is considered.
COARSE_SWITCH_WIDTH = 120.0

# ID 99 must be detected reliably before switching.
FINE_SWITCH_WIDTH = 60.0
FINE_CONFIRM_FRAMES = 5

# Final charging conditions.
FINE_STOP_WIDTH = 125.0
FINE_ALIGNMENT_TOL = 0.12
CHARGE_CONFIRM_FRAMES = 3

# Set this to match the robot's actual spawn orientation.
INITIAL_HEADING = math.pi / 2

# Obstacle avoidance.
STOP_DISTANCE = 0.20
CLEAR_DISTANCE = 0.40

AVOID_DISTANCE = 0.8
AVOID_TURN_RAD = math.radians(35)
AVOID_TURN_W = 0.50
AVOID_FORWARD_V = 0.12

# Fine docking speed limits.
FINE_MAX_V = 0.08
FINE_MAX_W = 0.45

DOCK_SAFETY_DISTANCE = 0.12

MAX_WHEEL_SPEED = 5.0
WHEEL_RADIUS = 0.033
BASE_VISUAL_CENTER_X = -0.064
BASE_VISUAL_CENTER_Y = 0.0

ANGLES = np.arange(-90, 91, 15)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def wrap_angle(angle):
    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


def angle_difference(a, b):
    return wrap_angle(a - b)


def marker_measurement(corners, ids, target_id, image_width):
    """
    Return:
        width: apparent marker width in pixels
        error: normalized horizontal centre error
    """

    if ids is None or corners is None:
        return 0.0, None

    for index, marker_id in enumerate(
        np.asarray(ids).reshape(-1)
    ):
        if int(marker_id) != target_id:
            continue

        points = np.asarray(
            corners[index],
            dtype=float,
        ).reshape(4, 2)

        top = np.linalg.norm(
            points[1] - points[0]
        )

        bottom = np.linalg.norm(
            points[2] - points[3]
        )

        width = float((top + bottom) / 2.0)

        centre_x = float(np.mean(points[:, 0]))
        image_centre = image_width / 2.0

        error = (
            centre_x - image_centre
        ) / image_centre

        return width, error

    return 0.0, None


def filter_target(corners, ids, target_id):
    """
    Pass ONLY the active marker to TagFollower.

    Other detected markers are ignored by navigation.
    """

    if ids is None or corners is None:
        return [], None

    selected_corners = []

    for index, marker_id in enumerate(
        np.asarray(ids).reshape(-1)
    ):
        if int(marker_id) == target_id:
            selected_corners.append(corners[index])

    if not selected_corners:
        return [], None

    selected_ids = np.full(
        (len(selected_corners), 1),
        target_id,
        dtype=np.int32,
    )

    return selected_corners, selected_ids


def draw_charged_banner(image):
    height, width = image.shape[:2]

    x1 = max(10, width // 2 - 230)
    x2 = min(width - 10, width // 2 + 230)

    y1 = min(205, height - 100)
    y2 = min(y1 + 85, height - 10)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 145, 0),
        -1,
    )

    cv2.putText(
        image,
        "ROBOT CHARGED",
        (x1 + 30, y1 + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (255, 255, 255),
        3,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-ghost", action="store_true", help="show the full robot ghost")
    parser.add_argument("--no-ghost", action="store_true", help="disable ghost rendering")
    args = parser.parse_args()

    xml_path = os.path.join(
        PROJECT_ROOT,
        "assets",
        "scene_turtlebot3_waffle_pi.xml",
    )

    if not os.path.isfile(xml_path):
        raise FileNotFoundError(
            f"Missing XML: {xml_path}"
        )

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    camera = RaspiCamera(model)
    imu = IMUSensor(model)
    track_helpers = load_track_helpers()
    diff_drive = DiffDriveController()

    follower = TagFollower(
        target_id=COARSE_ID,
        image_width=camera.width,
    )

    # ========================================================
    # SENSOR AND ACTUATOR LOOKUP
    # ========================================================

    lidar_ids = []

    for angle in ANGLES:

        name = f"lidar_{int(angle)}"

        sid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            name,
        )

        if sid < 0 or model.sensor_dim[sid] != 1:
            raise RuntimeError(
                f"Missing or invalid LiDAR: {name}"
            )

        lidar_ids.append(sid)

    def find_sensor(name):

        sid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            name,
        )

        if sid < 0:
            raise RuntimeError(
                f"Missing sensor: {name}"
            )

        if model.sensor_dim[sid] != 1:
            raise RuntimeError(
                f"Invalid sensor dimension: {name}"
            )

        return sid

    left_encoder_id = find_sensor(
        "wheel_left_encoder"
    )

    right_encoder_id = find_sensor(
        "wheel_right_encoder"
    )

    left_actuator = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "wheel_left",
    )

    right_actuator = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "wheel_right",
    )

    if left_actuator < 0 or right_actuator < 0:
        raise RuntimeError(
            "Missing wheel_left or wheel_right actuator"
        )

    # ========================================================
    # ROBOT STATE
    # ========================================================

    initial_robot_yaw = 2.0 * math.atan2(float(data.qpos[6]), float(data.qpos[3]))
    heading = INITIAL_HEADING
    ghost_encoder = WheelEncoder(model)
    ghost_ekf = RobotEKF(
        model=model,
        initial_state=np.array(
            [float(data.qpos[0]), float(data.qpos[1]), initial_robot_yaw]
        ),
    )
    pending_camera_pose = None

    last_sim_time = float(data.time)
    last_cam_time = -CAM_INTERVAL
    last_debug_time = -DEBUG_PRINT_INTERVAL

    desired_v = 0.0
    desired_w = 0.0

    mode = "FOLLOW"
    docking_phase = "COARSE"

    # Coarse approach must complete before fine detection
    # is allowed to trigger a target change.
    coarse_ready = False

    fine_detected_frames = 0
    charge_detected_frames = 0

    fine_locked = False
    robot_charged = False

    turn_start_heading = heading
    turn_direction = 1

    travelled_distance = 0.0
    avoid_start_distance = 0.0

    last_left_encoder = None
    last_right_encoder = None

    last_annotated = None
    last_message_mode = None

    # ========================================================
    # LIDAR HELPERS
    # ========================================================

    def lidar_distances():

        values = []

        for sid in lidar_ids:

            value = float(
                data.sensordata[
                    model.sensor_adr[sid]
                ]
            )

            if not np.isfinite(value) or value < 0:
                value = math.inf

            values.append(value)

        return np.asarray(values)

    def sector_min(distances, low, high):

        mask = (
            (ANGLES >= low)
            & (ANGLES <= high)
        )

        return float(np.min(distances[mask]))

    # ========================================================
    # CONTROL CALLBACK
    # ========================================================

    def control_loop(model, data):

        nonlocal follower
        nonlocal heading
        nonlocal last_sim_time, last_cam_time
        nonlocal last_debug_time
        nonlocal desired_v, desired_w
        nonlocal mode, docking_phase
        nonlocal coarse_ready
        nonlocal fine_detected_frames
        nonlocal charge_detected_frames
        nonlocal fine_locked, robot_charged
        nonlocal turn_start_heading, turn_direction
        nonlocal travelled_distance, avoid_start_distance
        nonlocal last_left_encoder, last_right_encoder
        nonlocal last_annotated, last_message_mode
        nonlocal ghost_encoder, ghost_ekf
        nonlocal pending_camera_pose

        now = float(data.time)

        # ====================================================
        # SIMULATION RESET
        # ====================================================

        if now < last_sim_time - 1e-9:

            follower = TagFollower(
                target_id=COARSE_ID,
                image_width=camera.width,
            )

            heading = INITIAL_HEADING

            last_cam_time = now - CAM_INTERVAL
            last_debug_time = now - DEBUG_PRINT_INTERVAL

            desired_v = 0.0
            desired_w = 0.0

            mode = "FOLLOW"
            docking_phase = "COARSE"

            coarse_ready = False
            fine_detected_frames = 0
            charge_detected_frames = 0

            fine_locked = False
            robot_charged = False

            turn_start_heading = heading
            turn_direction = 1

            travelled_distance = 0.0
            avoid_start_distance = 0.0

            last_left_encoder = None
            last_right_encoder = None
            ghost_encoder = WheelEncoder(model)
            ghost_ekf = RobotEKF(
                model=model,
                initial_state=np.array(
                    [float(data.qpos[0]), float(data.qpos[1]), initial_robot_yaw]
                ),
            )
            pending_camera_pose = None

            last_annotated = None
            last_message_mode = None

        dt = max(0.0, now - last_sim_time)
        last_sim_time = now
        # ====================================================
        # ROBOT CHARGED: STOP ALL MOVEMENT
        # ====================================================

        if robot_charged:

            desired_v = 0.0
            desired_w = 0.0

            data.ctrl[left_actuator] = 0.0
            data.ctrl[right_actuator] = 0.0

            if now - last_cam_time >= CAM_INTERVAL:

                last_cam_time = now

                if last_annotated is not None:

                    display = last_annotated.copy()
                    draw_charged_banner(display)

                    cv2.imshow(
                        "TurtleBot3 - Two Stage Docking",
                        display,
                    )

                    cv2.waitKey(1)

            return

        # ====================================================
        # IMU HEADING
        # ====================================================

        _, gyro = imu.read(data)

        gyro_z = float(gyro[2])

        if abs(gyro_z) > 0.001:
            heading += gyro_z * dt

        heading = wrap_angle(heading)

        # ====================================================
        # WHEEL ENCODER DISTANCE
        # ====================================================

        left_encoder = float(
            data.sensordata[
                model.sensor_adr[left_encoder_id]
            ]
        )

        right_encoder = float(
            data.sensordata[
                model.sensor_adr[right_encoder_id]
            ]
        )

        if (
            last_left_encoder is not None
            and last_right_encoder is not None
        ):

            dl = (
                left_encoder - last_left_encoder
            ) * WHEEL_RADIUS

            dr = (
                right_encoder - last_right_encoder
            ) * WHEEL_RADIUS

            ds = (dl + dr) / 2.0

            travelled_distance += abs(ds)

        last_left_encoder = left_encoder
        last_right_encoder = right_encoder

        # ====================================================
        # LIDAR
        # ====================================================

        distances = lidar_distances()

        front = sector_min(
            distances, -30, 30
        )

        left = sector_min(
            distances, 45, 90
        )

        right = sector_min(
            distances, -90, -45
        )

        # ====================================================
        # OBSTACLE AVOIDANCE
        # ====================================================

        if (
            mode == "FOLLOW"
            and docking_phase == "COARSE"
            and desired_v > 0
            and front < STOP_DISTANCE
        ):

            mode = "AVOID_TURN"

            turn_start_heading = heading

            turn_direction = (
                1 if left >= right else -1
            )

        if mode == "AVOID_TURN":

            turned = (
                turn_direction
                * angle_difference(
                    heading,
                    turn_start_heading,
                )
            )

            if turned >= AVOID_TURN_RAD:

                mode = "AVOID_FORWARD"

                avoid_start_distance = travelled_distance

                desired_v = 0.0
                desired_w = 0.0

            else:

                desired_v = 0.0

                desired_w = (
                    AVOID_TURN_W * turn_direction
                )

        elif mode == "AVOID_FORWARD":

            moved = (
                travelled_distance
                - avoid_start_distance
            )

            if front < STOP_DISTANCE:

                mode = "AVOID_TURN"

                turn_start_heading = heading

                turn_direction = (
                    1 if left >= right else -1
                )

                desired_v = 0.0
                desired_w = 0.0

            elif moved >= AVOID_DISTANCE:

                mode = "FOLLOW"

                desired_v = 0.0
                desired_w = 0.0

            else:

                desired_v = AVOID_FORWARD_V
                desired_w = 0.0

        # ====================================================
        # CAMERA PROCESSING
        # ====================================================

        if now - last_cam_time >= CAM_INTERVAL:

            last_cam_time = now

            frame = camera.capture_cv_frame(data)

            corners, ids, annotated = (
                camera.detect_markers(frame)
            )

            detected_ids = (
                []
                if ids is None
                else [
                    int(value)
                    for value in np.asarray(ids).reshape(-1)
                ]
            )

            pending_camera_pose = None
            if ids is not None:
                for marker_corners, marker_id in zip(
                    corners, np.asarray(ids).reshape(-1)
                ):
                    measurement = track_helpers.robot_pose_from_wall_marker(
                        camera, int(marker_id), marker_corners, 0.4
                    )
                    if measurement is not None:
                        pending_camera_pose = measurement
                        break

            coarse_width, coarse_error = marker_measurement(
                corners,
                ids,
                COARSE_ID,
                camera.width,
            )

            fine_width, fine_error = marker_measurement(
                corners,
                ids,
                FINE_ID,
                camera.width,
            )

            # =================================================
            # STAGE 1: APPROACH ID 0 FIRST
            # =================================================

            if docking_phase == "COARSE":

                # ID 99 is NOT allowed to become the target
                # until ID 0 reaches the coarse threshold.

                if (
                    not coarse_ready
                    and coarse_width >= COARSE_SWITCH_WIDTH
                ):

                    coarse_ready = True

                    print(
                        f"[{now:.2f}s] "
                        f"ID 0 approach complete. "
                        f"Width={coarse_width:.1f}px. "
                        "ID 99 handoff enabled."
                    )

                # Only check ID 99 after coarse approach.
                if coarse_ready and not fine_locked:

                    if fine_width >= FINE_SWITCH_WIDTH:
                        fine_detected_frames += 1
                    else:
                        fine_detected_frames = 0

                    if (
                        fine_detected_frames
                        >= FINE_CONFIRM_FRAMES
                    ):

                        fine_locked = True
                        docking_phase = "FINE"

                        follower.target_id = FINE_ID
                        follower.state = "TRACKING"
                        follower.target_width_px = FINE_STOP_WIDTH

                        mode = "FOLLOW"

                        desired_v = 0.0
                        desired_w = 0.0

                        print(
                            f"[{now:.2f}s] "
                            "TARGET SWITCH: "
                            "ID 0 -> ID 99"
                        )

            # =================================================
            # FILTER: ONLY ACTIVE TARGET REACHES FOLLOWER
            # =================================================

            active_id = (
                FINE_ID
                if fine_locked
                else COARSE_ID
            )

            tracking_corners, tracking_ids = filter_target(
                corners,
                ids,
                active_id,
            )

            # =================================================
            # STAGE 2: FOLLOW ACTIVE TARGET
            # =================================================

            if mode == "FOLLOW":

                desired_v, desired_w = (
                    follower.compute_commands(
                        tracking_corners,
                        tracking_ids,
                        heading,
                    )
                )

                if docking_phase == "COARSE":

                    if (
                        desired_v > 0
                        and front < STOP_DISTANCE
                    ):

                        mode = "AVOID_TURN"

                        turn_start_heading = heading

                        turn_direction = (
                            1 if left >= right else -1
                        )

                        desired_v = 0.0
                        desired_w = 0.0

                    elif (
                        desired_v > 0
                        and front < CLEAR_DISTANCE
                    ):

                        desired_v = min(
                            desired_v,
                            AVOID_FORWARD_V,
                        )

                elif docking_phase == "FINE":

                    # Limit speed for fine docking.
                    desired_v = float(
                        np.clip(
                            desired_v,
                            0.0,
                            FINE_MAX_V,
                        )
                    )

                    desired_w = float(
                        np.clip(
                            desired_w,
                            -FINE_MAX_W,
                            FINE_MAX_W,
                        )
                    )

                    # =========================================
                    # CHARGING CONDITIONS
                    # =========================================

                    close_enough = (
                        fine_width >= FINE_STOP_WIDTH
                    )

                    aligned = (
                        fine_error is not None
                        and abs(fine_error)
                        <= FINE_ALIGNMENT_TOL
                    )

                    if close_enough and aligned:

                        charge_detected_frames += 1

                    else:

                        charge_detected_frames = 0

                    if (
                        charge_detected_frames
                        >= CHARGE_CONFIRM_FRAMES
                    ):

                        robot_charged = True
                        docking_phase = "ROBOT CHARGED"
                        mode = "CHARGED"

                        desired_v = 0.0
                        desired_w = 0.0

                        print()
                        print("=" * 50)
                        print("                ROBOT CHARGED")
                        print("       Charging station reached")
                        print("       All robot movement stopped")
                        print("=" * 50)
                        print()

                    elif front < DOCK_SAFETY_DISTANCE:

                        desired_v = 0.0
                        desired_w = 0.0

            # =================================================
            # CAMERA DISPLAY
            # =================================================

            cv2.putText(
                annotated,
                f"Mode: {mode}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated,
                f"Phase: {docking_phase}",
                (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                annotated,
                f"Target ID: {active_id}",
                (15, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated,
                f"Detected: {detected_ids}",
                (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 0),
                1,
            )

            cv2.putText(
                annotated,
                f"ID0: {coarse_width:.1f}px "
                f"ID99: {fine_width:.1f}px",
                (15, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
            )

            cv2.putText(
                annotated,
                f"ID99 enabled: {coarse_ready}",
                (15, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 0),
                1,
            )

            cv2.putText(
                annotated,
                f"Front: {front:.2f}m",
                (15, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 255),
                1,
            )

            cv2.putText(
                annotated,
                f"Yaw: {math.degrees(heading):.1f} deg",
                (15, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
            )

            last_annotated = annotated.copy()

            if robot_charged:
                draw_charged_banner(annotated)

            cv2.imshow(
                "TurtleBot3 - Two Stage Docking",
                annotated,
            )

            cv2.waitKey(1)

            if (
                now - last_debug_time
                >= DEBUG_PRINT_INTERVAL
            ):

                print(
                    f"[{now:.2f}s] "
                    f"Detected={detected_ids} "
                    f"Target={active_id} "
                    f"Phase={docking_phase} "
                    f"Mode={mode} "
                    f"ID0={coarse_width:.1f}px "
                    f"ID99={fine_width:.1f}px "
                    f"CoarseReady={coarse_ready} "
                    f"ChargeConfirm={charge_detected_frames}/"
                    f"{CHARGE_CONFIRM_FRAMES}"
                )

                last_debug_time = now

        # ====================================================
        # APPLY WHEEL COMMANDS
        # ====================================================

        if robot_charged:
            desired_v = 0.0
            desired_w = 0.0

        left_wheel, right_wheel = (
            diff_drive.twist_to_wheel_velocities(
                desired_v,
                desired_w,
            )
        )

        data.ctrl[left_actuator] = float(
            np.clip(
                left_wheel,
                -MAX_WHEEL_SPEED,
                MAX_WHEEL_SPEED,
            )
        )

        data.ctrl[right_actuator] = float(
            np.clip(
                right_wheel,
                -MAX_WHEEL_SPEED,
                MAX_WHEEL_SPEED,
            )
        )

    def run_with_ghost(full):
        nonlocal pending_camera_pose
        ghost_data = mujoco.MjData(model)
        ghost_options = mujoco.MjvOption()
        ghost_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        ghost_perturb = mujoco.MjvPerturb()
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                start = time.time()
                mujoco.mj_step(model, data)
                ghost_left, ghost_right, _, _ = ghost_encoder.step(data)
                gyro_z = float(imu.read(data)[1][2])
                ghost_ekf.update(
                    ghost_left,
                    ghost_right,
                    model.opt.timestep,
                    imu_gyro_z=gyro_z,
                    imu_rate_std=0.03,
                )
                if pending_camera_pose is not None:
                    ghost_ekf.fuse_pose(
                        pending_camera_pose,
                        np.diag([0.02, 0.02, 0.05]),
                    )
                    pending_camera_pose = None
                pose = ghost_ekf.state.copy()
                viewer.user_scn.ngeom = 0
                if full:
                    ghost_data.qpos[0:3] = [pose[0], pose[1], 0.01]
                    ghost_data.qpos[3:7] = [
                        math.cos(pose[2] / 2.0), 0.0, 0.0, math.sin(pose[2] / 2.0)
                    ]
                    mujoco.mj_forward(model, ghost_data)
                    mujoco.mjv_addGeoms(
                        model, ghost_data, ghost_options, ghost_perturb,
                        mujoco.mjtCatBit.mjCAT_DYNAMIC, viewer.user_scn
                    )
                else:
                    c, s = math.cos(pose[2]), math.sin(pose[2])
                    viewer.user_scn.ngeom = 1
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_BOX,
                        np.array([0.14, 0.14, 0.05]),
                        np.array([pose[0] + BASE_VISUAL_CENTER_X*c - BASE_VISUAL_CENTER_Y*s,
                                  pose[1] + BASE_VISUAL_CENTER_X*s + BASE_VISUAL_CENTER_Y*c, 0.05]),
                        np.array([c, -s, 0, s, c, 0, 0, 0, 1]),
                        np.array([0, 1, 0, 0.35], dtype=np.float32),
                    )
                viewer.sync()
                time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))

    # ========================================================
    # LAUNCH
    # ========================================================

    try:

        mujoco.set_mjcb_control(control_loop)
        if not args.no_ghost:
            run_with_ghost(args.full_ghost)
        else:
            mujoco.viewer.launch(model, data)

    except KeyboardInterrupt:

        print("Simulation interrupted.")

    finally:

        mujoco.set_mjcb_control(None)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
