#!/usr/bin/env python3
"""Compare wheel-only and wall-ArUco-fused EKF estimates in the viewer.

The green ghost is wheel odometry only. The blue ghost is corrected by wall
markers. A configured wheel scale error makes drift easier to observe.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.estimation.ekf import RobotEKF, wrap_angle
from src.sensing.IMU import IMUSensor
from src.sensing.cam import RaspiCamera
from src.sensing.wheel_encoder import WheelEncoder

# Center of the robot base mesh relative to the free-joint origin in the XML.
BASE_VISUAL_CENTER_OFFSET = np.array([-0.064, 0.0], dtype=float)


def load_track_helpers():
    path = os.path.join(PROJECT_ROOT, "scripts", "run_track_a.py")
    spec = importlib.util.spec_from_file_location("run_track_a", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Track A helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_pose_box(scene, pose, color):
    index = scene.ngeom
    scene.ngeom += 1
    yaw = float(pose[2])
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    offset = np.array(
        [
            c * BASE_VISUAL_CENTER_OFFSET[0] - s * BASE_VISUAL_CENTER_OFFSET[1],
            s * BASE_VISUAL_CENTER_OFFSET[0] + c * BASE_VISUAL_CENTER_OFFSET[1],
        ]
    )
    mujoco.mjv_initGeom(
        scene.geoms[index],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.14, 0.14, 0.05],
        pos=[float(pose[0] + offset[0]), float(pose[1] + offset[1]), 0.05],
        mat=rotation,
        rgba=color,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imu", action="store_true")
    parser.add_argument(
        "--scale-error",
        type=float,
        default=0.0,
        help="relative wheel-scale error to inject for drift testing",
    )
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()
    helpers = load_track_helpers()

    model = mujoco.MjModel.from_xml_path(helpers.SCENE)
    data = mujoco.MjData(model)
    camera = RaspiCamera(model)
    imu = IMUSensor(model) if args.imu else None
    encoder = WheelEncoder(model)
    odom = RobotEKF(model=model)
    fused = RobotEKF(model=model)
    rng = np.random.default_rng(7)
    scales = 1.0 + rng.normal(0.0, args.scale_error, 2)
    dt = model.opt.timestep
    camera_period = max(1, round(1.0 / (30.0 * dt)))
    last_print = -1.0
    accepted = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < args.seconds:
            start = time.time()
            data.ctrl[:] = [-2.0, 3.0]
            mujoco.mj_step(model, data)
            d_left, d_right, _, _ = encoder.step(data)
            d_left *= scales[0]
            d_right *= scales[1]
            gyro_z = None if imu is None else float(imu.read(data)[1][2])
            odom_pose = odom.update(d_left, d_right, dt)
            fused_pose = fused.update(
                d_left,
                d_right,
                dt,
                imu_gyro_z=gyro_z,
                imu_rate_std=0.03 if imu is not None else None,
            )

            if int(round(data.time / dt)) % camera_period == 0:
                frame = camera.capture_cv_frame(data)
                corners, ids, _ = camera.detect_markers(frame)
                if ids is not None:
                    for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
                        measurement = helpers.robot_pose_from_wall_marker(
                            camera, int(marker_id), marker_corners, 0.4
                        )
                        if measurement is not None:
                            accepted_measurement = fused.fuse_pose(
                                measurement, np.diag([0.02, 0.02, 0.05])
                            )
                            if accepted_measurement:
                                accepted += 1
                            fused_pose = fused.state.copy()
                            break

            true_pose = np.array(
                [data.qpos[0], data.qpos[1], 2.0 * math.atan2(data.qpos[6], data.qpos[3])]
            )
            true_pose[2] = wrap_angle(true_pose[2])
            odom_error = odom_pose - true_pose
            fused_error = fused_pose - true_pose
            odom_error[2] = wrap_angle(odom_error[2])
            fused_error[2] = wrap_angle(fused_error[2])

            viewer.user_scn.ngeom = 0
            add_pose_box(viewer.user_scn, odom_pose, [0.0, 1.0, 0.0, 0.35])  #Green
            add_pose_box(viewer.user_scn, fused_pose, [0.1, 0.4, 1.0, 0.45]) #Blue
            if data.time - last_print >= 0.5:
                last_print = data.time
                print(
                    f"t={data.time:5.1f}s corrections={accepted:3d} "
                    f"odom_err={np.linalg.norm(odom_error[:2]):.3f}m "
                    f"fused_err={np.linalg.norm(fused_error[:2]):.3f}m "
                    f"fused_yaw_err={math.degrees(abs(fused_error[2])):.1f}deg"
                )
            viewer.sync()
            time.sleep(max(0.0, dt - (time.time() - start)))


if __name__ == "__main__":
    main()
