#!/usr/bin/env python3
"""Visual EKF check using wheel odometry and optional wall-marker corrections.

Run from the project root:

    python tests/test_ekf_ghostA.py
    python tests/test_ekf_ghostA.py --no-aruco --scale-error 0.02
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


def load_track_helpers():
    path = os.path.join(PROJECT_ROOT, "scripts", "run_track_a.py")
    spec = importlib.util.spec_from_file_location("run_track_a", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Track A helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-aruco", action="store_true")
    parser.add_argument("--imu", action="store_true")
    parser.add_argument("--scale-error", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()
    helpers = load_track_helpers()

    model = mujoco.MjModel.from_xml_path(helpers.SCENE)
    data = mujoco.MjData(model)
    data_ghost = mujoco.MjData(model)
    ghost_options = mujoco.MjvOption()
    ghost_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    ghost_perturb = mujoco.MjvPerturb()
    ghost_category = mujoco.mjtCatBit.mjCAT_DYNAMIC
    camera = RaspiCamera(model)
    imu = IMUSensor(model) if args.imu else None
    encoder = WheelEncoder(model)
    ekf = RobotEKF(model=model)
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
            gyro_z = None if imu is None else float(imu.read(data)[1][2])
            pose = ekf.update(
                d_left * scales[0],
                d_right * scales[1],
                dt,
                imu_gyro_z=gyro_z,
                imu_rate_std=0.03 if imu is not None else None,
            )

            if not args.no_aruco and int(round(data.time / dt)) % camera_period == 0:
                frame = camera.capture_cv_frame(data)
                corners, ids, _ = camera.detect_markers(frame)
                if ids is not None:
                    for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
                        measurement = helpers.robot_pose_from_wall_marker(
                            camera, int(marker_id), marker_corners, 0.4
                        )
                        if measurement is not None:
                            covariance = np.diag([0.02, 0.02, 0.05])
                            pose = ekf.fuse_pose(measurement, covariance)
                            accepted += 1
                            break

            true_pose = np.array(
                [data.qpos[0], data.qpos[1], 2.0 * math.atan2(data.qpos[6], data.qpos[3])]
            )
            true_pose[2] = wrap_angle(true_pose[2])
            error = pose - true_pose
            error[2] = wrap_angle(error[2])

            data_ghost.qpos[0] = pose[0]
            data_ghost.qpos[1] = pose[1]
            data_ghost.qpos[2] = 0.01
            data_ghost.qpos[3] = math.cos(pose[2] / 2.0)
            data_ghost.qpos[4:6] = 0.0
            data_ghost.qpos[6] = math.sin(pose[2] / 2.0)
            mujoco.mj_forward(model, data_ghost)

            viewer.user_scn.ngeom = 0
            mujoco.mjv_addGeoms(
                model,
                data_ghost,
                ghost_options,
                ghost_perturb,
                ghost_category,
                viewer.user_scn,
            )
            if data.time - last_print >= 0.5:
                last_print = data.time
                print(
                    f"t={data.time:5.1f}s corrections={accepted:3d} "
                    f"est=({pose[0]:+.2f},{pose[1]:+.2f},{math.degrees(pose[2]):+.1f}deg) "
                    f"error=({error[0]:+.2f},{error[1]:+.2f},{math.degrees(error[2]):+.1f}deg)"
                )
            viewer.sync()
            time.sleep(max(0.0, dt - (time.time() - start)))


if __name__ == "__main__":
    main()
