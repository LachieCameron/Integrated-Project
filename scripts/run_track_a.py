#!/usr/bin/env python3
"""Track A pipeline: fused EKF -> navigation -> LQR tracker -> MuJoCo.

Run from the project root::

    python scripts/run_track_a.py --trials 1 --plots results          # one nominal run + plots
    python scripts/run_track_a.py --trials 20 --seed 0                # randomised start poses
    python scripts/run_track_a.py --trials 20 --unknown-obstacles     # discover boxes with LiDAR
    python scripts/run_track_a.py --trials 20 --enc-noise 0.002 --odom-scale-err 0.02 --init-err 0.05
    python scripts/run_track_a.py --viewer                            # watch one run (mjpython on macOS)

Ground-truth rule: the controller and navigator receive only ``ekf.state`` and
LiDAR ranges.  Wheel odometry is the prediction source; optional IMU gyro and
ArUco observations of the wall markers and dock tags (known map in
``WALL_MARKERS``) are fused by ``RobotEKF`` through a 99% NIS gate.  ``data.qpos`` /
``data.xpos`` are read in exactly two places,
both marked ``# GROUND TRUTH``: placing the robot at its start pose, and the
evaluation log.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import math
import os
import sys
import time

import numpy as np
import mujoco

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.control.lqr import body_frame_error
from src.control.lqr import LQRTrajectoryTracker
from src.control.docking import DockingConfig, VisualDockingController
from src.control.navigation import NavConfig, Navigator, State, boxes_from_model
from src.estimation.ekf import GATE_CHI2_3DOF_99, RobotEKF
from src.kinematics.differential_drive import DifferentialDrive, wrap_angle
from src.sensing.lidar_scan import Lidar
from src.sensing.IMU import IMUSensor
from src.sensing.wheel_encoder import WheelEncoder
from src.utils.config import AppConfig, load_config
from randomize_scene import randomize_scene

SCENE = os.path.join(project_root, "assets", "scene_turtlebot3_waffle_pi.xml")
CTRL_DT = 0.02          # 50 Hz control / estimation rate
T_MAX = 150.0           # s, timeout per trial
DOCK_POS_TOL = 0.05     # m, strict docking threshold on the TRUE final position error
DOCK_YAW_TOL = 0.15     # rad, threshold on the TRUE final heading error
WALL_MARKERS = {
    22: ((0.0, -2.9, 0.5), math.pi),
    44: ((-2.9, 0.0, 0.5), math.pi / 2.0),
    66: ((2.9, 0.0, 0.5), -math.pi / 2.0),
    88: ((0.0, 2.9, 0.5), 0.0),
    0: ((2.4, 2.5, 0.3), 0.0),    # dock tag
    99: ((2.4, 2.5, 0.1), 0.0),   # fine dock tag
}
DOCK_MARKER_ID = 0
FINE_DOCK_MARKER_ID = 99
CAMERA_POSITION_IN_ROBOT = np.array([0.078, 0.0, 0.107])
# OpenCV camera axes (right, down, forward) expressed in robot axes
# (forward, left, up).
ROBOT_FROM_CAMERA = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
)
MAX_MARKER_REPROJECTION_ERROR_PX = 2.0


def yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def wall_marker_rotation(yaw: float) -> np.ndarray:
    """Map a marker frame (x right, y up, z outward) into world axes."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    ) @ np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )


def robot_pose_from_wall_marker(camera, marker_id, corners, marker_size_m):
    """Estimate robot [x, y, yaw] from one known wall marker."""
    marker = WALL_MARKERS.get(int(marker_id))
    if marker is None:
        return None
    marker_pose = camera.estimate_marker_pose(corners, marker_size_m)
    if marker_pose.reprojection_error_px > MAX_MARKER_REPROJECTION_ERROR_PX:
        return None
    marker_position, marker_yaw = marker
    marker_rotation = wall_marker_rotation(marker_yaw)
    camera_position, camera_rotation = camera.camera_pose_from_marker(
        np.asarray(marker_position), marker_rotation, marker_pose
    )
    robot_rotation = camera_rotation @ ROBOT_FROM_CAMERA.T
    robot_position = camera_position - robot_rotation @ CAMERA_POSITION_IN_ROBOT
    return np.array(
        [robot_position[0], robot_position[1],
         math.atan2(robot_rotation[1, 0], robot_rotation[0, 0])],
        dtype=float,
    )


_DEFAULT_CONFIG: AppConfig | None = None


def marker_size_for(marker_id: int, config: AppConfig | None = None) -> float:
    """Side length [m] of a marker, from the camera section of the config.

    With no ``config`` the default YAML is used, so helper scripts can call
    ``marker_size_for(marker_id)`` directly.
    """
    global _DEFAULT_CONFIG
    if config is None:
        if _DEFAULT_CONFIG is None:
            _DEFAULT_CONFIG = load_config(os.path.join(project_root, "config", "default.yaml"))
        config = _DEFAULT_CONFIG
    if marker_id == DOCK_MARKER_ID:
        return config.camera.marker_size_m
    if marker_id == FINE_DOCK_MARKER_ID:
        return config.camera.fine_marker_size_m
    return config.camera.wall_marker_size_m


def free_start_pose(model, rng, clearance=0.35, max_tries=1000):
    """Random start pose in free space (uses the static map, evaluation set-up only)."""
    nav = Navigator(model, CTRL_DT)
    for _ in range(max_tries):
        x, y = rng.uniform(-2.5, 2.0), rng.uniform(-2.5, 2.2)
        if nav.grid.clearance[nav.grid.to_index(x, y)] > clearance and not (x > 1.6 and y > 1.5):
            return np.array([x, y, rng.uniform(-math.pi, math.pi)])
    raise RuntimeError("could not find a free start pose")


def nav_config_from_app(config: AppConfig, unknown_obstacles: bool = False) -> NavConfig:
    """Translate validated YAML settings into the navigation configuration."""
    obstacle = config.obstacle_avoidance
    nav = NavConfig(
        v_max=config.lqr.max_linear_velocity,
        w_max=config.lqr.max_angular_velocity,
        stop_range=obstacle.stop_distance,
        slow_range=obstacle.slow_distance,
        use_prior_obstacles=not unknown_obstacles,
    )
    return nav


def robot_collisions(model, data) -> int:
    """Number of contacts between the robot and non-floor world geoms (ground truth, evaluation)."""
    robot_bodies = {model.body(n).id for n in ("base", "wheel_left", "wheel_right")}
    floor = model.geom("floor").id
    count = 0
    for c in data.contact[: data.ncon]:
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if (b1 in robot_bodies) == (b2 in robot_bodies):
            continue
        if floor in (c.geom1, c.geom2):
            continue
        count += 1
    return count


def run_trial(
    model: mujoco.MjModel,
    start_pose: np.ndarray | None = None,
    seed: int = 0,
    enc_noise: float = 0.0,
    odom_scale_err: float = 0.0,
    init_err: float = 0.0,
    unknown_obstacles: bool = False,
    ekf_wheel_base: float | None = None,
    nav_config: NavConfig | None = None,
    viewer=None,
    data: mujoco.MjData | None = None,
    camera: bool = False,
    aruco_fusion: bool = False,
    camera_dir: str | None = None,
    t_max: float = T_MAX,
    app_config: AppConfig | None = None,
    docking_mode: str = "lqr",
) -> dict:
    """Simulate one docking run and return metrics plus time series.

    Pass ``viewer`` (a passive MuJoCo viewer) together with the ``data`` it was
    launched with to watch the run in real time.
    """
    rng = np.random.default_rng(seed)
    data = data if data is not None else mujoco.MjData(model)
    n_sub = int(round(CTRL_DT / model.opt.timestep))
    drive = DifferentialDrive.from_model(model)
    lidar = Lidar(model)
    encoder = WheelEncoder(model)
    config = app_config or load_config(os.path.join(project_root, "config", "default.yaml"))
    imu = IMUSensor(model) if config.ekf.gyro_enabled else None
    cfg = nav_config or nav_config_from_app(config, unknown_obstacles)
    if docking_mode == "visual":
        cfg = dataclasses.replace(cfg, visual_docking=True)
    tracker = LQRTrajectoryTracker(
        CTRL_DT,
        q_diag=config.lqr.q,
        r_diag=config.lqr.r,
        v_max=config.lqr.max_linear_velocity,
        w_max=config.lqr.max_angular_velocity,
    )
    nav = Navigator(model, CTRL_DT, cfg, tracker=tracker)
    visual_docking = VisualDockingController(DockingConfig()) if docking_mode == "visual" else None
    cam = None
    camera_period_steps = 1
    if camera or camera_dir or aruco_fusion or docking_mode == "visual":
        import cv2
        from src.sensing.cam import RaspiCamera
        calibration_path = config.camera.calibration_file
        if calibration_path is not None and not os.path.isabs(calibration_path):
            calibration_path = os.path.join(project_root, calibration_path)
        cam = RaspiCamera(model, calibration_path=calibration_path)
        camera_period_steps = max(1, round(1.0 / (config.camera.detection_rate_hz * CTRL_DT)))
        if camera_dir:
            os.makedirs(camera_dir, exist_ok=True)

    if start_pose is None:
        start_pose = np.array([0.0, 0.0, 0.0])
    # GROUND TRUTH (set-up): place the robot.
    data.qpos[:2] = start_pose[:2]
    data.qpos[3:7] = yaw_to_quat(start_pose[2])
    mujoco.mj_forward(model, data)

    # EKF starts from the (possibly perturbed) start pose with a matching P0.
    ekf_start = start_pose + rng.normal(0.0, [init_err, init_err, init_err], 3)
    initial_covariance = np.diag(config.ekf.initial_covariance)
    if init_err > 0.0:
        initial_covariance = np.diag([init_err**2, init_err**2, init_err**2])
    ekf = RobotEKF(
        model=None if ekf_wheel_base else model,
        wheel_base=ekf_wheel_base,
        wheel_noise_coeff=config.ekf.wheel_noise_coeff,
        # Injected encoder noise (--enc-noise) is known to the filter via the floor term.
        wheel_noise_floor_std=max(enc_noise, config.ekf.wheel_noise_floor_std),
        initial_state=ekf_start,
        initial_covariance=initial_covariance,
    )
    scale_l, scale_r = 1.0 + rng.normal(0.0, odom_scale_err, 2) if odom_scale_err > 0 else (1.0, 1.0)

    log = {k: [] for k in ("t", "true", "est", "P_diag", "ref", "cmd", "err_ctrl", "err_true", "state", "front", "nees")}
    collisions = 0
    t0_wall = time.time()
    steps = int(t_max / CTRL_DT)
    out = None
    visible_dock = None
    for k in range(steps):
        # ---- sensing -----------------------------------------------------------------
        d_l, d_r, _, _ = encoder.step(data)
        d_l = d_l * scale_l + (rng.normal(0.0, enc_noise) if enc_noise else 0.0)
        d_r = d_r * scale_r + (rng.normal(0.0, enc_noise) if enc_noise else 0.0)
        ranges = lidar.read(data)
        gyro_z = None if imu is None else float(imu.read(data)[1][2])

        # ---- estimation (the ONLY pose the controller sees) -----------------------------
        pose_est = ekf.update(
            d_l,
            d_r,
            CTRL_DT,
            imu_gyro_z=gyro_z,
            imu_rate_std=config.ekf.gyro_rate_std if imu is not None else None,
        )

        marker_ids = None
        shown = None
        if cam is not None and k % camera_period_steps == 0:
            frame = cam.capture_cv_frame(data)
            corners, marker_ids, shown = cam.detect_markers(frame)
            visible_dock = None
            if docking_mode == "visual" and marker_ids is not None:
                for marker_corners, marker_id in zip(corners, marker_ids.reshape(-1)):
                    if int(marker_id) != FINE_DOCK_MARKER_ID:
                        continue
                    marker = np.asarray(marker_corners).reshape(4, 2)
                    center_error = (
                        float(np.mean(marker[:, 0])) - cam.width / 2.0
                    ) / (cam.width / 2.0)
                    visible_dock = (
                        float(np.linalg.norm(marker[1] - marker[0])),
                        abs(center_error),
                        center_error,
                    )
                    break
            if aruco_fusion and marker_ids is not None:
                for marker_corners, marker_id in zip(corners, marker_ids.reshape(-1)):
                    measurement = robot_pose_from_wall_marker(
                        cam, int(marker_id), marker_corners,
                        marker_size_for(int(marker_id), config),
                    )
                    if measurement is not None:
                        covariance = np.diag(config.ekf.camera_measurement_covariance)
                        # Returns the state either way; ekf.last_accepted records the gate decision.
                        pose_est = ekf.fuse_pose(measurement, covariance, gate_chi2=GATE_CHI2_3DOF_99)
                        break

        # ---- navigation + LQR -----------------------------------------------------------
        out = nav.update(pose_est, ranges)
        if visual_docking is not None and visible_dock is not None:
            width, center_error, signed_error = visible_dock
            v_visual, w_visual, docked = visual_docking.command(
                width,
                center_error,
                signed_error,
                out.front_range,
                pose_est[2],
                math.pi / 2.0,
            )
            out.v, out.w = v_visual, w_visual
            if docked:
                nav.state = State.DONE
                out.state = State.DONE
        elif visual_docking is not None and out.state != State.DONE:
            visual_docking.reset()
        wl, wr = drive.twist_to_wheel_speeds(out.v, out.w)
        data.ctrl[0], data.ctrl[1] = wl, wr

        # ---- camera view (display / saved frames only, ~25 Hz) -----------------------------
        if shown is not None:
            cv2.putText(shown, f"t={data.time:.1f}s {out.state} marker={'yes' if marker_ids is not None else 'no'}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if camera_dir and k % 10 == 0:
                cv2.imwrite(os.path.join(camera_dir, f"frame_{k:05d}.png"), shown)
            if camera:
                cv2.imshow("TurtleBot3 camera", shown)
                cv2.waitKey(1)

        # ---- physics ---------------------------------------------------------------------
        with (viewer.lock() if viewer is not None else contextlib.nullcontext()):
            for _ in range(n_sub):
                mujoco.mj_step(model, data)
        collisions += robot_collisions(model, data) > 0

        # ---- evaluation log (GROUND TRUTH, never fed back) --------------------------------
        true_pose = np.array([data.qpos[0], data.qpos[1],
                              2.0 * math.atan2(data.qpos[6], data.qpos[3])])  # GROUND TRUTH
        true_pose[2] = float(wrap_angle(true_pose[2]))
        diff = pose_est - true_pose
        diff[2] = float(wrap_angle(diff[2]))
        try:
            nees = float(diff @ np.linalg.solve(ekf.P, diff))
        except np.linalg.LinAlgError:
            nees = float("nan")
        err_true = body_frame_error(true_pose, out.ref_pose) if out.state == State.TRACK else np.zeros(3)
        log["t"].append(data.time); log["true"].append(true_pose); log["est"].append(pose_est.copy())
        log["P_diag"].append(np.diag(ekf.P).copy()); log["ref"].append(out.ref_pose.copy())
        log["cmd"].append((out.v, out.w)); log["err_ctrl"].append(out.error.copy())
        log["err_true"].append(err_true); log["state"].append(out.state)
        log["front"].append(out.front_range); log["nees"].append(nees)

        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(max(0.0, t0_wall + (k + 1) * CTRL_DT - time.time()))  # real-time pacing
        if out.state in (State.DONE, State.FAILED):
            break

    for key in ("true", "est", "P_diag", "ref", "cmd", "err_ctrl", "err_true"):
        log[key] = np.array(log[key])
    log["t"] = np.array(log["t"])
    log["nees"] = np.array(log["nees"])

    # ---- metrics ---------------------------------------------------------------------------
    final_true = log["true"][-1]
    dock = nav.dock_stop
    pos_err = float(np.hypot(final_true[0] - dock[0], final_true[1] - dock[1]))
    yaw_err = float(abs(wrap_angle(final_true[2] - math.pi / 2)))
    est_err = np.linalg.norm(log["est"][:, :2] - log["true"][:, :2], axis=1)
    track_mask = np.array([s == State.TRACK for s in log["state"]])
    et = log["err_true"][track_mask]
    ec = log["err_ctrl"][track_mask]
    docked = out.state == State.DONE
    metrics = {
        "completed": bool(docked and collisions == 0),
        "success": bool(docked and pos_err < DOCK_POS_TOL and yaw_err < DOCK_YAW_TOL and collisions == 0),
        "nav_state": out.state,
        "time_s": float(log["t"][-1]),
        "dock_pos_err_m": pos_err,
        "dock_lateral_err_m": float(abs(final_true[0] - dock[0])),
        "dock_yaw_err_rad": yaw_err,
        "collision_steps": int(collisions),
        "replans": nav.replans,
        "recovery_replans": nav.recovery_replans,
        "track_rmse_pos_true_m": float(np.sqrt(np.mean(et[:, 0] ** 2 + et[:, 1] ** 2))) if len(et) else float("nan"),
        "track_rmse_pos_est_m": float(np.sqrt(np.mean(ec[:, 0] ** 2 + ec[:, 1] ** 2))) if len(ec) else float("nan"),
        "track_rmse_yaw_true_rad": float(np.sqrt(np.mean(et[:, 2] ** 2))) if len(et) else float("nan"),
        "ekf_rmse_pos_m": float(np.sqrt(np.mean(est_err**2))),
        "ekf_final_pos_err_m": float(est_err[-1]),
        "mean_nees": float(np.nanmean(log["nees"])),
        "control_effort": float(np.sum(np.abs(np.diff(log["cmd"], axis=0))) ),
        "wall_s": time.time() - t0_wall,
    }
    return {"metrics": metrics, "log": log, "nav": nav, "start": start_pose}


def plot_trial(result: dict, model, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    log, nav = result["log"], result["nav"]
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    ax = axs[0, 0]
    for b in boxes_from_model(model):
        col = "tab:gray" if b.name.endswith("_wall") else "tab:red"
        ax.add_patch(Rectangle((b.cx - b.hx, b.cy - b.hy), 2 * b.hx, 2 * b.hy, color=col, alpha=0.5))
    ax.plot(log["ref"][:, 0], log["ref"][:, 1], "k--", lw=1, label="reference")
    ax.plot(log["true"][:, 0], log["true"][:, 1], "tab:green", label="true (eval only)")
    ax.plot(log["est"][:, 0], log["est"][:, 1], "tab:blue", lw=1, label="EKF estimate")
    ax.plot(*nav.dock_stop, "m*", ms=12, label="dock stop")
    ax.set_aspect("equal"); ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.legend(fontsize=7); ax.set_title("Trajectory")
    t = log["t"]
    ax = axs[0, 1]
    ax.plot(t, log["cmd"][:, 0], label="v [m/s]"); ax.plot(t, log["cmd"][:, 1], label="w [rad/s]")
    ax.set_title("LQR output"); ax.legend(); ax.set_xlabel("t [s]")
    ax = axs[0, 2]
    ax.plot(t, log["err_ctrl"][:, 0], label="ex (est)"); ax.plot(t, log["err_ctrl"][:, 1], label="ey (est)")
    ax.plot(t, log["err_true"][:, 0], ":", label="ex (true)"); ax.plot(t, log["err_true"][:, 1], ":", label="ey (true)")
    ax.set_title("Body-frame tracking error [m]"); ax.legend(fontsize=7); ax.set_xlabel("t [s]")
    ax = axs[1, 0]
    err = log["est"] - log["true"]; err[:, 2] = wrap_angle(err[:, 2])
    sig = np.sqrt(log["P_diag"])
    ax.plot(t, err[:, 0], label="x err"); ax.plot(t, err[:, 1], label="y err")
    ax.fill_between(t, -3 * sig[:, 0], 3 * sig[:, 0], alpha=0.15, label="�3� (x)")
    ax.set_title("EKF error vs �3�"); ax.legend(fontsize=7); ax.set_xlabel("t [s]")
    ax = axs[1, 1]
    ax.plot(t, np.degrees(err[:, 2])); ax.fill_between(t, -3 * np.degrees(sig[:, 2]), 3 * np.degrees(sig[:, 2]), alpha=0.15)
    ax.set_title("Yaw error [deg] vs �3�"); ax.set_xlabel("t [s]")
    ax = axs[1, 2]
    ax.plot(t, log["front"]); ax.set_title("LiDAR front range [m]"); ax.set_xlabel("t [s]")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def summarise(all_metrics: list[dict]) -> str:
    n = len(all_metrics)
    comp = np.array([m["completed"] for m in all_metrics])
    pos_all = np.array([m["dock_pos_err_m"] for m in all_metrics])
    strict = sum(m["success"] for m in all_metrics)
    done = [m for m in all_metrics if m["completed"]]

    def stat(key, rows, scale=1.0):
        v = np.array([m[key] for m in rows], float) * scale
        if len(v) == 0:
            return "n/a"
        return f"{np.nanmean(v):.3f} � {np.nanstd(v):.3f} (max {np.nanmax(v):.3f})"

    failed = [m for m in all_metrics if m["nav_state"] == State.FAILED]
    timeout = [m for m in all_metrics if m["nav_state"] not in (State.DONE, State.FAILED)]
    collided = [m for m in all_metrics if m["collision_steps"] > 0]
    lines = [
        f"trials: {n}",
        f"  completed (navigator DONE, no collision):  {len(done)}/{n} ({100.0 * len(done) / n:.0f}%)",
        f"     navigator FAILED: {len(failed)}   timeout: {len(timeout)}   collided: {len(collided)}",
        f"  completed AND true final pos err < 0.05 m: {int(np.sum(comp & (pos_all < 0.05)))}/{n}",
        f"  completed AND true final pos err < 0.10 m: {int(np.sum(comp & (pos_all < 0.10)))}/{n}",
        f"  completed AND true final pos err < 0.20 m: {int(np.sum(comp & (pos_all < 0.20)))}/{n}",
        f"  strict success (<{DOCK_POS_TOL} m, yaw<{DOCK_YAW_TOL} rad, no collision): {strict}/{n}",
        "  --- over all trials (what the LQR regulates vs. the real error) ---",
        f"  tracking RMSE w.r.t. estimate [m]  {stat('track_rmse_pos_est_m', all_metrics)}",
        f"  tracking RMSE w.r.t. truth    [m]  {stat('track_rmse_pos_true_m', all_metrics)}",
        f"  EKF position RMSE             [m]  {stat('ekf_rmse_pos_m', all_metrics)}",
        f"  mean NEES (3 dof, ideal 3)         {stat('mean_nees', all_metrics)}",
        f"  replans per trial (all kinds)      {stat('replans', all_metrics)}",
        "  --- completed trials only ---",
        f"  time to dock [s]                   {stat('time_s', done)}",
        f"  final pos err (true) [m]           {stat('dock_pos_err_m', done)}",
        f"  final lateral err [m]              {stat('dock_lateral_err_m', done)}",
        f"  final yaw err [deg]                {stat('dock_yaw_err_rad', done, 180 / math.pi)}",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=str, default=os.path.join(project_root, "config", "default.yaml"))
    ap.add_argument("--scene", type=str, default=SCENE, help="MuJoCo scene XML to load")
    ap.add_argument("--random-scene", action="store_true", help="randomize obstacle geoms in memory")
    ap.add_argument("--scene-seed", type=int, default=0)
    ap.add_argument("--obstacle-count", type=int, default=3)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--fixed-start", action="store_true", help="start at the origin facing +x (default for 1 trial)")
    ap.add_argument("--random-start", action="store_true")
    ap.add_argument("--unknown-obstacles", action="store_true", help="map holds only the walls; boxes are found by LiDAR")
    ap.add_argument("--enc-noise", type=float, default=None, help="std of added noise on wheel distance per step [m]")
    ap.add_argument("--odom-scale-err", type=float, default=None, help="std of per-wheel odometry scale error (e.g. 0.02)")
    ap.add_argument("--init-err", type=float, default=0.0, help="std of EKF initial pose error [m, rad]")
    ap.add_argument("--ekf-wheel-base", type=float, default=None,
                    help="effective wheel base for the EKF [m]; default is the geometric 0.288 m "
                         "(spin-in-place calibration in the simulator gives about 0.296 m)")
    ap.add_argument("--plots", type=str, default=None, help="directory for per-trial PNGs")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--camera", action="store_true", help="show the robot camera with ArUco detections (needs a display)")
    ap.add_argument("--aruco-fusion", action="store_true", help="use wall ArUco poses as EKF corrections")
    ap.add_argument(
        "--docking-mode",
        choices=("lqr", "visual"),
        default="lqr",
        help="use Navigator LQR docking, or shared fine-tag visual docking",
    )
    ap.add_argument("--save-camera", type=str, default=None, help="save camera frames (every 0.2 s) to this directory")
    args = ap.parse_args()

    config = load_config(args.config)
    trials = config.evaluation.trial_count if args.trials is None else args.trials
    seed = config.evaluation.random_seed if args.seed is None else args.seed
    enc_noise = config.evaluation.encoder_noise_std if args.enc_noise is None else args.enc_noise
    odom_scale_err = (
        config.evaluation.odometry_scale_error
        if args.odom_scale_err is None
        else args.odom_scale_err
    )
    scene_path = args.scene
    if not os.path.isabs(scene_path):
        scene_path = os.path.join(project_root, scene_path)
    if not os.path.isfile(scene_path):
        raise FileNotFoundError(f"Missing scene XML: {scene_path}")
    model = mujoco.MjModel.from_xml_path(scene_path)
    if args.random_scene:
        randomize_scene(model, seed=args.scene_seed, count=args.obstacle_count)
    random_start = args.random_start or (trials > 1 and not args.fixed_start)
    rng = np.random.default_rng(seed)
    results = []
    for i in range(trials):
        start = free_start_pose(model, rng) if random_start else np.array([0.0, 0.0, 0.0])
        kwargs = dict(start_pose=start, seed=seed * 1000 + i, enc_noise=enc_noise,
                      odom_scale_err=odom_scale_err, init_err=args.init_err,
                      unknown_obstacles=args.unknown_obstacles, ekf_wheel_base=args.ekf_wheel_base,
                      camera=args.camera, camera_dir=args.save_camera,
                      aruco_fusion=args.aruco_fusion,
                      docking_mode=args.docking_mode,
                      t_max=config.evaluation.timeout_s, app_config=config)
        if args.viewer:
            from mujoco import viewer as mj_viewer
            data = mujoco.MjData(model)
            with mj_viewer.launch_passive(model, data) as viewer:
                res = run_trial(model, viewer=viewer, data=data, **kwargs)
        else:
            res = run_trial(model, **kwargs)
        m = res["metrics"]
        results.append(m)
        print(f"[{i + 1:3d}/{trials}] start=({start[0]:+.2f},{start[1]:+.2f},"
              f"{math.degrees(start[2]):+.0f}deg) "
              f"{'OK  ' if m['completed'] else 'FAIL'} state={m['nav_state']:6s} t={m['time_s']:5.1f}s "
              f"pos_err={m['dock_pos_err_m']:.3f}m yaw_err={math.degrees(m['dock_yaw_err_rad']):.1f}� "
              f"rmse_track={m['track_rmse_pos_true_m']:.3f}m ekf_err={m['ekf_final_pos_err_m']:.3f}m "
              f"replans={m['replans']}({m['recovery_replans']}) coll={m['collision_steps']} ({m['wall_s']:.0f}s wall)")
        if args.plots:
            plot_trial(res, model, os.path.join(args.plots, f"trial_{i:02d}.png"))
    print()
    print(summarise(results))


if __name__ == "__main__":
    main()