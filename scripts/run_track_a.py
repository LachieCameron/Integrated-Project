#!/usr/bin/env python3
"""Track A pipeline: wheel-odometry EKF -> navigation -> LQR tracker -> MuJoCo.

Run from the project root::

    python scripts/run_track_a.py --trials 1 --plots results          # one nominal run + plots
    python scripts/run_track_a.py --trials 20 --seed 0                # randomised start poses
    python scripts/run_track_a.py --trials 20 --unknown-obstacles     # discover boxes with LiDAR
    python scripts/run_track_a.py --trials 20 --enc-noise 0.002 --odom-scale-err 0.02 --init-err 0.05
    python scripts/run_track_a.py --viewer                            # watch one run (mjpython on macOS)

Ground-truth rule: the controller and navigator receive only ``ekf.state`` and
LiDAR ranges.  ``data.qpos`` / ``data.xpos`` are read in exactly two places,
both marked ``# GROUND TRUTH``: placing the robot at its start pose, and the
evaluation log.
"""

from __future__ import annotations

import argparse
import contextlib
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
from src.control.navigation import NavConfig, Navigator, State, boxes_from_model
from src.estimation.ekf import RobotEKF
from src.kinematics.differential_drive import DifferentialDrive, wrap_angle
from src.sensing.lidar_scan import Lidar
from src.sensing.wheel_encoder import WheelEncoder

SCENE = os.path.join(project_root, "assets", "scene_turtlebot3_waffle_pi.xml")
CTRL_DT = 0.02          # 50 Hz control / estimation rate
T_MAX = 150.0           # s, timeout per trial
DOCK_POS_TOL = 0.05     # m, strict docking threshold on the TRUE final position error
DOCK_YAW_TOL = 0.15     # rad, threshold on the TRUE final heading error


def yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def free_start_pose(model, rng, clearance=0.35, max_tries=1000):
    """Random start pose in free space (uses the static map, evaluation set-up only)."""
    nav = Navigator(model, CTRL_DT)
    for _ in range(max_tries):
        x, y = rng.uniform(-2.5, 2.0), rng.uniform(-2.5, 2.2)
        if nav.grid.clearance[nav.grid.to_index(x, y)] > clearance and not (x > 1.6 and y > 1.5):
            return np.array([x, y, rng.uniform(-math.pi, math.pi)])
    raise RuntimeError("could not find a free start pose")


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
    camera_dir: str | None = None,
    t_max: float = T_MAX,
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
    cfg = nav_config or NavConfig()
    cfg.use_prior_obstacles = not unknown_obstacles
    nav = Navigator(model, CTRL_DT, cfg)
    cam = None
    if camera or camera_dir:
        import cv2
        from src.sensing.cam import RaspiCamera
        cam = RaspiCamera(model)   # display only: the camera is not an EKF input yet
        if camera_dir:
            os.makedirs(camera_dir, exist_ok=True)

    if start_pose is None:
        start_pose = np.array([0.0, 0.0, 0.0])
    # GROUND TRUTH (set-up): place the robot.
    data.qpos[:2] = start_pose[:2]
    data.qpos[3:7] = yaw_to_quat(start_pose[2])
    mujoco.mj_forward(model, data)

    # EKF starts from the (possibly perturbed) start pose with a matching P0.
    sigma_xy, sigma_th = max(init_err, 0.01), max(init_err, 0.01)
    ekf_start = start_pose + rng.normal(0.0, [init_err, init_err, init_err], 3)
    ekf = RobotEKF(
        model=None if ekf_wheel_base else model,
        wheel_base=ekf_wheel_base,
        wheel_distance_std=max(enc_noise, 0.002),
        initial_state=ekf_start,
        initial_covariance=np.diag([sigma_xy**2, sigma_xy**2, sigma_th**2]),
    )
    scale_l, scale_r = 1.0 + rng.normal(0.0, odom_scale_err, 2) if odom_scale_err > 0 else (1.0, 1.0)

    log = {k: [] for k in ("t", "true", "est", "P_diag", "ref", "cmd", "err_ctrl", "err_true", "state", "front", "nees")}
    collisions = 0
    t0_wall = time.time()
    steps = int(t_max / CTRL_DT)
    out = None
    for k in range(steps):
        # ---- sensing -----------------------------------------------------------------
        d_l, d_r, _, _ = encoder.step(data)
        d_l = d_l * scale_l + (rng.normal(0.0, enc_noise) if enc_noise else 0.0)
        d_r = d_r * scale_r + (rng.normal(0.0, enc_noise) if enc_noise else 0.0)
        ranges = lidar.read(data)

        # ---- estimation (the ONLY pose the controller sees) -----------------------------
        pose_est = ekf.update(d_l, d_r, CTRL_DT)

        # ---- navigation + LQR -----------------------------------------------------------
        out = nav.update(pose_est, ranges)
        wl, wr = drive.twist_to_wheel_speeds(out.v, out.w)
        data.ctrl[0], data.ctrl[1] = wl, wr

        # ---- camera view (display / saved frames only, ~25 Hz) -----------------------------
        if cam is not None and k % 2 == 0:
            frame = cam.capture_cv_frame(data)
            _, ids, shown = cam.detect_markers(frame)
            cv2.putText(shown, f"t={data.time:.1f}s {out.state} marker={'yes' if ids is not None else 'no'}",
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
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-start", action="store_true", help="start at the origin facing +x (default for 1 trial)")
    ap.add_argument("--random-start", action="store_true")
    ap.add_argument("--unknown-obstacles", action="store_true", help="map holds only the walls; boxes are found by LiDAR")
    ap.add_argument("--enc-noise", type=float, default=0.0, help="std of added noise on wheel distance per step [m]")
    ap.add_argument("--odom-scale-err", type=float, default=0.0, help="std of per-wheel odometry scale error (e.g. 0.02)")
    ap.add_argument("--init-err", type=float, default=0.0, help="std of EKF initial pose error [m, rad]")
    ap.add_argument("--ekf-wheel-base", type=float, default=None,
                    help="effective wheel base for the EKF [m]; default is the geometric 0.288 m "
                         "(spin-in-place calibration in the simulator gives about 0.296 m)")
    ap.add_argument("--plots", type=str, default=None, help="directory for per-trial PNGs")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--camera", action="store_true", help="show the robot camera with ArUco detections (needs a display)")
    ap.add_argument("--save-camera", type=str, default=None, help="save camera frames (every 0.2 s) to this directory")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(SCENE)
    random_start = args.random_start or (args.trials > 1 and not args.fixed_start)
    rng = np.random.default_rng(args.seed)
    results = []
    for i in range(args.trials):
        start = free_start_pose(model, rng) if random_start else np.array([0.0, 0.0, 0.0])
        kwargs = dict(start_pose=start, seed=args.seed * 1000 + i, enc_noise=args.enc_noise,
                      odom_scale_err=args.odom_scale_err, init_err=args.init_err,
                      unknown_obstacles=args.unknown_obstacles, ekf_wheel_base=args.ekf_wheel_base,
                      camera=args.camera, camera_dir=args.save_camera)
        if args.viewer:
            from mujoco import viewer as mj_viewer
            data = mujoco.MjData(model)
            with mj_viewer.launch_passive(model, data) as viewer:
                res = run_trial(model, viewer=viewer, data=data, **kwargs)
        else:
            res = run_trial(model, **kwargs)
        m = res["metrics"]
        results.append(m)
        print(f"[{i + 1:3d}/{args.trials}] start=({start[0]:+.2f},{start[1]:+.2f},{math.degrees(start[2]):+4.0f}�) "
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