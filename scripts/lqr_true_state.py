#!/usr/bin/env python3
"""Controller-only verification: run the navigator + LQR with the TRUE pose as the state.

This is NOT the project pipeline.  It deliberately feeds simulator ground truth to the
controller so that the LQR's own tracking error can be separated from the estimator's.
The real pipeline is ``scripts/run_track_a.py`` (controller sees ``ekf.state`` only).

    python scripts/eval_lqr_true_state.py --trials 20 --seed 1
"""

import argparse
import importlib.util
import math
import os
import sys

import mujoco
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.control.lqr import LQRTrajectoryTracker
from src.control.navigation import NavConfig, Navigator, State
from src.kinematics.differential_drive import DifferentialDrive
from src.sensing.lidar_scan import Lidar

_spec = importlib.util.spec_from_file_location("run_track_a", os.path.join(project_root, "scripts", "run_track_a.py"))
rta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rta)


def run(model, start, q_scale=1.0, r_scale=1.0, t_max=150.0):
    data = mujoco.MjData(model)
    drive = DifferentialDrive.from_model(model)
    lidar = Lidar(model)
    cfg = NavConfig()
    base = LQRTrajectoryTracker(rta.CTRL_DT)
    tracker = LQRTrajectoryTracker(rta.CTRL_DT, q_diag=np.diag(base.Q) * q_scale, r_diag=np.diag(base.R) * r_scale,
                                   v_max=cfg.v_max + 0.04, w_max=cfg.w_max)
    nav = Navigator(model, rta.CTRL_DT, cfg, tracker=tracker)
    data.qpos[:2] = start[:2]
    data.qpos[3:7] = rta.yaw_to_quat(start[2])
    mujoco.mj_forward(model, data)
    errs, cmds, collisions, out = [], [], 0, None
    for _ in range(int(t_max / rta.CTRL_DT)):
        pose = np.array([data.qpos[0], data.qpos[1], 2.0 * math.atan2(data.qpos[6], data.qpos[3])])  # TRUE STATE
        out = nav.update(pose, lidar.read(data))
        if out.state == State.TRACK:
            errs.append(out.error)
            cmds.append((out.v, out.w))
        data.ctrl[:] = drive.twist_to_wheel_speeds(out.v, out.w)
        for _ in range(int(round(rta.CTRL_DT / model.opt.timestep))):
            mujoco.mj_step(model, data)
        collisions += rta.robot_collisions(model, data) > 0
        if out.state in (State.DONE, State.FAILED):
            break
    e = np.array(errs)
    c = np.diff(np.array(cmds), axis=0)
    final = np.hypot(data.qpos[0] - nav.dock_stop[0], data.qpos[1] - nav.dock_stop[1])
    yaw = abs((2.0 * math.atan2(data.qpos[6], data.qpos[3]) - math.pi / 2 + math.pi) % (2 * math.pi) - math.pi)
    return dict(done=out.state == State.DONE, collisions=collisions, time=data.time, final=float(final), yaw=float(yaw),
                rmse=float(np.sqrt(np.mean(e[:, 0] ** 2 + e[:, 1] ** 2))) if len(e) else float("nan"),
                rough_v=float(np.sqrt(np.mean(c[:, 0] ** 2))), rough_w=float(np.sqrt(np.mean(c[:, 1] ** 2))),
                rmse_yaw=float(np.sqrt(np.mean(e[:, 2] ** 2))) if len(e) else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--q-scale", type=float, default=1.0, help="multiply Q (state cost)")
    ap.add_argument("--r-scale", type=float, default=1.0, help="multiply R (input cost)")
    args = ap.parse_args()
    model = mujoco.MjModel.from_xml_path(rta.SCENE)
    rng = np.random.default_rng(args.seed)
    rows = []
    for i in range(args.trials):
        start = rta.free_start_pose(model, rng)
        r = run(model, start, args.q_scale, args.r_scale)
        rows.append(r)
        print(f"[{i + 1:2d}/{args.trials}] done={r['done']} coll={r['collisions']} t={r['time']:.1f}s "
              f"track_rmse={r['rmse'] * 1000:.1f}mm dock_err={r['final'] * 1000:.1f}mm yaw_err={math.degrees(r['yaw']):.1f}deg")
    ok = [r for r in rows if r["done"] and r["collisions"] == 0]
    print(f"\ncompleted without collision: {len(ok)}/{len(rows)}")
    for key, unit, scale in (("rmse", "mm", 1000), ("rough_v", "mm/s/step", 1000), ("rough_w", "mrad/s/step", 1000), ("rmse_yaw", "deg", 180 / math.pi), ("final", "mm", 1000), ("yaw", "deg", 180 / math.pi), ("time", "s", 1)):
        v = np.array([r[key] for r in ok]) * scale
        print(f"  {key:9s} mean {v.mean():7.2f} {unit}  max {v.max():7.2f} {unit}")


if __name__ == "__main__":
    main()
