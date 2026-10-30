"""Read the 13-beam LiDAR and convert a scan to world-frame points.

(``src/sensing/lidar.py`` is the one-off script that patches the XML; it runs
on import, so the reusable scan helpers live here instead.)
"""

from __future__ import annotations

import numpy as np
import mujoco

ANGLES_DEG = tuple(range(-90, 91, 15))
ANGLES = np.deg2rad(ANGLES_DEG)
# LiDAR origin in the robot (axle-centre) frame; from the XML site position.
LIDAR_OFFSET = np.array([-0.049, 0.0])
CUTOFF = 3.0


class Lidar:
    #Thin wrapper over the MuJoCo rangefinder sensors ``lidar_<deg>``.

    def __init__(self, model: mujoco.MjModel):
        self.adr = []
        for deg in ANGLES_DEG:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{deg}")
            if sid < 0:
                raise RuntimeError(f"Missing LiDAR sensor lidar_{deg} in the XML")
            self.adr.append(int(model.sensor_adr[sid]))
        self.cutoff = CUTOFF

    def read(self, data: mujoco.MjData) -> np.ndarray:
        #Ranges [m] per beam; ``inf`` where there is no return.
        r = np.array([data.sensordata[a] for a in self.adr], dtype=float)
        r[~np.isfinite(r) | (r < 0.0) | (r >= self.cutoff - 1e-3)] = np.inf
        return r


def scan_to_world(pose: np.ndarray, ranges: np.ndarray, hits_only: bool = True) -> np.ndarray:
    #World-frame (N, 2) points for a scan taken at ``pose = [x, y, theta]``.
    ranges = np.asarray(ranges, float)
    mask = np.isfinite(ranges) if hits_only else np.ones_like(ranges, bool)
    r = np.where(np.isfinite(ranges), ranges, CUTOFF)[mask]
    a = ANGLES[mask]
    local = LIDAR_OFFSET[None, :] + np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
    c, s = np.cos(pose[2]), np.sin(pose[2])
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.asarray(pose[:2])[None, :]


def sector_min(ranges: np.ndarray, low_deg: float, high_deg: float) -> float:
    #Minimum range over beams with bearing in [low_deg, high_deg].
    deg = np.asarray(ANGLES_DEG)
    sel = (deg >= low_deg) & (deg <= high_deg)
    return float(np.min(ranges[sel]))


def forward_distance(ranges: np.ndarray, half_width_deg: float = 15.0) -> float:
    #Mean perpendicular distance to a wall ahead, from the central beams.
    deg = np.asarray(ANGLES_DEG)
    sel = np.abs(deg) <= half_width_deg
    r = np.asarray(ranges, float)[sel]
    if not np.all(np.isfinite(r)):
        return float("inf")
    return float(np.mean(r * np.cos(ANGLES[sel])))
