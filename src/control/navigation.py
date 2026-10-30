#Waypoint navigation and docking

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.ndimage import distance_transform_edt

from src.control.lqr import LQRTrajectoryTracker
from src.kinematics.differential_drive import wrap_angle
from collections import deque

from src.sensing.lidar_scan import LIDAR_OFFSET, forward_distance, scan_to_world, sector_min


def _rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


# ======================================================================
# Occupancy grid
# ======================================================================
@dataclass
class Box:
    #Oriented rectangle in the XY plane (centre, half-extents, yaw).

    cx: float
    cy: float
    hx: float
    hy: float
    yaw: float = 0.0
    name: str = ""


def boxes_from_model(model: mujoco.MjModel, include_obstacles: bool = True) -> list[Box]:
    #Extract static box geoms attached to the world body from the XML.

    mj_data = mujoco.MjData(model)
    mujoco.mj_forward(model, mj_data)
    boxes = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != 0 or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        is_wall = name.endswith("_wall")
        if not is_wall and not include_obstacles:
            continue
        xmat = mj_data.geom_xmat[gid].reshape(3, 3)
        pos = mj_data.geom_xpos[gid]
        size = model.geom_size[gid]
        boxes.append(
            Box(pos[0], pos[1], size[0], size[1], math.atan2(xmat[1, 0], xmat[0, 0]), name)
        )
    return boxes


class OccupancyGrid:
    #Square-cell occupancy grid with robot-radius inflation.

    def __init__(self, half_extent: float = 3.2, resolution: float = 0.05, inflation: float = 0.25,
                 gate: float = 0.12):
        self.gate = float(gate)   # LiDAR hits this close to known geometry are treated as noise on it
        self.res = float(resolution)
        self.half = float(half_extent)
        self.n = int(round(2 * half_extent / resolution))
        self.inflation = float(inflation)
        self.static = np.zeros((self.n, self.n), bool)   # [ix, iy]
        self.dynamic = np.zeros((self.n, self.n), bool)  # LiDAR-detected
        self._dist = None
        self._dirty = True

    # --- index helpers --------------------------------------------------
    def to_index(self, x: float, y: float) -> tuple[int, int]:
        return int((x + self.half) / self.res), int((y + self.half) / self.res)

    def to_world(self, ix: int, iy: int) -> tuple[float, float]:
        return (ix + 0.5) * self.res - self.half, (iy + 0.5) * self.res - self.half

    def in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.n and 0 <= iy < self.n

    # --- building the map ----------------------------------------------
    def add_boxes(self, boxes: list[Box]) -> None:
        centres = (np.arange(self.n) + 0.5) * self.res - self.half
        gx, gy = np.meshgrid(centres, centres, indexing="ij")
        for b in boxes:
            c, s = math.cos(b.yaw), math.sin(b.yaw)
            lx = c * (gx - b.cx) + s * (gy - b.cy)
            ly = -s * (gx - b.cx) + c * (gy - b.cy)
            # Include a thin box (e.g. a marker board) by padding to half a cell.
            pad = 0.5 * self.res
            self.static |= (np.abs(lx) <= b.hx + pad) & (np.abs(ly) <= b.hy + pad)
        self._static_dist = distance_transform_edt(~self.static) * self.res
        self._dirty = True

    def mark_points(self, points: np.ndarray) -> int:
        #Add LiDAR hit points to the dynamic layer; return number of new cells.

        new = 0
        for x, y in points:
            ix, iy = self.to_index(x, y)
            if not self.in_bounds(ix, iy) or self._static_dist[ix, iy] < self.gate:
                continue
            if not self.dynamic[ix, iy]:
                self.dynamic[ix, iy] = True
                new += 1
        if new:
            self._dirty = True
        return new

    def clear_rays(self, origin: np.ndarray, endpoints: np.ndarray, margin: float = 0.15) -> int:
        #Clear dynamic cells that LiDAR beams pass through (free-space evidence).

        cleared = 0
        origin = np.asarray(origin, float)
        for end in endpoints:
            vec = np.asarray(end, float) - origin
            length = float(np.hypot(*vec)) - margin
            if length <= 0.0:
                continue
            n = int(length / (0.5 * self.res)) + 1
            ts = np.linspace(0.0, length / (length + margin), n)
            for t in ts:
                ix, iy = self.to_index(origin[0] + t * vec[0], origin[1] + t * vec[1])
                if self.in_bounds(ix, iy) and self.dynamic[ix, iy]:
                    self.dynamic[ix, iy] = False
                    cleared += 1
        if cleared:
            self._dirty = True
        return cleared

    # --- queries ---------------------------------------------------------
    def _refresh(self) -> None:
        if self._dirty:
            occ = self.static | self.dynamic
            self._dist = distance_transform_edt(~occ) * self.res
            self._dirty = False

    @property
    def clearance(self) -> np.ndarray:
        #Distance [m] from each cell centre to the nearest occupied cell.
        self._refresh()
        return self._dist

    @property
    def blocked(self) -> np.ndarray:
        #Cells the robot centre may not enter (obstacles inflated by the robot radius).
        return self.clearance < self.inflation

    def point_blocked(self, x: float, y: float) -> bool:
        ix, iy = self.to_index(x, y)
        return (not self.in_bounds(ix, iy)) or bool(self.blocked[ix, iy])

    def segment_free(self, p0, p1) -> bool:
        n = max(2, int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / (0.5 * self.res)))
        for t in np.linspace(0.0, 1.0, n):
            if self.point_blocked(p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])):
                return False
        return True

    def nearest_free(self, x: float, y: float) -> tuple[int, int]:
        #Index of the nearest unblocked cell to (x, y).
        ix, iy = self.to_index(x, y)
        ix = min(max(ix, 0), self.n - 1)
        iy = min(max(iy, 0), self.n - 1)
        if not self.blocked[ix, iy]:
            return ix, iy
        _, idx = distance_transform_edt(self.blocked, return_indices=True)
        return int(idx[0, ix, iy]), int(idx[1, ix, iy])


# ======================================================================
# Global planner
# ======================================================================
def plan_path(grid: OccupancyGrid, start_xy, goal_xy, clearance_weight: float = 2.0):
    """
    Returns an (N, 2) array of waypoints beginning at start_xy and ending
    at goal_xy, or None if no path exists.
    """
    s = grid.nearest_free(*start_xy)
    g = grid.nearest_free(*goal_xy)
    blocked = grid.blocked
    clear = grid.clearance
    soft = 0.5  # m: extra cost within this distance of an obstacle

    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]
    open_heap = [(0.0, 0.0, s)]
    best = {s: 0.0}
    parent = {s: None}
    while open_heap:
        _, cost, cur = heapq.heappop(open_heap)
        if cur == g:
            break
        if cost > best.get(cur, math.inf):
            continue
        for dx, dy, step in moves:
            nxt = (cur[0] + dx, cur[1] + dy)
            if not grid.in_bounds(*nxt) or blocked[nxt]:
                continue
            if dx and dy and (blocked[cur[0] + dx, cur[1]] or blocked[cur[0], cur[1] + dy]):
                continue  # no corner cutting
            penalty = clearance_weight * max(0.0, soft - clear[nxt]) / soft
            new_cost = cost + step * (1.0 + penalty)
            if new_cost < best.get(nxt, math.inf):
                best[nxt] = new_cost
                parent[nxt] = cur
                h = math.hypot(g[0] - nxt[0], g[1] - nxt[1])
                heapq.heappush(open_heap, (new_cost + h, new_cost, nxt))
    else:
        return None
    if g not in parent:
        return None

    cells = []
    node = g
    while node is not None:
        cells.append(node)
        node = parent[node]
    cells.reverse()
    pts = [np.array(start_xy, float)] + [np.array(grid.to_world(*c)) for c in cells[1:-1]] + [np.array(goal_xy, float)]

    # Line-of-sight shortcutting on the inflated grid.
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not grid.segment_free(pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return np.array(out)


# ======================================================================
# Reference path with curvature and speed profile
# ======================================================================
class ReferencePath:
    #Arc-length parameterised path of lines joined by circular fillets.

    def __init__(self, waypoints: np.ndarray, fillet_radius: float, ds: float = 0.01):
        pts = np.asarray(waypoints, float)
        # Drop duplicate points.
        keep = [0] + [i for i in range(1, len(pts)) if np.linalg.norm(pts[i] - pts[i - 1]) > 1e-6]
        pts = pts[keep]
        if len(pts) < 2:
            raise ValueError("path needs at least two distinct waypoints")
        self.waypoints = pts
        self.fillet_radius = float(fillet_radius)
        self.ds = ds

        seg = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        seg_dir = seg / seg_len[:, None]
        n_seg = len(seg)

        # Trim distances at each interior corner.
        trim_in = np.zeros(n_seg)   # trimmed from the end of segment i
        trim_out = np.zeros(n_seg)  # trimmed from the start of segment i
        corner_info = {}
        for k in range(1, n_seg):
            d1, d2 = seg_dir[k - 1], seg_dir[k]
            phi = math.atan2(d1[0] * d2[1] - d1[1] * d2[0], float(d1 @ d2))
            if abs(phi) < 1e-3 or fillet_radius <= 0.0:
                continue
            t = fillet_radius * math.tan(abs(phi) / 2.0)
            t = min(t, 0.45 * seg_len[k - 1], 0.45 * seg_len[k])
            r_eff = t / math.tan(abs(phi) / 2.0)
            trim_in[k - 1] = max(trim_in[k - 1], t)
            trim_out[k] = max(trim_out[k], t)
            corner_info[k] = (phi, r_eff, t)

        xs, ys, ths, kaps = [], [], [], []

        def emit_line(p, direction, length):
            n = max(1, int(round(length / ds)))
            th = math.atan2(direction[1], direction[0])
            for i in range(n):
                q = p + direction * (length * i / n)
                xs.append(q[0]); ys.append(q[1]); ths.append(th); kaps.append(0.0)

        def emit_arc(p, th0, kappa, length):
            n = max(1, int(round(length / ds)))
            for i in range(n):
                a = length * i / n
                th = th0 + kappa * a
                x = p[0] + (math.sin(th) - math.sin(th0)) / kappa
                y = p[1] - (math.cos(th) - math.cos(th0)) / kappa
                xs.append(x); ys.append(y); ths.append(th); kaps.append(kappa)

        for k in range(n_seg):
            start = pts[k] + seg_dir[k] * trim_out[k]
            length = seg_len[k] - trim_out[k] - trim_in[k]
            if length > 1e-9:
                emit_line(start, seg_dir[k], length)
            if k + 1 in corner_info:
                phi, r_eff, t = corner_info[k + 1]
                arc_start = pts[k + 1] - seg_dir[k] * t
                kappa = math.copysign(1.0 / r_eff, phi)
                emit_arc(arc_start, math.atan2(seg_dir[k][1], seg_dir[k][0]), kappa, abs(phi) * r_eff)
        # Final point.
        xs.append(pts[-1][0]); ys.append(pts[-1][1])
        ths.append(math.atan2(seg_dir[-1][1], seg_dir[-1][0])); kaps.append(0.0)

        self.x = np.array(xs)
        self.y = np.array(ys)
        self.theta = np.unwrap(np.array(ths))
        self.kappa = np.array(kaps)
        step = np.hypot(np.diff(self.x), np.diff(self.y))
        self.s = np.concatenate([[0.0], np.cumsum(step)])
        self.length = float(self.s[-1])
        self.speed = np.full_like(self.s, np.nan)

    # -- speed profile -----------------------------------------------------
    def build_speed_profile(self, v_max, a_max, a_lat, w_max, v_min, v_zone=None, zone_length=0.0):
        #Forward/backward pass: v^2 limited by curvature, accel and a final slow zone.
        k = np.abs(self.kappa)
        v_lim = np.full_like(self.s, v_max)
        with np.errstate(divide="ignore"):
            v_lim = np.minimum(v_lim, np.where(k > 1e-6, np.sqrt(a_lat / np.maximum(k, 1e-6)), np.inf))
            v_lim = np.minimum(v_lim, np.where(k > 1e-6, w_max / np.maximum(k, 1e-6), np.inf))
        if v_zone is not None and zone_length > 0.0:
            v_lim[self.s >= self.length - zone_length] = np.minimum(
                v_lim[self.s >= self.length - zone_length], v_zone
            )
        v2 = v_lim**2
        v2[0] = min(v2[0], v_min**2)
        v2[-1] = 0.0
        ds = np.diff(self.s)
        for i in range(len(ds)):
            v2[i + 1] = min(v2[i + 1], v2[i] + 2.0 * a_max * ds[i])
        for i in range(len(ds) - 1, -1, -1):
            v2[i] = min(v2[i], v2[i + 1] + 2.0 * a_max * ds[i])
        self.speed = np.maximum(np.sqrt(v2), 0.0)
        self.v_min = v_min

    # -- queries -------------------------------------------------------------
    def _idx(self, s: float) -> int:
        return int(np.clip(round(s / self.ds), 0, len(self.s) - 1))

    def at(self, s: float):
        #Reference (pose[3], v_ref, w_ref) at arc length s.
        i = self._idx(s)
        if s >= self.length - 1e-9:
            return np.array([self.x[-1], self.y[-1], self.theta[-1]]), 0.0, 0.0
        v = max(self.speed[i], self.v_min)
        return np.array([self.x[i], self.y[i], self.theta[i]]), v, v * self.kappa[i]

    def speed_at(self, s: float) -> float:
        return float(max(self.speed[self._idx(s)], self.v_min))

    def points_ahead(self, s: float, horizon: float, step: float = 0.05) -> np.ndarray:
        idx = [self._idx(a) for a in np.arange(s, min(s + horizon, self.length) + 1e-9, step)]
        return np.stack([self.x[idx], self.y[idx]], axis=1)


# ======================================================================
# Navigator
# ======================================================================
@dataclass
class NavConfig:
    # Docking geometry (world frame).  The robot axle centre stops at
    # (dock_x, dock_stop_y) facing +y towards the marker.
    dock_x: float = 2.4
    dock_stop_y: float = 2.30
    station_face_y: float = 2.5  # marker / station front face (from the scene XML)
    dock_range_window: float = 0.3 # s of LiDAR samples averaged for the terminal range check
    approach_length: float = 1.1  # straight final leg before the stop pose
    dock_zone_length: float = 0.6  # slow zone at the end of the leg
    # Motion limits.  v_max leaves headroom under the 0.26 m/s wheel limit.
    v_max: float = 0.18
    v_dock: float = 0.07
    v_min_ref: float = 0.02
    a_max: float = 0.15
    a_lat: float = 0.10
    w_max: float = 1.0
    fillet_radius: float = 0.35
    # Tracking / virtual-target throttling.
    ex_hold: float = 0.15   # lag at which the virtual target stops advancing
    lost_error: float = 0.5    # m: replan if the position error exceeds this
    align_threshold: float = 0.6  # rad: rotate in place first if heading error is larger
    align_done: float = 0.12
    # LiDAR safety.
    stop_range: float = 0.32  # m from the LiDAR; front beams closer than this stop the robot
    slow_range: float = 0.70
    dock_stop_range: float = 0.15 # relaxed in the final dock zone (station is meant to be close)
    max_replans: int = 8
    settle_time: float = 3.0
    done_tolerance: float = 0.03
    # Map.
    use_prior_obstacles: bool = True
    inflation: float = 0.25
    resolution: float = 0.05


class State:
    PLAN = "PLAN"
    ALIGN = "ALIGN"
    TRACK = "TRACK"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass
class NavOutput:
    v: float
    w: float
    state: str
    s_ref: float = 0.0
    ref_pose: np.ndarray = field(default_factory=lambda: np.zeros(3))
    error: np.ndarray = field(default_factory=lambda: np.zeros(3))
    front_range: float = math.inf
    replans: int = 0


class Navigator:
    #Plan, track with LQR and dock, using the estimated pose and LiDAR only.

    def __init__(self, model: mujoco.MjModel, dt: float, config: NavConfig | None = None,
                 tracker: LQRTrajectoryTracker | None = None):
        self.cfg = config or NavConfig()
        self.dt = float(dt)
        self.tracker = tracker or LQRTrajectoryTracker(dt, v_max=self.cfg.v_max + 0.04, w_max=self.cfg.w_max)

        self.grid = OccupancyGrid(resolution=self.cfg.resolution, inflation=self.cfg.inflation)
        self.grid.add_boxes(boxes_from_model(model, include_obstacles=self.cfg.use_prior_obstacles))

        self.state = State.PLAN
        self.path: ReferencePath | None = None
        self.s_ref = 0.0
        self.replans = 0   # every replan, for logging
        self.recovery_replans = 0  # replans caused by being blocked or lost; these hit the cap
        self.settle_clock = 0.0
        self.stuck_clock = 0.0
        self._dock_leg_s0 = math.inf
        self.dock_entry = np.array([self.cfg.dock_x, self.cfg.dock_stop_y - self.cfg.approach_length])
        self.dock_stop = np.array([self.cfg.dock_x, self.cfg.dock_stop_y])
        self.history: list[str] = []
        # LiDAR range to the station face expected at the stop pose.
        self.dock_range = (self.cfg.station_face_y - self.cfg.dock_stop_y) + float(-LIDAR_OFFSET[0])
        self._dock_ranges: deque = deque(maxlen=max(1, int(round(self.cfg.dock_range_window / self.dt))))

    # ------------------------------------------------------------------
    def build_path(self, pose_est: np.ndarray) -> ReferencePath | None:
        #Plan from the estimated pose to the docking entry, then straight to the stop pose.
        cfg = self.cfg
        coarse = plan_path(self.grid, pose_est[:2], self.dock_entry)
        if coarse is None:
            return None
        # Straight docking leg from the entry point to the stop pose, facing +y.
        waypoints = np.vstack([coarse, self.dock_stop[None, :]])
        self._dock_leg_s0 = math.inf  # no docking-leg exemption while testing candidates
        for radius in (cfg.fillet_radius, 0.25, 0.15, 0.08):
            path = ReferencePath(waypoints, radius)
            path.build_speed_profile(cfg.v_max, cfg.a_max, cfg.a_lat, cfg.w_max, cfg.v_min_ref,
                                     cfg.v_dock, cfg.dock_zone_length)
            self._dock_leg_s0 = max(0.0, path.length - cfg.approach_length)
            if self._path_clear(path, 0.0):
                break
        # If no fillet radius clears the map the smallest one is used; the
        # inflation margin (robot radius + 0.03 m) absorbs a small violation.
        return path

    def _path_clear(self, path: ReferencePath, s_from: float) -> bool:
        #Is the non-docking part of the path ahead of s_from free of blocked cells?
        s_to = self._dock_leg_s0 - 0.05
        if s_from >= s_to:
            return True
        pts = path.points_ahead(s_from, s_to - s_from, step=0.05)
        return not any(self.grid.point_blocked(x, y) for x, y in pts)

    # ------------------------------------------------------------------
    def _replan(self, pose_est: np.ndarray, recovery: bool = True) -> bool:
        #Plan again from the estimated pose.

        self.replans += 1
        self.recovery_replans += int(recovery)
        if self.recovery_replans > self.cfg.max_replans or self.replans > 10 * self.cfg.max_replans:
            self.state = State.FAILED
            self.history.append("failed: replan limit")
            return False
        path = self.build_path(pose_est)
        if path is None:
            self.state = State.FAILED
            self.history.append("failed: no path")
            return False
        self.path = path
        self.s_ref = 0.0
        self._dock_ranges.clear()
        head_err = abs(float(wrap_angle(path.theta[0] - pose_est[2])))
        self.state = State.ALIGN if head_err > self.cfg.align_threshold else State.TRACK
        self.history.append(f"replan #{self.replans}")
        return True

    def update(self, pose_est: np.ndarray, lidar_ranges: np.ndarray) -> NavOutput:
        #One control step.  Returns the commanded twist (v, w).
        cfg = self.cfg
        pose_est = np.asarray(pose_est, float)
        front = sector_min(lidar_ranges, -30, 30)

        # --- map update from LiDAR (estimated pose, not ground truth) ------------
        # Free space first (beams pass through ghost cells left by earlier pose error),
        # then new obstacles.
        self.grid.clear_rays(pose_est[:2] + _rot(pose_est[2]) @ LIDAR_OFFSET,
                             scan_to_world(pose_est, lidar_ranges, hits_only=False))
        pts = scan_to_world(pose_est, lidar_ranges)
        if len(pts):
            pts = pts[np.hypot(pts[:, 0] - pose_est[0], pts[:, 1] - pose_est[1]) > 0.1]
            new_cells = self.grid.mark_points(pts)
        else:
            new_cells = 0

        if self.state == State.PLAN:
            if not self._replan(pose_est):
                return NavOutput(0.0, 0.0, self.state, front_range=front, replans=self.replans)
            self.replans = 0  # the first plan is not a *re*plan
            self.recovery_replans = 0
            self.history[-1] = "initial plan"

        if self.state in (State.DONE, State.FAILED):
            return NavOutput(0.0, 0.0, self.state, self.s_ref, front_range=front, replans=self.replans)

        path = self.path
        ref_pose, v_ref, w_ref = path.at(self.s_ref)

        # --- replan if newly discovered obstacles block the path -----------------
        if new_cells and not self._path_clear(path, self.s_ref):
            if not self._replan(pose_est, recovery=False):
                return NavOutput(0.0, 0.0, self.state, front_range=front, replans=self.replans)
            path = self.path
            ref_pose, v_ref, w_ref = path.at(self.s_ref)

        # --- ALIGN: rotate in place until roughly facing the path ---------------------
        if self.state == State.ALIGN:
            e_th = float(wrap_angle(ref_pose[2] - pose_est[2]))
            if abs(e_th) < cfg.align_done:
                self.state = State.TRACK
            else:
                w = float(np.clip(1.5 * e_th, -0.8, 0.8))
                if abs(w) < 0.15:
                    w = math.copysign(0.15, w)
                return NavOutput(0.0, w, self.state, self.s_ref, ref_pose,
                                 np.array([0.0, 0.0, e_th]), front, self.replans)

        # --- TRACK ------------------------------------------------------------------
        in_dock_leg = self.s_ref >= self._dock_leg_s0
        stop_range = cfg.dock_stop_range if in_dock_leg else cfg.stop_range
        slow_range = stop_range + (cfg.slow_range - cfg.stop_range) * (0.4 if in_dock_leg else 1.0)
        lidar_scale = float(np.clip((front - stop_range) / max(slow_range - stop_range, 1e-6), 0.0, 1.0))

        # Terminal condition from the LiDAR: the range to the station face is a direct
        # measurement, so the along-track stop does not depend on odometry drift.
        if in_dock_leg:
            d_fwd = forward_distance(lidar_ranges)
            if np.isfinite(d_fwd):
                self._dock_ranges.append(d_fwd)
            if (len(self._dock_ranges) == self._dock_ranges.maxlen
                    and float(np.mean(self._dock_ranges)) <= self.dock_range):
                self.state = State.DONE
                self.history.append("docked (LiDAR range to station)")
                v, w, e = self.tracker.control(pose_est, ref_pose, v_ref, w_ref)
                return NavOutput(0.0, 0.0, self.state, self.s_ref, ref_pose, e, front, self.replans)

        v, w, e = self.tracker.control(pose_est, ref_pose, v_ref, w_ref)
        v = min(v, cfg.v_max * max(lidar_scale, 0.0)) if lidar_scale < 1.0 else v

        # Virtual target: advance with the speed profile, throttled by lag and LiDAR.
        lag_scale = float(np.clip(1.0 - max(e[0], 0.0) / cfg.ex_hold, 0.0, 1.0))
        if self.s_ref < path.length:
            self.s_ref = min(path.length, self.s_ref + path.speed_at(self.s_ref) * self.dt * min(lag_scale, lidar_scale))

        # Blocked in front for a while.  Away from the dock the LiDAR map already holds the
        # obstacle, so replan around it.  In the docking leg the robot is lined up on the
        # station and the range check above did not fire, so it is not at the dock face
        # (e.g. it arrived beside the station): report failure instead of waiting forever.
        if lidar_scale <= 0.0:
            self.stuck_clock += self.dt
            if self.stuck_clock > 1.0:
                self.stuck_clock = 0.0
                if in_dock_leg:
                    self.state = State.FAILED
                    self.history.append("failed: blocked in docking leg, not at the station face")
                else:
                    self._replan(pose_est)
                return NavOutput(0.0, 0.0, self.state, self.s_ref, ref_pose, e, front, self.replans)
        else:
            self.stuck_clock = 0.0

        if math.hypot(e[0], e[1]) > cfg.lost_error and not in_dock_leg:
            self._replan(pose_est)
            return NavOutput(0.0, 0.0, self.state, self.s_ref, ref_pose, e, front, self.replans)

        # --- finish: hold at the end of the path until settled ---------------------------
        if self.s_ref >= path.length - 1e-9:
            self.settle_clock += self.dt
            pos_err = math.hypot(e[0], e[1])
            if pos_err < cfg.done_tolerance or self.settle_clock > cfg.settle_time:
                self.state = State.DONE
                self.history.append("docked (estimated pose)")
                return NavOutput(0.0, 0.0, self.state, self.s_ref, ref_pose, e, front, self.replans)
            # Hold: only the forward error is correctable at v_ref = 0.
            v = float(np.clip(1.0 * e[0], 0.0, cfg.v_dock))
            w = float(np.clip(1.5 * e[2], -0.3, 0.3))

        return NavOutput(v, w, self.state, self.s_ref, ref_pose, e, front, self.replans)