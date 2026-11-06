#!/usr/bin/env python3
"""Randomize obstacle geoms in a loaded MuJoCo model without writing XML."""

from __future__ import annotations

import math
import random

import mujoco
import numpy as np


ROOM_LIMIT = 2.45
MIN_SEPARATION = 0.18
PROTECTED_ZONES = ((0.0, 0.0, 0.65), (2.4, 2.5, 0.8))


def _overlaps(x, y, hx, hy, placed):
    for px, py, phx, phy in placed:
        if abs(x - px) < hx + phx + MIN_SEPARATION and abs(y - py) < hy + phy + MIN_SEPARATION:
            return True
    return any(
        math.hypot(x - px, y - py) < radius + max(hx, hy)
        for px, py, radius in PROTECTED_ZONES
    )


def _randomize_obstacle(model, geom_id, rng, placed):
    kind = rng.choice(
        (
            mujoco.mjtGeom.mjGEOM_BOX,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            mujoco.mjtGeom.mjGEOM_SPHERE,
        )
    )
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        hx = rng.uniform(0.25, 0.55)
        hy = rng.uniform(0.25, 0.55)
        size = (hx, hy, 0.3)
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        hx = hy = rng.uniform(0.25, 0.45)
        size = (hx, 0.3, 0.0)
    else:
        hx = hy = rng.uniform(0.25, 0.45)
        size = (hx, 0.0, 0.0)

    for _ in range(500):
        x = rng.uniform(-ROOM_LIMIT + hx, ROOM_LIMIT - hx)
        y = rng.uniform(-ROOM_LIMIT + hy, ROOM_LIMIT - hy)
        if not _overlaps(x, y, hx, hy, placed):
            break
    else:
        raise RuntimeError("could not place all obstacles without overlap")

    placed.append((x, y, hx, hy))
    model.geom_type[geom_id] = kind
    model.geom_pos[geom_id] = (x, y, 0.3)
    model.geom_size[geom_id] = size
    model.geom_rgba[geom_id] = (rng.random(), rng.random(), rng.random(), 1.0)
    yaw = rng.uniform(-math.pi, math.pi)
    model.geom_quat[geom_id] = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def randomize_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData | None = None,
    seed: int = 0,
    count: int = 3,
) -> None:
    """Randomize existing ``obstacle_*`` geoms in memory and forward the model."""
    if count < 0:
        raise ValueError("count must be non-negative")

    obstacles = sorted(
        (
            geom_id
            for geom_id in range(model.ngeom)
            if (
                (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
                .startswith("obstacle_")
            )
        ),
        key=lambda geom_id: mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        ),
    )
    if count > len(obstacles):
        raise ValueError(
            f"count={count} exceeds existing obstacle count={len(obstacles)}"
        )

    rng = random.Random(seed)
    placed = []
    for geom_id in obstacles[:count]:
        _randomize_obstacle(model, geom_id, rng, placed)

    mujoco.mj_forward(model, data if data is not None else mujoco.MjData(model))
