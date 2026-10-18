"""Constraint handling: repair mechanisms (penalties live in fitness.py)."""
import numpy as np


def repair_bounds_and_nofly(ch, env, cfg):
    """Always applied. Clip to the grid, clip gene ranges, snap waypoints that land in a
    no-fly cell to the nearest free cell (Lamarckian repair)."""
    S = cfg.grid_size
    np.clip(ch.xy, 0, S - 1, out=ch.xy)
    np.clip(ch.speed, *cfg.speed_bounds, out=ch.speed)
    np.clip(ch.power, *cfg.power_bounds, out=ch.power)
    ry, rx = env.cell_of(ch.xy)
    bad = env.nofly[ry, rx]
    if bad.any():
        ch.xy[bad] = env.nearest[:, ry[bad], rx[bad]].T.astype(float)
    return ch


def _route_stats(ch, env):
    pts = np.concatenate([np.tile(env.base, (ch.K, 1, 1)), ch.xy, np.tile(env.base, (ch.K, 1, 1))], 1)
    d = np.linalg.norm(np.diff(pts, axis=1), axis=2)
    v = np.concatenate([ch.speed, np.ones((ch.K, 1))], 1)
    return d.sum(1), (d / v).sum(1)


def repair_range_time(ch, env, cfg, rng):
    """Probabilistic repair of the range / deadline limits.
    * range: geometric contraction of the route towards the base (keeps route shape)
    * time : raise the cruise speed of the slow legs
    Residual violations are still punished by the penalty function."""
    for _ in range(3):
        length, time = _route_stats(ch, env)
        over = length > cfg.max_range
        if not over.any():
            break
        for k in np.where(over)[0]:
            s = max(0.5, 0.97 * cfg.max_range / length[k])
            ch.xy[k] = env.base + s * (ch.xy[k] - env.base)
        repair_bounds_and_nofly(ch, env, cfg)
    length, time = _route_stats(ch, env)
    for k in np.where(time > cfg.max_time)[0]:
        f = min(2.0, 1.05 * time[k] / cfg.max_time)
        ch.speed[k] = np.clip(ch.speed[k] * f, *cfg.speed_bounds)
    return ch
