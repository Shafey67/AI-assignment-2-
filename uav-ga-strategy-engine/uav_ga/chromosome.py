"""Hybrid real-valued / binary chromosome for a UAV fleet."""
import numpy as np


class Chromosome:
    """
    Encoding (K UAVs x L waypoints, 5 genes per waypoint = 5*K*L genes):

        xy     (K, L, 2)  real     waypoint coordinates (x, y); waypoint 0 is the
                                    UAV's deployment position (first loiter point)
        speed  (K, L)     real     cruise speed on the leg that ARRIVES at waypoint l
        power  (K, L)     real     sensor power profile on that leg
        active (K, L)     binary   sensor activation schedule for that leg

    Route flown: BASE -> wp_0 -> ... -> wp_{L-1} -> BASE (return leg at nominal
    speed, sensors off). Waypoint order is part of the genome, so routing and
    deployment are optimised jointly.
    """

    __slots__ = ("xy", "speed", "power", "active", "fitness", "metrics")

    def __init__(self, xy, speed, power, active):
        self.xy, self.speed, self.power, self.active = xy, speed, power, active
        self.fitness = -np.inf
        self.metrics = {}

    @property
    def K(self):
        return self.xy.shape[0]

    @property
    def L(self):
        return self.xy.shape[1]

    def copy(self):
        c = Chromosome(self.xy.copy(), self.speed.copy(), self.power.copy(), self.active.copy())
        c.fitness, c.metrics = self.fitness, dict(self.metrics)
        return c

    def vector(self, size):
        """Flat normalised vector (population-diversity metric)."""
        return np.concatenate([(self.xy / size).ravel(), self.speed.ravel() / 2.0, self.power.ravel()])

    @staticmethod
    def random(env, cfg, rng):
        """Random individual: ~60% of waypoints seeded in priority zones, rest uniform."""
        K, L, S = cfg.n_uavs, cfg.n_waypoints, cfg.grid_size
        xy = rng.uniform(0, S - 1, (K, L, 2))
        seeded = rng.random((K, L)) < 0.6
        n = int(seeded.sum())
        if n:
            xy[seeded] = env.sample_cell(rng, n)
        speed = rng.uniform(*cfg.speed_bounds, (K, L))
        power = rng.uniform(*cfg.power_bounds, (K, L))
        active = rng.random((K, L)) < 0.8
        return Chromosome(xy, speed, power, active)
