"""Discretised 2-D hazard grid: priorities, hazards, no-fly zones, base."""
import numpy as np
from scipy import ndimage


class Environment:
    """Grid world. Arrays are indexed [y, x]; points are (x, y)."""

    def __init__(self, size: int = 50):
        self.size = size
        yy, xx = np.mgrid[0:size, 0:size].astype(float)
        s = size / 50.0

        # --- priority map: background 0.1 + disaster zones ------------------
        self.priority = np.full((size, size), 0.1)
        zones = [(12, 38, 7, 1.0), (38, 36, 8, 0.8), (40, 12, 6, 0.6), (22, 22, 5, 0.9)]
        for cx, cy, r, lvl in zones:
            m = (xx - cx * s) ** 2 + (yy - cy * s) ** 2 <= (r * s) ** 2
            self.priority[m] = np.maximum(self.priority[m], lvl)

        # --- hazard map (wind / radiation pockets), values in [0, 1] -------
        self.hazard = np.zeros((size, size))
        for cx, cy, sig, amp in [(36, 34, 5, 0.9), (20, 30, 4, 0.7), (30, 10, 4.5, 0.6)]:
            self.hazard += amp * np.exp(-((xx - cx * s) ** 2 + (yy - cy * s) ** 2) / (2 * (sig * s) ** 2))
        self.hazard = np.clip(self.hazard, 0, 1)

        # --- no-fly zones (binary mask: True = blocked) -----------------------
        self.nofly = np.zeros((size, size), dtype=bool)
        self.nofly[int(14 * s):int(27 * s), int(26 * s):int(33 * s)] = True      # airport / hospital
        self.nofly[((xx - 8 * s) ** 2 + (yy - 22 * s) ** 2) <= (4 * s) ** 2] = True
        self.nofly[int(24 * s):int(33 * s), int(44 * s):int(50 * s)] = True      # restricted airspace

        self.base = np.array([4.0 * s, 4.0 * s])
        self.priority[self.nofly] = 0.0   # nothing to survey in a closed zone

        # nearest free cell lookup (used by the repair operator)
        _, (iy, ix) = ndimage.distance_transform_edt(self.nofly, return_indices=True)
        self.nearest = np.stack([ix, iy])           # (2, S, S): x,y of nearest free cell

        # priority-proportional sampling distribution (spatially-aware mutation / init)
        p = self.priority.ravel()
        self.cell_p = p / p.sum()
        self.total_priority = self.priority.sum()

    def sample_cell(self, rng, n=1):
        """Draw cell centres with probability proportional to priority."""
        idx = rng.choice(self.cell_p.size, size=n, p=self.cell_p)
        y, x = np.divmod(idx, self.size)
        return np.stack([x, y], axis=-1).astype(float) + rng.uniform(-0.5, 0.5, (n, 2))

    def cell_of(self, pts):
        c = np.clip(np.rint(pts).astype(int), 0, self.size - 1)
        return c[..., 1], c[..., 0]   # row (y), col (x)
