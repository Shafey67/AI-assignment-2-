"""Multi-objective fitness: weighted-sum scalarisation + penalty constraints."""
import numpy as np


class Evaluator:
    """
    F = w_cov*C - w_energy*E - w_time*T - w_hazard*H - penalties

        C  priority-weighted detection rate in [0,1]
        E  total energy / reference energy (K UAVs flying full range, nominal speed, sensors on)
        T  mission makespan / deadline
        H  hazard exposure integral / reference exposure
    Penalties (hard constraints): range overrun, deadline overrun, no-fly-zone
    flight length, each normalised and multiplied by lam_*.
    """

    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg
        K, L = cfg.n_uavs, cfg.n_waypoints
        self.base_k = np.tile(env.base, (K, 1, 1))
        self.e_ref = K * (cfg.max_range * cfg.e_fly * (1 + cfg.e_drag) + cfg.e_sensor * cfg.max_range)
        self.h_ref = 0.5 * K * cfg.max_range
        # disk stencil for the sensor footprint
        R = int(np.ceil(cfg.sensor_r_min + cfg.sensor_r_gain * cfg.power_bounds[1]))
        d = np.arange(-R, R + 1)
        dx, dy = np.meshgrid(d, d)
        self.dx, self.dy = dx.ravel(), dy.ravel()
        self.d2 = (self.dx ** 2 + self.dy ** 2).astype(float)
        self.n_evals = 0

    # ------------------------------------------------------------------
    def samples(self, ch):
        """Densely sample every leg of every route."""
        cfg = self.cfg
        pts = np.concatenate([self.base_k, ch.xy, self.base_k], axis=1)
        a, b = pts[:, :-1].reshape(-1, 2), pts[:, 1:].reshape(-1, 2)
        d = np.linalg.norm(b - a, axis=1)
        n = np.maximum(1, np.ceil(d / cfg.sample_step).astype(int))
        seg = np.repeat(np.arange(len(d)), n)
        start = np.cumsum(n) - n
        j = np.arange(n.sum()) - np.repeat(start, n)
        t = (j + 0.5) / np.repeat(n, n)
        P = a[seg] + t[:, None] * (b - a)[seg]
        return P, seg, d, d / n

    def _leg_params(self, ch):
        K = ch.K
        one, zero = np.ones((K, 1)), np.zeros((K, 1))
        v = np.concatenate([ch.speed, one], 1).ravel()
        p = np.concatenate([ch.power, zero], 1).ravel()
        on = np.concatenate([ch.active, zero.astype(bool)], 1).ravel()
        return v, p, on

    def detection_grid(self, P, seg, v, p, on, uav_of_seg=None, which=None):
        """Detection-quality grid (max over passes). Optionally restricted to one UAV."""
        cfg, S = self.cfg, self.cfg.grid_size
        grid = np.zeros(S * S)
        m = on[seg]
        if which is not None:
            m = m & (uav_of_seg[seg] == which)
        if not m.any():
            return grid.reshape(S, S)
        Ps, ss = P[m], seg[m]
        r = cfg.sensor_r_min + cfg.sensor_r_gain * p[ss]
        q = np.clip(p[ss] * np.sqrt(1.0 / v[ss]), 0.0, 1.0)   # slower + stronger sensor -> better
        cx, cy = np.rint(Ps[:, 0]).astype(int), np.rint(Ps[:, 1]).astype(int)
        X = cx[:, None] + self.dx[None, :]
        Y = cy[:, None] + self.dy[None, :]
        ok = (self.d2[None, :] <= (r ** 2)[:, None]) & (X >= 0) & (X < S) & (Y >= 0) & (Y < S)
        np.maximum.at(grid, (Y * S + X)[ok], np.broadcast_to(q[:, None], ok.shape)[ok])
        return grid.reshape(S, S)

    # ------------------------------------------------------------------
    def evaluate(self, ch):
        cfg, env = self.cfg, self.env
        K, L = ch.K, ch.L
        P, seg, d, ds = self.samples(ch)
        v, p, on = self._leg_params(ch)

        grid = self.detection_grid(P, seg, v, p, on)
        C = float((env.priority * grid).sum() / env.total_priority)

        # energy
        e_leg = d * cfg.e_fly * (1 + cfg.e_drag * v ** 2) + on * cfg.e_sensor * p ** 2 * (d / v)
        E = float(e_leg.sum() / self.e_ref)

        # time / range per UAV
        t_uav = (d / v).reshape(K, L + 1).sum(1)
        len_uav = d.reshape(K, L + 1).sum(1)
        T = float(t_uav.max() / cfg.max_time)

        # hazard exposure and no-fly flight length
        ry, rx = env.cell_of(P)
        H = float((env.hazard[ry, rx] * ds[seg]).sum() / self.h_ref)
        nf_len = float((env.nofly[ry, rx] * ds[seg]).sum())

        v_range = float(np.maximum(0, len_uav - cfg.max_range).sum() / cfg.max_range)
        v_time = float(np.maximum(0, t_uav - cfg.max_time).sum() / cfg.max_time)
        v_nofly = nf_len / cfg.max_range
        penalty = cfg.lam_range * v_range + cfg.lam_time * v_time + cfg.lam_nofly * v_nofly

        ch.fitness = (cfg.w_cov * C - cfg.w_energy * E - cfg.w_time * T
                      - cfg.w_hazard * H - penalty)
        ch.metrics = dict(coverage=C, energy=E, time=T, hazard=H, penalty=penalty,
                          range_viol=v_range, time_viol=v_time, nofly_len=nf_len,
                          feasible=bool(penalty < 1e-9))
        self.n_evals += 1
        return ch.fitness

    # ------------------------------------------------------------------
    def analyze(self, ch):
        """Detailed per-UAV picture of a solution (used for plots / reports)."""
        K, L = ch.K, ch.L
        P, seg, d, ds = self.samples(ch)
        v, p, on = self._leg_params(ch)
        uav_of_seg = np.repeat(np.arange(K), L + 1)
        grids = [self.detection_grid(P, seg, v, p, on, uav_of_seg, k) for k in range(K)]
        total = np.max(grids, axis=0)
        cnt = (np.array(grids) > 0).sum(0)
        covered = cnt >= 1
        hi = self.env.priority >= 0.5
        info = dict(
            grid=total, per_uav=grids,
            overlap=float((cnt >= 2).sum() / max(1, covered.sum())),
            hi_cov=float((total[hi] > 0).mean()),
            hi_det=float(total[hi].mean()),
            len_uav=d.reshape(K, L + 1).sum(1),
            time_uav=(d / v).reshape(K, L + 1).sum(1),
            samples=P, seg=seg,
        )
        return info
