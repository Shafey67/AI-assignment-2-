"""Central configuration (all tunable parameters live here)."""
from dataclasses import dataclass, replace


@dataclass
class Config:
    # ---- Mission / fleet ------------------------------------------------
    grid_size: int = 50            # 50 x 50 discretised hazard grid
    n_uavs: int = 4                # fleet size K
    n_waypoints: int = 8           # waypoints per UAV  L
    max_range: float = 110.0       # hard limit: path length per UAV incl. return (cells)
    max_time: float = 125.0        # hard limit: mission deadline per UAV (time units)

    # ---- Gene bounds -------------------------------------------------------
    speed_bounds: tuple = (0.5, 2.0)    # cells / time-unit
    power_bounds: tuple = (0.2, 1.0)    # normalised sensor power
    sensor_r_min: float = 1.0           # sensor radius = r_min + r_gain * power
    sensor_r_gain: float = 3.0

    # ---- Physical model ----------------------------------------------------
    sample_step: float = 1.0       # path sampling resolution (cells)
    e_fly: float = 1.0             # energy per cell at zero drag
    e_drag: float = 0.5            # energy per cell: e_fly*(1 + e_drag*v^2)
    e_sensor: float = 1.2          # sensor energy: e_sensor * power^2 * time

    # ---- Fitness weights (maximise coverage, minimise the rest) -----------
    w_cov: float = 0.55
    w_energy: float = 0.15
    w_time: float = 0.10
    w_hazard: float = 0.20
    lam_range: float = 4.0         # penalty multipliers for hard constraints
    lam_time: float = 4.0
    lam_nofly: float = 4.0

    # ---- Genetic algorithm ------------------------------------------------
    pop_size: int = 100
    generations: int = 300
    tournament_k: int = 3          # selection pressure
    elitism: int = 2
    p_crossover: float = 0.9
    p_mutation: float = 0.08       # per-waypoint mutation probability
    sigma_frac: float = 0.12       # Gaussian step as a fraction of grid size
    adaptive_mutation: bool = True # boost mutation when diversity collapses
    immigrant_frac: float = 0.05   # random immigrants per generation
    p_repair: float = 0.5          # probability range/time repair is applied
    early_stop: bool = True
    patience: int = 80             # stagnation termination (generations)
    tol: float = 1e-4

    seed: int = 0

    def but(self, **kw):
        """Return a modified copy."""
        return replace(self, **kw)
