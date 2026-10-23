"""Selection, spatial crossover and spatial mutation operators."""
import numpy as np
from .chromosome import Chromosome


# ---------------------------------------------------------------- selection
def tournament(pop, k, rng):
    """k-tournament selection: larger k = higher selection pressure."""
    idx = rng.integers(0, len(pop), k)
    return pop[max(idx, key=lambda i: pop[i].fitness)]


# --------------------------------------------------------------- crossover
def crossover(p1, p2, rng):
    """Per-UAV hybrid crossover. For every UAV one of three operators is chosen:

      0  route swap       - the two parents exchange the entire route of that UAV
      1  segment swap     - a contiguous waypoint segment (all genes) is exchanged
      2  geometric blend  - child waypoint lies on the line between the parents'
                            waypoints (alpha ~ U(-0.1, 1.1)); speed/power blended,
                            sensor-activation bit inherited uniformly
    """
    c1, c2 = p1.copy(), p2.copy()
    K, L = p1.K, p1.L
    for k in range(K):
        op = rng.integers(0, 3)
        if op == 0:
            for g in ("xy", "speed", "power", "active"):
                a, b = getattr(c1, g), getattr(c2, g)
                a[k], b[k] = b[k].copy(), a[k].copy()
        elif op == 1:
            i, j = sorted(rng.integers(0, L + 1, 2))
            if j == i:
                j = min(L, i + 1)
            for g in ("xy", "speed", "power", "active"):
                a, b = getattr(c1, g), getattr(c2, g)
                a[k, i:j], b[k, i:j] = b[k, i:j].copy(), a[k, i:j].copy()
        else:
            al = rng.uniform(-0.1, 1.1, (L, 1))
            xy1 = al * p1.xy[k] + (1 - al) * p2.xy[k]
            xy2 = (1 - al) * p1.xy[k] + al * p2.xy[k]
            c1.xy[k], c2.xy[k] = xy1, xy2
            a = al[:, 0]
            c1.speed[k] = a * p1.speed[k] + (1 - a) * p2.speed[k]
            c2.speed[k] = (1 - a) * p1.speed[k] + a * p2.speed[k]
            c1.power[k] = a * p1.power[k] + (1 - a) * p2.power[k]
            c2.power[k] = (1 - a) * p1.power[k] + a * p2.power[k]
            m = rng.random(L) < 0.5
            c1.active[k] = np.where(m, p1.active[k], p2.active[k])
            c2.active[k] = np.where(m, p2.active[k], p1.active[k])
    return c1, c2


# ---------------------------------------------------------------- mutation
def mutate(ch, env, cfg, rng, rate, sigma):
    """Spatially-aware mutation, applied independently to every waypoint with prob `rate`:

      gaussian shift  (50%)  small displacement of the waypoint (local refinement)
      priority jump   (20%)  relocate to a cell drawn ~ priority map (explore zones)
      order swap      (10%)  swap two waypoints of the UAV (re-route / un-tangle)
      param mutation  (20%)  Gaussian on speed & sensor power, flip activation bit
    """
    K, L = ch.K, ch.L
    hit = rng.random((K, L)) < rate
    for k, l in zip(*np.where(hit)):
        r = rng.random()
        if r < 0.5:
            ch.xy[k, l] += rng.normal(0, sigma, 2)
        elif r < 0.7:
            ch.xy[k, l] = env.sample_cell(rng, 1)[0]
        elif r < 0.8:
            m = rng.integers(0, L)
            for g in ("xy", "speed", "power", "active"):
                a = getattr(ch, g)
                a[k, [l, m]] = a[k, [m, l]]
        else:
            ch.speed[k, l] += rng.normal(0, 0.25)
            ch.power[k, l] += rng.normal(0, 0.12)
            if rng.random() < 0.3:
                ch.active[k, l] = ~ch.active[k, l]
    return ch


def diversity(pop, size, rng, n_pairs=60):
    """Mean normalised distance between random pairs of individuals."""
    V = np.array([c.vector(size) for c in pop])
    i = rng.integers(0, len(V), n_pairs)
    j = rng.integers(0, len(V), n_pairs)
    return float(np.mean(np.linalg.norm(V[i] - V[j], axis=1)))
