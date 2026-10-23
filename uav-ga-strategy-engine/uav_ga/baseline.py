"""Random-search baseline with the same sampler, repair and evaluation budget."""
import numpy as np
from .fitness import Evaluator
from .ga import make_individual


def random_search(env, cfg, n_evals, record_every=None, seed=None):
    """Draw n_evals random (repaired) individuals; return best + best-so-far trace."""
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    ev = Evaluator(env, cfg)
    best, trace = None, []
    for i in range(n_evals):
        c = make_individual(env, cfg, rng, ev)
        if best is None or c.fitness > best.fitness:
            best = c
        if record_every and (i + 1) % record_every == 0:
            trace.append(best.fitness)
    return best, trace
