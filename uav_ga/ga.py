"""Object-oriented Genetic Algorithm engine."""
import numpy as np
from .chromosome import Chromosome
from .fitness import Evaluator
from . import operators as op
from .constraints import repair_bounds_and_nofly, repair_range_time


def make_individual(env, cfg, rng, ev):
    ch = Chromosome.random(env, cfg, rng)
    repair_bounds_and_nofly(ch, env, cfg)
    repair_range_time(ch, env, cfg, rng)
    ev.evaluate(ch)
    return ch


class GeneticAlgorithm:
    def __init__(self, env, cfg, snapshot_gens=(), verbose=False):
        self.env, self.cfg = env, cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.ev = Evaluator(env, cfg)
        self.snapshot_gens = set(snapshot_gens)
        self.verbose = verbose
        self.history = dict(best=[], mean=[], worst=[], diversity=[], mut_rate=[], evals=[],
                            feasible_frac=[])
        self.snapshots = {}
        self.best = None
        self.stop_reason = ""

    # ------------------------------------------------------------------
    def _record(self, pop, g, rate):
        f = np.array([c.fitness for c in pop])
        h = self.history
        h["best"].append(float(f.max())); h["mean"].append(float(f.mean()))
        h["worst"].append(float(f.min()))
        h["diversity"].append(op.diversity(pop, self.cfg.grid_size, self.rng))
        h["mut_rate"].append(rate)
        h["evals"].append(self.ev.n_evals)
        h["feasible_frac"].append(float(np.mean([c.metrics["feasible"] for c in pop])))
        if g in self.snapshot_gens:
            self.snapshots[g] = max(pop, key=lambda c: c.fitness).copy()

    def run(self):
        cfg, env, rng = self.cfg, self.env, self.rng
        N = cfg.pop_size
        pop = [make_individual(env, cfg, rng, self.ev) for _ in range(N)]
        n_imm = int(round(cfg.immigrant_frac * N)) if cfg.immigrant_frac > 0 else 0
        rate, best_f, stall = cfg.p_mutation, -np.inf, 0
        div0 = None

        for g in range(cfg.generations):
            pop.sort(key=lambda c: -c.fitness)
            self._record(pop, g, rate)
            if div0 is None:
                div0 = self.history["diversity"][0]
            self.best = pop[0]

            # ---- termination criteria
            if pop[0].fitness > best_f + cfg.tol:
                best_f, stall = pop[0].fitness, 0
            else:
                stall += 1
            if cfg.early_stop and stall >= cfg.patience:
                self.stop_reason = f"stagnation ({cfg.patience} gens) at generation {g}"
                break

            # ---- adaptive mutation: respond to diversity collapse
            rel_div = self.history["diversity"][-1] / div0
            boost = 1.0
            if cfg.adaptive_mutation and rel_div < 0.35:
                boost = 2.5
            rate = min(0.4, cfg.p_mutation * boost)
            sigma = cfg.sigma_frac * cfg.grid_size * (1 - 0.6 * g / cfg.generations) * (1.5 if boost > 1 else 1)

            # ---- reproduction
            new = [c.copy() for c in pop[:cfg.elitism]]
            kids = []
            while len(new) + len(kids) < N - n_imm:
                p1 = op.tournament(pop, cfg.tournament_k, rng)
                p2 = op.tournament(pop, cfg.tournament_k, rng)
                if rng.random() < cfg.p_crossover:
                    c1, c2 = op.crossover(p1, p2, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                for c in (c1, c2):
                    op.mutate(c, env, cfg, rng, rate, sigma)
                    repair_bounds_and_nofly(c, env, cfg)
                    if rng.random() < cfg.p_repair:
                        repair_range_time(c, env, cfg, rng)
                    kids.append(c)
            kids = kids[: N - n_imm - len(new)]
            for c in kids:
                self.ev.evaluate(c)
            imm = [make_individual(env, cfg, rng, self.ev) for _ in range(n_imm)]
            pop = new + kids + imm

            if self.verbose and g % 10 == 0:
                print(f"gen {g:4d} best {self.history['best'][-1]: .4f} "
                      f"mean {self.history['mean'][-1]: .4f} div {self.history['diversity'][-1]:.3f}")

        pop.sort(key=lambda c: -c.fitness)
        self.best = pop[0]
        self._record(pop, len(self.history["best"]), rate)
        self.snapshots["final"] = pop[0].copy()
        if not self.stop_reason:
            self.stop_reason = f"max generations ({cfg.generations}) reached"
        self.final_population = pop
        return self.best
