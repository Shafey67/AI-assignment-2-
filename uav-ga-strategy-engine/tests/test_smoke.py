"""Run with:  python -m pytest -q   (or just: python tests/test_smoke.py)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from uav_ga import Config, Environment, GeneticAlgorithm, Chromosome, Evaluator
from uav_ga.constraints import repair_bounds_and_nofly, repair_range_time
from uav_ga.operators import crossover, mutate


def setup():
    return Environment(), Config(pop_size=20, generations=5), np.random.default_rng(0)


def test_chromosome_shapes():
    env, cfg, rng = setup()
    ch = Chromosome.random(env, cfg, rng)
    assert ch.xy.shape == (cfg.n_uavs, cfg.n_waypoints, 2)
    assert ch.active.dtype == bool


def test_repair_removes_nofly_waypoints_and_respects_range():
    env, cfg, rng = setup()
    ch = Chromosome.random(env, cfg, rng)
    repair_bounds_and_nofly(ch, env, cfg)
    ry, rx = env.cell_of(ch.xy)
    assert not env.nofly[ry, rx].any()
    repair_range_time(ch, env, cfg, rng)
    ev = Evaluator(env, cfg); ev.evaluate(ch)
    assert ch.metrics["range_viol"] < 0.2


def test_operators_keep_shapes():
    env, cfg, rng = setup()
    a, b = Chromosome.random(env, cfg, rng), Chromosome.random(env, cfg, rng)
    c1, c2 = crossover(a, b, rng)
    mutate(c1, env, cfg, rng, 0.5, 5.0)
    assert c1.xy.shape == a.xy.shape and c2.speed.shape == a.speed.shape


def test_ga_improves():
    env, cfg, _ = setup()
    ga = GeneticAlgorithm(env, cfg.but(generations=15)); ga.run()
    assert ga.history["best"][-1] >= ga.history["best"][0]


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
