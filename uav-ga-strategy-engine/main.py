"""Run one full GA mission-planning simulation and write figures to results/figures/.

    python main.py                       # opens each graph in a window + saves PNGs; default (pop 100, 300 generations)
    python main.py --pop 60 --gens 100   # quick run
"""
import argparse, json, os
from uav_ga import Config, Environment, GeneticAlgorithm
from uav_ga.fitness import Evaluator
from uav_ga import visualize as viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-show", action="store_true", help="only save PNGs, do not open windows")
    a = ap.parse_args()

    viz.SHOW = not a.no_show
    fig_dir = os.path.join(a.out, "figures"); os.makedirs(fig_dir, exist_ok=True)
    env = Environment()
    cfg = Config(pop_size=a.pop, generations=a.gens, seed=a.seed)
    snaps = sorted({0, max(1, a.gens // 30), a.gens // 6, a.gens // 2})
    ga = GeneticAlgorithm(env, cfg, snapshot_gens=snaps, verbose=True)
    best = ga.run()
    print("\nStopped:", ga.stop_reason)
    print("Best fitness:", round(best.fitness, 4)); print("Metrics:", best.metrics)

    ev = Evaluator(env, cfg)
    viz.plot_environment(env, f"{fig_dir}/01_environment.png")
    viz.plot_convergence(ga.history, f"{fig_dir}/02_convergence.png")
    viz.plot_evolution(env, ev, ga.snapshots, f"{fig_dir}/03_evolution_paths_heatmaps.png")
    viz.plot_final(env, ev, best, f"{fig_dir}/04_final_solution.png")
    json.dump(dict(history=ga.history, metrics=best.metrics, fitness=best.fitness, stop=ga.stop_reason),
              open(f"{a.out}/main_run.json", "w"), indent=1)
    print("Figures saved to", fig_dir)


if __name__ == "__main__":
    main()
