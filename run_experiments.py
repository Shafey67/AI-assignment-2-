"""Full experimental study used in the technical report.

    python run_experiments.py            # ~25 min on one CPU core
    python run_experiments.py --quick    # reduced seeds / generations (~3 min)

Produces: GA-vs-random-search comparison, uniqueness analysis, hyper-parameter
sensitivity (population size, mutation rate, selection pressure), diversity
ablation and fitness-weight trade-off study. Figures -> results/figures,
numbers -> results/experiments.json.
"""
import argparse, json, os, time
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
from uav_ga import Config, Environment, GeneticAlgorithm, random_search
from uav_ga.fitness import Evaluator
from uav_ga import visualize as viz

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default="results")
a = ap.parse_args()
Q = a.quick
FIG = os.path.join(a.out, "figures"); os.makedirs(FIG, exist_ok=True)
env = Environment()
T0 = time.time()
log = lambda *x: print(f"[{time.time() - T0:6.0f}s]", *x, flush=True)
R = {}

# ------------------------------------------------------------ 1. GA vs random search
N_SEEDS = 3 if Q else 8
base = Config(pop_size=60 if Q else 100, generations=60 if Q else 300, early_stop=False)
ga_runs, rs_runs, ga_curves, rs_curves = [], [], [], []
for s in range(N_SEEDS):
    cfg = base.but(seed=100 + s)
    ga = GeneticAlgorithm(env, cfg); b = ga.run()
    ga_runs.append(b); ga_curves.append(ga.history["best"])
    rb, tr = random_search(env, cfg, ga.ev.n_evals, record_every=cfg.pop_size, seed=500 + s)
    rs_runs.append(rb); rs_curves.append(tr)
    log(f"seed {s}: GA {b.fitness:.4f} | RS {rb.fitness:.4f}")
L = min(len(c) for c in ga_curves + rs_curves)
ga_curves = [c[:L] for c in ga_curves]; rs_curves = [c[:L] for c in rs_curves]
evals = (np.arange(L) + 1) * base.pop_size
ga_f = [c.fitness for c in ga_runs]; rs_f = [c.fitness for c in rs_runs]
mean_m = lambda runs: {k: float(np.mean([r.metrics[k] for r in runs])) for k in ["coverage", "energy", "time", "hazard"]}
viz.plot_ga_vs_rs(ga_curves, rs_curves, evals, ga_f, rs_f, mean_m(ga_runs), mean_m(rs_runs), f"{FIG}/05_ga_vs_random.png")
u = mannwhitneyu(ga_f, rs_f, alternative="greater")
R["comparison"] = dict(n_seeds=N_SEEDS, evals=int(evals[-1]), ga_mean=float(np.mean(ga_f)), ga_std=float(np.std(ga_f)),
                       rs_mean=float(np.mean(rs_f)), rs_std=float(np.std(rs_f)), mannwhitney_p=float(u.pvalue),
                       ga_metrics=mean_m(ga_runs), rs_metrics=mean_m(rs_runs),
                       ga_best=float(np.max(ga_f)), rs_best=float(np.max(rs_f)),
                       ga_feasible=int(sum(r.metrics["feasible"] for r in ga_runs)),
                       rs_feasible=int(sum(r.metrics["feasible"] for r in rs_runs)))

# ------------------------------------------------------------ 2. uniqueness analysis
ev = Evaluator(env, base)
def route_pts(ch):
    return ev.samples(ch)[0]
def chamfer(c1, c2):
    A, B = route_pts(c1), route_pts(c2)
    return 0.5 * (cKDTree(B).query(A)[0].mean() + cKDTree(A).query(B)[0].mean())
def pairwise(runs):
    d = [chamfer(runs[i], runs[j]) for i in range(len(runs)) for j in range(i + 1, len(runs))]
    return float(np.mean(d)), float(np.std(d))
def jaccard(runs):
    g = [ev.analyze(r)["grid"] > 0 for r in runs]
    return float(np.mean([(g[i] & g[j]).sum() / (g[i] | g[j]).sum() for i in range(len(g)) for j in range(i + 1, len(g))]))
ga_an = [ev.analyze(r) for r in ga_runs]; rs_an = [ev.analyze(r) for r in rs_runs]
R["uniqueness"] = dict(
    ga_chamfer=pairwise(ga_runs), rs_chamfer=pairwise(rs_runs),
    ga_jaccard=jaccard(ga_runs), rs_jaccard=jaccard(rs_runs),
    ga_cv=float(np.std(ga_f) / abs(np.mean(ga_f))), rs_cv=float(np.std(rs_f) / abs(np.mean(rs_f))),
    ga_overlap=float(np.mean([x["overlap"] for x in ga_an])), rs_overlap=float(np.mean([x["overlap"] for x in rs_an])),
    ga_hi_cov=float(np.mean([x["hi_cov"] for x in ga_an])), rs_hi_cov=float(np.mean([x["hi_cov"] for x in rs_an])),
    ga_hi_det=float(np.mean([x["hi_det"] for x in ga_an])), rs_hi_det=float(np.mean([x["hi_det"] for x in rs_an])),
    ga_mean_range_used=float(np.mean([x["len_uav"].mean() for x in ga_an])) / base.max_range,
    rs_mean_range_used=float(np.mean([x["len_uav"].mean() for x in rs_an])) / base.max_range)
viz.plot_uniqueness(env, ga_runs, rs_runs, f"{FIG}/06_uniqueness.png")
best_overall = max(ga_runs, key=lambda c: c.fitness)
viz.plot_final(env, ev, best_overall, f"{FIG}/04b_best_of_all_seeds.png", "Best evolved strategy over all seeds")
log("comparison + uniqueness done", R["uniqueness"])

# ------------------------------------------------------------ 3. hyper-parameter sensitivity
SS = 3 if Q else 5
sens_base = Config(pop_size=60, generations=60 if Q else 150, early_stop=False,
                   adaptive_mutation=False, immigrant_frac=0.0)   # canonical GA to isolate each factor
def sweep(name, make, values):
    out = {}
    for v in values:
        curves, finals = [], []
        for s in range(SS):
            ga = GeneticAlgorithm(env, make(sens_base.but(seed=900 + s), v)); b = ga.run()
            curves.append(ga.history["best"][:sens_base.generations]); finals.append(b.fitness)
        out[str(v)] = dict(curves=curves, final=finals)
        log(f"  {name}={v}: {np.mean(finals):.4f} ± {np.std(finals):.4f}")
    return out
sweeps = {
    "population size": sweep("pop", lambda c, v: c.but(pop_size=v), [20, 50, 100, 200] if not Q else [20, 60, 120]),
    "mutation rate": sweep("mut", lambda c, v: c.but(p_mutation=v), [0.01, 0.03, 0.08, 0.15, 0.30]),
    "tournament size": sweep("k", lambda c, v: c.but(tournament_k=v), [2, 3, 5, 10]),
}
viz.plot_sensitivity(sweeps, f"{FIG}/07_sensitivity.png")
R["sensitivity"] = {n: {k: dict(mean=float(np.mean(v["final"])), std=float(np.std(v["final"])),
                               cov=float(np.mean([0])) ) for k, v in d.items()} for n, d in sweeps.items()}
for n, d in sweeps.items():
    for k, v in d.items():
        R["sensitivity"][n][k].pop("cov", None)
        R["sensitivity"][n][k]["gen50"] = float(np.mean([c[min(49, len(c) - 1)] for c in v["curves"]]))

# ------------------------------------------------------------ 4. diversity ablation
abl_base = Config(pop_size=40, generations=60 if Q else 150, tournament_k=7, early_stop=False)
abl_cfg = {"full (adaptive + immigrants)": {},
           "no adaptive mutation": dict(adaptive_mutation=False),
           "no immigrants": dict(immigrant_frac=0.0),
           "neither": dict(adaptive_mutation=False, immigrant_frac=0.0)}
abl = {}
for name, kw in abl_cfg.items():
    curves, divs, fin = [], [], []
    for s in range(SS):
        ga = GeneticAlgorithm(env, abl_base.but(seed=700 + s, **kw)); b = ga.run()
        curves.append(ga.history["best"][:abl_base.generations]); divs.append(ga.history["diversity"][:abl_base.generations]); fin.append(b.fitness)
    abl[name] = dict(curves=curves, div=divs, final=fin)
    log(f"  ablation {name}: {np.mean(fin):.4f} ± {np.std(fin):.4f}, final diversity {np.mean([d[-1] for d in divs]):.3f}")
viz.plot_ablation(abl, f"{FIG}/08_diversity_ablation.png")
R["ablation"] = {k: dict(mean=float(np.mean(v["final"])), std=float(np.std(v["final"])),
                         div_end=float(np.mean([d[-1] for d in v["div"]]))) for k, v in abl.items()}

# ------------------------------------------------------------ 5. weight (trade-off) study
presets = {"coverage-first": dict(w_cov=0.80, w_energy=0.06, w_time=0.04, w_hazard=0.10),
           "balanced (default)": {},
           "safety / energy-saver": dict(w_cov=0.30, w_energy=0.30, w_time=0.10, w_hazard=0.30)}
sols, wstat = {}, {}
for name, kw in presets.items():
    runs = []
    for s in range(2 if Q else 3):
        ga = GeneticAlgorithm(env, Config(pop_size=60 if Q else 100, generations=60 if Q else 200, early_stop=False, seed=300 + s, **kw))
        runs.append(ga.run())
    # compare on the SAME raw objectives (not on their own weighted fitness)
    bestc = max(runs, key=lambda c: c.fitness); sols[name] = bestc
    wstat[name] = {k: float(np.mean([r.metrics[k] for r in runs])) for k in ["coverage", "energy", "time", "hazard"]}
    log(f"  weights {name}: {wstat[name]}")
viz.plot_weights(env, ev, sols, f"{FIG}/09_weight_tradeoff.png")
R["weights"] = wstat

json.dump(R, open(os.path.join(a.out, "experiments.json"), "w"), indent=1)
log("ALL DONE")
