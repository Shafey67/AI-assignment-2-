# UAV Strategy Engine — Genetic Algorithm for Multi-Objective Fleet Deployment

AI (BSCS-5) · Assignment 2 · Bahria University, Lahore Campus · CLO-4

A modular, object-oriented Python Genetic Algorithm that plans **deployment positions,
waypoint trajectories, speeds and sensor schedules** for a fleet of 4 UAVs surveying a
volatile 50 x 50 disaster grid. There is no single "correct" answer: the fitness
landscape trades off priority-zone coverage against energy, flight time and hazard exposure.

## Quick start

```bash
git clone <your-repo-url> && cd uav-ga-strategy-engine
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py                       # one full run (~25 s): graphs open in windows + saved to results/figures/
python main.py --no-show             # save PNGs only (no windows)
python main.py --pop 60 --gens 100   # quick run
python run_experiments.py            # full study for the report (~25 min, 1 core)
python run_experiments.py --quick    # reduced study (~2 min)
python tests/test_smoke.py           # sanity tests
```

Everything is seeded (`--seed`), so results are reproducible.

## Repository layout

```
main.py                 single GA run + convergence / path / heat-map figures
run_experiments.py      GA vs random search, uniqueness, sensitivity, ablation, weight study
uav_ga/
  config.py             all parameters (dataclass Config)
  environment.py        50x50 grid: priority map, hazard map, no-fly mask, base
  chromosome.py         hybrid real/binary chromosome (xy, speed, power, active)
  fitness.py            Evaluator: path simulation, sensor model, objectives, penalties
  constraints.py        repair mechanisms (no-fly snap, range contraction, speed-up)
  operators.py          tournament selection, spatial crossover, spatial mutation, diversity
  ga.py                 GeneticAlgorithm class (elitism, immigrants, adaptive mutation, termination)
  baseline.py           random-search baseline (same sampler / repair / budget)
  visualize.py          all plots
tests/test_smoke.py
results/                figures/*.png, experiments.json, main_run.json, report
```

## Design summary

**Chromosome** (K=4 UAVs x L=8 waypoints, 5 genes per waypoint = 160 genes)

| gene | type | meaning |
|---|---|---|
| `xy` | real (x, y) | waypoint coordinates; waypoint 0 is the UAV's *deployment position* |
| `speed` | real [0.5, 2.0] | cruise speed on the leg arriving at the waypoint |
| `power` | real [0.2, 1.0] | sensor power on that leg (radius = 1 + 3·power) |
| `active` | binary | sensor on/off schedule for that leg |

Route: `BASE → wp0 → … → wp7 → BASE`.

**Fitness** (maximise): `F = 0.55·C − 0.15·E − 0.10·T − 0.20·H − penalties`
- `C` priority-weighted detection rate, `E` total energy, `T` mission makespan, `H` hazard-exposure integral (all normalised).
- Hard constraints: path length ≤ 110 cells per UAV, deadline ≤ 125 time units, no flying through no-fly cells.

**Operators**
- Selection: k-tournament (k = selection pressure), 2 elites.
- Crossover (per UAV, random choice): route swap · waypoint-segment swap · geometric blend (point on the line between parent waypoints).
- Mutation (per waypoint): Gaussian shift · priority-map-guided jump · order swap · speed/power/activation mutation; step size anneals and the rate is boosted automatically when population diversity collapses; 5 % random immigrants per generation.
- Constraint handling: always-on repair of bounds / no-fly waypoints (snap to nearest free cell), probabilistic repair of range (geometric contraction to the base) and deadline (speed-up), residual violations penalised.
- Termination: maximum generations **or** stagnation (no improvement for `patience` generations).

See `report/UAV_GA_Technical_Report.docx` for rationale, hyper-parameter sensitivity and the GA vs random-search discussion.
