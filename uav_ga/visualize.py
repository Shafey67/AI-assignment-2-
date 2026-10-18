"""Matplotlib figures: environment, convergence, evolved paths, heatmaps, studies."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

UAV_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def _background(ax, env, heat=None, title=""):
    S = env.size
    if heat is None:
        ax.imshow(env.priority, origin="lower", cmap="Greens", vmin=0, vmax=1.4, alpha=0.55,
                  extent=(-0.5, S - 0.5, -0.5, S - 0.5))
    else:
        im = ax.imshow(heat, origin="lower", cmap="inferno", vmin=0, vmax=1,
                       extent=(-0.5, S - 0.5, -0.5, S - 0.5))
    ax.contour(env.hazard, levels=[0.25, 0.5], colors=["#e377c2", "#8c1c8c"], linewidths=[0.8, 1.2],
               origin="lower")
    ax.imshow(np.ma.masked_where(~env.nofly, env.nofly), origin="lower",
              cmap=ListedColormap(["#444444"]), alpha=0.75, extent=(-0.5, S - 0.5, -0.5, S - 0.5))
    ax.plot(*env.base, marker="*", ms=13, color="gold", mec="k", zorder=6)
    ax.set_xlim(-0.5, S - 0.5); ax.set_ylim(-0.5, S - 0.5)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def draw_paths(ax, env, ch, number=True, lw=1.8):
    for k in range(ch.K):
        col = UAV_COLORS[k % len(UAV_COLORS)]
        pts = np.vstack([env.base, ch.xy[k], env.base])
        for l in range(ch.L):
            ls = "-" if ch.active[k, l] else (0, (2, 2))
            ax.plot(pts[l:l + 2, 0], pts[l:l + 2, 1], ls=ls, color=col, lw=lw, alpha=0.95, zorder=4)
        ax.plot(pts[-1:, 0], pts[-1:, 1])
        ax.plot(pts[-2:, 0], pts[-2:, 1], ls=":", color=col, lw=lw * 0.7, zorder=4)
        ax.scatter(ch.xy[k, :, 0], ch.xy[k, :, 1], s=14, color=col, edgecolor="k", linewidth=0.4, zorder=5)
        ax.scatter(*ch.xy[k, 0], s=60, marker="D", color=col, edgecolor="k", zorder=6)  # deployment point
        if number:
            for l in range(ch.L):
                ax.text(ch.xy[k, l, 0] + 0.5, ch.xy[k, l, 1] + 0.5, str(l), fontsize=5, zorder=7)


def legend_handles(K):
    h = [Line2D([0], [0], color=UAV_COLORS[k], lw=2, label=f"UAV {k + 1}") for k in range(K)]
    h += [Line2D([0], [0], marker="D", color="w", mfc="grey", mec="k", ms=7, label="deployment pt (wp 0)"),
          Line2D([0], [0], ls="--", color="grey", label="sensor off"),
          Line2D([0], [0], color="#8c1c8c", lw=1.2, label="hazard 0.5 contour"),
          plt.Rectangle((0, 0), 1, 1, fc="#444444", alpha=0.75, label="no-fly zone"),
          Line2D([0], [0], marker="*", color="w", mfc="gold", mec="k", ms=11, label="base")]
    return h


def plot_environment(env, path):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    S = env.size
    for ax, (arr, cm, t) in zip(axs, [(env.priority, "Greens", "Disaster-zone priority"),
                                      (env.hazard, "RdPu", "Hazard map (wind / radiation)"),
                                      (env.nofly, "Greys", "No-fly zones (binary)")]):
        im = ax.imshow(arr, origin="lower", cmap=cm, extent=(-0.5, S - 0.5, -0.5, S - 0.5))
        ax.plot(*env.base, marker="*", ms=13, color="gold", mec="k")
        ax.set_title(t, fontsize=10); fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("50 x 50 operational grid (star = launch base)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_convergence(history, path, title="GA convergence"):
    g = np.arange(len(history["best"]))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    ax = axs[0]
    ax.fill_between(g, history["worst"], history["best"], color="#1f77b4", alpha=0.10, label="worst-best range")
    ax.plot(g, history["best"], color="#d62728", lw=2, label="best fitness")
    ax.plot(g, history["mean"], color="#1f77b4", lw=1.6, label="average fitness")
    lo = np.percentile(history["mean"], 2)
    ax.set_ylim(min(lo, -0.6) if lo > -1.5 else -1.5, max(history["best"]) + 0.05)
    ax.set_xlabel("generation"); ax.set_ylabel("fitness"); ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=.3)
    ax = axs[1]
    ax.plot(g, history["diversity"], color="#2ca02c", lw=1.8, label="population diversity")
    ax.set_xlabel("generation"); ax.set_ylabel("mean pairwise genotype distance", color="#2ca02c")
    ax2 = ax.twinx()
    ax2.plot(g, history["mut_rate"], color="#ff7f0e", lw=1.2, label="mutation rate")
    ax2.set_ylabel("per-waypoint mutation rate", color="#ff7f0e")
    ax.set_title("Diversity and adaptive mutation"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_evolution(env, ev, snaps, path):
    keys = sorted([k for k in snaps if k != "final"]) + ["final"]
    fig, axs = plt.subplots(2, len(keys), figsize=(3.3 * len(keys), 6.9))
    for j, k in enumerate(keys):
        ch = snaps[k]
        info = ev.analyze(ch)
        lab = "final" if k == "final" else f"generation {k}"
        _background(axs[0, j], env, None, f"{lab}\nfitness {ch.fitness:.3f}")
        draw_paths(axs[0, j], env, ch, number=False, lw=1.4)
        _background(axs[1, j], env, info["grid"], f"coverage heat-map: C = {ch.metrics['coverage']:.2f}")
    axs[0, 0].set_ylabel("flight paths", fontsize=9); axs[1, 0].set_ylabel("detection heat-map", fontsize=9)
    fig.legend(handles=legend_handles(snaps[keys[0]].K), loc="lower center", ncol=9, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=2.0); fig.savefig(path, dpi=150); plt.close(fig)


def plot_final(env, ev, ch, path, title="Best evolved fleet strategy"):
    info = ev.analyze(ch)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))
    _background(axs[0], env, None, title)
    draw_paths(axs[0], env, ch)
    _background(axs[1], env, info["grid"], "Detection-quality heat-map (yellow = high)")
    draw_paths(axs[1], env, ch, number=False, lw=0.8)
    m = ch.metrics
    txt = (f"fitness {ch.fitness:.3f}\ncoverage C {m['coverage']:.3f}\nenergy E {m['energy']:.3f}\n"
           f"time T {m['time']:.3f}\nhazard H {m['hazard']:.3f}\nfeasible: {m['feasible']}")
    axs[0].text(0.99, 0.01, txt, transform=axs[0].transAxes, va="bottom", ha="right", fontsize=7,
                bbox=dict(fc="white", alpha=0.85, ec="grey"))
    fig.legend(handles=legend_handles(ch.K), loc="lower center", ncol=9, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(path, dpi=150); plt.close(fig)


def _band(ax, X, Y, color, label):
    Y = np.asarray(Y); mu, sd = Y.mean(0), Y.std(0)
    ax.plot(X, mu, color=color, lw=2, label=label); ax.fill_between(X, mu - sd, mu + sd, color=color, alpha=0.2)


def plot_ga_vs_rs(ga_curves, rs_curves, evals, ga_final, rs_final, ga_m, rs_m, path):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    _band(axs[0], evals, ga_curves, "#d62728", "Genetic Algorithm")
    _band(axs[0], evals, rs_curves, "#7f7f7f", "Random search")
    axs[0].set_xlabel("fitness evaluations"); axs[0].set_ylabel("best-so-far fitness")
    axs[0].set_title("Equal evaluation budget (mean ± sd)"); axs[0].legend(); axs[0].grid(alpha=.3)
    bp = axs[1].boxplot([rs_final, ga_final], labels=["Random search", "GA"], patch_artist=True, widths=0.5)
    for p, c in zip(bp["boxes"], ["#bdbdbd", "#f4a6a6"]):
        p.set_facecolor(c)
    for i, d in enumerate([rs_final, ga_final], 1):
        axs[1].scatter(np.full(len(d), i) + np.random.default_rng(0).uniform(-.08, .08, len(d)), d, color="k", s=14, zorder=3)
    axs[1].set_ylabel("final best fitness"); axs[1].set_title("Final solution quality over seeds"); axs[1].grid(alpha=.3)
    names = ["coverage C", "energy E", "time T", "hazard H"]
    x = np.arange(4); w = 0.38
    axs[2].bar(x - w / 2, [rs_m[k] for k in ["coverage", "energy", "time", "hazard"]], w, color="#9e9e9e", label="Random search")
    axs[2].bar(x + w / 2, [ga_m[k] for k in ["coverage", "energy", "time", "hazard"]], w, color="#d62728", label="GA")
    axs[2].set_xticks(x); axs[2].set_xticklabels(names); axs[2].set_title("Mean objective values (normalised)")
    axs[2].legend(); axs[2].grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_sensitivity(sweeps, path):
    """sweeps: {name: {label: {'curves': [[...]], 'final': [...]}}}"""
    fig, axs = plt.subplots(2, len(sweeps), figsize=(5 * len(sweeps), 8))
    cmap = plt.get_cmap("viridis")
    for j, (name, d) in enumerate(sweeps.items()):
        labels = list(d.keys())
        for i, lab in enumerate(labels):
            c = cmap(i / max(1, len(labels) - 1) * 0.9)
            Y = np.asarray(d[lab]["curves"]); X = np.arange(Y.shape[1])
            axs[0, j].plot(X, Y.mean(0), color=c, lw=1.8, label=str(lab))
        axs[0, j].set_title(f"Convergence vs {name}"); axs[0, j].set_xlabel("generation")
        axs[0, j].set_ylabel("best fitness (mean of seeds)"); axs[0, j].legend(fontsize=8); axs[0, j].grid(alpha=.3)
        mu = [np.mean(d[l]["final"]) for l in labels]; sd = [np.std(d[l]["final"]) for l in labels]
        axs[1, j].errorbar(range(len(labels)), mu, yerr=sd, fmt="o-", color="#d62728", capsize=4)
        axs[1, j].set_xticks(range(len(labels))); axs[1, j].set_xticklabels([str(l) for l in labels])
        axs[1, j].set_xlabel(name); axs[1, j].set_ylabel("final best fitness (mean ± sd)")
        axs[1, j].set_title(f"Final quality vs {name}"); axs[1, j].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_ablation(abl, path):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    cols = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
    for (lab, d), c in zip(abl.items(), cols):
        _band(axs[0], np.arange(len(d["curves"][0])), d["curves"], c, lab)
        axs[1].plot(np.mean(d["div"], 0), color=c, lw=1.8, label=lab)
    axs[0].set_title("Best fitness"); axs[0].set_xlabel("generation"); axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)
    axs[1].set_title("Population diversity"); axs[1].set_xlabel("generation"); axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_weights(env, ev, sols, path):
    fig, axs = plt.subplots(1, len(sols), figsize=(4.6 * len(sols), 5))
    for ax, (name, ch) in zip(np.atleast_1d(axs), sols.items()):
        m = ch.metrics
        _background(ax, env, None, f"{name}\nC={m['coverage']:.2f}  E={m['energy']:.2f}  T={m['time']:.2f}  H={m['hazard']:.3f}")
        draw_paths(ax, env, ch, number=False, lw=1.5)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_uniqueness(env, ga_sols, rs_sols, path):
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, sols, t in zip(axs, [ga_sols, rs_sols], ["GA: best solutions from independent seeds", "Random search: best from independent seeds"]):
        _background(ax, env, None, t)
        for i, ch in enumerate(sols):
            col = plt.get_cmap("tab10")(i)
            for k in range(ch.K):
                pts = np.vstack([env.base, ch.xy[k], env.base])
                ax.plot(pts[:, 0], pts[:, 1], color=col, lw=0.9, alpha=0.75, zorder=4)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
