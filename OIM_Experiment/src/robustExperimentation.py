#!/usr/bin/env python3
"""
robustExperimentation.py
------------------------
Robust OIM experiments — Bifurcation analysis (Exp D) and
Statistical robustness with graph perturbations (Exp E).

Both experiments use OIMMaxCut (OIM_mu_v2.py) exclusively and produce
figures styled to match TikZ / PGFPlots output (serif fonts, white
background, black axes, clean grids).

Experiment D  —  Bifurcation Diagram
--------------------------------------
For every 2^N binary equilibrium phi* in {0, pi}^N the leading
Jacobian eigenvalue is:

    lambda_max(A(phi*, mu)) = lambda_max(D(phi*)) - mu

This follows because at binary points cos(2*phi*_i) = 1 for all i,
so  A = D(phi*) - mu*I  and eigenvalues shift linearly with mu.
The analysis is therefore ANALYTICAL — no simulation is needed for the
eigenvalue part.

Three sub-panels:
  1. All eigenvalue branches vs mu  (colour = cut quality via RdYlGn)
     Zero-crossing at mu = mu*_i marks each equilibrium's stability
     onset.  Optimal-cut branches are drawn thicker.
  2. n_stable(mu) — step function counting stable equilibria vs mu.
  3. Best cut found by simulation vs mu with 95% bootstrap CI.

Experiment E  —  Statistical Robustness
-----------------------------------------
Tests whether the threshold behaviour survives graph perturbations and
quantifies it statistically.

Graph family:  original + drop lightest edge + add random edge +
               ±10% weight noise + rewire one edge pair.

For each variant, sweeps mu, runs many random restarts, computes:
  - Mean Ising H ± 95% bootstrap CI
  - Fraction of runs reaching the exact optimum
  - Normalised gap  (mean_H - H*) / |H*|

Trend test: Spearman rho on mean_H vs mu  (monotonically increasing
H as mu grows = quality degrades).

Four sub-panels:
  1. Mean H ± CI vs mu          (one curve per variant)
  2. Fraction at exact optimum  (one curve per variant)
  3. Normalised optimality gap  (one curve per variant)
  4. Summary table: mu_bin, Spearman rho, p-value, verdict

Usage
-----
    python robustExperimentation.py --graph graph.txt
    python robustExperimentation.py --graph graph.txt --experiments D
    python robustExperimentation.py --graph graph.txt --experiments E
    python robustExperimentation.py --graph graph.txt --experiments DE \\
        --mu-min 0.2 --mu-max 5.0 --d-trials 60 --e-trials 80
"""

from __future__ import annotations

import argparse
import sys
from itertools import product as iproduct
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines   as mlines
import matplotlib.colors  as mcolors
from matplotlib.gridspec  import GridSpec
from scipy                import stats

# ── TikZ / PGFPlots global style ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family"       : "serif",
    "font.size"         : 10,
    "axes.edgecolor"    : "black",
    "axes.linewidth"    : 0.8,
    "xtick.direction"   : "in",
    "ytick.direction"   : "in",
    "xtick.color"       : "black",
    "ytick.color"       : "black",
    "text.color"        : "black",
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "white",
    "legend.framealpha" : 1.0,
    "legend.edgecolor"  : "#aaaaaa",
    "savefig.dpi"       : 150,
    "savefig.bbox"      : "tight",
})

WHITE  = "#ffffff"
BLACK  = "#000000"
GRAY   = "#aaaaaa"
LGRAY  = "#e6e6e6"
BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"
PURPLE = "#8172B2"
BROWN  = "#937860"

# ── Import OIMMaxCut (tries local, then package path) ─────────────────────────
try:
    from OIM_mu_v2 import OIMMaxCut
except ModuleNotFoundError:
    try:
        from OIM_Experiment.src.OIM_mu_v2 import OIMMaxCut
    except ModuleNotFoundError:
        raise ImportError(
            "Cannot find OIM_mu_v2.py — place it in the same directory "
            "or install OIM_Experiment.")

# ── Graph I/O ─────────────────────────────────────────────────────────────────
def _load_graph(path: str) -> np.ndarray:
    """Return positive weight matrix W (OIMMaxCut convention: W >= 0)."""
    try:
        try:
            from graph_utils import read_graph
        except ModuleNotFoundError:
            from OIM_Experiment.src.graph_utils import read_graph
        return np.array(read_graph(path), dtype=float)
    except (ModuleNotFoundError, ImportError):
        pass
    try:
        try:
            from graph_utils import read_graph
        except ModuleNotFoundError:
            from OIM_Experiment.src.graph_utils import read_graph
        return -np.array(read_graph(path), dtype=float)   # W = -J
    except (ModuleNotFoundError, ImportError):
        raise ImportError("Cannot find a graph reader (graph_utils or read_graphs).")


def _graph_info(W: np.ndarray) -> str:
    N     = W.shape[0]
    edges = int(np.sum(W > 0)) // 2
    return f"N={N}, {edges} edges, density={2*edges/max(N*(N-1), 1):.3f}"


# ── Axis / box helpers (matching run_experiment.py style) ─────────────────────
def _style_ax(ax, grid_axis: str = "both") -> None:
    ax.set_facecolor(WHITE)
    ax.tick_params(colors=BLACK, labelsize=9, direction="in")
    for sp in ax.spines.values():
        sp.set_edgecolor(BLACK)
        sp.set_linewidth(0.8)
    ax.grid(True, color=LGRAY, linewidth=0.5, axis=grid_axis, zorder=0)


def _tikz_box() -> dict:
    return dict(boxstyle="round,pad=0.35", facecolor=WHITE,
                edgecolor=GRAY, alpha=1.0)


# ── Shared computation helpers ─────────────────────────────────────────────────
def _all_configs(N: int) -> List[np.ndarray]:
    """All 2^N binary phase vectors in {0, pi}^N."""
    return [np.array(cfg, dtype=float) * np.pi
            for cfg in iproduct([0, 1], repeat=N)]


def _ground_H(W: np.ndarray) -> float:
    """Exact minimum Ising H over all 2^N binary assignments."""
    oim = OIMMaxCut(W, mu=1.0, seed=0)
    return float(min(oim.get_hamiltonian(theta=phi)
                     for phi in _all_configs(oim.n)))


def _ground_cut(W: np.ndarray) -> float:
    """Exact Max-Cut via brute force."""
    oim = OIMMaxCut(W, mu=1.0, seed=0)
    W_total = oim.get_w_total()
    return float((W_total - _ground_H(W)) / 2.0)


def _one_trial(oim: OIMMaxCut,
               t_span: Tuple[float, float] = (0., 50.),
               n_pts:  int = 500) -> Tuple[float, float, bool]:
    """Random restart. Returns (H_final, cut_final, is_binarized)."""
    phi0  = np.random.uniform(0., 2.*np.pi, oim.n)
    sol   = oim.simulate(phi0, t_span, n_pts)
    phi_f = sol.y[:, -1]
    H_f   = oim.get_hamiltonian(theta=phi_f)
    oim.theta = phi_f
    c_f   = oim.get_binary_cut_value()
    binar = oim.is_binarized(tol=0.01)
    return H_f, c_f, binar


def _bootstrap_ci(data: np.ndarray, stat_fn, n_boot: int = 500,
                  alpha: float = 0.05) -> Tuple[float, float]:
    """Non-parametric bootstrap CI for a scalar statistic."""
    if len(data) < 2:
        return (float("nan"), float("nan"))
    boots = [stat_fn(np.random.choice(data, len(data), replace=True))
             for _ in range(n_boot)]
    return (np.percentile(boots, 100*alpha/2),
            np.percentile(boots, 100*(1 - alpha/2)))


# =============================================================================
# Experiment D — Bifurcation Diagram
# =============================================================================
def experiment_D(W: np.ndarray,
                 mu_min:       float = 0.10,
                 mu_max:       float = 4.00,
                 n_mu_sim:     int   = 25,
                 n_trials_sim: int   = 50,
                 t_end:        float = 50.0,
                 n_pts:        int   = 500,
                 seed:         int   = 42) -> None:
    """
    Three-panel bifurcation diagram.

    Panel 1  (left, full height)
        lambda_max(A(phi*, mu)) = lambda_max(D(phi*)) - mu  vs  mu
        for every 2^N binary equilibrium.  ANALYTICAL — slope = -1.
        Colour encodes cut quality (RdYlGn: red=worst, green=optimal).
        Optimal-cut branches are drawn with thicker lines.
        Scatter dots mark the zero-crossings  (= mu*_i thresholds).

    Panel 2  (top right)
        n_stable(mu) — step function: number of equilibria with
        lambda_max(A) < 0, computed analytically from mu*_i values.

    Panel 3  (bottom right)
        Best cut found by simulation vs mu, with 95% bootstrap CI.
        Horizontal line = exact optimum from brute-force enumeration.
    """
    N   = W.shape[0]
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    # ── Analytical: mu*_i, H_i, cut_i for all 2^N equilibria ─────────────────
    print(f"[Exp D] Analytical scan: {2**N} equilibria ...")
    oim_ref = OIMMaxCut(W, mu=1.0, seed=0)
    configs = _all_configs(N)

    mu_stars, H_vals, cut_vals = [], [], []
    for phi in configs:
        mu_i = oim_ref.stability_threshold(phi)     # lambda_max(D(phi*))
        H_i  = oim_ref.get_hamiltonian(theta=phi)
        oim_ref.theta = phi.copy()
        c_i  = oim_ref.get_binary_cut_value()
        mu_stars.append(mu_i);  H_vals.append(H_i);  cut_vals.append(c_i)

    mu_stars  = np.array(mu_stars)
    H_vals    = np.array(H_vals)
    cut_vals  = np.array(cut_vals)
    H_ground  = float(H_vals.min())
    cut_opt   = float(cut_vals.max())
    mu_bin    = float(mu_stars.min())
    W_total   = float(oim_ref.get_w_total())

    print(f"  mu_bin = {mu_bin:.4f}  |  H_ground = {H_ground:.3f}"
          f"  |  cut_opt = {cut_opt:.1f}  |  2^N = {2**N}")

    # Normalise cut for colormap (0 = worst, 1 = optimal)
    cut_norm = (cut_vals - cut_vals.min()) / max(cut_vals.max() - cut_vals.min(), 1e-9)
    cmap     = plt.get_cmap("RdYlGn")  # red=worst, green=best
    colors   = cmap(cut_norm)

    # ── Simulation: best cut vs mu (with bootstrap CI) ────────────────────────
    print(f"[Exp D] Simulation sweep: {n_mu_sim} steps × {n_trials_sim} trials ...")
    mu_sim      = np.linspace(mu_min, mu_max, n_mu_sim)
    best_cut_s  = np.zeros(n_mu_sim)
    ci_lo_s     = np.zeros(n_mu_sim)
    ci_hi_s     = np.zeros(n_mu_sim)
    frac_opt_s  = np.zeros(n_mu_sim)

    for k, mu in enumerate(mu_sim):
        oim = OIMMaxCut(W, mu=mu, seed=seed)
        cuts = np.array([_one_trial(oim, (0., t_end), n_pts)[1]
                         for _ in range(n_trials_sim)])
        best_cut_s[k] = cuts.max()
        frac_opt_s[k] = float(np.mean(np.abs(cuts - cut_opt) < 0.5))
        lo, hi        = _bootstrap_ci(cuts, np.max, n_boot=300)
        ci_lo_s[k]    = lo;  ci_hi_s[k] = hi
        pct = 100*(k+1)/n_mu_sim
        bar = "█"*int(pct/5) + "░"*(20-int(pct/5))
        print(f"  [{bar}] {pct:5.1f}%  mu={mu:.3f}  "
              f"best_cut={cuts.max():.1f}  frac_opt={frac_opt_s[k]:.2f}",
              end="\r", flush=True)
    print()

    # ── Empirical mu_c: first mu where ≥50% of trials reach optimum ──────────
    opt_mask     = frac_opt_s >= 0.50
    mu_c_empiric = float(mu_sim[opt_mask][0]) if opt_mask.any() else None

    # ── Figure ────────────────────────────────────────────────────────────────
    mu_plot = np.linspace(mu_min, mu_max, 600)
    fig     = plt.figure(figsize=(14, 7.5), facecolor=WHITE,
                         num="Experiment D: Bifurcation Diagram")
    gs      = GridSpec(2, 2, figure=fig,
                       left=0.07, right=0.91, top=0.88, bottom=0.09,
                       wspace=0.32, hspace=0.48)
    ax_bif  = fig.add_subplot(gs[:, 0])    # full-height left
    ax_ns   = fig.add_subplot(gs[0, 1])    # top right
    ax_cut  = fig.add_subplot(gs[1, 1])    # bottom right

    for ax in (ax_bif, ax_ns, ax_cut):
        _style_ax(ax)

    # Panel 1: eigenvalue branches (analytical straight lines, slope −1) ──────
    is_opt = np.isclose(cut_vals, cut_opt)
    for i, (mu_i, col) in enumerate(zip(mu_stars, colors)):
        y   = mu_i - mu_plot                          # lambda_max(A) = mu*_i - mu
        lw  = 1.8 if is_opt[i] else 0.7
        alp = 0.90 if is_opt[i] else 0.40
        ax_bif.plot(mu_plot, y, color=col, lw=lw, alpha=alp, zorder=3)

    # Zero-crossing markers (scatter on y=0 at x=mu*_i)
    sc = ax_bif.scatter(
        mu_stars, np.zeros_like(mu_stars),
        c=cut_norm, cmap=cmap, s=25, zorder=5,
        edgecolors=BLACK, linewidths=0.4,
        label=r"Zero-crossing $\mu^*_i$ (stability onset)")

    # Reference lines
    ax_bif.axhline(0., color=BLACK, lw=1.2, ls="--",
                   label=r"$\lambda_{\max}(A)=0$ (stability boundary)", zorder=4)
    ax_bif.axvline(mu_bin, color=RED, lw=1.4, ls=":",
                   label=fr"$\mu_{{\rm bin}}={mu_bin:.4f}$ (Remark 7)", zorder=4)
    if mu_c_empiric is not None:
        ax_bif.axvline(mu_c_empiric, color=ORANGE, lw=1.4, ls="--",
                       label=fr"Empirical $\hat\mu_c={mu_c_empiric:.4f}$", zorder=4)

    # Axis labels / title
    ax_bif.set_xlabel(r"$\mu = 2K_s$", fontsize=12)
    ax_bif.set_ylabel(
        r"$\lambda_{\max}(A(\phi^*,\mu))=\lambda_{\max}(D(\phi^*))-\mu$",
        fontsize=11)
    ax_bif.set_title(
        fr"Bifurcation diagram — all $2^{{{N}}}$ binary equilibria"
        "\n"
        r"(slope $=-1$;  colour $=$ cut quality;  thick $=$ optimal)",
        fontsize=10)
    ax_bif.set_xlim(mu_min, mu_max)
    ax_bif.legend(fontsize=8.5, loc="upper right",
                  facecolor=WHITE, edgecolor=GRAY)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                            norm=mcolors.Normalize(cut_vals.min(), cut_vals.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_bif, shrink=0.55, pad=0.02)
    cbar.set_label("Cut value at $\\phi^*$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Panel 2: n_stable(mu) step function ─────────────────────────────────────
    mu_dense = np.linspace(mu_min, mu_max, 1200)
    n_stable = np.array([int(np.sum(mu_stars < mu)) for mu in mu_dense])
    n_total  = len(mu_stars)

    ax_ns.step(mu_dense, n_stable, where="post",
               color=BLUE, lw=2.0, label=r"$n_{\rm stable}(\mu)$")
    ax_ns.fill_between(mu_dense, 0, n_stable, step="post",
                       color=BLUE, alpha=0.12)
    ax_ns.axvline(mu_bin, color=RED, lw=1.2, ls=":",
                  label=fr"$\mu_{{\rm bin}}$")
    if mu_c_empiric is not None:
        ax_ns.axvline(mu_c_empiric, color=ORANGE, lw=1.2, ls="--",
                      label=fr"$\hat\mu_c$")
    ax_ns.set_xlabel(r"$\mu$", fontsize=11)
    ax_ns.set_ylabel("# stable equilibria", fontsize=10)
    ax_ns.set_title(f"Stable equilibria count  (total = {n_total})", fontsize=9.5)
    ax_ns.set_xlim(mu_min, mu_max)
    ax_ns.set_ylim(-0.5, n_total * 1.08)
    ax_ns.legend(fontsize=8, facecolor=WHITE, edgecolor=GRAY)

    # Panel 3: best cut from simulation ───────────────────────────────────────
    ax_cut.fill_between(mu_sim, ci_lo_s, ci_hi_s,
                        color=BLUE, alpha=0.18, label="95% bootstrap CI")
    ax_cut.plot(mu_sim, best_cut_s, "o-", color=BLUE,
                lw=2.0, ms=4, label="Best cut (simulation)")
    ax_cut.axhline(cut_opt, color=GREEN, lw=1.5, ls="--",
                   label=f"Optimal cut $= {cut_opt:.0f}$")
    ax_cut.axvline(mu_bin, color=RED, lw=1.2, ls=":",
                   label=fr"$\mu_{{\rm bin}}$")
    if mu_c_empiric is not None:
        ax_cut.axvline(mu_c_empiric, color=ORANGE, lw=1.2, ls="--",
                       label=fr"$\hat\mu_c$")
    ax_cut.text(0.97, 0.05,
                f"$H^*={H_ground:.1f}$\n"
                f"$W_{{\\rm total}}={W_total:.1f}$\n"
                f"Opt cut $= {cut_opt:.0f}$",
                transform=ax_cut.transAxes, ha="right", va="bottom",
                fontsize=8.5, bbox=_tikz_box())
    ax_cut.set_xlabel(r"$\mu$", fontsize=11)
    ax_cut.set_ylabel("Best cut found", fontsize=10)
    ax_cut.set_title(
        f"Simulation: best cut vs $\\mu$  ({n_trials_sim} trials/pt, "
        f"95% CI bootstrap)", fontsize=9.5)
    ax_cut.set_xlim(mu_min, mu_max)
    ax_cut.legend(fontsize=8, facecolor=WHITE, edgecolor=GRAY)

    fig.suptitle(
        f"OIM Bifurcation Analysis — {_graph_info(W)}",
        fontsize=12, fontweight="bold")
    plt.savefig("experiment_D_bifurcation.png")
    plt.savefig("experiment_D_bifurcation.pdf")
    print("[Exp D] Saved experiment_D_bifurcation.{png,pdf}.  Close to proceed.")
    plt.show()


# =============================================================================
# Experiment E — Statistical Robustness + Graph Perturbations
# =============================================================================
def _perturb(W: np.ndarray,
             mode: str,
             rng: np.random.Generator,
             noise_level: float = 0.10) -> Tuple[np.ndarray, str]:
    """Return (W_perturbed, label) for a given perturbation mode."""
    N   = W.shape[0]
    W_p = W.copy()

    if mode == "drop_edge":
        rows, cols = np.where(
            (W_p > 0) & (np.triu(np.ones((N, N), dtype=bool), 1)))
        if len(rows):
            idx = int(np.argmin(W_p[rows, cols]))
            W_p[rows[idx], cols[idx]] = W_p[cols[idx], rows[idx]] = 0.
        return W_p, "drop lightest edge"

    elif mode == "add_edge":
        rows, cols = np.where(
            (W_p == 0) & (np.triu(np.ones((N, N), dtype=bool), 1)))
        if len(rows):
            pick = int(rng.integers(len(rows)))
            i, j = rows[pick], cols[pick]
            w = float(rng.uniform(0.5, 1.5))
            W_p[i, j] = W_p[j, i] = w
        return W_p, "add random edge"

    elif mode == "weight_noise":
        noise = rng.normal(0., noise_level, (N, N))
        noise = (noise + noise.T) / 2.          # keep symmetric
        W_p   = np.maximum(W + W * noise, 0.)
        np.fill_diagonal(W_p, 0.)
        return W_p, f"weight noise ±{noise_level*100:.0f}%"

    elif mode == "rewire":
        rows, cols = np.where(
            (W_p > 0) & (np.triu(np.ones((N, N), dtype=bool), 1)))
        if len(rows) >= 2:
            idx = rng.choice(len(rows), 2, replace=False)
            i1, j1 = rows[idx[0]], cols[idx[0]]
            i2, j2 = rows[idx[1]], cols[idx[1]]
            # swap (i1,j1)↔(i1,j2) only if target slots are empty
            if (i1 != j2 and i2 != j1
                    and W_p[i1, j2] == 0. and W_p[i2, j1] == 0.):
                w1, w2 = W_p[i1, j1], W_p[i2, j2]
                W_p[i1, j1] = W_p[j1, i1] = 0.
                W_p[i2, j2] = W_p[j2, i2] = 0.
                W_p[i1, j2] = W_p[j2, i1] = w1
                W_p[i2, j1] = W_p[j1, i2] = w2
        return W_p, "rewired edge pair"

    else:
        raise ValueError(f"Unknown perturbation mode: {mode!r}")


def _sweep(W: np.ndarray,
           mu_vals:  np.ndarray,
           H_ground: float,
           n_trials: int,
           t_end:    float,
           n_pts:    int,
           n_boot:   int = 500,
           seed:     int = 42) -> Dict[str, np.ndarray]:
    """
    Run mu sweep for one graph variant.  Returns dict of arrays indexed
    by mu position:
      mean_H, std_H, ci_lo, ci_hi   — mean Ising H ± bootstrap 95% CI
      frac_opt                        — fraction of trials reaching H*
      norm_gap                        — (mean_H - H*) / |H*|
    """
    n_mu     = len(mu_vals)
    mean_H   = np.full(n_mu, np.nan)
    ci_lo    = np.full(n_mu, np.nan)
    ci_hi    = np.full(n_mu, np.nan)
    frac_opt = np.zeros(n_mu)
    norm_gap = np.full(n_mu, np.nan)

    for k, mu in enumerate(mu_vals):
        np.random.seed(seed + k)
        oim   = OIMMaxCut(W, mu=mu, seed=seed)
        H_arr = np.array([_one_trial(oim, (0., t_end), n_pts)[0]
                          for _ in range(n_trials)])
        if not len(H_arr):
            continue
        mean_H[k]   = float(H_arr.mean())
        lo, hi      = _bootstrap_ci(H_arr, np.mean, n_boot)
        ci_lo[k]    = lo;  ci_hi[k] = hi
        frac_opt[k] = float(np.mean(np.abs(H_arr - H_ground) < 0.5))
        if abs(H_ground) > 1e-9:
            norm_gap[k] = (mean_H[k] - H_ground) / abs(H_ground)
        else:
            norm_gap[k] = 0.

    return dict(mean_H=mean_H, ci_lo=ci_lo, ci_hi=ci_hi,
                frac_opt=frac_opt, norm_gap=norm_gap)


def experiment_E(W: np.ndarray,
                 mu_min:   float = 0.10,
                 mu_max:   float = 4.00,
                 n_mu:     int   = 20,
                 n_trials: int   = 60,
                 n_boot:   int   = 500,
                 t_end:    float = 50.0,
                 n_pts:    int   = 500,
                 seed:     int   = 42) -> None:
    """
    Statistical robustness across a family of graph variants.

    Graph family:
      1. original
      2. drop lightest edge
      3. add one random edge
      4. ±10% weight noise
      5. rewired edge pair

    For each variant, sweeps mu, bootstraps 95% CI on mean H,
    computes fraction reaching exact optimum, normalised gap, and
    applies a Spearman trend test on mean_H vs mu.

    Panels:
      1. Mean H ± 95% CI vs mu      (one curve per variant)
      2. Fraction at exact optimum
      3. Normalised gap (mean_H - H*) / |H*|
      4. Summary table
    """
    rng = np.random.default_rng(seed)
    N   = W.shape[0]

    # ── Ground truth (original graph) ─────────────────────────────────────────
    H_ground = _ground_H(W)
    cut_opt  = _ground_cut(W)
    oim_ref  = OIMMaxCut(W, mu=1.0, seed=0)
    mu_bin_orig = oim_ref.binarization_threshold()["mu_bin"]
    W_total     = oim_ref.get_w_total()

    print(f"[Exp E] N={N}  H*={H_ground:.3f}  cut_opt={cut_opt:.1f}"
          f"  mu_bin={mu_bin_orig:.4f}  W_total={W_total:.2f}")

    # ── Graph family ──────────────────────────────────────────────────────────
    perturb_modes = ["drop_edge", "add_edge", "weight_noise", "rewire"]
    family: List[Tuple[str, np.ndarray]] = [("original", W.copy())]
    for mode in perturb_modes:
        W_p, lbl = _perturb(W, mode, rng)
        family.append((lbl, W_p))

    # ── Sweep for each variant ─────────────────────────────────────────────────
    mu_vals  = np.linspace(mu_min, mu_max, n_mu)
    results  = {}
    mu_bins  = {}
    rho_pval = {}

    for name, W_v in family:
        print(f"\n[Exp E] Variant: «{name}»  ({_graph_info(W_v)})")
        oim_v = OIMMaxCut(W_v, mu=1.0, seed=0)
        mu_bins[name] = oim_v.binarization_threshold()["mu_bin"]
        H_gnd_v = _ground_H(W_v)
        print(f"  mu_bin = {mu_bins[name]:.4f}  |  H* = {H_gnd_v:.3f}")

        res = _sweep(W_v, mu_vals, H_gnd_v, n_trials, t_end, n_pts, n_boot, seed)
        results[name] = res

        valid = ~np.isnan(res["mean_H"])
        if valid.sum() >= 4:
            rho, pval = stats.spearmanr(mu_vals[valid], res["mean_H"][valid])
        else:
            rho, pval = np.nan, np.nan
        rho_pval[name] = (rho, pval)
        direction = "↑ H (degrades)" if rho > 0 else "↓ H (improves)"
        sig       = "sig." if pval < 0.05 else "n.s."
        print(f"  Spearman ρ={rho:.3f}  p={pval:.4f}  {direction}  {sig}")

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'─'*74}")
    print(f"  {'Variant':<26} {'mu_bin':>8}  {'ρ (Spearman)':>13}"
          f"  {'p-value':>10}  Verdict")
    print(f"{'─'*74}")
    for name, _ in family:
        mb       = mu_bins[name]
        rho, pval = rho_pval[name]
        verdict   = ("monotone ↑ H *" if (rho > 0.5 and pval < 0.05)
                     else ("monotone ↓ H *" if (rho < -0.5 and pval < 0.05)
                           else "no clear trend"))
        print(f"  {name:<26} {mb:>8.4f}  {rho:>13.4f}  {pval:>10.4f}  {verdict}")
    print(f"{'─'*74}\n")

    # ── Figure ────────────────────────────────────────────────────────────────
    palette    = [BLUE, ORANGE, GREEN, RED, PURPLE, BROWN]
    linestyles = ["-", "--", "-.", ":", (0,(3,1,1,1)), (0,(5,2))]

    fig = plt.figure(figsize=(15, 11.5), facecolor=WHITE,
                     num="Experiment E: Statistical Robustness")
    gs  = GridSpec(3, 2, figure=fig,
                   left=0.08, right=0.97, top=0.91, bottom=0.06,
                   wspace=0.30, hspace=0.50)
    ax_H     = fig.add_subplot(gs[0, :])   # full-width: mean H vs mu
    ax_fopt  = fig.add_subplot(gs[1, 0])   # fraction at optimum
    ax_gap   = fig.add_subplot(gs[1, 1])   # normalised gap
    ax_tbl   = fig.add_subplot(gs[2, :])   # summary table
    ax_tbl.axis("off")

    for ax in (ax_H, ax_fopt, ax_gap):
        _style_ax(ax, grid_axis="y")

    # ── Panel 1: mean H ± CI ─────────────────────────────────────────────────
    h_line_added = False
    for idx, (name, _) in enumerate(family):
        res   = results[name]
        valid = ~np.isnan(res["mean_H"])
        col   = palette[idx % len(palette)]
        ls    = linestyles[idx % len(linestyles)]

        ax_H.fill_between(mu_vals[valid],
                          res["ci_lo"][valid], res["ci_hi"][valid],
                          color=col, alpha=0.12)
        ax_H.plot(mu_vals[valid], res["mean_H"][valid],
                  ls=ls, color=col, lw=2.0, label=name)
        ax_H.axvline(mu_bins[name], color=col, lw=0.8, ls=":", alpha=0.6)

    ax_H.axhline(H_ground, color=GRAY, lw=1.5, ls="--",
                 label=f"$H^* = {H_ground:.1f}$ (ground state)")
    ax_H.set_xlabel(r"$\mu = 2K_s$", fontsize=12)
    ax_H.set_ylabel("Mean Ising $H$ (binarised)", fontsize=11)
    ax_H.set_title(
        f"Mean $H$ $\\pm$ 95\\% bootstrap CI vs $\\mu$ "
        f"— {n_trials} trials/pt  "
        r"| dotted verticals $=$ $\mu_{\rm bin}$ per variant",
        fontsize=10)
    ax_H.set_xlim(mu_min, mu_max)
    ax_H.legend(fontsize=8.5, loc="upper left", ncol=3,
                facecolor=WHITE, edgecolor=GRAY)

    # ── Panel 2: fraction at optimum ─────────────────────────────────────────
    for idx, (name, _) in enumerate(family):
        res = results[name]
        col = palette[idx % len(palette)]
        ls  = linestyles[idx % len(linestyles)]
        ax_fopt.plot(mu_vals, res["frac_opt"], ls=ls, color=col,
                     lw=1.8, label=name)

    ax_fopt.axvline(mu_bin_orig, color=BLACK, lw=1.0, ls=":",
                    label=fr"$\mu_{{\rm bin}}$ (original)")
    ax_fopt.set_xlabel(r"$\mu$", fontsize=11)
    ax_fopt.set_ylabel(r"Fraction reaching $H^*$", fontsize=10)
    ax_fopt.set_title(
        f"Fraction of runs reaching exact optimum  ({n_trials} trials/pt)",
        fontsize=9.5)
    ax_fopt.set_xlim(mu_min, mu_max)
    ax_fopt.set_ylim(-0.05, 1.10)
    ax_fopt.legend(fontsize=7.5, facecolor=WHITE, edgecolor=GRAY, ncol=2)

    # ── Panel 3: normalised gap ───────────────────────────────────────────────
    for idx, (name, _) in enumerate(family):
        res   = results[name]
        valid = ~np.isnan(res["norm_gap"])
        col   = palette[idx % len(palette)]
        ls    = linestyles[idx % len(linestyles)]
        ax_gap.plot(mu_vals[valid], res["norm_gap"][valid],
                    ls=ls, color=col, lw=1.8, label=name)

    ax_gap.axhline(0., color=BLACK, lw=1.0, ls="--",
                   label="gap $= 0$ (optimal)")
    ax_gap.axvline(mu_bin_orig, color=BLACK, lw=1.0, ls=":",
                   label=fr"$\mu_{{\rm bin}}$ (original)")
    ax_gap.set_xlabel(r"$\mu$", fontsize=11)
    ax_gap.set_ylabel(r"$(\bar H - H^*)\,/\,|H^*|$", fontsize=10)
    ax_gap.set_title("Normalised optimality gap vs $\\mu$", fontsize=9.5)
    ax_gap.set_xlim(mu_min, mu_max)
    ax_gap.legend(fontsize=7.5, facecolor=WHITE, edgecolor=GRAY, ncol=2)

    # ── Panel 4: summary table ────────────────────────────────────────────────
    col_labels = ["Variant", r"$\mu_{\rm bin}$",
                  r"Spearman $\rho$", r"$p$-value",
                  "Monotone?", "Significant?"]
    rows = []
    for name, _ in family:
        mb       = mu_bins[name]
        rho, pv  = rho_pval[name]
        mono     = ("↑ H (degrades)" if rho > 0 else "↓ H (improves)")
        sig      = "yes  (p<0.05)" if pv < 0.05 else "no"
        rows.append([name,
                     f"{mb:.4f}",
                     f"{rho:.3f}",
                     f"{pv:.4f}",
                     mono, sig])

    tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                       cellLoc="center", loc="center",
                       bbox=[0.0, 0.0, 1.0, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRAY)
        if r == 0:
            cell.set_facecolor(LGRAY)
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor(WHITE if r % 2 == 0 else "#f9f9f9")
    ax_tbl.set_title(
        r"Statistical summary — Spearman trend test on $\bar H$ vs $\mu$",
        fontsize=9.5, pad=6)

    fig.suptitle(
        f"OIM Statistical Robustness — {_graph_info(W)}"
        f"  |  {n_trials} trials/pt  |  bootstrap 95\\% CI  |  Spearman trend test",
        fontsize=12, fontweight="bold")
    plt.savefig("experiment_E_robustness.png")
    plt.savefig("experiment_E_robustness.pdf")
    print("[Exp E] Saved experiment_E_robustness.{png,pdf}.  Close to exit.")
    plt.show()



# =============================================================================
# Experiment F — Focused µ_bin Comparison (direct numerical proof)
# =============================================================================
def experiment_F(W: np.ndarray,
                 mu_min:     float = 0.10,
                 mu_max:     float = 4.00,
                 n_trials:   int   = 80,
                 n_boot:     int   = 600,
                 t_end:      float = 50.0,
                 n_pts:      int   = 500,
                 seed:       int   = 42,
                 multipliers: List[float] = None) -> None:
    """
    Focused comparison:  H at mu_bin  vs  H at k * mu_bin  (k > 1).

    For each graph variant the function:
      1. Identifies  mu_bin  (Remark 7 theory threshold).
      2. Runs  n_trials  random restarts at each anchor
             mu_bin * [1.0, 1.5, 2.0, 2.5, 3.0].
      3. Bootstraps 95% CI on mean H at every anchor.
      4. Runs a one-sided Mann–Whitney U test:
             H(mu_bin)  <  H(k * mu_bin)   for k > 1.

    Two windows are opened:
      Window 1  —  Grouped violin plots (H distributions per anchor & variant).
      Window 2  —  Delta-H bar chart + Mann–Whitney p-value heatmap.
    """
    if multipliers is None:
        multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    N   = W.shape[0]

    # ── Ground truth & graph family ───────────────────────────────────────────
    H_ground = _ground_H(W)
    print(f"[Exp F] N={N}  H*={H_ground:.3f}")

    perturb_modes = ["drop_edge", "add_edge", "weight_noise", "rewire"]
    family: List[Tuple[str, np.ndarray]] = [("original", W.copy())]
    for mode in perturb_modes:
        W_p, lbl = _perturb(W, mode, rng)
        family.append((lbl, W_p))

    n_var  = len(family)
    n_mult = len(multipliers)
    palette    = [BLUE, ORANGE, GREEN, RED, PURPLE, BROWN]
    linestyles = ["-", "--", "-.", ":", (0,(3,1,1,1))]

    # ── Data collection ───────────────────────────────────────────────────────
    # all_H[variant_idx][mult_idx] = np.ndarray of H values (length n_trials)
    all_H:    List[List[np.ndarray]] = []
    mu_bins:  List[float] = []
    anchors:  List[List[float]] = []  # actual mu values used

    for v_idx, (name, W_v) in enumerate(family):
        oim_v     = OIMMaxCut(W_v, mu=1.0, seed=0)
        mu_b      = oim_v.binarization_threshold()["mu_bin"]
        H_gnd_v   = _ground_H(W_v)
        mu_bins.append(mu_b)
        anch_row  = [mu_b * m for m in multipliers]
        anchors.append(anch_row)

        print(f"\n[Exp F] Variant: «{name}»  mu_bin={mu_b:.4f}  H*={H_gnd_v:.3f}")
        row_H: List[np.ndarray] = []
        for m_idx, mu_val in enumerate(anch_row):
            oim = OIMMaxCut(W_v, mu=mu_val, seed=seed)
            H_arr = np.array([_one_trial(oim, (0., t_end), n_pts)[0]
                               for _ in range(n_trials)])
            row_H.append(H_arr)
            print(f"  k={multipliers[m_idx]:.1f}×  mu={mu_val:.4f}  "
                  f"mean H={H_arr.mean():.3f}  ±{H_arr.std():.3f}",
                  end="  ")
            # Mann-Whitney vs baseline (k=1.0)
            if m_idx > 0:
                stat, pval = stats.mannwhitneyu(
                    row_H[0], H_arr, alternative="less")  # H(µ_bin) < H(k*µ_bin)
                print(f"MW p={pval:.4f} {'✓' if pval < 0.05 else '✗'}", end="")
            print()
        all_H.append(row_H)

    # ── Pre-compute bootstrap CIs and delta-H ─────────────────────────────────
    mean_H_mat  = np.zeros((n_var, n_mult))   # mean H per (variant, multiplier)
    ci_lo_mat   = np.zeros((n_var, n_mult))
    ci_hi_mat   = np.zeros((n_var, n_mult))
    delta_mat   = np.zeros((n_var, n_mult))   # delta = mean_H(k) - mean_H(1.0)
    dci_lo_mat  = np.zeros((n_var, n_mult))
    dci_hi_mat  = np.zeros((n_var, n_mult))
    pval_mat    = np.ones((n_var, n_mult))    # Mann-Whitney p-values

    for v in range(n_var):
        H_base = all_H[v][0]
        for m in range(n_mult):
            H_m = all_H[v][m]
            mean_H_mat[v, m] = H_m.mean()
            lo, hi = _bootstrap_ci(H_m, np.mean, n_boot)
            ci_lo_mat[v, m]  = lo
            ci_hi_mat[v, m]  = hi

            delta_mat[v, m]  = H_m.mean() - H_base.mean()

            # Bootstrap CI on delta
            boot_deltas = [
                np.random.choice(H_m, n_trials, replace=True).mean()
                - np.random.choice(H_base, n_trials, replace=True).mean()
                for _ in range(n_boot)]
            dci_lo_mat[v, m] = np.percentile(boot_deltas, 2.5)
            dci_hi_mat[v, m] = np.percentile(boot_deltas, 97.5)

            if m > 0:
                _, pval = stats.mannwhitneyu(H_base, H_m, alternative="less")
                pval_mat[v, m] = pval

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    hdr_mult = "  ".join([f"k={m:.1f}× p-val" for m in multipliers[1:]])
    print(f"  {'Variant':<26}  {'mu_bin':>8}  {hdr_mult}")
    print(f"{'─'*80}")
    for v, (name, _) in enumerate(family):
        p_strs = "  ".join([f"{pval_mat[v,m]:.4f}{'*' if pval_mat[v,m]<0.05 else ' '}"
                             for m in range(1, n_mult)])
        print(f"  {name:<26}  {mu_bins[v]:>8.4f}  {p_strs}")
    print(f"{'─'*80}\n* = significant at p < 0.05 (one-sided Mann–Whitney: H(µ_bin) < H(k·µ_bin))\n")

    mult_labels = [f"{m:.1f}×" for m in multipliers]
    short_names = [n[:18] for n, _ in family]

    # =========================================================================
    # WINDOW 1 — Grouped violin plots
    # =========================================================================
    fig1 = plt.figure(figsize=(16, 7), facecolor=WHITE,
                      num="Exp F — Window 1: H distributions at each µ anchor")
    gs1  = GridSpec(1, n_var, figure=fig1,
                    left=0.06, right=0.98, top=0.87, bottom=0.13,
                    wspace=0.28)

    for v, (name, _) in enumerate(family):
        ax = fig1.add_subplot(gs1[v])
        _style_ax(ax)

        data_v  = [all_H[v][m] for m in range(n_mult)]
        col     = palette[v % len(palette)]
        positions = np.arange(n_mult)

        parts = ax.violinplot(data_v, positions=positions,
                              showmeans=True, showmedians=False,
                              showextrema=False, widths=0.7)
        for i, pc in enumerate(parts["bodies"]):
            alpha = 0.85 if i == 0 else 0.35 + 0.15 * i
            pc.set_facecolor(col)
            pc.set_edgecolor(BLACK)
            pc.set_linewidth(0.6)
            pc.set_alpha(min(alpha, 0.85))
        parts["cmeans"].set_color(BLACK)
        parts["cmeans"].set_linewidth(1.5)

        # CI error bars on means
        means  = mean_H_mat[v]
        ax.errorbar(positions, means,
                    yerr=[means - ci_lo_mat[v], ci_hi_mat[v] - means],
                    fmt="none", ecolor=BLACK, elinewidth=1.2,
                    capsize=4, capthick=1.2, zorder=5)

        # Mark baseline (k=1) with a dashed horizontal
        ax.axhline(means[0], color=col, lw=1.0, ls="--", alpha=0.7,
                   label=f"$H(\\mu_{{\\rm bin}})={means[0]:.2f}$")

        # Significance stars above non-baseline violins
        y_top = max(h.max() for h in data_v) * 1.02
        for m in range(1, n_mult):
            if pval_mat[v, m] < 0.001:
                star = "***"
            elif pval_mat[v, m] < 0.01:
                star = "**"
            elif pval_mat[v, m] < 0.05:
                star = "*"
            else:
                star = "n.s."
            ax.text(m, y_top, star, ha="center", va="bottom",
                    fontsize=8, color=RED if star != "n.s." else GRAY)

        ax.set_xticks(positions)
        ax.set_xticklabels(mult_labels, fontsize=8.5)
        ax.set_xlabel(r"$k \times \mu_{\rm bin}$", fontsize=10)
        if v == 0:
            ax.set_ylabel("Ising $H$ (binarised)", fontsize=10)
        ax.set_title(f"{name[:20]}\n"
                     f"$\\mu_{{\\rm bin}}={mu_bins[v]:.4f}$",
                     fontsize=8.5)
        ax.legend(fontsize=7.5, loc="upper left",
                  facecolor=WHITE, edgecolor=GRAY)

    fig1.suptitle(
        fr"Exp F — $H$ distributions at $k \times \mu_{{\rm bin}}$  "
        f"({n_trials} trials/anchor)  "
        r"|  * / ** / *** = Mann–Whitney $H(\mu_{\rm bin}) < H(k\cdot\mu_{\rm bin})$",
        fontsize=11, fontweight="bold")

    plt.savefig("experiment_F_violins.png")
    plt.savefig("experiment_F_violins.pdf")
    print("[Exp F] Window 1 saved → experiment_F_violins.{png,pdf}")

    # =========================================================================
    # WINDOW 2 — Delta-H bar chart + p-value heatmap
    # =========================================================================
    fig2 = plt.figure(figsize=(15, 8), facecolor=WHITE,
                      num="Exp F — Window 2: Delta-H and significance")
    gs2  = GridSpec(1, 2, figure=fig2,
                    left=0.07, right=0.97, top=0.87, bottom=0.12,
                    wspace=0.38)
    ax_delta = fig2.add_subplot(gs2[0])
    ax_heat  = fig2.add_subplot(gs2[1])
    _style_ax(ax_delta)
    ax_heat.set_facecolor(WHITE)

    # ── Panel A: grouped bar chart of delta_H ─────────────────────────────────
    bar_w   = 0.14
    x_base  = np.arange(n_var)

    # Only plot k > 1 (m = 1..n_mult-1)
    n_delta = n_mult - 1
    offsets = np.linspace(-(n_delta-1)/2, (n_delta-1)/2, n_delta) * bar_w * 1.15

    for di, m in enumerate(range(1, n_mult)):
        deltas    = delta_mat[:, m]
        err_lo    = deltas - dci_lo_mat[:, m]
        err_hi    = dci_hi_mat[:, m] - deltas
        shade     = 0.45 + 0.15 * di     # progressively darker
        blues_cmap = plt.get_cmap('Blues')
        bar_col = blues_cmap(0.35 + 0.13 * di)  
        bars = ax_delta.bar(x_base + offsets[di], deltas,
                            width=bar_w, color=bar_col,
                            edgecolor=BLACK, linewidth=0.5,
                            label=f"$k={multipliers[m]:.1f}\\times\\mu_{{\\rm bin}}$",
                            zorder=3)
        ax_delta.errorbar(x_base + offsets[di], deltas,
                          yerr=[err_lo, err_hi],
                          fmt="none", ecolor=BLACK, elinewidth=1.0,
                          capsize=3, capthick=1.0, zorder=4)
        # significance stars
        for vi in range(n_var):
            pv = pval_mat[vi, m]
            star = ("***" if pv < 0.001 else
                    "**"  if pv < 0.01  else
                    "*"   if pv < 0.05  else "")
            if star:
                ax_delta.text(x_base[vi] + offsets[di],
                              deltas[vi] + err_hi[vi] + 0.05,
                              star, ha="center", va="bottom",
                              fontsize=7, color=RED)

    ax_delta.axhline(0., color=BLACK, lw=1.2, ls="--",
                     label="$\\Delta H = 0$ (no change)")
    ax_delta.set_xticks(x_base)
    ax_delta.set_xticklabels(short_names, rotation=22, ha="right", fontsize=8)
    ax_delta.set_ylabel(r"$\Delta H = \bar{H}(k\cdot\mu_{\rm bin})"
                        r"- \bar{H}(\mu_{\rm bin})$", fontsize=10)
    ax_delta.set_title(
        r"Quality degradation $\Delta H$ relative to $\mu_{\rm bin}$"
        "\n(95\\% bootstrap CI  |  * p<0.05, ** p<0.01, *** p<0.001)",
        fontsize=9.5)
    ax_delta.legend(fontsize=8, facecolor=WHITE, edgecolor=GRAY,
                    loc="upper left")

    # ── Panel B: p-value heatmap ──────────────────────────────────────────────
    # Show -log10(p) for k > 1
    pmat_sub  = pval_mat[:, 1:]    # shape (n_var, n_delta)
    log_p     = -np.log10(np.clip(pmat_sub, 1e-10, 1.0))
    im        = ax_heat.imshow(log_p, aspect="auto",
                               cmap="YlOrRd", vmin=0, vmax=max(4, log_p.max()))

    ax_heat.set_xticks(np.arange(n_delta))
    ax_heat.set_xticklabels([f"$k={multipliers[m]:.1f}\\times$"
                              for m in range(1, n_mult)], fontsize=9)
    ax_heat.set_yticks(np.arange(n_var))
    ax_heat.set_yticklabels(short_names, fontsize=8.5)
    ax_heat.set_xlabel(r"Anchor $k \times \mu_{\rm bin}$", fontsize=10)
    ax_heat.set_title(
        r"$-\log_{10}(p)$ heatmap"
        "\n"
        r"(Mann–Whitney: $H(\mu_{\rm bin}) < H(k\cdot\mu_{\rm bin})$)"
        "\ndarker = more significant",
        fontsize=9.5)

    cbar2 = fig2.colorbar(im, ax=ax_heat, shrink=0.75, pad=0.02)
    cbar2.set_label(r"$-\log_{10}(p)$", fontsize=9)
    cbar2.ax.tick_params(labelsize=8)
    cbar2.ax.axhline(-np.log10(0.05), color=BLACK, lw=1.5, ls="--")

    # Annotate cells
    for vi in range(n_var):
        for di in range(n_delta):
            pv  = pmat_sub[vi, di]
            lp  = log_p[vi, di]
            txt = ("***" if pv < 0.001 else
                   "**"  if pv < 0.01  else
                   "*"   if pv < 0.05  else "n.s.")
            col_txt = WHITE if lp > 2.0 else BLACK
            ax_heat.text(di, vi, txt, ha="center", va="center",
                         fontsize=9, color=col_txt, fontweight="bold")

    # Spine styling for heatmap
    for sp in ax_heat.spines.values():
        sp.set_edgecolor(BLACK)
        sp.set_linewidth(0.8)

    fig2.suptitle(
        fr"Exp F — Direct evidence: $H(\mu_{{\rm bin}}) < H(k\cdot\mu_{{\rm bin}})$"
        f"  |  {n_trials} trials/anchor  |  {_graph_info(W)}",
        fontsize=11, fontweight="bold")

    plt.savefig("experiment_F_significance.png")
    plt.savefig("experiment_F_significance.pdf")
    print("[Exp F] Window 2 saved → experiment_F_significance.{png,pdf}")
    plt.show()

# =============================================================================
# CLI
# =============================================================================
def _parse(argv=None):
    p = argparse.ArgumentParser(
        description="Robust OIM Experiments D (bifurcation) & E (robustness)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--graph",       "-g", required=True, metavar="FILE",
                   help="Path to graph file")
    p.add_argument("--experiments", "-e", default="DE",
                   help="Which to run: D, E, F, or any combination e.g. DEF")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--mu-min",      type=float, default=0.1)
    p.add_argument("--mu-max",      type=float, default=4.0)

    d = p.add_argument_group("Experiment D — bifurcation diagram")
    d.add_argument("--d-sim-steps", type=int,   default=25,
                   help="mu grid points for simulation panel")
    d.add_argument("--d-trials",    type=int,   default=50,
                   help="Random trials per mu step (simulation panel)")
    d.add_argument("--d-t-end",     type=float, default=50.0)
    d.add_argument("--d-n-points",  type=int,   default=500)

    e = p.add_argument_group("Experiment E — statistical robustness")
    e.add_argument("--e-mu-steps",  type=int,   default=20)
    e.add_argument("--e-trials",    type=int,   default=60,
                   help="Random trials per mu step per graph variant")
    e.add_argument("--e-n-boot",    type=int,   default=500,
                   help="Bootstrap resamples for CI")
    e.add_argument("--e-t-end",     type=float, default=50.0)
    e.add_argument("--e-n-points",  type=int,   default=500)

    f = p.add_argument_group("Experiment F — focused mu_bin comparison")
    f.add_argument("--f-trials",    type=int,   default=80,
                   help="Random trials per anchor point per variant")
    f.add_argument("--f-n-boot",    type=int,   default=600,
                   help="Bootstrap resamples")
    f.add_argument("--f-t-end",     type=float, default=50.0)
    f.add_argument("--f-n-points",  type=int,   default=500)
    f.add_argument("--f-multipliers", type=float, nargs="+",
                   default=[1.0, 1.5, 2.0, 2.5, 3.0],
                   help="mu_bin multipliers to evaluate (first must be 1.0)")

    return p.parse_args(argv)


def main(argv=None):
    args = _parse(argv)
    exps = args.experiments.upper()

    print(f"Loading graph: {args.graph}")
    W = _load_graph(args.graph)
    print(f"  {_graph_info(W)}")
    np.random.seed(args.seed)

    if "D" in exps:
        print("\n" + "="*60)
        print("EXPERIMENT D — Bifurcation Diagram")
        print("="*60)
        experiment_D(
            W,
            mu_min       = args.mu_min,
            mu_max       = args.mu_max,
            n_mu_sim     = args.d_sim_steps,
            n_trials_sim = args.d_trials,
            t_end        = args.d_t_end,
            n_pts        = args.d_n_points,
            seed         = args.seed)

    if "E" in exps:
        print("\n" + "="*60)
        print("EXPERIMENT E — Statistical Robustness")
        print("="*60)
        experiment_E(
            W,
            mu_min   = args.mu_min,
            mu_max   = args.mu_max,
            n_mu     = args.e_mu_steps,
            n_trials = args.e_trials,
            n_boot   = args.e_n_boot,
            t_end    = args.e_t_end,
            n_pts    = args.e_n_points,
            seed     = args.seed)

    if "F" in exps:
        print("\n" + "="*60)
        print("EXPERIMENT F — Focused mu_bin Comparison")
        print("="*60)
        experiment_F(
            W,
            mu_min      = args.mu_min,
            mu_max      = args.mu_max,
            n_trials    = args.f_trials,
            n_boot      = args.f_n_boot,
            t_end       = args.f_t_end,
            n_pts       = args.f_n_points,
            seed        = args.seed,
            multipliers = args.f_multipliers)

    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()
