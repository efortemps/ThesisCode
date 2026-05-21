#!/usr/bin/env python3
"""
maxcut_mubin_conjecture.py
──────────────────────────────────────────────────────────────────────────────
Statistical test of the conjecture:

    "The binarisation threshold mu_bin (the smallest mu at which any binary
     equilibrium becomes asymptotically stable) consistently achieves the
     global Max-Cut optimum."

Setting
───────
• N = 18 nodes — practical ceiling for exact 2^N equilibrium enumeration
  (2^18 = 262 144 spin configs; each needs an 18×18 eigen-decomposition)
• 50 random Erdos-Renyi graphs, antiferromagnetic edges only (J_ij = -W_ij)
  -> this is exactly the Max-Cut Ising formulation
• 10 random initial conditions per graph, drawn from uniform(-pi, pi)
• For each graph:
    1. Call OIMMaxCut.binarization_threshold()  [original, untouched method]
       -> gives mu_bin_exact + all_thresholds dict (one 2^N eigen-pass)
    2. Post-process all_thresholds in THIS file to derive best_cut, H, cut
       per equilibrium — no second eigendecomposition needed
    3. Run 10 ODE trajectories at mu_bin_exact; record which ones reach
       the global optimum

Figures
───────
Figure 1 — Histogram: per-graph count of ICs (0..n_init) converging to
           the global optimum at mu_bin_exact.  Right: CDF + summary stats.

Figure 2 — Spectral fingerprint: for 6 representative graphs, scatter
           (lambda_max(D(phi*)) vs H(phi*)) for all 2^N equilibria.
           Colour = cut quality.  Green = global-optimum configs.
           Red dashed line = mu_bin_exact.
           If optimum configs cluster LEFT of the line they are the FIRST
           equilibria stabilised as mu increases — the conjecture's backbone.

Figure 3 — Summary statistics:
           (a) mu_bin_exact distribution over 50 graphs
           (b) Approximation ratio: best cut at mu_bin_exact / true opt
           (c) Success-fraction bar chart (>=1 IC / all ICs / 0 ICs hit opt)

Usage
─────
python maxcut_mubin_conjecture.py [--N 18] [--n_graphs 50] [--p1 0.5]
                                  [--n_init 10] [--t_end 120] [--n_points 400]
                                  [--rtol 1e-6] [--atol 1e-6]
                                  [--bin_tol 0.05] [--seed 42] [--save]
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import ast
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

from OIM_Experiment.src.OIM_mu import OIMMaxCut

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 10,
    "axes.edgecolor"   : "black",
    "axes.linewidth"   : 0.8,
    "xtick.color"      : "black",
    "ytick.color"      : "black",
    "text.color"       : "black",
    "figure.facecolor" : "white",
    "axes.facecolor"   : "white",
    "legend.framealpha": 0.92,
    "legend.edgecolor" : "#b0b0b0",
    "legend.facecolor" : "white",
    "legend.labelcolor": "black",
})

WHITE    = "#ffffff"
BLACK    = "#000000"
GRAY     = "#b0b0b0"
LIGHT    = "#e6e6e6"
C_BLUE   = "#4C72B0"
C_ORANGE = "#DD8452"
C_GREEN  = "#55a868"
C_RED    = "#c44e52"
C_AMBER  = "#ffb74d"
C_PURPLE = "#8172b2"


def _ax_style(ax, title="", xlabel="", ylabel="", titlesize=10):
    ax.set_facecolor(WHITE)
    ax.tick_params(colors=BLACK, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(BLACK); sp.set_linewidth(0.8)
    ax.grid(True, color=LIGHT, linewidth=0.5, zorder=0)
    if title:  ax.set_title(title,  color=BLACK, fontsize=titlesize,
                             fontweight="bold", pad=5)
    if xlabel: ax.set_xlabel(xlabel, color=BLACK, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, color=BLACK, fontsize=10)


# ═══════════════════════════════════════════════════════════════════════════════
# Graph generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_er_graph(N: int, p1: float, seed: int) -> np.ndarray:
    """Unweighted ER graph. Returns symmetric W >= 0 with unit weights."""
    rng = np.random.default_rng(seed)
    W   = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < p1:
                W[i, j] = W[j, i] = 1.0
    return W


# ═══════════════════════════════════════════════════════════════════════════════
# Post-process binarization_threshold() output
# ═══════════════════════════════════════════════════════════════════════════════

def enrich_thresholds(W: np.ndarray, all_thresholds: dict) -> list:
    """
    Given the 'all_thresholds' dict returned by OIMMaxCut.binarization_threshold()
    — which maps str(list(bits)) -> lmax_D — compute cut and H for each
    equilibrium WITHOUT re-running any eigendecomposition.

    Returns
    -------
    per_eq : list of dicts, one per equilibrium, with keys:
        bits    : tuple of ints (0/1)
        lmax_D  : float  (already computed by binarization_threshold)
        cut     : float  (Max-Cut value of this spin assignment)
        H       : float  (Ising Hamiltonian = W_total - 2*cut, up to sign)
    Sorted by lmax_D ascending (easiest-to-stabilise first).
    """
    per_eq = []
    for key, lmax in all_thresholds.items():
        bits  = tuple(ast.literal_eval(key))           # "[0,1,0,...]" -> tuple
        sigma = np.array([1.0 if b == 0 else -1.0 for b in bits])
        cut   = 0.25 * float(np.sum(W * (1.0 - sigma[:, None] * sigma[None, :])))
        H     = 0.5  * float(sigma @ W @ sigma)
        per_eq.append({"bits": bits, "lmax_D": float(lmax), "cut": cut, "H": H})

    per_eq.sort(key=lambda x: x["lmax_D"])
    return per_eq


# ═══════════════════════════════════════════════════════════════════════════════
# Per-graph experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_graph(W: np.ndarray, g_idx: int, args) -> dict:
    """
    Full experiment for one graph.

    1. OIMMaxCut.binarization_threshold()  [original method, single 2^N pass]
       -> mu_bin_exact + all_thresholds dict
    2. enrich_thresholds()  [this file]
       -> per_eq list with (bits, lmax_D, cut, H) — no extra eigen calls
    3. ODE trajectories at mu_bin_exact + 0.1 (small additive noise)
    4. Count how many ICs land on the global optimum
    """
    N   = W.shape[0]
    oim = OIMMaxCut(W, mu=1.0, seed=args.seed)   # mu is a placeholder here

    # ── Step 1: binarization_threshold (original OIMMaxCut method) ────────────
    bin_data       = oim.binarization_threshold()
    mu_bin_exact   = bin_data["mu_bin"]
    all_thresholds = bin_data["all_thresholds"]   # {str(bits): lmax_D}

    # ── Step 2: enrich in this file (cut + H, no extra eigen calls) ───────────
    per_eq   = enrich_thresholds(W, all_thresholds)
    best_cut = max(row["cut"] for row in per_eq)
    w_total  = float(np.sum(np.triu(W)))
    n_opt_configs = sum(1 for row in per_eq if abs(row["cut"] - best_cut) < 1e-9)

    # ── Step 3: random initial conditions ─────────────────────────────────────
    rng   = np.random.default_rng(args.seed + g_idx * 1000)
    phi0s = [rng.uniform(-np.pi, np.pi, N) for _ in range(args.n_init)]

    # ── Step 4: ODE at mu_bin_exact ───────────────────────────────────────────
    oim.mu = mu_bin_exact + 0.1 # small additive noise.
    sols   = oim.simulate_many(phi0s,
                               t_span=(0., args.t_end),
                               n_points=args.n_points,
                               rtol=args.rtol, atol=args.atol)

    cuts_bin  = []
    bin_flags = []
    for sol in sols:
        theta_f = sol.y[:, -1]
        sigma   = np.sign(np.cos(theta_f)); sigma[sigma == 0] = 1.0
        cut     = 0.25 * float(np.sum(W * (1.0 - sigma[:, None] * sigma[None, :])))
        res     = float(np.max(np.abs(np.sin(theta_f))))
        cuts_bin.append(cut)
        bin_flags.append(res < args.bin_tol)

    cuts_bin  = np.array(cuts_bin)
    bin_flags = np.array(bin_flags)
    n_opt_bin = int(np.sum(np.abs(cuts_bin - best_cut) < 1e-6))

    return dict(
        g_idx        = g_idx,
        W            = W,
        N            = N,
        best_cut     = best_cut,
        w_total      = w_total,
        mu_bin_exact = mu_bin_exact,
        n_opt_configs= n_opt_configs,
        per_eq       = per_eq,           # list of {bits, lmax_D, cut, H}
        phi0s        = phi0s,
        cuts_bin     = cuts_bin.tolist(),
        bin_flags    = bin_flags.tolist(),
        n_opt_bin    = n_opt_bin,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Run full ensemble
# ═══════════════════════════════════════════════════════════════════════════════

def run_ensemble(args) -> list:
    print(f"\n{'='*65}")
    print(f"  Max-Cut mu_bin Conjecture Experiment")
    print(f"  N={args.N}  p1={args.p1}  n_graphs={args.n_graphs}")
    print(f"  n_init={args.n_init}  t_end={args.t_end}")
    print(f"  2^N = {2**args.N} equilibria per graph (enumerated exactly)")
    print(f"{'='*65}")
    print(f"  {'g':>3}  {'|E|':>4}  {'best_cut':>9}  {'mu_bin':>8}  {'n_opt@mu_bin':>13}")
    print("  " + "─" * 50)

    results = []
    t0 = time.perf_counter()

    for g_idx in range(args.n_graphs):
        W = generate_er_graph(args.N, args.p1, seed=args.seed + g_idx)
        r = run_one_graph(W, g_idx, args)
        results.append(r)
        n_edges = int(np.sum(W)) // 2
        print(f"  {g_idx+1:>3}  {n_edges:>4}  {r['best_cut']:>9.1f}  "
              f"{r['mu_bin_exact']:>8.4f}  "
              f"{r['n_opt_bin']:>5}/{args.n_init:>2}")

    elapsed = time.perf_counter() - t0
    print(f"\n  Total: {elapsed:.1f}s  ({elapsed/args.n_graphs:.1f}s/graph)")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Console summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(results, args):
    n_init   = args.n_init
    n_graphs = len(results)
    n_opt    = np.array([r["n_opt_bin"]    for r in results])
    mu_bins  = np.array([r["mu_bin_exact"] for r in results])
    approx   = np.array([max(r["cuts_bin"]) / max(r["best_cut"], 1e-9)
                          for r in results])

    print(f"\n{'='*65}")
    print(f"  SUMMARY  |  N={args.N}  p1={args.p1}  "
          f"{n_graphs} graphs  n_init={n_init}")
    print(f"{'='*65}")
    print(f"  mu_bin_exact:  mean={mu_bins.mean():.4f}  "
          f"std={mu_bins.std():.4f}  "
          f"range=[{mu_bins.min():.4f}, {mu_bins.max():.4f}]")
    print(f"\n  # ICs reaching global opt at mu_bin_exact:")
    for k in range(n_init + 1):
        cnt = int(np.sum(n_opt == k))
        bar = "█" * cnt
        print(f"    {k:>2}/{n_init:<2} {bar} ({cnt} graphs, "
              f"{100*cnt/n_graphs:.1f}%)")
    frac_any = float(np.mean(n_opt >= 1))
    frac_all = float(np.mean(n_opt == n_init))
    print(f"\n  At least 1 IC hits global opt : {frac_any*100:.1f}% of graphs")
    print(f"  All {n_init} ICs hit global opt  : {frac_all*100:.1f}% of graphs")
    print(f"  Mean ICs hitting opt           : {n_opt.mean():.2f} / {n_init}")
    print(f"  Approx ratio (best@mu_bin/opt) : "
          f"mean={approx.mean():.4f}  min={approx.min():.4f}")
    print(f"{'='*65}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Main histogram + CDF
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure1(results, args):
    n_init   = args.n_init
    n_graphs = len(results)
    n_opt    = np.array([r["n_opt_bin"] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=WHITE)
    fig.subplots_adjust(wspace=0.35, left=0.08, right=0.97, top=0.87, bottom=0.13)

    # ── Left: histogram ──────────────────────────────────────────────────────
    ax      = axes[0]
    bins    = np.arange(-0.5, n_init + 1.5, 1.0)
    counts, _ = np.histogram(n_opt, bins=bins)
    colours = [C_RED if k == 0 else (C_GREEN if k == n_init else C_BLUE)
               for k in range(n_init + 1)]
    bars = ax.bar(np.arange(n_init + 1), counts, color=colours,
                  edgecolor=BLACK, linewidth=0.6, alpha=0.82, zorder=3)
    for k, (bar, cnt) in enumerate(zip(bars, counts)):
        if cnt > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{cnt}\n({100*cnt/n_graphs:.0f}%)",
                    ha="center", va="bottom", fontsize=8, color=BLACK)

    ax.set_xticks(np.arange(n_init + 1))
    ax.set_xlim(-0.7, n_init + 0.7)
    ax.set_ylim(0, max(counts) * 1.28)
    ax.axvline(np.mean(n_opt), color=C_AMBER, linewidth=2.0,
               linestyle="--", zorder=5, label=f"mean = {np.mean(n_opt):.2f}")
    legend_patches = [
        mpatches.Patch(color=C_RED,   label="0 ICs reach optimum"),
        mpatches.Patch(color=C_BLUE,  label="1 to n_init-1 ICs"),
        mpatches.Patch(color=C_GREEN, label=f"All {n_init} ICs reach optimum"),
        mlines.Line2D([0], [0], color=C_AMBER, lw=2, ls="--",
                      label=f"mean = {np.mean(n_opt):.2f}"),
    ]
    ax.legend(handles=legend_patches, fontsize=8.5, loc="upper left")
    _ax_style(ax,
              title=(f"ICs (out of {n_init}) reaching the global Max-Cut optimum\n"),
              xlabel=f"# ICs converging to global optimum (out of {n_init})",
              ylabel="# graphs")

    # ── Right: CDF ───────────────────────────────────────────────────────────
    ax     = axes[1]
    x_vals = np.arange(n_init + 1)
    cdf    = np.array([np.mean(n_opt >= k) for k in x_vals])
    ax.step(x_vals, cdf, where="post", color=C_BLUE, linewidth=2.2)
    ax.fill_between(x_vals, cdf, step="post", alpha=0.15, color=C_BLUE)
    ax.set_xticks(x_vals)
    ax.set_ylim(-0.02, 1.10)
    ax.set_xlim(-0.3, n_init + 0.3)
    ax.text(0.97, 0.05,
            f"P(at least 1 IC hits opt) = {cdf[1]*100:.1f}%\n"
            f"P(ALL {n_init} ICs hit opt)  = {cdf[n_init]*100:.1f}%\n",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=WHITE,
                      edgecolor=GRAY, alpha=0.95))
    ax.legend(fontsize=9)
    _ax_style(ax,
              title=(f"Cumulative: P(# ICs >= k reaching global optimum)"),
              xlabel="k (minimum # ICs that find the optimum)",
              ylabel="fraction of graphs")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Spectral fingerprint (6 representative graphs)
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure2(results, args, n_show=6):
    """
    Scatter (lambda_max(D(phi*)) vs H(phi*)) for all 2^N equilibria.
    Colour = cut quality.  Green = global-optimum configs.
    Red dashed line = mu_bin_exact.

    Key diagnostic: if optimum configs (green) cluster to the LEFT of the
    red line, they have the SMALLEST lambda_max among all equilibria and are
    therefore the FIRST to be stabilised as mu increases from 0.
    This is the structural support for the conjecture.
    """
    n_opt_arr  = np.array([r["n_opt_bin"] for r in results])
    sorted_idx = np.argsort(n_opt_arr)
    mid        = len(sorted_idx) // 2
    show_idx   = list(dict.fromkeys(
        list(sorted_idx[:2]) +
        list(sorted_idx[mid-1:mid+1]) +
        list(sorted_idx[-2:])
    ))[:n_show]

    ncols = 3
    nrows = (len(show_idx) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(18, 5.5 * nrows), facecolor=WHITE)
    fig.subplots_adjust(hspace=0.50, wspace=0.38,
                        left=0.07, right=0.97, top=0.92, bottom=0.08)
    axes_flat = axes.ravel() if nrows > 1 else list(axes)

    for plot_idx, g_idx in enumerate(show_idx):
        ax     = axes_flat[plot_idx]
        r      = results[g_idx]
        per_eq = r["per_eq"]   # list of {bits, lmax_D, cut, H}

        lmax_all = np.array([row["lmax_D"] for row in per_eq])
        cut_all  = np.array([row["cut"]    for row in per_eq])
        H_all    = np.array([row["H"]      for row in per_eq])

        cmap_c = plt.get_cmap("RdYlGn")
        norm_c = mcolors.Normalize(vmin=cut_all.min(), vmax=cut_all.max())
        colours = [cmap_c(norm_c(c)) for c in cut_all]

        ax.scatter(lmax_all, H_all, c=colours, s=6, alpha=0.45,
                   zorder=2, edgecolors="none")

        opt_mask = np.abs(cut_all - r["best_cut"]) < 1e-6
        ax.scatter(lmax_all[opt_mask], H_all[opt_mask],
                   color=C_GREEN, s=40, zorder=5,
                   edgecolors=BLACK, linewidths=0.6,
                   label=f"global opt ({opt_mask.sum()} configs)")

        mu_b = r["mu_bin_exact"]
        ax.axvline(mu_b, color=C_RED, linewidth=1.8, linestyle="--",
                   zorder=6, label=f"$\\mu_{{\\rm bin}}={mu_b:.3f}$")

        opt_lmax  = lmax_all[opt_mask]
        frac_left = float(np.mean(opt_lmax <= mu_b + 1e-9))
        text_col  = C_GREEN if frac_left >= 0.5 else C_RED
        ax.text(0.97, 0.97,
                f"$n_{{\\rm opt}}@\\mu_{{\\rm bin}} = {r['n_opt_bin']}/{args.n_init}$\n"
                f"opt left of line: {frac_left*100:.0f}%",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, fontweight="bold", color=text_col,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE,
                          edgecolor=GRAY, alpha=0.92))

        ax.legend(fontsize=7, loc="lower right")
        _ax_style(ax,
                  title=(f"Graph {g_idx+1}  |E|={int(r['w_total'])}  "
                         f"opt cut={r['best_cut']:.0f}"),
                  xlabel="$\\lambda_{\\max}(D(\\phi^*))$",
                  ylabel="$H(\\phi^*)$",
                  titlesize=9)

    for i in range(len(show_idx), len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(
        f"Spectral fingerprint: $\\lambda_{{\\max}}(D)$ vs $H(\\phi^*)$ "
        f"for all $2^N$ equilibria",
        color=BLACK, fontsize=11, fontweight="bold")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Summary statistics
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure3(results, args):
    n_init   = args.n_init
    n_graphs = len(results)

    mu_bins = np.array([r["mu_bin_exact"] for r in results])
    n_opt   = np.array([r["n_opt_bin"]    for r in results])
    approx  = np.array([max(r["cuts_bin"]) / max(r["best_cut"], 1e-9)
                         for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5), facecolor=WHITE)
    fig.subplots_adjust(wspace=0.38, left=0.07, right=0.97,
                        top=0.87, bottom=0.13)

    # ── (a) mu_bin_exact distribution ────────────────────────────────────────
    ax = axes[0]
    ax.hist(mu_bins, bins=15, color=C_BLUE, alpha=0.72,
            edgecolor=BLACK, linewidth=0.5)
    ax.axvline(mu_bins.mean(), color=C_AMBER, linewidth=2.0, linestyle="--",
               label=f"mean = {mu_bins.mean():.3f}")
    ax.text(0.97, 0.97,
            f"mean = {mu_bins.mean():.3f}\n"
            f"std  = {mu_bins.std():.3f}\n"
            f"min  = {mu_bins.min():.3f}\n"
            f"max  = {mu_bins.max():.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=25,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE,
                      edgecolor=GRAY, alpha=0.95))
    ax.legend(fontsize=22, loc="upper left")
    _ax_style(ax,
              title=f"Distribution of $\\mu_{{\\rm bin}}$ over {n_graphs} graphs",
              xlabel="$\\mu_{\\rm bin}$ (exact, from $2^N$ enumeration)",
              ylabel="# graphs")

    # ── (c) Success-fraction bar chart ────────────────────────────────────────
    ax = axes[1]
    frac_any  = float(np.mean(n_opt >= 1))
    frac_all  = float(np.mean(n_opt == n_init))
    frac_none = float(np.mean(n_opt == 0))

    categories = [">=1 IC hits opt", f"All {n_init} ICs hit opt", "0 ICs hit opt"]
    values     = [frac_any * 100, frac_all * 100, frac_none * 100]
    colours_b  = [C_BLUE, C_GREEN, C_RED]
    bars = ax.bar(categories, values, color=colours_b, alpha=0.78,
                  edgecolor=BLACK, linewidth=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=25, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.axhline(100, color=GRAY, linewidth=0.9, linestyle="--", alpha=0.7)
    _ax_style(ax,
              title=(f"Graph proportions reaching global optimum"),
              xlabel="",
              ylabel="% of graphs")

    fig.suptitle(
        f"Summary Statistics",
        color=BLACK, fontsize=30, fontweight="bold")
    return fig

def _lighten(hex_col, factor=0.6):
    """Blend a hex colour toward white by `factor` (0=original, 1=white)."""
    hex_col = hex_col.lstrip("#")
    r, g, b = (int(hex_col[i:i+2], 16) for i in (0, 2, 4))
    r2 = int(r + (255 - r) * factor)
    g2 = int(g + (255 - g) * factor)
    b2 = int(b + (255 - b) * factor)
    return f"#{r2:02x}{g2:02x}{b2:02x}"

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Structural comparison: 4 worst vs 1 best performing graph
# ═══════════════════════════════════════════════════════════════════════════════

def graph_structural_metrics(r: dict) -> dict:
    """
    Compute all structural metrics for one graph result dict.

    Properties computed
    ───────────────────
    Basic
      n_edges        : number of edges
      density        : |E| / (N*(N-1)/2)
      w_total        : sum of edge weights / 2

    Degree statistics
      degrees        : (N,) array of node degrees
      deg_mean/std/min/max

    Spectral (graph Laplacian L = diag(degree) - W)
      lap_eigenvalues  : all N eigenvalues of L, sorted ascending
      fiedler          : 2nd smallest eigenvalue of L (algebraic connectivity)
                         > 0 iff graph is connected
      lap_spectral_gap : lambda_2 - lambda_1 of L
      lap_lambda_max   : largest eigenvalue of L

    Adjacency spectrum
      adj_eigenvalues  : eigenvalues of W sorted descending
      adj_lambda_max   : largest eigenvalue of W (spectral radius)
      adj_symmetric    : True if spectrum is symmetric (±λ), i.e. bipartite

    Bipartiteness
      is_bipartite     : True if 2-colorable (BFS check)
      bipart_ratio     : best_cut / w_total  (= 1 for bipartite graphs)

    Frustration (OIM context)
      frustration_index : fraction of edges that are "unsatisfied" at best cut
                          = (w_total - best_cut) / w_total
                          0 for bipartite, > 0 for frustrated graphs

    Clustering
      n_triangles      : number of triangles  (tr(W^3) / 6)
      mean_clustering  : mean local clustering coefficient
                         c_i = (# triangles through i) / (deg_i*(deg_i-1)/2)

    OIM-specific
      mu_bin_exact     : binarisation threshold (from experiment)
      n_opt_bin        : # ICs that found the global optimum at mu_bin
      n_opt_configs    : number of globally optimal spin configurations
      best_cut         : global Max-Cut value
    """
    W        = r["W"]
    N        = r["N"]
    best_cut = r["best_cut"]
    w_total  = r["w_total"]

    # ── Basic ─────────────────────────────────────────────────────────────────
    n_edges  = int(np.sum(W)) // 2
    density  = n_edges / max(N * (N - 1) // 2, 1)

    # ── Degree ────────────────────────────────────────────────────────────────
    degrees  = W.sum(axis=1)
    deg_mean = float(degrees.mean())
    deg_std  = float(degrees.std())
    deg_min  = int(degrees.min())
    deg_max  = int(degrees.max())

    # ── Laplacian spectrum ────────────────────────────────────────────────────
    L          = np.diag(degrees) - W
    lap_eigs   = np.sort(np.linalg.eigvalsh(L))
    fiedler    = float(lap_eigs[1]) if N > 1 else 0.0
    lap_gap    = float(lap_eigs[1] - lap_eigs[0]) if N > 1 else 0.0
    lap_lmax   = float(lap_eigs[-1])

    # ── Adjacency spectrum ────────────────────────────────────────────────────
    adj_eigs   = np.sort(np.linalg.eigvalsh(W))[::-1]   # descending
    adj_lmax   = float(adj_eigs[0])
    # Bipartite iff spectrum is symmetric: all eigenvalues come in ±λ pairs
    adj_sym    = bool(np.allclose(np.sort(adj_eigs),
                                   -np.sort(adj_eigs[::-1]), atol=1e-6))

    # ── Bipartiteness (BFS 2-coloring) ────────────────────────────────────────
    colour   = -np.ones(N, dtype=int)
    is_bip   = True
    colour[0] = 0
    queue     = [0]
    while queue and is_bip:
        v = queue.pop(0)
        for u in range(N):
            if W[v, u] > 0:
                if colour[u] == -1:
                    colour[u] = 1 - colour[v]
                    queue.append(u)
                elif colour[u] == colour[v]:
                    is_bip = False
                    break

    bipart_ratio     = best_cut / max(w_total, 1e-9)
    frustration_idx  = 1.0 - bipart_ratio   # 0 for bipartite

    # ── Clustering ────────────────────────────────────────────────────────────
    W3          = W @ W @ W
    n_triangles = int(round(np.trace(W3) / 6.0))

    clust = []
    for i in range(N):
        di = int(degrees[i])
        if di < 2:
            clust.append(0.0)
        else:
            # number of edges among neighbours of i
            nbrs   = np.where(W[i] > 0)[0]
            e_nbrs = 0
            for a in nbrs:
                for b in nbrs:
                    if a < b and W[a, b] > 0:
                        e_nbrs += 1
            clust.append(2.0 * e_nbrs / (di * (di - 1)))
    mean_clust = float(np.mean(clust))

    return dict(
        n_edges=n_edges, density=density, w_total=w_total,
        degrees=degrees,
        deg_mean=deg_mean, deg_std=deg_std, deg_min=deg_min, deg_max=deg_max,
        lap_eigenvalues=lap_eigs, fiedler=fiedler,
        lap_spectral_gap=lap_gap, lap_lambda_max=lap_lmax,
        adj_eigenvalues=adj_eigs, adj_lambda_max=adj_lmax,
        adj_symmetric=adj_sym,
        is_bipartite=is_bip, bipart_ratio=bipart_ratio,
        frustration_index=frustration_idx,
        n_triangles=n_triangles, mean_clustering=mean_clust,
        mu_bin_exact=r["mu_bin_exact"],
        n_opt_bin=r["n_opt_bin"], n_init=len(r["cuts_bin"]),
        n_opt_configs=r["n_opt_configs"],
        best_cut=best_cut,
    )


def make_figure5(results: list, args) -> plt.Figure:

    n_init   = args.n_init
    n_graphs = len(results)

    n_opt_arr = np.array([r["n_opt_bin"] for r in results])
    mu_arr    = np.array([r["mu_bin_exact"] for r in results])

    worst_idxs = sorted(range(n_graphs),
                        key=lambda i: (n_opt_arr[i], -mu_arr[i]))[:4]
    best_idx   = sorted(range(n_graphs),
                        key=lambda i: (-n_opt_arr[i], mu_arr[i]))[0]

    col_idxs  = worst_idxs + [best_idx]
    col_roles = ["worst"] * 4 + ["best"]
    n_cols    = 5

    metrics = [graph_structural_metrics(results[i]) for i in col_idxs]

    # ── NEW LAYOUT (split figure) ─────────────────────────────────────────────
    fig = plt.figure(figsize=(26, 22), facecolor=WHITE)

    outer = gridspec.GridSpec(2, 1, figure=fig,
                              height_ratios=[3.2, 1.8],
                              hspace=0.28,
                              left=0.04, right=0.98,
                              top=0.94, bottom=0.04)

    gs_top = gridspec.GridSpecFromSubplotSpec(
        3, n_cols, subplot_spec=outer[0],
        hspace=0.52, wspace=0.38
    )

    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, n_cols, subplot_spec=outer[1],
        wspace=0.38
    )

    col_header_colours = {
        "worst": C_RED,
        "best":  C_GREEN,
    }

    # ─────────────────────────────────────────────────────────────────────────
    for col, (g_idx, role, m) in enumerate(zip(col_idxs, col_roles, metrics)):
        r       = results[g_idx]
        W       = r["W"]
        N       = r["N"]
        hcol    = col_header_colours[role]

        # ── Row 0: Adjacency ────────────────────────────────────────────────
        ax0 = fig.add_subplot(gs_top[0, col])
        ax0.set_facecolor(WHITE)

        deg_order = np.argsort(-m["degrees"])
        W_sorted  = W[np.ix_(deg_order, deg_order)]

        im = ax0.imshow(W_sorted, cmap="Blues", vmin=0, vmax=1,
                        interpolation="nearest", aspect="auto")

        for k in range(N + 1):
            ax0.axhline(k - 0.5, color=LIGHT, linewidth=0.3)
            ax0.axvline(k - 0.5, color=LIGHT, linewidth=0.3)

        if N <= 12:
            labels = [str(deg_order[i]) for i in range(N)]
            ax0.set_xticks(range(N)); ax0.set_xticklabels(labels, fontsize=5.5)
            ax0.set_yticks(range(N)); ax0.set_yticklabels(labels, fontsize=5.5)
        else:
            ax0.set_xticks([]); ax0.set_yticks([])

        for sp in ax0.spines.values():
            sp.set_edgecolor(hcol); sp.set_linewidth(2.0 if role == "best" else 1.2)

        role_label = "★ BEST" if role == "best" else f"✗ WORST #{col+1}"
        ax0.set_title(
            f"{role_label}  —  Graph {g_idx+1}\n"
            f"$n_{{\\rm opt}}@\\mu_{{\\rm bin}} = {m['n_opt_bin']}/{n_init}$",
            color=hcol, fontsize=9, fontweight="bold", pad=4)

        if col == n_cols - 1:
            cb = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=6)

        # ── Row 1: Degree ───────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs_top[1, col])
        ax1.set_facecolor(WHITE)

        deg_vals, deg_counts = np.unique(m["degrees"].astype(int),
                                          return_counts=True)

        ax1.bar(deg_vals, deg_counts, color=hcol, alpha=0.75,
                edgecolor=BLACK, linewidth=0.5)

        ax1.axvline(m["deg_mean"], color=C_AMBER, linestyle="--",
                    label=f"{m['deg_mean']:.1f}")

        ax1.set_title("Degree distribution", fontsize=8.5, fontweight="bold")
        ax1.grid(True, axis="y", color=LIGHT, linewidth=0.5)
        ax1.legend(fontsize=7)

        # ── Row 2: Spectrum ─────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs_top[2, col])
        ax2.set_facecolor(WHITE)

        eigs = m["lap_eigenvalues"]
        x    = np.arange(N)

        ax2.plot(x, eigs, marker="o", color=hcol)
        ax2.fill_between(x, eigs, alpha=0.2, color=hcol)

        ax2.scatter([1], [m["fiedler"]], color=C_AMBER, zorder=5)

        ax2.set_title("Laplacian spectrum", fontsize=8.5, fontweight="bold")
        ax2.grid(True, color=LIGHT)

        # ── Row 3: PAPER-STYLE PANEL ────────────────────────────────────────
        ax3 = fig.add_subplot(gs_bot[0, col])
        ax3.axis("off")
        ax3.set_facecolor(WHITE)

        # ---- Title bar (paper style) ----
        ax3.text(0.0, 1.02,
                 "Key structural metrics",
                 fontsize=9.5, fontweight="bold",
                 color=hcol, ha="left", va="bottom",
                 transform=ax3.transAxes)

        # ---- Sections ----
        lines = [
            ("Graph", ""),
            (f"N = {N}", f"|E| = {m['n_edges']}"),
            (f"density = {m['density']:.3f}", f"W = {m['w_total']:.0f}"),

            ("", ""),
            ("Degree", ""),
            (f"{m['deg_mean']:.2f} ± {m['deg_std']:.2f}",
             f"[{m['deg_min']}, {m['deg_max']}]"),

            ("", ""),
            ("Structure", ""),
            (f"Bipartite: {'YES' if m['is_bipartite'] else 'NO'}",
             f"ratio = {m['bipart_ratio']:.3f}"),
            (f"Frustration = {m['frustration_index']:.3f}", ""),

            ("", ""),
            ("Spectrum", ""),
            (f"λ₂ = {m['fiedler']:.3f}",
             f"λ_max(L) = {m['lap_lambda_max']:.3f}"),

            ("", ""),
            ("Dynamics", ""),
            (f"μ_bin = {m['mu_bin_exact']:.3f}",
             f"#opt = {m['n_opt_configs']}"),
            (f"Best cut = {m['best_cut']:.1f}",
             f"{m['bipart_ratio']:.3f}"),
            (f"IC→opt = {m['n_opt_bin']}/{m['n_init']}", ""),
        ]

        # ---- Render text grid (clean paper look) ----
        y = 0.95
        dy = 0.055

        for left, right in lines:
            if left == "" and right == "":
                y -= dy * 0.5
                continue

            is_header = right == "" and not left.startswith("λ")

            if is_header:
                ax3.text(0.0, y, left,
                         fontsize=8.5, fontweight="bold",
                         color=hcol, ha="left", va="center",
                         transform=ax3.transAxes)
            else:
                ax3.text(0.0, y, left,
                         fontsize=8.2, color=BLACK,
                         ha="left", va="center",
                         transform=ax3.transAxes)

                if right:
                    ax3.text(0.55, y, right,
                             fontsize=8.2, color=BLACK,
                             ha="left", va="center",
                             transform=ax3.transAxes)

            y -= dy

        # subtle box around panel
        for spine in ax3.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(LIGHT)
            spine.set_linewidth(0.8)

    # ── Separator ───────────────────────────────────────────────────────────
    x_sep = (gs_top[0, 3].get_position(fig).x1 +
             gs_top[0, 4].get_position(fig).x0) / 2

    fig.add_artist(plt.Line2D([x_sep, x_sep], [0.03, 0.97],
                             transform=fig.transFigure,
                             linestyle="--", color=BLACK, alpha=0.4))

    # ── Title ───────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Figure 5 — Structural comparison (worst vs best)",
        fontsize=12, fontweight="bold")

    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Max-Cut mu_bin conjecture: does exact mu_bin achieve the global optimum?")
    parser.add_argument("--N",        type=int,   default=18)
    parser.add_argument("--n_graphs", type=int,   default=50)
    parser.add_argument("--p1",       type=float, default=0.5)
    parser.add_argument("--n_init",   type=int,   default=10)
    parser.add_argument("--t_end",    type=float, default=120.0)
    parser.add_argument("--n_points", type=int,   default=400)
    parser.add_argument("--rtol",     type=float, default=1e-6)
    parser.add_argument("--atol",     type=float, default=1e-6)
    parser.add_argument("--bin_tol",  type=float, default=0.05)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--save",     action="store_true")
    args = parser.parse_args()

    results = run_ensemble(args)
    print_summary(results, args)

    fig1 = make_figure1(results, args)   # histogram + CDF
    fig2 = make_figure2(results, args)   # spectral fingerprint
    fig3 = make_figure3(results, args)   # summary statistics
    fig5 = make_figure5(results, args)   # structural comparison: 4 worst vs best

    if args.save:
        tag = f"N{args.N}_p{args.p1:.2f}_ng{args.n_graphs}"
        for name, fig in [("histogram",  fig1),
                           ("spectral",   fig2),
                           ("summary",    fig3),
                           ("worst_traj", fig4),
                           ("structure",  fig5)]:
            for ext in ("pdf", "png"):
                fname = f"oim_conjecture_{name}_{tag}.{ext}"
                fig.savefig(fname, bbox_inches="tight", dpi=150)
                print(f"  Saved: {fname}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
