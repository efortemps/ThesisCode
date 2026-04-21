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
    3. ODE trajectories at mu_bin_exact
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
    oim.mu = mu_bin_exact
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
              title=(f"# ICs (out of {n_init}) reaching the global Max-Cut optimum\n"
                     f"at exact $\\mu_{{\\rm bin}}$ -- {n_graphs} ER graphs, $N={args.N}$"),
              xlabel=f"# ICs converging to global optimum (out of {n_init})",
              ylabel="# graphs")

    # ── Right: CDF ───────────────────────────────────────────────────────────
    ax     = axes[1]
    x_vals = np.arange(n_init + 1)
    cdf    = np.array([np.mean(n_opt >= k) for k in x_vals])
    ax.step(x_vals, cdf, where="post", color=C_BLUE, linewidth=2.2,
            label="P(# ICs >= k)")
    ax.fill_between(x_vals, cdf, step="post", alpha=0.15, color=C_BLUE)
    ax.axhline(1.0, color=GRAY, linewidth=0.9, linestyle="--")
    ax.axvline(1, color=C_GREEN, linewidth=1.5, linestyle=":",
               label=f"P(>=1 IC) = {cdf[1]:.2f}")
    ax.set_xticks(x_vals)
    ax.set_ylim(-0.02, 1.10)
    ax.set_xlim(-0.3, n_init + 0.3)
    ax.text(0.97, 0.05,
            f"P(at least 1 IC hits opt) = {cdf[1]*100:.1f}%\n"
            f"P(ALL {n_init} ICs hit opt)  = {cdf[n_init]*100:.1f}%\n"
            f"Mean # ICs = {np.mean(n_opt):.2f} / {n_init}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=WHITE,
                      edgecolor=GRAY, alpha=0.95))
    ax.legend(fontsize=9)
    _ax_style(ax,
              title=(f"Cumulative: P(# ICs >= k reaching global optimum)\n"
                     f"across {n_graphs} graphs at exact $\\mu_{{\\rm bin}}$"),
              xlabel="k (minimum # ICs that find the optimum)",
              ylabel="fraction of graphs")

    fig.suptitle(
        f"Conjecture Test: exact $\\mu_{{\\rm bin}}$ achieves global Max-Cut | "
        f"$N={args.N}$, $p_1={args.p1}$, {n_graphs} ER graphs | "
        f"$n_{{\\rm init}}={n_init}$,  $2^N={2**args.N}$ exact equilibria",
        color=BLACK, fontsize=11, fontweight="bold")
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
        f"for all $2^N$ equilibria | $N={args.N}$\n",
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

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor=WHITE)
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
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE,
                      edgecolor=GRAY, alpha=0.95))
    ax.legend(fontsize=8.5)
    _ax_style(ax,
              title=f"Distribution of $\\mu_{{\\rm bin}}$ over {n_graphs} graphs",
              xlabel="$\\mu_{\\rm bin}$ (exact, from $2^N$ enumeration)",
              ylabel="# graphs")

    # ── (b) Approximation ratio at mu_bin_exact ───────────────────────────────
    ax = axes[1]
    ax.hist(approx, bins=20, color=C_GREEN, alpha=0.72,
            edgecolor=BLACK, linewidth=0.5)
    ax.axvline(approx.mean(), color=C_AMBER, linewidth=2.0, linestyle="--",
               label=f"mean = {approx.mean():.4f}")
    ax.axvline(1.0, color=C_RED, linewidth=1.5, linestyle=":",
               label="global opt (= 1.0)")
    ax.text(0.03, 0.97,
            f"mean  = {approx.mean():.4f}\n"
            f"std   = {approx.std():.4f}\n"
            f"min   = {approx.min():.4f}\n"
            f"ratio=1: {int(np.sum(approx >= 1-1e-6))}/{n_graphs} graphs",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE,
                      edgecolor=GRAY, alpha=0.95))
    ax.legend(fontsize=8.5)
    _ax_style(ax,
              title=(f"Approximation ratio: best cut at $\\mu_{{\\rm bin}}$ / opt\n"
                     f"over {n_graphs} graphs"),
              xlabel="best cut (at $\\mu_{\\rm bin}$) / global opt",
              ylabel="# graphs")

    # ── (c) Success-fraction bar chart ────────────────────────────────────────
    ax = axes[2]
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
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.axhline(100, color=GRAY, linewidth=0.9, linestyle="--", alpha=0.7)
    _ax_style(ax,
              title=(f"How often does $\\mu_{{\\rm bin}}$ reach the optimum?\n"
                     f"Across {n_graphs} graphs, {n_init} ICs each"),
              xlabel="",
              ylabel="% of graphs")

    fig.suptitle(
        f"Summary Statistics | $N={args.N}$, $p_1={args.p1}$, "
        f"{n_graphs} ER graphs | $n_{{\\rm init}}={n_init}$",
        color=BLACK, fontsize=11, fontweight="bold")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Worst-graph trajectories + styled convergence table
# ═══════════════════════════════════════════════════════════════════════════════

# State-type colour palette (background tints for table rows)
_STATE_BG = {
    "M2-binary":     "#d4eac8",   # soft green
    "M1-half":       "#fde8c8",   # soft orange
    "M1-mixed":      "#fde8c8",
    "Type-III":      "#e8c8de",   # soft purple
    "not-converged": "#e8e8e8",   # grey
}
_STATE_BADGE = {          # stronger colour for the type-name cell itself
    "M2-binary":     "#55a868",
    "M1-half":       "#DD8452",
    "M1-mixed":      "#e377c2",
    "Type-III":      "#8172b2",
    "not-converged": "#b0b0b0",
}


def _classify_sol(sol, W, best_cut, per_eq, bin_tol=0.05):
    """
    Classify one ODE solution into the convergence categories used
    throughout the codebase, and find the nearest exact M2 equilibrium.

    Returns a dict with all fields needed for the table.
    """
    theta    = sol.y[:, -1]
    sigma    = np.sign(np.cos(theta)); sigma[sigma == 0] = 1.0
    bits     = tuple(0 if s > 0 else 1 for s in sigma)
    cut      = 0.25 * float(np.sum(W * (1.0 - sigma[:, None] * sigma[None, :])))
    H        = 0.5  * float(sigma @ W @ sigma)
    residual = float(np.max(np.abs(np.sin(theta))))

    # per-spin atom types
    def _atom(th):
        s, c = np.sin(th), np.cos(th)
        if abs(s) < bin_tol:
            return "zero" if c > 0 else "pi"
        if abs(abs(s) - 1.0) < bin_tol:
            return "half"
        return "other"

    atoms  = [_atom(th) for th in theta]
    n_zero = atoms.count("zero"); n_pi = atoms.count("pi")
    n_half = atoms.count("half"); n_oth = atoms.count("other")

    if n_zero + n_pi == len(theta):
        stype = "M2-binary"
    elif n_half == len(theta):
        stype = "M1-half"
    elif n_half > 0 and n_oth == 0:
        stype = "M1-mixed"
    elif n_oth > 0:
        stype = "Type-III"
    else:
        stype = "not-converged"

    is_opt = abs(cut - best_cut) < 1e-6

    # nearest M2 equilibrium (by wrapped L2 distance in phase space)
    nearest_bits, nearest_dist = None, np.inf
    phi_end = np.array([0.0 if b == 0 else np.pi for b in bits])
    for row in per_eq:
        diff = theta - row["phi"] if hasattr(row, "__contains__") and "phi" in row \
               else theta - np.array([0.0 if b == 0 else np.pi for b in row["bits"]])
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        d    = float(np.linalg.norm(diff))
        if d < nearest_dist:
            nearest_dist = d
            nearest_bits = row["bits"]

    return dict(
        bits=bits, cut=cut, H=H, residual=residual,
        is_binary=(stype == "M2-binary"), is_opt=is_opt,
        state_type=stype,
        nearest_bits=nearest_bits, nearest_dist=nearest_dist,
    )


def _draw_styled_table(ax, conv_list, best_cut, n_init, mu, mu_bin):
    """
    Render a two-section styled convergence table inside an axis-off axes.

    Section 1 — Per-trajectory rows
    Section 2 — Unique terminal states summary
    """
    ax.set_facecolor(WHITE)
    ax.axis("off")

    # ── colour helpers ────────────────────────────────────────────────────────
    def _bg(stype):   return _STATE_BG.get(stype, WHITE)
    def _badge(stype): return _STATE_BADGE.get(stype, GRAY)

    # ── Section 1: per-trajectory data ───────────────────────────────────────
    hdr1 = ["#", "Type", "Bits  (φ*)", "H", "Cut", "✓", "Res.",
            "Nearest M2", "Dist"]
    rows1 = []
    for i, c in enumerate(conv_list):
        bits_s   = "".join(str(b) for b in c["bits"])
        near_s   = "".join(str(b) for b in c["nearest_bits"]) \
                   if c["nearest_bits"] else "—"
        opt_mark = "★" if c["is_opt"] else ("✓" if c["is_binary"] else "✗")
        rows1.append([
            str(i),
            c["state_type"],
            bits_s,
            f"{c['H']:.2f}",
            f"{c['cut']:.2f}",
            opt_mark,
            f"{c['residual']:.4f}",
            near_s,
            f"{c['nearest_dist']:.3f}",
        ])

    # ── Section 2: summary ───────────────────────────────────────────────────
    from collections import Counter
    summary = {}
    for c in conv_list:
        key = (c["state_type"], c["bits"])
        if key not in summary:
            summary[key] = {"stype": c["state_type"], "bits": c["bits"],
                            "H": c["H"], "cut": c["cut"],
                            "count": 0, "residuals": [], "is_opt": c["is_opt"]}
        summary[key]["count"]     += 1
        summary[key]["residuals"].append(c["residual"])
    # sort by cut descending
    summ_vals = sorted(summary.values(), key=lambda x: -x["cut"])

    hdr2  = ["Type", "Bits", "H", "Cut", "n", "%", "Mean res.", "→ Opt?"]
    rows2 = []
    for s in summ_vals:
        bits_s = "".join(str(b) for b in s["bits"])
        rows2.append([
            s["stype"],
            bits_s,
            f"{s['H']:.2f}",
            f"{s['cut']:.2f}",
            str(s["count"]),
            f"{100*s['count']/n_init:.0f}%",
            f"{float(np.mean(s['residuals'])):.4f}",
            "★ YES" if s["is_opt"] else "no",
        ])

    # ── Geometry ─────────────────────────────────────────────────────────────
    # We place both tables in the axes using ax.table with bbox.
    # Total vertical space: [0, 1].  Allocate:
    #   title1  row height
    #   header1 row height
    #   n_traj  data rows
    #   gap
    #   title2  row height
    #   header2 row height
    #   n_summ  data rows
    #   footer (μ info)

    rh   = 0.048   # data-row height (axes fraction)
    hh   = 0.054   # header row
    th   = 0.050   # section title
    gap  = 0.040
    foot = 0.055
    n1   = len(rows1); n2 = len(rows2)

    total = 2*th + 2*hh + (n1 + n2)*rh + gap + foot
    # rescale so everything fits in [0.01, 0.99]
    if total > 0.98:
        s = 0.98 / total
        rh *= s; hh *= s; th *= s; gap *= s; foot *= s

    y = 0.99   # cursor, top-down

    # ── helper: draw one table section ───────────────────────────────────────
    def _section(y_cursor, title_txt, col_labels, data_rows,
                  type_col_idx, opt_col_idx=None):
        nc = len(col_labels); nr = len(data_rows)

        # Draw a full-width dark banner manually
        from matplotlib.patches import FancyBboxPatch
        banner_h = th * 0.9
        banner = FancyBboxPatch(
            (0.0, y_cursor - banner_h),
            1.0, banner_h,
            boxstyle="square,pad=0",
            transform=ax.transAxes,
            facecolor="#3a3a3a", edgecolor="none", alpha=0.88,
            clip_on=False, zorder=3)
        ax.add_patch(banner)
        ax.text(0.5, y_cursor - banner_h / 2,
                title_txt,
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=WHITE,
                zorder=5)

        y_cursor -= th
        table_height = hh + nr * rh
        bbox = [0.0, y_cursor - table_height, 1.0, table_height]

        tbl = ax.table(cellText=data_rows,
                       colLabels=col_labels,
                       bbox=bbox,
                       cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.8)

        # Style header row
        for j in range(nc):
            cell = tbl[0, j]
            cell.set_facecolor("#dce6f0")
            cell.set_text_props(fontweight="bold", color="#1a1a2e")
            cell.set_edgecolor("#9ab0cc")
            cell.set_linewidth(0.6)

        # Style data rows
        for i, row in enumerate(data_rows, start=1):
            stype    = row[type_col_idx]
            bg_col   = _bg(stype)
            bdg_col  = _badge(stype)
            is_opt_r = (opt_col_idx is not None and "★" in row[opt_col_idx])

            for j in range(nc):
                cell = tbl[i, j]
                # Alternate row shading: slightly darker on odd rows
                shade = bg_col if (i % 2 == 1) else _lighten(bg_col, 0.55)
                cell.set_facecolor(shade)
                cell.set_edgecolor("#c8c8c8")
                cell.set_linewidth(0.4)
                cell.get_text().set_color(BLACK)

            # Type badge column — bolder colour
            tbl[i, type_col_idx].set_facecolor(bdg_col)
            tbl[i, type_col_idx].get_text().set_color(WHITE)
            tbl[i, type_col_idx].get_text().set_fontweight("bold")
            tbl[i, type_col_idx].get_text().set_fontsize(7.0)

            # Optimum column special styling
            if opt_col_idx is not None:
                cell = tbl[i, opt_col_idx]
                if "★" in row[opt_col_idx]:
                    cell.set_facecolor("#55a868")
                    cell.get_text().set_color(WHITE)
                    cell.get_text().set_fontweight("bold")
                elif "✓" in row[opt_col_idx]:
                    cell.set_facecolor("#9dcf9a")
                elif "✗" in row[opt_col_idx]:
                    cell.set_facecolor("#f7b89a")

        return y_cursor - table_height

    # ── Draw Section 1 ────────────────────────────────────────────────────────
    y = _section(y, "Per-trajectory convergence",
                 hdr1, rows1,
                 type_col_idx=1, opt_col_idx=5)

    y -= gap

    # ── Draw Section 2 ────────────────────────────────────────────────────────
    y = _section(y, "Summary — unique terminal states",
                 hdr2, rows2,
                 type_col_idx=0, opt_col_idx=7)

    y -= gap * 0.5

    # ── Footer: μ status line ─────────────────────────────────────────────────
    diff    = mu - mu_bin
    above   = diff > 0
    status  = f"above  (binarises ✓)" if above else f"below  (may not binarise ✗)"
    status_col = C_GREEN if above else C_ORANGE
    footer_txt = (f"$\\mu = {mu:.4f}$   |   "
                  f"$\\mu_{{\\rm bin}} = {mu_bin:.4f}$   |   "
                  f"$\\mu - \\mu_{{\\rm bin}} = {diff:+.4f}$   "
                  f"({status})")
    ax.text(0.5, max(y - 0.01, 0.01),
            footer_txt,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=8, color=status_col,
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="#f5f5f5",
                      edgecolor=status_col, linewidth=0.9, alpha=0.95))

    # ── Type legend at very bottom ────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=_badge(k), edgecolor=GRAY,
                       label=k, alpha=0.88)
        for k in _STATE_BADGE
    ]
    ax.legend(handles=legend_handles,
              loc="lower center",
              bbox_to_anchor=(0.5, 0.0),
              ncol=3, fontsize=7.5,
              facecolor=WHITE, edgecolor=GRAY,
              framealpha=0.92, labelcolor=BLACK)


def _lighten(hex_col, factor=0.6):
    """Blend a hex colour toward white by `factor` (0=original, 1=white)."""
    hex_col = hex_col.lstrip("#")
    r, g, b = (int(hex_col[i:i+2], 16) for i in (0, 2, 4))
    r2 = int(r + (255 - r) * factor)
    g2 = int(g + (255 - g) * factor)
    b2 = int(b + (255 - b) * factor)
    return f"#{r2:02x}{g2:02x}{b2:02x}"


def make_figure4(results, args):
    """
    Figure 4 — Worst-performing graph analysis.

    Left panel  : Phase trajectories θ_i(t) for all n_init ICs at μ_bin,
                  identical style to the mu-slider experiment.
    Right panel : Styled two-section convergence table:
                  Section 1 — one row per trajectory (state, bits, H, cut, ★)
                  Section 2 — unique terminal states summary

    The "worst" graph is the one with the lowest n_opt_bin (fewest ICs
    reaching the global optimum).  Ties are broken by the smallest n_opt_bin
    / most frustrated graph (highest mu_bin_exact).
    """
    # ── Pick worst graph ──────────────────────────────────────────────────────
    n_opt_arr = np.array([r["n_opt_bin"] for r in results])
    mu_arr    = np.array([r["mu_bin_exact"] for r in results])
    # primary sort: n_opt ascending; secondary: mu_bin descending (harder)
    worst_idx = int(
        sorted(range(len(results)),
               key=lambda i: (n_opt_arr[i], -mu_arr[i]))[0]
    )
    r = results[worst_idx]

    # ── Re-simulate at mu_bin_exact (phi0s were stored) ──────────────────────
    W       = r["W"]
    N       = r["N"]
    mu_b    = r["mu_bin_exact"]
    phi0s   = r["phi0s"]
    per_eq  = r["per_eq"]

    oim  = OIMMaxCut(W, mu=mu_b, seed=args.seed)
    sols = oim.simulate_many(phi0s,
                              t_span=(0., args.t_end),
                              n_points=args.n_points,
                              rtol=args.rtol, atol=args.atol)

    # Classify every trajectory
    conv_list = [_classify_sol(s, W, r["best_cut"], per_eq, args.bin_tol)
                 for s in sols]

    n_opt_here = sum(1 for c in conv_list if c["is_opt"])
    n_bin_here = sum(1 for c in conv_list if c["is_binary"])

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(24, max(9, 2.2 + 0.42 * args.n_init)),
                     facecolor=WHITE)
    gs  = gridspec.GridSpec(1, 2, figure=fig,
                       width_ratios=[1.55, 1.0],
                       wspace=0.06,
                       left=0.04, right=0.99, top=0.88, bottom=0.09)
    ax_phase = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[0, 1])

    ax_phase.set_facecolor(WHITE)
    ax_table.set_facecolor(WHITE)
    for sp in ax_phase.spines.values():
        sp.set_edgecolor(BLACK); sp.set_linewidth(0.8)
    ax_phase.grid(True, color=LIGHT, linewidth=0.5, zorder=0)

    # ── Left: phase trajectories ──────────────────────────────────────────────
    SPIN_COLS = plt.get_cmap("tab20")(np.linspace(0, 1, max(N, 2)))
    PI_TICKS  = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    PI_LABELS = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]

    t = sols[0].t
    for sol in sols:
        for spin in range(N):
            ax_phase.plot(t, sol.y[spin],
                          color=SPIN_COLS[spin % 20],
                          alpha=0.42, linewidth=0.95, zorder=2)

    for yref, lw_r in [(np.pi, 1.1), (np.pi / 2, 0.65),
                        (0.0, 1.4), (-np.pi / 2, 0.65), (-np.pi, 0.9)]:
        ax_phase.axhline(yref, color=GRAY, linestyle="--",
                         linewidth=lw_r, alpha=0.75, zorder=1)
        if abs(abs(yref) - np.pi / 2) < 1e-9:
            lbl = r"$\pi/2$" if yref > 0 else r"$-\pi/2$"
            ax_phase.text(t[-1] * 0.996, yref + 0.10, lbl,
                          ha="right", va="bottom", fontsize=7.5, color=GRAY)

    ax_phase.set_yticks(PI_TICKS)
    ax_phase.set_yticklabels(PI_LABELS, fontsize=10, color=BLACK)
    ax_phase.set_ylim(-4.2, 4.2)
    ax_phase.set_xlim(t[0], t[-1])
    ax_phase.tick_params(colors=BLACK, labelsize=9)

    # Status badge (top-right)
    is_all_bin = (n_bin_here == args.n_init)
    status_txt = ("BINARISED ✓" if is_all_bin
                  else f"NOT YET BINARISED ✗   "
                       f"M2:{n_bin_here}  opt:{n_opt_here}")
    ax_phase.text(0.98, 0.97, status_txt,
                  transform=ax_phase.transAxes, ha="right", va="top",
                  fontsize=9, fontweight="bold",
                  color=C_GREEN if is_all_bin else C_ORANGE,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE,
                            edgecolor=GRAY, alpha=0.95))

    # Info badge (top-left)
    n_edges = int(np.sum(W)) // 2
    ax_phase.text(0.01, 0.97,
                  f"$\\mu = {mu_b:.4f}$  |  "
                  f"$\\mu_{{\\rm bin}} = {mu_b:.4f}$  |  "
                  f"$W_{{\\rm tot}} = {r['w_total']:.0f}$  |  "
                  f"best cut $= {r['best_cut']:.0f}$",
                  transform=ax_phase.transAxes, ha="left", va="top",
                  fontsize=9.5,
                  bbox=dict(boxstyle="round,pad=0.28", facecolor=WHITE,
                            edgecolor=GRAY, alpha=0.93))

    # Spin legend
    spin_patches = [mpatches.Patch(color=SPIN_COLS[s % 20], label=f"spin {s}")
                    for s in range(N)]
    ax_phase.legend(handles=spin_patches, loc="lower right", fontsize=7.5,
                    ncol=max(1, N // 5), framealpha=0.90)

    ax_phase.set_xlabel("time  $t$", color=BLACK, fontsize=11)
    ax_phase.set_ylabel("phase  $\\theta_i(t)$  (rad)", color=BLACK, fontsize=11)
    ax_phase.set_title(
        f"Phase dynamics — worst-performing graph  (graph {worst_idx+1} / {len(results)})  |  "
        f"$\\mu = {mu_b:.4f}$  |  "
        f"$N={N}$,  $|E|={n_edges}$  |  "
        f"opt reached: {n_opt_here}/{args.n_init} ICs",
        color=BLACK, fontsize=10, fontweight="bold", pad=5)

    # ── Right: styled convergence table ───────────────────────────────────────
    ax_table.axis("off")
    _draw_styled_table(ax_table, conv_list, r["best_cut"],
                       args.n_init, mu_b, mu_b)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Figure 4 — Worst-performing graph analysis  |  "
        f"Graph {worst_idx+1} / {len(results)}  |  "
        f"$N={N}$,  $p_1={args.p1}$  |  "
        f"$\\mu_{{\\rm bin}} = {mu_b:.4f}$  |  "
        f"{n_opt_here} / {args.n_init} ICs reach global optimum "
        f"(cut $= {r['best_cut']:.0f}$)",
        color=BLACK, fontsize=11, fontweight="bold")

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
    fig4 = make_figure4(results, args)   # worst-graph trajectories + table

    if args.save:
        tag = f"N{args.N}_p{args.p1:.2f}_ng{args.n_graphs}"
        for name, fig in [("histogram",  fig1),
                           ("spectral",   fig2),
                           ("summary",    fig3),
                           ("worst_traj", fig4)]:
            for ext in ("pdf", "png"):
                fname = f"oim_conjecture_{name}_{tag}.{ext}"
                fig.savefig(fname, bbox_inches="tight", dpi=150)
                print(f"  Saved: {fname}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
