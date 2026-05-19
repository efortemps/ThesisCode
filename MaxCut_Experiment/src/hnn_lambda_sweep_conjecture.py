#!/usr/bin/env python3
"""
hnn_lambda_sweep_conjecture.py
──────────────────────────────────────────────────────────────────────────────
Hopfield-Network Max-Cut experiment: λ sweep over random ER graphs.
(λ = 1/u₀ is the gain parameter, thesis convention.)

Unlike the OIM, the Hopfield network has no single "optimal" gain parameter
λ analogous to μ_bin. This experiment therefore sweeps a fixed set of λ
values and asks, for each graph, which λ (if any) most reliably finds the
global Max-Cut optimum.

Setting
───────
• N = 20 nodes — small enough to find the true global optimum by brute-force
  over all 2^20 ≈ 1 M spin configurations (vectorised, chunked for memory).
• n_graphs random Erdős–Rényi graphs (antiferromagnetic: J = −W, W ≥ 0).
• lam_values : fixed sweep grid [0.01, 0.1, 1.0, 2.0, 5.0] (5 values)
• n_init initial conditions per (graph, λ), drawn from uniform(−1, 1).
• Same ICs used for every λ on a given graph — isolates gain effect.

HNN ODE
───────
du_i/dt = −u_i − Σ_j W_ij tanh(λ u_j)    (τ = 1,  λ = 1/u₀)

Interpretation of λ:
Large λ → steep tanh (near hard threshold) → fast, aggressive binarisation
Small λ → shallow tanh (near linear)       → slow, may not binarise

Stability of the origin s = 0:
Hessian at origin: H(0) = W + (1/λ) I
λ_min(H(0)) < 0  iff  1/λ < |λ_min(W)|,  i.e.  λ > 1/|λ_min(W)| =: λ_bin
→ origin is UNSTABLE iff λ > λ_bin  (GOOD: system leaves and binarises)
→ origin is STABLE   iff λ < λ_bin  (BAD:  system may stagnate near s=0)

Figures produced
────────────────
Figure 1 — Per-λ histograms + CDF
  Row 0: one histogram per λ value showing distribution of n_opt
         (# ICs out of n_init reaching the global optimum)

Figure 3 — Sweep summary
  (a) Approximation ratio box-plots: (best cut found) / (true opt) per λ
  (b) Success-fraction bar chart: % graphs where ≥1 IC hits opt, per λ
  (c) Mean n_opt ± std per λ (line plot showing the sweep profile)

Figure 5 — Structural comparison: 4 worst vs 1 best graph
  Worst = lowest max_{λ} n_opt (even the best λ rarely finds the opt)
  Best  = highest max_{λ} n_opt

Usage
─────
python hnn_lambda_sweep_conjecture.py
    [--N 20] [--n_graphs 20] [--p1 0.5]
    [--lam_values 100 10 1.0 0.5 0.2]
    [--n_init 10] [--t_end 80] [--n_points 400]
    [--rtol 1e-6] [--atol 1e-8]
    [--bin_tol 0.05] [--seed 42] [--save]
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from scipy.integrate import solve_ivp

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":   16,
    "axes.titlesize":   20,
    "axes.labelsize":   18,
    "xtick.labelsize":   15,
    "ytick.labelsize":   15,
    "legend.fontsize":   14,
    "figure.titlesize":  22,
    "axes.edgecolor":    "black",
    "axes.linewidth":    0.8,
    "xtick.color":       "black",
    "ytick.color":       "black",
    "text.color":        "black",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "legend.framealpha": 0.92,
    "legend.edgecolor":  "#b0b0b0",
    "legend.facecolor":  "white",
    "legend.labelcolor": "black",
})

WHITE   = "#ffffff"
BLACK   = "#000000"
GRAY    = "#b0b0b0"
LIGHT   = "#e6e6e6"
C_BLUE  = "#4C72B0"
C_ORANGE= "#DD8452"
C_GREEN = "#55a868"
C_RED   = "#c44e52"
C_AMBER = "#ffb74d"
C_PURPLE= "#8172b2"

# ── State-type colour palettes ────────────────────────────────────────────────
_STATE_BG = {
    "M2-binary":    "#d4eac8",
    "M1-mixed":     "#fde8c8",
    "Type-III":     "#e8c8de",
    "not-converged":"#e8e8e8",
}
_STATE_BADGE = {
    "M2-binary":    "#55a868",
    "M1-mixed":     "#e377c2",
    "Type-III":     "#8172b2",
    "not-converged":"#b0b0b0",
}

# Colour per λ value (up to 8 values)
_LAM_COLOURS = [C_RED, C_ORANGE, C_AMBER, C_GREEN, C_BLUE,
                C_PURPLE, "#17becf", "#bcbd22"]


def _ax_style(ax, title="", xlabel="", ylabel="", titlesize=20):
    ax.set_facecolor(WHITE)
    ax.tick_params(colors=BLACK, labelsize=15)
    for sp in ax.spines.values():
        sp.set_edgecolor(BLACK); sp.set_linewidth(0.8)
    ax.grid(True, color=LIGHT, linewidth=0.5, zorder=0)
    if title:   ax.set_title(title,   color=BLACK, fontsize=titlesize,
                             fontweight="bold", pad=5)
    if xlabel:  ax.set_xlabel(xlabel, color=BLACK, fontsize=18)
    if ylabel:  ax.set_ylabel(ylabel, color=BLACK, fontsize=18)


def _lighten(hex_col, factor=0.6):
    hex_col = hex_col.lstrip("#")
    r, g, b = (int(hex_col[i:i+2], 16) for i in (0, 2, 4))
    return (f"#{int(r+(255-r)*factor):02x}"
            f"{int(g+(255-g)*factor):02x}"
            f"{int(b+(255-b)*factor):02x}")


# ═══════════════════════════════════════════════════════════════════════════════
# Graph generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_er_graph(N: int, p1: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < p1:
                W[i, j] = W[j, i] = 1.0
    return W


# ═══════════════════════════════════════════════════════════════════════════════
# True global optimum (brute-force, vectorised, chunked for N=20)
# ═══════════════════════════════════════════════════════════════════════════════

def find_global_optimum(W: np.ndarray, chunk: int = 1 << 16) -> float:
    """
    Find the true Max-Cut value by evaluating all 2^N spin configurations.

    Vectorised and chunked to keep peak memory under ~100 MB even for N=20.
    For N=20: 2^20 = 1 048 576 configs, processed in chunks of 2^16 = 65 536.

    cut(s) = 0.25 * (Σ W_ij − sᵀWs)
    """
    n       = W.shape[0]
    n_total = 1 << n
    W32     = W.astype(np.float32)
    W_sum   = float(np.sum(W))
    best    = 0.0

    shifts = np.arange(n, dtype=np.int32)
    for start in range(0, n_total, chunk):
        end  = min(start + chunk, n_total)
        idx  = np.arange(start, end, dtype=np.int32)
        bits = ((idx[:, None] >> shifts[None, :]) & 1).astype(np.float32)
        s    = 2.0 * bits - 1.0                           # shape (chunk, N), ±1
        quad = (s @ W32 * s).sum(axis=1)                  # shape (chunk,)
        best = max(best, float((0.25 * (W_sum - quad)).max()))

    return best


# ═══════════════════════════════════════════════════════════════════════════════
# HNN ODE and convergence
# ═══════════════════════════════════════════════════════════════════════════════

def _hnn_rhs(t, u, W, lam):
    """τ=1 Hopfield ODE: du/dt = −u − W tanh(λ u)."""
    return -u - W @ np.tanh(lam * u)


def simulate_hnn(W: np.ndarray, lam: float, u_init: np.ndarray,
                 t_end: float, n_points: int,
                 rtol: float, atol: float):
    t_eval = np.linspace(0.0, t_end, n_points)
    return solve_ivp(
        _hnn_rhs, (0.0, t_end), u_init,
        args=(W, lam),
        method="RK45", t_eval=t_eval,
        rtol=rtol, atol=atol, dense_output=False,
    )


def classify_hnn_sol(sol, W, lam, best_cut, bin_tol=0.05):
    """
    Classify the terminal state of one HNN trajectory.

    State variables: u (membrane potential), activation s = tanh(λ u).

    Terminal state types
    ────────────────────
    M2-binary   : all |s_i| ≥ 1 − bin_tol (converged to ±1 corners)
    M1-mixed    : some |s_i| ≈ 1, some |s_i| ≈ 0, none intermediate
    Type-III    : at least one s_i at a continuous intermediate value
    not-converged: all |s_i| < bin_tol (stuck near origin)
    """
    s_final = np.tanh(lam * sol.y[:, -1])
    sigma   = np.sign(s_final); sigma[sigma == 0] = 1.0
    bits    = tuple(1 if s > 0 else 0 for s in sigma)

    cut      = 0.25 * float(np.sum(W * (1.0 - sigma[:, None] * sigma[None, :])))
    H        = 0.5  * float(sigma @ W @ sigma)
    residual = float(np.max(np.abs(np.abs(s_final) - 1.0)))

    n_near1 = sum(1 for s in s_final if abs(abs(s) - 1.0) < bin_tol)
    n_near0 = sum(1 for s in s_final if abs(s) < bin_tol)
    n_inter = len(s_final) - n_near1 - n_near0

    if np.max(np.abs(s_final)) < 1e-3:
        stype = "not-converged"
    elif n_near1 == len(s_final):
        stype = "M2-binary"
    elif n_inter == 0:
        stype = "M1-mixed"
    else:
        stype = "Type-III"

    return dict(
        bits=bits, s_final=s_final, cut=cut, H=H, residual=residual,
        is_binary=(stype == "M2-binary"),
        is_opt=(stype == "M2-binary" and abs(cut - best_cut) < 1e-6),
        state_type=stype,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Per-graph experiment
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_graph(W: np.ndarray, g_idx: int, args,
                  lam_values: list) -> dict:
    """
    Run the full HNN experiment for one graph across all λ values.

    Steps
    ─────
    1. Find the true global optimum cut (brute force over 2^N configs).
    2. Sample n_init initial conditions u_init ~ uniform(−1, 1)^N.
       The SAME u_inits are used for every λ value.
    3. For each λ:
       a. Simulate n_init ODE trajectories.
       b. Classify each terminal state.
       c. Count n_opt (# trajectories reaching the global optimum cut).
    4. Identify best_lam = argmax_{λ} n_opt.
    """
    N       = W.shape[0]
    w_total = float(np.sum(np.triu(W)))

    # Step 1: true global optimum
    t0     = time.perf_counter()
    best_cut = find_global_optimum(W)
    t_enum = time.perf_counter() - t0

    # Step 2: fixed ICs
    rng     = np.random.default_rng(args.seed + g_idx * 1000)
    u_inits = [rng.uniform(-1.0, 1.0, N) for _ in range(args.n_init)]

    # Step 3: sweep λ values
    per_lam = {}
    for lam in lam_values:
        sols = [simulate_hnn(W, lam, u_init, args.t_end, args.n_points,
                             args.rtol, args.atol)
                for u_init in u_inits]
        conv = [classify_hnn_sol(s, W, lam, best_cut, args.bin_tol)
                for s in sols]
        cuts_lam  = np.array([c["cut"] for c in conv])
        bin_flags = np.array([c["is_binary"] for c in conv])
        n_opt     = int(np.sum([c["is_opt"] for c in conv]))

        per_lam[lam] = dict(
            conv=conv, cuts=cuts_lam.tolist(),
            bin_fraction=float(bin_flags.mean()),
            n_opt=n_opt,
            sols=sols,
        )

    # Step 4: summary across λ
    n_opt_per_lam = {lam: per_lam[lam]["n_opt"] for lam in lam_values}
    best_lam      = max(lam_values, key=lambda l: n_opt_per_lam[l])
    best_n_opt    = n_opt_per_lam[best_lam]

    all_cuts_flat = [c for lam in lam_values for c in per_lam[lam]["cuts"]]
    best_found    = float(max(all_cuts_flat))
    approx_ratio  = best_found / max(best_cut, 1e-9)

    return dict(
        g_idx=g_idx, W=W, N=N,
        best_cut=best_cut, w_total=w_total,
        lam_values=lam_values,
        per_lam=per_lam,
        n_opt_per_lam=n_opt_per_lam,
        best_lam=best_lam,
        best_n_opt=best_n_opt,
        approx_ratio=approx_ratio,
        best_found=best_found,
        u_inits=u_inits,
        t_enum=t_enum,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Run full ensemble
# ═══════════════════════════════════════════════════════════════════════════════

def run_ensemble(args, lam_values: list) -> list:
    print(f"\n{'='*70}")
    print(f"  HNN λ-Sweep Max-Cut Experiment")
    print(f"  N={args.N}  p1={args.p1}  n_graphs={args.n_graphs}")
    print(f"  λ values: {lam_values}")
    print(f"  n_init={args.n_init}  t_end={args.t_end}")
    print(f"  True global opt by 2^{args.N}={2**args.N} brute-force enumeration")
    print(f"{'='*70}")

    lam_hdr = "  ".join(f"λ={v:.4g}→n_opt" for v in lam_values)
    print(f"  {'g':>3}  {'|E|':>4}  {'opt':>5}  {lam_hdr}  best_λ")
    print("  " + "─" * 65)

    results = []
    t_total = time.perf_counter()

    for g_idx in range(args.n_graphs):
        W = generate_er_graph(args.N, args.p1, seed=args.seed + g_idx)
        r = run_one_graph(W, g_idx, args, lam_values)
        results.append(r)
        n_edges  = int(np.sum(W)) // 2
        n_opts   = "  ".join(f"{r['n_opt_per_lam'][lam]:>12}" for lam in lam_values)
        print(f"  {g_idx+1:>3}  {n_edges:>4}  {r['best_cut']:>5.1f}  "
              f"{n_opts}  λ={r['best_lam']:.4g}")

    elapsed = time.perf_counter() - t_total
    print(f"\n  Total: {elapsed:.1f}s ({elapsed/args.n_graphs:.1f}s/graph)\n")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Console summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(results: list, lam_values: list, args):
    n_init   = args.n_init
    n_graphs = len(results)

    print(f"\n{'='*70}")
    print(f"  SUMMARY | N={args.N}  p1={args.p1}  {n_graphs} graphs  n_init={n_init}")
    print(f"{'='*70}")

    for lam in lam_values:
        n_opt    = np.array([r["n_opt_per_lam"][lam] for r in results])
        frac_any = float(np.mean(n_opt >= 1))
        frac_all = float(np.mean(n_opt == n_init))
        print(f"\n  λ = {lam:.4g}  (u₀ = {1.0/lam:.4g})")
        print(f"    >=1 IC hits opt : {frac_any*100:.1f}%  |  "
              f"all {n_init} ICs : {frac_all*100:.1f}%  |  "
              f"mean n_opt = {n_opt.mean():.2f}")

    print(f"\n  Best λ distribution across graphs:")
    best_lams = [r["best_lam"] for r in results]
    for lam in lam_values:
        cnt = sum(1 for v in best_lams if v == lam)
        print(f"    λ={lam:.4g} is best for {cnt}/{n_graphs} graphs")

    approx = np.array([r["approx_ratio"] for r in results])
    print(f"\n  Approx ratio (best found / true opt):")
    print(f"    mean={approx.mean():.4f}  std={approx.std():.4f}  "
          f"min={approx.min():.4f}  "
          f"=1.0: {int(np.sum(approx >= 1-1e-6))}/{n_graphs}")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Per-λ histograms
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure1(results: list, lam_values: list, args) -> plt.Figure:
    """1 row × n_λ columns — per-λ histogram of n_opt."""
    n_init = args.n_init
    n_lam  = len(lam_values)

    fig = plt.figure(figsize=(4.2 * n_lam, 8), facecolor=WHITE)
    gs  = gridspec.GridSpec(1, n_lam, figure=fig,
                            wspace=0.38,
                            left=0.05, right=0.98, top=0.88, bottom=0.14)

    bins       = np.arange(-0.5, n_init + 1.5, 1.0)
    x_vals     = np.arange(n_init + 1)
    tick_step  = max(1, n_init // 5)
    sparse_ticks = x_vals[::tick_step]

    for col, (lam, col_c) in enumerate(zip(lam_values, _LAM_COLOURS)):
        n_opt  = np.array([r["n_opt_per_lam"][lam] for r in results])
        counts, _ = np.histogram(n_opt, bins=bins)

        ax0 = fig.add_subplot(gs[0, col])
        bar_colours = [C_RED if k == 0 else
                       (C_GREEN if k == n_init else col_c)
                       for k in range(n_init + 1)]
        bars = ax0.bar(x_vals, counts, color=bar_colours,
                       edgecolor=BLACK, linewidth=0.5, alpha=0.80, zorder=3)
        for k, (bar, cnt) in enumerate(zip(bars, counts)):
            if cnt > 0:
                ax0.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.15,
                         f"{cnt}", ha="center", va="bottom",
                         fontsize=8, color=BLACK)

        ax0.axvline(n_opt.mean(), color=C_AMBER, linewidth=1.8,
                    linestyle="--", zorder=5,
                    label=f"mean={n_opt.mean():.2f}")
        ax0.set_xticks(sparse_ticks)
        ax0.set_xticklabels([str(t) for t in sparse_ticks],
                            fontsize=9, rotation=45, ha="right")
        ax0.set_xlim(-0.7, n_init + 0.7)
        ax0.set_ylim(0, max(counts.max(), 1) * 1.30)
        ax0.legend(fontsize=15, loc="upper left")
        _ax_style(ax0,
                  title=rf"$\lambda = {lam}$",
                  xlabel=rf"# ICs $\to$ opt (/{n_init})",
                  ylabel="# graphs")

    fig.suptitle(
        r"HNN $\lambda$ Sweep — Histogram Summary",
        color=BLACK, fontsize=22, fontweight="bold")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Sweep summary
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure3(results: list, lam_values: list, args) -> plt.Figure:
    n_init   = args.n_init
    n_graphs = len(results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor=WHITE)
    fig.subplots_adjust(wspace=0.38, left=0.08, right=0.97,
                        top=0.87, bottom=0.13)

    x_pos = np.arange(len(lam_values))

    # ── (a) Success-fraction grouped bar chart ────────────────────────────
    ax = axes[0]
    frac_none = [float(np.mean(
        [r["n_opt_per_lam"][lam] == 0      for r in results])) * 100
                 for lam in lam_values]
    frac_any  = [float(np.mean(
        [r["n_opt_per_lam"][lam] >= 1      for r in results])) * 100
                 for lam in lam_values]
    frac_all  = [float(np.mean(
        [r["n_opt_per_lam"][lam] == n_init for r in results])) * 100
                 for lam in lam_values]

    w = 0.25
    ax.bar(x_pos - w, frac_none, width=w, color=C_RED,   alpha=0.75,
           edgecolor=BLACK, linewidth=0.5, label="0 ICs hit opt")
    ax.bar(x_pos,     frac_any,  width=w, color=C_BLUE,  alpha=0.75,
           edgecolor=BLACK, linewidth=0.5, label=r"$\geq$1 IC hits opt")
    ax.bar(x_pos + w, frac_all,  width=w, color=C_GREEN, alpha=0.75,
           edgecolor=BLACK, linewidth=0.5, label=f"all {n_init} ICs hit opt")

    for pos, f_none, f_any, f_all in zip(x_pos, frac_none, frac_any, frac_all):
        for offset, val in [(-w, f_none), (0, f_any), (w, f_all)]:
            if val > 2:
                ax.text(pos + offset, val + 1, f"{val:.0f}",
                        ha="center", va="bottom", fontsize=8,
                        color=BLACK, fontweight="bold")

    ax.set_ylim(0, 115)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([rf"$\lambda={v}$" for v in lam_values],
                       fontsize=9, rotation=20)
    ax.axhline(100, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.legend(fontsize=15, loc= "upper right")
    _ax_style(ax,
              title=rf"Success fractions per $\lambda$"
                    f"\nover {n_graphs} graphs",
              xlabel=r"$\lambda$",
              ylabel="% of graphs")

    # ── (b) Mean n_opt ± std vs λ (sweep profile) ────────────────────────
    ax = axes[1]
    means = np.array([np.mean([r["n_opt_per_lam"][lam] for r in results])
                      for lam in lam_values])
    stds  = np.array([np.std ([r["n_opt_per_lam"][lam] for r in results])
                      for lam in lam_values])

    ax.fill_between(x_pos, means - stds, means + stds,
                    alpha=0.18, color=C_BLUE)
    ax.plot(x_pos, means, color=C_BLUE, linewidth=2.2,
            marker="o", markersize=7, zorder=3, label=r"mean $\pm$ std")
    ax.errorbar(x_pos, means, yerr=stds, fmt="none",
                ecolor=C_BLUE, elinewidth=1.2, capsize=4, capthick=1.2)

    best_col = int(np.argmax(means))
    ax.scatter([x_pos[best_col]], [means[best_col]],
               color=C_GREEN, s=100, zorder=5, edgecolors=BLACK,
               linewidths=0.7,
               label=rf"best $\lambda={lam_values[best_col]}$")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([rf"$\lambda={v}$" for v in lam_values],
                       fontsize=9, rotation=20)
    ax.set_ylim(-0.3, n_init + 0.5)
    ax.axhline(n_init, color=GRAY, linewidth=0.8, linestyle="--",
               label=f"n_init = {n_init}")
    ax.legend(fontsize=15, loc="upper right")
    _ax_style(ax,
              title=r"Mean # ICs $\to$ global opt",
              xlabel=r"$\lambda$",
              ylabel=rf"mean # ICs $\to$ global opt (/{n_init})")

    fig.suptitle(
        r"HNN $\lambda$ Sweep Summary",
        color=BLACK, fontsize=22, fontweight="bold")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Graph structural metrics
# ═══════════════════════════════════════════════════════════════════════════════

def graph_structural_metrics(r: dict) -> dict:
    W        = r["W"]
    N        = r["N"]
    best_cut = r["best_cut"]
    w_total  = r["w_total"]

    n_edges = int(np.sum(W)) // 2
    density = n_edges / max(N * (N - 1) // 2, 1)
    degrees = W.sum(axis=1)

    L        = np.diag(degrees) - W
    lap_eigs = np.sort(np.linalg.eigvalsh(L))
    fiedler  = float(lap_eigs[1]) if N > 1 else 0.0
    lap_gap  = float(lap_eigs[1] - lap_eigs[0]) if N > 1 else 0.0
    lap_lmax = float(lap_eigs[-1])

    adj_eigs  = np.sort(np.linalg.eigvalsh(W))[::-1]
    adj_lmax  = float(adj_eigs[0])
    lam_min_W = float(np.linalg.eigvalsh(W)[0])
    u0_bin    = float(abs(lam_min_W))             # = 1/λ_bin
    lam_bin   = 1.0 / u0_bin if u0_bin > 0 else float("inf")

    # Bipartiteness (BFS)
    colour = -np.ones(N, dtype=int); colour[0] = 0
    queue  = [0]; is_bip = True
    while queue and is_bip:
        v = queue.pop(0)
        for u in range(N):
            if W[v, u] > 0:
                if colour[u] == -1:
                    colour[u] = 1 - colour[v]; queue.append(u)
                elif colour[u] == colour[v]:
                    is_bip = False; break
    bipart_ratio = best_cut / max(w_total, 1e-9)
    frustration  = 1.0 - bipart_ratio

    W3          = W @ W @ W
    n_triangles = int(round(np.trace(W3) / 6.0))
    clust = []
    for i in range(N):
        di = int(degrees[i])
        if di < 2: clust.append(0.0); continue
        nbrs   = np.where(W[i] > 0)[0]
        e_nbrs = sum(1 for a in nbrs for b in nbrs if a < b and W[a, b] > 0)
        clust.append(2.0 * e_nbrs / (di * (di - 1)))
    mean_clust = float(np.mean(clust))

    return dict(
        n_edges=n_edges, density=density, w_total=w_total,
        degrees=degrees,
        deg_mean=float(degrees.mean()), deg_std=float(degrees.std()),
        deg_min=int(degrees.min()), deg_max=int(degrees.max()),
        lap_eigenvalues=lap_eigs, fiedler=fiedler,
        lap_spectral_gap=lap_gap, lap_lambda_max=lap_lmax,
        adj_eigenvalues=adj_eigs, adj_lambda_max=adj_lmax,
        lam_min_W=lam_min_W, u0_bin=u0_bin, lam_bin=lam_bin,
        is_bipartite=is_bip, bipart_ratio=bipart_ratio,
        frustration_index=frustration,
        n_triangles=n_triangles, mean_clustering=mean_clust,
        # HNN-specific
        best_cut=best_cut,
        best_n_opt=r["best_n_opt"],
        best_lam=r["best_lam"],
        n_opt_per_lam=r["n_opt_per_lam"],
        n_init=len(r["u_inits"]),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Structural comparison: 4 worst vs 1 best
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure5(results: list, lam_values: list, args) -> plt.Figure:
    n_init   = args.n_init
    n_graphs = len(results)

    n_opt_best_arr = np.array([r["best_n_opt"] for r in results])
    lam_bin_arr    = np.array([
        1.0 / abs(float(np.linalg.eigvalsh(r["W"])[0]))
        if abs(float(np.linalg.eigvalsh(r["W"])[0])) > 0 else 1e9
        for r in results
    ])

    worst_idxs = sorted(range(n_graphs),
                        key=lambda i: (n_opt_best_arr[i], -lam_bin_arr[i]))[:4]
    best_idx   = sorted(range(n_graphs),
                        key=lambda i: (-n_opt_best_arr[i], lam_bin_arr[i]))[0]

    col_idxs  = worst_idxs + [best_idx]
    col_roles = ["worst"] * 4 + ["best"]
    n_cols    = 5

    metrics = [graph_structural_metrics(results[i]) for i in col_idxs]

    plt.rcParams.update({
    "font.size":         22,   # base size — fallback for anything not listed below
    "axes.titlesize":    30,   # subplot titles (e.g. "λ = 10.0")
    "axes.labelsize":    25,   # x-axis and y-axis labels
    "xtick.labelsize":   20,   # x tick numbers
    "ytick.labelsize":   20,   # y tick numbers
    "legend.fontsize":   25,   # legend text
    "figure.titlesize":  30,   # fig.suptitle() — the big title above all panels
    })
    
    fig   = plt.figure(figsize=(26, 22), facecolor="white")
    outer = gridspec.GridSpec(2, 1, figure=fig,
                              height_ratios=[3.3, 1.7],
                              hspace=0.22,
                              left=0.04, right=0.98,
                              top=0.94,  bottom=0.04)

    gs_top = gridspec.GridSpecFromSubplotSpec(
        3, n_cols, subplot_spec=outer[0], hspace=0.45, wspace=0.32)
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, n_cols, subplot_spec=outer[1], wspace=0.32)

    role_col = {"worst": "#c44e52", "best": "#4c9a5f"}

    for col, (g_idx, role, m) in enumerate(zip(col_idxs, col_roles, metrics)):
        r    = results[g_idx]
        W    = r["W"]
        N    = r["N"]
        hcol = role_col[role]

        # ── Row 0: Adjacency ─────────────────────────────────────────────
        ax0 = fig.add_subplot(gs_top[0, col])
        ax0.set_facecolor("white")
        order = np.argsort(-m["degrees"])
        W_s   = W[np.ix_(order, order)]
        im    = ax0.imshow(W_s, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax0.set_xticks([]); ax0.set_yticks([])
        for sp in ax0.spines.values():
            sp.set_edgecolor(hcol)
            sp.set_linewidth(1.6 if role == "best" else 1.0)
        title = "BEST" if role == "best" else f"WORST {col+1}"
        ax0.set_title(f"{title} — Graph {g_idx+1}",
                      color=hcol, fontweight="bold")
        if col == n_cols - 1:
            cb = fig.colorbar(im, ax=ax0, fraction=0.04, pad=0.02)
            cb.ax.tick_params(labelsize=7)

        # ── Row 1: Degree ────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs_top[1, col])
        vals, counts = np.unique(m["degrees"].astype(int), return_counts=True)
        ax1.bar(vals, counts, color=hcol, alpha=0.8,
                edgecolor="black", linewidth=0.4)
        ax1.axvline(m["deg_mean"], linestyle="--", linewidth=1.2)
        ax1.set_title("Degree", fontweight="bold")
        ax1.grid(axis="y", linewidth=0.4, alpha=0.5)
        for sp in ax1.spines.values():
            sp.set_linewidth(0.6)

        # ── Row 2: Spectrum ──────────────────────────────────────────────
        ax2  = fig.add_subplot(gs_top[2, col])
        eigs = m["lap_eigenvalues"]
        x    = np.arange(N)
        ax2.plot(x, eigs, linewidth=1.4, color=hcol)
        ax2.scatter(x, eigs, s=10, color=hcol)
        ax2.scatter([1], [m["fiedler"]], s=35, edgecolor="black", zorder=5)
        ax2.set_title("Spectrum", fontweight="bold")
        ax2.grid(linewidth=0.4, alpha=0.5)
        for sp in ax2.spines.values():
            sp.set_linewidth(0.6)

        # ── Row 3: Paper panel ───────────────────────────────────────────
        ax3 = fig.add_subplot(gs_bot[0, col])
        ax3.axis("off")
        xL, xR = 0.02, 0.60
        y, dy  = 0.94, 0.052

        def line(left, right="", bold=False):
            nonlocal y
            ax3.text(xL, y, left, ha="left", va="center",
                     fontsize=8.2, fontweight="bold" if bold else "normal",
                     family="monospace")
            if right:
                ax3.text(xR, y, right, ha="left", va="center",
                         fontsize=8.2, family="monospace")
            y -= dy

        line("GRAPH", bold=True)
        line(f"N={N}",              f"E={m['n_edges']}")
        line(f"density={m['density']:.3f}", f"W={m['w_total']:.0f}")
        y -= dy * 0.4

        line("DEGREE", bold=True)
        line(f"mean={m['deg_mean']:.2f}", f"std={m['deg_std']:.2f}")
        line(f"min={m['deg_min']}",       f"max={m['deg_max']}")
        y -= dy * 0.4

        line("STRUCTURE", bold=True)
        line(f"bipartite={'Y' if m['is_bipartite'] else 'N'}",
             f"ratio={m['bipart_ratio']:.3f}")
        line(f"frustr={m['frustration_index']:.3f}")
        y -= dy * 0.4

        line("SPECTRUM", bold=True)
        line(f"lambda2={m['fiedler']:.3f}",
             f"lambda_max={m['lap_lambda_max']:.3f}")
        line(rf"lambda_bin={m['lam_bin']:.3f}",
             f"(u0_bin={m['u0_bin']:.3f})")
        y -= dy * 0.4

        line("PERFORMANCE", bold=True)
        line(f"cut*={m['best_cut']:.1f}",
             f"found={results[g_idx]['best_found']:.1f}")
        line(f"ratio={results[g_idx]['approx_ratio']:.3f}",
             rf"lam*={m['best_lam']:.4g}")
        line(f"n_opt={m['best_n_opt']}/{n_init}")

        for sp in ax3.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.6)
            sp.set_edgecolor("#cccccc")

    # ── Separator ────────────────────────────────────────────────────────────
    x_sep = (gs_top[0, 3].get_position(fig).x1 +
             gs_top[0, 4].get_position(fig).x0) / 2
    fig.add_artist(mlines.Line2D(
        [x_sep, x_sep], [0.05, 0.96],
        transform=fig.transFigure,
        linestyle="--", linewidth=1.2, color="black", alpha=0.4))

    fig.suptitle("Structural comparison of worst vs best graphs",
                 fontsize=12, fontweight="bold")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "HNN λ-sweep Max-Cut conjecture — N=20 ER graphs. "
            "λ = 1/u₀ is the gain parameter (thesis convention)."
        ))
    parser.add_argument("--N",          type=int,   default=20)
    parser.add_argument("--n_graphs",   type=int,   default=20)
    parser.add_argument("--p1",         type=float, default=0.5)
    parser.add_argument("--lam_values", type=float, nargs="+",
                        default=[10000, 1000, 100, 10, 1.0, 0.5, 0.2],
                        help="λ values to sweep (thesis convention, λ=1/u₀)")
    parser.add_argument("--n_init",     type=int,   default=10)
    parser.add_argument("--t_end",      type=float, default=80.0)
    parser.add_argument("--n_points",   type=int,   default=400)
    parser.add_argument("--rtol",       type=float, default=1e-6)
    parser.add_argument("--atol",       type=float, default=1e-8)
    parser.add_argument("--bin_tol",    type=float, default=0.05)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--save",       action="store_true")
    args = parser.parse_args()

    lam_values = sorted(args.lam_values)

    results = run_ensemble(args, lam_values)
    print_summary(results, lam_values, args)

    fig1 = make_figure1(results, lam_values, args)
    fig3 = make_figure3(results, lam_values, args)
    fig5 = make_figure5(results, lam_values, args)

    if args.save:
        tag = f"N{args.N}_p{args.p1:.2f}_ng{args.n_graphs}"
        for name, fig in [("hist", fig1), ("summary", fig3), ("structure", fig5)]:
            for ext in ("pdf", "png"):
                fname = f"hnn_lam_sweep_{name}_{tag}.{ext}"
                fig.savefig(fname, bbox_inches="tight", dpi=150)
                print(f"  Saved: {fname}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
