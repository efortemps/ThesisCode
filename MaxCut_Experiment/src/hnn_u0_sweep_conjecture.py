#!/usr/bin/env python3
"""
hnn_u0_sweep_conjecture.py
──────────────────────────────────────────────────────────────────────────────
Hopfield-Network Max-Cut experiment: u0 sweep over random ER graphs.

Unlike the OIM, the Hopfield network has no single "optimal" gain parameter
u0 analogous to mu_bin.  This experiment therefore sweeps a fixed set of u0
values and asks, for each graph, which u0 (if any) most reliably finds the
global Max-Cut optimum.

Setting
───────
• N = 20 nodes  — small enough to find the true global optimum by brute-force
  over all 2^20 ≈ 1 M spin configurations (vectorised, chunked for memory).
• n_graphs random Erdős–Rényi graphs (antiferromagnetic: J = −W, W ≥ 0).
• u0_values : fixed sweep grid  [0.01, 0.1, 1.0, 2.0, 5.0]  (5 values)
• n_init initial conditions per (graph, u0), drawn from uniform(−1, 1).
• Same ICs used for every u0 on a given graph — isolates gain effect.

HNN ODE
───────
  du_i/dt = −u_i − Σ_j W_ij tanh(u_j / u0)        (tau = 1)

  Interpretation of u0:
    Small u0 → steep tanh (near hard threshold)  → fast, aggressive binarisation
    Large u0 → shallow tanh (near linear)        → slow, may not binarise

  Stability of the origin s = 0:
    Hessian at origin: H(0) = W + u0 I
    λ_min(H(0)) < 0  iff  u0 < |λ_min(W)|  =: u0_bin
    → origin is UNSTABLE iff u0 < u0_bin  (GOOD: system leaves and binarises)
    → origin is STABLE   iff u0 > u0_bin  (BAD: system may stagnate near s=0)

Figures produced
────────────────
  Figure 1 — Per-u0 histograms + CDF
    Row 0: one histogram per u0 value showing distribution of n_opt
           (# ICs out of n_init reaching the global optimum)
    Row 1: one CDF per u0 value

  Figure 3 — Sweep summary (2 panels — no u0_bin distribution panel)
    (a) Approximation ratio box-plots: (best cut found) / (true opt) per u0
    (b) Success-fraction bar chart: % graphs where ≥1 IC hits opt, per u0
    (c) Mean n_opt ± std per u0  (line plot showing the sweep profile)

  Figure 5 — Structural comparison: 4 worst vs 1 best graph
    (same layout as maxcut_mubin_conjecture Figure 5)
    Worst = lowest  max_{u0} n_opt  (even the best u0 rarely finds the opt)
    Best  = highest max_{u0} n_opt

Usage
─────
python hnn_u0_sweep_conjecture.py
    [--N 20] [--n_graphs 20] [--p1 0.5]
    [--u0_values 0.01 0.1 1.0 2.0 5.0]
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

# ── Global style (mirrors OIM experiment) ────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.edgecolor":     "black",
    "axes.linewidth":     0.8,
    "xtick.color":        "black",
    "ytick.color":        "black",
    "text.color":         "black",
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#b0b0b0",
    "legend.facecolor":   "white",
    "legend.labelcolor":  "black",
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

# ── State-type colour palettes ────────────────────────────────────────────────
_STATE_BG = {
    "M2-binary":     "#d4eac8",
    "M1-mixed":      "#fde8c8",
    "Type-III":      "#e8c8de",
    "not-converged": "#e8e8e8",
}
_STATE_BADGE = {
    "M2-binary":     "#55a868",
    "M1-mixed":      "#e377c2",
    "Type-III":      "#8172b2",
    "not-converged": "#b0b0b0",
}

# Colour per u0 value (up to 8 values)
_U0_COLOURS = [C_RED, C_ORANGE, C_AMBER, C_GREEN, C_BLUE,
               C_PURPLE, "#17becf", "#bcbd22"]


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


def _lighten(hex_col, factor=0.6):
    hex_col = hex_col.lstrip("#")
    r, g, b = (int(hex_col[i:i+2], 16) for i in (0, 2, 4))
    return f"#{int(r+(255-r)*factor):02x}{int(g+(255-g)*factor):02x}{int(b+(255-b)*factor):02x}"


# ═══════════════════════════════════════════════════════════════════════════════
# Graph generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_er_graph(N: int, p1: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W   = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < p1:
                W[i, j] = W[j, i] = 1.0
    return W


# ═══════════════════════════════════════════════════════════════════════════════
# True global optimum  (brute-force, vectorised, chunked for N=20)
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
        s    = 2.0 * bits - 1.0          # shape (chunk, N), values ±1
        # cut = 0.25 * (W_sum - diag(s W sᵀ)) = 0.25*(W_sum - (s@W * s).sum(1))
        quad = (s @ W32 * s).sum(axis=1) # shape (chunk,)
        best = max(best, float((0.25 * (W_sum - quad)).max()))

    return best


# ═══════════════════════════════════════════════════════════════════════════════
# HNN ODE and convergence
# ═══════════════════════════════════════════════════════════════════════════════

def _hnn_rhs(t, u, W, u0):
    """tau=1 Hopfield ODE:  du/dt = −u − W tanh(u/u0)."""
    return -u - W @ np.tanh(u / u0)


def simulate_hnn(W: np.ndarray, u0: float, u_init: np.ndarray,
                 t_end: float, n_points: int,
                 rtol: float, atol: float):
    t_eval = np.linspace(0.0, t_end, n_points)
    return solve_ivp(
        _hnn_rhs, (0.0, t_end), u_init,
        args=(W, u0),
        method="RK45", t_eval=t_eval,
        rtol=rtol, atol=atol, dense_output=False,
    )


def classify_hnn_sol(sol, W, u0, best_cut, bin_tol=0.05):
    """
    Classify the terminal state of one HNN trajectory.

    State variables: u (membrane potential), activation s = tanh(u/u0).

    Terminal state types
    ────────────────────
    M2-binary    : all |s_i| ≥ 1 − bin_tol   (converged to ±1 corners)
    M1-mixed     : some |s_i| ≈ 1, some |s_i| ≈ 0, none intermediate
    Type-III     : at least one s_i at a continuous intermediate value
    not-converged: all |s_i| < bin_tol (stuck near origin)

    Returns dict with: bits, s_final, cut, H, residual,
                       is_binary, is_opt, state_type.
    """
    s_final = np.tanh(sol.y[:, -1] / u0)
    sigma   = np.sign(s_final); sigma[sigma == 0] = 1.0
    bits    = tuple(1 if s > 0 else 0 for s in sigma)

    cut      = 0.25 * float(np.sum(W * (1.0 - sigma[:, None] * sigma[None, :])))
    H        = 0.5  * float(sigma @ W @ sigma)
    residual = float(np.max(np.abs(np.abs(s_final) - 1.0)))

    n_near1 = sum(1 for s in s_final if abs(abs(s) - 1.0) < bin_tol)
    n_near0 = sum(1 for s in s_final if abs(s) < bin_tol)
    n_inter = len(s_final) - n_near1 - n_near0   # intermediate values

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
                  u0_values: list) -> dict:
    """
    Run the full HNN experiment for one graph across all u0 values.

    Steps
    ─────
    1. Find the true global optimum cut (brute force over 2^N configs).
    2. Sample n_init initial conditions u_init ~ uniform(−1, 1)^N.
       The SAME u_inits are used for every u0 value.
    3. For each u0:
       a. Simulate n_init ODE trajectories.
       b. Classify each terminal state.
       c. Count n_opt (# trajectories reaching the global optimum cut).
    4. Identify best_u0 = argmax_{u0} n_opt.
    """
    N       = W.shape[0]
    w_total = float(np.sum(np.triu(W)))

    # Step 1: true global optimum
    t0       = time.perf_counter()
    best_cut = find_global_optimum(W)
    t_enum   = time.perf_counter() - t0

    # Step 2: fixed ICs
    rng    = np.random.default_rng(args.seed + g_idx * 1000)
    u_inits = [rng.uniform(-1.0, 1.0, N) for _ in range(args.n_init)]

    # Step 3: sweep u0 values
    per_u0 = {}
    for u0 in u0_values:
        sols = [simulate_hnn(W, u0, u_init, args.t_end, args.n_points,
                              args.rtol, args.atol)
                for u_init in u_inits]
        conv = [classify_hnn_sol(s, W, u0, best_cut, args.bin_tol)
                for s in sols]
        cuts_u0   = np.array([c["cut"] for c in conv])
        bin_flags = np.array([c["is_binary"] for c in conv])
        n_opt     = int(np.sum([c["is_opt"] for c in conv]))

        per_u0[u0] = dict(
            conv=conv, cuts=cuts_u0.tolist(),
            bin_fraction=float(bin_flags.mean()),
            n_opt=n_opt,
            sols=sols,
        )

    # Step 4: summary across u0
    n_opt_per_u0 = {u0: per_u0[u0]["n_opt"] for u0 in u0_values}
    best_u0      = max(u0_values, key=lambda u: n_opt_per_u0[u])
    best_n_opt   = n_opt_per_u0[best_u0]
    # Approximation ratio: best cut achieved across ALL u0 / true opt
    all_cuts_flat = [c for u0 in u0_values for c in per_u0[u0]["cuts"]]
    best_found    = float(max(all_cuts_flat))
    approx_ratio  = best_found / max(best_cut, 1e-9)

    return dict(
        g_idx=g_idx, W=W, N=N,
        best_cut=best_cut, w_total=w_total,
        u0_values=u0_values,
        per_u0=per_u0,
        n_opt_per_u0=n_opt_per_u0,
        best_u0=best_u0,
        best_n_opt=best_n_opt,
        approx_ratio=approx_ratio,
        best_found=best_found,
        u_inits=u_inits,
        t_enum=t_enum,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Run full ensemble
# ═══════════════════════════════════════════════════════════════════════════════

def run_ensemble(args, u0_values: list) -> list:
    print(f"\n{'='*70}")
    print(f"  HNN u0-Sweep Max-Cut Experiment")
    print(f"  N={args.N}  p1={args.p1}  n_graphs={args.n_graphs}")
    print(f"  u0 values: {u0_values}")
    print(f"  n_init={args.n_init}  t_end={args.t_end}")
    print(f"  True global opt by 2^{args.N}={2**args.N} brute-force enumeration")
    print(f"{'='*70}")

    u0_hdr = "  ".join(f"u0={v:.2f}→n_opt" for v in u0_values)
    print(f"  {'g':>3}  {'|E|':>4}  {'opt':>5}  {u0_hdr}  best_u0")
    print("  " + "─" * 65)

    results = []
    t_total = time.perf_counter()

    for g_idx in range(args.n_graphs):
        W = generate_er_graph(args.N, args.p1, seed=args.seed + g_idx)
        r = run_one_graph(W, g_idx, args, u0_values)
        results.append(r)
        n_edges = int(np.sum(W)) // 2
        n_opts  = "  ".join(f"{r['n_opt_per_u0'][u0]:>10}" for u0 in u0_values)
        print(f"  {g_idx+1:>3}  {n_edges:>4}  {r['best_cut']:>5.1f}  "
              f"{n_opts}  u0={r['best_u0']:.2f}")

    elapsed = time.perf_counter() - t_total
    print(f"\n  Total: {elapsed:.1f}s  ({elapsed/args.n_graphs:.1f}s/graph)\n")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Console summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(results: list, u0_values: list, args):
    n_init   = args.n_init
    n_graphs = len(results)

    print(f"\n{'='*70}")
    print(f"  SUMMARY  |  N={args.N}  p1={args.p1}  {n_graphs} graphs  "
          f"n_init={n_init}")
    print(f"{'='*70}")

    for u0 in u0_values:
        n_opt = np.array([r["n_opt_per_u0"][u0] for r in results])
        frac_any = float(np.mean(n_opt >= 1))
        frac_all = float(np.mean(n_opt == n_init))
        print(f"\n  u0 = {u0:.3f}")
        print(f"    >=1 IC hits opt : {frac_any*100:.1f}%  |  "
              f"all {n_init} ICs : {frac_all*100:.1f}%  |  "
              f"mean n_opt = {n_opt.mean():.2f}")

    print(f"\n  Best u0 distribution across graphs:")
    best_u0s = [r["best_u0"] for r in results]
    for u0 in u0_values:
        cnt = sum(1 for v in best_u0s if v == u0)
        print(f"    u0={u0:.3f} is best for {cnt}/{n_graphs} graphs")

    approx = np.array([r["approx_ratio"] for r in results])
    print(f"\n  Approx ratio (best found / true opt):")
    print(f"    mean={approx.mean():.4f}  std={approx.std():.4f}  "
          f"min={approx.min():.4f}  "
          f"=1.0: {int(np.sum(approx >= 1-1e-6))}/{n_graphs}")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Per-u0 histograms + CDFs
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure1(results: list, u0_values: list, args) -> plt.Figure:
    """
    2 rows × n_u0 columns  +  one summary CDF column.

    Row 0: per-u0 histogram of n_opt (# ICs reaching global opt)
    Row 1: per-u0 CDF of n_opt

    Each column corresponds to one u0 value.  A final column overlays all
    CDFs for direct comparison.
    """
    n_init   = args.n_init
    n_graphs = len(results)
    n_u0     = len(u0_values)
    n_cols   = n_u0 + 1   # +1 for the overlay CDF

    fig = plt.figure(figsize=(4.2 * n_cols, 10), facecolor=WHITE)
    gs  = gridspec.GridSpec(2, n_cols, figure=fig,
                            hspace=0.55, wspace=0.35,
                            left=0.05, right=0.98, top=0.91, bottom=0.08)

    bins   = np.arange(-0.5, n_init + 1.5, 1.0)
    x_vals = np.arange(n_init + 1)

    for col, (u0, col_c) in enumerate(zip(u0_values, _U0_COLOURS)):
        n_opt = np.array([r["n_opt_per_u0"][u0] for r in results])
        counts, _ = np.histogram(n_opt, bins=bins)

        # ── Row 0: histogram ─────────────────────────────────────────────────
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
                         fontsize=7.5, color=BLACK)

        ax0.axvline(n_opt.mean(), color=C_AMBER, linewidth=1.8,
                    linestyle="--", zorder=5,
                    label=f"mean={n_opt.mean():.2f}")
        ax0.set_xticks(x_vals)
        ax0.set_xlim(-0.7, n_init + 0.7)
        ax0.set_ylim(0, max(counts.max(), 1) * 1.30)
        ax0.legend(fontsize=7.5, loc="upper left")
        _ax_style(ax0,
                  title=f"$u_0 = {u0}$",
                  xlabel=f"# ICs → opt  (/{n_init})",
                  ylabel="# graphs")

    # ── Row 1: CDFs ──────────────────────────────────────────────────────────
    ax_ov = fig.add_subplot(gs[1, n_u0])   # overlay CDF (last column, row 1)
    ax_ov.set_facecolor(WHITE)
    for sp in ax_ov.spines.values():
        sp.set_edgecolor(BLACK); sp.set_linewidth(0.8)
    ax_ov.grid(True, color=LIGHT, linewidth=0.5)

    for col, (u0, col_c) in enumerate(zip(u0_values, _U0_COLOURS)):
        n_opt = np.array([r["n_opt_per_u0"][u0] for r in results])
        cdf   = np.array([np.mean(n_opt >= k) for k in x_vals])

        # ── per-u0 CDF panel ──────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[1, col])
        ax1.step(x_vals, cdf, where="post", color=col_c, linewidth=2.0,
                 label="P(# ICs ≥ k)")
        ax1.fill_between(x_vals, cdf, step="post", alpha=0.15, color=col_c)
        ax1.axhline(1.0, color=GRAY, linewidth=0.8, linestyle="--")
        ax1.axvline(1, color=C_GREEN, linewidth=1.3, linestyle=":",
                    label=f"P(≥1) = {cdf[1]:.2f}")
        ax1.set_xticks(x_vals)
        ax1.set_xlim(-0.3, n_init + 0.3)
        ax1.set_ylim(-0.02, 1.10)
        ax1.text(0.97, 0.06,
                 f"P(≥1) = {cdf[1]*100:.1f}%\n"
                 f"P(all) = {cdf[n_init]*100:.1f}%",
                 transform=ax1.transAxes, ha="right", va="bottom",
                 fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE,
                           edgecolor=GRAY, alpha=0.92))
        ax1.legend(fontsize=7.5)
        _ax_style(ax1, title=f"CDF  $u_0 = {u0}$",
                  xlabel="k  (min # ICs → opt)", ylabel="P(# ICs ≥ k)")

        # Overlay on the shared comparison panel
        ax_ov.step(x_vals, cdf, where="post", color=col_c, linewidth=2.0,
                   label=f"$u_0={u0}$", zorder=3)

    ax_ov.axhline(1.0, color=GRAY, linewidth=0.8, linestyle="--")
    ax_ov.set_xticks(x_vals)
    ax_ov.set_xlim(-0.3, n_init + 0.3)
    ax_ov.set_ylim(-0.02, 1.10)
    ax_ov.legend(fontsize=8, loc="lower left")
    ax_ov.set_title("CDF comparison\n(all $u_0$ values overlaid)",
                    color=BLACK, fontsize=10, fontweight="bold", pad=5)
    ax_ov.set_xlabel("k  (min # ICs → opt)", color=BLACK, fontsize=10)
    ax_ov.set_ylabel("P(# ICs ≥ k)", color=BLACK, fontsize=10)
    ax_ov.tick_params(colors=BLACK, labelsize=9)

    # Top row: overlay histogram (last column, row 0)
    ax_oh = fig.add_subplot(gs[0, n_u0])
    ax_oh.set_facecolor(WHITE)
    for sp in ax_oh.spines.values():
        sp.set_edgecolor(BLACK); sp.set_linewidth(0.8)
    ax_oh.grid(True, color=LIGHT, linewidth=0.5)
    for u0, col_c in zip(u0_values, _U0_COLOURS):
        n_opt = np.array([r["n_opt_per_u0"][u0] for r in results])
        ax_oh.plot(x_vals, [np.mean(n_opt >= k) for k in x_vals],
                   color=col_c, linewidth=1.8, marker="o", markersize=4,
                   label=f"$u_0={u0}$")
    ax_oh.axhline(1.0, color=GRAY, linewidth=0.8, linestyle="--")
    ax_oh.set_xticks(x_vals); ax_oh.set_xlim(-0.3, n_init + 0.3)
    ax_oh.set_ylim(-0.02, 1.10)
    ax_oh.legend(fontsize=7.5)
    ax_oh.set_title("P(# ICs ≥ k) for all $u_0$",
                    color=BLACK, fontsize=10, fontweight="bold", pad=5)
    ax_oh.set_xlabel("k", color=BLACK, fontsize=10)
    ax_oh.set_ylabel("fraction of graphs", color=BLACK, fontsize=10)
    ax_oh.tick_params(colors=BLACK, labelsize=9)

    fig.suptitle(
        f"HNN u₀ Sweep — Conjecture Test  |  $N={args.N}$,  $p_1={args.p1}$,  "
        f"{n_graphs} ER graphs  |  $n_{{\\rm init}}={n_init}$  |  "
        f"$2^N={2**args.N}$ brute-force global opt",
        color=BLACK, fontsize=11, fontweight="bold")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Sweep summary  (2 original panels + 1 new sweep-profile panel)
#            Note: the mu_bin distribution panel is REMOVED (no equivalent)
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure3(results: list, u0_values: list, args) -> plt.Figure:
    """
    Three panels  (the mu_bin distribution panel from the OIM version is absent):

    (a) Approximation ratio box-plots per u0
        = (best cut found at u0) / (true global optimum)
        Shows whether each u0 reliably finds the optimum.

    (b) Success-fraction bar chart per u0
        Three grouped bars per u0: % graphs with 0 ICs, ≥1 IC, all ICs at opt.

    (c) Mean n_opt ± std per u0  (line + shaded band)
        Directly shows the sweep profile — is there a "sweet spot" u0?
    """
    n_init   = args.n_init
    n_graphs = len(results)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor=WHITE)
    fig.subplots_adjust(wspace=0.38, left=0.07, right=0.97,
                        top=0.87, bottom=0.13)

    x_pos = np.arange(len(u0_values))

    # ── (a) Approximation ratio box-plots ─────────────────────────────────────
    ax = axes[0]
    box_data = []
    for u0 in u0_values:
        ratios = [max(r["per_u0"][u0]["cuts"]) / max(r["best_cut"], 1e-9)
                  for r in results]
        box_data.append(ratios)

    bp = ax.boxplot(box_data, positions=x_pos, patch_artist=True,
                    medianprops=dict(color=BLACK, linewidth=1.5),
                    whiskerprops=dict(color=GRAY),
                    capprops=dict(color=GRAY), widths=0.5)
    for patch, col in zip(bp["boxes"], _U0_COLOURS):
        patch.set_facecolor(col); patch.set_alpha(0.68)

    ax.axhline(1.0, color=C_RED, linewidth=1.5, linestyle=":",
               label="global opt (= 1.0)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"$u_0={v}$" for v in u0_values], fontsize=8.5,
                        rotation=20)
    ax.legend(fontsize=8.5)
    _ax_style(ax,
              title=("Approximation ratio: best cut at each $u_0$ / global opt\n"
                     f"over {n_graphs} graphs  (box = IQR, whiskers = 10/90th pct)"),
              xlabel="$u_0$",
              ylabel="best cut / true global opt")

    # ── (b) Success-fraction grouped bar chart ────────────────────────────────
    ax = axes[1]
    frac_none = [float(np.mean(
        [r["n_opt_per_u0"][u0] == 0 for r in results])) * 100
        for u0 in u0_values]
    frac_any  = [float(np.mean(
        [r["n_opt_per_u0"][u0] >= 1 for r in results])) * 100
        for u0 in u0_values]
    frac_all  = [float(np.mean(
        [r["n_opt_per_u0"][u0] == n_init for r in results])) * 100
        for u0 in u0_values]

    w = 0.25
    ax.bar(x_pos - w, frac_none, width=w, color=C_RED,    alpha=0.75,
           edgecolor=BLACK, linewidth=0.5, label="0 ICs hit opt")
    ax.bar(x_pos,     frac_any,  width=w, color=C_BLUE,   alpha=0.75,
           edgecolor=BLACK, linewidth=0.5, label="≥1 IC hits opt")
    ax.bar(x_pos + w, frac_all,  width=w, color=C_GREEN,  alpha=0.75,
           edgecolor=BLACK, linewidth=0.5, label=f"all {n_init} ICs hit opt")

    # Value labels on top of bars
    for pos, f_none, f_any, f_all in zip(x_pos, frac_none, frac_any, frac_all):
        for offset, val, col in [(-w, f_none, C_RED),
                                   (0, f_any, C_BLUE),
                                   (w, f_all, C_GREEN)]:
            if val > 2:
                ax.text(pos + offset, val + 1, f"{val:.0f}",
                        ha="center", va="bottom", fontsize=7.5,
                        color=BLACK, fontweight="bold")

    ax.set_ylim(0, 115)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"$u_0={v}$" for v in u0_values], fontsize=8.5,
                        rotation=20)
    ax.axhline(100, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.legend(fontsize=8.5)
    _ax_style(ax,
              title=(f"Success fractions per $u_0$\n"
                     f"({n_graphs} graphs, {n_init} ICs each)"),
              xlabel="$u_0$",
              ylabel="% of graphs")

    # ── (c) Mean n_opt ± std vs u0  (sweep profile) ───────────────────────────
    ax = axes[2]
    means = np.array([np.mean([r["n_opt_per_u0"][u0] for r in results])
                      for u0 in u0_values])
    stds  = np.array([np.std ([r["n_opt_per_u0"][u0] for r in results])
                      for u0 in u0_values])

    ax.fill_between(x_pos, means - stds, means + stds,
                    alpha=0.18, color=C_BLUE)
    ax.plot(x_pos, means, color=C_BLUE, linewidth=2.2,
            marker="o", markersize=7, zorder=3, label="mean ± std")
    ax.errorbar(x_pos, means, yerr=stds, fmt="none",
                ecolor=C_BLUE, elinewidth=1.2, capsize=4, capthick=1.2)

    # Highlight the u0 with highest mean
    best_col = int(np.argmax(means))
    ax.scatter([x_pos[best_col]], [means[best_col]],
               color=C_GREEN, s=100, zorder=5, edgecolors=BLACK,
               linewidths=0.7,
               label=f"best $u_0={u0_values[best_col]}$")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"$u_0={v}$" for v in u0_values], fontsize=8.5,
                        rotation=20)
    ax.set_ylim(-0.3, n_init + 0.5)
    ax.axhline(n_init, color=GRAY, linewidth=0.8, linestyle="--",
               label=f"n_init = {n_init}")
    ax.legend(fontsize=8.5)
    _ax_style(ax,
              title=("Mean # ICs → global opt ± std\n"
                     "Sweep profile: is there a best $u_0$?"),
              xlabel="$u_0$",
              ylabel=f"mean # ICs → global opt  (/{n_init})")

    fig.suptitle(
        f"HNN Sweep Summary  |  $N={args.N}$,  $p_1={args.p1}$,  "
        f"{n_graphs} ER graphs  |  $n_{{\\rm init}}={n_init}$",
        color=BLACK, fontsize=11, fontweight="bold")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Graph structural metrics  (same as maxcut_mubin_conjecture.py)
# ═══════════════════════════════════════════════════════════════════════════════

def graph_structural_metrics(r: dict) -> dict:
    W        = r["W"]
    N        = r["N"]
    best_cut = r["best_cut"]
    w_total  = r["w_total"]

    n_edges  = int(np.sum(W)) // 2
    density  = n_edges / max(N * (N - 1) // 2, 1)
    degrees  = W.sum(axis=1)

    L        = np.diag(degrees) - W
    lap_eigs = np.sort(np.linalg.eigvalsh(L))
    fiedler  = float(lap_eigs[1]) if N > 1 else 0.0
    lap_gap  = float(lap_eigs[1] - lap_eigs[0]) if N > 1 else 0.0
    lap_lmax = float(lap_eigs[-1])

    adj_eigs   = np.sort(np.linalg.eigvalsh(W))[::-1]
    adj_lmax   = float(adj_eigs[0])
    lam_min_W  = float(np.linalg.eigvalsh(W)[0])
    u0_bin     = float(abs(lam_min_W))          # origin stability threshold

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
        deg_min=int(degrees.min()),     deg_max=int(degrees.max()),
        lap_eigenvalues=lap_eigs, fiedler=fiedler,
        lap_spectral_gap=lap_gap, lap_lambda_max=lap_lmax,
        adj_eigenvalues=adj_eigs, adj_lambda_max=adj_lmax,
        lam_min_W=lam_min_W, u0_bin=u0_bin,
        is_bipartite=is_bip, bipart_ratio=bipart_ratio,
        frustration_index=frustration,
        n_triangles=n_triangles, mean_clustering=mean_clust,
        # HNN-specific
        best_cut=best_cut,
        best_n_opt=r["best_n_opt"],
        best_u0=r["best_u0"],
        n_opt_per_u0=r["n_opt_per_u0"],
        n_init=len(r["u_inits"]),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Structural comparison: 4 worst vs 1 best
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure5(results: list, u0_values: list, args) -> plt.Figure:

    n_init   = args.n_init
    n_graphs = len(results)

    n_opt_best_arr = np.array([r["best_n_opt"] for r in results])
    u0_bin_arr     = np.array([abs(float(np.linalg.eigvalsh(r["W"])[0]))
                                for r in results])

    worst_idxs = sorted(range(n_graphs),
                        key=lambda i: (n_opt_best_arr[i], -u0_bin_arr[i]))[:4]
    best_idx   = sorted(range(n_graphs),
                        key=lambda i: (-n_opt_best_arr[i], u0_bin_arr[i]))[0]

    col_idxs  = worst_idxs + [best_idx]
    col_roles = ["worst"] * 4 + ["best"]
    n_cols    = 5

    metrics = [graph_structural_metrics(results[i]) for i in col_idxs]

    # ── GLOBAL STYLE (Nature/IEEE-like) ──────────────────────────────────────
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
    })

    fig = plt.figure(figsize=(26, 22), facecolor="white")

    outer = gridspec.GridSpec(2, 1, figure=fig,
                              height_ratios=[3.3, 1.7],
                              hspace=0.22,
                              left=0.04, right=0.98,
                              top=0.94, bottom=0.04)

    gs_top = gridspec.GridSpecFromSubplotSpec(
        3, n_cols, subplot_spec=outer[0],
        hspace=0.45, wspace=0.32
    )

    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, n_cols, subplot_spec=outer[1],
        wspace=0.32
    )

    role_col = {"worst": "#c44e52", "best": "#4c9a5f"}

    # ─────────────────────────────────────────────────────────────────────────
    for col, (g_idx, role, m) in enumerate(zip(col_idxs, col_roles, metrics)):
        r    = results[g_idx]
        W    = r["W"]
        N    = r["N"]
        hcol = role_col[role]

        # ── Row 0: Adjacency ────────────────────────────────────────────────
        ax0 = fig.add_subplot(gs_top[0, col])
        ax0.set_facecolor("white")

        order = np.argsort(-m["degrees"])
        W_s   = W[np.ix_(order, order)]

        im = ax0.imshow(W_s, cmap="Blues", vmin=0, vmax=1, aspect="auto")

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

        # ── Row 1: Degree ───────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs_top[1, col])

        vals, counts = np.unique(m["degrees"].astype(int), return_counts=True)
        ax1.bar(vals, counts, color=hcol, alpha=0.8, edgecolor="black", linewidth=0.4)

        ax1.axvline(m["deg_mean"], linestyle="--", linewidth=1.2)

        ax1.set_title("Degree", fontweight="bold")
        ax1.grid(axis="y", linewidth=0.4, alpha=0.5)

        for sp in ax1.spines.values():
            sp.set_linewidth(0.6)

        # ── Row 2: Spectrum ─────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs_top[2, col])

        eigs = m["lap_eigenvalues"]
        x    = np.arange(N)

        ax2.plot(x, eigs, linewidth=1.4, color=hcol)
        ax2.scatter(x, eigs, s=10, color=hcol)

        ax2.scatter([1], [m["fiedler"]], s=35, edgecolor="black", zorder=5)

        ax2.set_title("Spectrum", fontweight="bold")
        ax2.grid(linewidth=0.4, alpha=0.5)

        for sp in ax2.spines.values():
            sp.set_linewidth(0.6)

        # ── Row 3: PAPER PANEL (aligned) ────────────────────────────────────
        ax3 = fig.add_subplot(gs_bot[0, col])
        ax3.axis("off")

        # fixed x anchors (THIS gives perfect alignment)
        xL = 0.02
        xR = 0.60

        y  = 0.94
        dy = 0.052

        def line(left, right="", bold=False):
            nonlocal y
            ax3.text(xL, y, left,
                     ha="left", va="center",
                     fontsize=8.2,
                     fontweight="bold" if bold else "normal",
                     family="monospace")

            if right:
                ax3.text(xR, y, right,
                         ha="left", va="center",
                         fontsize=8.2,
                         family="monospace")
            y -= dy

        # Build compact aligned panel
        line("GRAPH", bold=True)
        line(f"N={N}", f"E={m['n_edges']}")
        line(f"density={m['density']:.3f}", f"W={m['w_total']:.0f}")

        y -= dy * 0.4

        line("DEGREE", bold=True)
        line(f"mean={m['deg_mean']:.2f}", f"std={m['deg_std']:.2f}")
        line(f"min={m['deg_min']}", f"max={m['deg_max']}")

        y -= dy * 0.4

        line("STRUCTURE", bold=True)
        line(f"bipartite={'Y' if m['is_bipartite'] else 'N'}",
             f"ratio={m['bipart_ratio']:.3f}")
        line(f"frustr={m['frustration_index']:.3f}")

        y -= dy * 0.4

        line("SPECTRUM", bold=True)
        line(f"lambda2={m['fiedler']:.3f}",
             f"lambda_max={m['lap_lambda_max']:.3f}")
        line(f"u0_bin={m['u0_bin']:.3f}")

        y -= dy * 0.4

        line("PERFORMANCE", bold=True)
        line(f"cut*={m['best_cut']:.1f}",
             f"found={results[g_idx]['best_found']:.1f}")
        line(f"ratio={results[g_idx]['approx_ratio']:.3f}",
             f"u0*={m['best_u0']}")
        line(f"n_opt={m['best_n_opt']}/{n_init}")

        # subtle border (Nature style)
        for sp in ax3.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.6)
            sp.set_edgecolor("#cccccc")

    # ── Separator ───────────────────────────────────────────────────────────
    x_sep = (gs_top[0, 3].get_position(fig).x1 +
             gs_top[0, 4].get_position(fig).x0) / 2

    fig.add_artist(mlines.Line2D([x_sep, x_sep], [0.05, 0.96],
                                transform=fig.transFigure,
                                linestyle="--", linewidth=1.2,
                                color="black", alpha=0.4))

    fig.suptitle("Structural comparison of worst vs best graphs",
                 fontsize=12, fontweight="bold")

    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HNN u0-sweep Max-Cut conjecture — N=20 ER graphs")
    parser.add_argument("--N",          type=int,   default=20)
    parser.add_argument("--n_graphs",   type=int,   default=20)
    parser.add_argument("--p1",         type=float, default=0.5)
    parser.add_argument("--u0_values",  type=float, nargs="+",
                        default=[0.01, 0.1, 1.0, 2.0, 5.0])
    parser.add_argument("--n_init",     type=int,   default=10)
    parser.add_argument("--t_end",      type=float, default=80.0)
    parser.add_argument("--n_points",   type=int,   default=400)
    parser.add_argument("--rtol",       type=float, default=1e-6)
    parser.add_argument("--atol",       type=float, default=1e-8)
    parser.add_argument("--bin_tol",    type=float, default=0.05)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--save",       action="store_true")
    args = parser.parse_args()

    u0_values = sorted(args.u0_values)

    results = run_ensemble(args, u0_values)
    print_summary(results, u0_values, args)

    fig1 = make_figure1(results, u0_values, args)
    fig3 = make_figure3(results, u0_values, args)
    fig5 = make_figure5(results, u0_values, args)

    if args.save:
        tag = f"N{args.N}_p{args.p1:.2f}_ng{args.n_graphs}"
        for name, fig in [("hist_cdf",  fig1),
                           ("summary",   fig3),
                           ("structure", fig5)]:
            for ext in ("pdf", "png"):
                fname = f"hnn_sweep_{name}_{tag}.{ext}"
                fig.savefig(fname, bbox_inches="tight", dpi=150)
                print(f"  Saved: {fname}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
