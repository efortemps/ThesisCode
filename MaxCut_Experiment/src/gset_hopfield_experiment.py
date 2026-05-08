#!/usr/bin/env python3
"""
gset_hopfield_experiment.py
──────────────────────────────────────────────────────────────────────────────
G-Set Max-Cut experiment using a Continuous Hopfield-Tank Network (HNN).

Key difference vs. OIM
───────────────────────
In the OIM, each equilibrium φ* has its own per-state stability threshold
λ_max(D(φ*)), so increasing μ selectively stabilises individual corners.
In the HNN there is only ONE global threshold: the gain parameter u0.
  • Large u0 → tanh ≈ linear  → the network sits near the origin, no binarisation.
  • Small u0 → tanh ≈ sign     → strong saturation; all corners become attractors
                                  simultaneously.
  • u0_bin is the critical u0 below which all trajectories binarise.

Integration — RK45 via scipy.integrate.solve_ivp
──────────────────────────────────────────────────
The Euler integrator requires n_steps × dt wall-time and is
bottlenecked by Python loop overhead.  For G-Set graphs (N ≥ 800) at small
u0 the stability limit forces dt < 1e-4, meaning ≥ 50 000 steps just to
reach t = 5 — making the experiment run for many hours.

RK45 (adaptive Runge-Kutta 4/5) from scipy solves this:
  • Adapts its internal step size automatically — large steps when the
    state changes slowly, small steps only during fast transients.
  • For the HNN after the initial binarisation transient the state barely
    moves, so RK45 takes very large steps → effectively 10–100× fewer
    evaluations than fixed-step Euler.
  • No stability concerns: RK45 chooses a step size that respects error
    tolerances at each u0 value independently.
  • Total cost: O(n_evals × N²) where n_evals ≈ 100–500 per trajectory
    instead of n_steps = 20 000–500 000 for Euler.

Procedure
─────────
Phase 1 — find û0_bin empirically
  Sweep u0 downward from u0_start to u0_min.
  At each u0 run n_init trajectories from random initial conditions.
  Stop at the first u0 where EVERY trajectory binarises.
  → û0_bin

Phase 2 — quality sweep below û0_bin
  Dense sweep of u0 ≤ û0_bin.
  At each value run n_init trajectories, record best and mean cut found.

Output
──────
• Console report: per-u0 best/mean cut, binarisation fraction, timing
• Figure 1 — Phase 1 scan: binarisation fraction + cut vs u0 (decreasing)
• Figure 2 — Phase 2 quality sweep: best/mean cut vs u0 (decreasing)

Usage
─────
python gset_hopfield_experiment.py --graph G1.txt [options]

--graph PATH          G-Set file (required)
--u0_start   FLOAT    Phase-1 scan start         (default: 2.0)
--u0_min     FLOAT    Phase-1 scan lower bound   (default: 0.001)
--u0_step    FLOAT    Phase-1 step size (log)    (default: 5 steps/decade)
--n_u0_2     INT      Phase-2 number of u0 points(default: 15)
--u0_2_factor FLOAT   Phase-2 lower = û0_bin / f (default: 4.0)
--n_init     INT      trajectories per u0         (default: 10)
--t_end      FLOAT    ODE integration horizon     (default: 30.0)
--rtol       FLOAT    RK45 relative tolerance     (default: 1e-4)
--atol       FLOAT    RK45 absolute tolerance     (default: 1e-6)
--seed       INT      RNG seed                    (default: 42)
--known_opt  FLOAT    known optimum cut (optional, for gap reporting)
--save               save figures as PDF + PNG
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ── plotting style (mirrors OIM experiment) ───────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
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

WHITE     = "#ffffff"
BLACK     = "#000000"
GRAY      = "#b0b0b0"
LIGHT     = "#e6e6e6"
C_STABLE  = "#4C72B0"   # blue  — binarised / good cut
C_UNSTABLE= "#DD8452"   # orange — not binarised
C_U0_BIN  = "#c44e52"   # red   — û0_bin marker
C_BEST    = "#55a868"   # green — best-cut marker


# ═══════════════════════════════════════════════════════════════════════════════
# G-Set parser  (identical to OIM experiment)
# ═══════════════════════════════════════════════════════════════════════════════
def parse_gset(path: str):
    """
    Parse a G-Set benchmark file.

    Format::
        N  M           ← nodes, edges  (first non-comment line)
        u  v  w        ← edges (1-indexed, weight w)
        ...

    Returns
    -------
    n      : int
    W      : (N, N) symmetric weight matrix  (W_ij = |w|)
    edges  : list of (u, v, w)   (0-indexed)
    """
    edges_raw = []
    n = m = None

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("c "):
                continue
            tok = line.split()
            if n is None:
                n, m = int(tok[0]), int(tok[1])
                continue
            if len(tok) >= 3:
                u, v, w = int(tok[0]) - 1, int(tok[1]) - 1, float(tok[2])
            elif len(tok) == 2:
                u, v, w = int(tok[0]) - 1, int(tok[1]) - 1, 1.0
            else:
                continue
            edges_raw.append((u, v, w))

    if n is None:
        raise ValueError(f"Could not parse G-Set file: {path}")

    W = np.zeros((n, n), dtype=float)
    for u, v, w in edges_raw:
        W[u, v] += abs(w)
        W[v, u] += abs(w)

    return n, W, [(u, v, w) for u, v, w in edges_raw]


# ═══════════════════════════════════════════════════════════════════════════════
# Single-u0 run
# ═══════════════════════════════════════════════════════════════════════════════
def _hnn_rhs(t: float, u: np.ndarray, W: np.ndarray, u0: float) -> np.ndarray:
    """
    HNN ODE right-hand side (standalone, avoids Python method-call overhead).

        du/dt = −u − W tanh(u/u0)          (tau = 1)

    Called directly by scipy solve_ivp — no HopfieldNetMaxCut wrapper needed
    during integration, which eliminates the per-step Python dispatch overhead
    that made the Euler loop slow.
    """
    return -u - W @ np.tanh(u / u0)


def _binary_cut(u: np.ndarray, W: np.ndarray, u0: float) -> float:
    """Cut value after hard binarisation  σ = sign(tanh(u/u0))."""
    sigma = np.sign(np.tanh(u / u0))
    sigma[sigma == 0] = 1.0
    return 0.25 * float(np.sum(W * (1.0 - sigma[:, None] * sigma[None, :])))


def _is_binarised(u: np.ndarray, u0: float, tol: float = 0.05) -> bool:
    """True iff every |tanh(u_i/u0)| > 1 − tol."""
    return bool(np.all(np.abs(np.tanh(u / u0)) > 1.0 - tol))


def _partition(u: np.ndarray, u0: float) -> np.ndarray:
    """Hard spin assignment  σ ∈ {−1, +1}^N."""
    s = np.sign(np.tanh(u / u0))
    s[s == 0] = 1.0
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# Single-u0 run  (RK45, fixes rebuild bug)
# ═══════════════════════════════════════════════════════════════════════════════
def run_u0(W: np.ndarray, u0: float, u0_inits: list,
           t_end: float, rtol: float, atol: float,
           seed: int, bin_tol: float = 0.05) -> dict:
    """
    Run n_init Hopfield trajectories at a given u0 using RK45.

    Why RK45 instead of Euler
    ──────────────────────────
    The Euler loop  `for _ in range(n_steps): net.update()`  has two costs:
      1. Python function-call overhead  (~5 μs × n_steps per trajectory)
      2. n_steps must be large enough that n_steps × dt reaches t_end while
         satisfying the Euler stability condition dt < 2/(1 + λ_max(W)/u0).
         At u0=0.01 on G1 this forces dt < 4e-4, so ≥ 12 500 steps for t=5.

    RK45 (scipy) adapts its step size internally:
      • During fast transients (early binarisation) it takes small steps.
      • Once the state barely moves it takes very large steps.
      • Total evaluations: ~100–500 per trajectory vs. 20 000+ for Euler.
      • No stability concerns: the solver guarantees the requested tolerances.

    Bug fix: original code ran n_steps AGAIN to rebuild the best partition.
    Here, each trajectory's terminal state u[:, −1] is stored in a list and
    the best partition is read directly — no extra integration needed.

    Parameters
    ----------
    W         : (N, N) symmetric weight matrix
    u0        : gain parameter
    u0_inits  : list of (N,) initial membrane-potential vectors
    t_end     : integration horizon in dimensionless time  (default: 30)
    rtol/atol : RK45 tolerances
    seed      : (unused here, kept for API compatibility)
    bin_tol   : binarisation tolerance  |tanh(u_i/u0)| > 1 − bin_tol

    Returns dict with keys:
        u0, best_cut, mean_cut, std_cut, best_partition,
        bin_fraction, all_cuts, all_bin, t_elapsed, n_func_evals
    """
    t0 = time.perf_counter()
    N  = W.shape[0]

    cuts         = []
    bin_flags    = []
    terminal_us  = []   # store terminal u-state for each IC (fixes rebuild bug)
    total_evals  = 0

    for u_init in u0_inits:
        sol = solve_ivp(
            _hnn_rhs,
            t_span=(0.0, t_end),
            y0=u_init,
            method="RK45",
            args=(W, u0),
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )
        u_final = sol.y[:, -1]
        terminal_us.append(u_final)
        total_evals += sol.nfev

        cuts.append(_binary_cut(u_final, W, u0))
        bin_flags.append(_is_binarised(u_final, u0, bin_tol))

    cuts      = np.array(cuts, dtype=float)
    bin_flags = np.array(bin_flags)

    best_idx  = int(np.argmax(cuts))
    bin_cuts  = cuts[bin_flags]

    # Read partition directly from stored terminal state — no re-integration
    partition = _partition(terminal_us[best_idx], u0)

    return dict(
        u0            = u0,
        best_cut      = float(cuts.max()),
        mean_cut      = float(bin_cuts.mean()) if len(bin_cuts) else float(cuts.mean()),
        std_cut       = float(bin_cuts.std())  if len(bin_cuts) else float(cuts.std()),
        best_partition= partition,
        bin_fraction  = float(bin_flags.mean()),
        all_cuts      = cuts.tolist(),
        all_bin       = bin_flags.tolist(),
        t_elapsed     = time.perf_counter() - t0,
        n_func_evals  = total_evals,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Analytical û0_bin  (global threshold from Hessian at origin)
# ═══════════════════════════════════════════════════════════════════════════════
def analytical_u0_bin(W: np.ndarray) -> float:
    """
    At the origin u=0, s=0 and the linearised Hopfield dynamics are:

        du/dt = -(I + W/u0) u / τ

    Stability of the origin requires all eigenvalues of (I + W/u0) to be
    positive, i.e.  u0 > -λ_min(W).  If λ_min(W) ≤ 0 (the usual case for
    a non-trivial graph), the origin becomes unstable for

        u0 < u0_bin_theory ≡ -λ_min(W) = |λ_min(W)|

    This is the GLOBAL threshold: below it, ALL corners simultaneously
    become stable fixed points.
    """
    lmin = float(np.linalg.eigvalsh(W).min())
    return max(0.0, -lmin)    # = |λ_min| when λ_min < 0


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — downward scan to find û0_bin
# ═══════════════════════════════════════════════════════════════════════════════
def phase1_find_u0_bin(W, u0_inits, args):
    """
    Sweep u0 DOWNWARD from u0_start, logarithmically spaced,
    stopping at the first value where all trajectories binarise.

    Returns
    -------
    u0_hat_bin : float
    records    : list of run_u0 dicts
    """
    n_pts = int(np.ceil(np.log10(args.u0_start / args.u0_min)
                        * args.u0_steps_per_decade)) + 1
    u0_grid = np.logspace(np.log10(args.u0_start),
                          np.log10(args.u0_min),
                          num=max(n_pts, 10))

    n_init = len(u0_inits)
    records = []
    u0_hat_bin = None

    print(f"\n{'='*65}")
    print(f" PHASE 1 — empirical û0_bin search (downward sweep, RK45)")
    print(f" u0 grid: [{args.u0_start:.4f}, {args.u0_min:.6f}] "
          f"({len(u0_grid)} pts, log-spaced)")
    print(f" n_init={n_init}  t_end={args.t_end}  "
          f"rtol={args.rtol:.0e}  atol={args.atol:.0e}")
    print(f"{'='*65}")
    hdr = (f"  {'u0':>10} {'bin_frac':>9} {'best_cut':>10} "
           f"{'mean_cut':>10} {'nfev':>7} {'t(s)':>6}")
    print(hdr)
    print("  " + "─" * (10 + 9 + 10 + 10 + 7 + 6 + 8))

    for u0 in u0_grid:
        rec = run_u0(W, u0, u0_inits,
                     args.t_end, args.rtol, args.atol,
                     args.seed)
        records.append(rec)

        flag = " ← û0_bin ✓" if rec["bin_fraction"] == 1.0 and u0_hat_bin is None else ""
        print(f"  {u0:>10.5f} {rec['bin_fraction']:>9.3f} "
              f"{rec['best_cut']:>10.2f} {rec['mean_cut']:>10.2f} "
              f"{rec['n_func_evals']:>7} "
              f"{rec['t_elapsed']:>6.1f}s{flag}")

        if rec["bin_fraction"] == 1.0 and u0_hat_bin is None:
            u0_hat_bin = u0
            break

    if u0_hat_bin is None:
        print(f"\n [warn] No full binarisation found down to u0={args.u0_min:.6f}.")
        print(f"         Using the u0 with highest bin_fraction as fallback.")
        best_rec = max(records, key=lambda r: (r["bin_fraction"], r["best_cut"]))
        u0_hat_bin = best_rec["u0"]

    print(f"\n → û0_bin = {u0_hat_bin:.5f}  "
          f"(first fully-binarising u0 found)\n")
    return u0_hat_bin, records


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — quality sweep ≤ û0_bin
# ═══════════════════════════════════════════════════════════════════════════════
def phase2_quality_sweep(W, u0_inits, u0_hat_bin, args):
    """
    Dense sweep of u0 values ≤ û0_bin (log-spaced downward).
    """
    u0_lo  = u0_hat_bin / args.u0_2_factor
    u0_arr = np.logspace(np.log10(u0_hat_bin),
                         np.log10(max(u0_lo, 1e-6)),
                         num=args.n_u0_2)

    print(f"{'='*65}")
    print(f" PHASE 2 — quality sweep u0 ∈ [{u0_lo:.5f}, {u0_hat_bin:.5f}]"
          f"  ({args.n_u0_2} pts, RK45)")
    print(f" n_init={len(u0_inits)}  t_end={args.t_end}  "
          f"rtol={args.rtol:.0e}  atol={args.atol:.0e}")
    print(f"{'='*65}")
    hdr = (f"  {'u0':>10} {'bin_frac':>9} {'best_cut':>10} "
           f"{'mean_cut':>10} {'std_cut':>8} {'nfev':>7} {'t(s)':>6}")
    print(hdr)
    print("  " + "─" * (10 + 9 + 10 + 10 + 8 + 7 + 6 + 8))

    records = []
    for u0 in u0_arr:
        rec = run_u0(W, u0, u0_inits,
                     args.t_end, args.rtol, args.atol,
                     args.seed)
        records.append(rec)
        print(f"  {u0:>10.5f} {rec['bin_fraction']:>9.3f} "
              f"{rec['best_cut']:>10.2f} {rec['mean_cut']:>10.2f} "
              f"{rec['std_cut']:>8.3f} {rec['n_func_evals']:>7} "
              f"{rec['t_elapsed']:>6.1f}s")

    best = max(records, key=lambda r: r["best_cut"])
    print(f"\n → Best cut in Phase 2: {best['best_cut']:.2f} "
          f"at u0 = {best['u0']:.5f}")
    if args.known_opt:
        gap = 100.0 * (args.known_opt - best["best_cut"]) / args.known_opt
        print(f"   Approx ratio: {best['best_cut'] / args.known_opt:.4f} "
              f"(gap = {gap:.2f}% vs known opt = {args.known_opt:.0f})")
    print()
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# Axis styling helper
# ═══════════════════════════════════════════════════════════════════════════════
def _ax_style(ax, title="", xlabel="", ylabel="", titlesize=11):
    ax.set_facecolor(WHITE)
    ax.tick_params(colors=BLACK, labelsize=10)
    for sp in ax.spines.values():
        sp.set_edgecolor(BLACK)
        sp.set_linewidth(0.8)
    ax.grid(True, color=LIGHT, linewidth=0.6, zorder=0)
    if title:  ax.set_title(title,  color=BLACK, fontsize=titlesize, pad=5)
    if xlabel: ax.set_xlabel(xlabel, color=BLACK, fontsize=11)
    if ylabel: ax.set_ylabel(ylabel, color=BLACK, fontsize=11)


# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════
def make_figures(args, n, w_total, u0_hat_bin,
                 u0_theory, p1_records, p2_records):
    """
    Figure 1 — Phase 1 scan (binarisation + cut vs u0, decreasing x-axis)
    Figure 2 — Phase 2 quality sweep
    """
    stem = Path(args.graph).stem

    def _arrays(records):
        u0_arr  = np.array([r["u0"]          for r in records])
        bin_arr = np.array([r["bin_fraction"] for r in records])
        best_arr= np.array([r["best_cut"]     for r in records])
        mean_arr= np.array([r["mean_cut"]     for r in records])
        std_arr = np.array([r["std_cut"]      for r in records])
        return u0_arr, bin_arr, best_arr, mean_arr, std_arr

    p1_u0, p1_bin, p1_best, p1_mean, _ = _arrays(p1_records)
    p2_u0, p2_bin, p2_best, p2_mean, p2_std = _arrays(p2_records)

    best_p2_idx  = int(np.argmax(p2_best))
    u0_best_cut  = p2_u0[best_p2_idx]
    best_cut     = p2_best[best_p2_idx]

    def _vline_annotate(ax, x, label, color, ls, yrel=0.05):
        ax.axvline(x, color=color, linewidth=2.0, linestyle=ls, zorder=5)
        ylims = ax.get_ylim()
        y = ylims[0] + yrel * (ylims[1] - ylims[0])
        ax.text(x, y, f" {label}", color=color, fontsize=8.5,
                ha="left", va="bottom", rotation=90)

    # ── Figure 1 ─────────────────────────────────────────────────────────────
    fig1, axes1 = plt.subplots(2, 1, figsize=(13, 8), facecolor=WHITE)
    fig1.subplots_adjust(hspace=0.42, left=0.09, right=0.97,
                         top=0.90, bottom=0.09)

    # top: binarisation fraction vs u0
    ax = axes1[0]
    ax.plot(p1_u0, p1_bin, color=C_STABLE, linewidth=2.0,
            marker="o", markersize=5, zorder=3, label="bin. fraction")
    ax.fill_between(p1_u0, p1_bin, alpha=0.12, color=C_STABLE)
    ax.axhline(1.0, color=GRAY, linewidth=0.9, linestyle="--")
    ax.set_xscale("log")
    ax.invert_xaxis()          # decreasing u0 → left to right = "more binarised"
    ax.set_ylim(-0.05, 1.12)
    _ax_style(ax,
              title=(f"Phase 1 — binarisation fraction vs $u_0$ | "
                     f"$N={n}$, $n_{{\\rm init}}={args.n_init}$"),
              xlabel="$u_0$ (log scale, decreasing →)",
              ylabel="fraction of fully binarised trajectories")
    _vline_annotate(ax, u0_hat_bin,
                    f"$\\hat{{u}}_{{0,bin}}={u0_hat_bin:.4f}$",
                    C_U0_BIN, "--")
    if u0_theory > 0:
        _vline_annotate(ax, u0_theory,
                        f"theory$={u0_theory:.4f}$",
                        BLACK, ":", yrel=0.55)
    ax.legend(fontsize=9, loc="lower left")

    # bottom: best + mean cut vs u0
    ax = axes1[1]
    ax.plot(p1_u0, p1_best, color=C_STABLE, linewidth=2.0,
            marker="o", markersize=5, label="best cut (binarised)", zorder=3)
    ax.plot(p1_u0, p1_mean, color=C_UNSTABLE, linewidth=1.4,
            marker="s", markersize=4, linestyle="--",
            label="mean cut (binarised)", zorder=2)
    ax.set_xscale("log")
    ax.invert_xaxis()
    _ax_style(ax,
              title="Phase 1 — cut quality vs $u_0$ (scan to first full binarisation)",
              xlabel="$u_0$ (log scale, decreasing →)",
              ylabel="cut value")
    _vline_annotate(ax, u0_hat_bin,
                    f"$\\hat{{u}}_{{0,bin}}={u0_hat_bin:.4f}$",
                    C_U0_BIN, "--")
    if args.known_opt:
        ax.axhline(args.known_opt, color=BLACK, linewidth=1.3,
                   linestyle=":", label=f"known opt = {args.known_opt:.0f}")
    ax.legend(fontsize=9, loc="lower left")

    fig1.suptitle(
        f"HNN G-Set experiment | {stem} | $N={n}$ | "
        f"$W_{{\\rm tot}}={w_total:.0f}$ | "
        f"$\\hat{{u}}_{{0,bin}}={u0_hat_bin:.4f}$",
        color=BLACK, fontsize=12, fontweight="bold")

    # ── Figure 2 ─────────────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(2, 1, figsize=(13, 8), facecolor=WHITE)
    fig2.subplots_adjust(hspace=0.42, left=0.09, right=0.97,
                         top=0.90, bottom=0.09)

    # top: cut quality
    ax = axes2[0]
    ax.plot(p2_u0, p2_best, color=C_STABLE, linewidth=2.0,
            marker="o", markersize=5, label="best cut", zorder=3)
    ax.fill_between(p2_u0, p2_mean - p2_std, p2_mean + p2_std,
                    alpha=0.15, color=C_STABLE)
    ax.plot(p2_u0, p2_mean, color=C_STABLE, linewidth=1.2,
            linestyle="--", label="mean ± std (binarised)", zorder=2)
    ax.set_xscale("log")
    ax.invert_xaxis()
    _ax_style(ax,
              title=f"Phase 2 — cut quality vs $u_0$ ($u_0 \\leq \\hat{{u}}_{{0,bin}}$)",
              xlabel="$u_0$ (log scale, decreasing →)",
              ylabel="cut value")
    _vline_annotate(ax, u0_hat_bin,
                    f"$\\hat{{u}}_{{0,bin}}={u0_hat_bin:.4f}$",
                    C_U0_BIN, "--")
    _vline_annotate(ax, u0_best_cut,
                    f"best-cut $u_0={u0_best_cut:.4f}$ (cut={best_cut:.0f})",
                    C_BEST, ":", yrel=0.55)
    if args.known_opt:
        ax.axhline(args.known_opt, color=BLACK, linewidth=1.3,
                   linestyle=":", label=f"known opt = {args.known_opt:.0f}")
        ax2r = ax.twinx()
        ax2r.plot(p2_u0, p2_best / args.known_opt,
                  color=C_UNSTABLE, linewidth=1.4, linestyle="-.")
        ax2r.set_ylabel("approx. ratio", color=C_UNSTABLE, fontsize=10)
        ax2r.tick_params(axis="y", colors=C_UNSTABLE, labelsize=9)
        ax2r.set_ylim(0, 1.15)
        for sp in ax2r.spines.values():
            sp.set_edgecolor(BLACK); sp.set_linewidth(0.8)
    ax.legend(fontsize=9, loc="lower left")

    # bottom: binarisation fraction
    ax = axes2[1]
    ax.plot(p2_u0, p2_bin, color=C_STABLE, linewidth=2.0,
            marker="o", markersize=5, zorder=3)
    ax.fill_between(p2_u0, p2_bin, alpha=0.12, color=C_STABLE)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(-0.05, 1.12)
    _ax_style(ax,
              title="Phase 2 — binarisation fraction vs $u_0$",
              xlabel="$u_0$ (log scale, decreasing →)",
              ylabel="bin. fraction")
    _vline_annotate(ax, u0_hat_bin,
                    f"$\\hat{{u}}_{{0,bin}}={u0_hat_bin:.4f}$",
                    C_U0_BIN, "--")

    fig2.suptitle(
        f"HNN G-Set experiment | {stem} | $N={n}$ | "
        f"$W_{{\\rm tot}}={w_total:.0f}$ | "
        f"best cut found = {best_cut:.0f}"
        + (f" / opt = {args.known_opt:.0f}" if args.known_opt else ""),
        color=BLACK, fontsize=12, fontweight="bold")

    return fig1, fig2


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
def print_summary(args, n, w_total, u0_hat_bin, u0_theory,
                  p1_records, p2_records):
    best_p2 = max(p2_records, key=lambda r: r["best_cut"])
    print(f"{'='*65}")
    print(f" SUMMARY | {Path(args.graph).stem} | N={n} W_tot={w_total:.0f}")
    print(f"{'='*65}")
    print(f" û0_bin (empirical, Phase 1)   : {u0_hat_bin:.5f}")
    print(f" u0_bin (theory, |λ_min(W)|)   : {u0_theory:.5f}")
    print(f" Best cut at û0_bin            : {p2_records[0]['best_cut']:.2f}")
    print(f" Best cut in Phase 2 sweep     : {best_p2['best_cut']:.2f} "
          f"at u0={best_p2['u0']:.5f}")
    if args.known_opt:
        r1 = p2_records[0]["best_cut"] / args.known_opt
        r2 = best_p2["best_cut"]       / args.known_opt
        print(f" Approx ratio at û0_bin        : {r1:.4f}")
        print(f" Best approx ratio             : {r2:.4f}")
        print(f" Known optimum                 : {args.known_opt:.0f}")
    h = p2_records[0]["best_cut"] >= best_p2["best_cut"] - 1e-6
    print(f"\n Hypothesis (û0_bin gives best cut): "
          f"{'✓ HOLDS' if h else '✗ DOES NOT HOLD'}")
    print(f" Cut at û0_bin = {p2_records[0]['best_cut']:.2f} | "
          f"Best = {best_p2['best_cut']:.2f} at u0 = {best_p2['u0']:.5f}")
    print(f"{'='*65}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="G-Set Hopfield-Tank Max-Cut experiment: u0 sweep (RK45)")
    parser.add_argument("--graph",   required=True,
                        help="Path to G-Set benchmark file")
    parser.add_argument("--u0_start", type=float, default=2.0,
                        help="Phase-1 scan start (default: 2.0)")
    parser.add_argument("--u0_min",  type=float, default=0.001,
                        help="Phase-1 minimum u0 (default: 0.001)")
    parser.add_argument("--u0_steps_per_decade", type=int, default=5,
                        help="Log-grid steps per decade in Phase-1 (default: 5)")
    parser.add_argument("--n_u0_2",  type=int, default=15,
                        help="Phase-2 number of u0 points (default: 15)")
    parser.add_argument("--u0_2_factor", type=float, default=4.0,
                        help="Phase-2 lower = û0_bin / factor (default: 4.0)")
    parser.add_argument("--n_init",  type=int, default=10,
                        help="Trajectories per u0 (default: 10)")
    # RK45 parameters (replace Euler n_steps / timestep)
    parser.add_argument("--t_end",   type=float, default=30.0,
                        help="ODE integration horizon in dimensionless time "
                             "(default: 30.0 — RK45 takes large steps once "
                             "binarised, so this is cheap)")
    parser.add_argument("--rtol",    type=float, default=1e-4,
                        help="RK45 relative tolerance (default: 1e-4)")
    parser.add_argument("--atol",    type=float, default=1e-6,
                        help="RK45 absolute tolerance (default: 1e-6)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--known_opt", type=float, default=None,
                        help="Known optimum cut value (optional)")
    parser.add_argument("--save", action="store_true",
                        help="Save figures as PDF + PNG")
    args = parser.parse_args()

    # ── load graph ──────────────────────────────────────────────────────────
    print(f"\nLoading G-Set graph: {args.graph}")
    n, W, edges = parse_gset(args.graph)
    w_total = float(np.sum(W)) / 2.0
    print(f" N={n}  |E|={len(edges)}  W_total={w_total:.0f}")

    # ── analytical threshold ────────────────────────────────────────────────
    u0_theory = analytical_u0_bin(W)
    print(f" Theoretical u0_bin = |λ_min(W)| = {u0_theory:.5f}")
    print(f"   (origin unstable for u0 < u0_bin → all corners become attractors)")

    # ── fixed initial conditions (same u_init across all u0) ───────────────
    rng      = np.random.default_rng(args.seed)
    u0_inits = [rng.uniform(-0.5e-4, 0.5e-4, n)
                for _ in range(args.n_init)]
    print(f"\n {args.n_init} initial conditions drawn from "
          f"uniform(-5e-5, 5e-5) (seed={args.seed})")
    print(f" Same u_inits used across all u0 — isolates u0 effect from IC noise.")
    print(f" Integration: RK45  t_end={args.t_end}  "
          f"rtol={args.rtol:.0e}  atol={args.atol:.0e}")

    # ── Phase 1 ─────────────────────────────────────────────────────────────
    t_total = time.perf_counter()
    u0_hat_bin, p1_records = phase1_find_u0_bin(W, u0_inits, args)

    # ── Phase 2 ─────────────────────────────────────────────────────────────
    p2_records = phase2_quality_sweep(W, u0_inits, u0_hat_bin, args)

    print(f" Total wall time: {time.perf_counter() - t_total:.1f}s\n")

    # ── summary ─────────────────────────────────────────────────────────────
    print_summary(args, n, w_total, u0_hat_bin, u0_theory,
                  p1_records, p2_records)

    # ── figures ─────────────────────────────────────────────────────────────
    fig1, fig2 = make_figures(args, n, w_total, u0_hat_bin,
                               u0_theory, p1_records, p2_records)

    if args.save:
        stem = Path(args.graph).stem
        for tag, fig in [("phase1_scan", fig1), ("phase2_sweep", fig2)]:
            for ext in ("pdf", "png"):
                fname = f"hnn_gset_{stem}_{tag}.{ext}"
                fig.savefig(fname, bbox_inches="tight", dpi=150)
                print(f" Saved: {fname}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
