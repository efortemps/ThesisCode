#!/usr/bin/env python3
"""
gset_hopfield_lambda_experiment.py
──────────────────────────────────────────────────────────────────────────────
G-Set Max-Cut experiment using a Continuous Hopfield-Tank Network (HNN).

Gain convention: λ  (large λ = steep tanh = strong binarisation)
──────────────────────────────────────────────────────────────────────────────
ODE:  du_i/dt = −u_i − Σ_j W_ij tanh(λ u_j)

Binarisation threshold:
  H(0) = W + (1/λ) I  →  origin unstable iff λ > λ_bin = 1/|λ_min(W)|

Procedure
─────────
Phase 1 — find λ̂_bin empirically
  Sweep λ UPWARD (log-spaced) from lambda_start to lambda_max.
  Stop at the first λ where ALL trajectories binarise.

Phase 2 — quality sweep above λ̂_bin
  Dense sweep of λ in [λ̂_bin, λ̂_bin × lambda_2_factor].
  Record best and mean cut at each point.

Integration: RK45 (scipy solve_ivp, adaptive step-size).

Usage
─────
python gset_hopfield_lambda_experiment.py --graph G1.txt [options]

--graph                PATH    G-Set file (required)
--lambda_start         FLOAT   Phase-1 start λ              (default: 0.5)
--lambda_max           FLOAT   Phase-1 upper bound λ        (default: 1000.0)
--lambda_steps_per_decade INT  log-grid density Phase-1     (default: 5)
--n_lambda_2           INT     Phase-2 number of λ points   (default: 15)
--lambda_2_factor      FLOAT   Phase-2 upper = λ̂_bin×factor (default: 4.0)
--n_init               INT     trajectories per λ            (default: 10)
--t_end                FLOAT   integration horizon T         (default: 120.0)
--rtol                 FLOAT   RK45 relative tolerance       (default: 1e-4)
--atol                 FLOAT   RK45 absolute tolerance       (default: 1e-6)
--seed                 INT     RNG seed                      (default: 42)
--known_opt            FLOAT   known optimum cut (optional)
--save                         save figures as PDF + PNG
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ── plotting style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          16,
    "axes.titlesize":     18,
    "axes.labelsize":     16,
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
C_LAM_BIN = "#c44e52"   # red   — λ̂_bin marker
C_BEST    = "#55a868"   # green — best-cut marker


# ═══════════════════════════════════════════════════════════════════════════════
# G-Set parser
# ═══════════════════════════════════════════════════════════════════════════════
def parse_gset(path: str):
    """
    Parse a G-Set benchmark file.

    Format::
        N  M          ← nodes, edges (first non-comment line)
        u  v  w       ← edges (1-indexed, weight w)
        ...

    Returns
    -------
    n     : int
    W     : (N, N) symmetric weight matrix  (W_ij = |w|)
    edges : list of (u, v, w) tuples (0-indexed)
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

    return n, W, edges_raw


# ═══════════════════════════════════════════════════════════════════════════════
# ODE helpers  (λ convention)
# ═══════════════════════════════════════════════════════════════════════════════
def _hnn_rhs(t: float, u: np.ndarray, W: np.ndarray, lam: float) -> np.ndarray:
    """du/dt = −u − W tanh(λ u)  (τ = 1)"""
    return -u - W @ np.tanh(lam * u)


def _binary_cut(u: np.ndarray, W: np.ndarray, lam: float) -> float:
    """Cut value after hard binarisation σ = sign(tanh(λ u))."""
    sigma = np.sign(np.tanh(lam * u))
    sigma[sigma == 0] = 1.0
    return 0.25 * float(np.sum(W * (1.0 - sigma[:, None] * sigma[None, :])))


def _is_binarised(u: np.ndarray, lam: float, tol: float = 0.05) -> bool:
    """True iff ‖s(T) − sgn(s(T))‖∞ < tol,  where s = tanh(λ u)."""
    s = np.tanh(lam * u)
    return bool(np.all(np.abs(s - np.sign(s)) < tol))


def _partition(u: np.ndarray, lam: float) -> np.ndarray:
    """Hard spin assignment σ ∈ {−1, +1}^N."""
    s = np.sign(np.tanh(lam * u))
    s[s == 0] = 1.0
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# Single-λ run  (RK45)
# ═══════════════════════════════════════════════════════════════════════════════
def run_lam(W: np.ndarray, lam: float, u_inits: list,
            t_end: float, rtol: float, atol: float,
            seed: int, bin_tol: float = 0.05) -> dict:
    """
    Run n_init Hopfield trajectories at a given λ using RK45.

    Parameters
    ----------
    W        : (N, N) symmetric weight matrix
    lam      : gain parameter λ
    u_inits  : list of (N,) initial membrane-potential vectors
    t_end    : integration horizon T
    rtol/atol: RK45 tolerances
    seed     : kept for API compatibility (ICs passed explicitly)
    bin_tol  : binarisation tolerance (‖s−sgn(s)‖∞ < bin_tol)

    Returns dict with keys:
        lam, best_cut, mean_cut, std_cut, best_partition,
        bin_fraction, all_cuts, all_bin, t_elapsed, n_func_evals
    """
    t0 = time.perf_counter()

    cuts        = []
    bin_flags   = []
    terminal_us = []
    total_evals = 0

    for u_init in u_inits:
        sol = solve_ivp(
            _hnn_rhs,
            t_span=(0.0, t_end),
            y0=u_init,
            method="RK45",
            args=(W, lam),
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )
        u_final = sol.y[:, -1]
        terminal_us.append(u_final)
        total_evals += sol.nfev

        cuts.append(_binary_cut(u_final, W, lam))
        bin_flags.append(_is_binarised(u_final, lam, bin_tol))

    cuts      = np.array(cuts,      dtype=float)
    bin_flags = np.array(bin_flags, dtype=bool)

    best_idx  = int(np.argmax(cuts))
    bin_cuts  = cuts[bin_flags]
    partition = _partition(terminal_us[best_idx], lam)

    return dict(
        lam           = lam,
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
# Analytical λ_bin
# ═══════════════════════════════════════════════════════════════════════════════
def analytical_lambda_bin(W: np.ndarray) -> float:
    """
    Theoretical binarisation threshold from the Hessian at the origin:

        H(0) = W + (1/λ) I

    Origin is UNSTABLE (binarisation) iff λ > λ_bin = 1/|λ_min(W)|.

    Returns 0.0 if λ_min(W) ≥ 0 (origin always stable).
    """
    lmin = float(np.linalg.eigvalsh(W).min())
    if lmin >= 0:
        return 0.0
    return 1.0 / (-lmin)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — upward scan to find λ̂_bin
# ═══════════════════════════════════════════════════════════════════════════════
def phase1_find_lambda_bin(W, u_inits, args):
    """
    Sweep λ UPWARD from lambda_start to lambda_max (log-spaced).
    Stop at the first λ where ALL trajectories binarise.

    Returns
    -------
    lam_hat_bin : float
    records     : list of run_lam dicts
    """
    n_pts = int(np.ceil(
        np.log10(args.lambda_max / args.lambda_start)
        * args.lambda_steps_per_decade)) + 1
    lam_grid = np.logspace(
        np.log10(args.lambda_start),
        np.log10(args.lambda_max),
        num=max(n_pts, 10))

    n_init      = len(u_inits)
    records     = []
    lam_hat_bin = None

    print(f"\n{'='*65}")
    print(f" PHASE 1 — empirical λ̂_bin search (upward sweep, RK45)")
    print(f" λ grid: [{args.lambda_start:.4f}, {args.lambda_max:.2f}]"
          f"  ({len(lam_grid)} pts, log-spaced)")
    print(f" n_init={n_init}  t_end={args.t_end}"
          f"  rtol={args.rtol:.0e}  atol={args.atol:.0e}")
    print(f"{'='*65}")
    hdr = (f"  {'lambda':>10} {'bin_frac':>9} {'best_cut':>10}"
           f" {'mean_cut':>10} {'nfev':>7} {'t(s)':>6}")
    print(hdr)
    print("  " + "─" * (10 + 9 + 10 + 10 + 7 + 6 + 8))

    for lam in lam_grid:
        rec = run_lam(W, lam, u_inits,
                      args.t_end, args.rtol, args.atol,
                      args.seed)
        records.append(rec)

        flag = "  ← λ̂_bin ✓" if (rec["bin_fraction"] == 1.0
                                    and lam_hat_bin is None) else ""
        print(f"  {lam:>10.5f} {rec['bin_fraction']:>9.3f}"
              f" {rec['best_cut']:>10.2f} {rec['mean_cut']:>10.2f}"
              f" {rec['n_func_evals']:>7}"
              f" {rec['t_elapsed']:>6.1f}s{flag}")

        if rec["bin_fraction"] == 1.0 and lam_hat_bin is None:
            lam_hat_bin = lam
            break

    if lam_hat_bin is None:
        print(f"\n  [warn] No full binarisation found up to λ={args.lambda_max:.2f}.")
        print(f"  Using the λ with highest bin_fraction as fallback.")
        best_rec    = max(records, key=lambda r: (r["bin_fraction"], r["best_cut"]))
        lam_hat_bin = best_rec["lam"]

    print(f"\n  → λ̂_bin = {lam_hat_bin:.5f}  (first fully-binarising λ)\n")
    return lam_hat_bin, records


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — quality sweep above λ̂_bin
# ═══════════════════════════════════════════════════════════════════════════════
def phase2_quality_sweep(W, u_inits, lam_hat_bin, args):
    """
    Dense sweep of λ in [λ̂_bin, λ̂_bin × lambda_2_factor] (log-spaced upward).
    At each λ run the same n_init trajectories and record cut quality.

    Returns list of run_lam dicts.
    """
    lam_hi  = lam_hat_bin * args.lambda_2_factor
    lam_arr = np.logspace(
        np.log10(lam_hat_bin),
        np.log10(max(lam_hi, lam_hat_bin * 1.001)),
        num=args.n_lambda_2)

    print(f"{'='*65}")
    print(f" PHASE 2 — quality sweep  λ ∈ [{lam_hat_bin:.5f}, {lam_hi:.5f}]"
          f"  ({args.n_lambda_2} pts, RK45)")
    print(f" n_init={len(u_inits)}  t_end={args.t_end}"
          f"  rtol={args.rtol:.0e}  atol={args.atol:.0e}")
    print(f"{'='*65}")
    hdr = (f"  {'lambda':>10} {'bin_frac':>9} {'best_cut':>10}"
           f" {'mean_cut':>10} {'std_cut':>8} {'nfev':>7} {'t(s)':>6}")
    print(hdr)
    print("  " + "─" * (10 + 9 + 10 + 10 + 8 + 7 + 6 + 8))

    records = []
    for lam in lam_arr:
        rec = run_lam(W, lam, u_inits,
                      args.t_end, args.rtol, args.atol,
                      args.seed)
        records.append(rec)
        print(f"  {lam:>10.5f} {rec['bin_fraction']:>9.3f}"
              f" {rec['best_cut']:>10.2f} {rec['mean_cut']:>10.2f}"
              f" {rec['std_cut']:>8.3f} {rec['n_func_evals']:>7}"
              f" {rec['t_elapsed']:>6.1f}s")

    best = max(records, key=lambda r: r["best_cut"])
    print(f"\n  → Best cut in Phase 2: {best['best_cut']:.2f}"
          f"  at λ = {best['lam']:.5f}")
    if args.known_opt:
        gap = 100.0 * (args.known_opt - best["best_cut"]) / args.known_opt
        print(f"     Approx ratio: {best['best_cut'] / args.known_opt:.4f}"
              f"  (gap = {gap:.2f}%  vs known opt = {args.known_opt:.0f})")
    print()
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# Axis styling helper
# ═══════════════════════════════════════════════════════════════════════════════
def _ax_style(ax, title="", xlabel="", ylabel="", titlesize=18):
    ax.set_facecolor(WHITE)
    ax.tick_params(colors=BLACK, labelsize=14)
    for sp in ax.spines.values():
        sp.set_edgecolor(BLACK)
        sp.set_linewidth(0.8)
    ax.grid(True, color=LIGHT, linewidth=0.6, zorder=0)
    if title:  ax.set_title( title,  color=BLACK, fontsize=titlesize, pad=5)
    if xlabel: ax.set_xlabel(xlabel, color=BLACK, fontsize=16)
    if ylabel: ax.set_ylabel(ylabel, color=BLACK, fontsize=16)


def _vline_annotate(ax, x, label, color, ls, yrel=0.05):
    ax.axvline(x, color=color, linewidth=2.0, linestyle=ls, zorder=5)
    ylims = ax.get_ylim()
    y = ylims[0] + yrel * (ylims[1] - ylims[0])
    ax.text(x, y, f" {label}", color=color, fontsize=12,
            ha="left", va="bottom", rotation=90)


# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════
def make_figures(args, n, w_total, lam_hat_bin, lam_theory,
                 p1_records, p2_records):
    """
    Figure 1 — Phase 1 scan  (binarisation fraction + cut quality vs λ)
    Figure 2 — Phase 2 quality sweep (cut quality + binarisation vs λ)
    """
    stem = Path(args.graph).stem

    def _arrays(records):
        lam_arr  = np.array([r["lam"]          for r in records])
        bin_arr  = np.array([r["bin_fraction"]  for r in records])
        best_arr = np.array([r["best_cut"]      for r in records])
        mean_arr = np.array([r["mean_cut"]      for r in records])
        std_arr  = np.array([r["std_cut"]       for r in records])
        return lam_arr, bin_arr, best_arr, mean_arr, std_arr

    p1_lam, p1_bin, p1_best, p1_mean, _       = _arrays(p1_records)
    p2_lam, p2_bin, p2_best, p2_mean, p2_std  = _arrays(p2_records)

    best_p2_idx  = int(np.argmax(p2_best))
    lam_best_cut = p2_lam[best_p2_idx]
    best_cut     = p2_best[best_p2_idx]

    # ── Figure 1 ──────────────────────────────────────────────────────────
    fig1, axes1 = plt.subplots(2, 1, figsize=(13, 8), facecolor=WHITE)
    fig1.subplots_adjust(hspace=0.42, left=0.09, right=0.97,
                         top=0.90, bottom=0.09)

    # top: binarisation fraction vs λ
    ax = axes1[0]
    ax.plot(p1_lam, p1_bin, color=C_STABLE, linewidth=2.0,
            marker="o", markersize=5, zorder=3, label="bin. fraction")
    ax.fill_between(p1_lam, p1_bin, alpha=0.12, color=C_STABLE)
    ax.axhline(1.0, color=GRAY, linewidth=0.9, linestyle="--")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.12)
    _ax_style(ax,
              title=(f"Binarisation fraction vs $\\lambda$"),
              xlabel=r"$\lambda$",
              ylabel="fraction of fully binarised trajectories")
    _vline_annotate(ax, lam_hat_bin,
                    f"$\\hat{{\\lambda}}_{{\\rm bin}}={lam_hat_bin:.4f}$",
                    C_LAM_BIN, "--")
    ax.legend(fontsize=13, loc="upper right")

    # bottom: cut quality vs λ
    ax = axes1[1]
    ax.plot(p1_lam, p1_best, color=C_STABLE, linewidth=2.0,
            marker="o", markersize=5, label="best cut", zorder=3)
    ax.plot(p1_lam, p1_mean, color=C_UNSTABLE, linewidth=1.4,
            marker="s", markersize=4, linestyle="--",
            label="mean cut (binarised)", zorder=2)
    ax.set_xscale("log")
    _ax_style(ax,
              title=r"Cut quality vs $\lambda$",
              xlabel=r"$\lambda$",
              ylabel="cut value")
    _vline_annotate(ax, lam_hat_bin,
                    f"$\\hat{{\\lambda}}_{{\\rm bin}}={lam_hat_bin:.4f}$",
                    C_LAM_BIN, "--")
    if args.known_opt:
        ax.axhline(args.known_opt, color=BLACK, linewidth=1.3,
                   linestyle=":", label=f"known opt = {args.known_opt:.0f}")
    ax.legend(fontsize=13, loc="upper right")

    fig1.suptitle(
        f"HNN G-Set experiment  |  {stem}  |  $N={n}$  |"
        f"  $W_{{\\rm tot}}={w_total:.0f}$",
        color=BLACK, fontsize=20, fontweight="bold")

    # ── Figure 2 ──────────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(2, 1, figsize=(13, 8), facecolor=WHITE)
    fig2.subplots_adjust(hspace=0.42, left=0.09, right=0.97,
                         top=0.90, bottom=0.09)

    # top: cut quality
    ax = axes2[0]
    ax.plot(p2_lam, p2_best, color=C_STABLE, linewidth=2.0,
            marker="o", markersize=5, label="best cut", zorder=3)
    ax.fill_between(p2_lam, p2_mean - p2_std, p2_mean + p2_std,
                    alpha=0.15, color=C_STABLE)
    ax.plot(p2_lam, p2_mean, color=C_STABLE, linewidth=1.2,
            linestyle="--", label=r"mean $\pm$ std (binarised)", zorder=2)
    ax.set_xscale("log")
    _ax_style(ax,
              title=(r"Phase 2 — cut quality vs $\lambda$"
                     r"  ($\lambda \geq \hat{\lambda}_{\rm bin}$)"),
              xlabel=r"$\lambda$  (log scale, increasing $\rightarrow$)",
              ylabel="cut value")
    _vline_annotate(ax, lam_hat_bin,
                    f"$\\hat{{\\lambda}}_{{\\rm bin}}={lam_hat_bin:.4f}$",
                    C_LAM_BIN, "--")
    _vline_annotate(ax, lam_best_cut,
                    f"best-cut $\\lambda={lam_best_cut:.4f}$  (cut={best_cut:.0f})",
                    C_BEST, ":", yrel=0.55)
    if args.known_opt:
        ax.axhline(args.known_opt, color=BLACK, linewidth=1.3,
                   linestyle=":", label=f"known opt = {args.known_opt:.0f}")
        ax2r = ax.twinx()
        ax2r.plot(p2_lam, p2_best / args.known_opt,
                  color=C_UNSTABLE, linewidth=1.4, linestyle="-.")
        ax2r.set_ylabel("approx. ratio", color=C_UNSTABLE, fontsize=14)
        ax2r.tick_params(axis="y", colors=C_UNSTABLE, labelsize=13)
        ax2r.set_ylim(0, 1.15)
        for sp in ax2r.spines.values():
            sp.set_edgecolor(BLACK)
            sp.set_linewidth(0.8)
    ax.legend(fontsize=13, loc="lower right")

    # bottom: binarisation fraction
    ax = axes2[1]
    ax.plot(p2_lam, p2_bin, color=C_STABLE, linewidth=2.0,
            marker="o", markersize=5, zorder=3)
    ax.fill_between(p2_lam, p2_bin, alpha=0.12, color=C_STABLE)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.12)
    _ax_style(ax,
              title=r"Phase 2 — binarisation fraction vs $\lambda$",
              xlabel=r"$\lambda$  (log scale, increasing $\rightarrow$)",
              ylabel="bin. fraction")
    _vline_annotate(ax, lam_hat_bin,
                    f"$\\hat{{\\lambda}}_{{\\rm bin}}={lam_hat_bin:.4f}$",
                    C_LAM_BIN, "--")

    fig2.suptitle(
        f"HNN G-Set experiment  |  {stem}  |  $N={n}$  |"
        f"  $W_{{\\rm tot}}={w_total:.0f}$  |"
        f"  best cut = {best_cut:.0f}"
        + (f"  /  opt = {args.known_opt:.0f}" if args.known_opt else ""),
        color=BLACK, fontsize=20, fontweight="bold")

    return fig1, fig2


# ═══════════════════════════════════════════════════════════════════════════════
# Console summary
# ═══════════════════════════════════════════════════════════════════════════════
def print_summary(args, n, w_total, lam_hat_bin, lam_theory,
                  p1_records, p2_records):
    best_p2 = max(p2_records, key=lambda r: r["best_cut"])
    print(f"{'='*65}")
    print(f" SUMMARY  |  {Path(args.graph).stem}  |  N={n}  W_tot={w_total:.0f}")
    print(f"{'='*65}")
    print(f"  λ̂_bin  (empirical, Phase 1)    : {lam_hat_bin:.5f}")
    print(f"  λ_bin  (theory, 1/|λ_min(W)|) : {lam_theory:.5f}")
    print(f"  Best cut at λ̂_bin             : {p2_records[0]['best_cut']:.2f}")
    print(f"  Best cut in Phase 2 sweep     : {best_p2['best_cut']:.2f}"
          f"  at λ = {best_p2['lam']:.5f}")
    if args.known_opt:
        r1 = p2_records[0]["best_cut"] / args.known_opt
        r2 = best_p2["best_cut"]       / args.known_opt
        print(f"  Approx ratio at λ̂_bin        : {r1:.4f}")
        print(f"  Best approx ratio             : {r2:.4f}")
        print(f"  Known optimum                 : {args.known_opt:.0f}")
    holds = p2_records[0]["best_cut"] >= best_p2["best_cut"] - 1e-6
    print(f"\n  Hypothesis (λ̂_bin gives best cut): "
          f"{'✓ HOLDS' if holds else '✗ DOES NOT HOLD'}")
    print(f"  Cut at λ̂_bin = {p2_records[0]['best_cut']:.2f}  |"
          f"  Best = {best_p2['best_cut']:.2f}  at λ = {best_p2['lam']:.5f}")
    print(f"{'='*65}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="G-Set Hopfield-Tank Max-Cut experiment — λ convention (RK45)")
    parser.add_argument("--graph",    required=True,
                        help="Path to G-Set benchmark file")
    parser.add_argument("--lambda_start", type=float, default=0.5,
                        help="Phase-1 λ scan start (default: 0.5)")
    parser.add_argument("--lambda_max",   type=float, default=1000.0,
                        help="Phase-1 λ upper bound (default: 1000.0)")
    parser.add_argument("--lambda_steps_per_decade", type=int, default=5,
                        help="Log-grid steps per decade in Phase-1 (default: 5)")
    parser.add_argument("--n_lambda_2",    type=int,   default=15,
                        help="Phase-2 number of λ points (default: 15)")
    parser.add_argument("--lambda_2_factor", type=float, default=4.0,
                        help="Phase-2 upper = λ̂_bin × factor (default: 4.0)")
    parser.add_argument("--n_init", type=int, default=10,
                        help="Trajectories per λ (default: 10)")
    parser.add_argument("--t_end",  type=float, default=120.0,
                        help="Integration horizon T (default: 120.0)")
    parser.add_argument("--rtol",   type=float, default=1e-4,
                        help="RK45 relative tolerance (default: 1e-4)")
    parser.add_argument("--atol",   type=float, default=1e-6,
                        help="RK45 absolute tolerance (default: 1e-6)")
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--known_opt", type=float, default=None,
                        help="Known optimum cut value (optional)")
    parser.add_argument("--save",   action="store_true",
                        help="Save figures as PDF + PNG")
    args = parser.parse_args()

    # ── load graph ──────────────────────────────────────────────────────
    print(f"\nLoading G-Set graph: {args.graph}")
    n, W, edges = parse_gset(args.graph)
    w_total = float(np.sum(W)) / 2.0
    print(f"  N={n}  |E|={len(edges)}  W_total={w_total:.0f}")

    # ── analytical threshold ────────────────────────────────────────────
    lam_theory = analytical_lambda_bin(W)
    print(f"  Theoretical λ_bin = 1/|λ_min(W)| = {lam_theory:.5f}")
    print(f"  (origin unstable for λ > λ_bin → all corners become attractors)")

    # ── fixed initial conditions  U(-1, 1) ─────────────────────────────
    rng     = np.random.default_rng(args.seed)
    u_inits = [rng.uniform(-1.0, 1.0, n) for _ in range(args.n_init)]
    print(f"\n  {args.n_init} initial conditions drawn from U(-1, 1)"
          f"  (seed={args.seed})")
    print(f"  Same ICs reused at every λ — isolates effect of gain from IC noise.")
    print(f"  Integration: RK45  t_end={args.t_end}"
          f"  rtol={args.rtol:.0e}  atol={args.atol:.0e}")

    # ── Phase 1 ─────────────────────────────────────────────────────────
    t_total = time.perf_counter()
    lam_hat_bin, p1_records = phase1_find_lambda_bin(W, u_inits, args)

    # ── Phase 2 ─────────────────────────────────────────────────────────
    p2_records = phase2_quality_sweep(W, u_inits, lam_hat_bin, args)

    print(f"  Total wall time: {time.perf_counter() - t_total:.1f}s\n")

    # ── summary ─────────────────────────────────────────────────────────
    print_summary(args, n, w_total, lam_hat_bin, lam_theory,
                  p1_records, p2_records)

    # ── figures ─────────────────────────────────────────────────────────
    fig1, fig2 = make_figures(args, n, w_total, lam_hat_bin,
                               lam_theory, p1_records, p2_records)

    if args.save:
        stem = Path(args.graph).stem
        for tag, fig in [("phase1_scan", fig1), ("phase2_sweep", fig2)]:
            for ext in ("pdf", "png"):
                fname = f"hnn_gset_{stem}_{tag}.{ext}"
                fig.savefig(fname, bbox_inches="tight", dpi=150)
                print(f"  Saved: {fname}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
