#!/usr/bin/env python3
"""
gset_oim_experiment.py
──────────────────────────────────────────────────────────────────────────────
Large-graph OIM Max-Cut experiment on G-Set benchmarks.

Hypothesis under test
─────────────────────
  For small graphs (N ≤ 10) the first μ at which ALL trajectories binarise
  (call it μ̂_bin) tends to yield the best reachable cut value.
  Does this hold for larger G-Set graphs?

Procedure  (avoids the 2^N equilibrium enumeration)
────────────────────────────────────────────────────
  Phase 1 — find μ̂_bin empirically
    Sweep μ over a coarse grid starting from a small value.
    At each μ, run n_init ODE trajectories from uniform(−π, π).
    Stop at the first μ where EVERY trajectory binarises within t_end.
    → μ̂_bin  (empirical lower bound on the true μ_bin)

  Phase 2 — quality sweep above μ̂_bin
    Sweep a denser grid of μ values ≥ μ̂_bin.
    At each μ, run n_init trajectories and record the best cut found.
    Compare cut quality across the whole range.

Output
──────
  • Console report: per-μ best/mean cut, binarisation fraction, timing
  • Figure 1 — Phase 1 scan: binarisation fraction vs μ with μ̂_bin marker
  • Figure 2 — Phase 2 quality sweep: best/mean cut vs μ, reference lines
  • Figure 3 — Best trajectory at μ̂_bin and at best-cut μ side by side

G-Set file format  (standard Rudy/Biq-Mac format)
──────────────────────────────────────────────────
  N  M          ← number of nodes, number of edges
  u  v  w       ← edges (1-indexed; weight w, usually ±1)
  ...

Usage
─────
  python gset_oim_experiment.py --graph G1.txt [options]

  --graph       PATH    G-Set file (required)
  --mu_start    FLOAT   start of Phase-1 coarse scan (default: 0.01)
  --mu_step     FLOAT   step size for Phase-1 scan   (default: 0.05)
  --mu_max_scan FLOAT   upper limit of Phase-1 scan  (default: 10.0)
  --n_mu2       INT     number of μ points in Phase-2 sweep (default: 30)
  --mu2_factor  FLOAT   Phase-2 upper bound = μ̂_bin × mu2_factor (default: 4.0)
  --n_init      INT     trajectories per μ            (default: 20)
  --t_end       FLOAT   ODE integration horizon       (default: 200)
  --n_points    INT     time-grid points per traj.    (default: 600)
  --rtol        FLOAT   RK45 relative tolerance       (default: 1e-6)
  --atol        FLOAT   RK45 absolute tolerance       (default: 1e-6)
  --seed        INT     RNG seed                      (default: 42)
  --known_opt   FLOAT   known optimum cut (for gap reporting, optional)
  --save        FLAG    save figures as PDF + PNG

──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from OIM_Experiment.src.OIM_mu import OIMMaxCut

# ── TikZ-like style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
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

WHITE      = "#ffffff"
BLACK      = "#000000"
GRAY       = "#b0b0b0"
LIGHT      = "#e6e6e6"
C_STABLE   = "#4C72B0"    # blue  — binarised / good cut
C_UNSTABLE = "#DD8452"    # orange — not binarised
C_MU_BIN   = "#c44e52"    # red   — μ̂_bin marker
C_BEST     = "#55a868"    # green — best-cut marker
C_MU_LINE  = "#ffb74d"    # amber — current μ


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
# G-Set parser  (Rudy/Biq-Mac format: first line = N M, then u v w)
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
    W      : (N, N) symmetric weight matrix  (W_ij = |w|, signs handled by OIM)
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

    # Build weight matrix.
    # G-Set uses w = 1 for max-cut (sometimes w = -1 for min-cut formulations).
    # The OIM maximises cut, so we use |w| as coupling weight.
    W = np.zeros((n, n), dtype=float)
    for u, v, w in edges_raw:
        W[u, v] += abs(w)
        W[v, u] += abs(w)

    edges_0idx = [(u, v, w) for u, v, w in edges_raw]
    return n, W, edges_0idx


# ═══════════════════════════════════════════════════════════════════════════════
# Single-μ run  (the core experiment unit)
# ═══════════════════════════════════════════════════════════════════════════════
def run_mu(W: np.ndarray, mu: float, phi0s: list,
           t_end: float, n_points: int,
           rtol: float, atol: float, seed: int,
           bin_tol: float = 0.05) -> dict:
    """
    Simulate n_init trajectories at a given μ and return aggregate statistics.

    Returns dict with keys:
      mu            : float — μ used
      best_cut      : float — best cut over all trajectories
      mean_cut      : float — mean cut over binarised trajectories (or all)
      std_cut       : float
      best_bits     : tuple — spin assignment achieving best_cut
      best_sol      : scipy solution achieving best_cut
      bin_fraction  : float — fraction of trajectories that fully binarised
      all_cuts      : list of float
      all_bin       : list of bool
      t_elapsed     : float — wall time in seconds
    """
    t0  = time.perf_counter()
    oim = OIMMaxCut(W, mu=mu, seed=seed)
    n   = W.shape[0]

    # Run all trajectories
    sols = oim.simulate_many(phi0s,
                             t_span=(0., t_end),
                             n_points=n_points,
                             rtol=rtol, atol=atol)

    cuts     = []
    bin_mask = []
    for sol in sols:
        theta = sol.y[:, -1]
        sigma = np.sign(np.cos(theta))
        sigma[sigma == 0] = 1.0
        cut      = 0.25 * float(np.sum(W * (1.0 - sigma[:, None] * sigma[None, :])))
        residual = float(np.max(np.abs(np.sin(theta))))
        cuts.append(cut)
        bin_mask.append(residual < bin_tol)

    cuts     = np.array(cuts)
    bin_mask = np.array(bin_mask)
    best_idx = int(np.argmax(cuts))

    # spin assignment for best trajectory
    theta_best = sols[best_idx].y[:, -1]
    sigma_best = np.sign(np.cos(theta_best))
    sigma_best[sigma_best == 0] = 1.0
    bits_best = tuple(0 if s > 0 else 1 for s in sigma_best)

    bin_cuts = cuts[bin_mask]

    return dict(
        mu           = mu,
        best_cut     = float(cuts.max()),
        mean_cut     = float(bin_cuts.mean()) if len(bin_cuts) else float(cuts.mean()),
        std_cut      = float(bin_cuts.std())  if len(bin_cuts) else float(cuts.std()),
        best_bits    = bits_best,
        best_sol     = sols[best_idx],
        bin_fraction = float(bin_mask.mean()),
        all_cuts     = cuts.tolist(),
        all_bin      = bin_mask.tolist(),
        t_elapsed    = time.perf_counter() - t0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — empirical μ̂_bin search
# ═══════════════════════════════════════════════════════════════════════════════
def phase1_find_mu_bin(W, phi0s, args, bar_w=38):
    """
    Coarse upward sweep: stop at first μ where bin_fraction == 1.0.

    Returns
    -------
    mu_hat_bin  : float   — first fully-binarising μ found
    scan_records: list    — one dict per μ tried (from run_mu)
    """
    n_init   = len(phi0s)
    mu       = args.mu_start
    records  = []
    mu_hat_bin = None

    print(f"\n{'='*65}")
    print(f"  PHASE 1 — empirical μ̂_bin search")
    print(f"  μ grid: [{args.mu_start:.4f}, {args.mu_max_scan:.4f}]  "
          f"step={args.mu_step:.4f}")
    print(f"  n_init={n_init}  t_end={args.t_end}  "
          f"bin_tol=0.05")
    print(f"{'='*65}")
    hdr = (f"  {'μ':>8}  {'bin_frac':>9}  {'best_cut':>10}  "
           f"{'mean_cut':>10}  {'t(s)':>6}")
    print(hdr)
    print("  " + "─" * (8 + 9 + 10 + 10 + 6 + 12))

    while mu <= args.mu_max_scan + 1e-9:
        rec = run_mu(W, mu, phi0s,
                     args.t_end, args.n_points,
                     args.rtol, args.atol, args.seed)
        records.append(rec)

        flag = "  ← μ̂_bin ✓" if rec["bin_fraction"] == 1.0 and mu_hat_bin is None else ""
        print(f"  {mu:>8.4f}  {rec['bin_fraction']:>9.3f}  "
              f"{rec['best_cut']:>10.2f}  {rec['mean_cut']:>10.2f}  "
              f"{rec['t_elapsed']:>6.1f}s{flag}")

        if rec["bin_fraction"] == 1.0 and mu_hat_bin is None:
            mu_hat_bin = mu
            break   # stop Phase 1 as soon as we find full binarisation

        mu = round(mu + args.mu_step, 10)   # avoid float drift

    if mu_hat_bin is None:
        print(f"\n  [warn] No full binarisation found up to μ={args.mu_max_scan:.4f}.")
        print(f"  Using the μ with highest bin_fraction as a fallback.")
        best_rec   = max(records, key=lambda r: (r["bin_fraction"], r["best_cut"]))
        mu_hat_bin = best_rec["mu"]

    print(f"\n  → μ̂_bin = {mu_hat_bin:.4f}  "
          f"(empirical first fully-binarising μ)\n")
    return mu_hat_bin, records


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — quality sweep ≥ μ̂_bin
# ═══════════════════════════════════════════════════════════════════════════════
def phase2_quality_sweep(W, phi0s, mu_hat_bin, args):
    """
    Dense sweep of μ values ≥ μ̂_bin.  Records best and mean cut at each point.

    Returns list of dicts (from run_mu) for each μ in the sweep.
    """
    mu_hi  = mu_hat_bin * args.mu2_factor
    mu_arr = np.linspace(mu_hat_bin, mu_hi, args.n_mu2)

    print(f"{'='*65}")
    print(f"  PHASE 2 — quality sweep  μ ∈ [{mu_hat_bin:.4f}, {mu_hi:.4f}]"
          f"  ({args.n_mu2} steps)")
    print(f"  n_init={len(phi0s)}  t_end={args.t_end}")
    print(f"{'='*65}")
    hdr = (f"  {'μ':>8}  {'bin_frac':>9}  {'best_cut':>10}  "
           f"{'mean_cut':>10}  {'std_cut':>8}  {'t(s)':>6}")
    print(hdr)
    print("  " + "─" * (8 + 9 + 10 + 10 + 8 + 6 + 12))

    records = []
    for mu in mu_arr:
        rec = run_mu(W, mu, phi0s,
                     args.t_end, args.n_points,
                     args.rtol, args.atol, args.seed)
        records.append(rec)
        print(f"  {mu:>8.4f}  {rec['bin_fraction']:>9.3f}  "
              f"{rec['best_cut']:>10.2f}  {rec['mean_cut']:>10.2f}  "
              f"{rec['std_cut']:>8.3f}  {rec['t_elapsed']:>6.1f}s")

    best_overall = max(records, key=lambda r: r["best_cut"])
    print(f"\n  → Best cut found in Phase 2: {best_overall['best_cut']:.2f} "
          f"at μ = {best_overall['mu']:.4f}")
    if args.known_opt:
        gap = 100.0 * (args.known_opt - best_overall["best_cut"]) / args.known_opt
        print(f"  → Approximation ratio: "
              f"{best_overall['best_cut']/args.known_opt:.4f}  "
              f"(gap = {gap:.2f}%  vs known opt = {args.known_opt:.0f})")
    print()
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════════
def make_figures(args, n, w_total,
                 mu_hat_bin,
                 p1_records, p2_records):
    """
    Three figures.

    Figure 1 — Phase 1 scan
    Figure 2 — Phase 2 quality sweep
    Figure 3 — Best trajectory at μ̂_bin vs best-cut μ
    """
    p1_mu   = np.array([r["mu"]          for r in p1_records])
    p1_bin  = np.array([r["bin_fraction"] for r in p1_records])
    p1_best = np.array([r["best_cut"]    for r in p1_records])
    p1_mean = np.array([r["mean_cut"]    for r in p1_records])

    p2_mu   = np.array([r["mu"]          for r in p2_records])
    p2_bin  = np.array([r["bin_fraction"] for r in p2_records])
    p2_best = np.array([r["best_cut"]    for r in p2_records])
    p2_mean = np.array([r["mean_cut"]    for r in p2_records])
    p2_std  = np.array([r["std_cut"]     for r in p2_records])

    best_p2_idx = int(np.argmax(p2_best))
    mu_best_cut = p2_mu[best_p2_idx]
    best_cut    = p2_best[best_p2_idx]

    stem = Path(args.graph).stem

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 1 — Phase 1 scan
    # ══════════════════════════════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(2, 1, figsize=(13, 8), facecolor=WHITE)
    fig1.subplots_adjust(hspace=0.42, left=0.09, right=0.97,
                         top=0.90, bottom=0.09)

    # top: binarisation fraction vs μ
    ax = axes1[0]
    ax.plot(p1_mu, p1_bin, color=C_STABLE, linewidth=2.0,
            marker="o", markersize=5, zorder=3, label="bin. fraction")
    ax.axhline(1.0, color=GRAY, linewidth=0.9, linestyle="--")
    ax.axvline(mu_hat_bin, color=C_MU_BIN, linewidth=2.0,
               linestyle="--", zorder=5,
               label=f"$\\hat{{\\mu}}_{{\\rm bin}}={mu_hat_bin:.4f}$")
    ax.fill_between(p1_mu, p1_bin, alpha=0.12, color=C_STABLE)
    ax.set_ylim(-0.05, 1.12)
    _ax_style(ax,
              title=f"Phase 1 — binarisation fraction vs $\\mu$  |  "
                    f"$N={n}$,  $n_{{\\rm init}}={args.n_init}$",
              xlabel="$\\mu$",
              ylabel="fraction of fully binarised trajectories")
    ax.legend(fontsize=9, loc="lower right")

    # bottom: best + mean cut vs μ (Phase 1 scan only)
    ax = axes1[1]
    ax.plot(p1_mu, p1_best, color=C_STABLE,   linewidth=2.0,
            marker="o", markersize=5, label="best cut", zorder=3)
    ax.plot(p1_mu, p1_mean, color=C_UNSTABLE, linewidth=1.4,
            marker="s", markersize=4, linestyle="--",
            label="mean cut (binarised)", zorder=2)
    ax.axvline(mu_hat_bin, color=C_MU_BIN, linewidth=2.0,
               linestyle="--", zorder=5,
               label=f"$\\hat{{\\mu}}_{{\\rm bin}}={mu_hat_bin:.4f}$")
    if args.known_opt:
        ax.axhline(args.known_opt, color=BLACK, linewidth=1.3,
                   linestyle=":", label=f"known opt = {args.known_opt:.0f}")
    _ax_style(ax,
              title="Phase 1 — cut quality vs $\\mu$  (scan up to first full binarisation)",
              xlabel="$\\mu$",
              ylabel="cut value")
    ax.legend(fontsize=9, loc="lower right")

    fig1.suptitle(
        f"OIM G-Set experiment  |  {stem}  |  $N={n}$  |  "
        f"$W_{{\\rm tot}}={w_total:.0f}$  |  "
        f"$\\hat{{\\mu}}_{{\\rm bin}}={mu_hat_bin:.4f}$",
        color=BLACK, fontsize=12, fontweight="bold")

    # ══════════════════════════════════════════════════════════════════════════
    # Figure 2 — Phase 2 quality sweep
    # ══════════════════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(2, 1, figsize=(13, 8), facecolor=WHITE)
    fig2.subplots_adjust(hspace=0.42, left=0.09, right=0.97,
                         top=0.90, bottom=0.09)

    # top: cut quality
    ax = axes2[0]
    ax.plot(p2_mu, p2_best, color=C_STABLE,   linewidth=2.0,
            marker="o", markersize=5, label="best cut", zorder=3)
    ax.fill_between(p2_mu, p2_mean - p2_std, p2_mean + p2_std,
                    alpha=0.15, color=C_STABLE)
    ax.plot(p2_mu, p2_mean, color=C_STABLE, linewidth=1.2,
            linestyle="--", label="mean ± std (binarised)", zorder=2)

    # mark μ̂_bin
    ax.axvline(mu_hat_bin, color=C_MU_BIN, linewidth=2.0,
               linestyle="--", zorder=5,
               label=f"$\\hat{{\\mu}}_{{\\rm bin}}={mu_hat_bin:.4f}$")

    # mark best-cut μ
    ax.axvline(mu_best_cut, color=C_BEST, linewidth=1.8,
               linestyle=":", zorder=6,
               label=f"best-cut $\\mu={mu_best_cut:.4f}$ (cut={best_cut:.0f})")

    if args.known_opt:
        ax.axhline(args.known_opt, color=BLACK, linewidth=1.3,
                   linestyle=":", label=f"known opt = {args.known_opt:.0f}")
        # approximation ratio on right axis
        ax2r = ax.twinx()
        ax2r.plot(p2_mu, p2_best / args.known_opt,
                  color=C_UNSTABLE, linewidth=1.4,
                  linestyle="-.", label="approx. ratio")
        ax2r.set_ylabel("approx. ratio  (best / opt)", color=C_UNSTABLE, fontsize=10)
        ax2r.tick_params(axis="y", colors=C_UNSTABLE, labelsize=9)
        ax2r.set_ylim(0, 1.15)
        for sp in ax2r.spines.values():
            sp.set_edgecolor(BLACK); sp.set_linewidth(0.8)
        ax2r.legend(fontsize=8, loc="lower left")

    _ax_style(ax,
              title="Phase 2 — cut quality vs $\\mu$  ($\\mu \\geq \\hat{\\mu}_{\\rm bin}$)",
              xlabel="$\\mu$",
              ylabel="cut value")
    ax.legend(fontsize=9, loc="lower right")

    # bottom: binarisation fraction
    ax = axes2[1]
    ax.plot(p2_mu, p2_bin, color=C_STABLE, linewidth=2.0,
            marker="o", markersize=5, zorder=3)
    ax.fill_between(p2_mu, p2_bin, alpha=0.12, color=C_STABLE)
    ax.axvline(mu_hat_bin, color=C_MU_BIN, linewidth=2.0,
               linestyle="--", zorder=5,
               label=f"$\\hat{{\\mu}}_{{\\rm bin}}={mu_hat_bin:.4f}$")
    ax.set_ylim(-0.05, 1.12)
    _ax_style(ax,
              title="Phase 2 — binarisation fraction vs $\\mu$",
              xlabel="$\\mu$",
              ylabel="bin. fraction")
    ax.legend(fontsize=9, loc="lower right")

    fig2.suptitle(
        f"OIM G-Set experiment  |  {stem}  |  $N={n}$  |  "
        f"$W_{{\\rm tot}}={w_total:.0f}$  |  "
        f"best cut found = {best_cut:.0f}"
        + (f"  /  opt = {args.known_opt:.0f}" if args.known_opt else ""),
        color=BLACK, fontsize=12, fontweight="bold")
    
    return fig1, fig2


# ═══════════════════════════════════════════════════════════════════════════════
# Summary report
# ═══════════════════════════════════════════════════════════════════════════════
def print_summary(args, n, w_total, mu_hat_bin, p1_records, p2_records):
    best_p1 = max(p1_records, key=lambda r: r["best_cut"])
    best_p2 = max(p2_records, key=lambda r: r["best_cut"])
    print(f"{'='*65}")
    print(f"  SUMMARY  |  {Path(args.graph).stem}  |  N={n}  W_tot={w_total:.0f}")
    print(f"{'='*65}")
    print(f"  μ̂_bin  (empirical, Phase 1)  : {mu_hat_bin:.4f}")
    print(f"  Best cut at μ̂_bin            : {p2_records[0]['best_cut']:.2f}")
    print(f"  Best cut in Phase 2 sweep    : {best_p2['best_cut']:.2f} "
          f"at μ={best_p2['mu']:.4f}")
    if args.known_opt:
        r1 = p2_records[0]["best_cut"] / args.known_opt
        r2 = best_p2["best_cut"]       / args.known_opt
        print(f"  Approx ratio at μ̂_bin       : {r1:.4f}")
        print(f"  Best approx ratio            : {r2:.4f}")
        print(f"  Known optimum                : {args.known_opt:.0f}")

    hypothesis_holds = (p2_records[0]["best_cut"] >= best_p2["best_cut"] - 1e-6)
    print(f"\n  Hypothesis (μ̂_bin gives best cut): "
          f"{'✓ HOLDS' if hypothesis_holds else '✗ DOES NOT HOLD'}")
    print(f"    Cut at μ̂_bin  = {p2_records[0]['best_cut']:.2f}")
    print(f"    Best cut      = {best_p2['best_cut']:.2f}  "
          f"at μ = {best_p2['mu']:.4f}")
    print(f"{'='*65}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="G-Set OIM Max-Cut experiment: "
                    "find μ̂_bin empirically then sweep cut quality")
    parser.add_argument("--graph",       required=True,
                        help="Path to G-Set benchmark file")
    parser.add_argument("--mu_start",    type=float, default=0.01,
                        help="Phase-1 scan start (default: 0.01)")
    parser.add_argument("--mu_step",     type=float, default=0.05,
                        help="Phase-1 scan step size (default: 0.05)")
    parser.add_argument("--mu_max_scan", type=float, default=10.0,
                        help="Phase-1 scan upper bound (default: 10.0)")
    parser.add_argument("--n_mu2",       type=int,   default=30,
                        help="Phase-2 number of μ points (default: 30)")
    parser.add_argument("--mu2_factor",  type=float, default=3.0,
                        help="Phase-2 upper bound = μ̂_bin × factor (default: 3.0)")
    parser.add_argument("--n_init",      type=int,   default=20,
                        help="Trajectories per μ (default: 20)")
    parser.add_argument("--t_end",       type=float, default=200.0,
                        help="ODE horizon (default: 200)")
    parser.add_argument("--n_points",    type=int,   default=600,
                        help="Time-grid points per trajectory (default: 600)")
    parser.add_argument("--rtol",        type=float, default=1e-6,
                        help="RK45 rtol (default: 1e-6)")
    parser.add_argument("--atol",        type=float, default=1e-6,
                        help="RK45 atol (default: 1e-6)")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--known_opt",   type=float, default=None,
                        help="Known optimum cut value (optional, for gap reporting)")
    parser.add_argument("--save",        action="store_true",
                        help="Save figures as PDF + PNG")
    args = parser.parse_args()

    # ── load graph ─────────────────────────────────────────────────────────────
    print(f"\nLoading G-Set graph: {args.graph}")
    n, W, edges = parse_gset(args.graph)
    w_total     = float(np.sum(W)) / 2.0
    print(f"  N={n}  |E|={len(edges)}  W_total={w_total:.0f}")
    print(f"  NOTE: 2^N = 2^{n} — equilibrium enumeration is infeasible for N>{18}.")
    print(f"  Using empirical ODE-based approach.")

    # ── fixed initial conditions (same across all μ for fair comparison) ───────
    rng   = np.random.default_rng(args.seed)
    phi0s = [rng.uniform(-np.pi, np.pi, n) for _ in range(args.n_init)]
    print(f"\n  {args.n_init} initial conditions sampled from uniform(−π, π)  "
          f"(seed={args.seed})")
    print(f"  Using the SAME phi0s for every μ — isolates μ effect from IC randomness.")

    # ── Phase 1 ────────────────────────────────────────────────────────────────
    t_total = time.perf_counter()
    mu_hat_bin, p1_records = phase1_find_mu_bin(W, phi0s, args)

    # ── Phase 2 ────────────────────────────────────────────────────────────────
    p2_records = phase2_quality_sweep(W, phi0s, mu_hat_bin, args)

    print(f"  Total wall time: {time.perf_counter() - t_total:.1f}s\n")

    # ── summary ────────────────────────────────────────────────────────────────
    print_summary(args, n, w_total, mu_hat_bin, p1_records, p2_records)

    # ── figures ────────────────────────────────────────────────────────────────
    # pass n_init into args so figures can read it
    args.n_init_val = args.n_init
    fig1, fig2 = make_figures(args, n, w_total,
                                    mu_hat_bin, p1_records, p2_records)

    if args.save:
        stem = Path(args.graph).stem
        for tag, fig in [("phase1_scan", fig1),
                         ("phase2_sweep", fig2)]:
            for ext in ("pdf", "png"):
                fname = f"oim_gset_{stem}_{tag}.{ext}"
                fig.savefig(fname, bbox_inches="tight", dpi=150)
                print(f"  Saved: {fname}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
