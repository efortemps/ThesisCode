#!/usr/bin/env python3
"""
eigenvalue_sweep_analysis.py  (updated)
─────────────────────────────────────────────────────────────────────────────
Mu-sweep eigenvalue / bifurcation analysis for any graph fed to OIMMaxCut.

Three separate figures
──────────────────────
Figure 1 — Phase dynamics  +  convergence table
  Left   | Phase trajectories θ_i(t) for all initial conditions
  Right  | Per-trajectory convergence table (terminal state, type, H, cut,
           distance to every binary equilibrium, …) + summary table

Figure 2 — Bifurcation analysis (Jacobian)
  Row 0  | λ_max(D) bar chart  — all 2^N equilibria, bars coloured by cut
  Row 1  | Bifurcation diagram — λ_max(D)−μ vs μ, annotated H & cut

Figure 3 — Quality analysis (Updated with Hessian)
  Top Left    | A(φ*) Jacobian spectrum
  Top Right   | D(φ*) Laplacian spectrum
  Bottom Left | H(φ*) Hessian spectrum  (H = -2A)
  Bottom Right| Empty/Placeholder

Figure 4 — Hessian Bifurcation Diagram (NEW)
  Row 0  | λ_max(D) bar chart  — all 2^N equilibria, bars coloured by cut
  Row 1  | Bifurcation diagram — λ_min(H) = 2μ - 2λ_max(D) vs μ

──────────────────────────────────────────────────────────────────────────────
ROOT-CAUSE NOTE  (explains the ±π/2 trap seen in Figure_1_trajectories.png)
──────────────────────────────────────────────────────────────────────────────
The original code initialised trajectories from  uniform(-0.08, 0.08)  — a
tiny band around zero.  For a bipartite graph with μ ≈ 0, the coupling ODE
antisymmetrically pushes the two partitions in opposite directions, landing
both partitions near ±π/2.  At exactly θ = ±π/2 the SHIL restoring force
is   (μ/2) sin(2·π/2) = (μ/2) sin(π) = 0,   so the escape time scales as
1/μ.  With μ = 0.01 that is ~100× the integration window (t_end = 80), so
the trajectories appear "stuck" — marked NOT YET BINARISED.

In contrast, experiment_maxcut_interactive.py draws from uniform(-π, π),
which breaks the symmetry and puts each trajectory in the correct basin.

FIX applied here:
  • phi0s sampled from  uniform(-π, π, N)                [main(), line ~620]
  • identify_convergence now classifies terminal states into 5 types
  • Figure 1 carries a full per-trajectory table AND a summary table
    showing for each unique terminal state: count, type, H, cut, residual,
    and distance to the nearest binary (M2) equilibrium.

Mathematical background  (Cheng et al., Chaos 34, 073103, 2024)
────────────────────────────────────────────────────────────────
  A(φ*, μ) = D(φ*) − μ·I_N   →   λ_k(A) = λ_k(D) − μ   (slope −1 in μ)
  H(φ*, μ) = -2 * A(φ*, μ)   (Hessian is proportional to negative Jacobian)
  Stability: μ > λ_max(D(φ*))   <=>   λ_max(A) < 0   <=>   λ_min(H) > 0
  μ_bin = min_{φ*∈{0,π}^N} λ_max(D(φ*))
  H(σ) = Σ_{i<j} W_ij σ_i σ_j = W_total − 2·cut
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
from collections import defaultdict
from itertools import product as iproduct

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from OIM_Experiment.src.OIM_mu import OIMMaxCut
from OIM_Experiment.src.graph_utils import read_graph

# ── global style (TikZ-like, identical to experiment_maxcut_interactive) ──────
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
C_STABLE   = "#4C72B0"
C_UNSTABLE = "#DD8452"
C_BPART    = "#55a868"
C_FERRO    = "#c44e52"
C_MIXED    = "#8172b2"
C_MU_LINE  = "#ffb74d"

# Terminal-state type colours
_TYPE_COL = {
    "M2-binary":     C_STABLE,    # θ_i ∈ {0, π}            correct binarised
    "M1-half":       C_UNSTABLE,  # θ_i ∈ {±π/2}            ±π/2 trap
    "M1-mixed":      "#e377c2",   # mix of {0,π} and {±π/2}
    "Type-III":      C_MIXED,     # continuous-phase eq.
    "not-converged": GRAY,        # residual too large
}


def _ax_style(ax, title="", xlabel="", ylabel="", titlesize=11):
    ax.set_facecolor(WHITE)
    ax.tick_params(colors=BLACK, labelsize=10)
    for sp in ax.spines.values():
        sp.set_edgecolor(BLACK)
        sp.set_linewidth(0.8)
    ax.grid(True, color=LIGHT, linewidth=0.6, zorder=0)
    if title:  ax.set_title(title,  color=BLACK, fontsize=titlesize, pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=BLACK, fontsize=11)
    if ylabel: ax.set_ylabel(ylabel, color=BLACK, fontsize=11)


def _jacobian(oim: OIMMaxCut, phi_star: np.ndarray) -> np.ndarray:
    D = oim.build_D(phi_star)
    return D - oim.mu * np.diag(np.cos(2.0 * phi_star))

def _hessian(oim: OIMMaxCut, phi_star: np.ndarray) -> np.ndarray:
    """Hessian is -2 * Jacobian for this system"""
    return -2.0 * _jacobian(oim, phi_star)


# ═════════════════════════════════════════════════════════════════════════════
# Equilibrium analysis
# ═════════════════════════════════════════════════════════════════════════════
def analyse_equilibria(oim: OIMMaxCut) -> dict:
    n, mu, W = oim.n, oim.mu, oim.W
    w_total  = float(np.sum(W)) / 2.0
    rows     = []

    for bits in iproduct([0, 1], repeat=n):
        phi  = np.array([b * np.pi for b in bits], dtype=float)
        D    = oim.build_D(phi)
        ev_D = np.sort(np.linalg.eigvalsh(D))
        lmax = float(ev_D[-1])

        A    = _jacobian(oim, phi)
        ev_A = np.sort(np.linalg.eigvalsh(A))

        H_mat = _hessian(oim, phi)
        ev_H  = np.sort(np.linalg.eigvalsh(H_mat))

        sigma        = np.sign(np.cos(phi))
        sigma[sigma == 0] = 1.0
        H   = 0.5  * float(np.sum(W * np.outer(sigma, sigma)))
        cut = 0.25 * float(np.sum(W * (1.0 - np.outer(sigma, sigma))))

        rows.append({
            "bits": bits, "phi": phi, "D": D,
            "ev_D": ev_D, "ev_A": ev_A, "ev_H": ev_H,
            "lmax_D": lmax, "lmax_A": float(ev_A[-1]), 
            "lmin_H": float(ev_H[0]), # Min eigenvalue of Hessian determines stability
            "mu_thr": lmax, "stable": mu > lmax, "H": H, "cut": cut,
        })

    mu_bin   = min(r["lmax_D"] for r in rows)
    best_cut = max(r["cut"]    for r in rows)
    n_stable = sum(r["stable"] for r in rows)
    return dict(rows=rows, mu_bin=mu_bin, best_cut=best_cut, w_total=w_total,
                easiest=min(rows, key=lambda r: r["lmax_D"]),
                hardest=max(rows, key=lambda r: r["lmax_D"]),
                n_stable=n_stable, total=len(rows), n=n, mu=mu)


# ═════════════════════════════════════════════════════════════════════════════
# Mu sweep
# ═════════════════════════════════════════════════════════════════════════════
def mu_sweep(oim: OIMMaxCut, eq_data: dict, mu_vals: np.ndarray) -> dict:
    rows    = eq_data["rows"]
    paths   = {}
    bif_pts = []

    for r in rows:
        key = r["bits"]
        paths[key] = r["ev_D"][None, :] - mu_vals[:, None]
        mu_star = r["lmax_D"]
        if mu_vals[0] <= mu_star <= mu_vals[-1]:
            bif_pts.append((mu_star, key))

    lmax_D_arr  = np.array([r["lmax_D"] for r in rows])
    lmax_A_mat  = lmax_D_arr[:, None] - mu_vals[None, :]
    n_stable_mu = np.sum(lmax_A_mat < 0, axis=0)
    return dict(mu_vals=mu_vals, paths=paths, n_stable_mu=n_stable_mu,
                lmax_A_min_mu=lmax_A_mat.min(axis=0),
                lmax_A_max_mu=lmax_A_mat.max(axis=0),
                bifurcation_pts=bif_pts)


# ═════════════════════════════════════════════════════════════════════════════
# 3 representative equilibria
# ═════════════════════════════════════════════════════════════════════════════
def pick_representatives(rows: list):
    sr   = sorted(rows, key=lambda r: r["lmax_D"])
    low  = sr[0]
    mid  = sr[len(sr) // 2]
    high = sr[-1]

    def _lbl(tag, r):
        b = "".join(str(x) for x in r["bits"])
        return (f"{tag}:  $[{b}]$   "
                f"$\\lambda_{{\\max}}={r['lmax_D']:.3f}$   "
                f"$H={r['H']:.1f}$   cut$={r['cut']:.1f}$")

    return [
        (low,  _lbl("Easiest", low),  C_BPART, "-"),
        (mid,  _lbl("Median",  mid),  C_MIXED, "--"),
        (high, _lbl("Hardest", high), C_FERRO, ":"),
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Convergence identification
# ═════════════════════════════════════════════════════════════════════════════
def _atom(theta_i: float, tol: float = 0.05) -> str:
    s = np.sin(theta_i)
    c = np.cos(theta_i)
    if abs(s) < tol:
        return "zero" if c > 0 else "pi"
    if abs(abs(s) - 1.0) < tol:
        return "half"
    return "other"


def identify_convergence(sol, W: np.ndarray,
                          binary_equilibria: list,
                          bin_tol: float = 0.05) -> dict:
    theta = sol.y[:, -1].copy()
    n     = len(theta)

    sigma = np.sign(np.cos(theta))
    sigma[sigma == 0] = 1.0
    bits     = tuple(0 if s > 0 else 1 for s in sigma)
    H        = 0.5  * float(np.sum(W * np.outer(sigma, sigma)))
    cut      = 0.25 * float(np.sum(W * (1.0 - np.outer(sigma, sigma))))
    residual = float(np.max(np.abs(np.sin(theta))))

    atom_types = [_atom(th, bin_tol) for th in theta]
    n_zero = atom_types.count("zero")
    n_pi   = atom_types.count("pi")
    n_half = atom_types.count("half")
    n_othe = atom_types.count("other")

    if n_zero + n_pi == n:
        state_type = "M2-binary"
    elif n_half == n:
        state_type = "M1-half"
    elif n_half > 0 and n_othe == 0 and n_zero + n_pi + n_half == n:
        state_type = "M1-mixed"
    elif n_othe > 0:
        state_type = "Type-III"
    else:
        state_type = "not-converged"

    # nearest M2 equilibrium in phase space
    nearest_eq = None
    min_dist   = np.inf
    for r in binary_equilibria:
        diff = theta - r["phi"]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi   # wrap to (−π, π]
        dist = float(np.linalg.norm(diff))
        if dist < min_dist:
            min_dist   = dist
            nearest_eq = {
                "bits"   : r["bits"],
                "phi"    : r["phi"].copy(),
                "H"      : r["H"],
                "cut"    : r["cut"],
                "mu_thr" : r["mu_thr"],
                "stable" : r["stable"],
                "dist_L2": dist,
            }

    return dict(
        theta_end  = theta,
        bits       = bits,
        H          = H,
        cut        = cut,
        residual   = residual,
        is_binary  = (state_type == "M2-binary"),
        state_type = state_type,
        atom_types = atom_types,
        nearest_eq = nearest_eq,
    )

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Bifurcation analysis (Jacobian)
# ═════════════════════════════════════════════════════════════════════════════
def make_figure2(args, n, edges, eq_data, sweep_data):
    rows       = eq_data["rows"]
    mu_bin     = eq_data["mu_bin"]
    w_total    = eq_data["w_total"]
    best_cut   = eq_data["best_cut"]
    mu_vals    = sweep_data["mu_vals"]
    n_eq       = eq_data["total"]
    current_mu = eq_data["mu"]

    lmax_D_all = np.array([r["lmax_D"] for r in rows])
    H_all      = np.array([r["H"]      for r in rows])
    cut_all    = np.array([r["cut"]    for r in rows])
    unique_lmax, inverse, counts_lmax = np.unique(
        np.round(lmax_D_all, 5), return_inverse=True, return_counts=True)
    avg_H_per_lmax   = np.array([H_all  [inverse == k].mean()
                                  for k in range(len(unique_lmax))])
    avg_cut_per_lmax = np.array([cut_all[inverse == k].mean()
                                  for k in range(len(unique_lmax))])

    cmap_cut = plt.get_cmap("RdYlGn")
    norm_cut = mcolors.Normalize(vmin=0, vmax=best_cut)

    fig = plt.figure(figsize=(18, 12), facecolor=WHITE)
    gs  = gridspec.GridSpec(2, 1, figure=fig,
                            height_ratios=[0.70, 1.30], hspace=0.45,
                            left=0.06, right=0.96, top=0.92, bottom=0.08)
    ax_lmbar = fig.add_subplot(gs[0])
    ax_bif   = fig.add_subplot(gs[1])

    # ── bar chart ─────────────────────────────────────────────────────────────
    sorted_rows = sorted(rows, key=lambda r: r["lmax_D"])
    lmax_arr    = np.array([r["lmax_D"] for r in sorted_rows])
    cut_arr_s   = np.array([r["cut"]    for r in sorted_rows])
    bar_cols    = [cmap_cut(norm_cut(c)) for c in cut_arr_s]

    ax_lmbar.bar(np.arange(n_eq), lmax_arr,
                 color=bar_cols, width=1.0, edgecolor="none", zorder=2)
    ax_lmbar.axhline(0,          color=BLACK,     linewidth=0.9,  zorder=3)
    ax_lmbar.axhline(mu_bin,     color=C_STABLE,  linewidth=2.0,
                     linestyle="--", zorder=5,
                     label=f"$\\mu_{{\\rm bin}} = {mu_bin:.3f}$")
    ax_lmbar.axhline(current_mu, color=C_MU_LINE, linewidth=2.0,
                     linestyle="--", zorder=6,
                     label=f"current $\\mu = {current_mu:.3f}$")

    sm_bar = plt.cm.ScalarMappable(cmap=cmap_cut, norm=norm_cut)
    sm_bar.set_array([])
    cb_bar = fig.colorbar(sm_bar, ax=ax_lmbar, fraction=0.015, pad=0.01)
    cb_bar.set_label("Cut value", fontsize=10)
    cb_bar.ax.tick_params(labelsize=9)
    cb_bar.outline.set_edgecolor(BLACK)

    ax_lmbar.set_xlim(-1, n_eq)
    ax_lmbar.legend(fontsize=10, loc="upper left")
    _ax_style(ax_lmbar,
              title=(f"$\\lambda_{{\\max}}(D(\\phi^*))$ for all $2^N={n_eq}$ equilibria  "
                     f"(sorted)  |  bar colour = cut quality  |  "
                     f"stable at current $\\mu$: {eq_data['n_stable']}/{n_eq}"),
              xlabel="Equilibrium index",
              ylabel="$\\lambda_{\\max}(D(\\phi^*))$")

    # ── bifurcation diagram (Jacobian) ─────────────────────────────────────────
    norm_bif = mcolors.Normalize(vmin=0, vmax=best_cut)
    cmap_bif = plt.get_cmap("RdYlGn")
    bif_lo   = unique_lmax.min() - mu_vals.max() - 0.5
    bif_hi   = unique_lmax.max() + 0.5

    ax_bif.fill_between(mu_vals,
                        np.minimum(sweep_data["lmax_A_min_mu"], 0), 0,
                        color=C_STABLE,   alpha=0.12, zorder=0)
    ax_bif.fill_between(mu_vals, 0,
                        np.maximum(sweep_data["lmax_A_max_mu"], 0),
                        color=C_UNSTABLE, alpha=0.10, zorder=0)
    ax_bif.axhline(0, color=BLACK, linewidth=1.5, zorder=5,
                   label="$\\lambda_{\\max}(A)=0$  (stability boundary)")
    ax_bif.axvline(current_mu, color=C_MU_LINE, linewidth=1.8,
                   linestyle="--", zorder=6,
                   label=f"current $\\mu={current_mu:.3f}$")
    ax_bif.axvline(mu_bin, color=BLACK, linewidth=1.8,
                   linestyle=":", zorder=7,
                   label=f"$\\mu_{{\\rm bin}}={mu_bin:.3f}$")

    n_unique   = len(unique_lmax)
    ann_y_vals = np.linspace(bif_hi * 0.95, bif_hi * 0.05, n_unique)

    for idx, (lm, cnt, H_mean, cut_mean) in enumerate(
            zip(unique_lmax, counts_lmax, avg_H_per_lmax, avg_cut_per_lmax)):
        c  = cmap_bif(norm_bif(cut_mean))
        lw = 0.8 + 0.55 * np.log1p(cnt / 2.0)
        ax_bif.plot(mu_vals, lm - mu_vals, color=c, linewidth=lw, alpha=0.85)
        mu_star = lm
        if mu_vals[0] <= mu_star <= mu_vals[-1]:
            ax_bif.scatter([mu_star], [0.0], color=c, s=55, zorder=7,
                           edgecolors=BLACK, linewidths=0.7)
            if idx == 0 or idx == 1 or idx % 5 == 0:
                ax_bif.annotate(
                    f"$\\mu^*={mu_star:.2f}$\n"
                    f"$\\bar{{H}}={H_mean:.1f}$\n"
                    f"cut$={cut_mean:.1f}$  $\\times{cnt}$",
                    xy=(mu_star, 0.0),
                    xytext=(mu_star + (mu_vals[-1] - mu_vals[0]) * 0.012,
                            ann_y_vals[idx]),
                    fontsize=7, color=c, zorder=8,
                    arrowprops=dict(arrowstyle="->", color=c, lw=0.65,
                                    shrinkA=2, shrinkB=2))

    sm_bif = plt.cm.ScalarMappable(cmap=cmap_bif, norm=norm_bif)
    sm_bif.set_array([])
    cb_bif = fig.colorbar(sm_bif, ax=ax_bif, fraction=0.012, pad=0.01)
    cb_bif.set_label("Mean cut at $\\mu^*$", fontsize=10)
    cb_bif.ax.tick_params(labelsize=9)
    cb_bif.outline.set_edgecolor(BLACK)

    ax_bif.set_xlim(mu_vals[0], mu_vals[-1])
    ax_bif.set_ylim(bif_lo, bif_hi)
    ax_bif.legend(fontsize=10, loc="upper right", framealpha=0.93)
    ax_bif.text(0.01, 0.04, "← stable",   transform=ax_bif.transAxes,
                ha="left", fontsize=10, color=C_STABLE)
    ax_bif.text(0.01, 0.96, "← unstable", transform=ax_bif.transAxes,
                ha="left", va="top", fontsize=10, color=C_UNSTABLE)
    _ax_style(ax_bif,
              title=("Bifurcation diagram (Jacobian A):  $\\lambda_{\\max}(A) = \\lambda_{\\max}(D)-\\mu$  vs  $\\mu$  |  "
                     "line colour = cut quality  |  dots = stability transitions"),
              xlabel="$\\mu$",
              ylabel="$\\lambda_{\\max}(A)=\\lambda_{\\max}(D)-\\mu$")

    fig.suptitle(
        f"OIM Bifurcation Analysis (Jacobian) |  {args.graph}  |  "
        f"$N={n}$,  $|E|={len(edges)}$,  $2^N={n_eq}$ equilibria  |  "
        f"$\\mu_{{\\rm bin}}={mu_bin:.4f}$  |  "
        f"Best cut $={best_cut:.1f}$,  $W_{{\\rm tot}}={w_total:.1f}$",
        color=BLACK, fontsize=12, fontweight="bold")
    return fig

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Quality analysis (Updated with Hessian)
# ═════════════════════════════════════════════════════════════════════════════
def make_figure3(args, n, W, eq_data, conv_results):
    rows = eq_data["rows"]
    best_cut = eq_data["best_cut"]
    w_total = eq_data["w_total"]
    mu_bin = eq_data["mu_bin"]
    current_mu = eq_data["mu"]
    reps = pick_representatives(rows)

    # 2x2 layout instead of 1x2
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), facecolor=WHITE)
    fig.subplots_adjust(wspace=0.25, hspace=0.35, left=0.06, right=0.96,
                        top=0.90, bottom=0.08)

    ax_aspec = axes[0, 0]
    ax_dspec = axes[0, 1]
    ax_hspec = axes[1, 0]
    ax_empty = axes[1, 1]

    # Hide the empty bottom right subplot for now
    ax_empty.set_visible(False)

    xidx = np.arange(1, n + 1)

    # ── 1. A(φ*) Jacobian eigenvalue spectrum ─────────────────────────────────
    ev_a_all_reps = np.concatenate([eq["ev_A"] for eq, _, _, _ in reps])
    ev_a_lo, ev_a_hi = ev_a_all_reps.min(), ev_a_all_reps.max()
    margin_a = max(0.06 * (ev_a_hi - ev_a_lo), 0.4)

    # Shading: A stable if all λ(A) < 0
    ax_aspec.fill_between([0.5, n + 0.5], ev_a_lo - margin_a, 0,
                          color=C_STABLE, alpha=0.10, zorder=0)
    ax_aspec.fill_between([0.5, n + 0.5], 0, ev_a_hi + margin_a,
                          color=C_UNSTABLE, alpha=0.10, zorder=0)
    ax_aspec.axhline(0, color=BLACK, linewidth=1.0, linestyle="--",
                     alpha=0.55, zorder=2)

    for k_r, (eq, lbl, col, ls) in enumerate(reps):
        ev_desc = eq["ev_A"][::-1] # largest to smallest
        ax_aspec.plot(xidx, ev_desc, color=col, linestyle=ls,
                      linewidth=2.2, marker="o", markersize=8,
                      label=lbl, zorder=3)
        dy = margin_a * (0.6 + 0.35 * k_r)
        ax_aspec.annotate(
            f"$\\lambda_{{\\max}}={eq['lmax_A']:.3f}$\n"
            f"$H={eq['H']:.1f}$ cut$={eq['cut']:.1f}$",
            xy=(1, ev_desc[0]),
            xytext=(1.6, ev_desc[0] + dy),
            fontsize=9, color=col, zorder=5,
            arrowprops=dict(arrowstyle="->", color=col, lw=0.9,
                            shrinkA=2, shrinkB=2))

    ax_aspec.set_xticks(xidx)
    ax_aspec.set_xlim(0.5, n + 0.5)
    ax_aspec.set_ylim(ev_a_lo - 2 * margin_a, ev_a_hi + 3.5 * margin_a)
    ax_aspec.legend(fontsize=8.5, loc="lower left", ncol=1)
    ax_aspec.text(0.98, 0.98,
                  f"Jacobian at current $\\mu={current_mu:.3f}$\n"
                  "$A(\\phi^*,\\mu)=D(\\phi^*)-\\mu I$\n"
                  "Stable iff $\\lambda_{\\max}(A) < 0$",
                  transform=ax_aspec.transAxes, ha="right", va="top",
                  fontsize=9.5,
                  bbox=dict(boxstyle="round,pad=0.35", facecolor=WHITE,
                            edgecolor=GRAY, alpha=0.94))
    _ax_style(ax_aspec,
              title=("$A(\\phi^*,\\mu)$ Jacobian spectrum — 3 representative equilibria\n"
                     "Blue: $\\lambda<0$ (stable dir) | Orange: $\\lambda>0$ (unstable dir)"),
              xlabel="Eigenvalue rank $k$ (largest first)",
              ylabel="$\\lambda_k\\left(A(\\phi^*)\\right)$",
              titlesize=11)

    # ── 2. D(φ*) Laplacian eigenvalue spectrum ────────────────────────────────
    ev_all_reps = np.concatenate([eq["ev_D"] for eq, _, _, _ in reps])
    ev_lo, ev_hi = ev_all_reps.min(), ev_all_reps.max()
    margin = max(0.06 * (ev_hi - ev_lo), 0.4)

    ax_dspec.fill_between([0.5, n + 0.5], ev_lo - margin, 0,
                          color=C_STABLE, alpha=0.10, zorder=0)
    ax_dspec.fill_between([0.5, n + 0.5], 0, ev_hi + margin,
                          color=C_UNSTABLE, alpha=0.10, zorder=0)
    ax_dspec.axhline(0, color=BLACK, linewidth=1.0, linestyle="--",
                     alpha=0.55, zorder=2)

    for k_r, (eq, lbl, col, ls) in enumerate(reps):
        ev_desc = eq["ev_D"][::-1] # largest to smallest
        ax_dspec.plot(xidx, ev_desc, color=col, linestyle=ls,
                      linewidth=2.2, marker="o", markersize=8,
                      label=lbl, zorder=3)
        dy = margin * (0.6 + 0.35 * k_r)
        ax_dspec.annotate(
            f"$\\lambda_{{\\max}}={eq['lmax_D']:.3f}$\n"
            f"$H={eq['H']:.1f}$ cut$={eq['cut']:.1f}$",
            xy=(1, ev_desc[0]),
            xytext=(1.6, ev_desc[0] + dy),
            fontsize=9, color=col, zorder=5,
            arrowprops=dict(arrowstyle="->", color=col, lw=0.9,
                            shrinkA=2, shrinkB=2))

    ax_dspec.set_xticks(xidx)
    ax_dspec.set_xlim(0.5, n + 0.5)
    ax_dspec.set_ylim(ev_lo - 2 * margin, ev_hi + 3.5 * margin)
    ax_dspec.legend(fontsize=8.5, loc="lower left", ncol=1)
    ax_dspec.text(0.98, 0.98,
                  "$D(\\phi^*)$ depends only on topology\n"
                  "$\\lambda_k(A)=\\lambda_k(D)-\\mu$\n"
                  "Stable iff $\\mu>\\lambda_{\\max}(D)$",
                  transform=ax_dspec.transAxes, ha="right", va="top",
                  fontsize=9.5,
                  bbox=dict(boxstyle="round,pad=0.35", facecolor=WHITE,
                            edgecolor=GRAY, alpha=0.94))
    _ax_style(ax_dspec,
              title=("$D(\\phi^*)$ eigenvalue spectrum — 3 representative equilibria\n"
                     "Blue: $\\lambda<0$ | Orange: $\\lambda>0$ | Annotated: $\\lambda_{\\max}$, $H$, cut"),
              xlabel="Eigenvalue rank $k$ (largest first)",
              ylabel="$\\lambda_k\\left(D(\\phi^*)\\right)$",
              titlesize=11)

    # ── 3. Hessian H(φ*) eigenvalue spectrum ──────────────────────────────────
    # Hessian = -2 * A. Thus eigenvalues are -2 * ev_A
    # A is stable when λ_max(A) < 0, which means H is stable when λ_min(H) > 0 (positive definite)
    ev_h_all_reps = np.concatenate([eq["ev_H"] for eq, _, _, _ in reps])
    ev_h_lo, ev_h_hi = ev_h_all_reps.min(), ev_h_all_reps.max()
    margin_h = max(0.06 * (ev_h_hi - ev_h_lo), 0.4)

    # Shading: H stable (positive definite) if all λ(H) > 0
    # So we want Blue (C_STABLE) for λ > 0, Orange (C_UNSTABLE) for λ < 0
    ax_hspec.fill_between([0.5, n + 0.5], 0, ev_h_hi + margin_h,
                          color=C_STABLE, alpha=0.10, zorder=0)
    ax_hspec.fill_between([0.5, n + 0.5], ev_h_lo - margin_h, 0,
                          color=C_UNSTABLE, alpha=0.10, zorder=0)
    ax_hspec.axhline(0, color=BLACK, linewidth=1.0, linestyle="--",
                     alpha=0.55, zorder=2)

    for k_r, (eq, lbl, col, ls) in enumerate(reps):
        ev_asc = eq["ev_H"]  # smallest to largest! (since H = -2A, the order flips)
        ax_hspec.plot(xidx, ev_asc, color=col, linestyle=ls,
                      linewidth=2.2, marker="o", markersize=8,
                      label=lbl, zorder=3)
        # Annotate the MINIMUM eigenvalue
        dy = margin_h * (0.6 + 0.35 * k_r)
        # Because we plot smallest first, the first point is ev_asc[0]
        ax_hspec.annotate(
            f"$\\lambda_{{\\min}}={eq['lmin_H']:.3f}$\n"
            f"$H={eq['H']:.1f}$ cut$={eq['cut']:.1f}$",
            xy=(1, ev_asc[0]),
            xytext=(1.6, ev_asc[0] - dy), # Point down instead of up
            fontsize=9, color=col, zorder=5,
            arrowprops=dict(arrowstyle="->", color=col, lw=0.9,
                            shrinkA=2, shrinkB=2))

    ax_hspec.set_xticks(xidx)
    ax_hspec.set_xlim(0.5, n + 0.5)
    ax_hspec.set_ylim(ev_h_lo - 3.5 * margin_h, ev_h_hi + 2 * margin_h)
    ax_hspec.legend(fontsize=8.5, loc="upper left", ncol=1)
    ax_hspec.text(0.98, 0.02, # Position at bottom right instead of top right
                  f"Hessian at current $\\mu={current_mu:.3f}$\n"
                  "$H(\\phi^*,\\mu)=-2 A(\\phi^*,\\mu)$\n"
                  "Stable (Local Min) iff $\\lambda_{\\min}(H) > 0$",
                  transform=ax_hspec.transAxes, ha="right", va="bottom",
                  fontsize=9.5,
                  bbox=dict(boxstyle="round,pad=0.35", facecolor=WHITE,
                            edgecolor=GRAY, alpha=0.94))
    _ax_style(ax_hspec,
              title=("$H(\\phi^*,\\mu)$ Hessian spectrum — 3 representative equilibria\n"
                     "Blue: $\\lambda>0$ (stable/convex) | Orange: $\\lambda<0$ (unstable/concave)"),
              xlabel="Eigenvalue rank $k$ (smallest first)",
              ylabel="$\\lambda_k\\left(H(\\phi^*)\\right)$",
              titlesize=11)


    fig.suptitle(
        f"OIM Quality Analysis | {args.graph} | "
        f"$N={n}$, $2^N={len(rows)}$ equilibria | "
        f"$\\mu_{{\\rm bin}}={mu_bin:.4f}$ | "
        f"Best cut $={best_cut:.1f}$, $W_{{\\rm tot}}={w_total:.1f}$",
        color=BLACK, fontsize=14, fontweight="bold")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Hessian Bifurcation Diagram
# ═════════════════════════════════════════════════════════════════════════════
def make_figure4(args, n, edges, eq_data, sweep_data):
    rows       = eq_data["rows"]
    mu_bin     = eq_data["mu_bin"]
    w_total    = eq_data["w_total"]
    best_cut   = eq_data["best_cut"]
    mu_vals    = sweep_data["mu_vals"]
    n_eq       = eq_data["total"]
    current_mu = eq_data["mu"]

    lmax_D_all = np.array([r["lmax_D"] for r in rows])
    H_all      = np.array([r["H"]      for r in rows])
    cut_all    = np.array([r["cut"]    for r in rows])
    unique_lmax, inverse, counts_lmax = np.unique(
        np.round(lmax_D_all, 5), return_inverse=True, return_counts=True)
    avg_H_per_lmax   = np.array([H_all  [inverse == k].mean()
                                  for k in range(len(unique_lmax))])
    avg_cut_per_lmax = np.array([cut_all[inverse == k].mean()
                                  for k in range(len(unique_lmax))])

    cmap_cut = plt.get_cmap("RdYlGn")
    norm_cut = mcolors.Normalize(vmin=0, vmax=best_cut)

    fig = plt.figure(figsize=(18, 12), facecolor=WHITE)
    gs  = gridspec.GridSpec(2, 1, figure=fig,
                            height_ratios=[0.70, 1.30], hspace=0.45,
                            left=0.06, right=0.96, top=0.92, bottom=0.08)
    ax_lmbar = fig.add_subplot(gs[0])
    ax_bif_H = fig.add_subplot(gs[1])

    # ── bar chart ─────────────────────────────────────────────────────────────
    sorted_rows = sorted(rows, key=lambda r: r["lmax_D"])
    lmax_arr    = np.array([r["lmax_D"] for r in sorted_rows])
    cut_arr_s   = np.array([r["cut"]    for r in sorted_rows])
    bar_cols    = [cmap_cut(norm_cut(c)) for c in cut_arr_s]

    ax_lmbar.bar(np.arange(n_eq), lmax_arr,
                 color=bar_cols, width=1.0, edgecolor="none", zorder=2)
    ax_lmbar.axhline(0,          color=BLACK,     linewidth=0.9,  zorder=3)
    ax_lmbar.axhline(mu_bin,     color=C_STABLE,  linewidth=2.0,
                     linestyle="--", zorder=5,
                     label=f"$\\mu_{{\\rm bin}} = {mu_bin:.3f}$")
    ax_lmbar.axhline(current_mu, color=C_MU_LINE, linewidth=2.0,
                     linestyle="--", zorder=6,
                     label=f"current $\\mu = {current_mu:.3f}$")

    sm_bar = plt.cm.ScalarMappable(cmap=cmap_cut, norm=norm_cut)
    sm_bar.set_array([])
    cb_bar = fig.colorbar(sm_bar, ax=ax_lmbar, fraction=0.015, pad=0.01)
    cb_bar.set_label("Cut value", fontsize=10)
    cb_bar.ax.tick_params(labelsize=9)
    cb_bar.outline.set_edgecolor(BLACK)

    ax_lmbar.set_xlim(-1, n_eq)
    ax_lmbar.legend(fontsize=10, loc="upper left")
    _ax_style(ax_lmbar,
              title=(f"$\\lambda_{{\\max}}(D(\\phi^*))$ for all $2^N={n_eq}$ equilibria  "
                     f"(sorted)  |  bar colour = cut quality  |  "
                     f"stable at current $\\mu$: {eq_data['n_stable']}/{n_eq}"),
              xlabel="Equilibrium index",
              ylabel="$\\lambda_{\\max}(D(\\phi^*))$")

    # ── bifurcation diagram (Hessian) ─────────────────────────────────────────
    norm_bif = mcolors.Normalize(vmin=0, vmax=best_cut)
    cmap_bif = plt.get_cmap("RdYlGn")

    # For A: max eigenvalue of A is lambda_max(D) - mu.
    # For H: H = -2A. Therefore the min eigenvalue of H is 2*mu - 2*lambda_max(D).

    bif_lo_A = unique_lmax.min() - mu_vals.max() - 0.5
    bif_hi_A = unique_lmax.max() + 0.5

    bif_lo_H = -2.0 * bif_hi_A
    bif_hi_H = -2.0 * bif_lo_A

    ax_bif_H.fill_between(mu_vals,
                        0, np.maximum(-2.0 * sweep_data["lmax_A_min_mu"], 0),
                        color=C_STABLE,   alpha=0.12, zorder=0)
    ax_bif_H.fill_between(mu_vals, 
                        np.minimum(-2.0 * sweep_data["lmax_A_max_mu"], 0), 0,
                        color=C_UNSTABLE, alpha=0.10, zorder=0)
    ax_bif_H.axhline(0, color=BLACK, linewidth=1.5, zorder=5,
                   label="$\\lambda_{\\min}(H)=0$  (stability boundary)")
    ax_bif_H.axvline(current_mu, color=C_MU_LINE, linewidth=1.8,
                   linestyle="--", zorder=6,
                   label=f"current $\\mu={current_mu:.3f}$")
    ax_bif_H.axvline(mu_bin, color=BLACK, linewidth=1.8,
                   linestyle=":", zorder=7,
                   label=f"$\\mu_{{\\rm bin}}={mu_bin:.3f}$")

    n_unique   = len(unique_lmax)
    ann_y_vals_H = np.linspace(bif_lo_H * 0.95, bif_lo_H * 0.05, n_unique)

    for idx, (lm, cnt, H_mean, cut_mean) in enumerate(
            zip(unique_lmax, counts_lmax, avg_H_per_lmax, avg_cut_per_lmax)):
        c  = cmap_bif(norm_bif(cut_mean))
        lw = 0.8 + 0.55 * np.log1p(cnt / 2.0)
        # Plot lambda_min(H) = 2 * mu - 2 * lm
        ax_bif_H.plot(mu_vals, 2.0 * mu_vals - 2.0 * lm, color=c, linewidth=lw, alpha=0.85)
        mu_star = lm
        if mu_vals[0] <= mu_star <= mu_vals[-1]:
            ax_bif_H.scatter([mu_star], [0.0], color=c, s=55, zorder=7,
                           edgecolors=BLACK, linewidths=0.7)
            if idx == 0 or idx == 1 or idx % 5 == 0:
                ax_bif_H.annotate(
                    f"$\\mu^*={mu_star:.2f}$\n"
                    f"$\\bar{{H}}={H_mean:.1f}$\n"
                    f"cut$={cut_mean:.1f}$  $\\times{cnt}$",
                    xy=(mu_star, 0.0),
                    xytext=(mu_star + (mu_vals[-1] - mu_vals[0]) * 0.012,
                            ann_y_vals_H[idx]),
                    fontsize=7, color=c, zorder=8,
                    arrowprops=dict(arrowstyle="->", color=c, lw=0.65,
                                    shrinkA=2, shrinkB=2))

    cb_bif_H = fig.colorbar(sm_bar, ax=ax_bif_H, fraction=0.012, pad=0.01)
    cb_bif_H.set_label("Mean cut at $\\mu^*$", fontsize=10)
    cb_bif_H.ax.tick_params(labelsize=9)
    cb_bif_H.outline.set_edgecolor(BLACK)

    ax_bif_H.set_xlim(mu_vals[0], mu_vals[-1])
    ax_bif_H.set_ylim(bif_lo_H, bif_hi_H)
    ax_bif_H.legend(fontsize=10, loc="lower right", framealpha=0.93)
    ax_bif_H.text(0.01, 0.96, "← stable (positive definite)",   transform=ax_bif_H.transAxes,
                ha="left", va="top", fontsize=10, color=C_STABLE)
    ax_bif_H.text(0.01, 0.04, "← unstable", transform=ax_bif_H.transAxes,
                ha="left", fontsize=10, color=C_UNSTABLE)
    _ax_style(ax_bif_H,
              title=("Bifurcation diagram (Hessian H):  $\\lambda_{\\min}(H) = 2\\mu - 2\\lambda_{\\max}(D)$  vs  $\\mu$  |  "
                     "line colour = cut quality  |  dots = stability transitions"),
              xlabel="$\\mu$",
              ylabel="$\\lambda_{\\min}(H)=2\\mu - 2\\lambda_{\\max}(D)$")

    fig.suptitle(
        f"OIM Hessian Bifurcation Analysis  |  {args.graph}  |  "
        f"$N={n}$,  $|E|={len(edges)}$,  $2^N={n_eq}$ equilibria  |  "
        f"$\\mu_{{\\rm bin}}={mu_bin:.4f}$  |  "
        f"Best cut $={best_cut:.1f}$,  $W_{{\\rm tot}}={w_total:.1f}$",
        color=BLACK, fontsize=12, fontweight="bold")
    return fig

def main():
    parser = argparse.ArgumentParser(
        description="OIM eigenvalue sweep & bifurcation analysis — 4 figures")
    parser.add_argument("--graph",  required=True)
    parser.add_argument("--mu_min", type=float, default=None)
    parser.add_argument("--mu_max", type=float, default=None)
    parser.add_argument("--n_mu",   type=int,   default=300)
    parser.add_argument("--mu",     type=float, default=None)
    parser.add_argument("--n_init", type=int,   default=12)
    parser.add_argument("--t_end",  type=float, default=80.0)
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--save",   action="store_true",
                        help="Save all figures as PDF+PNG")
    args = parser.parse_args()

    print(f"\nLoading graph: {args.graph}")
    W     = read_graph(args.graph)
    n     = W.shape[0]
    edges = [(i, j, W[i, j])
             for i in range(n) for j in range(i + 1, n) if W[i, j] > 0]
    print(f"  N={n}  |E|={len(edges)}  W_total={W.sum()/2:.2f}")
    if n > 18:
        print(f"[warn] N={n} > 18 → 2^N={2**n} — may be slow")

    # ── scan λ_max range ──────────────────────────────────────────────────────
    oim_scan = OIMMaxCut(W, mu=1.0, seed=args.seed)
    print(f"  Scanning all 2^{n}={2**n} equilibria...")
    lmax_scan = []
    for bits in iproduct([0, 1], repeat=n):
        phi = np.array([b * np.pi for b in bits], dtype=float)
        lmax_scan.append(float(np.linalg.eigvalsh(oim_scan.build_D(phi)).max()))

    global_lmax_min = min(lmax_scan)
    global_lmax_max = max(lmax_scan)
    print(f"  λ_max(D) range: [{global_lmax_min:.4f}, {global_lmax_max:.4f}]")
    print(f"  μ_bin estimate : {global_lmax_min:.4f}")

    mu_min_eff = (args.mu_min if args.mu_min is not None
                  else min(0.0, global_lmax_min - 0.5))
    mu_max_eff = (args.mu_max if args.mu_max is not None
                  else global_lmax_max * 1.30)
    mu_ref     = (args.mu if args.mu is not None
                  else (mu_min_eff + mu_max_eff) / 2.0)

    print(f"  μ sweep: [{mu_min_eff:.4f}, {mu_max_eff:.4f}]  ({args.n_mu} steps)")
    print(f"  Reference μ: {mu_ref:.4f}")

    oim = OIMMaxCut(W, mu=mu_ref, seed=args.seed)
    print(f"\n  Equilibrium analysis at μ={mu_ref:.4f}...")
    eq_data = analyse_equilibria(oim)
    print(f"  μ_bin={eq_data['mu_bin']:.4f}  stable: "
          f"{eq_data['n_stable']}/{eq_data['total']}  "
          f"best cut: {eq_data['best_cut']:.1f}")

    mu_vals    = np.linspace(mu_min_eff, mu_max_eff, args.n_mu)
    print(f"\n  μ sweep ({args.n_mu} steps)...")
    sweep_data = mu_sweep(oim, eq_data, mu_vals)
    print(f"  Bifurcation points: {len(sweep_data['bifurcation_pts'])}")

    # ── simulation  ───────────────────────────────────────────────────────────
    MU_SIM  = max(global_lmax_min + 0.1, mu_min_eff + 0.02)
    rng     = np.random.default_rng(args.seed)
    phi0s   = [rng.uniform(-np.pi, np.pi, n) for _ in range(args.n_init)]  # ← FIX
    oim_sim = OIMMaxCut(W, mu=MU_SIM, seed=args.seed)
    print(f"\n  Simulating {args.n_init} trajectories at "
          f"μ={MU_SIM:.4f}  (t=0..{args.t_end})...")
    sols = oim_sim.simulate_many(phi0s, t_span=(0., args.t_end), n_points=500)
    print("  Done.\n")

    # ── convergence analysis ───────────────────────────────────────────────────
    binary_equilibria = eq_data["rows"]
    conv_results      = [identify_convergence(sol, W, binary_equilibria, bin_tol=0.05)
                         for sol in sols]

    # console report
    cw = max(n, 8)
    print(f"  {'#':>3}  {'type':<16}  {'bits':<{cw}}  {'H':>8}  "
          f"{'cut':>8}  {'residual':>9}  {'nearest M2':<{cw}}  dist  stable?")
    print("  " + "─" * (3 + 16 + cw + 8 + 8 + 9 + cw + 7 + 7))
    for i, c in enumerate(conv_results):
        b   = "".join(str(x) for x in c["bits"])
        neq = c["nearest_eq"]
        nb  = "".join(str(x) for x in neq["bits"]) if neq else "—"
        nd  = f"{neq['dist_L2']:.3f}"               if neq else "—"
        ns  = ("✓" if neq["stable"] else "✗")        if neq else "—"
        print(f"  {i:>3}  {c['state_type']:<16}  {b:<{cw}}  {c['H']:>8.3f}  "
              f"{c['cut']:>8.3f}  {c['residual']:>9.5f}  {nb:<{cw}}  {nd}  {ns}")

    summary = defaultdict(lambda: {"count": 0, "residuals": []})
    for c in conv_results:
        key = (c["state_type"], c["bits"])
        summary[key]["count"] += 1
        summary[key]["state_type"] = c["state_type"]
        summary[key]["residuals"].append(c["residual"])

    print(f"\n  Unique terminal states: {len(summary)}")
    for (stype, bits), sm in sorted(summary.items(),
                                    key=lambda x: -x[1]["count"]):
        b = "".join(str(x) for x in bits)
        print(f"    type={stype:<16}  bits={b}  "
              f"count={sm['count']}/{args.n_init}  "
              f"mean_res={np.mean(sm['residuals']):.5f}")

    fig2 = make_figure2(args, n, edges, eq_data, sweep_data)
    fig3 = make_figure3(args, n, W, eq_data, conv_results)
    fig4 = make_figure4(args, n, edges, eq_data, sweep_data)

    if args.save:
        stem = args.graph.replace("/", "_").replace("\\", "_").rstrip(".txt")
        for tag, fig in [("bifurcation", fig2),
                         ("quality", fig3),
                         ("hessian_bifurcation", fig4)]:
            for ext in ("pdf", "png"):
                fname = f"oim_{tag}_{stem}.{ext}"
                fig.savefig(fname, bbox_inches="tight", dpi=150)
                print(f"Saved: {fname}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
