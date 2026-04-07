#!/usr/bin/env python3
"""
eigenvalue_sweep_analysis.py
─────────────────────────────────────────────────────────────────────────────
Mu-sweep eigenvalue / bifurcation analysis for any graph fed to OIMMaxCut.

Three separate figures
──────────────────────
Figure 1 — Phase dynamics
  Left  | Phase trajectories θ_i(t) for all initial conditions
  Right | ODE convergence table (reached equilibria per trajectory)

Figure 2 — Bifurcation analysis
  Row 0 | λ_max(D) bar chart  — all 2^N equilibria, bars coloured by cut
  Row 1 | Bifurcation diagram — λ_max(D)−μ vs μ, annotated H & cut

Figure 3 — Quality analysis
  Left  | H vs cut scatter    — all 2^N equilibria, ODE hits overlaid
  Right | D(φ*) spectrum      — 3 representative equilibria

Mathematical background (Cheng et al., Chaos 34, 073103, 2024)
──────────────────────────────────────────────────────────────
A(φ*, μ) = D(φ*) − μ·I_N   →   λ_k(A) = λ_k(D) − μ   (slope −1 in μ)
Stability: μ > λ_max(D(φ*))
μ_bin = min_{φ*∈{0,π}^N} λ_max(D(φ*))
H(σ) = Σ_{i<j} W_ij σ_i σ_j = W_total − 2·cut
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
from itertools import product as iproduct

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from OIM_Experiment.src.OIM_mu import OIMMaxCut
from OIM_Experiment.src.graph_utils import read_graph

# ── global style ──────────────────────────────────────────────────────────────
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


# ── silent Jacobian ───────────────────────────────────────────────────────────
def _jacobian(oim: OIMMaxCut, phi_star: np.ndarray) -> np.ndarray:
    D = oim.build_D(phi_star)
    return D - oim.mu * np.diag(np.cos(2.0 * phi_star))


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

        sigma        = np.sign(np.cos(phi))
        sigma[sigma == 0] = 1.0
        H   = 0.5  * float(np.sum(W * np.outer(sigma, sigma)))
        cut = 0.25 * float(np.sum(W * (1.0 - np.outer(sigma, sigma))))

        rows.append({
            "bits": bits, "phi": phi, "D": D,
            "ev_D": ev_D, "ev_A": ev_A, "lmax_D": lmax,
            "lmax_A": float(ev_A[-1]), "mu_thr": lmax,
            "stable": mu > lmax, "H": H, "cut": cut,
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
        paths[key] = r["ev_D"][None, :] - mu_vals[:, None]   # (M, N)
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
# 3 representative equilibria spanning the λ_max(D) range
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
def identify_convergence(sol, W: np.ndarray, tol: float = 0.05):
    theta    = sol.y[:, -1]
    residual = float(np.max(np.abs(np.sin(theta))))
    sigma    = np.sign(np.cos(theta))
    sigma[sigma == 0] = 1.0
    bits = tuple(0 if s > 0 else 1 for s in sigma)
    H    = 0.5  * float(np.sum(W * np.outer(sigma, sigma)))
    cut  = 0.25 * float(np.sum(W * (1.0 - np.outer(sigma, sigma))))
    return bits, H, cut, residual < tol, residual


# ═════════════════════════════════════════════════════════════════════════════
# ── FIGURE 1  Phase dynamics + convergence table ──────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
def make_figure1(args, n, W, eq_data, sols, MU_SIM, conv_results):

    mu_bin   = eq_data["mu_bin"]
    w_total  = eq_data["w_total"]
    best_cut = eq_data["best_cut"]

    fig = plt.figure(figsize=(18, 7), facecolor=WHITE)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32,
                            left=0.06, right=0.97, top=0.88, bottom=0.13)
    ax_phase = fig.add_subplot(gs[0, :2])   # wide left
    ax_conv  = fig.add_subplot(gs[0, 2])    # table right

    # ── phase trajectories ────────────────────────────────────────────────────
    SPIN_COLS = plt.get_cmap("tab20")(np.linspace(0, 1, max(n, 2)))
    PI_TICKS  = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    PI_LABELS = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]

    t = sols[0].t
    for sol in sols:
        for spin in range(n):
            ax_phase.plot(t, sol.y[spin],
                          color=SPIN_COLS[spin % 20],
                          alpha=0.42, linewidth=1.0)

    for yref, lw_r in [(np.pi, 1.1), (0.0, 1.4), (-np.pi, 0.9)]:
        ax_phase.axhline(yref, color=GRAY, linestyle="--",
                         linewidth=lw_r, alpha=0.85)

    ax_phase.set_yticks(PI_TICKS)
    ax_phase.set_yticklabels(PI_LABELS, fontsize=10, color=BLACK)
    ax_phase.set_ylim(-4.2, 4.2)
    ax_phase.set_xlim(t[0], t[-1])

    is_bin = all(c[3] for c in conv_results)
    ax_phase.text(0.98, 0.97,
                  "BINARISED ✓" if is_bin else "NOT YET BINARISED ✗",
                  transform=ax_phase.transAxes, ha="right", va="top",
                  fontsize=11, fontweight="bold",
                  color=C_STABLE if is_bin else C_UNSTABLE,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE,
                            edgecolor=GRAY, alpha=0.95))
    ax_phase.text(0.01, 0.97,
                  f"$\\mu={MU_SIM:.4f}$  |  "
                  f"$\\mu_{{\\rm bin}}={mu_bin:.4f}$  |  "
                  f"$W_{{\\rm tot}}={w_total:.1f}$  |  "
                  f"Best cut $={best_cut:.1f}$",
                  transform=ax_phase.transAxes, ha="left", va="top",
                  fontsize=9.5,
                  bbox=dict(boxstyle="round,pad=0.28", facecolor=WHITE,
                            edgecolor=GRAY, alpha=0.93))

    spin_patches = [mpatches.Patch(color=SPIN_COLS[s % 20], label=f"spin {s}")
                    for s in range(n)]
    ax_phase.legend(handles=spin_patches, loc="lower right", fontsize=8,
                    ncol=max(1, n // 5), framealpha=0.90)
    _ax_style(ax_phase,
              title=(f"Phase dynamics  $\\mu={MU_SIM:.4f}$  "
                     f"({'above' if MU_SIM > mu_bin else 'below'} "
                     f"$\\mu_{{\\rm bin}}={mu_bin:.4f}$)  |  "
                     f"{args.n_init} random initial conditions"),
              xlabel="time $t$",
              ylabel="phase $\\theta_i(t)$  (rad)")

    # ── convergence table ─────────────────────────────────────────────────────
    ax_conv.set_facecolor(WHITE)
    ax_conv.axis("off")
    for sp in ax_conv.spines.values():
        sp.set_edgecolor(BLACK); sp.set_linewidth(0.8)
    ax_conv.set_title("ODE convergence — reached equilibria",
                      color=BLACK, fontsize=11, pad=6)

    cw    = max(n, 8)
    sep   = "─" * (3 + cw + 8 + 8 + 5 + 8)
    lines = [f"{'#':>3}  {'φ* bits':<{cw}}  {'H':>7}  "
             f"{'cut':>7}  {'bin?':>4}  res", sep]

    for i, (bits, H_c, cut_c, binarized, residual) in enumerate(conv_results):
        b  = "".join(str(x) for x in bits)
        bs = "✓" if binarized else "✗"
        lines.append(f"{i:>3}  {b:<{cw}}  {H_c:>7.2f}  "
                     f"{cut_c:>7.2f}  {bs:>4}  {residual:.3f}")

    unique_conv = {}
    for bits, H_c, cut_c, binarized, _ in conv_results:
        if bits not in unique_conv:
            unique_conv[bits] = {"H": H_c, "cut": cut_c, "count": 0}
        unique_conv[bits]["count"] += 1

    lines += ["", f"Unique equilibria reached: {len(unique_conv)}",
              f"{'φ* bits':<{cw}}  {'H':>7}  {'cut':>7}  {'n':>3}",
              "─" * (cw + 8 + 8 + 5)]
    for bits, info in sorted(unique_conv.items(), key=lambda x: -x[1]["cut"]):
        b = "".join(str(x) for x in bits)
        lines.append(f"{b:<{cw}}  {info['H']:>7.2f}  "
                     f"{info['cut']:>7.2f}  {info['count']:>3}")

    ax_conv.text(0.03, 0.97, "\n".join(lines),
                 transform=ax_conv.transAxes, va="top", ha="left",
                 fontsize=7.5, fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=WHITE,
                           edgecolor=GRAY, alpha=0.95))

    fig.suptitle(
        f"OIM Phase Dynamics  |  {args.graph}  |  "
        f"$N={n}$,  $\\mu_{{\\rm bin}}={mu_bin:.4f}$  |  "
        f"Best cut $={best_cut:.1f}$,  $W_{{\\rm tot}}={w_total:.1f}$",
        color=BLACK, fontsize=12, fontweight="bold")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# ── FIGURE 2  Bifurcation analysis ────────────────────────────────────────────
# Row 0 | λ_max(D) bar chart  — full width
# Row 1 | Bifurcation diagram — full width
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
                            height_ratios=[0.70, 1.30],
                            hspace=0.45,
                            left=0.06, right=0.96, top=0.92, bottom=0.08)
    ax_lmbar = fig.add_subplot(gs[0])
    ax_bif   = fig.add_subplot(gs[1])

    # ══════════════════════════════════════════════════════════════════════════
    # Row 0 — λ_max(D) bar chart
    # ══════════════════════════════════════════════════════════════════════════
    sorted_rows = sorted(rows, key=lambda r: r["lmax_D"])
    lmax_arr    = np.array([r["lmax_D"] for r in sorted_rows])
    cut_arr_s   = np.array([r["cut"]    for r in sorted_rows])
    bar_cols    = [cmap_cut(norm_cut(c)) for c in cut_arr_s]

    ax_lmbar.bar(np.arange(n_eq), lmax_arr,
                 color=bar_cols, width=1.0, edgecolor="none", zorder=2)
    ax_lmbar.axhline(0,          color=BLACK,     linewidth=0.9, zorder=3)
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
              title=(f"$\\lambda_{{\\max}}(D(\\phi^*))$ for all $2^N = {n_eq}$ equilibria "
                     f"(sorted by $\\lambda_{{\\max}}$)  —  "
                     f"bar colour = cut quality  |  "
                     f"stable at current $\\mu$: {eq_data['n_stable']}/{n_eq}"),
              xlabel="Equilibrium index",
              ylabel="$\\lambda_{\\max}(D(\\phi^*))$")

    # ══════════════════════════════════════════════════════════════════════════
    # Row 1 — Bifurcation diagram
    # ══════════════════════════════════════════════════════════════════════════
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
                   label="$\\lambda_{\\max}(A) = 0$  (stability boundary)")
    ax_bif.axvline(current_mu, color=C_MU_LINE, linewidth=1.8,
                   linestyle="--", zorder=6,
                   label=f"current $\\mu = {current_mu:.3f}$")
    ax_bif.axvline(mu_bin, color=BLACK, linewidth=1.8,
                   linestyle=":", zorder=7,
                   label=f"$\\mu_{{\\rm bin}} = {mu_bin:.3f}$")

    # spread annotation y-positions to avoid overlap
    n_unique   = len(unique_lmax)
    ann_y_vals = np.linspace(bif_hi * 0.95, bif_hi * 0.05, n_unique)

    for idx, (lm, cnt, H_mean, cut_mean) in enumerate(
            zip(unique_lmax, counts_lmax, avg_H_per_lmax, avg_cut_per_lmax)):
        c  = cmap_bif(norm_bif(cut_mean))
        lw = 0.8 + 0.55 * np.log1p(cnt / 2.0)
        ax_bif.plot(mu_vals, lm - mu_vals, color=c, linewidth=lw, alpha=0.85)

        mu_star = lm
        if mu_vals[0] <= mu_star <= mu_vals[-1]:
            ax_bif.scatter([mu_star], [0.0],
                           color=c, s=55, zorder=7,
                           edgecolors=BLACK, linewidths=0.7)
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

    # right axis: # stable equilibria
    ax_bif2 = ax_bif.twinx()
    ax_bif2.plot(mu_vals, sweep_data["n_stable_mu"],
                 color=C_STABLE, linewidth=2.5, linestyle="-",
                 alpha=0.90, zorder=4, label="# stable eq.")
    ax_bif2.set_ylabel("# stable equilibria", color=C_STABLE, fontsize=11)
    ax_bif2.tick_params(axis="y", colors=C_STABLE, labelsize=9)
    ax_bif2.set_ylim(-0.5, n_eq + 0.5)
    for sp in ax_bif2.spines.values():
        sp.set_edgecolor(BLACK); sp.set_linewidth(0.8)
    ax_bif2.legend(fontsize=9, loc="center right")

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
              title=("Bifurcation diagram:  $\\lambda_{\\max}(D) - \\mu$  vs  $\\mu$  |  "
                     "line colour = cut quality  |  "
                     "dots annotated with $\\bar{H}$, cut, multiplicity  |  "
                     "right axis = # stable equilibria"),
              xlabel="$\\mu$  (SHIL / binarisation parameter)",
              ylabel="$\\lambda_{\\max}(A) = \\lambda_{\\max}(D) - \\mu$")

    fig.suptitle(
        f"OIM Bifurcation Analysis  |  {args.graph}  |  "
        f"$N={n}$,  $|E|={len(edges)}$,  $2^N={n_eq}$ equilibria  |  "
        f"$\\mu_{{\\rm bin}}={mu_bin:.4f}$  |  "
        f"Best cut $={best_cut:.1f}$,  $W_{{\\rm tot}}={w_total:.1f}$",
        color=BLACK, fontsize=12, fontweight="bold")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# ── FIGURE 3  Quality analysis ────────────────────────────────────────────────
# Left  | H vs cut scatter
# Right | D(φ*) eigenvalue spectrum
# ═════════════════════════════════════════════════════════════════════════════
def make_figure3(args, n, W, eq_data, conv_results):

    rows       = eq_data["rows"]
    best_cut   = eq_data["best_cut"]
    w_total    = eq_data["w_total"]
    mu_bin     = eq_data["mu_bin"]
    current_mu = eq_data["mu"]
    reps       = pick_representatives(rows)

    H_all       = np.array([r["H"]      for r in rows])
    cut_all     = np.array([r["cut"]    for r in rows])
    stable_mask = np.array([r["stable"] for r in rows])

    fig, (ax_hcut, ax_dspec) = plt.subplots(
        1, 2, figsize=(16, 7), facecolor=WHITE)
    fig.subplots_adjust(wspace=0.32, left=0.07, right=0.97,
                        top=0.88, bottom=0.12)

    # ══════════════════════════════════════════════════════════════════════════
    # Left — H vs cut scatter
    # ══════════════════════════════════════════════════════════════════════════
    ax_hcut.scatter(cut_all[~stable_mask], H_all[~stable_mask],
                    c=C_UNSTABLE, s=20, alpha=0.45, zorder=2,
                    label=f"unstable at $\\mu={current_mu:.2f}$")
    ax_hcut.scatter(cut_all[stable_mask], H_all[stable_mask],
                    c=C_STABLE, s=45, alpha=0.85, zorder=3,
                    edgecolors=BLACK, linewidths=0.5,
                    label=f"stable at $\\mu={current_mu:.2f}$")

    cut_line = np.linspace(cut_all.min() - 0.3, cut_all.max() + 0.3, 300)
    ax_hcut.plot(cut_line, w_total - 2.0 * cut_line,
                 color=GRAY, linewidth=1.3, linestyle="-", alpha=0.65,
                 zorder=1, label=f"$H = {w_total:.0f} - 2\\cdot$cut")

    ax_hcut.axvline(best_cut,    color=C_BPART, linewidth=1.6,
                    linestyle="--", alpha=0.8,
                    label=f"best cut $={best_cut:.1f}$")
    ax_hcut.axhline(H_all.min(), color=C_FERRO, linewidth=1.6,
                    linestyle="--", alpha=0.8,
                    label=f"min $H={H_all.min():.1f}$")

    # overlay ODE convergence (unique equilibria only)
    seen = set()
    for bits, H_c, cut_c, binarized, _ in conv_results:
        if bits in seen:
            continue
        seen.add(bits)
        mk  = "*" if binarized else "^"
        col = C_STABLE if binarized else C_UNSTABLE
        ax_hcut.scatter([cut_c], [H_c], marker=mk,
                        s=180 if binarized else 110,
                        color=col, edgecolors=BLACK,
                        linewidths=0.8, zorder=6)
        b_str = "".join(str(b) for b in bits)
        ax_hcut.annotate(
            f"$[{b_str}]$",
            xy=(cut_c, H_c),
            xytext=(cut_c + best_cut * 0.025,
                    H_c - (H_all.max() - H_all.min()) * 0.04),
            fontsize=9, color=col,
            arrowprops=dict(arrowstyle="->", color=col, lw=0.7,
                            shrinkA=2, shrinkB=2))

    star_p = mpatches.Patch(color=C_STABLE,   label="ODE → binarized (★)")
    tri_p  = mpatches.Patch(color=C_UNSTABLE, label="ODE → not binarized (▲)")
    hnd, lbl = ax_hcut.get_legend_handles_labels()
    ax_hcut.legend(handles=hnd + [star_p, tri_p], fontsize=9, loc="upper right")
    _ax_style(ax_hcut,
              title=(f"Ising Hamiltonian $H$ vs cut value  —  "
                     f"all $2^N={len(rows)}$ equilibria\n"
                     f"$H = W_{{\\rm tot}} - 2\\cdot$cut  |  "
                     f"$W_{{\\rm tot}}={w_total:.1f}$  |  "
                     f"best cut $={best_cut:.1f}$  |  "
                     f"$\\mu_{{\\rm bin}}={mu_bin:.3f}$"),
              xlabel="Cut value",
              ylabel="$H(\\sigma)$",
              titlesize=11)

    # ══════════════════════════════════════════════════════════════════════════
    # Right — D(φ*) eigenvalue spectrum
    # ══════════════════════════════════════════════════════════════════════════
    xidx        = np.arange(1, n + 1)
    ev_all_reps = np.concatenate([eq["ev_D"] for eq, _, _, _ in reps])
    ev_lo, ev_hi = ev_all_reps.min(), ev_all_reps.max()
    margin = max(0.06 * (ev_hi - ev_lo), 0.4)

    ax_dspec.fill_between([0.5, n + 0.5], ev_lo - margin, 0,
                          color=C_STABLE,   alpha=0.10, zorder=0)
    ax_dspec.fill_between([0.5, n + 0.5], 0, ev_hi + margin,
                          color=C_UNSTABLE, alpha=0.10, zorder=0)
    ax_dspec.axhline(0, color=BLACK, linewidth=1.0, linestyle="--",
                     alpha=0.55, zorder=2)

    for k_r, (eq, lbl, col, ls) in enumerate(reps):
        ev_desc = eq["ev_D"][::-1]
        ax_dspec.plot(xidx, ev_desc, color=col, linestyle=ls,
                      linewidth=2.2, marker="o", markersize=8,
                      label=lbl, zorder=3)
        dy = margin * (0.6 + 0.35 * k_r)
        ax_dspec.annotate(
            f"$\\lambda_{{\\max}}={eq['lmax_D']:.3f}$\n"
            f"$H={eq['H']:.1f}$   cut$={eq['cut']:.1f}$",
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
                  "$A(\\phi^*,\\mu) = D(\\phi^*) - \\mu I$\n"
                  "$\\lambda_k(A) = \\lambda_k(D) - \\mu$\n"
                  "Stable iff $\\mu > \\lambda_{\\max}(D)$",
                  transform=ax_dspec.transAxes, ha="right", va="top",
                  fontsize=9.5,
                  bbox=dict(boxstyle="round,pad=0.35", facecolor=WHITE,
                            edgecolor=GRAY, alpha=0.94))
    _ax_style(ax_dspec,
              title=("$D(\\phi^*)$ eigenvalue spectrum  —  "
                     "3 representative equilibria\n"
                     "Blue shading: $\\lambda<0$  |  "
                     "Orange: $\\lambda>0$  |  "
                     "Annotation: $\\lambda_{\\max}$, $H$, cut"),
              xlabel="Eigenvalue rank $k$  (largest first)",
              ylabel="$\\lambda_k\\left((D(\\phi^*)\\right)$",
              titlesize=11)

    fig.suptitle(
        f"OIM Quality Analysis  |  {args.graph}  |  "
        f"$N={n}$,  $2^N={len(rows)}$ equilibria  |  "
        f"$\\mu_{{\\rm bin}}={mu_bin:.4f}$  |  "
        f"Best cut $={best_cut:.1f}$,  $W_{{\\rm tot}}={w_total:.1f}$",
        color=BLACK, fontsize=12, fontweight="bold")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="OIM eigenvalue sweep & bifurcation analysis — 3 figures")
    parser.add_argument("--graph",  required=True)
    parser.add_argument("--mu_min", type=float, default=None)
    parser.add_argument("--mu_max", type=float, default=None)
    parser.add_argument("--n_mu",   type=int,   default=300)
    parser.add_argument("--mu",     type=float, default=None)
    parser.add_argument("--n_init", type=int,   default=12)
    parser.add_argument("--t_end",  type=float, default=80.0)
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--save",   action="store_true",
                        help="Save all 3 figures as PDF+PNG")
    args = parser.parse_args()

    # ── load ──────────────────────────────────────────────────────────────────
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

    # ── analysis ──────────────────────────────────────────────────────────────
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

    # ── simulation ────────────────────────────────────────────────────────────
    MU_SIM  = max(global_lmax_min + 0.01, mu_min_eff + 0.02)
    rng     = np.random.default_rng(args.seed)
    phi0s   = [rng.uniform(-0.08, 0.08, n) for _ in range(args.n_init)]
    oim_sim = OIMMaxCut(W, mu=MU_SIM, seed=args.seed)
    print(f"\n  Simulating {args.n_init} trajectories at "
          f"μ={MU_SIM:.4f}  (t=0..{args.t_end})...")
    sols = oim_sim.simulate_many(phi0s, t_span=(0., args.t_end), n_points=500)
    print("  Done.\n")

    # ── convergence ───────────────────────────────────────────────────────────
    cw = max(n, 8)
    print(f"  {'#':>3}  {'φ* bits':<{cw}}  {'H':>8}  "
          f"{'cut':>8}  {'bin?':>5}  residual")
    print("  " + "─" * (3 + cw + 8 + 8 + 5 + 12))
    conv_results = []
    for i, sol in enumerate(sols):
        bits, H_c, cut_c, binarized, residual = identify_convergence(sol, W)
        conv_results.append((bits, H_c, cut_c, binarized, residual))
        b = "".join(str(x) for x in bits)
        print(f"  {i:>3}  {b:<{cw}}  {H_c:>8.3f}  {cut_c:>8.3f}  "
              f"{'✓ yes' if binarized else '✗ no ':>5}  {residual:.4f}")

    unique_conv = {}
    for bits, H_c, cut_c, binarized, _ in conv_results:
        if bits not in unique_conv:
            unique_conv[bits] = {"H": H_c, "cut": cut_c, "count": 0}
        unique_conv[bits]["count"] += 1
    print(f"\n  Unique equilibria reached: {len(unique_conv)}")
    for bits, info in sorted(unique_conv.items(), key=lambda x: -x[1]["cut"]):
        b = "".join(str(x) for x in bits)
        print(f"    φ*={b}  H={info['H']:.3f}  "
              f"cut={info['cut']:.3f}  "
              f"count={info['count']}/{args.n_init}")

    # ── figures ───────────────────────────────────────────────────────────────
    fig1 = make_figure1(args, n, W, eq_data, sols, MU_SIM, conv_results)
    fig2 = make_figure2(args, n, edges, eq_data, sweep_data)
    fig3 = make_figure3(args, n, W, eq_data, conv_results)

    if args.save:
        stem = args.graph.replace("/", "_").replace("\\", "_").rstrip(".txt")
        for tag, fig in [("dynamics", fig1),
                         ("bifurcation", fig2),
                         ("quality", fig3)]:
            for ext in ("pdf", "png"):
                fname = f"oim_{tag}_{stem}.{ext}"
                fig.savefig(fname, bbox_inches="tight", dpi=150)
                print(f"Saved: {fname}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
