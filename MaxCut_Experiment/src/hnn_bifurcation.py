#!/usr/bin/env python3
"""
hnn_bifurcation.py
──────────────────────────────────────────────────────────────────────────────
Bifurcation analysis for the Continuous Hopfield Network (HNN) applied to
Max-Cut — the exact structural analogue of Bashar et al. (2023) Figures 2–3
for the OIM.

Mathematical derivation
───────────────────────
The HNN energy at a spin configuration σ ∈ {±1}^N is the Lyapunov function

    E(s; u0) = ½ sᵀWs + u0 Σᵢ [sᵢ arctanh(sᵢ) + ½ log(1 − sᵢ²)]

whose Hessian in s-space evaluated at σ (Allibhoy et al. 2025, Eq. 9) is

    H(σ, u0) = L(σ) + 2u0 · I_N

where  L(σ) = diag(A(σ)·1) − A(σ)  is the signed Laplacian of the signed
graph Gₛ(σ), with signed adjacency  A_ij(σ) = J_ij σᵢσⱼ = −W_ij σᵢσⱼ.

Because L(σ) does NOT depend on u0, all N eigenvalues shift rigidly:

    λk(H(σ, u0)) = λk(L(σ)) + 2u0        [straight lines, slope +2]

Stability of σ requires all eigenvalues > 0, i.e.

    u0 > −λmin(L(σ)) / 2  =:  u0*(σ)    [per-equilibrium threshold]

The global binarisation threshold is

    u0_bin = max_{σ} u0*(σ) = |λmin(W)|   (origin instability threshold)

This is equivalent to:
    u0 < u0_bin  →  origin unstable (binarises to corners)
    u0 > u0*(σ)  →  equilibrium σ is stable

Comparison with OIM (Bashar 2023)
──────────────────────────────────
    OIM Jacobian:  J(φ*,Ks) = D(φ*) − Ks·I    [slope −1, lines go DOWN]
    HNN Hessian:   H(σ, u0) = L(σ)  + 2u0·I   [slope +2, lines go UP ]

    OIM threshold for σ:  Ks*(σ) = λmax(D(φ*))
    HNN threshold for σ:  u0*(σ) = −λmin(L(σ))/2  = Ks*(σ)/2

    Key structural difference:
      OIM — increasing Ks DESTABILISES configs selectively (high-threshold
            first), creating a Goldilocks zone around μ_bin.
      HNN — increasing u0 STABILISES more and more configs simultaneously,
            so no selective stabilisation window exists.

Figures
───────
Figure 1 — λmin(H) paths vs u0  (analogue of Bashar Fig. 2)
    Left  : All 2^N configurations, each a straight line λmin(L(σ)) + 2u0.
            Coloured by cut quality (RdYlGn colormap).
            Horizontal dashed line at λ=0.  Vertical line at u0_bin.
    Right : Zoomed panel — only globally optimal configurations.
            Shows how many (if any) are already stable at u0_bin.

Figure 2 — λmin(H) vs Ising energy H(σ)  (analogue of Bashar Fig. 3)
    Three columns for three u0 values  [0.01·u0_bin, u0_bin, 3·u0_bin].
    For each unique energy level h:
      • orange diamond = max  λmin(H(σ, u0))  over all σ with H(σ)=h
      • blue  circle   = min  λmin(H(σ, u0))  over all σ with H(σ)=h
    Red box marks the globally optimal energy level.
    Horizontal dashed line at λ=0 (stability boundary).

Figure 3 — Per-configuration threshold scatter  (new, no OIM equivalent)
    x-axis : u0*(σ) = per-equilibrium stability threshold
    y-axis : H(σ)   = Ising energy (lower = better cut)
    Colour : cut value
    Annotations: global optimum configurations highlighted in green.
    This shows whether optimal configs have lower thresholds than
    suboptimal ones — the structural support (or refutation) of the
    conjecture that optimal configs are easiest to stabilise.

Usage
─────
  python hnn_bifurcation.py --graph data/10node.txt
  python hnn_bifurcation.py --n 12 --seed 42
  python hnn_bifurcation.py --graph data/10node.txt --u0_max 8.0 --save

CLI options
───────────
  --graph     PATH    graph file (N  /  u v [w]  format)
  --n         INT     random ER graph of size N  (overridden by --graph)
  --p1        FLOAT   ER edge probability  (default: 0.5)
  --seed      INT     RNG seed  (default: 42)
  --u0_min    FLOAT   left edge of u0 sweep  (default: 0)
  --u0_max    FLOAT   right edge  (default: auto = 3 × u0_bin)
  --n_u0      INT     number of u0 points  (default: 300)
  --save      FLAG    save all figures as PDF + PNG
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import sys
import time
from itertools import product as iproduct

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch

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


def _ax_style(ax, title="", xlabel="", ylabel="", titlesize=11):
    ax.set_facecolor(WHITE)
    ax.tick_params(colors=BLACK, labelsize=10)
    for sp in ax.spines.values():
        sp.set_edgecolor(BLACK); sp.set_linewidth(0.8)
    ax.grid(True, color=LIGHT, linewidth=0.6, zorder=0)
    if title:  ax.set_title(title,  color=BLACK, fontsize=titlesize,
                             fontweight="bold", pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=BLACK, fontsize=11)
    if ylabel: ax.set_ylabel(ylabel, color=BLACK, fontsize=11)


# ═══════════════════════════════════════════════════════════════════════════════
# Graph I/O
# ═══════════════════════════════════════════════════════════════════════════════

def read_graph(filepath: str) -> np.ndarray:
    """Read graph from  N / u v [w]  format (same as OIM experiments)."""
    with open(filepath) as f:
        lines = [l.strip() for l in f
                 if l.strip() and not l.startswith("#")]
    n = int(lines[0])
    W = np.zeros((n, n), dtype=float)
    for line in lines[1:]:
        parts = line.split()
        u, v  = int(parts[0]), int(parts[1])
        w     = float(parts[2]) if len(parts) > 2 else 1.0
        W[u, v] = W[v, u] = w
    return W


def generate_er_graph(n: int, p1: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W   = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p1:
                W[i, j] = W[j, i] = 1.0
    return W


# ═══════════════════════════════════════════════════════════════════════════════
# Core computation: signed Laplacian and per-equilibrium threshold
# ═══════════════════════════════════════════════════════════════════════════════

def signed_laplacian(W: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Signed Laplacian  L(σ) = diag(A·1) − A  where  A_ij = J_ij σᵢσⱼ = −W_ij σᵢσⱼ.

    Properties
    ----------
    • H(σ, u0) = L(σ) + 2u0·I   [Allibhoy et al. 2025, Eq. 9]
    • H(σ) = −½ tr(L(σ))         [Ising Hamiltonian from trace]
    • L(σ) ≥ 0 iff σ is a global minimiser (frustration-free case)
    """
    N    = W.shape[0]
    A    = -W * np.outer(sigma, sigma)   # signed adjacency  A_ij = J_ij σᵢσⱼ
    np.fill_diagonal(A, 0.0)
    deg  = A.sum(axis=1)
    return np.diag(deg) - A


def enumerate_equilibria(W: np.ndarray) -> dict:
    """
    Enumerate all 2^N spin configurations σ ∈ {±1}^N and compute for each:
      • H(σ)      — Ising Hamiltonian  = ½ σᵀWσ  (= −½ tr L(σ))
      • cut(σ)    — Max-Cut value  = ¼ Σ Wᵢⱼ(1 − σᵢσⱼ)
      • eigs_L    — all N eigenvalues of L(σ), sorted ascending
      • u0_thr    — per-equilibrium stability threshold  = max(0, −λmin(L)/2)
      • lmin_L    — λmin(L(σ))

    The entire bifurcation family  λk(H(σ,u0)) = λk(L(σ)) + 2u0  is then
    reconstructed for free from eigs_L for any u0 grid.

    Returns a dict with parallel arrays of length 2^N.
    """
    N       = W.shape[0]
    n_eq    = 1 << N
    w_total = float(np.sum(W)) / 2.0

    H_arr      = np.empty(n_eq)
    cut_arr    = np.empty(n_eq)
    u0thr_arr  = np.empty(n_eq)
    lmin_arr   = np.empty(n_eq)
    eigs_arr   = np.empty((n_eq, N))    # sorted ascending
    sigma_arr  = np.empty((n_eq, N))

    print(f"\n  Enumerating all 2^{N} = {n_eq} spin configurations...")
    t0 = time.perf_counter()

    for idx, bits in enumerate(iproduct([0, 1], repeat=N)):
        sigma = np.array([1.0 if b == 0 else -1.0 for b in bits])
        L     = signed_laplacian(W, sigma)
        eigs  = np.linalg.eigvalsh(L)          # ascending order

        cut   = 0.25 * float(np.sum(W * (1.0 - np.outer(sigma, sigma))))
        H_val = 0.5  * float(sigma @ W @ sigma)   # = −½ tr(L)

        H_arr[idx]     = H_val
        cut_arr[idx]   = cut
        eigs_arr[idx]  = eigs
        lmin_arr[idx]  = eigs[0]
        u0thr_arr[idx] = max(0.0, -eigs[0] / 2.0)
        sigma_arr[idx] = sigma

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s")

    best_cut   = float(cut_arr.max())
    opt_mask   = np.abs(cut_arr - best_cut) < 1e-9
    H_min      = float(H_arr.min())
    u0_bin     = float(abs(np.linalg.eigvalsh(W)[0]))   # global threshold

    print(f"  Best cut = {best_cut:.1f}  |  "
          f"# optimal configs = {opt_mask.sum()}  |  "
          f"H_min = {H_min:.2f}")
    print(f"  u0_bin = |λmin(W)| = {u0_bin:.4f}")
    print(f"  u0* range: [{u0thr_arr.min():.4f}, {u0thr_arr.max():.4f}]")

    return dict(
        N=N, n_eq=n_eq, w_total=w_total,
        H=H_arr, cut=cut_arr,
        eigs_L=eigs_arr, lmin_L=lmin_arr, u0_thr=u0thr_arr,
        sigma=sigma_arr,
        best_cut=best_cut, H_min=H_min,
        opt_mask=opt_mask,
        u0_bin=u0_bin,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — λmin(H) paths vs u0  (Bashar Fig. 2 analogue)
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure1(eq: dict, u0_vals: np.ndarray, graph_name: str) -> plt.Figure:
    """
    Two-panel figure:
      Left  — all 2^N straight-line paths  λmin(H(σ,u0)) = λmin(L(σ)) + 2u0
              coloured by cut quality, dashed λ=0 line, u0_bin vertical.
      Right — zoomed to globally optimal configurations only.

    Each configuration σ contributes exactly one path (its λmin path),
    which is a straight line with slope +2 and y-intercept λmin(L(σ)).
    Crossing zero at u0*(σ) = −λmin(L(σ))/2.
    """
    N       = eq["N"]
    n_eq    = eq["n_eq"]
    u0_bin  = eq["u0_bin"]
    best_cut = eq["best_cut"]
    opt_mask = eq["opt_mask"]
    lmin_L   = eq["lmin_L"]      # λmin(L(σ)) for every config, shape (n_eq,)
    cut_arr  = eq["cut"]

    # λmin(H(σ, u0)) = λmin(L(σ)) + 2u0  — shape (n_eq, n_u0)
    lmin_H = lmin_L[:, None] + 2.0 * u0_vals[None, :]

    cmap_cut = plt.get_cmap("RdYlGn")
    norm_cut = mcolors.Normalize(vmin=0, vmax=best_cut)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=WHITE)
    fig.subplots_adjust(wspace=0.35, left=0.07, right=0.97,
                        top=0.88, bottom=0.10)

    for col, (ax, mask, subtitle) in enumerate(zip(
        axes,
        [np.ones(n_eq, dtype=bool), opt_mask],
        [f"All $2^N = {n_eq}$ spin configurations",
         f"Globally optimal configurations only ({opt_mask.sum()} configs)"]
    )):
        # ── Draw paths ───────────────────────────────────────────────────────
        # Sort by cut value so high-quality lines are drawn on top
        idx_sorted = np.argsort(cut_arr[mask])
        mask_idx   = np.where(mask)[0][idx_sorted]

        lw   = 0.55 if mask.sum() > 50 else 1.5
        alpha = 0.40 if mask.sum() > 50 else 0.80

        for k in mask_idx:
            c   = cmap_cut(norm_cut(cut_arr[k]))
            lw_ = 1.8 if opt_mask[k] else lw
            al_ = 0.90 if opt_mask[k] else alpha
            ax.plot(u0_vals, lmin_H[k], color=c,
                    linewidth=lw_, alpha=al_, zorder=3)

        # ── λ = 0 stability boundary ─────────────────────────────────────────
        ax.axhline(0, color=BLACK, linewidth=1.8, linestyle="--",
                   zorder=5, label="$\\lambda = 0$  (stability boundary)")

        # ── u0_bin vertical line ──────────────────────────────────────────────
        ax.axvline(u0_bin, color=C_RED, linewidth=2.0, linestyle="--",
                   zorder=6,
                   label=f"$u_{{0,\\rm bin}} = {u0_bin:.3f}$\n"
                         f"$= |\\lambda_{{\\min}}(W)|$")

        # ── Shade stable / unstable regions ──────────────────────────────────
        y_lo = lmin_H[mask].min() - 0.5
        y_hi = lmin_H[mask].max() + 0.5
        ax.fill_between(u0_vals, y_lo, 0,
                        color=C_ORANGE, alpha=0.06, zorder=0,
                        label="Unstable region  ($\\lambda < 0$)")
        ax.fill_between(u0_vals, 0, y_hi,
                        color=C_GREEN,  alpha=0.06, zorder=0,
                        label="Stable region  ($\\lambda > 0$)")

        # ── How many configs are stable at u0_bin? ────────────────────────────
        n_stable_at_bin = int(np.sum(lmin_H[:, np.argmin(np.abs(u0_vals - u0_bin))] > 0))
        ax.text(0.97, 0.03,
                f"Stable at $u_{{0,\\rm bin}}$: {n_stable_at_bin}/{n_eq}\n"
                f"($\\lambda_{{\\min}}(H) > 0$)",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=WHITE,
                          edgecolor=GRAY, alpha=0.95))

        ax.set_xlim(u0_vals[0], u0_vals[-1])
        ax.set_ylim(y_lo, y_hi)
        ax.legend(fontsize=9, loc="upper left")
        _ax_style(ax,
                  title=subtitle,
                  xlabel="gain  $u_0$",
                  ylabel="$\\lambda_{\\min}(H(\\sigma, u_0)) = "
                         "\\lambda_{\\min}(L(\\sigma)) + 2u_0$")

        # ── Right-side annotation explaining the slope ────────────────────────
        if col == 0:
            ax.text(0.97, 0.97,
                    "Each line: one spin config $\\sigma$\n"
                    "Slope = $+2$ (fixed, same for all)\n"
                    "y-intercept = $\\lambda_{\\min}(L(\\sigma))$\n"
                    "Crosses 0 at $u_0^*(\\sigma) = -\\lambda_{\\min}(L)/2$\n\n"
                    "Colour = cut quality  (green = high cut)",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor=WHITE,
                              edgecolor=GRAY, alpha=0.95))

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_cut, norm=norm_cut)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.02, shrink=0.75)
    cbar.set_label("Cut value", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_edgecolor(BLACK)

    fig.suptitle(
        f"HNN Bifurcation Analysis — $\\lambda_{{\\min}}(H(\\sigma,u_0))$ vs $u_0$  |  "
        f"{graph_name}  |  $N={N}$,  $2^N={n_eq}$ equilibria  |  "
        f"best cut $= {best_cut:.1f}$  |  "
        f"$u_{{0,\\rm bin}} = {u0_bin:.4f}$",
        color=BLACK, fontsize=12, fontweight="bold")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — λmin(H) vs Ising energy H(σ)  (Bashar Fig. 3 analogue)
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure2(eq: dict, u0_vals: np.ndarray, graph_name: str) -> plt.Figure:
    """
    3 panels — one per representative u0 value:
      0.01 · u0_bin  →  "No configuration stable"  (sub-threshold)
      1.0  · u0_bin  →  "Some configurations stable"  (at threshold)
      3.0  · u0_bin  →  "All configurations stable"  (above threshold)

    For each energy level h:
      orange diamond = max  λmin(H(σ, u0))  over all σ with H(σ) = h
      blue   circle  = min  λmin(H(σ, u0))  over all σ with H(σ) = h
    Red box marks the globally optimal energy level.

    Relationship to Bashar Fig. 3:
      - OIM: x = H(σ), y = max/min λmax(J(φ*))  across configs at energy h
      - HNN: x = H(σ), y = max/min λmin(H(σ,u0)) across configs at energy h
      The sign convention is flipped: for OIM, stability ↔ λmax(J) < 0;
      for HNN, stability ↔ λmin(H) > 0.
    """
    N        = eq["N"]
    n_eq     = eq["n_eq"]
    u0_bin   = eq["u0_bin"]
    H_arr    = eq["H"]
    lmin_L   = eq["lmin_L"]
    best_cut = eq["best_cut"]
    H_min    = eq["H_min"]

    # Three representative u0 values
    u0_refs = [
        max(u0_vals[0], 0.01 * u0_bin),
        u0_bin,
        3.0 * u0_bin,
    ]
    u0_labels = [
        f"$u_0 = 0.01 \\cdot u_{{0,\\rm bin}}$\nNo configs stable",
        f"$u_0 = u_{{0,\\rm bin}} = {u0_bin:.3f}$\nSome configs stable",
        f"$u_0 = 3 \\cdot u_{{0,\\rm bin}}$\nAll configs stable",
    ]

    # Unique energy levels (round to avoid float noise)
    H_rounded      = np.round(H_arr, 4)
    unique_H       = np.unique(H_rounded)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=WHITE)
    fig.subplots_adjust(wspace=0.38, left=0.07, right=0.97,
                        top=0.86, bottom=0.13)

    # Global y-limits across all panels for visual consistency
    lmin_all_refs = np.array([lmin_L + 2.0 * u0 for u0 in u0_refs])
    y_lo = lmin_all_refs.min() - 0.3
    y_hi = lmin_all_refs.max() + 0.3

    for col, (u0_ref, lbl) in enumerate(zip(u0_refs, u0_labels)):
        ax = axes[col]

        # λmin(H(σ, u0_ref)) for every config
        lmin_H_ref = lmin_L + 2.0 * u0_ref   # shape (n_eq,)

        # Per energy level: min and max of λmin(H)
        H_max_lam, H_min_lam = [], []
        H_plot = []
        for h in unique_H:
            mask_h = H_rounded == h
            vals   = lmin_H_ref[mask_h]
            H_plot.append(h)
            H_max_lam.append(vals.max())
            H_min_lam.append(vals.min())

        H_plot    = np.array(H_plot)
        H_max_lam = np.array(H_max_lam)
        H_min_lam = np.array(H_min_lam)

        # Plot as scatter (mirrors Bashar Fig. 3 scatter style)
        ax.scatter(H_plot, H_max_lam,
                   color=C_ORANGE, marker="D", s=55, zorder=4,
                   edgecolors=BLACK, linewidths=0.5,
                   label=f"Largest $\\lambda_{{\\min}}$ at energy $h$")
        ax.scatter(H_plot, H_min_lam,
                   color=C_BLUE,   marker="o", s=45, zorder=4,
                   edgecolors=BLACK, linewidths=0.5,
                   label=f"Smallest $\\lambda_{{\\min}}$ at energy $h$")

        # Connect max/min with thin vertical lines for readability
        for h, lmax_v, lmin_v in zip(H_plot, H_max_lam, H_min_lam):
            ax.plot([h, h], [lmin_v, lmax_v],
                    color=GRAY, linewidth=0.6, zorder=2, alpha=0.60)

        # λ = 0 boundary
        ax.axhline(0, color=BLACK, linewidth=1.5, linestyle="--",
                   zorder=5, alpha=0.8, label="$\\lambda = 0$")

        # Red box: globally optimal energy level
        opt_h   = H_min
        h_step  = (H_plot[1] - H_plot[0]) if len(H_plot) > 1 else 1.0
        box_w   = abs(h_step) * 1.1
        box_lo  = min(H_min_lam[H_plot == opt_h].min() if (H_plot == opt_h).any() else -1,
                      -0.5)
        box_hi  = max(H_max_lam[H_plot == opt_h].max() if (H_plot == opt_h).any() else 1,
                       0.5)
        rect = plt.Rectangle(
            (opt_h - box_w / 2, box_lo - 0.3),
            box_w, (box_hi - box_lo + 0.6),
            linewidth=2.0, edgecolor=C_RED, facecolor="none",
            zorder=6, label=f"Global opt  ($H = {opt_h:.1f}$)")
        ax.add_patch(rect)

        # Stability annotation
        n_stable = int(np.sum(lmin_H_ref > 0))
        n_opt_stable = int(np.sum(
            (lmin_H_ref > 0) & (np.abs(H_arr - H_min) < 1e-6)))
        ax.text(0.97, 0.97,
                f"$u_0 = {u0_ref:.4f}$\n"
                f"Stable configs: {n_stable}/{n_eq}\n"
                f"Stable opt configs: {n_opt_stable}/{int(eq['opt_mask'].sum())}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=WHITE,
                          edgecolor=GRAY, alpha=0.95))

        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(H_plot.min() - 1, H_plot.max() + 1)
        ax.legend(fontsize=8, loc="lower right")
        _ax_style(ax,
                  title=lbl,
                  xlabel="Ising energy  $H(\\sigma) = \\frac{1}{2}\\sigma^\\top W\\sigma$",
                  ylabel="$\\lambda_{\\min}(H(\\sigma, u_0))$")

        # Shade stable/unstable
        ax.fill_between([H_plot.min() - 1, H_plot.max() + 1],
                        y_lo, 0,
                        color=C_ORANGE, alpha=0.07, zorder=0)
        ax.fill_between([H_plot.min() - 1, H_plot.max() + 1],
                        0, y_hi,
                        color=C_GREEN, alpha=0.07, zorder=0)
        ax.text(0.03, 0.06, "UNSTABLE\n$\\lambda_{\\min}<0$",
                transform=ax.transAxes, fontsize=8, color=C_ORANGE,
                fontweight="bold")
        ax.text(0.03, 0.88, "STABLE\n$\\lambda_{\\min}>0$",
                transform=ax.transAxes, fontsize=8, color=C_GREEN,
                fontweight="bold")

    fig.suptitle(
        f"HNN Bifurcation — $\\lambda_{{\\min}}(H(\\sigma,u_0))$ vs Ising energy  |  "
        f"{graph_name}  |  $N={N}$,  $2^N={n_eq}$ equilibria  |  "
        f"$u_{{0,\\rm bin}} = {u0_bin:.4f}$",
        color=BLACK, fontsize=12, fontweight="bold")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Per-equilibrium threshold scatter  (no OIM equivalent)
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure3(eq: dict, graph_name: str) -> plt.Figure:
    """
    Two panels:

    Left — Scatter:  u0*(σ) vs H(σ)  for all 2^N configs.
      x: per-equilibrium threshold  u0*(σ) = max(0, −λmin(L(σ))/2)
      y: Ising energy  H(σ)
      Colour: cut quality
      Green stars: globally optimal configs
      Red line: u0_bin (the global threshold = max of all u0*)

    Right — Histogram of u0*(σ) coloured by cut quality.
      Shows the distribution of per-config thresholds.
      Vertical lines: u0_bin, and mean threshold of globally optimal configs.

    Key diagnostic: if optimal configs cluster at LOW u0*(σ) (left of plot),
    they have the SMALLEST thresholds and are stabilised FIRST as u0 increases.
    If they cluster at HIGH u0*(σ), they are the LAST to be stabilised, and
    there is a dangerous window where suboptimal configs are stable but
    optimal ones are not.
    """
    N        = eq["N"]
    n_eq     = eq["n_eq"]
    u0_bin   = eq["u0_bin"]
    u0_thr   = eq["u0_thr"]
    H_arr    = eq["H"]
    cut_arr  = eq["cut"]
    opt_mask = eq["opt_mask"]
    best_cut = eq["best_cut"]

    cmap_cut = plt.get_cmap("RdYlGn")
    norm_cut = mcolors.Normalize(vmin=0, vmax=best_cut)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=WHITE)
    fig.subplots_adjust(wspace=0.35, left=0.07, right=0.97,
                        top=0.88, bottom=0.12)

    # ── Left: scatter u0*(σ) vs H(σ) ─────────────────────────────────────────
    ax = axes[0]
    non_opt = ~opt_mask

    ax.scatter(u0_thr[non_opt], H_arr[non_opt],
               c=[cmap_cut(norm_cut(c)) for c in cut_arr[non_opt]],
               s=8, alpha=0.35, zorder=2, edgecolors="none",
               label="All configs")

    # Globally optimal configs on top with star markers
    ax.scatter(u0_thr[opt_mask], H_arr[opt_mask],
               color=C_GREEN, marker="*", s=180, zorder=5,
               edgecolors=BLACK, linewidths=0.6,
               label=f"Global opt  ({opt_mask.sum()} configs)")

    # u0_bin vertical line
    ax.axvline(u0_bin, color=C_RED, linewidth=2.0, linestyle="--",
               zorder=6, label=f"$u_{{0,\\rm bin}} = {u0_bin:.3f}$")

    # Mean threshold of globally optimal configs
    mean_opt_thr = float(u0_thr[opt_mask].mean())
    ax.axvline(mean_opt_thr, color=C_GREEN, linewidth=1.5,
               linestyle=":", zorder=5,
               label=f"Mean $u_0^*$ (opt) $= {mean_opt_thr:.3f}$")

    # Annotation: are optimal configs easy or hard to stabilise?
    frac_opt_low = float(np.mean(u0_thr[opt_mask] <= u0_thr.mean()))
    message = ("Optimal configs have\n"
               f"LOWER-than-average $u_0^*$\n→ easiest to stabilise ✓"
               if frac_opt_low >= 0.5 else
               "Optimal configs have\n"
               f"HIGHER-than-average $u_0^*$\n→ hardest to stabilise ✗")
    msg_col = C_GREEN if frac_opt_low >= 0.5 else C_RED
    ax.text(0.97, 0.97, message,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, fontweight="bold", color=msg_col,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=WHITE,
                      edgecolor=msg_col, alpha=0.95, linewidth=1.2))

    ax.legend(fontsize=9, loc="lower right")
    _ax_style(ax,
              title=("Per-equilibrium stability threshold  $u_0^*(\\sigma)$  vs  "
                     "Ising energy  $H(\\sigma)$\n"
                     "Green stars = global optimum configs  |  "
                     "Colour = cut quality"),
              xlabel="$u_0^*(\\sigma) = \\max(0,\\, -\\lambda_{\\min}(L(\\sigma))/2)$",
              ylabel="Ising energy  $H(\\sigma) = \\frac{1}{2}\\sigma^\\top W\\sigma$")

    # ── Right: histogram of u0*(σ) ───────────────────────────────────────────
    ax = axes[1]

    # Colour bars by mean cut quality in each bin
    bins = np.linspace(u0_thr.min(), u0_thr.max() + 1e-9, 35)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    bin_idx     = np.digitize(u0_thr, bins) - 1
    bin_idx     = np.clip(bin_idx, 0, len(bins) - 2)

    for b in range(len(bins) - 1):
        mask_b = bin_idx == b
        if not mask_b.any():
            continue
        mean_cut_b = float(cut_arr[mask_b].mean())
        ax.bar(bin_centres[b], mask_b.sum(),
               width=(bins[1] - bins[0]) * 0.88,
               color=cmap_cut(norm_cut(mean_cut_b)),
               alpha=0.80, edgecolor="none", zorder=2)

    # Mark u0_bin
    ax.axvline(u0_bin, color=C_RED, linewidth=2.0, linestyle="--",
               zorder=6, label=f"$u_{{0,\\rm bin}} = {u0_bin:.3f}$")

    # Mark mean threshold of optimal configs
    ax.axvline(mean_opt_thr, color=C_GREEN, linewidth=1.8,
               linestyle=":", zorder=5,
               label=f"Mean $u_0^*$ (opt) $= {mean_opt_thr:.3f}$")

    # Mark mean threshold of all configs
    ax.axvline(float(u0_thr.mean()), color=C_AMBER, linewidth=1.4,
               linestyle="-.", zorder=5,
               label=f"Mean $u_0^*$ (all) $= {u0_thr.mean():.3f}$")

    # Overlay optimal configs as rug plot
    for thr in u0_thr[opt_mask]:
        ax.axvline(thr, color=C_GREEN, linewidth=1.0,
                   alpha=0.60, ymin=0, ymax=0.08, zorder=6)

    ax.text(0.97, 0.97,
            f"$u_0^*$ of optimal configs:\n"
            f"  mean  = {mean_opt_thr:.4f}\n"
            f"  range = [{u0_thr[opt_mask].min():.4f}, "
            f"{u0_thr[opt_mask].max():.4f}]\n\n"
            f"$u_0^*$ of all configs:\n"
            f"  mean = {u0_thr.mean():.4f}  std = {u0_thr.std():.4f}\n\n"
            f"Frac. opt configs with\n"
            f"$u_0^* \\leq$ u0_bin: "
            f"{float(np.mean(u0_thr[opt_mask] <= u0_bin))*100:.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=WHITE,
                      edgecolor=GRAY, alpha=0.95))

    ax.legend(fontsize=9, loc="upper left")
    _ax_style(ax,
              title=("Distribution of per-config thresholds $u_0^*(\\sigma)$\n"
                     "Colour = mean cut quality of configs in bin  |  "
                     "Green ticks = optimal config thresholds"),
              xlabel="$u_0^*(\\sigma)$",
              ylabel="# configurations")

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_cut, norm=norm_cut)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.02, shrink=0.75)
    cbar.set_label("Cut value", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_edgecolor(BLACK)

    fig.suptitle(
        f"HNN Per-Equilibrium Threshold Analysis  |  {graph_name}  |  "
        f"$N={N}$,  $2^N={n_eq}$ equilibria  |  "
        f"best cut $= {eq['best_cut']:.1f}$  |  "
        f"$u_{{0,\\rm bin}} = {u0_bin:.4f}$",
        color=BLACK, fontsize=12, fontweight="bold")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Console summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(eq: dict, graph_name: str):
    N       = eq["N"]
    n_eq    = eq["n_eq"]
    u0_bin  = eq["u0_bin"]
    u0_thr  = eq["u0_thr"]
    H_arr   = eq["H"]
    cut_arr = eq["cut"]
    opt     = eq["opt_mask"]

    print(f"\n{'='*65}")
    print(f"  HNN Bifurcation Summary  |  {graph_name}  |  N={N}")
    print(f"{'='*65}")
    print(f"  2^N = {n_eq} equilibria enumerated")
    print(f"  Best cut = {eq['best_cut']:.1f}  |  "
          f"H_min = {eq['H_min']:.2f}  |  "
          f"# optimal configs = {opt.sum()}")
    print(f"\n  Global threshold:  u0_bin = |λmin(W)| = {u0_bin:.4f}")
    print(f"\n  Per-config thresholds u0*(σ):")
    print(f"    All configs  :  mean={u0_thr.mean():.4f}  "
          f"std={u0_thr.std():.4f}  "
          f"range=[{u0_thr.min():.4f}, {u0_thr.max():.4f}]")
    print(f"    Opt configs  :  mean={u0_thr[opt].mean():.4f}  "
          f"std={u0_thr[opt].std():.4f}  "
          f"range=[{u0_thr[opt].min():.4f}, {u0_thr[opt].max():.4f}]")

    # Key question: are optimal configs easy to stabilise?
    frac_opt_below_bin = float(np.mean(u0_thr[opt] <= u0_bin))
    print(f"\n  Fraction of optimal configs with u0*(σ) ≤ u0_bin: "
          f"{frac_opt_below_bin*100:.1f}%")
    if frac_opt_below_bin >= 0.5:
        print(f"  → Optimal configs are EASIER to stabilise than average ✓")
        print(f"    (supports the conjecture: low-cut ↔ low threshold)")
    else:
        print(f"  → Optimal configs are HARDER to stabilise than average ✗")
        print(f"    (optimal configs require LARGER u0 than suboptimal ones)")

    print(f"\n  Stability counts at three u0 levels:")
    for label, u0_ref in [("0.01·u0_bin", 0.01 * u0_bin),
                           ("u0_bin",      u0_bin),
                           ("3·u0_bin",    3.0 * u0_bin)]:
        lmin_H = eq["lmin_L"] + 2.0 * u0_ref
        n_st   = int(np.sum(lmin_H > 0))
        n_opt_st = int(np.sum((lmin_H > 0) & opt))
        print(f"    u0 = {label:15s} ({u0_ref:.4f}):  "
              f"{n_st:>6}/{n_eq} stable  |  "
              f"{n_opt_st}/{opt.sum()} optimal stable")
    print(f"{'='*65}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="HNN bifurcation plot: λmin(H(σ,u0)) vs u0 for all 2^N configs")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--graph", type=str,
                     help="Path to graph file  (N / u v [w] format)")
    grp.add_argument("--n",    type=int,
                     help="Random ER graph of size N")
    parser.add_argument("--p1",      type=float, default=0.5)
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--u0_min",  type=float, default=0.0)
    parser.add_argument("--u0_max",  type=float, default=None,
                        help="Right edge of u0 sweep (default: 3 × u0_bin)")
    parser.add_argument("--n_u0",    type=int,   default=300)
    parser.add_argument("--save",    action="store_true")
    args = parser.parse_args()

    # ── Load / generate graph ─────────────────────────────────────────────────
    if args.graph is not None:
        W          = read_graph(args.graph)
        graph_name = args.graph
    else:
        if args.n > 18:
            print(f"[warn] N={args.n} > 18 → 2^N={2**args.n}  may be slow")
        W          = generate_er_graph(args.n, args.p1, args.seed)
        graph_name = f"ER(N={args.n}, p={args.p1}, seed={args.seed})"

    n = W.shape[0]
    print(f"\nGraph: {graph_name}")
    print(f"  N={n}  |E|={int(np.sum(W))//2}  W_tot={np.sum(W)/2:.1f}")
    if n > 18:
        print(f"  [warn] 2^N = {2**n} — enumeration will be slow")

    # ── Enumerate equilibria ──────────────────────────────────────────────────
    eq = enumerate_equilibria(W)
    u0_bin = eq["u0_bin"]

    # ── Build u0 grid ─────────────────────────────────────────────────────────
    u0_max = args.u0_max if args.u0_max is not None else 3.0 * u0_bin
    u0_max = max(u0_max, u0_bin * 1.5)   # at least 1.5× u0_bin
    u0_vals = np.linspace(args.u0_min, u0_max, args.n_u0)
    print(f"\n  u0 sweep: [{args.u0_min:.4f}, {u0_max:.4f}]  ({args.n_u0} pts)")

    # ── Console summary ───────────────────────────────────────────────────────
    print_summary(eq, graph_name)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("  Generating figures...")
    fig1 = make_figure1(eq, u0_vals, graph_name)
    fig2 = make_figure2(eq, u0_vals, graph_name)
    fig3 = make_figure3(eq, graph_name)

    if args.save:
        stem = (graph_name.replace("/", "_").replace(".", "_")
                          .replace("(", "").replace(")", "")
                          .replace(",", "").replace(" ", "_"))
        for tag, fig in [("paths",      fig1),
                         ("energy_vs_H", fig2),
                         ("thresholds",  fig3)]:
            for ext in ("pdf", "png"):
                fname = f"hnn_bifurcation_{tag}_{stem}.{ext}"
                fig.savefig(fname, bbox_inches="tight", dpi=150)
                print(f"  Saved: {fname}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
