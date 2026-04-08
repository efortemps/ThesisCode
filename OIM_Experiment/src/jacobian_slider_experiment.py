#!/usr/bin/env python3
"""
jacobian_slider_experiment.py
─────────────────────────────────────────────────────────────────────────────
Interactive μ-slider for OIM Eigenvalue Spectrum (Jacobian vs D matrix).

Visualizes how the Jacobian spectrum A(φ*, μ) = D(φ*) - μI shifts live
as μ is changed via a slider, while the D(φ*) spectrum remains static.
"""

import argparse
from itertools import product as iproduct

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider

from OIM_Experiment.src.OIM_mu import OIMMaxCut
from OIM_Experiment.src.graph_utils import read_graph

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#b0b0b0",
    "legend.facecolor": "white",
    "legend.labelcolor": "black",
})

WHITE = "#ffffff"
BLACK = "#000000"
GRAY = "#b0b0b0"
LIGHT = "#e6e6e6"
C_STABLE = "#4C72B0"
C_UNSTABLE = "#DD8452"
C_BPART = "#55a868"
C_MIXED = "#8172b2"
C_FERRO = "#c44e52"
C_MU_LINE = "#ffb74d"

# ── helpers ───────────────────────────────────────────────────────────────────
def _jacobian(oim, phi):
    D = oim.build_D(phi)
    return D - oim.mu * np.diag(np.cos(2.0 * phi))

def analyse_equilibria(oim):
    n, mu, W = oim.n, oim.mu, oim.W
    w_total = float(np.sum(W)) / 2.0
    rows = []

    for bits in iproduct([0, 1], repeat=n):
        phi = np.array([b * np.pi for b in bits], dtype=float)
        D = oim.build_D(phi)
        ev_D = np.sort(np.linalg.eigvalsh(D))
        lmax = float(ev_D[-1])

        sigma = np.sign(np.cos(phi))
        sigma[sigma == 0] = 1.0
        H = 0.5 * float(np.sum(W * np.outer(sigma, sigma)))
        cut = 0.25 * float(np.sum(W * (1.0 - np.outer(sigma, sigma))))

        rows.append(dict(
            bits=bits, phi=phi, D=D, ev_D=ev_D, 
            lmax_D=lmax, mu_thr=lmax, H=H, cut=cut
        ))

    mu_bin = min(r["lmax_D"] for r in rows)
    best_cut = max(r["cut"] for r in rows)
    n_stable = sum(1 for r in rows if mu > r["lmax_D"])

    return dict(rows=rows, mu_bin=mu_bin, best_cut=best_cut, 
                w_total=w_total, n_stable=n_stable, total=len(rows), n=n)

def pick_representatives(rows):
    sr = sorted(rows, key=lambda r: r["lmax_D"])
    low = sr[0]
    mid = sr[len(sr) // 2]
    high = sr[-1]

    def _lbl(tag, r):
        b = "".join(str(x) for x in r["bits"])
        return (f"{tag}: $[{b}]$ "
                f"$\\lambda_{{max}}={r['lmax_D']:.3f}$ "
                f"$H={r['H']:.1f}$ cut$={r['cut']:.1f}$")

    return [
        (low, _lbl("Easiest", low), C_BPART, "-"),
        (mid, _lbl("Median", mid), C_MIXED, "--"),
        (high, _lbl("Hardest", high), C_FERRO, ":"),
    ]

def _ax_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, color=LIGHT, linewidth=0.6, zorder=0)

# ── main interactive figure ───────────────────────────────────────────────────
def make_interactive_jacobian(eq_data, args):
    rows = eq_data["rows"]
    mu_bin = eq_data["mu_bin"]
    w_total = eq_data["w_total"]
    best_cut = eq_data["best_cut"]
    n = eq_data["n"]
    n_eq = eq_data["total"]

    reps = pick_representatives(rows)

    global_lmax_min = min(r["lmax_D"] for r in rows)
    global_lmax_max = max(r["lmax_D"] for r in rows)

    mu_min = args.mu_min if args.mu_min is not None else max(0.0, global_lmax_min - 0.5)
    mu_max = args.mu_max if args.mu_max is not None else global_lmax_max * 1.30
    init_mu = mu_bin

    fig = plt.figure(figsize=(18, 8.5), facecolor=WHITE)
    ax_aspec = fig.add_axes((0.05, 0.15, 0.42, 0.72))
    ax_dspec = fig.add_axes((0.55, 0.15, 0.42, 0.72))
    ax_slider = fig.add_axes((0.15, 0.04, 0.70, 0.03))

    for ax in (ax_aspec, ax_dspec):
        ax.set_facecolor(WHITE)
        ax.tick_params(colors=BLACK, labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor(BLACK)
            sp.set_linewidth(0.8)

    xidx = np.arange(1, n + 1)
    ev_all = np.concatenate([eq["ev_D"] for eq, _, _, _ in reps])
    ev_lo, ev_hi = ev_all.min(), ev_all.max()
    margin = max(0.06 * (ev_hi - ev_lo), 0.4)

    # =========================================================================
    # Right Plot: D(φ*) spectrum (Static lines, dynamic threshold)
    # =========================================================================
    ax_dspec.fill_between([0.5, n + 0.5], ev_lo - margin, 0, color=C_STABLE, alpha=0.10, zorder=0)
    ax_dspec.fill_between([0.5, n + 0.5], 0, ev_hi + margin, color=C_UNSTABLE, alpha=0.10, zorder=0)
    ax_dspec.axhline(0, color=BLACK, linewidth=1.0, linestyle="--", alpha=0.55, zorder=2)

    for k_r, (eq, lbl, col, ls) in enumerate(reps):
        ev_desc = eq["ev_D"][::-1]
        ax_dspec.plot(xidx, ev_desc, color=col, linestyle=ls, lw=2.2, marker="o", ms=8, label=lbl, zorder=3)
        dy = margin * (0.6 + 0.35 * k_r)
        ax_dspec.annotate(
            f"$\\lambda_{{max}}={eq['lmax_D']:.3f}$\n$H={eq['H']:.1f}$ cut$={eq['cut']:.1f}$",
            xy=(1, ev_desc[0]), xytext=(1.6, ev_desc[0] + dy),
            fontsize=9, color=col, zorder=5, arrowprops=dict(arrowstyle="->", color=col, lw=0.9)
        )

    hline_mu_D = ax_dspec.axhline(init_mu, color=C_MU_LINE, linewidth=2.0, linestyle="--", zorder=4, label="Current $\\mu$")

    ax_dspec.set_xticks(xidx)
    ax_dspec.set_xlim(0.5, n + 0.5)
    ax_dspec.set_ylim(ev_lo - 2 * margin, ev_hi + 3.5 * margin)
    ax_dspec.legend(fontsize=8.5, loc="lower left")

    d_text = ax_dspec.text(
        0.98, 0.98, "", transform=ax_dspec.transAxes, ha="right", va="top",
        fontsize=9.5, bbox=dict(boxstyle="round,pad=0.35", facecolor=WHITE, edgecolor=GRAY, alpha=0.94)
    )

    _ax_style(ax_dspec, 
              title="$D(\\phi^*)$ eigenvalue spectrum\n(Topology dependent, independent of $\\mu$)", 
              xlabel="Eigenvalue rank $k$ (largest first)", 
              ylabel="$\\lambda_k\\left(D(\\phi^*)\\right)$")

    # =========================================================================
    # Left Plot: A(φ*, μ) spectrum (Dynamic lines, static boundary)
    # =========================================================================
    # We must fix the Y axis to encompass the maximum possible shift.
    a_ylim_lo = ev_lo - mu_max - 2 * margin
    a_ylim_hi = ev_hi - mu_min + 3.5 * margin

    ax_aspec.fill_between([0.5, n + 0.5], a_ylim_lo, 0, color=C_STABLE, alpha=0.10, zorder=0)
    ax_aspec.fill_between([0.5, n + 0.5], 0, a_ylim_hi, color=C_UNSTABLE, alpha=0.10, zorder=0)
    ax_aspec.axhline(0, color=BLACK, linewidth=1.5, linestyle="--", zorder=2, label="Stability boundary (0)")

    a_lines = []
    a_annots = []
    dy_list = []

    for k_r, (eq, lbl, col, ls) in enumerate(reps):
        ev_desc = eq["ev_D"][::-1] - init_mu
        line, = ax_aspec.plot(xidx, ev_desc, color=col, linestyle=ls, lw=2.2, marker="o", ms=8, label=lbl, zorder=3)
        a_lines.append(line)

        dy = margin * (0.6 + 0.35 * k_r)
        dy_list.append(dy)
        ann = ax_aspec.annotate(
            f"$\\lambda_{{max}}={eq['lmax_D'] - init_mu:.3f}$\n$H={eq['H']:.1f}$ cut$={eq['cut']:.1f}$",
            xy=(1, ev_desc[0]), xytext=(1.6, ev_desc[0] + dy),
            fontsize=9, color=col, zorder=5, arrowprops=dict(arrowstyle="->", color=col, lw=0.9)
        )
        a_annots.append(ann)

    ax_aspec.set_xticks(xidx)
    ax_aspec.set_xlim(0.5, n + 0.5)
    ax_aspec.set_ylim(a_ylim_lo, a_ylim_hi)
    ax_aspec.legend(fontsize=8.5, loc="lower left")

    a_text = ax_aspec.text(
        0.98, 0.98, "", transform=ax_aspec.transAxes, ha="right", va="top",
        fontsize=9.5, bbox=dict(boxstyle="round,pad=0.35", facecolor=WHITE, edgecolor=GRAY, alpha=0.94)
    )

    _ax_style(ax_aspec, 
              title="$A(\\phi^*,\\mu)$ Jacobian spectrum\n(Dynamically shifts by $-\\mu$)", 
              xlabel="Eigenvalue rank $k$ (largest first)", 
              ylabel="$\\lambda_k\\left(A(\\phi^*)\\right) = \\lambda_k(D) - \\mu$")

    # =========================================================================
    # Slider & Update Logic
    # =========================================================================
    slider = Slider(ax_slider, '$\\mu$', mu_min, mu_max, valinit=init_mu, color=C_STABLE, track_color=LIGHT)
    ax_slider.set_facecolor(WHITE)
    slider.label.set_color(BLACK) 
    slider.valtext.set_color(BLACK)

    # Add mu_bin marker to slider
    ax_slider.axvline(mu_bin, color=C_MU_LINE, linewidth=2.5, zorder=5)
    rel = np.clip((mu_bin - mu_min) / (mu_max - mu_min + 1e-12), 0, 1)
    ax_slider.text(rel, 1.05, f"$\\mu_{{bin}}={mu_bin:.3f}$", transform=ax_slider.transAxes,
                   ha="center", va="bottom", fontsize=8.5, color=C_MU_LINE)

    def update(val):
        mu = slider.val

        # 1. Update D threshold
        hline_mu_D.set_ydata([mu, mu])
        n_stable = sum(1 for r in rows if mu > r["lmax_D"])
        d_text.set_text(
            f"Current $\\mu = {mu:.4f}$\n"
            f"Stable iff $\\mu > \\lambda_{{max}}(D)$\n"
            f"Stable equilibria: {n_stable} / {n_eq}"
        )

        # 2. Update A lines & annotations
        for k_r, (eq, lbl, col, ls) in enumerate(reps):
            ev_a = eq["ev_D"][::-1] - mu
            a_lines[k_r].set_ydata(ev_a)

            # Update annotation arrow and text dynamically
            a_annots[k_r].xy = (1, ev_a[0])
            a_annots[k_r].set_position((1.6, ev_a[0] + dy_list[k_r]))
            a_annots[k_r].set_text(
                f"$\\lambda_{{max}}={eq['lmax_D'] - mu:.3f}$\n"
                f"$H={eq['H']:.1f}$ cut$={eq['cut']:.1f}$"
            )

        a_text.set_text(
            f"Jacobian at $\\mu = {mu:.4f}$\n"
            f"Stable iff $\\lambda_{{max}}(A) < 0$\n"
            f"Stable equilibria: {n_stable} / {n_eq}"
        )

        fig.suptitle(
            f"OIM Interactive Eigenvalue Spectrum | {args.graph} | "
            f"$N={n}$, $2^N={n_eq}$ equilibria | $\\mu_{{bin}}={mu_bin:.4f}$ | "
            f"Best cut $={best_cut:.1f}$, $W_{{tot}}={w_total:.1f}$",
            color=BLACK, fontsize=13, fontweight="bold", y=0.98
        )

        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(init_mu)

    # FIX: Attach slider to figure to prevent garbage collection
    fig.slider = slider
    return fig

# ── entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Interactive μ-slider for Jacobian/D matrix spectra")
    parser.add_argument("--graph", required=True, help="Path to graph .txt file")
    parser.add_argument("--mu_min", type=float, default=None)
    parser.add_argument("--mu_max", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\nLoading graph: {args.graph}")
    W = read_graph(args.graph)
    n = W.shape[0]
    if n > 18:
        print(f"[warn] N={n} > 18 — 2^N={2**n} equilibrium scan may be slow")

    # We just run analyse_equilibria at mu=1.0, the exact mu doesn't matter for D matrix eigenvalues
    oim_ref = OIMMaxCut(W, mu=1.0, seed=args.seed)
    print(f" Scanning 2^{n}={2**n} equilibria for λ_max(D) spectrum...")
    eq_data = analyse_equilibria(oim_ref)

    print(f" μ_bin={eq_data['mu_bin']:.4f} | best cut={eq_data['best_cut']:.1f}")
    print("\n Launching interactive window ...")

    fig = make_interactive_jacobian(eq_data, args)
    plt.show()

if __name__ == "__main__":
    main()
