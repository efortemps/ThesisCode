#!/usr/bin/env python3
"""
combined_hnn_lambda_slider_experiment.py

Continuous Hopfield (Hopfield-Tank) Max-Cut experiment with:
- lambda-slider for gain  (lambda = 1/u0, thesis convention).
- Phase trajectories for multiple initial conditions.
- Hessian spectra at origin and at representative equilibria
  (best / median / worst cut states).

Run example:
    python -m MaxCut_Experiment.src.combined_hnn_lambda_slider_experiment \
        --graph MaxCut_Experiment/data/8node.txt
"""

import argparse
from itertools import product as iproduct

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp

# ── global style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          13,
    "axes.titlesize":     14,
    "axes.labelsize":     13,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "legend.fontsize":    11,
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

WHITE, BLACK, GRAY, LIGHT = "#ffffff", "#000000", "#b0b0b0", "#e6e6e6"
C_STABLE, C_UNSTABLE, C_BIN_OK, C_LAMBIN = "#4C72B0", "#DD8452", "#55a868", "#c44e52"
C_MIXED,  C_LAM_LINE, C_GREEN,  C_ORANGE  = "#8172b2", "#ffb74d", "#2e7d32", "#e65100"

_STATE_BADGE = {
    "M2-binary":    "#4C72B0",
    "M1-mixed":     "#e377c2",
    "Type-III":     "#8172b2",
    "not-converged":"#888888",
}
_STATE_BG = {
    "M2-binary":    "#dce8f7",
    "M1-mixed":     "#fce4f4",
    "Type-III":     "#ece4f9",
    "not-converged":"#ebebeb",
}


# ── graph IO ───────────────────────────────────────────────────────────────────
def read_graph(filepath):
    """
    Read graph in the format:

        # comments...
        N
        u v [w]
        ...

    If w is omitted, weight = 1.0.
    """
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f
                 if line.strip() and not line.startswith("#")]
    n = int(lines[0])
    W = np.zeros((n, n), dtype=float)
    for line in lines[1:]:
        parts = line.split()
        u, v = int(parts[0]), int(parts[1])
        w = float(parts[2]) if len(parts) > 2 else 1.0
        W[u, v] = W[v, u] = w
    return W


# ── helpers ────────────────────────────────────────────────────────────────────
def _lighten(hex_col, factor=0.5):
    hex_col = hex_col.lstrip("#")
    r, g, b = [int(hex_col[i:i+2], 16) for i in (0, 2, 4)]
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def hessian_at_s(W, lam, s):
    """
    H_ss(s) = W + diag((1/λ) / (1 - s_i^2)),   s ∈ (-1,1)^n.

    With u0 = 1/λ:
        H_ij = W_ij + δ_ij * u0 / (1 - s_i^2)
    """
    u0  = 1.0 / lam
    s_c = np.clip(s, -1.0 + 1e-10, 1.0 - 1e-10)
    return W + np.diag(u0 / (1.0 - s_c**2))


def hessian_at_origin(W, lam):
    """
    H_ss at s = 0:  H(0) = W + (1/λ) I = W + u0 I.

    Origin is unstable (good for binarisation) iff λ > λ_bin = 1/|λ_min(W)|,
    equivalently u0 < u0_bin = |λ_min(W)|.
    """
    u0 = 1.0 / lam
    n  = W.shape[0]
    return W + u0 * np.eye(n)


def scan_equilibria(W):
    """
    Enumerate all 2^n binary spin configurations.

    Global binarisation threshold in λ-convention:
        λ_bin = 1 / |λ_min(W)|
    Interpretation:
        λ > λ_bin  =>  origin UNSTABLE  =>  system binarises  (GOOD)
        λ < λ_bin  =>  origin STABLE    =>  may stagnate near 0
    """
    n = W.shape[0]
    lam_min_W = float(np.linalg.eigvalsh(W)[0])
    u0_bin    = float(abs(lam_min_W))   # = |λ_min(W)|
    lam_bin   = 1.0 / u0_bin if u0_bin > 0 else float("inf")

    rows     = []
    best_cut = 0.0
    for bits in iproduct([0, 1], repeat=n):
        s   = np.array([1.0 if b == 1 else -1.0 for b in bits], dtype=float)
        cut = 0.25 * float(np.sum(W * (1.0 - np.outer(s, s))))
        H_ising = 0.5 * float(s @ W @ s)
        best_cut = max(best_cut, cut)
        rows.append(dict(bits=bits, s=s, cut=cut, H_ising=H_ising))

    return dict(
        rows=rows,
        best_cut=best_cut,
        w_total=np.sum(W) / 2.0,
        n=n,
        total=len(rows),
        lam_bin=lam_bin,
        u0_bin=u0_bin,          # kept for internal use
        lam_min_W=lam_min_W,
    )


# ── dynamics and convergence ───────────────────────────────────────────────────
def simulate_trajectory(W, lam, u_init, t_end, n_points):
    """
    Solve  τ du/dt = -u - W s,   s = tanh(λ u),   τ = 1.
    """
    def rhs(t, u):
        return -u - W @ np.tanh(lam * u)

    t_eval = np.linspace(0.0, t_end, n_points)
    return solve_ivp(rhs, (0.0, t_end), u_init,
                     method="RK45", t_eval=t_eval,
                     rtol=1e-6, atol=1e-8)


def identify_convergence(sol, W, lam, best_cut, bintol=0.05):
    s_final = np.tanh(lam * sol.y[:, -1])
    sigma   = np.sign(s_final)
    sigma[sigma == 0] = 1.0

    nz = sum(1 for s in s_final if abs(s) < bintol)
    no = sum(1 for s in s_final if abs(abs(s) - 1.0) < bintol)
    nh = len(s_final) - nz - no

    if np.max(np.abs(s_final)) < 1e-3:
        stype = "not-converged"
    elif no == len(s_final):
        stype = "M2-binary"
    elif nh == 0:
        stype = "M1-mixed"
    else:
        stype = "Type-III"

    cut = 0.0 if stype == "not-converged" else \
          0.25 * float(np.sum(W * (1.0 - np.outer(sigma, sigma))))

    residual = float(np.max(np.abs(np.abs(s_final) - 1.0)))
    bits     = tuple(1 if s > 0 else 0 for s in sigma)

    return dict(
        bits=bits,
        cut=cut,
        residual=residual,
        is_binary=(stype == "M2-binary"),
        is_opt=(stype == "M2-binary" and abs(cut - best_cut) < 1e-6),
        state_type=stype,
    )


def precompute(lam_list, W, s_inits, t_end, n_points, best_cut):
    """
    Precompute trajectories for every λ in lam_list.

    s_inits holds fixed initial conditions in s-space ~ U(-1,1).
    For each λ they are converted to u-space via u = (1/λ) arctanh(s),
    so the starting activation is identical across the entire λ sweep.
    """
    results = []
    print(f"Pre-computing {len(lam_list)} λ values...")
    for lam in lam_list:
        u0     = 1.0 / lam
        print(f"  λ={lam:.4f}", end="\r", flush=True)
        u_inits = [u0 * np.arctanh(s) for s in s_inits]
        sols    = [simulate_trajectory(W, lam, u_init, t_end, n_points)
                   for u_init in u_inits]
        conv    = [identify_convergence(s, W, lam, best_cut) for s in sols]
        results.append(dict(lam=lam, u0=u0, sols=sols, conv=conv))
    print("  100.0% done.")
    return results


# ── equilibria representatives ─────────────────────────────────────────────────
def pick_representatives(rows):
    sr   = sorted(rows, key=lambda r: r["cut"])
    best, worst = sr[-1], sr[0]
    mid_cuts = [r for r in sr if worst["cut"] < r["cut"] < best["cut"]]
    mid = mid_cuts[len(mid_cuts) // 2] if mid_cuts else sr[len(sr) // 2]

    def _lbl(tag, r):
        bits_str = "".join(str(x) for x in r["bits"])
        return f"{tag}: $[{bits_str}]$ cut$={r['cut']:.1f}$"

    return [
        (best,  _lbl("Easiest", best),  C_BIN_OK, "-"),
        (mid,   _lbl("Median",  mid),   C_MIXED,  "--"),
        (worst, _lbl("Hardest", worst), C_LAMBIN, ":"),
    ]


# ── styled convergence table ───────────────────────────────────────────────────
def _draw_styled_table(ax, conv_list, best_cut, n_init, lam, lam_bin):
    ax.cla()
    ax.set_facecolor(WHITE)
    ax.axis("off")

    summary = {}
    for c in conv_list:
        key = (c["state_type"], c["bits"])
        if key not in summary:
            summary[key] = {
                "stype":    c["state_type"],
                "bits":     c["bits"],
                "cut":      c["cut"],
                "count":    0,
                "residuals":[],
                "is_opt":   c["is_opt"],
            }
        summary[key]["count"] += 1
        summary[key]["residuals"].append(c["residual"])

    rows2 = [
        [
            s["stype"],
            "".join(str(b) for b in s["bits"]),
            f"{s['cut']:.2f}",
            str(s["count"]),
            f"{100.0 * s['count'] / n_init:.0f}%",
            f"{float(np.mean(s['residuals'])):.4f}",
            "★ YES" if s["is_opt"] else "no",
        ]
        for s in sorted(summary.values(), key=lambda x: -x["cut"])
    ]

    # ── geometry constants ───────────────────────────────────────────────
    BANNER_H = 0.055
    HDR_H    = 0.060
    ROW_H    = 0.055
    STATUS_H = 0.06
    LEGEND_H = 0.08
    GAP      = 0.03

    n_rows  = len(rows2)
    table_h = HDR_H + ROW_H * n_rows
    block_h = BANNER_H + table_h + GAP + STATUS_H + GAP + LEGEND_H
    y_banner_top = 0.5 + block_h / 2.0

    def _section(y, title, hdrs, data, tcol, ocol=None):
        banner = FancyBboxPatch(
            (0.0, y - BANNER_H), 1.0, BANNER_H,
            boxstyle="square,pad=0",
            transform=ax.transAxes,
            facecolor="#3a3a3a", edgecolor="none", alpha=0.88, clip_on=False,
        )
        ax.add_patch(banner)
        ax.text(
            0.5, y - BANNER_H / 2.0, title,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=9.0, fontweight="bold", color=WHITE,
        )

        table_bottom = y - BANNER_H - table_h
        tbl = ax.table(
            cellText=data, colLabels=hdrs,
            bbox=[0.0, table_bottom, 1.0, table_h],
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.0)

        for j in range(len(hdrs)):
            cell = tbl[0, j]
            cell.set_facecolor("#dce6f0")
            cell.set_text_props(fontweight="bold", color="#1a1a2e")

        for i, row in enumerate(data, start=1):
            bg = _STATE_BG.get(row[tcol], WHITE)
            for j in range(len(hdrs)):
                cell  = tbl[i, j]
                shade = bg if (i % 2 == 1) else _lighten(bg, 0.55)
                cell.set_facecolor(shade)
                cell.set_edgecolor("#c8c8c8")
                cell.set_linewidth(0.4)

            bc = tbl[i, tcol]
            bc.set_facecolor(_STATE_BADGE.get(row[tcol], GRAY))
            bc.get_text().set_color(WHITE)
            bc.get_text().set_fontweight("bold")

            if ocol is not None:
                oc = tbl[i, ocol]
                v  = row[ocol]
                if "★" in v:
                    oc.set_facecolor("#55a868")
                    oc.get_text().set_color(WHITE)
                    oc.get_text().set_fontweight("bold")
                elif "✓" in v:
                    oc.set_facecolor("#9dcf9a")
                elif "✗" in v:
                    oc.set_facecolor("#f7b89a")

        return table_bottom

    bottom = _section(
        y_banner_top,
        "Summary — unique terminal states",
        ["Type", "Bits", "Cut", "n", "%", "Mean res.", "→ Opt?"],
        rows2, tcol=0, ocol=6,
    )

    # ── status text ──────────────────────────────────────────────────────
    diff          = lam - lam_bin
    origin_stable = diff < 0          # λ < λ_bin  =>  origin stable  (bad)
    status        = ("origin STABLE — may stagnate ✗" if origin_stable
                     else "origin UNSTABLE — binarises ✓")
    sc = C_ORANGE if origin_stable else C_GREEN

    ax.text(
        0.5, bottom - GAP,
        rf"$\lambda = {lam:.4f}$ | "
        rf"$\lambda_{{\mathrm{{bin}}}} = {lam_bin:.4f}$ | "
        rf"diff $= {diff:+.4f}$ ({status})",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=9.0, color=sc,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5f5f5", edgecolor=sc),
    )

    # ── legend ───────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=_STATE_BADGE[k], edgecolor=GRAY, label=k)
        for k in _STATE_BADGE
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, bottom - GAP - STATUS_H - GAP),
        ncol=2, fontsize=8.5,
        framealpha=0.92, edgecolor=GRAY,
    )


# ── phase-dynamics drawing ─────────────────────────────────────────────────────
def _draw_phase(ax, sols, conv, lam, lam_bin, w_total, best_cut, n, n_init):
    ax.cla()
    ax.set_facecolor(WHITE)
    ax.grid(True, color=LIGHT, linewidth=0.6, zorder=0)
    for sp in ax.spines.values():
        sp.set_edgecolor(BLACK)
        sp.set_linewidth(0.8)

    spin_cols = plt.get_cmap("tab20")(np.linspace(0, 1, max(n, 2)))

    for sol in sols:
        t = sol.t
        for s in range(n):
            ax.plot(
                t, np.tanh(lam * sol.y[s]),
                color=spin_cols[s % 20],
                alpha=0.42, linewidth=0.95, zorder=2,
            )

    for yref, lw_r in [(1.0, 1.1), (0.0, 1.4), (-1.0, 0.9)]:
        ax.axhline(yref, color=GRAY, linestyle="--",
                   linewidth=lw_r, alpha=0.75, zorder=1)

    ax.set(
        yticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
        ylim=(-1.1, 1.1),
        xlim=(sols[0].t[0], sols[0].t[-1]),
        xlabel="time $t$",
        ylabel=r"activation $s_i(t) = \tanh(\lambda\, u_i)$",
    )

    n_bin = sum(1 for c in conv if c["is_binary"])
    all_b = (n_bin == n_init)

    ax.text(
        0.98, 0.97,
        "BINARISED ✓" if all_b else f"NOT BINARISED ✗ ({n_bin}/{n_init})",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, fontweight="bold",
        color=C_BIN_OK if all_b else C_UNSTABLE,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE, edgecolor=GRAY),
    )

    ax.legend(
        handles=[
            mpatches.Patch(color=spin_cols[s % 20], label=f"spin {s}")
            for s in range(n)
        ],
        loc="lower right", fontsize=8, ncol=max(1, n // 5), framealpha=0.90,
    )

# ── shared slider controller ───────────────────────────────────────────────────
class SharedLambdaController:
    def __init__(self, lam_arr):
        self.lam_arr  = np.asarray(lam_arr, dtype=float)
        self.index    = len(self.lam_arr) // 2
        self._sliders = []
        self._updaters= []
        self._syncing = False

    def register_slider(self, slider):
        self._sliders.append(slider)

        def _cb(val):
            if self._syncing:
                return
            idx = int(np.argmin(np.abs(self.lam_arr - val)))
            self._set(idx, source=slider)

        slider.on_changed(_cb)

    def register_updater(self, fn):
        self._updaters.append(fn)

    def _set(self, idx, source=None):
        idx       = int(np.clip(idx, 0, len(self.lam_arr) - 1))
        self.index = idx
        lam       = float(self.lam_arr[idx])

        self._syncing = True
        try:
            for s in self._sliders:
                if s is not source and float(s.val) != lam:
                    s.set_val(lam)
        finally:
            self._syncing = False

        for fn in self._updaters:
            fn(idx)

    def trigger(self):
        self._set(self.index)


# ── Figure 1: phase dynamics ───────────────────────────────────────────────────
def make_phase_figure(ctrl, results, eq_data, args):
    fig     = plt.figure("Phase Dynamics", figsize=(13, 9.5), facecolor=WHITE)
    ax_phase  = fig.add_axes((0.07, 0.13, 0.88, 0.78))
    ax_slider = fig.add_axes((0.10, 0.035, 0.80, 0.030))

    lam_arr = ctrl.lam_arr
    lam_bin = eq_data["lam_bin"]

    slider = Slider(
        ax_slider, r"$\lambda$",
        float(lam_arr[0]), float(lam_arr[-1]),
        valinit=float(lam_arr[ctrl.index]),
        valstep=lam_arr,
        color=C_STABLE, track_color=LIGHT,
    )

    ax_slider.axvline(lam_bin, color=C_LAMBIN, linewidth=2.5, zorder=5)
    rel = float(np.clip(
        (lam_bin - lam_arr[0]) / (lam_arr[-1] - lam_arr[0] + 1e-12), 0, 1
    ))
    ax_slider.text(
        rel, 1.05,
        rf"$\lambda_{{\mathrm{{bin}}}}={lam_bin:.3f}$",
        transform=ax_slider.transAxes,
        ha="center", va="bottom", fontsize=10, color=C_LAMBIN,
    )

    def update(idx):
        rec = results[idx]
        _draw_phase(
            ax_phase, rec["sols"], rec["conv"],
            rec["lam"], lam_bin,
            eq_data["w_total"], eq_data["best_cut"],
            eq_data["n"], args.n_init,
        )
        fig.suptitle(
            rf"Continuous Hopfield  Phase Dynamics"
            rf"  |  $\lambda_{{\mathrm{{bin}}}} = {lam_bin:.4f}$",
            fontweight="bold", fontsize=14, y=0.99,
        )
        fig.canvas.draw_idle()

    ctrl.register_slider(slider)
    ctrl.register_updater(update)
    fig.slider = slider
    return fig


# ── Figure 1b: equilibrium table ──────────────────────────────────────────────
def make_table_figure(ctrl, results, eq_data, args):
    fig     = plt.figure("Equilibrium Table", figsize=(11, 9.5), facecolor=WHITE)
    ax_table  = fig.add_axes((0.03, 0.13, 0.94, 0.78))
    ax_slider = fig.add_axes((0.10, 0.035, 0.80, 0.030))

    lam_arr = ctrl.lam_arr
    lam_bin = eq_data["lam_bin"]

    slider = Slider(
        ax_slider, r"$\lambda$",
        float(lam_arr[0]), float(lam_arr[-1]),
        valinit=float(lam_arr[ctrl.index]),
        valstep=lam_arr,
        color=C_STABLE, track_color=LIGHT,
    )

    ax_slider.axvline(lam_bin, color=C_LAMBIN, linewidth=2.5, zorder=5)
    rel = float(np.clip(
        (lam_bin - lam_arr[0]) / (lam_arr[-1] - lam_arr[0] + 1e-12), 0, 1
    ))
    ax_slider.text(
        rel, 1.05,
        rf"$\lambda_{{\mathrm{{bin}}}}={lam_bin:.3f}$",
        transform=ax_slider.transAxes,
        ha="center", va="bottom", fontsize=10, color=C_LAMBIN,
    )

    def update(idx):
        rec = results[idx]
        _draw_styled_table(
            ax_table, rec["conv"], eq_data["best_cut"],
            args.n_init, rec["lam"], lam_bin,
        )
        fig.suptitle(
            rf"Equilibrium Table  |  $N={eq_data['n']}$,  "
            rf"$2^N={eq_data['total']}$ equilibria  |  "
            rf"$\lambda_{{\mathrm{{bin}}}}={lam_bin:.4f}$",
            fontweight="bold", fontsize=14, y=0.99,
        )
        fig.canvas.draw_idle()

    ctrl.register_slider(slider)
    ctrl.register_updater(update)
    fig.slider = slider
    return fig


# ── Figure 2: Hessian spectra ──────────────────────────────────────────────────
def make_spectrum_figure(ctrl, eq_data, W, args):
    fig = plt.figure("Hessian Spectra", figsize=(18, 8.5), facecolor=WHITE)

    ax_orig   = fig.add_axes((0.05, 0.15, 0.42, 0.72))
    ax_eq     = fig.add_axes((0.55, 0.15, 0.42, 0.72))
    ax_slider = fig.add_axes((0.15, 0.04, 0.70, 0.03))

    for ax in (ax_orig, ax_eq):
        ax.set_facecolor(WHITE)
        ax.grid(True, color=LIGHT, linewidth=0.6)
        ax.axhline(0, color=BLACK, linewidth=1.5, linestyle="--", zorder=2)
        for sp in ax.spines.values():
            sp.set_edgecolor(BLACK)
            sp.set_linewidth(0.8)

    n     = eq_data["n"]
    xidx  = np.arange(1, n + 1)
    ev_W  = np.sort(np.linalg.eigvalsh(W))
    margin = 0.5

    # ── origin panel ────────────────────────────────────────────────────
    line_orig, = ax_orig.plot(
        xidx, np.zeros(n),
        color=C_MIXED, marker="o", ms=8, lw=2.2,
        label=r"Origin $s=0$",
    )
    ann_orig = ax_orig.annotate(
        "", xy=(1, 0), xytext=(1.6, margin),
        fontsize=10, color=C_MIXED,
        arrowprops=dict(arrowstyle="->", color=C_MIXED, lw=0.9),
    )

    ax_orig.set(
        xlim=(0.5, n + 0.5),
        ylim=(ev_W[0] - margin, ev_W[-1] + ctrl.lam_arr[-1] * 0.1 + margin),
        title=r"Hessian spectrum at Origin ($s = 0$)",
        xlabel="eigenvalue index $k$",
        ylabel=r"$\lambda_k \left(H(s)\right)$",
    )
    ax_orig.legend(fontsize=11, loc="upper left")

    # ── equilibria panel ─────────────────────────────────────────────────
    reps     = pick_representatives(eq_data["rows"])
    e_lines  = []
    e_annots = []

    for k_r, (eq, lbl, col, ls) in enumerate(reps):
        line, = ax_eq.plot(
            xidx, np.zeros(n),
            color=col, linestyle=ls, lw=2.2, marker="o", ms=8,
            label=lbl, zorder=3,
        )
        e_lines.append(line)
        e_annots.append(ax_eq.annotate(
            "", xy=(1, 0), xytext=(1.6, margin),
            fontsize=10, color=col,
            arrowprops=dict(arrowstyle="->", color=col, lw=0.9),
        ))

    ax_eq.set(
        xlim=(0.5, n + 0.5),
        ylim=(-1, ev_W[-1] + 10),
        title=r"Hessian spectrum $H(\tilde{s}^*)$",
        xlabel="eigenvalue index $k$",
        ylabel=r"$\lambda_k \left(H(\tilde{s}^*)\right)$",
    )
    ax_eq.legend(fontsize=11, loc="lower right")

    # ── slider ───────────────────────────────────────────────────────────
    lam_arr = ctrl.lam_arr
    lam_bin = eq_data["lam_bin"]

    slider = Slider(
        ax_slider, r"$\lambda$",
        float(lam_arr[0]), float(lam_arr[-1]),
        valinit=float(lam_arr[ctrl.index]),
        valstep=lam_arr,
        color=C_STABLE, track_color=LIGHT,
    )

    ax_slider.axvline(lam_bin, color=C_LAM_LINE, linewidth=2.5)
    rel = float(np.clip(
        (lam_bin - lam_arr[0]) / (lam_arr[-1] - lam_arr[0] + 1e-12), 0, 1
    ))
    ax_slider.text(
        rel, 1.05,
        rf"$\lambda_{{\mathrm{{bin}}}}={lam_bin:.3f}$",
        transform=ax_slider.transAxes,
        ha="center", va="bottom", fontsize=10, color=C_LAM_LINE,
    )

    def update(idx):
        lam = float(lam_arr[idx])
        u0  = 1.0 / lam

        # origin spectrum
        ev_orig = np.sort(np.linalg.eigvalsh(hessian_at_origin(W, lam)))
        line_orig.set_ydata(ev_orig)
        ann_orig.xy = (1, ev_orig[0])
        ann_orig.set_position((1.6, ev_orig[0] + margin))
        ann_orig.set_text(rf"$\lambda_{{\min}}={ev_orig[0]:.3f}$")

        # equilibria spectra
        max_ev = 2.0
        for k_r, (eq, lbl, col, ls) in enumerate(reps):
            s = np.array(
                [1.0 if b == 1 else -1.0 for b in eq["bits"]], dtype=float
            )
            # Fixed-point iteration: s* = tanh(-W s* / u0) = tanh(-λ W s*)
            for _ in range(15):
                s = np.tanh(-(W @ s) * lam)
            ev_H = np.sort(np.linalg.eigvalsh(hessian_at_s(W, lam, s)))
            e_lines[k_r].set_ydata(ev_H)
            e_annots[k_r].xy = (1, ev_H[0])
            e_annots[k_r].set_position((1.6, ev_H[0] + margin * (1 + k_r * 0.5)))
            e_annots[k_r].set_text(rf"$\lambda_{{\min}}={ev_H[0]:.3f}$")
            max_ev = max(max_ev, ev_H[-1])

        ax_eq.set_ylim(-1, max_ev + 2)

        fig.suptitle(
            rf"Continuous Hopfield Hessian Spectra  |  "
            rf"$\lambda_{{\mathrm{{bin}}}}={lam_bin:.4f}$  |  ",
            fontweight="bold", fontsize=14, y=0.99,
        )
        fig.canvas.draw_idle()

    ctrl.register_slider(slider)
    ctrl.register_updater(update)
    fig.slider = slider
    return fig


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Continuous Hopfield λ-slider experiment "
            "(phase dynamics + Hessian spectra). "
            "λ = 1/u₀ is the gain (thesis convention)."
        ),
    )
    parser.add_argument("--graph",    required=True)
    parser.add_argument("--lam_min",  type=float, default=1.0,
                        help="Minimum λ (default 1.0)")
    parser.add_argument("--lam_max",  type=float, default=None,
                        help="Maximum λ (default 1.5 * λ_bin)")
    parser.add_argument("--n_lam",    type=int,   default=30,
                        help="Number of λ steps (default 30)")
    parser.add_argument("--n_init",   type=int,   default=10)
    parser.add_argument("--t_end",    type=float, default=20.0)
    parser.add_argument("--n_points", type=int,   default=500)
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    W = read_graph(args.graph)
    n = W.shape[0]

    print(f"Loaded graph {args.graph} with N={n}")
    eq_data = scan_equilibria(W)
    lam_bin = eq_data["lam_bin"]
    print(
        f"  2^N = {eq_data['total']} binary states | "
        f"best cut = {eq_data['best_cut']:.1f} | "
        f"λ_bin ≈ {lam_bin:.4f}  (= 1/u0_bin = 1/{eq_data['u0_bin']:.4f})"
    )

    lam_max  = args.lam_max if args.lam_max is not None else lam_bin * 1.50
    lam_list = list(np.linspace(args.lam_min, lam_max, args.n_lam))
    print(f"  λ sweep: [{args.lam_min:.4f}, {lam_max:.4f}] ({args.n_lam} steps)")

    rng    = np.random.default_rng(args.seed)
    S_CLIP = 1.0 - 1e-3
    s_inits = [
        np.clip(rng.uniform(-1.0, 1.0, n), -S_CLIP, S_CLIP)
        for _ in range(args.n_init)
    ]
    print(f"  {args.n_init} initial conditions sampled from U(-1,1) in s-space"
          f" (clipped to ±{S_CLIP})")

    results = precompute(lam_list, W, s_inits, args.t_end, args.n_points,
                         eq_data["best_cut"])

    ctrl = SharedLambdaController(lam_list)
    make_phase_figure(ctrl, results, eq_data, args)
    make_table_figure(ctrl, results, eq_data, args)
    make_spectrum_figure(ctrl, eq_data, W, args)
    ctrl.trigger()
    print("Windows launched.  Moving either slider syncs all figures.")
    plt.show()


if __name__ == "__main__":
    main()
