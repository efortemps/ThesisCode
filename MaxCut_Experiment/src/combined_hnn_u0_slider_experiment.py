#!/usr/bin/env python3
"""
combined_hnn_u0_slider_experiment.py

Continuous Hopfield (Hopfield–Tank) Max-Cut experiment with:
- u0-slider for gain.
- Phase trajectories for multiple initial conditions.
- Hessian spectra at origin and at representative equilibria
  (best / median / worst cut states).
"""

import argparse
from itertools import product as iproduct

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp

# ── global style (mirrors OIM experiment) ─────────────────────────────────────
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

WHITE, BLACK, GRAY, LIGHT = "#ffffff", "#000000", "#b0b0b0", "#e6e6e6"
C_STABLE, C_UNSTABLE, C_BIN_OK, C_U0BIN = "#4C72B0", "#DD8452", "#55a868", "#c44e52"
C_MIXED, C_U0_LINE, C_GREEN, C_ORANGE = "#8172b2", "#ffb74d", "#2e7d32", "#e65100"

_STATE_BADGE = {
    "M2-binary": "#4C72B0",
    "M1-mixed": "#e377c2",
    "Type-III": "#8172b2",
    "not-converged": "#888888",
}
_STATE_BG = {
    "M2-binary": "#dce8f7",
    "M1-mixed": "#fce4f4",
    "Type-III": "#ece4f9",
    "not-converged": "#ebebeb",
}


# ── graph IO ──────────────────────────────────────────────────────────────────
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


# ── helpers ───────────────────────────────────────────────────────────────────
def _lighten(hex_col, factor=0.5):
    hex_col = hex_col.lstrip("#")
    r, g, b = [int(hex_col[i:i+2], 16) for i in (0, 2, 4)]
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


# Hessian in s-space at arbitrary s
def hessian_at_s(W, u0, s):
    """
    H_ss(s) = W + diag(u0 / (1 - s_i^2)), with s in (-1,1)^n.
    """
    s_c = np.clip(s, -1.0 + 1e-10, 1.0 - 1e-10)
    return W + np.diag(u0 / (1.0 - s_c**2))


def hessian_at_origin(W, u0):
    """
    H_ss at s = 0.

    At s_i = 0 we have 1 - s_i^2 = 1, so:
        H = W + u0 * I.
    """
    n = W.shape[0]
    return W + u0 * np.eye(n)


def scan_equilibria(W):
    """
    Enumerate all 2^n binary spin configurations and, for each, compute:
      - cut value
      - Ising Hamiltonian H(s) = 0.5 * s^T W s
      - The per-equilibrium stability threshold  u0_i* (analogous to OIM's
        lambda_max(D(phi*)) per equilibrium).

    Per-equilibrium stability threshold
    ─────────────────────────────────────
    The Hessian at a *binary* corner s* = pm 1 is:
        H_ij(s*) = W_ij + delta_ij * u0 / (1 - s_i*^2)
    Because s_i* = pm 1 exactly, (1 - s_i*^2) = 0, so the Hessian diverges
    at the binary corners themselves.  The meaningful quantity is the LIMIT
    as u0 -> 0, reached by evaluating H at the *continuous fixed point*
    s_tilde*(u0) near s* (found by fixed-point iteration).

    However, a cleaner per-equilibrium threshold can be derived in u-space:
    The Jacobian of the ODE f(u) = -u - W tanh(u/u0) at u* is
        J_ij(u*) = -delta_ij - W_ij / u0 * (1 - s_i*^2)
    At a binary corner u_i* -> pm infty:  (1 - s_i*^2) -> 0, so J -> -I
    (always stable).  Therefore every binary corner is a stable equilibrium
    for small enough u0 -- there is NO per-corner instability analogous to
    OIM's mu_i* threshold.  This is the key difference from OIM.

    For comparison with OIM we instead report, for every binary state:
        lambda_min_H0 = lambda_min(W) + u0  [Hessian at origin]
    which is negative iff u0 < u0_bin = |lambda_min(W)| — i.e. the GLOBAL
    threshold, same for every equilibrium.  Per-equilibrium analysis for HNN
    requires finding the continuous fixed point (see make_spectrum_figure).

    Global binarisation threshold
    ──────────────────────────────
    u0_bin = |lambda_min(W)|
    Interpretation (OPPOSITE sign from OIM's mu_bin):
      u0 < u0_bin  =>  origin is UNSTABLE  =>  system binarises  (GOOD)
      u0 > u0_bin  =>  origin is STABLE    =>  system may stagnate near 0
    """
    n = W.shape[0]
    # Threshold: origin H(0) = W + u0*I changes from indefinite to PD at u0_bin
    lam_min_W = float(np.linalg.eigvalsh(W)[0])
    u0_bin    = float(abs(lam_min_W))   # = max(0, -lam_min_W)

    rows = []
    best_cut = 0.0
    for bits in iproduct([0, 1], repeat=n):
        s   = np.array([1.0 if b == 1 else -1.0 for b in bits], dtype=float)
        cut = 0.25 * float(np.sum(W * (1.0 - np.outer(s, s))))
        H_ising = 0.5 * float(s @ W @ s)   # Ising Hamiltonian; min <=> max cut
        best_cut = max(best_cut, cut)
        rows.append(dict(bits=bits, s=s, cut=cut, H_ising=H_ising))

    return dict(
        rows=rows,
        best_cut=best_cut,
        w_total=np.sum(W) / 2.0,
        n=n,
        total=len(rows),
        u0_bin=u0_bin,
        lam_min_W=lam_min_W,
    )


# ── dynamics and convergence ──────────────────────────────────────────────────
def simulate_trajectory(W, u0, u_init, t_end, n_points):
    """
    Solve tau du/dt = -u - W s,  s = tanh(u / u0), with tau=1.
    """
    def rhs(t, u):
        return -u - W @ np.tanh(u / u0)

    t_eval = np.linspace(0.0, t_end, n_points)
    return solve_ivp(rhs, (0.0, t_end), u_init,
                     method="RK45", t_eval=t_eval,
                     rtol=1e-6, atol=1e-8)


def identify_convergence(sol, W, u0, best_cut, bintol=0.05):
    """
    Given a solution sol, classify the terminal state:
    - bits (sign of s_final).
    - cut value.
    - residual = max_i | |s_i| - 1 |.
    - type:
        M2-binary    : all |s_i| ≈ 1  (converged to ±1 corners)
        M1-mixed     : some |s_i| ≈ 1, some |s_i| ≈ 0, none intermediate
        Type-III     : at least one s_i at an intermediate value (≠ 0 or ±1)
        not-converged: all |s_i| near 0 (stuck at origin)

    Note: the original code had `elif nz + no + nh == len(s_final)` as the
    Type-III guard, which is a tautology (nh is defined as n - nz - no), so
    Type-III was NEVER reachable. Fixed below.
    """
    s_final = np.tanh(sol.y[:, -1] / u0)
    sigma = np.sign(s_final)
    sigma[sigma == 0] = 1.0

    cut = 0.25 * float(np.sum(W * (1.0 - np.outer(sigma, sigma))))

    nz = sum(1 for s in s_final if abs(s) < bintol)           # near 0
    no = sum(1 for s in s_final if abs(abs(s) - 1.0) < bintol) # near ±1
    nh = len(s_final) - nz - no                                 # intermediate

    if np.max(np.abs(s_final)) < 1e-3:
        stype = "not-converged"       # all activations near 0
    elif no == len(s_final):
        stype = "M2-binary"           # every spin at ±1
    elif nh == 0:
        stype = "M1-mixed"            # mix of 0 and ±1, nothing intermediate
    else:
        stype = "Type-III"            # at least one spin at a continuous value

    residual = float(np.max(np.abs(np.abs(s_final) - 1.0)))
    bits = tuple(1 if s > 0 else 0 for s in sigma)

    return dict(
        bits=bits,
        cut=cut,
        residual=residual,
        is_binary=(stype == "M2-binary"),
        is_opt=(stype == "M2-binary" and abs(cut - best_cut) < 1e-6),
        state_type=stype,
    )


def precompute(u0_list, W, u_inits, t_end, n_points, best_cut):
    """
    Precompute trajectories and convergence info for all u0 in u0_list.
    """
    results = []
    print(f"Pre-computing {len(u0_list)} u0 values...")
    for u0 in u0_list:
        print(f" u0={u0:.4f}", end="\r", flush=True)
        sols = [simulate_trajectory(W, u0, u0_init, t_end, n_points)
                for u0_init in u_inits]
        conv = [identify_convergence(s, W, u0, best_cut) for s in sols]
        results.append(dict(u0=u0, sols=sols, conv=conv))
    print(" 100.0% done.")
    return results


# ── equilibria representatives ────────────────────────────────────────────────
def pick_representatives(rows):
    """
    Pick three representative equilibria:
      - best cut (easiest),
      - median cut,
      - worst cut (hardest).
    """
    sr = sorted(rows, key=lambda r: r["cut"])
    best, worst = sr[-1], sr[0]
    mid_cuts = [r for r in sr if worst["cut"] < r["cut"] < best["cut"]]
    mid = mid_cuts[len(mid_cuts) // 2] if mid_cuts else sr[len(sr) // 2]

    def _lbl(tag, r):
        bits_str = "".join(str(x) for x in r["bits"])
        return f"{tag}: $[{bits_str}]$ cut$={r['cut']:.1f}$"

    return [
        (best,  _lbl("Easiest", best),  C_BIN_OK, "-"),
        (mid,   _lbl("Median",  mid),   C_MIXED,  "--"),
        (worst, _lbl("Hardest", worst), C_U0BIN,  ":"),
    ]


# ── styled convergence table ──────────────────────────────────────────────────
def _draw_styled_table(ax, conv_list, best_cut, n_init, u0, u0_bin):
    ax.cla()
    ax.set_facecolor(WHITE)
    ax.axis("off")

    rows1 = [
        [
            str(i),
            c["state_type"],
            "".join(str(b) for b in c["bits"]),
            f"{c['cut']:.2f}",
            "★" if c["is_opt"] else ("✓" if c["is_binary"] else "✗"),
            f"{c['residual']:.4f}",
        ]
        for i, c in enumerate(conv_list)
    ]

    summary = {}
    for c in conv_list:
        key = (c["state_type"], c["bits"])
        if key not in summary:
            summary[key] = {
                "stype": c["state_type"],
                "bits": c["bits"],
                "cut": c["cut"],
                "count": 0,
                "residuals": [],
                "is_opt": c["is_opt"],
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

    def _section(y, title, hdrs, data, tcol, ocol=None):
        banner = FancyBboxPatch(
            (0.0, y - 0.045), 1.0, 0.045,
            boxstyle="square,pad=0",
            transform=ax.transAxes,
            facecolor="#3a3a3a",
            edgecolor="none",
            alpha=0.88,
            clip_on=False,
        )
        ax.add_patch(banner)
        ax.text(
            0.5, y - 0.0225, title,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=8.5, fontweight="bold",
            color=WHITE,
        )

        bottom = y - 0.05 - 0.048 * len(data)
        tbl = ax.table(
            cellText=data,
            colLabels=hdrs,
            bbox=[0.0, bottom, 1.0, 0.05 + 0.048 * len(data)],
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.8)

        # header row
        for j in range(len(hdrs)):
            cell = tbl[0, j]
            cell.set_facecolor("#dce6f0")
            cell.set_text_props(fontweight="bold", color="#1a1a2e")

        # body
        for i, row in enumerate(data, start=1):
            bg = _STATE_BG.get(row[tcol], WHITE)
            for j in range(len(hdrs)):
                cell = tbl[i, j]
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
                v = row[ocol]
                if "★" in v:
                    oc.set_facecolor("#55a868")
                    oc.get_text().set_color(WHITE)
                    oc.get_text().set_fontweight("bold")
                elif "✓" in v:
                    oc.set_facecolor("#9dcf9a")
                elif "✗" in v:
                    oc.set_facecolor("#f7b89a")

        return bottom

    y = 0.99
    y = _section(
        y,
        "Per-trajectory convergence",
        ["#", "Type", "Bits (s*)", "Cut", "✓", "Res."],
        rows1,
        tcol=1,
        ocol=4,
    ) - 0.04

    y = _section(
        y,
        "Summary — unique terminal states",
        ["Type", "Bits", "Cut", "n", "%", "Mean res.", "→ Opt?"],
        rows2,
        tcol=0,
        ocol=6,
    ) - 0.02

    diff = u0 - u0_bin
    # u0 < u0_bin  =>  origin is unstable  =>  binarisation FAVOURED (good)
    # u0 > u0_bin  =>  origin is stable    =>  network may stagnate at 0 (bad)
    origin_stable = diff > 0
    status = "origin STABLE — may stagnate ✗" if origin_stable else "origin UNSTABLE — binarises ✓"
    sc = C_ORANGE if origin_stable else C_GREEN

    ax.text(
        0.5, max(y, 0.01),
        rf"$u_0 = {u0:.4f}$ | "
        rf"$u_{{0,\mathrm{{bin}}}} = {u0_bin:.4f}$ | "
        rf"diff = {diff:+.4f} ({status})",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=8, color=sc,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#f5f5f5",
            edgecolor=sc,
        ),
    )

    legend_handles = [
        mpatches.Patch(
            facecolor=_STATE_BADGE[k],
            edgecolor=GRAY,
            label=k,
        )
        for k in _STATE_BADGE
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        fontsize=7.5,
        framealpha=0.92,
        facecolor=WHITE,
        edgecolor=GRAY,
    )


# ── phase-dynamics drawing ────────────────────────────────────────────────────
def _draw_phase(ax, sols, conv, u0, u0_bin, w_total, best_cut, n, n_init):
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
                t,
                np.tanh(sol.y[s] / u0),
                color=spin_cols[s % 20],
                alpha=0.42,
                linewidth=0.95,
                zorder=2,
            )

    for yref, lw_r in [(1.0, 1.1), (0.0, 1.4), (-1.0, 0.9)]:
        ax.axhline(
            yref, color=GRAY, linestyle="--",
            linewidth=lw_r, alpha=0.75, zorder=1,
        )

    ax.set(
        yticks=[-1.0, -0.5, 0.0, 0.5, 1.0],
        ylim=(-1.1, 1.1),
        xlim=(sols[0].t[0], sols[0].t[-1]),
        xlabel="time $t$",
        ylabel=r"activation $s_i(t)$",
    )

    n_bin = sum(1 for c in conv if c["is_binary"])
    all_b = (n_bin == n_init)

    ax.text(
        0.98, 0.97,
        "BINARISED ✓" if all_b else f"NOT BINARISED ✗ ({n_bin}/{n_init})",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9, fontweight="bold",
        color=C_BIN_OK if all_b else C_UNSTABLE,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor=WHITE,
            edgecolor=GRAY,
        ),
    )

    ax.text(
        0.01, 0.97,
        rf"$u_0={u0:.4f}$ | "
        rf"$u_{{0,\mathrm{{bin}}}}={u0_bin:.4f}$ | "
        rf"$W_{{\mathrm{{tot}}}}={w_total:.1f}$ | "
        rf"best cut={best_cut:.1f}",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9.5,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor=WHITE,
            edgecolor=GRAY,
        ),
    )

    ax.legend(
        handles=[
            mpatches.Patch(
                color=spin_cols[s % 20],
                label=f"spin {s}",
            )
            for s in range(n)
        ],
        loc="lower right",
        fontsize=7.5,
        ncol=max(1, n // 5),
        framealpha=0.90,
    )

    ax.set_title(
        rf"Phase dynamics $u_0 = {u0:.4f}$ | {n_init} initial conditions",
        color=BLACK,
        fontsize=11,
        pad=6,
    )


# ── shared slider controller ──────────────────────────────────────────────────
class SharedMuController:
    def __init__(self, u0_arr):
        self.u0_arr = np.asarray(u0_arr, dtype=float)
        self.index = len(self.u0_arr) // 2
        self._sliders = []
        self._updaters = []
        self._syncing = False

    def register_slider(self, slider):
        self._sliders.append(slider)

        def _cb(val):
            if self._syncing:
                return
            idx = int(np.argmin(np.abs(self.u0_arr - val)))
            self._set(idx, source=slider)

        slider.on_changed(_cb)

    def register_updater(self, fn):
        self._updaters.append(fn)

    def _set(self, idx, source=None):
        idx = int(np.clip(idx, 0, len(self.u0_arr) - 1))
        self.index = idx
        u0 = float(self.u0_arr[idx])

        self._syncing = True
        try:
            for s in self._sliders:
                if s is not source and float(s.val) != u0:
                    s.set_val(u0)
        finally:
            self._syncing = False

        for fn in self._updaters:
            fn(idx)

    def trigger(self):
        self._set(self.index)


# ── Figure 1: phase dynamics + table ──────────────────────────────────────────
def make_phase_figure(ctrl, results, eq_data, args):
    fig = plt.figure("Phase Dynamics", figsize=(22, 9.5), facecolor=WHITE)

    ax_phase = fig.add_axes((0.04, 0.13, 0.54, 0.78))
    ax_table = fig.add_axes((0.61, 0.13, 0.38, 0.78))
    ax_slider = fig.add_axes((0.10, 0.035, 0.80, 0.030))

    u0_arr = ctrl.u0_arr

    slider = Slider(
        ax_slider, "$u_0$",
        float(u0_arr[0]),
        float(u0_arr[-1]),
        valinit=float(u0_arr[ctrl.index]),
        valstep=u0_arr,
        color=C_STABLE,
        track_color=LIGHT,
    )

    ax_slider.axvline(eq_data["u0_bin"], color=C_U0BIN, linewidth=2.5, zorder=5)
    rel = float(
        np.clip(
            (eq_data["u0_bin"] - u0_arr[0]) / (u0_arr[-1] - u0_arr[0] + 1e-12),
            0,
            1,
        )
    )
    ax_slider.text(
        rel, 1.05,
        rf"$u_{{0,\mathrm{{bin}}}}={eq_data['u0_bin']:.3f}$",
        transform=ax_slider.transAxes,
        ha="center", va="bottom",
        fontsize=8.5, color=C_U0BIN,
    )

    def update(idx):
        rec = results[idx]
        u0 = rec["u0"]
        _draw_phase(
            ax_phase, rec["sols"], rec["conv"],
            u0, eq_data["u0_bin"], eq_data["w_total"],
            eq_data["best_cut"], eq_data["n"], args.n_init,
        )
        _draw_styled_table(
            ax_table, rec["conv"], eq_data["best_cut"],
            args.n_init, u0, eq_data["u0_bin"],
        )
        fig.suptitle(
            rf"Continuous Hopfield $u_0$-slider | {args.graph} | "
            rf"$N={eq_data['n']}$, $2^N={eq_data['total']}$ eq | "
            rf"$u_{{0,\mathrm{{bin}}}}={eq_data['u0_bin']:.4f}$ | "
            rf"best cut={eq_data['best_cut']:.1f}",
            fontweight="bold",
            y=0.99,
        )
        fig.canvas.draw_idle()

    ctrl.register_slider(slider)
    ctrl.register_updater(update)
    fig.slider = slider

    return fig


# ── Figure 2: Hessian spectra ─────────────────────────────────────────────────
def make_spectrum_figure(ctrl, eq_data, W, args):
    fig = plt.figure("Hessian Spectra", figsize=(18, 8.5), facecolor=WHITE)

    ax_orig = fig.add_axes((0.05, 0.15, 0.42, 0.72))
    ax_eq = fig.add_axes((0.55, 0.15, 0.42, 0.72))
    ax_slider = fig.add_axes((0.15, 0.04, 0.70, 0.03))

    for ax in (ax_orig, ax_eq):
        ax.set_facecolor(WHITE)
        ax.grid(True, color=LIGHT, linewidth=0.6)
        ax.axhline(0, color=BLACK, linewidth=1.5, linestyle="--", zorder=2)
        for sp in ax.spines.values():
            sp.set_edgecolor(BLACK)
            sp.set_linewidth(0.8)

    n = eq_data["n"]
    xidx = np.arange(1, n + 1)

    # spectrum of W for reference
    ev_W = np.sort(np.linalg.eigvalsh(W))
    margin = 0.5

    # origin spectrum line
    line_orig, = ax_orig.plot(
        xidx,
        np.zeros(n),
        color=C_MIXED,
        marker="o",
        ms=8,
        lw=2.2,
        label="Origin $s=0$",
    )
    ann_orig = ax_orig.annotate(
        "",
        xy=(1, 0),
        xytext=(1.6, margin),
        fontsize=9,
        color=C_MIXED,
        arrowprops=dict(arrowstyle="->", color=C_MIXED, lw=0.9),
    )
    orig_text = ax_orig.text(
        0.98, 0.98, "",
        transform=ax_orig.transAxes,
        ha="right", va="top",
        fontsize=9.5,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor=WHITE,
            edgecolor=GRAY,
        ),
    )

    ax_orig.set(
        xlim=(0.5, n + 0.5),
        ylim=(ev_W[0] - margin, ev_W[-1] + ctrl.u0_arr[-1] + margin),
        title=(
            r"Hessian $H(0) = W + u_0 I$ spectrum at Origin ($s=0$)" "\n"
            r"$\lambda_{\min}(H) > 0 \Leftrightarrow u_0 > u_{0,\mathrm{bin}}$"
            " — origin stable (BAD: network stagnates)"
        ),
    )

    # equilibria spectra
    reps = pick_representatives(eq_data["rows"])

    e_lines = []
    e_annots = []
    for k_r, (eq, lbl, col, ls) in enumerate(reps):
        line, = ax_eq.plot(
            xidx,
            np.zeros(n),
            color=col,
            linestyle=ls,
            lw=2.2,
            marker="o",
            ms=8,
            label=lbl,
            zorder=3,
        )
        e_lines.append(line)
        e_annots.append(
            ax_eq.annotate(
                "",
                xy=(1, 0),
                xytext=(1.6, margin),
                fontsize=9,
                color=col,
                arrowprops=dict(arrowstyle="->", color=col, lw=0.9),
            )
        )

    ax_eq.set(
        xlim=(0.5, n + 0.5),
        ylim=(-1, ev_W[-1] + ctrl.u0_arr[-1] * 20),
        title=(
            r"Hessian $H(\tilde{s}^*)$ at continuous fixed points near binary corners" "\n"
            r"$\tilde{s}^*$ found via $s \leftarrow \tanh(-Ws/u_0)$ (fixed-point iteration)"
        ),
    )
    ax_eq.legend(fontsize=8.5, loc="lower right")

    u0_arr = ctrl.u0_arr
    slider = Slider(
        ax_slider, "$u_0$",
        float(u0_arr[0]),
        float(u0_arr[-1]),
        valinit=float(u0_arr[ctrl.index]),
        valstep=u0_arr,
        color=C_STABLE,
        track_color=LIGHT,
    )

    ax_slider.axvline(eq_data["u0_bin"], color=C_U0_LINE, linewidth=2.5)
    rel = float(
        np.clip(
            (eq_data["u0_bin"] - u0_arr[0]) / (u0_arr[-1] - u0_arr[0] + 1e-12),
            0,
            1,
        )
    )
    ax_slider.text(
        rel, 1.05,
        rf"$u_{{0,\mathrm{{bin}}}}={eq_data['u0_bin']:.3f}$",
        transform=ax_slider.transAxes,
        ha="center", va="bottom",
        fontsize=8.5, color=C_U0_LINE,
    )

    def update(idx):
        u0 = float(u0_arr[idx])
        # origin
        ev_orig = np.sort(np.linalg.eigvalsh(hessian_at_origin(W, u0)))
        line_orig.set_ydata(ev_orig)
        ann_orig.xy = (1, ev_orig[0])
        ann_orig.set_position((1.6, ev_orig[0] + margin))
        ann_orig.set_text(rf"$\lambda_{{min}}={ev_orig[0]:.3f}$")

        status = "STABLE — stagnates ✗" if ev_orig[0] > 0 else "UNSTABLE — binarises ✓"
        orig_text.set_text(
            rf"Origin $s=0$ at $u_0 = {u0:.4f}$" "\n"
            rf"$\lambda_{{\min}}(H) = {ev_orig[0]:.4f}$" "\n"
            rf"Status: {status}" "\n"
            rf"$u_{{0,\mathrm{{bin}}}} = {eq_data['u0_bin']:.4f}$"
        )

        # equilibria
        max_ev = 2.0
        for k_r, (eq, lbl, col, ls) in enumerate(reps):
            s = np.array(
                [1.0 if b == 1 else -1.0 for b in eq["bits"]],
                dtype=float,
            )
            # Relax towards the continuous fixed point near this binary corner.
            # Fixed-point condition from du/dt = 0:
            #   u* = -W s*   ->   s* = tanh(u*/u0) = tanh(-W s*/u0)
            # The original code had +W@s/u0 (wrong sign), which converges to
            # fixed points of the wrong map and can flip the corner entirely.
            for _ in range(15):
                s = np.tanh(-(W @ s) / u0)
            ev_H = np.sort(np.linalg.eigvalsh(hessian_at_s(W, u0, s)))
            e_lines[k_r].set_ydata(ev_H)
            e_annots[k_r].xy = (1, ev_H[0])
            e_annots[k_r].set_position(
                (1.6, ev_H[0] + margin * (1 + k_r * 0.5))
            )
            e_annots[k_r].set_text(
                rf"$\lambda_{{min}}={ev_H[0]:.3f}$"
            )
            max_ev = max(max_ev, ev_H[-1])

        ax_eq.set_ylim(-1, min(max_ev + 2, 100))

        fig.suptitle(
            rf"Continuous Hopfield Hessian Spectra | "
            rf"$u_{{0,\mathrm{{bin}}}}={eq_data['u0_bin']:.4f}$ | "
            rf"$u_0={u0:.4f}$",
            fontweight="bold",
            y=0.98,
        )
        fig.canvas.draw_idle()

    ctrl.register_slider(slider)
    ctrl.register_updater(update)
    fig.slider = slider

    return fig


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Continuous Hopfield u0-slider experiment "
                    "(phase dynamics + Hessian spectra).",
    )
    parser.add_argument("--graph", required=True)
    parser.add_argument("--u0_min", type=float, default=0.05)
    parser.add_argument("--u0_max", type=float, default=None)
    parser.add_argument("--n_u0", type=int, default=30)
    parser.add_argument("--n_init", type=int, default=10)
    parser.add_argument("--t_end", type=float, default=20.0)
    parser.add_argument("--n_points", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    W = read_graph(args.graph)
    n = W.shape[0]

    print(f"Loaded graph {args.graph} with N={n}")
    eq_data = scan_equilibria(W)
    u0_bin = eq_data["u0_bin"]
    print(
        f" 2^N = {eq_data['total']} binary states | "
        f"best cut = {eq_data['best_cut']:.1f} | "
        f"u0_bin ≈ {u0_bin:.4f}"
    )

    u0_max = args.u0_max if args.u0_max is not None else u0_bin * 1.50
    u0_list = list(np.linspace(args.u0_min, u0_max, args.n_u0))
    print(f" u0 sweep: [{args.u0_min:.4f}, {u0_max:.4f}] ({args.n_u0} steps)")

    rng = np.random.default_rng(args.seed)
    # initial conditions: uniform in [-1,1]^n as requested
    u_inits = [rng.uniform(-1.0, 1.0, n) for _ in range(args.n_init)]
    print(f" {args.n_init} initial conditions sampled from uniform(-1, 1)")

    results = precompute(
        u0_list, W, u_inits,
        args.t_end, args.n_points,
        eq_data["best_cut"],
    )

    ctrl = SharedMuController(u0_list)
    make_phase_figure(ctrl, results, eq_data, args)
    make_spectrum_figure(ctrl, eq_data, W, args)
    ctrl.trigger()
    print("Windows launched. Moving either slider syncs the other.")
    plt.show()


if __name__ == "__main__":
    main()