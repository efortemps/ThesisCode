#!/usr/bin/env python3
"""
combined_oim_mu_slider_experiment.py
─────────────────────────────────────────────────────────────────────────────
Combined interactive μ-sweep for OIM:
  • Figure 1 — Phase dynamics + styled convergence table
  • Figure 2 — Jacobian / D-matrix eigenvalue spectra

Both figures share the same discrete μ grid. Moving the slider in either
window synchronises the other instantly.

Usage
─────
python combined_oim_mu_slider_experiment.py --graph data/10node.txt
python combined_oim_mu_slider_experiment.py --graph data/10node.txt \
    --mu_values 0.01 0.5 1.0 2.0 3.0 4.0 6.0
python combined_oim_mu_slider_experiment.py --graph data/10node.txt \
    --mu_min 0.0 --mu_max 7.0 --n_mu 30 --t_end 120 --n_init 16

CLI options
───────────
--graph      PATH        graph file (required)
--mu_min     FLOAT       lower bound  (default: max(0, μ_bin − 0.5))
--mu_max     FLOAT       upper bound  (default: 1.3 × max λ_max(D))
--n_mu       INT         evenly-spaced steps (default: 20)
--mu_values  FLOAT ...   explicit list — overrides --mu_min/max/n_mu
--n_init     INT         random initial conditions per μ (default: 12)
--t_end      FLOAT       ODE horizon (default: 80)
--n_points   INT         time-grid points per trajectory (default: 500)
--seed       INT         RNG seed (default: 42)
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
from itertools import product as iproduct

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Slider

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
C_BIN_OK   = "#55a868"
C_MUBIN    = "#c44e52"
C_MIXED    = "#8172b2"
C_MU_LINE  = "#ffb74d"
C_GREEN    = "#2e7d32"
C_ORANGE   = "#e65100"

_STATE_BADGE = {
    "M2-binary":     "#4C72B0",
    "M1-half":       "#DD8452",
    "M1-mixed":      "#e377c2",
    "Type-III":      "#8172b2",
    "not-converged": "#888888",
}
_STATE_BG = {
    "M2-binary":     "#dce8f7",
    "M1-half":       "#fde8d8",
    "M1-mixed":      "#fce4f4",
    "Type-III":      "#ece4f9",
    "not-converged": "#ebebeb",
}

PI_TICKS  = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
PI_LABELS = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]


# ── colour helper ─────────────────────────────────────────────────────────────
def _lighten(hex_col, factor=0.5):
    """Blend hex_col toward white by factor."""
    hex_col = hex_col.lstrip("#")
    r, g, b = [int(hex_col[i:i+2], 16) for i in (0, 2, 4)]
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


# ── physics helpers ───────────────────────────────────────────────────────────
def _jacobian(oim, phi):
    D = oim.build_D(phi)
    return D - oim.mu * np.diag(np.cos(2.0 * phi))


def analyse_equilibria(oim):
    n, mu, W = oim.n, oim.mu, oim.W
    w_total = float(np.sum(W)) / 2.0
    rows = []
    for bits in iproduct([0, 1], repeat=n):
        phi  = np.array([b * np.pi for b in bits], dtype=float)
        D    = oim.build_D(phi)
        ev_D = np.sort(np.linalg.eigvalsh(D))
        lmax = float(ev_D[-1])
        A    = _jacobian(oim, phi)
        ev_A = np.sort(np.linalg.eigvalsh(A))

        sigma = np.sign(np.cos(phi))
        sigma[sigma == 0] = 1.0
        H   = 0.5  * float(np.sum(W * np.outer(sigma, sigma)))
        cut = 0.25 * float(np.sum(W * (1.0 - np.outer(sigma, sigma))))

        rows.append(dict(
            bits=bits,
            phi=phi,
            D=D,
            ev_D=ev_D,
            ev_A=ev_A,
            lmax_D=lmax,
            lmax_A=float(ev_A[-1]),
            mu_thr=lmax,
            stable=(mu > lmax),
            H=H,
            cut=cut,
        ))

    mu_bin   = min(r["lmax_D"] for r in rows)
    best_cut = max(r["cut"] for r in rows)
    n_stable = sum(r["stable"] for r in rows)
    return dict(
        rows=rows,
        mu_bin=mu_bin,
        best_cut=best_cut,
        w_total=w_total,
        n_stable=n_stable,
        total=len(rows),
        n=n,
        mu=mu,
    )


def _atom(theta_i, tol=0.05):
    s, c = np.sin(theta_i), np.cos(theta_i)
    if abs(s) < tol:
        return "zero" if c > 0 else "pi"
    if abs(abs(s) - 1.0) < tol:
        return "half"
    return "other"


def identify_convergence(sol, W, eq_rows, best_cut, bintol=0.05):
    theta    = sol.y[:, -1].copy()
    n        = len(theta)
    residual = float(np.max(np.abs(np.sin(theta))))
    sigma    = np.sign(np.cos(theta))
    sigma[sigma == 0] = 1.0
    bits     = tuple(0 if s > 0 else 1 for s in sigma)

    H   = 0.5  * float(np.sum(W * np.outer(sigma, sigma)))
    cut = 0.25 * float(np.sum(W * (1.0 - np.outer(sigma, sigma))))

    atoms = [_atom(t, bintol) for t in theta]
    nz = atoms.count("zero")
    np_ = atoms.count("pi")
    nh = atoms.count("half")
    no = atoms.count("other")

    if nz + np_ == n:
        stype = "M2-binary"
    elif nh == n:
        stype = "M1-half"
    elif nh > 0 and no == 0 and nz + np_ + nh == n:
        stype = "M1-mixed"
    elif no > 0:
        stype = "Type-III"
    else:
        stype = "not-converged"

    nearest_bits = None
    nearest_dist = np.inf
    for r in eq_rows:
        diff = (theta - r["phi"] + np.pi) % (2*np.pi) - np.pi
        d = float(np.linalg.norm(diff))
        if d < nearest_dist:
            nearest_dist = d
            nearest_bits = r["bits"]

    return dict(
        theta=theta,
        bits=bits,
        H=H,
        cut=cut,
        residual=residual,
        is_binary=(stype == "M2-binary"),
        is_opt=(stype == "M2-binary" and abs(cut - best_cut) < 1e-6),
        state_type=stype,
        nearest_bits=nearest_bits,
        nearest_dist=float(nearest_dist),
    )


def precompute(mu_list, W, phi0s, t_end, n_points, eq_rows, best_cut, seed):
    results = []
    total = len(mu_list)
    bar_w = 40

    print(f"\nPre-computing {total} μ values ({len(phi0s)} trajectories × t_end={t_end}):")
    for i, mu in enumerate(mu_list):
        filled = int(bar_w * (i / total))
        bar = "█" * filled + "░" * (bar_w - filled)
        pct = 100 * i / total
        print(f" [{bar}] {pct:5.1f}%  μ={mu:.4f} ", end="\r", flush=True)

        oim  = OIMMaxCut(W, mu=mu, seed=seed)
        sols = oim.simulate_many(phi0s, t_span=(0., t_end), n_points=n_points)
        n_st = sum(1 for r in eq_rows if mu > r["lmax_D"])
        conv = [identify_convergence(s, W, eq_rows, best_cut) for s in sols]
        results.append(dict(mu=mu, sols=sols, conv=conv, n_stable=n_st))

    print(f" [{'█'*bar_w}] 100.0%  done.")
    return results


def pick_representatives(rows):
    sr   = sorted(rows, key=lambda r: r["lmax_D"])
    low  = sr[0]
    mid  = sr[len(sr)//2]
    high = sr[-1]

    def _lbl(tag, r):
        b = "".join(str(x) for x in r["bits"])
        return (
            f"{tag}: $[{b}]$  "
            f"$\\lambda_{{max}}={r['lmax_D']:.3f}$  "
            f"$H={r['H']:.1f}$  cut$={r['cut']:.1f}$"
        )

    return [
        (low,  _lbl("Easiest", low),  C_BIN_OK, "-"),
        (mid,  _lbl("Median",  mid),  C_MIXED,  "--"),
        (high, _lbl("Hardest", high), C_MUBIN,  ":"),
    ]


# ── styled convergence table ──────────────────────────────────────────────────
def _draw_styled_table(ax, conv_list, best_cut, n_init, mu, mu_bin):
    ax.cla()
    ax.set_facecolor(WHITE)
    ax.axis("off")

    def _bg(st):    return _STATE_BG.get(st, WHITE)
    def _badge(st): return _STATE_BADGE.get(st, GRAY)

    hdr1  = ["#", "Type", "Bits (φ*)", "H", "Cut", "✓", "Res.", "Nearest M2"]
    rows1 = []
    for i, c in enumerate(conv_list):
        bits_s   = "".join(str(b) for b in c["bits"])
        near_s   = "".join(str(b) for b in c["nearest_bits"]) if c["nearest_bits"] else "—"
        opt_mark = "★" if c["is_opt"] else ("✓" if c["is_binary"] else "✗")
        rows1.append([
            str(i),
            c["state_type"],
            bits_s,
            f"{c['H']:.2f}",
            f"{c['cut']:.2f}",
            opt_mark,
            f"{c['residual']:.4f}",
            near_s,
        ])

    summary = {}
    for c in conv_list:
        key = (c["state_type"], c["bits"])
        if key not in summary:
            summary[key] = {
                "stype": c["state_type"],
                "bits": c["bits"],
                "H": c["H"],
                "cut": c["cut"],
                "count": 0,
                "residuals": [],
                "is_opt": c["is_opt"],
            }
        summary[key]["count"] += 1
        summary[key]["residuals"].append(c["residual"])

    summ_vals = sorted(summary.values(), key=lambda x: -x["cut"])

    hdr2  = ["Type", "Bits", "H", "Cut", "n", "%", "Mean res.", "→ Opt?"]
    rows2 = []
    for s in summ_vals:
        bits_s = "".join(str(b) for b in s["bits"])
        rows2.append([
            s["stype"],
            bits_s,
            f"{s['H']:.2f}",
            f"{s['cut']:.2f}",
            str(s["count"]),
            f"{100*s['count']/n_init:.0f}%",
            f"{float(np.mean(s['residuals'])):.4f}",
            "★ YES" if s["is_opt"] else "no",
        ])

    rh = 0.048
    hh = 0.054
    th = 0.050
    gap = 0.040
    foot = 0.060
    n1 = len(rows1)
    n2 = len(rows2)

    total_h = 2*th + 2*hh + (n1+n2)*rh + gap + foot
    if total_h > 0.98:
        s = 0.98 / total_h
        rh *= s
        hh *= s
        th *= s
        gap *= s
        foot *= s

    def _section(y_cursor, title_txt, col_labels, data_rows, type_col_idx, opt_col_idx=None):
        nc = len(col_labels)
        nr = len(data_rows)

        banner_h = th * 0.9
        banner = FancyBboxPatch(
            (0.0, y_cursor - banner_h),
            1.0, banner_h,
            boxstyle="square,pad=0",
            transform=ax.transAxes,
            facecolor="#3a3a3a",
            edgecolor="none",
            alpha=0.88,
            clip_on=False,
            zorder=3
        )
        ax.add_patch(banner)
        ax.text(
            0.5, y_cursor - banner_h/2,
            title_txt,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=WHITE, zorder=5
        )

        y_cursor -= th
        table_h = hh + nr * rh
        bbox = [0.0, y_cursor - table_h, 1.0, table_h]

        tbl = ax.table(cellText=data_rows, colLabels=col_labels, bbox=bbox, cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.8)

        for j in range(nc):
            cell = tbl[0, j]
            cell.set_facecolor("#dce6f0")
            cell.set_text_props(fontweight="bold", color="#1a1a2e")
            cell.set_edgecolor("#9ab0cc")
            cell.set_linewidth(0.6)

        for i, row in enumerate(data_rows, start=1):
            stype = row[type_col_idx]
            bg    = _bg(stype)

            for j in range(nc):
                cell = tbl[i, j]
                shade = bg if (i % 2 == 1) else _lighten(bg, 0.55)
                cell.set_facecolor(shade)
                cell.set_edgecolor("#c8c8c8")
                cell.set_linewidth(0.4)
                cell.get_text().set_color(BLACK)

            bc = tbl[i, type_col_idx]
            bc.set_facecolor(_badge(stype))
            bc.get_text().set_color(WHITE)
            bc.get_text().set_fontweight("bold")
            bc.get_text().set_fontsize(7.0)

            if opt_col_idx is not None:
                oc = tbl[i, opt_col_idx]
                v  = row[opt_col_idx]
                if "★" in v:
                    oc.set_facecolor("#55a868")
                    oc.get_text().set_color(WHITE)
                    oc.get_text().set_fontweight("bold")
                elif "✓" in v:
                    oc.set_facecolor("#9dcf9a")
                elif "✗" in v:
                    oc.set_facecolor("#f7b89a")

        return y_cursor - table_h

    y = 0.99
    y = _section(y, "Per-trajectory convergence", hdr1, rows1, type_col_idx=1, opt_col_idx=5)
    y -= gap
    y = _section(y, "Summary — unique terminal states", hdr2, rows2, type_col_idx=0, opt_col_idx=7)
    y -= gap * 0.5

    diff   = mu - mu_bin
    above  = diff > 0
    status = "above  (binarises ✓)" if above else "below  (may not binarise ✗)"
    sc     = C_GREEN if above else C_ORANGE

    ax.text(
        0.5, max(y - 0.01, 0.01),
        (f"$\\mu = {mu:.4f}$   |   "
         f"$\\mu_{{\\rm bin}} = {mu_bin:.4f}$   |   "
         f"$\\mu - \\mu_{{\\rm bin}} = {diff:+.4f}$   ({status})"),
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=8, color=sc,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#f5f5f5",
            edgecolor=sc,
            linewidth=0.9,
            alpha=0.95
        )
    )

    legend_handles = [
        mpatches.Patch(facecolor=_badge(k), edgecolor=GRAY, label=k, alpha=0.88)
        for k in _STATE_BADGE
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        fontsize=7.5,
        facecolor=WHITE,
        edgecolor=GRAY,
        framealpha=0.92,
        labelcolor=BLACK
    )


# ── phase-dynamics drawing ────────────────────────────────────────────────────
def _draw_phase(ax, sols, conv, mu, mu_bin, w_total, best_cut, n, n_init):
    ax.cla()
    ax.set_facecolor(WHITE)
    for sp in ax.spines.values():
        sp.set_edgecolor(BLACK)
        sp.set_linewidth(0.8)
    ax.grid(True, color=LIGHT, linewidth=0.6, zorder=0)

    spin_cols = plt.get_cmap("tab20")(np.linspace(0, 1, max(n, 2)))
    t = sols[0].t

    for sol in sols:
        for s in range(n):
            ax.plot(t, sol.y[s], color=spin_cols[s % 20], alpha=0.42, linewidth=0.95, zorder=2)

    for yref, lw_r in [(np.pi,1.1),(np.pi/2,0.6),(0.,1.4),(-np.pi/2,0.6),(-np.pi,0.9)]:
        ax.axhline(yref, color=GRAY, linestyle="--", linewidth=lw_r, alpha=0.75, zorder=1)
        if abs(abs(yref) - np.pi/2) < 1e-9:
            lbl = r"$\pi/2$" if yref > 0 else r"$-\pi/2$"
            ax.text(t[-1]*0.995, yref+0.10, lbl, ha="right", va="bottom", fontsize=7.5, color=GRAY)

    ax.set_yticks(PI_TICKS)
    ax.set_yticklabels(PI_LABELS, fontsize=10, color=BLACK)
    ax.set_ylim(-4.2, 4.2)
    ax.set_xlim(t[0], t[-1])
    ax.tick_params(colors=BLACK, labelsize=9)

    n_bin  = sum(1 for c in conv if c["is_binary"])
    n_half = sum(1 for c in conv if c["state_type"] == "M1-half")
    n_t3   = sum(1 for c in conv if c["state_type"] == "Type-III")
    n_nc   = sum(1 for c in conv if c["state_type"] == "not-converged")
    all_b  = (n_bin == n_init)

    status = (
        "BINARISED ✓" if all_b
        else f"NOT YET BINARISED ✗  M2:{n_bin}  π/2:{n_half}  T3:{n_t3}  NC:{n_nc}"
    )
    ax.text(
        0.98, 0.97, status,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, fontweight="bold",
        color=C_BIN_OK if all_b else C_UNSTABLE,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE, edgecolor=GRAY, alpha=0.95)
    )

    ax.text(
        0.01, 0.97,
        f"$\\mu={mu:.4f}$ | $\\mu_{{\\rm bin}}={mu_bin:.4f}$ | "
        f"$W_{{\\rm tot}}={w_total:.1f}$ | best cut$={best_cut:.1f}$",
        transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.28", facecolor=WHITE, edgecolor=GRAY, alpha=0.93)
    )

    ax.legend(
        handles=[mpatches.Patch(color=spin_cols[s%20], label=f"spin {s}") for s in range(n)],
        loc="lower right",
        fontsize=7.5,
        ncol=max(1, n//5),
        framealpha=0.90
    )
    ax.set_xlabel("time $t$", color=BLACK, fontsize=11)
    ax.set_ylabel(r"phase $\theta_i(t)$ (rad)", color=BLACK, fontsize=11)
    above_below = "above" if mu > mu_bin else "below"
    ax.set_title(
        f"Phase dynamics "
        f"{n_init} initial conditions",
        color=BLACK, fontsize=11, pad=6
    )


# ── shared μ-sync controller ──────────────────────────────────────────────────
class SharedMuController:
    def __init__(self, mu_arr):
        self.mu_arr   = np.asarray(mu_arr, dtype=float)
        self.index    = len(self.mu_arr) // 2
        self._sliders  = []
        self._updaters = []
        self._syncing  = False

    def register_slider(self, slider):
        self._sliders.append(slider)

        def _cb(val):
            if self._syncing:
                return
            idx = int(np.argmin(np.abs(self.mu_arr - val)))
            self._set(idx, source=slider)

        slider.on_changed(_cb)

    def register_updater(self, fn):
        self._updaters.append(fn)

    def _set(self, idx, source=None):
        idx = int(np.clip(idx, 0, len(self.mu_arr) - 1))
        self.index = idx
        mu = float(self.mu_arr[idx])

        self._syncing = True
        try:
            for s in self._sliders:
                if s is not source and float(s.val) != mu:
                    s.set_val(mu)
        finally:
            self._syncing = False

        for fn in self._updaters:
            fn(idx)

    def trigger(self):
        self._set(self.index)


# ── Figure 1: phase dynamics + styled table ───────────────────────────────────
def make_phase_figure(ctrl, results, eq_data, args):
    mu_arr   = ctrl.mu_arr
    mu_bin   = eq_data["mu_bin"]
    w_total  = eq_data["w_total"]
    best_cut = eq_data["best_cut"]
    n        = eq_data["n"]
    n_eq     = eq_data["total"]
    n_init   = args.n_init

    fig = plt.figure(figsize=(22, 9.5), facecolor=WHITE)
    ax_phase  = fig.add_axes((0.04, 0.13, 0.54, 0.78))
    ax_table  = fig.add_axes((0.61, 0.13, 0.38, 0.78))
    ax_slider = fig.add_axes((0.10, 0.035, 0.80, 0.030))

    for ax in (ax_phase, ax_table):
        ax.set_facecolor(WHITE)
        for sp in ax.spines.values():
            sp.set_edgecolor(BLACK)
            sp.set_linewidth(0.8)

    slider = Slider(
        ax_slider, "$\\mu$",
        mu_arr[0], mu_arr[-1],
        valinit=float(mu_arr[ctrl.index]),
        valstep=mu_arr,
        color=C_STABLE,
        track_color=LIGHT
    )
    ax_slider.set_facecolor(WHITE)
    slider.label.set_color(BLACK)
    slider.valtext.set_color(BLACK)
    ax_slider.axvline(mu_bin, color=C_MUBIN, linewidth=2.5, zorder=5)
    rel = float(np.clip((mu_bin - mu_arr[0]) / (mu_arr[-1] - mu_arr[0] + 1e-12), 0, 1))
    ax_slider.text(
        rel, 1.05, f"$\\mu_{{\\rm bin}}={mu_bin:.3f}$",
        transform=ax_slider.transAxes,
        ha="center", va="bottom", fontsize=8.5, color=C_MUBIN
    )

    def update(idx):
        rec = results[idx]
        mu = rec["mu"]
        _draw_phase(ax_phase, rec["sols"], rec["conv"], mu, mu_bin, w_total, best_cut, n, n_init)
        _draw_styled_table(ax_table, rec["conv"], best_cut, n_init, mu, mu_bin)
        fig.suptitle(
            f"OIM μ-slider | {args.graph} | $N={n}$, $2^N={n_eq}$ equilibria | "
            f"$\\mu_{{\\rm bin}}={mu_bin:.4f}$ | "
            f"best cut$={best_cut:.1f}$, $W_{{\\rm tot}}={w_total:.1f}$ | "
            f"step {idx+1}/{len(results)}  ($\\mu={mu:.4f}$)",
            color=BLACK, fontsize=11, fontweight="bold", y=0.99
        )
        fig.canvas.draw_idle()

    ctrl.register_slider(slider)
    ctrl.register_updater(update)
    fig.slider = slider
    return fig


# ── Figure 2: Jacobian / D-matrix spectra ────────────────────────────────────
def make_spectrum_figure(ctrl, eq_data, args):
    mu_arr   = ctrl.mu_arr
    rows     = eq_data["rows"]
    mu_bin   = eq_data["mu_bin"]
    w_total  = eq_data["w_total"]
    best_cut = eq_data["best_cut"]
    n        = eq_data["n"]
    n_eq     = eq_data["total"]
    reps     = pick_representatives(rows)

    fig = plt.figure(figsize=(18, 8.5), facecolor=WHITE)
    ax_aspec  = fig.add_axes((0.05, 0.15, 0.42, 0.72))
    ax_dspec  = fig.add_axes((0.55, 0.15, 0.42, 0.72))
    ax_slider = fig.add_axes((0.15, 0.04, 0.70, 0.03))

    for ax in (ax_aspec, ax_dspec):
        ax.set_facecolor(WHITE)
        ax.tick_params(colors=BLACK, labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor(BLACK)
            sp.set_linewidth(0.8)

    xidx   = np.arange(1, n + 1)
    ev_all = np.concatenate([eq["ev_D"] for eq, _, _, _ in reps])
    ev_lo, ev_hi = ev_all.min(), ev_all.max()
    margin = max(0.06 * (ev_hi - ev_lo), 0.4)
    mu_min = float(mu_arr[0])
    mu_max = float(mu_arr[-1])
    init_mu = float(mu_arr[ctrl.index])

    ax_dspec.fill_between([0.5, n+0.5], ev_lo-margin, 0, color=C_STABLE, alpha=0.10, zorder=0)
    ax_dspec.fill_between([0.5, n+0.5], 0, ev_hi+margin, color=C_UNSTABLE, alpha=0.10, zorder=0)
    ax_dspec.axhline(0, color=BLACK, linewidth=1.0, linestyle="--", alpha=0.55, zorder=2)

    for k_r, (eq, lbl, col, ls) in enumerate(reps):
        ev_desc = eq["ev_D"][::-1]
        ax_dspec.plot(xidx, ev_desc, color=col, linestyle=ls, lw=2.2, marker="o", ms=8, label=lbl, zorder=3)
        dy = margin * (0.6 + 0.35*k_r)
        ax_dspec.annotate(
            f"$\\lambda_{{max}}={eq['lmax_D']:.3f}$\n$H={eq['H']:.1f}$  cut$={eq['cut']:.1f}$",
            xy=(1, ev_desc[0]), xytext=(1.6, ev_desc[0]+dy),
            fontsize=9, color=col, zorder=5,
            arrowprops=dict(arrowstyle="->", color=col, lw=0.9)
        )

    hline_mu = ax_dspec.axhline(init_mu, color=C_MU_LINE, linewidth=2.0, linestyle="--", zorder=4, label="Current $\\mu$")
    ax_dspec.set_xticks(xidx)
    ax_dspec.set_xlim(0.5, n+0.5)
    ax_dspec.set_ylim(ev_lo - 2*margin, ev_hi + 3.5*margin)
    ax_dspec.legend(fontsize=8.5, loc="lower left")
    d_text = ax_dspec.text(
        0.98, 0.98, "", transform=ax_dspec.transAxes,
        ha="right", va="top", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=WHITE, edgecolor=GRAY, alpha=0.94)
    )
    ax_dspec.set_title("$D(\\phi^*)$ eigenvalue spectrum\n(Topology dependent, independent of $\\mu$)", fontsize=11, pad=10)
    ax_dspec.set_xlabel("Eigenvalue rank $k$ (largest first)", fontsize=10)
    ax_dspec.set_ylabel("$\\lambda_k\\left(D(\\phi^*)\\right)$", fontsize=10)
    ax_dspec.grid(True, color=LIGHT, linewidth=0.6, zorder=0)

    a_ylim_lo = ev_lo - mu_max - 2*margin
    a_ylim_hi = ev_hi - mu_min + 3.5*margin
    ax_aspec.fill_between([0.5, n+0.5], a_ylim_lo, 0, color=C_STABLE, alpha=0.10, zorder=0)
    ax_aspec.fill_between([0.5, n+0.5], 0, a_ylim_hi, color=C_UNSTABLE, alpha=0.10, zorder=0)
    ax_aspec.axhline(0, color=BLACK, linewidth=1.5, linestyle="--", zorder=2, label="Stability boundary (0)")

    a_lines = []
    a_annots = []
    dy_list = []
    for k_r, (eq, lbl, col, ls) in enumerate(reps):
        ev_d = eq["ev_D"][::-1] - init_mu
        line, = ax_aspec.plot(xidx, ev_d, color=col, linestyle=ls, lw=2.2, marker="o", ms=8, label=lbl, zorder=3)
        a_lines.append(line)
        dy = margin * (0.6 + 0.35*k_r)
        dy_list.append(dy)
        ann = ax_aspec.annotate(
            f"$\\lambda_{{max}}={eq['lmax_D']-init_mu:.3f}$\n$H={eq['H']:.1f}$  cut$={eq['cut']:.1f}$",
            xy=(1, ev_d[0]), xytext=(1.6, ev_d[0]+dy),
            fontsize=9, color=col, zorder=5,
            arrowprops=dict(arrowstyle="->", color=col, lw=0.9)
        )
        a_annots.append(ann)

    ax_aspec.set_xticks(xidx)
    ax_aspec.set_xlim(0.5, n+0.5)
    ax_aspec.set_ylim(a_ylim_lo, a_ylim_hi)
    ax_aspec.legend(fontsize=8.5, loc="lower left")
    a_text = ax_aspec.text(
        0.98, 0.98, "", transform=ax_aspec.transAxes,
        ha="right", va="top", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=WHITE, edgecolor=GRAY, alpha=0.94)
    )
    ax_aspec.set_title("$A(\\phi^*,\\mu)$ Jacobian spectrum\n(Dynamically shifts by $-\\mu$)", fontsize=11, pad=10)
    ax_aspec.set_xlabel("Eigenvalue rank $k$ (largest first)", fontsize=10)
    ax_aspec.set_ylabel("$\\lambda_k\\left(A(\\phi^*)\\right) = \\lambda_k(D) - \\mu$", fontsize=10)
    ax_aspec.grid(True, color=LIGHT, linewidth=0.6, zorder=0)

    slider = Slider(
        ax_slider, "$\\mu$",
        mu_arr[0], mu_arr[-1],
        valinit=init_mu,
        valstep=mu_arr,
        color=C_STABLE,
        track_color=LIGHT
    )
    ax_slider.set_facecolor(WHITE)
    slider.label.set_color(BLACK)
    slider.valtext.set_color(BLACK)
    ax_slider.axvline(mu_bin, color=C_MU_LINE, linewidth=2.5, zorder=5)
    rel = float(np.clip((mu_bin - mu_arr[0]) / (mu_arr[-1] - mu_arr[0] + 1e-12), 0, 1))
    ax_slider.text(
        rel, 1.05, f"$\\mu_{{bin}}={mu_bin:.3f}$",
        transform=ax_slider.transAxes,
        ha="center", va="bottom", fontsize=8.5, color=C_MU_LINE
    )

    def update(idx):
        mu = float(mu_arr[idx])
        n_stable = sum(1 for r in rows if mu > r["lmax_D"])

        hline_mu.set_ydata([mu, mu])
        d_text.set_text(
            f"Current $\\mu = {mu:.4f}$\n"
            f"Stable iff $\\mu > \\lambda_{{max}}(D)$\n"
            f"Stable equilibria: {n_stable} / {n_eq}"
        )

        for k_r, (eq, lbl, col, ls) in enumerate(reps):
            ev_a = eq["ev_D"][::-1] - mu
            a_lines[k_r].set_ydata(ev_a)
            a_annots[k_r].xy = (1, ev_a[0])
            a_annots[k_r].set_position((1.6, ev_a[0] + dy_list[k_r]))
            a_annots[k_r].set_text(
                f"$\\lambda_{{max}}={eq['lmax_D']-mu:.3f}$\n"
                f"$H={eq['H']:.1f}$  cut$={eq['cut']:.1f}$"
            )

        a_text.set_text(
            f"Jacobian at $\\mu = {mu:.4f}$\n"
            f"Stable iff $\\lambda_{{max}}(A) < 0$\n"
            f"Stable equilibria: {n_stable} / {n_eq}"
        )

        fig.suptitle(
            f"OIM Eigenvalue Spectrum | {args.graph} | "
            f"$N={n}$, $2^N={n_eq}$ equilibria | "
            f"$\\mu_{{bin}}={mu_bin:.4f}$ | "
            f"Best cut$={best_cut:.1f}$, $W_{{tot}}={w_total:.1f}$ | "
            f"$\\mu={mu:.4f}$",
            color=BLACK, fontsize=13, fontweight="bold", y=0.98
        )
        fig.canvas.draw_idle()

    ctrl.register_slider(slider)
    ctrl.register_updater(update)
    fig.slider = slider
    return fig


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Combined OIM μ-slider: phase dynamics + eigenvalue spectra"
    )
    parser.add_argument("--graph", required=True)
    parser.add_argument("--mu_min", type=float, default=None)
    parser.add_argument("--mu_max", type=float, default=None)
    parser.add_argument("--n_mu", type=int, default=20)
    parser.add_argument("--mu_values", type=float, nargs="+", default=None)
    parser.add_argument("--n_init", type=int, default=12)
    parser.add_argument("--t_end", type=float, default=80.0)
    parser.add_argument("--n_points", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\nLoading graph: {args.graph}")
    W = read_graph(args.graph)
    n = W.shape[0]
    edges = [(i, j, W[i,j]) for i in range(n) for j in range(i+1, n) if W[i,j] > 0]
    print(f" N={n}  |E|={len(edges)}  W_total={W.sum()/2:.2f}")
    if n > 18:
        print(f"[warn] N={n} > 18 — 2^N={2**n} equilibrium scan may be slow")

    print(f" Scanning 2^{n}={2**n} equilibria for λ_max range...")
    oim_scan = OIMMaxCut(W, mu=1.0, seed=args.seed)
    lmax_scan = []
    for bits in iproduct([0, 1], repeat=n):
        phi = np.array([b*np.pi for b in bits], dtype=float)
        lmax_scan.append(float(np.linalg.eigvalsh(oim_scan.build_D(phi)).max()))

    global_lmax_min = min(lmax_scan)
    global_lmax_max = max(lmax_scan)
    print(f" λ_max(D) range : [{global_lmax_min:.4f}, {global_lmax_max:.4f}]")

    if args.mu_values is not None:
        mu_list = sorted(set(args.mu_values))
        print(f" Using {len(mu_list)} explicit μ values: {[f'{v:.3f}' for v in mu_list]}")
    else:
        mu_min = args.mu_min if args.mu_min is not None else max(0., global_lmax_min - 0.5)
        mu_max = args.mu_max if args.mu_max is not None else global_lmax_max * 1.30
        mu_list = list(np.linspace(mu_min, mu_max, args.n_mu))
        print(f" μ sweep: [{mu_min:.4f}, {mu_max:.4f}]  ({args.n_mu} steps)")

    mu_ref  = float(np.median(mu_list))
    oim_ref = OIMMaxCut(W, mu=mu_ref, seed=args.seed)
    print(f"\n Equilibrium analysis at μ_ref={mu_ref:.4f}...")
    eq_data = analyse_equilibria(oim_ref)
    print(
        f" μ_bin={eq_data['mu_bin']:.4f}  best cut={eq_data['best_cut']:.1f}  "
        f"stable at ref μ: {eq_data['n_stable']}/{eq_data['total']}"
    )

    rng = np.random.default_rng(args.seed)
    phi0s = [rng.uniform(-np.pi, np.pi, n) for _ in range(args.n_init)]
    print(f" {args.n_init} initial conditions sampled from uniform(−π, π)")

    results = precompute(
        mu_list, W, phi0s,
        args.t_end, args.n_points,
        eq_data["rows"], eq_data["best_cut"], args.seed
    )

    ctrl = SharedMuController(mu_list)
    make_phase_figure(ctrl, results, eq_data, args)
    make_spectrum_figure(ctrl, eq_data, args)
    ctrl.trigger()

    print("\nBoth windows launched. Moving either slider syncs the other.\n")
    plt.show()


if __name__ == "__main__":
    main()