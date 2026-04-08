#!/usr/bin/env python3
"""
mu_slider_experiment.py
─────────────────────────────────────────────────────────────────────────────
Interactive μ-sweep for OIM phase dynamics (reproduces Figure 1).

Pre-computes ODE trajectories at every requested μ value, then renders an
interactive Matplotlib window with:
  • Phase-trajectory panel  (same layout as Figure 1 left)
  • Convergence-table panel (same layout as Figure 1 right)
  • μ slider at the bottom  (snaps to pre-computed values)
  • ◀ / ▶ step buttons
  • μ_bin marker on the slider track

Usage
─────
# Uniform sweep from μ_bin to 1.3·λ_max in 20 steps
python mu_slider_experiment.py --graph data/10node.txt

# Explicit μ values
python mu_slider_experiment.py --graph data/10node.txt \\
    --mu_values 0.01 0.5 1.0 2.0 3.0 4.0 6.0

# Denser sweep, longer integration
python mu_slider_experiment.py --graph data/10node.txt \\
    --mu_min 0.0 --mu_max 7.0 --n_mu 30 --t_end 120 --n_init 16

CLI options
───────────
--graph      PATH          graph file (required)
--mu_min     FLOAT         lower bound (default: max(0, μ_bin − 0.5))
--mu_max     FLOAT         upper bound (default: 1.3 × max λ_max(D))
--n_mu       INT           number of evenly-spaced μ steps (default: 20)
--mu_values  FLOAT ...     explicit μ list — overrides --mu_min/max/n_mu
--n_init     INT           random initial conditions per μ (default: 12)
--t_end      FLOAT         ODE integration horizon (default: 80)
--n_points   INT           time-grid points per trajectory (default: 500)
--seed       INT           RNG seed (default: 42)
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import sys
from itertools import product as iproduct

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button

from OIM_Experiment.src.OIM_mu import OIMMaxCut
from OIM_Experiment.src.graph_utils import read_graph

# ── style ─────────────────────────────────────────────────────────────────────
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
C_MU_LINE  = "#ffb74d"
C_MUBIN    = "#c44e52"

TYPE_COL = {
    "M2-binary":   C_BIN_OK,
    "M1-half":     C_UNSTABLE,
    "M1-mixed":    "#e377c2",
    "Type-III":    "#8172b2",
    "not-converged": GRAY,
}

PI_TICKS  = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
PI_LABELS = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]


# ── helpers ───────────────────────────────────────────────────────────────────
def _jacobian(oim, phi):
    D = oim.build_D(phi)
    return D - oim.mu * np.diag(np.cos(2.0 * phi))


def analyse_equilibria(oim):
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
        rows.append(dict(bits=bits, phi=phi, D=D, ev_D=ev_D, ev_A=ev_A,
                         lmax_D=lmax, lmax_A=float(ev_A[-1]), mu_thr=lmax,
                         stable=mu > lmax, H=H, cut=cut))
    mu_bin   = min(r["lmax_D"] for r in rows)
    best_cut = max(r["cut"]    for r in rows)
    n_stable = sum(r["stable"] for r in rows)
    return dict(rows=rows, mu_bin=mu_bin, best_cut=best_cut,
                w_total=w_total, n_stable=n_stable,
                total=len(rows), n=n, mu=mu)


def _atom(theta_i, tol=0.05):
    s, c = np.sin(theta_i), np.cos(theta_i)
    if abs(s) < tol:
        return "zero" if c > 0 else "pi"
    if abs(abs(s) - 1.0) < tol:
        return "half"
    return "other"


def identify_convergence(sol, W, eq_rows, bintol=0.05):
    theta    = sol.y[:, -1].copy()
    n        = len(theta)
    residual = float(np.max(np.abs(np.sin(theta))))
    sigma    = np.sign(np.cos(theta));  sigma[sigma == 0] = 1.0
    bits     = tuple(0 if s > 0 else 1 for s in sigma)
    H        = 0.5  * float(np.sum(W * np.outer(sigma, sigma)))
    cut      = 0.25 * float(np.sum(W * (1.0 - np.outer(sigma, sigma))))
    atoms    = [_atom(t, bintol) for t in theta]
    nz = atoms.count("zero");  np_ = atoms.count("pi")
    nh = atoms.count("half");  no  = atoms.count("other")
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
    # nearest M2 equilibrium
    nearest = None
    min_dist = np.inf
    for r in eq_rows:
        diff = theta - r["phi"]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        d = float(np.linalg.norm(diff))
        if d < min_dist:
            min_dist = d
            nearest  = dict(bits=r["bits"], H=r["H"], cut=r["cut"],
                            mu_thr=r["mu_thr"], stable=r["stable"],
                            dist_L2=d)
    return dict(theta=theta, bits=bits, H=H, cut=cut, residual=residual,
                is_binary=(stype == "M2-binary"), state_type=stype,
                nearest=nearest)


# ── pre-computation ───────────────────────────────────────────────────────────
def precompute(mu_list, W, phi0s, t_end, n_points, eq_rows, seed):
    """
    Simulate n_init trajectories for each μ in mu_list.
    Returns list of dicts: {mu, sols, conv, n_stable}.
    """
    n = W.shape[0]
    results = []
    total   = len(mu_list)
    bar_w   = 40

    print(f"\n  Pre-computing {total} μ values  "
          f"({len(phi0s)} trajectories × t_end={t_end}):")

    for i, mu in enumerate(mu_list):
        # progress bar
        filled = int(bar_w * (i / total))
        bar    = "█" * filled + "░" * (bar_w - filled)
        pct    = 100 * i / total
        print(f"  [{bar}] {pct:5.1f}%  μ={mu:.4f}   ", end="\r", flush=True)

        oim  = OIMMaxCut(W, mu=mu, seed=seed)
        sols = oim.simulate_many(phi0s, t_span=(0., t_end), n_points=n_points)

        # stability at this μ (reuse eq_rows, just count)
        n_stable = sum(1 for r in eq_rows if mu > r["lmax_D"])

        conv = [identify_convergence(s, W, eq_rows) for s in sols]
        results.append(dict(mu=mu, sols=sols, conv=conv, n_stable=n_stable))

    print(f"  [{'█'*bar_w}] 100.0%  done.           ")
    return results


# ── drawing helpers ───────────────────────────────────────────────────────────
def _draw_phase(ax, sols, conv, mu, mu_bin, w_total, best_cut, n, n_init):
    ax.cla()
    ax.set_facecolor(WHITE)
    for sp in ax.spines.values():
        sp.set_edgecolor(BLACK); sp.set_linewidth(0.8)
    ax.grid(True, color=LIGHT, linewidth=0.6, zorder=0)

    SPIN_COLS = plt.get_cmap("tab20")(np.linspace(0, 1, max(n, 2)))
    t = sols[0].t

    for sol in sols:
        for spin in range(n):
            ax.plot(t, sol.y[spin],
                    color=SPIN_COLS[spin % 20],
                    alpha=0.42, linewidth=0.95, zorder=2)

    for yref, lw_r in [(np.pi, 1.1), (np.pi/2, 0.6),
                       (0.0, 1.4), (-np.pi/2, 0.6), (-np.pi, 0.9)]:
        ax.axhline(yref, color=GRAY, linestyle="--",
                   linewidth=lw_r, alpha=0.75, zorder=1)
        if abs(abs(yref) - np.pi/2) < 1e-9:
            lbl = "$\\pi/2$" if yref > 0 else "$-\\pi/2$"
            ax.text(t[-1] * 0.995, yref + 0.10, lbl,
                    ha="right", va="bottom", fontsize=7.5, color=GRAY)

    ax.set_yticks(PI_TICKS)
    ax.set_yticklabels(PI_LABELS, fontsize=10, color=BLACK)
    ax.set_ylim(-4.2, 4.2)
    ax.set_xlim(t[0], t[-1])
    ax.tick_params(colors=BLACK, labelsize=9)

    # convergence summary counts
    n_bin  = sum(1 for c in conv if c["is_binary"])
    n_half = sum(1 for c in conv if c["state_type"] == "M1-half")
    n_t3   = sum(1 for c in conv if c["state_type"] == "Type-III")
    n_nc   = sum(1 for c in conv if c["state_type"] == "not-converged")
    is_all_bin = (n_bin == n_init)

    status = ("BINARISED ✓" if is_all_bin
              else f"NOT YET BINARISED ✗   "
                   f"M2:{n_bin}  π/2-trap:{n_half}  "
                   f"TypeIII:{n_t3}  no-conv:{n_nc}")
    ax.text(0.98, 0.97, status,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, fontweight="bold",
            color=C_BIN_OK if is_all_bin else C_UNSTABLE,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE,
                      edgecolor=GRAY, alpha=0.95))

    above_below = "above" if mu > mu_bin else "below"
    ax.text(0.01, 0.97,
            f"$\\mu={mu:.4f}$  |  $\\mu_{{\\rm bin}}={mu_bin:.4f}$  |  "
            f"$W_{{\\rm tot}}={w_total:.1f}$  |  best cut$={best_cut:.1f}$",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=WHITE,
                      edgecolor=GRAY, alpha=0.93))

    spin_patches = [mpatches.Patch(color=SPIN_COLS[s % 20], label=f"spin {s}")
                    for s in range(n)]
    ax.legend(handles=spin_patches, loc="lower right", fontsize=7.5,
              ncol=max(1, n // 5), framealpha=0.90)

    ax.set_xlabel("time $t$", color=BLACK, fontsize=11)
    ax.set_ylabel("phase $\\theta_i(t)$  (rad)", color=BLACK, fontsize=11)
    ax.set_title(
        f"Phase dynamics  $\\mu = {mu:.4f}$ "
        f"({above_below} $\\mu_{{\\rm bin}} = {mu_bin:.4f}$)  |  "
        f"{n_init} initial conditions",
        color=BLACK, fontsize=11, pad=6)


def _draw_table(ax, conv, mu, mu_bin, n, n_init):
    ax.cla()
    ax.set_facecolor(WHITE)
    ax.axis("off")
    for sp in ax.spines.values():
        sp.set_edgecolor(BLACK); sp.set_linewidth(0.8)
    ax.set_title("Convergence — terminal states",
                 color=BLACK, fontsize=11, pad=6)

    cw    = max(n, 8)
    sep   = "─" * (3 + 14 + cw + 7 + 7 + 5 + 8)
    lines = [f"{'#':>3}  {'type':<14}  {'φ* bits':<{cw}}  "
             f"{'H':>6}  {'cut':>6}  {'bin':>4}  res", sep]

    for i, c in enumerate(conv):
        b  = "".join(str(x) for x in c["bits"])
        bs = "✓" if c["is_binary"] else "✗"
        lines.append(f"{i:>3}  {c['state_type']:<14}  {b:<{cw}}  "
                     f"{c['H']:>6.2f}  {c['cut']:>6.2f}  {bs:>4}  "
                     f"{c['residual']:.3f}")

    # unique terminal states summary
    summary = {}
    for c in conv:
        key = (c["state_type"], c["bits"])
        if key not in summary:
            summary[key] = {"st": c["state_type"], "H": c["H"],
                            "cut": c["cut"], "count": 0, "res": []}
        summary[key]["count"] += 1
        summary[key]["res"].append(c["residual"])

    lines += ["",
              f"Unique terminal states: {len(summary)}",
              f"{'type':<14}  {'bits':<{cw}}  {'H':>6}  "
              f"{'cut':>6}  {'n':>3}  mean res",
              "─" * (14 + cw + 6 + 6 + 3 + 10)]
    for (st, bits), info in sorted(summary.items(),
                                   key=lambda x: -x[1]["cut"]):
        b = "".join(str(x) for x in bits)
        lines.append(f"{info['st']:<14}  {b:<{cw}}  {info['H']:>6.2f}  "
                     f"{info['cut']:>6.2f}  {info['count']:>3}  "
                     f"{np.mean(info['res']):.4f}")

    # μ vs μ_bin status line
    diff = mu - mu_bin
    lines += ["",
              f"μ − μ_bin = {diff:+.4f}  "
              f"({'above ✓' if diff > 0 else 'below ✗'} binarisation threshold)"]

    ax.text(0.03, 0.97, "\n".join(lines),
            transform=ax.transAxes, va="top", ha="left",
            fontsize=7.2, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=WHITE,
                      edgecolor=GRAY, alpha=0.95))

    # type legend
    patches = [mpatches.Patch(facecolor=v, edgecolor=GRAY, label=k)
               for k, v in TYPE_COL.items()]
    ax.legend(handles=patches, loc="lower center",
              bbox_to_anchor=(0.5, 0.0), ncol=3,
              fontsize=7.5, facecolor=WHITE, edgecolor=GRAY,
              framealpha=0.92, labelcolor=BLACK)


# ── main interactive figure ───────────────────────────────────────────────────
def make_interactive(results, eq_data, args):
    mu_arr   = np.array([r["mu"] for r in results])
    mu_bin   = eq_data["mu_bin"]
    w_total  = eq_data["w_total"]
    best_cut = eq_data["best_cut"]
    n        = eq_data["n"]
    n_eq     = eq_data["total"]
    n_init   = args.n_init

    init_mu = mu_arr[len(mu_arr) // 2]

    # ── figure + axes (add_axes tuples, exactly like reference) ──────────────
    fig = plt.figure(figsize=(22, 9.5), facecolor=WHITE)

    ax_phase  = fig.add_axes((0.04, 0.13, 0.57, 0.78))
    ax_table  = fig.add_axes((0.64, 0.13, 0.35, 0.78))
    ax_slider = fig.add_axes((0.10, 0.035, 0.80, 0.030))

    for ax in (ax_phase, ax_table):
        ax.set_facecolor(WHITE)
        ax.tick_params(colors=BLACK, labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor(BLACK)
            sp.set_linewidth(0.8)
        ax.grid(True, color=LIGHT, linewidth=0.6, alpha=1.0)

    # ── slider (exactly like reference) ──────────────────────────────────────
    # valstep must be the full array of allowed values, NOT a scalar step size.
    # A scalar valstep snaps to vmin + n*step, which drifts off the precomputed
    # mu_arr values due to floating-point accumulation, so slider.val never
    # actually changes and the on_changed callback stops firing.
    slider = Slider(ax_slider, "$\\mu$", mu_arr[0], mu_arr[-1],
                    valinit=init_mu, valstep=mu_arr,
                    color=C_STABLE, track_color=LIGHT)
    ax_slider.set_facecolor(WHITE)
    slider.label.set_color(BLACK)
    slider.valtext.set_color(BLACK)

    # μ_bin marker on the slider track (data coordinate — no transform)
    ax_slider.axvline(mu_bin, color=C_MUBIN, linewidth=2.5, zorder=5)
    rel = np.clip((mu_bin - mu_arr[0]) / (mu_arr[-1] - mu_arr[0] + 1e-12), 0, 1)
    ax_slider.text(rel, 1.05, f"$\\mu_{{\\rm bin}}={mu_bin:.3f}$",
                   transform=ax_slider.transAxes,
                   ha="center", va="bottom", fontsize=8.5, color=C_MUBIN)

    # ── update callback (mirrors reference exactly) ───────────────────────────
    def update(val):
        idx = int(np.argmin(np.abs(mu_arr - slider.val)))   # use slider.val
        rec = results[idx]
        mu  = rec["mu"]

        _draw_phase(ax_phase, rec["sols"], rec["conv"],
                    mu, mu_bin, w_total, best_cut, n, n_init)
        _draw_table(ax_table, rec["conv"], mu, mu_bin, n, n_init)

        fig.suptitle(
            f"OIM μ-slider experiment  |  {args.graph}  |  "
            f"$N={n}$,  $2^N={n_eq}$ equilibria  |  "
            f"$\\mu_{{\\rm bin}}={mu_bin:.4f}$  |  "
            f"best cut$={best_cut:.1f}$,  $W_{{\\rm tot}}={w_total:.1f}$  |  "
            f"step {idx+1}/{len(results)}  ($\\mu={mu:.4f}$)",
            color=BLACK, fontsize=11, fontweight="bold", y=0.99)

        fig.canvas.draw_idle()

    slider.on_changed(update)

    # initial render + show (exactly like reference)
    update(init_mu)
    fig.slider = slider
    return fig


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Interactive μ-slider for OIM phase dynamics (Figure 1 live)")
    parser.add_argument("--graph",     required=True,
                        help="Path to graph .txt file")
    parser.add_argument("--mu_min",    type=float, default=None,
                        help="Lower bound of μ sweep (default: max(0, μ_bin−0.5))")
    parser.add_argument("--mu_max",    type=float, default=None,
                        help="Upper bound (default: 1.3 × max λ_max(D))")
    parser.add_argument("--n_mu",      type=int,   default=20,
                        help="Number of evenly-spaced μ steps (default: 20)")
    parser.add_argument("--mu_values", type=float, nargs="+", default=None,
                        help="Explicit μ list — overrides --mu_min/max/n_mu")
    parser.add_argument("--n_init",    type=int,   default=12,
                        help="Random initial conditions per μ (default: 12)")
    parser.add_argument("--t_end",     type=float, default=80.0,
                        help="ODE integration horizon (default: 80)")
    parser.add_argument("--n_points",  type=int,   default=500,
                        help="Time-grid points per trajectory (default: 500)")
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    # ── load graph ─────────────────────────────────────────────────────────────
    print(f"\nLoading graph: {args.graph}")
    W     = read_graph(args.graph)
    n     = W.shape[0]
    edges = [(i, j, W[i, j])
             for i in range(n) for j in range(i + 1, n) if W[i, j] > 0]
    print(f"  N={n}  |E|={len(edges)}  W_total={W.sum()/2:.2f}")
    if n > 18:
        print(f"[warn] N={n} > 18 — 2^N={2**n} equilibrium scan may be slow")

    # ── scan λ_max range ───────────────────────────────────────────────────────
    oim_scan = OIMMaxCut(W, mu=1.0, seed=args.seed)
    print(f"  Scanning 2^{n}={2**n} equilibria for λ_max range...")
    lmax_scan = []
    for bits in iproduct([0, 1], repeat=n):
        phi = np.array([b * np.pi for b in bits], dtype=float)
        lmax_scan.append(float(np.linalg.eigvalsh(oim_scan.build_D(phi)).max()))

    global_lmax_min = min(lmax_scan)
    global_lmax_max = max(lmax_scan)
    print(f"  λ_max(D) range : [{global_lmax_min:.4f}, {global_lmax_max:.4f}]")
    print(f"  μ_bin estimate : {global_lmax_min:.4f}")

    # ── build μ list ───────────────────────────────────────────────────────────
    if args.mu_values is not None:
        mu_list = sorted(set(args.mu_values))
        print(f"  Using {len(mu_list)} explicit μ values: "
              f"{[f'{v:.3f}' for v in mu_list]}")
    else:
        mu_min = (args.mu_min if args.mu_min is not None
                  else max(0.0, global_lmax_min - 0.5))
        mu_max = (args.mu_max if args.mu_max is not None
                  else global_lmax_max * 1.30)
        mu_list = list(np.linspace(mu_min, mu_max, args.n_mu))
        print(f"  μ sweep: [{mu_min:.4f}, {mu_max:.4f}]  ({args.n_mu} steps)")

    # ── reference equilibrium analysis (at midpoint μ) ─────────────────────────
    mu_ref  = float(np.median(mu_list))
    oim_ref = OIMMaxCut(W, mu=mu_ref, seed=args.seed)
    print(f"\n  Equilibrium analysis (reference μ={mu_ref:.4f})...")
    eq_data = analyse_equilibria(oim_ref)
    print(f"  μ_bin={eq_data['mu_bin']:.4f}  "
          f"best cut={eq_data['best_cut']:.1f}  "
          f"stable at ref μ: {eq_data['n_stable']}/{eq_data['total']}")

    # ── fixed initial conditions (same across all μ) ──────────────────────────
    rng   = np.random.default_rng(args.seed)
    phi0s = [rng.uniform(-np.pi, np.pi, n) for _ in range(args.n_init)]
    print(f"\n  {args.n_init} initial conditions sampled from uniform(−π, π)")

    # ── pre-computation ────────────────────────────────────────────────────────
    results = precompute(mu_list, W, phi0s,
                         args.t_end, args.n_points,
                         eq_data["rows"], args.seed)

    # ── launch interactive window ──────────────────────────────────────────────
    print("\n  Launching interactive window …  "
          "(use slider or ◀/▶ buttons to sweep μ)\n")
    fig = make_interactive(results, eq_data, args)
    plt.show()


if __name__ == "__main__":
    main()
