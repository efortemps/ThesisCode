#!/usr/bin/env python3
"""
experiment_maxcut_interactive.py
---------------------------------
Interactive Max-Cut explorer using OIMMaxCut (OIM_mu_v2.py).

Layout (3-panel right column)
------------------------------
  LEFT           : Phase evolution theta_i(t) for all spins — redraws from cache.
  RIGHT-TOP      : Per-equilibrium stability threshold bar chart (Theorem 2)
                     bars        = mu*_i = lambda_max(D(phi*_i))
                     white dot.  = mu_bin = min_i(mu*_i)  [binarisation threshold]
                     orange dash = current mu
                     red/green shading below/above mu_bin
  RIGHT-BOTTOM   : Final Hamiltonian H per initial condition
                     H = sum_{i<j} W_ij sigma_i sigma_j  (= Lyapunov energy at bin.)
                     OIM minimises H  <=>  maximises cut
                     H = W_total - 2*cut
  SLIDER         : Sweeps mu in [mu_min, mu_max].

JSON save: every slider move → timestamped snapshot via MaxCutExperimentLogger.

Usage
-----
    python experiment_maxcut_interactive.py [graph.txt] [options]
    --mu_min  --mu_max  --n_mu  --n_init  --t_end  --n_points  --seed
"""

import argparse
from pathlib import Path
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.widgets import Slider

from OIM_Experiment.src.OIM_mu_v2 import OIMMaxCut
from OIM_Experiment.src.graph_utils import read_graph
from OIM_Experiment.src.experiment_logger2 import MaxCutExperimentLogger

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("graph",      nargs="?", default="3node.txt")
parser.add_argument("--mu_min",   type=float, default=0.2)
parser.add_argument("--mu_max",   type=float, default=5.0)
parser.add_argument("--n_mu",     type=int,   default=20)
parser.add_argument("--n_init",   type=int,   default=12)
parser.add_argument("--t_end",    type=float, default=60.0)
parser.add_argument("--n_points", type=int,   default=500)
parser.add_argument("--seed",     type=int,   default=42)
args = parser.parse_args()

# ── Setup ──────────────────────────────────────────────────────────────────────
rng         = np.random.default_rng(args.seed)
W           = read_graph(args.graph)
N           = W.shape[0]
SPIN_COLORS = plt.get_cmap("tab20")(np.linspace(0, 1, max(N, 2)))

print("=" * 65)
print("OIM MAX-CUT — Interactive Explorer")
print(f"  Graph    : {args.graph}  ({N} nodes)")
print(f"  mu range : [{args.mu_min}, {args.mu_max}]  ({args.n_mu} steps)")
print(f"  N_init   : {args.n_init}   t_end : {args.t_end}")
print("=" * 65)

# ── Thresholds via binarization_threshold() ───────────────────────────────────
print(f"\nComputing thresholds for all 2^{N} = {2**N} Type-I M2 equilibria...")
oim_ref  = OIMMaxCut(W, mu=1.0, seed=0)
bin_info = oim_ref.binarization_threshold()

mu_bin          = bin_info["mu_bin"]
Ks_bin          = bin_info["Ks_bin"]
thresholds_dict = bin_info["all_thresholds"]
easiest_eq      = bin_info["easiest_eq"]
hardest_eq      = bin_info["hardest_eq"]
W_total         = oim_ref.get_w_total()

eq_labels     = list(thresholds_dict.keys())
eq_thresholds = list(thresholds_dict.values())

sorted_idx   = np.argsort(eq_thresholds)
MAX_BARS     = min(32, len(eq_labels))
s_labels     = [eq_labels[i]     for i in sorted_idx][:MAX_BARS]
s_thresholds = [eq_thresholds[i] for i in sorted_idx][:MAX_BARS]
n_bars       = len(s_labels)

print(f"\n  {'Equilibrium phi*':<22} {'mu*_i':>8}  {'Ks*_i':>8}")
print(f"  {'-'*42}")
for lbl, thr in zip(s_labels, s_thresholds):
    print(f"  {lbl:<22} {thr:>8.4f}  {thr/2:>8.4f}")
print(f"\n  mu_bin = {mu_bin:.4f}  (global binarisation threshold, Remark 7)")
print(f"  Ks_bin = {Ks_bin:.4f}  (equivalent Ks)")
print(f"  W_total= {W_total:.4f}  (total edge weight — used in H = W_total - 2*cut)")

# ── Logger ─────────────────────────────────────────────────────────────────────
logger = MaxCutExperimentLogger()
logger.start_mu_sweep_experiment(args, Path(args.graph).stem, N)

# ── Pre-compute all trajectories ──────────────────────────────────────────────
MU_VALUES = np.linspace(args.mu_min, args.mu_max, args.n_mu)
MU_STEP   = float(MU_VALUES[1] - MU_VALUES[0])
T_SPAN    = (0., args.t_end)
phi0_list = [rng.uniform(-np.pi, np.pi, N) for _ in range(args.n_init)]

print(f"\nPre-computing {args.n_mu} x {args.n_init} trajectories — please wait...")
t0        = time.time()
all_trajs  = []    # list[list[sol]]
all_cuts   = []    # list[np.ndarray]   final binary cut per IC
all_H      = []    # list[np.ndarray]   final Hamiltonian per IC
all_binar  = []    # list[list[bool]]   binarised? per IC

for k_idx, mu in enumerate(MU_VALUES):
    oim  = OIMMaxCut(W, mu=mu, seed=args.seed)
    sols = oim.simulate_many(phi0_list, t_span=T_SPAN, n_points=args.n_points)
    cuts, Hs, bins = [], [], []
    for sol in sols:
        oim.theta = sol.y[:, -1]
        cuts.append(oim.get_binary_cut_value())
        Hs.append(oim.get_hamiltonian())
        bins.append(oim.is_binarized())
    all_trajs.append(sols)
    all_cuts.append(np.array(cuts))
    all_H.append(np.array(Hs))
    all_binar.append(bins)
    pct = 100 * (k_idx + 1) / args.n_mu
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    print(f"  [{bar}] {pct:5.1f}%  mu={mu:.3f}  "
          f"best cut={max(cuts):.1f}  min H={min(Hs):.2f}  "
          f"{'[BINARISED]' if mu > mu_bin else ''}",
          flush=True)

print(f"\nReady in {time.time()-t0:.1f}s — opening window...\n")

# ── Theme ──────────────────────────────────────────────────────────────────────
BG, PANEL, WHITE = "#111827", "#1e293b", "#f1f5f9"
ACCENT           = "#e94560"
PI_TICKS  = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
PI_LABELS = ["-pi", "-pi/2", "0", "pi/2", "pi"]

# ── Figure layout ──────────────────────────────────────────────────────────────
# Left : phase (full height)
# Right-top    : threshold bars  (top 52%)
# Right-bottom : Hamiltonian plot (bottom 36%)
# Bottom strip : slider
fig = plt.figure(figsize=(17, 8.5), facecolor=BG)
ax_phase  = fig.add_axes((0.04, 0.10, 0.44, 0.83))          # left column
ax_thresh = fig.add_axes((0.54, 0.50, 0.44, 0.43))          # right top
ax_hamil  = fig.add_axes((0.54, 0.10, 0.44, 0.34))          # right bottom
ax_slider = fig.add_axes((0.15, 0.025, 0.70, 0.030))        # slider

for ax in (ax_phase, ax_thresh, ax_hamil):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE)
    for sp in ax.spines.values():
        sp.set_edgecolor("#334155")

# ══════════════════════════════════════════════════════════════════════════════
# Static right-top: threshold bar chart
# ══════════════════════════════════════════════════════════════════════════════
init_mu = MU_VALUES[args.n_mu // 2]
y_max   = max(s_thresholds) * 1.30 + 0.3

ax_thresh.axhspan(0,      mu_bin, alpha=0.07, color="#ef5350", zorder=0)
ax_thresh.axhspan(mu_bin, y_max,  alpha=0.07, color="#4caf50", zorder=0)

bar_colors_init = ["#4caf50" if init_mu > v else "#ef5350" for v in s_thresholds]
bar_rects = ax_thresh.bar(range(n_bars), s_thresholds,
                           color=bar_colors_init, alpha=0.82, zorder=2)

ax_thresh.axhline(mu_bin, color=WHITE, linestyle=":", linewidth=2.0, zorder=4,
                  label=f"mu_bin = {mu_bin:.4f}  [Remark 7]")
hline_mu = ax_thresh.axhline(y=init_mu, color="#ffb74d", linestyle="--",
                               linewidth=2.2, zorder=5, label="Current  mu")

ax_thresh.text(n_bars * 0.97, mu_bin * 0.48,
               "NOT binarised\n(no eq. stable)",
               ha="right", va="center", fontsize=7.5, color="#ef9a9a",
               bbox=dict(boxstyle="round,pad=0.25", facecolor="#0f172a", alpha=0.80))
ax_thresh.text(n_bars * 0.97, mu_bin + (y_max - mu_bin) * 0.50,
               "BINARISED\n(≥1 eq. stable)",
               ha="right", va="center", fontsize=7.5, color="#a5d6a7",
               bbox=dict(boxstyle="round,pad=0.25", facecolor="#0f172a", alpha=0.80))

ax_thresh.set_xlim(-0.6, n_bars - 0.4)
ax_thresh.set_ylim(0, y_max)
ax_thresh.set_xticks(range(n_bars))
ax_thresh.set_xticklabels(s_labels, rotation=45, ha="right",
                           fontsize=max(5, 8 - N // 4), color=WHITE)
ax_thresh.set_ylabel("mu*_i = lambda_max(D)  [Th.2]",
                      color=WHITE, fontsize=9)
ax_thresh.set_title(
    f"Stability thresholds — mu_bin = {mu_bin:.4f}  [Remark 7]",
    color=WHITE, fontsize=9.5)
ax_thresh.tick_params(colors=WHITE)
ax_thresh.grid(True, alpha=0.15, color=WHITE, axis="y")

thresh_legend = [
    mpatches.Patch(facecolor="#4caf50", alpha=0.82, label="Stable at current mu"),
    mpatches.Patch(facecolor="#ef5350", alpha=0.82, label="Unstable at current mu"),
    mlines.Line2D([0],[0], color=WHITE,     linestyle=":",  lw=2,
               label=f"mu_bin={mu_bin:.4f}"),
    mlines.Line2D([0],[0], color="#ffb74d", linestyle="--", lw=2,
               label="Current mu"),
]
ax_thresh.legend(handles=thresh_legend, facecolor="#0f172a", labelcolor=WHITE,
                  fontsize=7.5, loc="upper left", framealpha=0.90, ncol=2)

status_txt = ax_thresh.text(
    0.98, 0.04, "", transform=ax_thresh.transAxes,
    ha="right", va="bottom", fontsize=8, color=WHITE,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a", alpha=0.88))

# ══════════════════════════════════════════════════════════════════════════════
# Static right-bottom: Hamiltonian bar chart (data filled dynamically)
# ══════════════════════════════════════════════════════════════════════════════
# Pre-compute H range for stable y-limits
all_H_flat = np.concatenate(all_H)
H_min_glob = all_H_flat.min() - 0.5
H_max_glob = all_H_flat.max() + 0.5
H_opt_line = W_total - 2 * max(max(c) for c in all_cuts)  # best H ever seen

init_Hs     = all_H[args.n_mu // 2]
ham_colors  = ["#4caf50" if b else "#ef5350"
               for b in all_binar[args.n_mu // 2]]
ham_bars    = ax_hamil.bar(range(args.n_init), init_Hs,
                            color=ham_colors, alpha=0.85)
hline_H_opt = ax_hamil.axhline(H_opt_line, color="#ffb74d", linestyle="--",
                                linewidth=1.5, label=f"Best H seen = {H_opt_line:.2f}")
hline_H_wt  = ax_hamil.axhline(W_total, color="#90caf9", linestyle=":",
                                linewidth=1.2, label=f"H_max (no cut) = {W_total:.2f}")
hline_H_wt2 = ax_hamil.axhline(-W_total, color="#a5d6a7", linestyle=":",
                                linewidth=1.2, label=f"H_min (perfect) = {-W_total:.2f}")

ax_hamil.set_xlim(-0.6, args.n_init - 0.4)
ax_hamil.set_ylim(H_min_glob - 0.5, H_max_glob + 0.5)
ax_hamil.set_xticks(range(args.n_init))
ax_hamil.set_xticklabels([f"IC {i}" for i in range(args.n_init)],
                          fontsize=7, color=WHITE, rotation=30, ha="right")
ax_hamil.set_ylabel("H = sum W_ij si sj", color=WHITE, fontsize=9)
ax_hamil.set_title(
    f"Final Hamiltonian per IC   (H = W_total - 2·cut,  lower H = better cut)",
    color=WHITE, fontsize=9.5)
ax_hamil.tick_params(colors=WHITE)
ax_hamil.grid(True, alpha=0.13, color=WHITE, axis="y")

ham_legend = [
    mpatches.Patch(facecolor="#4caf50", alpha=0.85, label="Binarised IC"),
    mpatches.Patch(facecolor="#ef5350", alpha=0.85, label="Not binarised IC"),
    mlines.Line2D([0],[0], color="#ffb74d", linestyle="--", lw=1.5,
               label=f"Best H seen = {H_opt_line:.2f}"),
    mlines.Line2D([0],[0], color="#90caf9", linestyle=":",  lw=1.2,
               label=f"H_max (no cut) = {W_total:.2f}"),
]
ax_hamil.legend(handles=ham_legend, facecolor="#0f172a", labelcolor=WHITE,
                 fontsize=7.5, loc="upper right", framealpha=0.90, ncol=2)

hamil_stat = ax_hamil.text(
    0.01, 0.95, "", transform=ax_hamil.transAxes,
    ha="left", va="top", fontsize=8, color=WHITE,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f172a", alpha=0.85))

# ── Slider ─────────────────────────────────────────────────────────────────────
slider = Slider(ax_slider, "  mu  ", args.mu_min, args.mu_max,
                valinit=init_mu, valstep=MU_STEP,
                color=ACCENT, track_color="#0f172a")
ax_slider.set_facecolor("#0f172a")
slider.label.set_color(WHITE)
slider.valtext.set_color(WHITE)

# ══════════════════════════════════════════════════════════════════════════════
# Left panel: phase evolution (redrawn on each slider move)
# ══════════════════════════════════════════════════════════════════════════════
def draw_phase(idx):
    ax_phase.clear()
    ax_phase.set_facecolor(PANEL)
    mu   = MU_VALUES[idx]
    sols = all_trajs[idx]
    cuts = all_cuts[idx]
    t    = sols[0].t

    for sol in sols:
        for spin in range(N):
            ax_phase.plot(t, sol.y[spin], color=SPIN_COLORS[spin % 20],
                          alpha=0.22, linewidth=0.65)

    ax_phase.axhline( np.pi, color="#ef9a9a", linestyle="--",
                      linewidth=1.2, alpha=0.85, label="pi")
    ax_phase.axhline(0,      color="#90caf9", linestyle="--",
                      linewidth=1.2, alpha=0.85, label="0")
    ax_phase.axhline(-np.pi, color="#ef9a9a", linestyle="--",
                      linewidth=1.2, alpha=0.50)

    binarized = mu > mu_bin
    ax_phase.text(0.97, 0.96,
        "BINARISED \u2713" if binarized else "NOT binarised \u2717",
        transform=ax_phase.transAxes, ha="right", va="top",
        fontsize=10, fontweight="bold",
        color="#4caf50" if binarized else "#ef5350",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a", alpha=0.90))

    ax_phase.text(0.97, 0.04,
        f"best cut = {max(cuts):.1f}\n"
        f"mean cut = {np.mean(cuts):.2f} \u00b1 {np.std(cuts):.2f}\n"
        f"mu = {mu:.4f}\n"
        f"mu_bin = {mu_bin:.4f}",
        transform=ax_phase.transAxes, ha="right", va="bottom",
        fontsize=8, color=WHITE,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a", alpha=0.85))

    ax_phase.set_xlim(t[0], t[-1])
    ax_phase.set_ylim(-4.5, 4.5)
    ax_phase.set_yticks(PI_TICKS)
    ax_phase.set_yticklabels(PI_LABELS, color=WHITE, fontsize=9)
    xt = np.linspace(t[0], t[-1], 5)
    ax_phase.set_xticks(xt)
    ax_phase.set_xticklabels([f"{v:.0f}" for v in xt], color=WHITE, fontsize=9)
    ax_phase.set_xlabel("time  t", color=WHITE, fontsize=11)
    ax_phase.set_ylabel("phase  theta  (rad)", color=WHITE, fontsize=11)
    ax_phase.set_title(
        f"Phase evolution — mu = {mu:.4f}  "
        f"({'above' if binarized else 'below'} mu_bin = {mu_bin:.4f})",
        color=WHITE, fontsize=11)
    ax_phase.tick_params(colors=WHITE)
    ax_phase.grid(True, alpha=0.14, color=WHITE, linewidth=0.5)
    for sp in ax_phase.spines.values():
        sp.set_edgecolor("#334155")
    spin_patches = [mpatches.Patch(color=SPIN_COLORS[s % 20], label=f"spin {s}")
                    for s in range(N)]
    ax_phase.legend(handles=spin_patches, loc="lower left", fontsize=7,
                    facecolor="#0f172a", labelcolor=WHITE,
                    framealpha=0.85, ncol=max(1, N // 5))


def save_state(idx):
    mu   = float(MU_VALUES[idx])
    cuts = all_cuts[idx]
    Hs   = all_H[idx]
    logger.save_interactive_state({
        "graph_file"       : args.graph,
        "n_nodes"          : N,
        "mu"               : mu,
        "Ks_equiv"         : mu / 2.0,
        "mu_bin"           : float(mu_bin),
        "Ks_bin"           : float(Ks_bin),
        "W_total"          : float(W_total),
        "binarised"        : bool(mu > mu_bin),
        "easiest_eq"       : easiest_eq,
        "hardest_eq"       : hardest_eq,
        "seed"             : args.seed,
        "n_init"           : args.n_init,
        "t_span"           : [0.0, args.t_end],
        "n_points"         : args.n_points,
        "best_cut"         : float(max(cuts)),
        "mean_cut"         : float(np.mean(cuts)),
        "std_cut"          : float(np.std(cuts)),
        "min_H"            : float(min(Hs)),
        "mean_H"           : float(np.mean(Hs)),
        "H_per_IC"         : Hs.tolist(),
        "mu_range"         : [float(args.mu_min), float(args.mu_max)],
        "per_eq_thresholds": thresholds_dict,
    })


def update(val):
    idx = int(np.argmin(np.abs(MU_VALUES - slider.val)))
    mu  = MU_VALUES[idx]
    cuts = all_cuts[idx]
    Hs   = all_H[idx]
    bins = all_binar[idx]

    # ── Redraw phase panel ────────────────────────────────────────────
    draw_phase(idx)

    # ── Update threshold bars ─────────────────────────────────────────
    hline_mu.set_ydata([mu, mu])
    for rect, thr in zip(bar_rects, s_thresholds):
        rect.set_facecolor("#4caf50" if mu > thr else "#ef5350")

    n_stable  = sum(1 for thr in s_thresholds if mu > thr)
    binarized = mu > mu_bin
    status_txt.set_text(
        f"mu = {mu:.4f}\n"
        f"mu_bin = {mu_bin:.4f}\n"
        f"{'BINARISED \u2713' if binarized else 'not binarised \u2717'}\n"
        f"Stable eq: {n_stable} / {n_bars}")
    status_txt.set_color("#4caf50" if binarized else "#ef5350")

    # ── Update Hamiltonian bars ───────────────────────────────────────
    for bar, h, b in zip(ham_bars, Hs, bins):
        bar.set_height(h)
        bar.set_facecolor("#4caf50" if b else "#ef5350")

    H_best = float(min(Hs))
    hamil_stat.set_text(
        f"min H = {H_best:.2f}  (best cut = {(W_total - H_best)/2:.1f})\n"
        f"mean H = {np.mean(Hs):.2f}  W_total = {W_total:.2f}\n"
        f"H = W_total - 2\u00b7cut  \u2192  lower H = better cut")

    save_state(idx)
    fig.canvas.draw_idle()


slider.on_changed(update)
fig.suptitle(
    f"OIM Max-Cut — Interactive Explorer   {args.graph}   (mu parametrisation)",
    color=WHITE, fontsize=13, fontweight="bold", y=1.002)

update(init_mu)
plt.show()
