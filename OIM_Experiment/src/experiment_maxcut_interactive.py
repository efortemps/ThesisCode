#!/usr/bin/env python3
"""
experiment_maxcut_interactive_v2.py
------------------------------------
Interactive Max-Cut explorer using OIMMaxCut (OIM_mu_v2.py).

Layout
------
  LEFT-TOP     : Phase evolution theta_i(t) for all spins
  LEFT-BOTTOM  : Stability threshold bar chart (Theorem 2 / Remark 7)
  RIGHT        : Equilibrium landscape — cut & H per equilibrium vs mu*
                   X  = stability threshold mu*_i  (= lambda_max(D(phi*_i)))
                   Y  = cut value at that equilibrium
                   secondary Y-right = H value
                   circle size ∝ number of equilibria at that (mu*, cut) cluster
                   green  = stable at current mu   (mu > mu*_i)
                   red    = unstable at current mu (mu < mu*_i)
                   orange vertical line = current mu (sweeps with slider)
  SLIDER       : Sweeps mu in [mu_min, mu_max].

Usage
-----
    python experiment_maxcut_interactive_v2.py [--graph graph.txt] [options]
"""

import ast
import argparse
from pathlib import Path
from collections import defaultdict
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines  as mlines
from matplotlib.widgets import Slider

# ── TikZ / PGFPlots global style ───────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 10,
    "axes.edgecolor"   : "black",
    "axes.linewidth"   : 0.8,
    "xtick.color"      : "black",
    "ytick.color"      : "black",
    "text.color"       : "black",
    "figure.facecolor" : "white",
    "axes.facecolor"   : "white",
})

from OIM_Experiment.src.OIM_mu          import OIMMaxCut
from OIM_Experiment.src.graph_utils        import read_graph
from OIM_Experiment.src.experiment_logger import MaxCutExperimentLogger

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--graph",    nargs="?", default="3node.txt")
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

# ── Thresholds + equilibrium H/cut ────────────────────────────────────────────
print(f"\nComputing thresholds for all 2^{N} = {2**N} equilibria...")
oim_ref   = OIMMaxCut(W, mu=1.0, seed=0)
bin_info  = oim_ref.binarization_threshold()

mu_bin          = bin_info["mu_bin"]
Ks_bin          = bin_info["Ks_bin"]
thresholds_dict = bin_info["all_thresholds"]
easiest_eq      = bin_info["easiest_eq"]
hardest_eq      = bin_info["hardest_eq"]
W_total         = oim_ref.get_w_total()

eq_labels     = list(thresholds_dict.keys())
eq_thresholds = list(thresholds_dict.values())

sorted_idx   = np.argsort(eq_thresholds)

# At most 20 bars to avoid overcrowding — if more, we keep the ones with lowest thresholds (most likely to become stable)
MAX_BARS     = min(2**(10), len(eq_labels))
###
s_labels     = [eq_labels[i]     for i in sorted_idx][:MAX_BARS]
s_thresholds = [eq_thresholds[i] for i in sorted_idx][:MAX_BARS]
n_bars       = len(s_labels)

# Compute H and cut at each equilibrium (static — independent of mu)
print(f"\n  {'Equilibrium phi*':<28} {'mu*_i':>8}  {'H':>7}  {'cut':>7}")
print(f"  {'-'*56}")

s_H_values   = []
s_cut_values = []
for lbl in s_labels:
    bits          = ast.literal_eval(lbl)
    phi_star      = np.array([b * np.pi for b in bits])
    oim_ref.theta = phi_star
    h_val  = oim_ref.get_hamiltonian()
    c_val  = oim_ref.get_binary_cut_value()
    s_H_values.append(h_val)
    s_cut_values.append(c_val)
    thr = thresholds_dict[lbl]
    print(f"  {lbl:<28} {thr:>8.4f}  {h_val:>7.3f}  {c_val:>7.3f}")

s_H_values   = np.array(s_H_values)
s_cut_values = np.array(s_cut_values)

# Aggregate: cluster equilibria that share the same (mu*, cut) point
# key=(mu*_i rounded to 4dp, cut) -> count, H, list of labels
cluster_map = defaultdict(lambda: {"count": 0, "H": None, "cut": None,
                                    "mu_star": None, "labels": []})
for lbl, thr, h, c in zip(s_labels, s_thresholds, s_H_values, s_cut_values):
    key = (round(thr, 4), round(c, 4))
    cluster_map[key]["count"]   += 1
    cluster_map[key]["H"]        = h
    cluster_map[key]["cut"]      = c
    cluster_map[key]["mu_star"]  = thr
    cluster_map[key]["labels"].append(lbl)

clusters = list(cluster_map.values())
cl_mu    = np.array([cl["mu_star"] for cl in clusters])
cl_cut   = np.array([cl["cut"]     for cl in clusters])
cl_H     = np.array([cl["H"]       for cl in clusters])
cl_count = np.array([cl["count"]   for cl in clusters])

print(f"\n  Unique (mu*, cut) clusters: {len(clusters)}")
print(f"  mu_bin   = {mu_bin:.4f}")
print(f"  W_total  = {W_total:.4f}")

# ── Logger ─────────────────────────────────────────────────────────────────────
logger = MaxCutExperimentLogger()
logger.start_mu_sweep_experiment(args, Path(args.graph).stem, N)

# ── Pre-compute all trajectories ──────────────────────────────────────────────
MU_VALUES = np.linspace(args.mu_min, args.mu_max, args.n_mu)
MU_STEP   = float(MU_VALUES[1] - MU_VALUES[0])
T_SPAN    = (0., args.t_end)
phi0_list = [rng.uniform(-np.pi, np.pi, N) for _ in range(args.n_init)]

print(f"\nPre-computing {args.n_mu} x {args.n_init} trajectories — please wait...")
t0 = time.time()
all_trajs, all_cuts, all_H, all_binar = [], [], [], []

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
    print(f"  [{bar}] {pct:5.1f}%  mu={mu:.3f}  best cut={max(cuts):.1f}  "
          f"min H={min(Hs):.2f}  {'[BINARISED]' if mu > mu_bin else ''}",
          flush=True)

print(f"\nReady in {time.time()-t0:.1f}s — opening window...\n")

# simulated best cut/H per mu (for info overlay)
sim_best_cut = np.array([float(np.max(c)) for c in all_cuts])
sim_min_H    = np.array([float(np.min(h)) for h in all_H])

# ── TikZ-like Theme ────────────────────────────────────────────────────────────
WHITE  = "#ffffff"
BLACK  = "#000000"
GRAY   = "#b0b0b0"
LIGHT  = "#e6e6e6"

BG     = BLACK          # text / axes
PANEL  = WHITE          # axes background
ACCENT = "#1f77b4"      # subtle blue (TikZ-like default)
PI_TICKS  = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
PI_LABELS = ["-pi", "-pi/2", "0", "pi/2", "pi"]

# ── Figure layout ──────────────────────────────────────────────────────────────
#   LEFT-TOP    : phase      [0.03, 0.43, 0.40, 0.50]
#   LEFT-BOTTOM : thresholds [0.03, 0.10, 0.40, 0.28]
#   RIGHT       : landscape  [0.48, 0.10, 0.50, 0.83]
#   SLIDER      : [0.15, 0.025, 0.70, 0.030]
fig       = plt.figure(figsize=(18, 8.5), facecolor=WHITE)
ax_phase  = fig.add_axes((0.03, 0.43, 0.40, 0.50))
ax_thresh = fig.add_axes((0.03, 0.10, 0.40, 0.28))
ax_land   = fig.add_axes((0.48, 0.10, 0.49, 0.83))
ax_slider = fig.add_axes((0.15, 0.025, 0.70, 0.030))

for ax in (ax_phase, ax_thresh, ax_land):
    ax.set_facecolor(WHITE)
    ax.tick_params(colors=BLACK, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(BLACK)
        sp.set_linewidth(0.8)

    # TikZ-like subtle grid
    ax.grid(True, color=LIGHT, linewidth=0.6, alpha=1.0)

# ══════════════════════════════════════════════════════════════════════════════
# LEFT-BOTTOM: stability threshold bar chart
# ══════════════════════════════════════════════════════════════════════════════
init_mu = MU_VALUES[args.n_mu // 2]
y_max   = max(s_thresholds) * 1.30 + 0.3

ax_thresh.axhspan(0,      mu_bin, alpha=0.07, color="#DD8452", zorder=0)
ax_thresh.axhspan(mu_bin, y_max,  alpha=0.07, color="#4C72B0", zorder=0)

bar_colors_init = ["#4C72B0" if init_mu > v else "#DD8452" for v in s_thresholds]
bar_rects = ax_thresh.bar(range(n_bars), s_thresholds,
                           color=bar_colors_init, alpha=0.82, zorder=2)

ax_thresh.axhline(mu_bin, color=BLACK, linestyle=":", linewidth=2.0, zorder=4)
hline_mu_thresh = ax_thresh.axhline(y=init_mu, color="#ffb74d",
                                     linestyle="--", linewidth=2.0, zorder=5)

ax_thresh.text(n_bars * 0.97, mu_bin * 0.48,
               "NOT binarised", ha="right", va="center",
               fontsize=7, color="#DD8452",
               bbox=dict(boxstyle="round,pad=0.2", facecolor=WHITE, edgecolor=GRAY, alpha=1.0))
ax_thresh.text(n_bars * 0.97, mu_bin + (y_max - mu_bin) * 0.55,
               "BINARISED", ha="right", va="center",
               fontsize=7, color="#4C72B0",
               bbox=dict(boxstyle="round,pad=0.2", facecolor=WHITE, edgecolor=GRAY, alpha=1.0))

ax_thresh.set_xlim(-0.6, n_bars - 0.4)
ax_thresh.set_ylim(0, y_max)
ax_thresh.set_xticks(range(n_bars))
ax_thresh.set_xticklabels(s_labels, rotation=60, ha="right",
                           fontsize=max(4, 7 - N // 4), color=BG)
ax_thresh.set_ylabel("mu*_i  [Theorem 2]", color=BLACK, fontsize=8)
ax_thresh.set_title(f"Stability thresholds   mu_bin = {mu_bin:.4f}  [Remark 7]",
                     color=BG, fontsize=9)
ax_thresh.tick_params(colors=BG)
ax_thresh.grid(True, color=LIGHT, axis="y")
ax_thresh.legend(handles=[
    mpatches.Patch(facecolor="#4C72B0", alpha=0.82, label="Stable"),
    mpatches.Patch(facecolor="#DD8452", alpha=0.82, label="Unstable"),
    mlines.Line2D([0],[0], color=BLACK,     ls=":", lw=2, label=f"mu_bin"),
    mlines.Line2D([0],[0], color="#ffb74d", ls="--",lw=2, label="Current mu"),
], facecolor=WHITE, edgecolor=GRAY, labelcolor=BLACK, fontsize=7,
   loc="upper left", framealpha=0.90, ncol=2)

thresh_stat = ax_thresh.text(
    0.98, 0.06, "", transform=ax_thresh.transAxes,
    ha="right", va="bottom", fontsize=7.5, color=BG,
    bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE, edgecolor=GRAY, alpha=1.0))

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT: Equilibrium Landscape  (cut & H vs mu*)
# ══════════════════════════════════════════════════════════════════════════════
#
#  X axis : mu*_i — the stability threshold for each equilibrium
#           Points to the LEFT of the current-mu line are stable (green)
#           Points to the RIGHT are unstable (red)
#
#  Y axis : cut value at that equilibrium
#           (secondary right Y axis: corresponding H value)
#
#  Circle size : proportional to the NUMBER of equilibria in that cluster
#
#  Orange vertical line : current mu (moves with slider)
#
# Reference lines: mu_bin (white dotted), best possible cut (blue dotted)
# ─────────────────────────────────────────────────────────────────────────────

# -- Build initial scatter colors based on init_mu ----------------------------
def cluster_colors(mu_val):
    return ["#4C72B0" if mu_val > ms else "#DD8452" for ms in cl_mu]

init_colors  = cluster_colors(init_mu)
# Circle area: scale so smallest cluster is visible, largest not overwhelming
MIN_SIZE, MAX_SIZE = 120, 900
sizes = MIN_SIZE + (cl_count - cl_count.min()) / max(cl_count.max() - cl_count.min(), 1) \
        * (MAX_SIZE - MIN_SIZE)

scatter_dots = ax_land.scatter(
    cl_mu, cl_cut,
    s=sizes, c=init_colors, alpha=0.85,
    zorder=4, edgecolors=GRAY, linewidths=0.5)

# Annotate each cluster: show H value + count
for i, cl in enumerate(clusters):
    ax_land.annotate(
        f"H={cl['H']:.1f}\n(×{cl['count']})",
        xy=(cl["mu_star"], cl["cut"]),
        xytext=(6, 6), textcoords="offset points",
        fontsize=7, color=BG,
        bbox=dict(boxstyle="round,pad=0.2", facecolor=WHITE, edgecolor=GRAY, alpha=1.0))

# Static reference lines
ax_land.axhline(max(cl_cut), color=GRAY, linestyle=":", linewidth=1.2,
                alpha=0.7, label=f"Best cut = {max(cl_cut):.0f}")
ax_land.axvline(mu_bin, color=BLACK, linestyle=":", linewidth=1.8,
                alpha=0.8, label=f"mu_bin = {mu_bin:.4f}")

# Dynamic: current-mu vertical line
vline_land = ax_land.axvline(x=init_mu, color="#ffb74d", linestyle="--",
                              linewidth=2.2, zorder=6, label="Current mu")

# Shaded background: left of mu_bin = red zone, right = green zone
ax_land.axvspan(args.mu_min, mu_bin, alpha=0.05, color="#DD8452", zorder=0)
ax_land.axvspan(mu_bin, args.mu_max * 1.05, alpha=0.05, color="#4C72B0", zorder=0)

# Axes
mu_pad = (args.mu_max - args.mu_min) * 0.08
ax_land.set_xlim(args.mu_min - mu_pad, args.mu_max + mu_pad)
cut_pad = (max(cl_cut) - min(cl_cut)) * 0.25 + 0.5
ax_land.set_ylim(min(cl_cut) - cut_pad, max(cl_cut) + cut_pad)
ax_land.set_xlabel("mu*_i  (stability threshold per equilibrium)",
                    color=BG, fontsize=10)
ax_land.set_ylabel("Cut value at equilibrium phi*", color=BG, fontsize=10)
ax_land.set_title(
    "Equilibrium Landscape   cut & H  vs  stability threshold mu*\n"
    "Left of orange line = stable at current mu   |   circle size = # equiv. equilibria",
    color=BG, fontsize=10)
ax_land.tick_params(colors=BG)
ax_land.grid(True, color=LIGHT, linewidth=0.6)

# Secondary right Y axis: H = W_total - 2*cut
ax_land2 = ax_land.twinx()
ax_land2.set_facecolor(PANEL)
ax_land2.tick_params(colors=BLACK, labelsize=8)
for sp in ax_land2.spines.values():
    sp.set_edgecolor(BLACK)
    sp.set_linewidth(0.8)
y1, y2 = ax_land.get_ylim()
ax_land2.set_ylim(W_total - 2*y1, W_total - 2*y2)  # H = W_total - 2*cut
ax_land2.set_ylabel("H = W_total − 2 · cut", color=BLACK, fontsize=9)

ax_land.legend(handles=[
    mpatches.Patch(facecolor="#4C72B0", alpha=0.85,
                   label="Equilibrium  stable at current mu"),
    mpatches.Patch(facecolor="#DD8452", alpha=0.85,
                   label="Equilibrium  unstable at current mu"),
    mlines.Line2D([0],[0], color="#ffb74d", ls="--", lw=2.2,
                  label="Current mu"),
    mlines.Line2D([0],[0], color=BLACK, ls=":", lw=1.8,
                  label=f"mu_bin = {mu_bin:.4f}"),
    mlines.Line2D([0],[0], color=GRAY, ls=":", lw=1.2,
                  label=f"Best cut = {max(cl_cut):.0f}"),
    mlines.Line2D([0],[0], color="none", marker="o", markersize=8,
                  markerfacecolor=BLACK, markeredgecolor=GRAY,
                  label="Size ∝ # equiv. equilibria"),
], facecolor=WHITE, edgecolor=GRAY, labelcolor=BLACK, fontsize=8,
   loc="lower right", framealpha=0.92, ncol=1)

land_stat = ax_land.text(
    0.02, 0.97, "", transform=ax_land.transAxes,
    ha="left", va="top", fontsize=9, color=BG,
    bbox=dict(boxstyle="round,pad=0.4", facecolor=WHITE, edgecolor=GRAY, alpha=1.0))

# ── Slider ─────────────────────────────────────────────────────────────────────
slider = Slider(ax_slider, "  mu  ", args.mu_min, args.mu_max,
                valinit=init_mu, valstep=MU_STEP,
                color=ACCENT, track_color=LIGHT)
ax_slider.set_facecolor(WHITE)
slider.label.set_color(BLACK)
slider.valtext.set_color(BLACK)

# ══════════════════════════════════════════════════════════════════════════════
# LEFT-TOP: phase evolution (redrawn on each slider move)
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

    ax_phase.axhline( np.pi, color=GRAY, linestyle="--",
                      linewidth=1.1, alpha=0.85)
    ax_phase.axhline(0,      color=GRAY, linestyle="--",
                      linewidth=1.1, alpha=0.85)
    ax_phase.axhline(-np.pi, color=GRAY, linestyle="--",
                      linewidth=1.1, alpha=0.50)

    binarized = mu > mu_bin
    ax_phase.text(0.97, 0.96,
        "BINARISED \u2713" if binarized else "NOT binarised \u2717",
        transform=ax_phase.transAxes, ha="right", va="top",
        fontsize=9, fontweight="bold",
        color="#4C72B0" if binarized else "#DD8452",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE, edgecolor=GRAY, alpha=1.0))

    ax_phase.text(0.97, 0.04,
        f"best cut = {max(cuts):.1f}    mu = {mu:.4f}\n"
        f"mean cut = {np.mean(cuts):.2f} \u00b1 {np.std(cuts):.2f}    "
        f"mu_bin = {mu_bin:.4f}",
        transform=ax_phase.transAxes, ha="right", va="bottom",
        fontsize=7.5, color=BLACK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", alpha=0.85))

    ax_phase.set_xlim(t[0], t[-1])
    ax_phase.set_ylim(-4.5, 4.5)
    ax_phase.set_yticks(PI_TICKS)
    ax_phase.set_yticklabels(PI_LABELS, color=BLACK, fontsize=8)
    xt = np.linspace(t[0], t[-1], 5)
    ax_phase.set_xticks(xt)
    ax_phase.set_xticklabels([f"{v:.0f}" for v in xt], color=BLACK, fontsize=8)
    ax_phase.set_xlabel("time  t", color=BLACK, fontsize=10)
    ax_phase.set_ylabel("phase  theta  (rad)", color=BLACK, fontsize=10)
    ax_phase.set_title(
        f"Phase evolution — mu = {mu:.4f}  "
        f"({'above' if binarized else 'below'} mu_bin = {mu_bin:.4f})",
        color=BLACK, fontsize=10)
    ax_phase.tick_params(colors=BLACK)
    ax_phase.grid(True, color=LIGHT, linewidth=0.6)
    for sp in ax_phase.spines.values():
        sp.set_edgecolor(BLACK)
    spin_patches = [mpatches.Patch(color=SPIN_COLORS[s % 20], label=f"spin {s}")
                    for s in range(N)]
    ax_phase.legend(handles=spin_patches, loc="lower left", fontsize=6.5,
                    facecolor=WHITE, edgecolor=GRAY, labelcolor=BLACK,
                    framealpha=0.85, ncol=max(1, N // 5))


def save_state(idx):
    mu   = float(MU_VALUES[idx])
    cuts = all_cuts[idx]
    Hs   = all_H[idx]
    logger.save_interactive_state({
        "graph_file"        : args.graph,
        "n_nodes"           : N,
        "mu"                : mu,
        "Ks_equiv"          : mu / 2.0,
        "mu_bin"            : float(mu_bin),
        "Ks_bin"            : float(Ks_bin),
        "W_total"           : float(W_total),
        "binarised"         : bool(mu > mu_bin),
        "easiest_eq"        : easiest_eq,
        "hardest_eq"        : hardest_eq,
        "seed"              : args.seed,
        "n_init"            : args.n_init,
        "best_cut"          : float(max(cuts)),
        "sim_min_H"         : float(min(Hs)),
        "eq_H_values"       : s_H_values.tolist(),
        "eq_cut_values"     : s_cut_values.tolist(),
        "per_eq_thresholds" : thresholds_dict,
    })


def update(val):
    idx       = int(np.argmin(np.abs(MU_VALUES - slider.val)))
    mu        = MU_VALUES[idx]
    cuts      = all_cuts[idx]
    Hs        = all_H[idx]
    binarized = mu > mu_bin

    # ── Phase panel ───────────────────────────────────────────────────
    draw_phase(idx)

    # ── Threshold bars ────────────────────────────────────────────────
    hline_mu_thresh.set_ydata([mu, mu])
    for rect, thr in zip(bar_rects, s_thresholds):
        rect.set_facecolor("#4C72B0" if mu > thr else "#DD8452")
    n_stable = sum(1 for thr in s_thresholds if mu > thr)
    thresh_stat.set_text(
        f"mu = {mu:.4f}   {'BINARISED \u2713' if binarized else 'not binarised \u2717'}\n"
        f"Stable eq: {n_stable} / {n_bars}")
    thresh_stat.set_color("#4C72B0" if binarized else "#DD8452")

    # ── Landscape: update scatter colors + vertical line ─────────────
    new_colors = cluster_colors(mu)
    scatter_dots.set_facecolor(new_colors)

    vline_land.set_xdata([mu, mu])

    # Build info box
    stable_clusters = [(cl["cut"], cl["H"], cl["count"])
                       for cl, ms in zip(clusters, cl_mu) if mu > ms]
    if stable_clusters:
        best_cut_now = max(c for c, h, _ in stable_clusters)
        best_H_now   = min(h for c, h, _ in stable_clusters)
        n_eq_stable  = sum(cnt for c, h, cnt in stable_clusters)
        info = (f"mu = {mu:.4f}   |   {n_stable} cluster(s) stable   "
                f"({n_eq_stable} equilibria)\n"
                f"Best reachable cut  = {best_cut_now:.1f}   "
                f"H = {best_H_now:.3f}\n"
                f"Simulated best cut  = {max(cuts):.1f}   "
                f"H = {min(Hs):.3f}\n"
                f"H = W_total \u2212 2\u00b7cut   (W_total = {W_total:.2f})")
    else:
        info = (f"mu = {mu:.4f}   |   NOT binarised\n"
                f"No equilibrium stable yet   (mu_bin = {mu_bin:.4f})\n"
                f"Simulated best cut  = {max(cuts):.1f}   "
                f"H = {min(Hs):.3f}")
    land_stat.set_text(info)

    save_state(idx)
    fig.canvas.draw_idle()


slider.on_changed(update)
fig.suptitle(
    f"OIM Max-Cut — Interactive Explorer   |   {args.graph}   "
    f"|   mu parametrisation",
    color=BLACK, fontsize=13, fontweight="bold", y=1.002)

update(init_mu)
plt.show()
