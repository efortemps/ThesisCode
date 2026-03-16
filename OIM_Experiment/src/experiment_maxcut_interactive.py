#!/usr/bin/env python3
"""
experiment_maxcut_interactive.py
---------------------------------
Interactive Max-Cut explorer using OIMMaxCut (OIM_mu_v2.py).

Layout
------
  LEFT  : Phase evolution theta_i(t) for all N spins from N_INIT random ICs.
           Redraws instantly from cache when the slider moves.
  RIGHT : Per-equilibrium stability threshold bar chart with a horizontal
           line showing the current mu value — analogous to the threshold
           diagram in experiment1_interactive.py.
  SLIDER: Sweeps mu in [MU_MIN, MU_MAX] in N_MU snapped steps.

JSON save
---------
  Each time the slider is released, the current experiment state is written
  to  experiment_maxcut_interactive_state.json  in the working directory.

Usage
-----
    python experiment_maxcut_interactive.py [graph.txt] [options]

    --mu_min   float  (default 0.2)
    --mu_max   float  (default 5.0)
    --n_mu     int    (default 20)
    --n_init   int    (default 12)
    --t_end    float  (default 60.0)
    --n_points int    (default 500)
    --seed     int    (default 42)

Requires OIM_mu_v2.py and graph_utils.py in the same directory.
"""

import argparse
from pathlib import Path
import json
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
from itertools import product as iproduct

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
rng = np.random.default_rng(args.seed)
W   = read_graph(args.graph)
N   = W.shape[0]
SPIN_COLORS = plt.get_cmap("tab20")(np.linspace(0, 1, N))

print("=" * 65)
print(f"OIM MAX-CUT — Interactive Explorer")
print(f"  Graph : {args.graph}  ({N} nodes)")
print(f"  mu range : [{args.mu_min}, {args.mu_max}]  ({args.n_mu} steps)")
print(f"  N_init   : {args.n_init}   t_end : {args.t_end}")
print("=" * 65)

# ── Binarisation thresholds per equilibrium (fixed) ───────────────────────────
print(f"\nComputing thresholds for all 2^{N} = {2**N} Type-I equilibria...")
oim_ref = OIMMaxCut(W, mu=1.0, seed=0)
eq_bits_list = list(iproduct([0, 1], repeat=N))
eq_labels    = [str(list(b)) for b in eq_bits_list]
eq_thresholds= [float(oim_ref.stability_threshold(
                    np.array([b * np.pi for b in bits])))
                for bits in eq_bits_list]
mu_star = min(eq_thresholds)
print(f"  mu* = {mu_star:.6f}")

# Sort by threshold for the bar chart
sorted_idx  = np.argsort(eq_thresholds)
s_labels    = [eq_labels[i]    for i in sorted_idx]
s_thresholds= [eq_thresholds[i] for i in sorted_idx]
# cap display at 32 bars (avoid overcrowding for large graphs)
MAX_BARS = min(32, len(s_labels))
s_labels     = s_labels[:MAX_BARS]
s_thresholds = s_thresholds[:MAX_BARS]

# ── Pre-compute trajectories for each mu step ─────────────────────────────────
MU_VALUES = np.linspace(args.mu_min, args.mu_max, args.n_mu)
MU_STEP   = float(MU_VALUES[1] - MU_VALUES[0])
T_SPAN    = (0., args.t_end)

# ── Logger: initialise sweep experiment ──────────────────────────────────────
logger   = MaxCutExperimentLogger()
logger.start_mu_sweep_experiment(args, Path(args.graph).stem, N)

phi0_list = [rng.uniform(-np.pi, np.pi, N) for _ in range(args.n_init)]

print(f"\nPre-computing {args.n_mu} x {args.n_init} trajectories — please wait...")
t0 = time.time()

all_trajs  = []   # list[list[sol]] — indexed by int
all_cuts   = []   # list[np.ndarray] of binary cut values

for k_idx, mu in enumerate(MU_VALUES):
    oim  = OIMMaxCut(W, mu=mu, seed=args.seed)
    sols = oim.simulate_many(phi0_list, t_span=T_SPAN, n_points=args.n_points)
    cuts = []
    for sol in sols:
        oim.theta = sol.y[:, -1]
        cuts.append(oim.get_binary_cut_value())
    all_trajs.append(sols)
    all_cuts.append(np.array(cuts))
    pct = 100 * (k_idx + 1) / args.n_mu
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    print(f"  [{bar}] {pct:5.1f}%  mu={mu:.3f}  "
          f"best cut={max(cuts):.1f}  "
          f"{'[BINARISED]' if mu > mu_star else ''}",
          flush=True)

print(f"\nReady in {time.time()-t0:.1f}s — opening window...\n")

# ── Theme ──────────────────────────────────────────────────────────────────────
BG, PANEL, WHITE = "#111827", "#1e293b", "#f1f5f9"
ACCENT = "#e94560"

# ── Figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 7.5), facecolor=BG)
ax_phase  = fig.add_axes((0.04, 0.13, 0.44, 0.79))   # left : phase evolution
ax_thresh = fig.add_axes((0.54, 0.13, 0.44, 0.79))   # right: threshold bars
ax_slider = fig.add_axes((0.15, 0.03, 0.70, 0.035))  # bottom: slider

for ax in (ax_phase, ax_thresh):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE)
    for sp in ax.spines.values():
        sp.set_edgecolor("#334155")

# ── Static right panel: stability threshold bar chart ─────────────────────────
bar_colors_init = ["#4caf50" if (MU_VALUES[args.n_mu // 2] > v) else "#ef5350"
                   for v in s_thresholds]
bar_rects = ax_thresh.bar(range(len(s_thresholds)), s_thresholds,
                           color=bar_colors_init, alpha=0.80)

# Global threshold line (horizontal at mu_star)
ax_thresh.axhline(mu_star, color=WHITE, linestyle=":", linewidth=1.5,
                  label=f"mu* = {mu_star:.4f}")

# Dynamic current-mu line
hline_mu = ax_thresh.axhline(y=MU_VALUES[args.n_mu // 2], color=ACCENT,
                               linestyle="--", linewidth=2.2, zorder=10,
                               label="Current mu")

ax_thresh.set_xlim(-0.8, len(s_thresholds) - 0.2)
ax_thresh.set_ylim(0, max(s_thresholds) * 1.25 + 0.5)
ax_thresh.set_xticks(range(len(s_labels)))
ax_thresh.set_xticklabels(s_labels, rotation=45, ha="right",
                            fontsize=max(5, 8 - N // 4), color=WHITE)
ax_thresh.set_ylabel("mu*_i  =  lambda_max(D(phi*_i))", color=WHITE, fontsize=10)
ax_thresh.set_title(
    f"Per-equilibrium stability thresholds — {args.graph}  (Theorem 2)",
    color=WHITE, fontsize=10)
ax_thresh.tick_params(colors=WHITE)
ax_thresh.grid(True, alpha=0.16, color=WHITE, axis="y")
ax_thresh.legend(facecolor="#0f172a", labelcolor=WHITE, fontsize=9,
                  loc="upper left", framealpha=0.9)

# Status annotation
status_txt = ax_thresh.text(
    0.98, 0.06, "", transform=ax_thresh.transAxes,
    ha="right", va="bottom", fontsize=9, color=WHITE,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f172a", alpha=0.85))

# ── Slider ─────────────────────────────────────────────────────────────────────
slider = Slider(ax_slider, "  mu  ", args.mu_min, args.mu_max,
                valinit=MU_VALUES[args.n_mu // 2],
                valstep=MU_STEP, color=ACCENT, track_color="#0f172a")
ax_slider.set_facecolor("#0f172a")
slider.label.set_color(WHITE); slider.valtext.set_color(WHITE)

# ── Phase evolution drawing (left panel) ──────────────────────────────────────
PI_TICKS  = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
PI_LABELS = ["-pi", "-pi/2", "0", "pi/2", "pi"]

def draw_phase(idx):
    """Redraw left panel from cached data at mu index idx."""
    ax_phase.clear()
    ax_phase.set_facecolor(PANEL)

    mu   = MU_VALUES[idx]
    sols = all_trajs[idx]
    cuts = all_cuts[idx]
    t    = sols[0].t

    for sol in sols:
        for spin in range(N):
            ax_phase.plot(t, sol.y[spin],
                          color=SPIN_COLORS[spin % 20],
                          alpha=0.22, linewidth=0.65)

    ax_phase.axhline( np.pi, color="#ef9a9a", linestyle="--",
                      linewidth=1.2, alpha=0.85, label="pi")
    ax_phase.axhline(0,      color="#90caf9", linestyle="--",
                      linewidth=1.2, alpha=0.85, label="0")
    ax_phase.axhline(-np.pi, color="#ef9a9a", linestyle="--",
                      linewidth=1.2, alpha=0.50)

    binarized = mu > mu_star
    ax_phase.text(
        0.97, 0.95,
        "BINARISED ✓" if binarized else "NOT binarised ✗",
        transform=ax_phase.transAxes, ha="right", va="top",
        fontsize=10, fontweight="bold",
        color="#4caf50" if binarized else "#ef5350",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a", alpha=0.90))

    ax_phase.text(
        0.97, 0.07,
        f"best cut = {max(cuts):.1f}\nmean cut = {np.mean(cuts):.2f} +/- {np.std(cuts):.2f}",
        transform=ax_phase.transAxes, ha="right", va="bottom",
        fontsize=9, color=WHITE,
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
        f"({'above' if binarized else 'below'} mu* = {mu_star:.4f})",
        color=WHITE, fontsize=11)
    ax_phase.tick_params(colors=WHITE)
    ax_phase.grid(True, alpha=0.14, color=WHITE, linewidth=0.5)
    ax_phase.legend(loc="upper left", fontsize=8.5,
                    facecolor="#0f172a", labelcolor=WHITE, framealpha=0.90)
    for sp in ax_phase.spines.values():
        sp.set_edgecolor("#334155")

    # Spin colour legend (small patches, bottom-left)
    patches = [mpatches.Patch(color=SPIN_COLORS[s % 20], label=f"spin {s}")
               for s in range(N)]
    ax_phase.legend(handles=patches, loc="lower left", fontsize=7,
                    facecolor="#0f172a", labelcolor=WHITE,
                    framealpha=0.85, ncol=max(1, N // 5))


def save_state(idx):
    """Save current experiment state via MaxCutExperimentLogger."""
    mu   = float(MU_VALUES[idx])
    cuts = all_cuts[idx]
    state = {
        "graph_file"       : args.graph,
        "n_nodes"          : N,
        "mu"               : mu,
        "Ks_equiv"         : mu / 2.0,
        "mu_star"          : float(mu_star),
        "binarised"        : bool(mu > mu_star),
        "seed"             : args.seed,
        "n_init"           : args.n_init,
        "t_span"           : [0.0, args.t_end],
        "n_points"         : args.n_points,
        "best_cut"         : float(max(cuts)),
        "mean_cut"         : float(np.mean(cuts)),
        "std_cut"          : float(np.std(cuts)),
        "n_binarised"      : int(sum(
            np.all(np.abs(np.sin(all_trajs[idx][i].y[:, -1])) < 1e-2)
            for i in range(args.n_init))),
        "mu_range"         : [float(args.mu_min), float(args.mu_max)],
        "all_mu_thresholds": {eq_labels[j]: eq_thresholds[j]
                              for j in range(len(eq_labels))},
    }
    logger.save_interactive_state(state)


def update(val):
    idx = int(np.argmin(np.abs(MU_VALUES - slider.val)))
    mu  = MU_VALUES[idx]

    # Redraw phase panel
    draw_phase(idx)

    # Update threshold bar colours + mu line
    hline_mu.set_ydata([mu, mu])
    for rect, thr in zip(bar_rects, s_thresholds):
        rect.set_facecolor("#4caf50" if mu > thr else "#ef5350")

    # Update status annotation
    n_stable = sum(1 for thr in s_thresholds if mu > thr)
    status_txt.set_text(
        f"mu = {mu:.4f}\n"
        f"mu* = {mu_star:.4f}\n"
        f"{'▲ BINARISED' if mu > mu_star else '▼ not binarised'}\n"
        f"Stable eq: {n_stable} / {len(s_thresholds)}")

    # Save state to JSON on every slider change
    save_state(idx)
    fig.canvas.draw_idle()


slider.on_changed(update)
fig.suptitle(
    f"OIM Max-Cut — Interactive Explorer   {args.graph}  (mu parametrisation)",
    color=WHITE, fontsize=13, fontweight="bold", y=1.003)

# Initial draw
update(MU_VALUES[args.n_mu // 2])
plt.show()
