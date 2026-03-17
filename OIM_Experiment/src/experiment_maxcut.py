#!/usr/bin/env python3
"""
experiment_maxcut.py
--------------------
Simple Max-Cut experiment using OIMMaxCut (OIM_mu_v2.py).

Printed output
--------------
  - Per-equilibrium stability threshold  mu*_i = lambda_max(D(phi*_i))
    for every Type-I M2 equilibrium (one per spin assignment).
  - Global binarisation threshold  mu_bin = min_i(mu*_i)  (Remark 7).
  - Clear verdict: is the current mu above/below mu_bin?

Plots
-----
  1. Phase evolution  theta_i(t)  for all spins and all ICs.
  2. Threshold bar chart showing:
       - Each bar  = per-equilibrium stability threshold  mu*_i
       - Orange dashed line = current mu
       - White dotted line  = global binarisation threshold  mu_bin
       - Green/Red shaded region to show binarised/not-binarised zone

Usage
-----
    python experiment_maxcut.py [graph.txt] [--mu 2.0] [--n_init 15] [--seed 42]
"""

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from pathlib import Path

from OIM_Experiment.src.OIM_mu_v2 import OIMMaxCut
from OIM_Experiment.src.graph_utils import read_graph
from OIM_Experiment.src.experiment_logger2 import MaxCutExperimentLogger

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="OIM Max-Cut experiment (single mu)")
parser.add_argument("graph",      nargs="?", default="3node.txt")
parser.add_argument("--mu",       type=float, default=2.0)
parser.add_argument("--n_init",   type=int,   default=15)
parser.add_argument("--t_end",    type=float, default=50.0)
parser.add_argument("--n_points", type=int,   default=500)
parser.add_argument("--seed",     type=int,   default=42)
args = parser.parse_args()

# ── Setup ──────────────────────────────────────────────────────────────────────
rng          = np.random.default_rng(args.seed)
W            = read_graph(args.graph)
N            = W.shape[0]
dataset_name = Path(args.graph).stem

# ── Logger ─────────────────────────────────────────────────────────────────────
oim_tmp = OIMMaxCut(W, mu=args.mu, seed=args.seed)
logger  = MaxCutExperimentLogger()
exp_dir = logger.start_experiment(oim_tmp, args, dataset_name=dataset_name)

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("OIM MAX-CUT EXPERIMENT  (mu parametrisation)")
print("=" * 65)
print(f"  Graph    : {args.graph}  ({N} nodes)")
print(f"  mu       : {args.mu}  (Ks_equiv = {args.mu/2:.4f},  K = 1)")
print(f"  N_init   : {args.n_init}   t_end : {args.t_end}   seed : {args.seed}")

# ── Compute all thresholds via OIMMaxCut.binarization_threshold() ─────────────
print(f"\nComputing thresholds for all 2^{N} = {2**N} Type-I M2 equilibria...")
oim_ref    = OIMMaxCut(W, mu=1.0, seed=0)
bin_info   = oim_ref.binarization_threshold()

mu_bin          = bin_info["mu_bin"]          # global binarisation threshold
Ks_bin          = bin_info["Ks_bin"]          # equivalent Ks
thresholds_dict = bin_info["all_thresholds"]  # {pattern: mu*_i}
easiest_eq      = bin_info["easiest_eq"]
hardest_eq      = bin_info["hardest_eq"]

eq_labels     = list(thresholds_dict.keys())
eq_thresholds = list(thresholds_dict.values())

# ── Print threshold table ─────────────────────────────────────────────────────
print(f"\n  {'Equilibrium phi*':<20} {'mu*_i':>10}  {'Ks*_i':>8}  Status at mu={args.mu}")
print(f"  {'-'*60}")
for label, thr in sorted(thresholds_dict.items(), key=lambda x: x[1]):
    verdict = "STABLE ✓" if args.mu > thr else "unstable ✗"
    print(f"  {label:<20} {thr:>10.4f}  {thr/2:>8.4f}  {verdict}")

print(f"\n  ┌─────────────────────────────────────────────────────┐")
print(f"  │  Per-equilibrium stability threshold (Theorem 2):   │")
print(f"  │    mu*_i = lambda_max( D(phi*_i) )                  │")
print(f"  │    A specific phi*_i is stable iff  mu > mu*_i      │")
print(f"  ├─────────────────────────────────────────────────────┤")
print(f"  │  Global BINARISATION threshold (Remark 7):          │")
print(f"  │    mu_bin = min_i( mu*_i ) = {mu_bin:<8.4f}              │")
print(f"  │    Ks_bin = mu_bin / 2    = {Ks_bin:<8.4f}              │")
print(f"  │    The system binarises iff  mu > mu_bin            │")
print(f"  ├─────────────────────────────────────────────────────┤")
print(f"  │  Current mu = {args.mu:<8.4f}                              │")
print(f"  │  Status: {'BINARISED ✓  (mu > mu_bin)' if args.mu > mu_bin else 'NOT binarised ✗  (mu <= mu_bin)':<38}│")
print(f"  │  Easiest equilibrium to stabilise: {easiest_eq:<18} │")
print(f"  │  Hardest equilibrium to stabilise: {hardest_eq:<18} │")
print(f"  └─────────────────────────────────────────────────────┘")

# ── Simulate ──────────────────────────────────────────────────────────────────
phi0_list = [rng.uniform(-np.pi, np.pi, N) for _ in range(args.n_init)]
oim       = OIMMaxCut(W, mu=args.mu, seed=args.seed)
t0        = time.time()
sols      = oim.simulate_many(phi0_list, t_span=(0., args.t_end), n_points=args.n_points)
print(f"\n  Simulated {args.n_init} trajectories in {time.time()-t0:.2f}s")

# ── Per-IC results ────────────────────────────────────────────────────────────
print(f"\n  {'IC':<4} {'Cut':>6} {'Binarised':>10} {'Energy':>12}")
print(f"  {'-'*38}")
records  = []
best_cut = -1.0
best_idx = 0
for i, sol in enumerate(sols):
    oim.theta = sol.y[:, -1]
    cut       = oim.get_binary_cut_value()
    binar     = oim.is_binarized()
    energ     = oim.get_energy()
    tag       = "✓" if binar else "✗"
    print(f"  {i:<4} {cut:>6.3f} {tag:>10} {energ:>12.4f}")
    records.append({"ic": i, "cut": float(cut), "binarized": bool(binar),
                    "energy": float(energ), "phases_final": sol.y[:, -1].tolist()})
    if cut > best_cut:
        best_cut, best_idx = cut, i

cuts = [r["cut"] for r in records]
print(f"\n  Best cut   : {max(cuts):.3f}")
print(f"  Mean cut   : {np.mean(cuts):.3f}  ± {np.std(cuts):.3f}")
print(f"  Binarised  : {sum(r['binarized'] for r in records)}/{args.n_init}")
print("=" * 65)

# ── Logger: save ──────────────────────────────────────────────────────────────
oim.theta    = np.array(records[best_idx]["phases_final"])
partition    = oim.get_spins().astype(int).tolist()
final_energy = records[best_idx]["energy"]
logger.log_results(partition, best_cut, final_energy, mu_star=mu_bin)
csv_path = logger.log_all_runs(records)
print(f"  Runs saved -> {csv_path}")

# ── Theme ─────────────────────────────────────────────────────────────────────
BG, PANEL, WHITE = "#111827", "#1e293b", "#f1f5f9"
SPIN_COLORS = plt.get_cmap("tab20")(np.linspace(0, 1, max(N, 2)))

# ════════════════════════════════════════════════════════════════════════
# Plot 1 — Phase evolution
# ════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
ax.set_facecolor(PANEL)
t = sols[0].t
for sol in sols:
    for spin in range(N):
        ax.plot(t, sol.y[spin], color=SPIN_COLORS[spin % 20],
                alpha=0.25, linewidth=0.7)
ax.axhline( np.pi, color="#ef9a9a", linestyle="--", linewidth=1.2, alpha=0.8, label="pi")
ax.axhline(0,      color="#90caf9", linestyle="--", linewidth=1.2, alpha=0.8, label="0")
ax.axhline(-np.pi, color="#ef9a9a", linestyle="--", linewidth=1.2, alpha=0.5)

is_bin = args.mu > mu_bin
ax.text(0.97, 0.96,
        "BINARISED ✓" if is_bin else "NOT binarised ✗",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, fontweight="bold",
        color="#4caf50" if is_bin else "#ef5350",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a", alpha=0.9))
ax.text(0.97, 0.06,
        f"Best cut = {max(cuts):.1f}\n"
        f"mu = {args.mu}\n"
        f"mu_bin = {mu_bin:.4f}  (binarisation threshold)",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8.5, color=WHITE,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a", alpha=0.85))

ax.set_xlim(t[0], t[-1])
ax.set_ylim(-4.5, 4.5)
ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_yticklabels(["-pi", "-pi/2", "0", "pi/2", "pi"], color=WHITE, fontsize=9)
xt = np.linspace(t[0], t[-1], 5)
ax.set_xticks(xt)
ax.set_xticklabels([f"{v:.0f}" for v in xt], color=WHITE, fontsize=9)
ax.set_xlabel("time  t", color=WHITE, fontsize=11)
ax.set_ylabel("phase  theta  (rad)", color=WHITE, fontsize=11)
ax.set_title(
    f"Phase evolution — {args.graph},  N={N},  mu={args.mu}  "
    f"({'above' if is_bin else 'below'} mu_bin={mu_bin:.4f})",
    color=WHITE, fontsize=11)
ax.tick_params(colors=WHITE)
ax.grid(True, alpha=0.14, color=WHITE)
spin_patches = [mpatches.Patch(color=SPIN_COLORS[s % 20], label=f"spin {s}")
                for s in range(N)]
ax.legend(handles=spin_patches, loc="lower left", fontsize=8,
          facecolor="#0f172a", labelcolor=WHITE, framealpha=0.9, ncol=max(1, N // 5))
for sp in ax.spines.values(): sp.set_edgecolor("#334155")
plt.tight_layout()
phase_fig = "experiment_maxcut_phases.png"
plt.savefig(phase_fig, dpi=140, bbox_inches="tight", facecolor=BG)
logger.save_plot(phase_fig)
print(f"  Phase plot -> {logger.get_output_subdir('plots')}/{phase_fig}")

# ════════════════════════════════════════════════════════════════════════
# Plot 2 — Threshold bar chart
# ════════════════════════════════════════════════════════════════════════
sorted_idx   = np.argsort(eq_thresholds)
s_labels     = [eq_labels[i]     for i in sorted_idx][:min(32, len(eq_labels))]
s_thresholds = [eq_thresholds[i] for i in sorted_idx][:min(32, len(eq_labels))]
n_bars       = len(s_labels)

fig2, ax2 = plt.subplots(figsize=(max(9, n_bars * 0.55), 6), facecolor=BG)
ax2.set_facecolor(PANEL)

# Coloured background regions
y_max = max(s_thresholds) * 1.30 + 0.3
ax2.axhspan(0,        mu_bin,   alpha=0.07, color="#ef5350", zorder=0)  # red: not binarised
ax2.axhspan(mu_bin,   y_max,    alpha=0.07, color="#4caf50", zorder=0)  # green: binarised

# Bars — green if current mu already stabilises this eq, red otherwise
bar_colors = ["#4caf50" if args.mu > v else "#ef5350" for v in s_thresholds]
ax2.bar(range(n_bars), s_thresholds, color=bar_colors, alpha=0.85, zorder=2)

# Current mu line
ax2.axhline(args.mu, color="#ffb74d", linestyle="--", linewidth=2.2, zorder=3,
            label=f"Current  mu = {args.mu}")

# Global binarisation threshold line
ax2.axhline(mu_bin, color=WHITE, linestyle=":", linewidth=2.0, zorder=4,
            label=f"Binarisation threshold  mu_bin = {mu_bin:.4f}  (Remark 7)")

# Region labels
ax2.text(n_bars - 0.3, mu_bin * 0.5,
         "NOT binarised\n(no eq. stable)",
         ha="right", va="center", fontsize=8, color="#ef9a9a",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f172a", alpha=0.75))
ax2.text(n_bars - 0.3, mu_bin + (y_max - mu_bin) * 0.55,
         "BINARISED\n(≥1 eq. stable)",
         ha="right", va="center", fontsize=8, color="#a5d6a7",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f172a", alpha=0.75))

ax2.set_xlim(-0.6, n_bars - 0.4)
ax2.set_ylim(0, y_max)
ax2.set_xticks(range(n_bars))
ax2.set_xticklabels(s_labels, rotation=45, ha="right",
                    fontsize=max(5, 8 - N // 4), color=WHITE)
ax2.set_ylabel(
    "Per-equilibrium stability threshold"
    "mu*_i = lambda_max( D(phi*_i) )  [Theorem 2]",
    color=WHITE, fontsize=10)
ax2.set_title(
    f"Stability thresholds — {args.graph}  |  "
    f"Binarisation threshold  mu_bin = min_i(mu*_i) = {mu_bin:.4f}  [Remark 7]",
    color=WHITE, fontsize=10)
ax2.tick_params(colors=WHITE)
ax2.grid(True, alpha=0.14, color=WHITE, axis="y")

legend_elements = [
    mpatches.Patch(facecolor="#4caf50", alpha=0.85, label="Stable at current mu"),
    mpatches.Patch(facecolor="#ef5350", alpha=0.85, label="Unstable at current mu"),
    mlines.Line2D([0],[0], color="#ffb74d", linestyle="--", lw=2,
               label=f"Current mu = {args.mu}"),
    mlines.Line2D([0],[0], color=WHITE, linestyle=":", lw=2,
               label=f"mu_bin = {mu_bin:.4f}  (global binarisation threshold)"),
]
ax2.legend(handles=legend_elements, facecolor="#0f172a", labelcolor=WHITE,
           fontsize=8.5, loc="upper left", framealpha=0.9)
for sp in ax2.spines.values(): sp.set_edgecolor("#334155")
plt.tight_layout()
thresh_fig = "experiment_maxcut_thresholds.png"
plt.savefig(thresh_fig, dpi=140, bbox_inches="tight", facecolor=BG)
logger.save_plot(thresh_fig)
print(f"  Thresh plot -> {logger.get_output_subdir('plots')}/{thresh_fig}")
print("=" * 65)
