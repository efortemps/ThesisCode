#!/usr/bin/env python3
"""
experiment_maxcut.py
--------------------
Simple Max-Cut experiment using OIMMaxCut (OIM_mu_v2.py).

For a given graph (loaded from a .txt edge-list) and a single value of mu:
  1. Loads the graph and computes the per-equilibrium binarisation thresholds.
  2. Runs simulate_many() from n_init random initial conditions.
  3. Prints a results table: cut value, binarisation status, energy.
  4. Plots phase evolution theta_i(t) for all spins.
  5. Plots stability threshold bar chart (mu* per Type-I equilibrium).
  6. Saves all data via MaxCutExperimentLogger (metadata.json, RESULTS.txt,
     runs.csv, and plots/).

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
from itertools import product as iproduct
from pathlib import Path

from OIM_Experiment.src.OIM_mu_v2 import OIMMaxCut
from OIM_Experiment.src.graph_utils import read_graph
from OIM_Experiment.src.experiment_logger2 import MaxCutExperimentLogger

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="OIM Max-Cut experiment (single mu)")
parser.add_argument("graph",        nargs="?", default="3node.txt")
parser.add_argument("--mu",         type=float, default=2.0)
parser.add_argument("--n_init",     type=int,   default=15)
parser.add_argument("--t_end",      type=float, default=50.0)
parser.add_argument("--n_points",   type=int,   default=500)
parser.add_argument("--seed",       type=int,   default=42)
args = parser.parse_args()

# ── Setup ──────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(args.seed)
W   = read_graph(args.graph)
N   = W.shape[0]
dataset_name = Path(args.graph).stem

# ── Logger: start experiment ───────────────────────────────────────────────────
oim_tmp = OIMMaxCut(W, mu=args.mu, seed=args.seed)
logger  = MaxCutExperimentLogger()
exp_dir = logger.start_experiment(oim_tmp, args, dataset_name=dataset_name)

print("=" * 65)
print("OIM MAX-CUT EXPERIMENT  (mu parametrisation)")
print("=" * 65)
print(f"  Graph    : {args.graph}  ({N} nodes)")
print(f"  mu       : {args.mu}  (Ks_equiv = {args.mu/2:.4f},  K = 1)")
print(f"  N_init   : {args.n_init}   t_end : {args.t_end}   seed : {args.seed}")

# ── Binarisation threshold ─────────────────────────────────────────────────────
print(f"\nComputing thresholds for all 2^{N} = {2**N} Type-I equilibria...")
oim_ref = OIMMaxCut(W, mu=1.0, seed=0)
eq_bits_list      = list(iproduct([0, 1], repeat=N))
eq_labels         = [str(list(b)) for b in eq_bits_list]
eq_thresholds     = [float(oim_ref.stability_threshold(
                        np.array([b * np.pi for b in bits])))
                     for bits in eq_bits_list]
thresholds_per_eq = {eq_labels[i]: eq_thresholds[i] for i in range(len(eq_labels))}
mu_star           = min(eq_thresholds)

print(f"  mu* = {mu_star:.6f}  "
      f"→  {'BINARISED' if args.mu > mu_star else 'NOT binarised'}"
      f"  (mu={args.mu} {'>' if args.mu > mu_star else '<='} mu*={mu_star:.4f})")

# ── Simulate ──────────────────────────────────────────────────────────────────
phi0_list = [rng.uniform(-np.pi, np.pi, N) for _ in range(args.n_init)]
oim       = OIMMaxCut(W, mu=args.mu, seed=args.seed)
t0        = time.time()
sols      = oim.simulate_many(phi0_list, t_span=(0., args.t_end),
                               n_points=args.n_points)
print(f"  Simulated {args.n_init} trajectories in {time.time()-t0:.2f}s")

# ── Collect per-IC results ────────────────────────────────────────────────────
print(f"\n{'IC':<5} {'Cut (binary)':<16} {'Binarised':<12} Energy (final)")
print("-" * 52)
records    = []
best_cut   = -1.0
best_idx   = 0
for i, sol in enumerate(sols):
    oim.theta = sol.y[:, -1]
    cut       = oim.get_binary_cut_value()
    binar     = oim.is_binarized()
    energ     = oim.get_energy()
    print(f"  {i:<3}  {cut:<16.3f} {'Yes' if binar else 'No':<12} {energ:.4f}")
    records.append({
        "ic"          : i,
        "cut"         : float(cut),
        "binarized"   : bool(binar),
        "energy"      : float(energ),
        "phases_final": sol.y[:, -1].tolist(),
    })
    if cut > best_cut:
        best_cut, best_idx = cut, i

cuts = [r["cut"] for r in records]
print(f"\n  Best cut   : {max(cuts):.3f}")
print(f"  Mean cut   : {np.mean(cuts):.3f}  \u00b1 {np.std(cuts):.3f}")
print(f"  Binarised  : {sum(r['binarized'] for r in records)}/{args.n_init}")

# Best-run partition for log_results
oim.theta   = np.array(records[best_idx]["phases_final"])
partition   = oim.get_spins().astype(int).tolist()
final_energy= records[best_idx]["energy"]

# ── Logger: save results and runs ─────────────────────────────────────────────
logger.log_results(partition, best_cut, final_energy, mu_star=mu_star)
csv_path = logger.log_all_runs(records)
print(f"\nRuns saved   -> {csv_path}")

# ── Theme ──────────────────────────────────────────────────────────────────────
BG, PANEL, WHITE = "#111827", "#1e293b", "#f1f5f9"
SPIN_COLORS = plt.get_cmap("tab20")(np.linspace(0, 1, N))

# ── Plot 1: Phase evolution ────────────────────────────────────────────────────
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
is_bin = args.mu > mu_star
ax.text(0.97, 0.94, "BINARISED \u2713" if is_bin else "NOT binarised \u2717",
        transform=ax.transAxes, ha="right", va="top", fontsize=10, fontweight="bold",
        color="#4caf50" if is_bin else "#ef5350",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a", alpha=0.9))
ax.text(0.97, 0.07,
        f"Best cut = {max(cuts):.1f}\nmu = {args.mu}   mu* = {mu_star:.4f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=WHITE,
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
ax.set_title(f"Phase evolution — {args.graph},  N={N},  mu={args.mu},  mu*={mu_star:.4f}",
             color=WHITE, fontsize=11)
ax.tick_params(colors=WHITE)
ax.grid(True, alpha=0.14, color=WHITE)
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=SPIN_COLORS[s % 20], label=f"spin {s}") for s in range(N)]
ax.legend(handles=patches, loc="lower left", fontsize=8,
          facecolor="#0f172a", labelcolor=WHITE, framealpha=0.9, ncol=max(1, N//5))
for sp in ax.spines.values(): sp.set_edgecolor("#334155")
plt.tight_layout()
phase_fig = "experiment_maxcut_phases.png"
plt.savefig(phase_fig, dpi=140, bbox_inches="tight", facecolor=BG)
logger.save_plot(phase_fig)
print(f"Phase plot   -> {logger.get_output_subdir('plots')}/{phase_fig}")

# ── Plot 2: Stability thresholds ──────────────────────────────────────────────
sorted_idx   = np.argsort(eq_thresholds)
s_labels     = [eq_labels[i]     for i in sorted_idx][:min(32, len(eq_labels))]
s_thresholds = [eq_thresholds[i] for i in sorted_idx][:min(32, len(eq_labels))]

fig2, ax2 = plt.subplots(figsize=(max(8, len(s_labels) * 0.55), 5), facecolor=BG)
ax2.set_facecolor(PANEL)
colors_bar = ["#4caf50" if args.mu > v else "#ef5350" for v in s_thresholds]
ax2.bar(range(len(s_thresholds)), s_thresholds, color=colors_bar, alpha=0.85)
ax2.axhline(args.mu,  color="#ffb74d", linestyle="--", linewidth=2.0,
            label=f"mu = {args.mu}")
ax2.axhline(mu_star,  color=WHITE,     linestyle=":",  linewidth=1.5,
            label=f"mu* = {mu_star:.4f}")
ax2.set_xticks(range(len(s_labels)))
ax2.set_xticklabels(s_labels, rotation=45, ha="right",
                    fontsize=max(5, 8 - N // 4), color=WHITE)
ax2.set_ylabel("Stability threshold  mu*_i = lambda_max(D(phi*_i))",
               color=WHITE, fontsize=10)
ax2.set_title(f"Per-equilibrium stability thresholds — {args.graph}", color=WHITE, fontsize=11)
ax2.tick_params(colors=WHITE)
ax2.grid(True, alpha=0.14, color=WHITE, axis="y")
ax2.legend(facecolor="#0f172a", labelcolor=WHITE, fontsize=9)
for sp in ax2.spines.values(): sp.set_edgecolor("#334155")
plt.tight_layout()
thresh_fig = "experiment_maxcut_thresholds.png"
plt.savefig(thresh_fig, dpi=140, bbox_inches="tight", facecolor=BG)
logger.save_plot(thresh_fig)
print(f"Thresh plot  -> {logger.get_output_subdir('plots')}/{thresh_fig}")
print("=" * 65)
