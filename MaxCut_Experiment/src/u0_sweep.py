"""
u0_sweep.py
-----------
Parameter sweep over the gain u0 for the Hopfield-Tank Max-Cut network.

Usage:
    python u0_sweep.py --data data/k33.txt --steps 300000 --seeds 5
    python u0_sweep.py --random 10 --steps 300000 --seeds 10

For each u0 value the script:
    1. Runs the Hopfield ODE for --steps steps
    2. Records the final binary cut value and final energy
    3. Repeats over --seeds different random seeds (to average out initialisation noise)
    4. Saves results to a CSV and produces two plots:
        - Mean binary cut value vs u0  (with min/max band)
        - Final energy vs u0
"""

import argparse
import numpy as np
import csv
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from MaxCut_Experiment.src.Hopfield    import HopfieldNetMaxCut
from MaxCut_Experiment.src.graph_utils import read_graph, random_graph, verify_cut


# ── U0 values to sweep ────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description='u0 parameter sweep for Max-Cut')
    parser.add_argument('--steps',  type=int,   default=300_000,
                        help='Euler steps per run')
    parser.add_argument('--seeds',  type=int,   default=5,
                        help='Number of random seeds per u0 value')
    parser.add_argument('--n_u0',   type=int,   default=30,
                        help='Number of u0 values to sweep')
    parser.add_argument('--data',   type=str,   default=None,
                        help='Path to graph edge-list file')
    parser.add_argument('--random', type=int,   default=None,
                        help='Generate a random G(n,p) graph with this many nodes')
    parser.add_argument('--prob',   type=float, default=0.5,
                        help='Edge probability for random graph')
    parser.add_argument('--output', type=str,   default='experiments_maxcut/u0_sweep',
                        help='Output directory for plots and CSV')
    return parser.parse_args()

def run_single(W, u0, seed, n_steps):
    """
    Run one experiment with a given u0 and seed.
    Returns (binary_cut_value, final_energy).
    """
    net = HopfieldNetMaxCut(W, seed=seed, u0=u0)
    for _ in range(n_steps):
        net.update()
    return net.get_binary_cut_value(), net.get_energy()


def main():
    args = get_args()
    os.makedirs(args.output, exist_ok=True)
    U0_VALUES = np.logspace(np.log10(0.005), np.log10(5.0), args.n_u0).tolist()

    # ── Load graph ────────────────────────────────────────────────────────────
    if args.data:
        W            = read_graph(args.data)
        graph_label  = os.path.splitext(os.path.basename(args.data))[0]
    elif args.random:
        W            = random_graph(args.random, edge_prob=args.prob, seed=0)
        graph_label  = f"random_{args.random}n_p{args.prob}"
    else:
        raise ValueError("Provide --data <file>  or  --random <n_nodes>")

    n      = len(W)

    print("=" * 60)
    print("U0 PARAMETER SWEEP — Hopfield-Tank Max-Cut")
    print("=" * 60)
    print(f"Graph           : {graph_label}  ({n} nodes)")
    print(f"Steps per run   : {args.steps}")
    print(f"Seeds per u0    : {args.seeds}")
    print("=" * 60)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    results = []   

    total_runs = len(U0_VALUES) * args.seeds
    run_idx    = 0

    for u0 in U0_VALUES:
        cuts, energies = [], []
        for seed in range(args.seeds):
            cut, energy = run_single(W, u0, seed, args.steps)
            cuts.append(cut)
            energies.append(energy)
            results.append({
                'u0':     u0,
                'seed':   seed,
                'cut':    cut,
                'energy': energy,
            })
            run_idx += 1
            print(f"  [{run_idx:3d}/{total_runs}]  u0={u0:.4f}  seed={seed}  "
                  f"cut={cut:.2f}  energy={energy:.4f}", end='\r')

        mean_cut = np.mean(cuts)
        print(f"  u0={u0:.4f}  mean_cut={mean_cut:.2f}  "
              f"[{min(cuts):.2f}, {max(cuts):.2f}]         ")

    print(f"\n✓ Sweep complete ({total_runs} runs)")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path   = os.path.join(args.output, f"u0_sweep_{graph_label}_{timestamp}.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['u0', 'seed', 'cut', 'energy'])
        writer.writeheader()
        writer.writerows(results)

    print(f"✓ Results saved to: {csv_path}")

    # ── Aggregate per u0 ──────────────────────────────────────────────────────
    u0_arr      = np.array(U0_VALUES)
    mean_cuts   = np.array([np.mean([r['cut']    for r in results if r['u0'] == u0])
                             for u0 in U0_VALUES])
    min_cuts    = np.array([np.min( [r['cut']    for r in results if r['u0'] == u0])
                             for u0 in U0_VALUES])
    max_cuts    = np.array([np.max( [r['cut']    for r in results if r['u0'] == u0])
                             for u0 in U0_VALUES])
    mean_energy = np.array([np.mean([r['energy'] for r in results if r['u0'] == u0])
                             for u0 in U0_VALUES])

    # ── Plot 1: mean binary cut value vs u0 ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(u0_arr, min_cuts, max_cuts,
                    alpha=0.25, color='steelblue', label='min/max across seeds')
    ax.plot(u0_arr, mean_cuts, 'o-', color='steelblue',
            linewidth=2, markersize=6, label='Mean cut value')

    ax.set_xscale('log')
    ax.set_xlabel('u0  (log scale)', fontsize=12)
    ax.set_ylabel('Binary cut value', fontsize=12)
    ax.set_title(f'Max-Cut quality vs gain u0  —  {graph_label}', fontsize=13)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    plot1_path = os.path.join(args.output, f"cut_vs_u0_{graph_label}_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(plot1_path, dpi=150)
    plt.close()
    print(f"✓ Plot saved to: {plot1_path}")

    # ── Plot 2: final energy vs u0 ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(u0_arr, mean_energy, 's-', color='darkorange',
            linewidth=2, markersize=6, label='Mean final energy')
    ax.set_xscale('log')
    ax.set_xlabel('u0  (log scale)', fontsize=12)
    ax.set_ylabel('Final Lyapunov energy E', fontsize=12)
    ax.set_title(f'Final energy vs gain u0  —  {graph_label}', fontsize=13)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    plot2_path = os.path.join(args.output, f"energy_vs_u0_{graph_label}_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(plot2_path, dpi=150)
    plt.close()
    print(f"✓ Plot saved to: {plot2_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    best_idx = int(np.argmax(mean_cuts))
    print(f"Best mean cut : {mean_cuts[best_idx]:.2f}  at  u0 = {U0_VALUES[best_idx]}")
    print("=" * 60)


if __name__ == '__main__':
    main()
