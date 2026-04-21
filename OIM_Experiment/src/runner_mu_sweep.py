import argparse
import numpy as np
import csv
from pathlib import Path

from OIM_Experiment.src.OIM_mu import OIMMaxCut
from OIM_Experiment.src.graph_utils import read_graph, random_graph, verify_cut
from OIM_Experiment.src.visualization import MuSweepVisualizer
from OIM_Experiment.src.experiment_logger import MaxCutExperimentLogger


def get_args():
    parser = argparse.ArgumentParser(
        description="OIM mu-sweep experiment: iterate over mu values and study binarization."
    )
    # ── Graph source ────────────────────────────────────────────────
    parser.add_argument("--graph", type=str, default=None,
                        help="Path to graph edge-list file (.txt).")
    parser.add_argument("--random", type=int, default=None,
                        help="Generate a random G(n,p) graph with this many nodes.")
    parser.add_argument("--prob", type=float, default=0.5,
                        help="Edge probability for the random graph (default 0.5).")
    # ── mu sweep ────────────────────────────────────────────────────
    parser.add_argument("--mu_min", type=float, default=0.01,
                        help="Minimum mu value (default 0.01).")
    parser.add_argument("--mu_max", type=float, default=20.0,
                        help="Maximum mu value (default 20.0).")
    parser.add_argument("--n_mu", type=int, default=200,
                        help="Number of mu values to sweep (default 200).")
    # ── Simulation ──────────────────────────────────────────────────
    parser.add_argument("--steps", type=int, default=8_000,
                        help="Number of Euler integration steps per mu (default 8 000).")
    parser.add_argument("--binarization_tol", type=float, default=1e-2,
                        help="Tolerance for binarization check |sin(theta_i)| < tol (default 1e-2).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42).")
    parser.add_argument("--init_mode", type=str, default="small_random",
                        choices=["small_random"],
                        help="Initialisation strategy (default small_random).")
    # ── Output ──────────────────────────────────────────────────────
    parser.add_argument("--output", type=str, default="experiments_MaxCut_OIM",
                        help="Base output directory (default experiments_MaxCut_OIM).")
    return parser.parse_args()


def run_single(W, mu, steps, seed, init_mode, binarization_tol):
    """Run one OIM simulation for a given mu; return result dict."""
    net = OIMMaxCut(W, mu=mu, seed=seed, init_mode=init_mode)
    for _ in range(steps):
        net.update()
    return {
        "mu":                    mu,
        "binarization_residual": net.binarization_residual(),
        "is_binarized":          net.is_binarized(tol=binarization_tol),
        "energy":                net.get_energy(),
        "binary_cut_value":      net.get_binary_cut_value(),
        "phases":                net.theta.copy(),
        "hessian_eigenvalues":   np.linalg.eigvalsh(net.get_hessian()).tolist(),
        "gradient_norm":         float(np.linalg.norm(net.get_gradient())),
    }


def main():
    args = get_args()

    print("=" * 60)
    print("OIM MU-SWEEP EXPERIMENT")
    print("=" * 60)
    print(f"  mu range    : [{args.mu_min}, {args.mu_max}]  ({args.n_mu} values)")
    print(f"  steps/mu    : {args.steps}")
    print(f"  binar. tol  : {args.binarization_tol}")
    print(f"  seed        : {args.seed}")
    print("=" * 60)

    # ── Load / generate graph ────────────────────────────────────────
    if args.graph:
        print(f"\nLoading graph from '{args.graph}'...")
        W = read_graph(args.graph)
        dataset_name = Path(args.graph).stem
    elif args.random:
        print(f"\nGenerating random G({args.random}, p={args.prob}) graph...")
        W = random_graph(args.random, edge_prob=args.prob, seed=args.seed)
        dataset_name = f"random_{args.random}nodes_p{args.prob}"
    else:
        raise ValueError("Provide --graph <file.txt> or --random <n_nodes>.")

    n = len(W)
    n_edges = int(np.sum(W > 0) / 2)
    print(f"  Graph : {n} nodes, {n_edges} edges")

    # ── Logger ───────────────────────────────────────────────────────
    logger = MaxCutExperimentLogger(base_output_dir=args.output)
    experiment_dir = logger.start_mu_sweep_experiment(args, dataset_name, n)
    print(f"  Output: {experiment_dir}")

    # ── mu sweep ─────────────────────────────────────────────────────
    mu_values = np.linspace(args.mu_min, args.mu_max, args.n_mu)
    results = {k: [] for k in [
        "mu", "binarization_residual", "is_binarized",
        "energy", "binary_cut_value", "phases", "hessian_eigenvalues", "gradient_norm"
    ]}

    print("\nRunning mu-sweep...")
    for idx, mu in enumerate(mu_values):
        r = run_single(W, mu, args.steps, args.seed, args.init_mode, args.binarization_tol)
        for k in results:
            results[k].append(r[k])
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  [{idx+1:4d}/{args.n_mu}]  mu={mu:.3f}  "
                  f"residual={r['binarization_residual']:.4f}  "
                  f"binarized={r['is_binarized']}", end="\r")

    print(f"\n✓ Sweep complete ({args.n_mu} mu values).")

    # ── Save CSV ─────────────────────────────────────────────────────
    csv_path = logger.log_mu_sweep_results(results, experiment_dir)
    print(f"✓ Results saved to: {csv_path}")

    # ── Visualize ────────────────────────────────────────────────────
    vis_output = logger.get_output_subdir("visualization")
    visualizer = MuSweepVisualizer(output_dir=vis_output)
    plot_path = visualizer.generate_sweep_plot(results, W, dataset_name)
    print(f"✓ Plot saved to: {plot_path}")

    # ── Summary ──────────────────────────────────────────────────────
    mu_arr = np.array(results["mu"])
    binarized_mask = np.array(results["is_binarized"])
    first_binarized = mu_arr[binarized_mask][0] if binarized_mask.any() else None

    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)
    print(f"  Graph              : {dataset_name}  ({n} nodes, {n_edges} edges)")
    print(f"  mu range           : [{args.mu_min:.3f}, {args.mu_max:.3f}]")
    print(f"  Binarized at mu >= : {first_binarized:.4f}" if first_binarized else
          "  Never fully binarized in this range.")
    print(f"  Fraction binarized : {binarized_mask.mean()*100:.1f}%")
    print(f"  Output directory   : {experiment_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
