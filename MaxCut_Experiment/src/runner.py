import argparse
import numpy as np
from pathlib import Path

from MaxCut_Experiment.src.Hopfield import HopfieldNetMaxCut
from MaxCut_Experiment.src.graph_utils import read_graph, random_graph, verify_cut
from MaxCut_Experiment.src.visualization import MaxCutVisualizer
from MaxCut_Experiment.src.experiment_logger import MaxCutExperimentLogger


def get_args():
    parser = argparse.ArgumentParser(description='Hopfield-Tank Max-Cut Solver')
    parser.add_argument('--u0', type=float, default=0.05, help='Gain parameter for tanh activation (default 0.05)')
    parser.add_argument('--steps',  type=int,   default=300_000,
                        help='Number of Euler integration steps')
    parser.add_argument('--freq',   type=int,   default=3_000,
                        help='Snapshot frequency (steps between snapshots)')
    parser.add_argument('--seed',   type=int,   default=42,
                        help='Random seed')
    parser.add_argument('--data',   type=str,   default=None,
                        help='Path to graph edge-list file (.txt)')
    parser.add_argument('--init_mode',type=str,default='small_random',
    choices=['small_random', 'large_random', 'bad_partition',
             'ferromagnetic', 'min_eigenvec'],
    help='Initialisation strategy for membrane potentials.')

    parser.add_argument('--random', type=int,   default=None,
                        help='Generate a random G(n,p) graph with this many nodes')
    parser.add_argument('--prob',   type=float, default=0.5,
                        help='Edge probability for the random graph (default 0.5)')
    parser.add_argument('--video',  action='store_true',
                        help='Generate an mp4 video from snapshots (requires ffmpeg)')
    parser.add_argument('--fps',    type=int,   default=10,
                        help='Frames per second for the video')
    parser.add_argument('--output', type=str,   default='experiments_maxcut',
                        help='Base output directory')
    return parser.parse_args()


def main():
    args = get_args()

    print("=" * 60)
    print("HOPFIELD-TANK MAX-CUT SOLVER")
    print("=" * 60)
    print(f"Steps             : {args.steps}")
    print(f"Snapshot frequency: {args.freq}")
    print(f"Seed              : {args.seed}")
    print("=" * 60)

    # ---- Load or generate graph ----
    if args.data:
        print(f"\nLoading graph from '{args.data}'...")
        W            = read_graph(args.data)
        dataset_name = Path(args.data).stem
    elif args.random:
        print(f"\nGenerating random G({args.random}, p={args.prob}) graph...")
        W            = random_graph(args.random, edge_prob=args.prob, seed=args.seed)
        dataset_name = f"random_{args.random}nodes_p{args.prob}"
    else:
        raise ValueError("Provide --data <file>  or  --random <n_nodes>")

    n       = len(W)
    n_edges = int(np.sum(W > 0) / 2)
    print(f"✓ Graph: {n} nodes, {n_edges} edges")

    # ---- Initialise network ----
    print("\nInitialising Hopfield-Tank Max-Cut network...")
    net = HopfieldNetMaxCut(W, seed=args.seed, u0 = args.u0, init_mode=args.init_mode)
    print(f"✓ Network initialised ({n} neurons)")

    # ---- Initialise logger and visualizer ----
    logger         = MaxCutExperimentLogger(base_output_dir=args.output)
    experiment_dir = logger.start_experiment(net, args, dataset_name=dataset_name)

    vis_output = logger.get_output_subdir('visualization')
    visualizer = MaxCutVisualizer(output_dir=vis_output)

    # ---- Run simulation ----
    print("\nRunning simulation...")
    for step in range(args.steps):
        net.update()

        if step % args.freq == 0:
            visualizer.add_snapshot(net.get_net_state(), net.get_net_configuration())

        if step % 10_000 == 0:
            print(f"  Step {step}/{args.steps}  |  "
                  f"E = {net.get_energy():.4f}  "
                  f"Cut = {net.get_cut_value():.2f}", end='\r')

    print(f"\n✓ Simulation complete ({args.steps} steps)")

    # ---- Generate visualizations ----
    visualizer.generate_images(W)
    if args.video:
        visualizer.generate_video(fps=args.fps)

    # ---- Decode and evaluate result ----
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    partition      = net.get_partition().tolist()
    binary_cut     = net.get_binary_cut_value()
    final_energy   = net.get_energy()
    verified_cut   = verify_cut(W, np.array(partition))   # independent check

    set_A = [i for i, s in enumerate(partition) if s ==  1]
    set_B = [i for i, s in enumerate(partition) if s == -1]

    print(f"Binary cut value (formula)  : {binary_cut:.4f}")
    print(f"Binary cut value (verified) : {verified_cut:.4f}")
    print(f"Final Lyapunov energy       : {final_energy:.6f}")
    print(f"Partition A ({len(set_A)} nodes): {set_A}")
    print(f"Partition B ({len(set_B)} nodes): {set_B}")

    # ---- Log results ----
    logger.log_results(partition, binary_cut, final_energy)
    logger.append_to_comparison_csv()

    print(f"\n✓ All results saved to: {experiment_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
