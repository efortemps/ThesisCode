import argparse
from pathlib import Path
from TSP_Experiment.src.Hopfield_2 import HopfieldNet
from TSP_Experiment.src.inputProcessing import read_data, distance_matrix, normalize, distance
from TSP_Experiment.src.visualization import SimpleVisualizer
from TSP_Experiment.src.experiment_logger import ExperimentLogger

def get_args():
    parser = argparse.ArgumentParser(description='Hopfield-Tank TSP Solver')
    parser.add_argument('--steps', type=int, default=5000, help='Number of steps')
    parser.add_argument('--freq', type=int, default=50, help='Snapshot frequency')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--data', type=str, default='data/cities.txt', help='Path to city data file')
    parser.add_argument('--video', action='store_true', help='Generate video')
    parser.add_argument('--fps', type=int, default=10, help='Video frame rate')
    parser.add_argument('--output', type=str, default='experiments', help='Base output directory')
    return parser.parse_args()

def check_tour_validity(tour, n_cities):
    """Check if tour is a valid Hamiltonian cycle."""
    return len(tour) == n_cities and len(set(tour)) == n_cities

def main():
    args = get_args()
    
    print("="*60)
    print("HOPFIELD-TANK TSP SOLVER")
    print("="*60)
    print(f"Steps: {args.steps}")
    print(f"Snapshot frequency: {args.freq}")
    print(f"Seed: {args.seed}")
    print(f"Data file: {args.data}")
    print("="*60)

    # Load data
    print("\nLoading city data...")
    coordinates = read_data(args.data)
    distances = normalize(distance_matrix(coordinates))
    print(f"✓ Loaded {len(coordinates)} cities")

    # Initialize network
    print("\nInitializing Hopfield network...")
    net = HopfieldNet(distances, args.seed)
    print(f"✓ Network initialized ({len(distances)}×{len(distances)} neurons)")

    # Initialize experiment logger
    logger = ExperimentLogger(base_output_dir=args.output)
    experiment_dir = logger.start_experiment(net, args, dataset_name=Path(args.data).stem)

    # Initialize visualizer with experiment-specific output dir
    vis_output = logger.get_output_subdir('visualization')
    visualizer = SimpleVisualizer(output_dir=vis_output)

    # Run simulation
    print("\nRunning simulation...")
    for step in range(args.steps):
        net.update()
        
        if step % args.freq == 0:
            visualizer.add_snapshot(
                net.get_net_state(),
                net.get_net_configuration()
            )
        
        if step % 100 == 0:
            print(f"  Step {step}/{args.steps}", end='\r')
    
    print(f"\n✓ Simulation complete ({args.steps} steps)")

    # Generate visualizations
    visualizer.generate_images(coordinates, distances)
    
    if args.video:
        visualizer.generate_video(fps=args.fps)

    # Compute results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    activations = net.activations()
    final_energy = net.get_energy()
    
    # Decode tour
    tour = [int(activations[:, pos].argmax()) for pos in range(len(activations[0]))]
    
    # Compute tour distance
    total_distance = 0.0
    for i in range(len(tour)):
        city1, city2 = tour[i], tour[(i+1) % len(tour)]
        x1, y1 = coordinates[city1]
        x2, y2 = coordinates[city2]
        total_distance += distance((x1, y1), (x2, y2))
    
    is_valid = check_tour_validity(tour, len(coordinates))
    
    # Log results
    logger.log_results(tour, total_distance, final_energy, is_valid)
    
    # Display on terminal
    print(f"Final energy: {final_energy:.6f}")
    print(f"Tour valid: {'✓ YES' if is_valid else '✗ NO'}")
    print(f"Decoded tour: {tour}")
    print(f"Total distance: {total_distance:.6f}")
    print(f"\n All results saved to: {experiment_dir}")
    print("="*60)

if __name__ == '__main__':
    main()
