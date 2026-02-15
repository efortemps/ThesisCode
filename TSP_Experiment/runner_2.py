import argparse
from Hopfield_2 import HopfieldNet
from inputProcessing import read_data, distance_matrix, normalize, normalize_cords, distance
from visualization import SimpleVisualizer

def get_args():
    parser = argparse.ArgumentParser(description='Hopfield-Tank TSP Solver')
    parser.add_argument('--steps', type=int, default=5000, help='Number of steps')
    parser.add_argument('--freq', type=int, default=50, help='Snapshot frequency')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--data', type=str, default='data/cities.txt', help='Path to city data file')
    parser.add_argument('--video', action='store_true', help='Generate video')
    parser.add_argument('--fps', type=int, default=10, help='Video frame rate')
    return parser.parse_args()

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
    
    # Load and prepare data
    print("\nLoading city data...")
    coordinates = read_data(args.data)
    # coordinates_normalized = normalize_cords(coordinates)
    distances = normalize(distance_matrix(coordinates))
    print(f"✓ Loaded {len(coordinates)} cities")
    
    # Initialize network
    print("\nInitializing Hopfield network...")
    net = HopfieldNet(distances, args.seed, args.sigma, args.method)
    print(f"✓ Network initialized ({len(distances)}×{len(distances)} neurons)")
    
    # Initialize visualizer
    visualizer = SimpleVisualizer(output_dir='output')
    
    # Run simulation
    print("\nRunning simulation...")
    for step in range(args.steps):
        net.update()
        
        # Save snapshot
        if step % args.freq == 0:
            visualizer.add_snapshot(
                net.get_net_state(),
                net.get_net_configuration()
            )
        
        # Progress display
        if step % 500 == 0:
            print(f"  Step {step}/{args.steps}", end='\r')
    
    print(f"\n✓ Simulation complete ({args.steps} steps)")

    # Generate visualizations
    # If coordinates are larger than 1, use coordinates_normalized
    visualizer.generate_images(coordinates, distances)
    
    if args.video:
        visualizer.generate_video(fps=args.fps)
    
    # Display final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    activations = net.activations()
    print(f"Activation matrix shape: {activations.shape}")
    print(f"Max activation: {activations.max():.3f}")
    print(f"Min activation: {activations.min():.3f}")
    
    # Decode tour
    tour = []
    for pos in range(len(activations[0])):
        city = activations[:, pos].argmax()
        tour.append(city)
    print(f"Decoded tour: {tour}")
    total_distance = 0
    for i in range(len(tour)): 
        city1 = tour[i]
        city2 = tour[(i+1) % len(tour)]
        x1, y1 = coordinates[city1]
        x2, y2 = coordinates[city2]
        dist = distance((x1, y1), (x2, y2))
        total_distance += dist

    print(f"Total distance of the tour is : {total_distance}")
if __name__ == '__main__':
    main()
