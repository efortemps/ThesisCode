import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt
import subprocess as sp
import os
from pathlib import Path
import shutil


class SimpleVisualizer:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, 'images')
        
        # Create directories
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        
        self.snapshots = []
        self.energy_history = []  # NEW: Track energy over time

    def add_snapshot(self, net_state, net_config):
        """Store a snapshot of the network state"""
        # Extract energy from net_state if available, else compute it separately
        energy = net_state.get('energy', None)
        
        self.snapshots.append({
            'state': net_state,
            'config': net_config
        })
        self.energy_history.append(energy)  # NEW: Store energy value

    def generate_images(self, coordinates, distances):
        """Generate all images from stored snapshots"""
        print(f"\nGenerating {len(self.snapshots)} images...")
        for idx, snapshot in enumerate(self.snapshots):
            self._plot_snapshot(idx, snapshot, coordinates, distances)
            print(f"Image {idx+1}/{len(self.snapshots)}", end='\r')
        print("\nImages generated!")

    def _plot_snapshot(self, index, snapshot, coordinates, distances):
        """Create a 5-panel visualization for one snapshot"""
        net_state = snapshot['state']
        net_config = snapshot['config']
        
        # Create figure with 5 subplots (NEW: changed from 4 to 5)
        fig = plt.figure(figsize=(37.5, 10), dpi=50)  # Increased width to fit 5 plots
        
        # Title with all parameters
        title = f"a={net_config['a']} b={net_config['b']} c={net_config['c']} d={net_config['d']} " \
                f"u0={net_config['u0']} timestep={net_config['timestep']} " \
                f"Snapshot {index}"
        plt.suptitle(title)

        # 1. Activations heatmap
        plt.subplot(1, 5, 1)
        plt.imshow(net_state['activations'], cmap='hot', vmin=0, vmax=1, interpolation='nearest')
        plt.title('Activations')
        plt.colorbar()

        # 2. Inputs heatmap
        plt.subplot(1, 5, 2)
        plt.imshow(net_state['inputs'], cmap='coolwarm', vmin=-0.075, vmax=0.075, interpolation='nearest')
        plt.title('Neuron Inputs')
        plt.colorbar()

        # 3. Distance matrix
        plt.subplot(1, 5, 3)
        plt.imshow(distances, cmap='plasma', vmin=0, vmax=1, interpolation='nearest')
        plt.title('Distance Matrix')
        plt.colorbar()

        # 4. Tour graph
        plt.subplot(1, 5, 4)
        tour_points = self._decode_tour(net_state['activations'], coordinates)
        if tour_points and len(tour_points) > 1:
            # Draw edges between consecutive cities
            for i in range(len(tour_points) - 1):
                p1, p2 = tour_points[i], tour_points[i+1]
                plt.arrow(p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1],
                        head_width=0.0, head_length=0.0, width=0.0001, fc='k', ec='k')
            
            # NEW: Draw the closing edge from last city back to first city
            if len(tour_points) > 2:  # Only close if we have at least 3 cities
                p_last = tour_points[-1]
                p_first = tour_points[0]
                plt.arrow(p_last[0], p_last[1], p_first[0]-p_last[0], p_first[1]-p_last[1],
                        head_width=0.0, head_length=0.0, width=0.0001, fc='r', ec='r')  # Red for visibility
            
            xs = [p[0] for p in tour_points]
            ys = [p[1] for p in tour_points]
            plt.xlim(min(xs), max(xs))
            plt.ylim(min(ys), max(ys))
        plt.title('Tour Graph')

        # 5. Energy evolution (NEW)
        plt.subplot(1, 5, 5)
        energy_up_to_now = [e for e in self.energy_history[:index+1] if e is not None]
        if energy_up_to_now:
            indices = list(range(len(energy_up_to_now)))
            plt.plot(indices, energy_up_to_now, 'b-', linewidth=2)
            plt.scatter([index], [energy_up_to_now[-1]], c='red', s=100, zorder=5)  # Current point
            plt.xlabel('Snapshot Index')
            plt.ylabel('Energy E')
            plt.title(f'Energy Evolution\nCurrent E = {energy_up_to_now[-1]:.4f}')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No energy data', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Energy Evolution')

        # Save image
        image_path = os.path.join(self.images_dir, f'img{index}.png')
        plt.savefig(image_path, bbox_inches='tight')
        plt.close()

    def _decode_tour(self, activations, coordinates):
        """Decode tour from activation matrix"""
        points = []
        n = len(activations)
        for pos in range(n):
            for city in range(n):
                if activations[city][pos] > 0.6:
                    points.append(coordinates[city])
                    break
        return points

    def generate_video(self, fps=10, video_name='hopfield_tsp.mp4'):
        """Generate video from images using ffmpeg"""
        print("\nGenerating video with ffmpeg...")
        video_path = os.path.join(self.output_dir, video_name)
        image_pattern = os.path.join(self.images_dir, 'img%d.png')
        
        # ffmpeg command
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-loglevel', 'error',  # Only show errors
            '-r', str(fps),  # Frame rate
            '-i', image_pattern,  # Input pattern
            '-vframes', str(len(self.snapshots)),  # Number of frames
            '-vcodec', 'libx264',  # Codec
            '-pix_fmt', 'yuv420p',  # Pixel format (for compatibility)
            video_path
        ]
        
        try:
            sp.run(cmd, check=True)
            if Path(video_path).is_file():
                print(f"✓ Video created: {video_path}")
                shutil.rmtree(self.images_dir)
            else:
                print("✗ Video creation failed")
        except FileNotFoundError:
            print("✗ ffmpeg not found. Please install ffmpeg:")
            print("  - Ubuntu/Debian: sudo apt-get install ffmpeg")
            print("  - macOS: brew install ffmpeg")
            print("  - Windows: Download from https://ffmpeg.org/download.html")
        except sp.CalledProcessError as e:
            print(f"✗ ffmpeg error: {e}")
