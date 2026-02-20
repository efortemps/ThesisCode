import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import subprocess as sp
import os
from pathlib import Path
import shutil


class MaxCutVisualizer:
    """
    Snapshot visualizer for the Hopfield-Tank Max-Cut network.
    Mirrors SimpleVisualizer from the TSP experiment.

    5-panel layout per snapshot:
        1. Spin activations  s_i = tanh(u_i/u0) ∈ (-1,+1), coloured by side
        2. Membrane potentials u_i per node
        3. Weight matrix W (heatmap)
        4. Graph partition drawing — circular layout, cut edges in green
        5. Energy and continuous cut value over time (twin-axis)
    """

    def __init__(self, output_dir='output'):
        self.output_dir  = output_dir
        self.images_dir  = os.path.join(output_dir, 'images')

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        self.snapshots      = []
        self.energy_history = []  
        self.cut_history    = []   

    # ------------------------------------------------------------------
    # Public interface (mirrors TSP SimpleVisualizer)
    # ------------------------------------------------------------------

    def add_snapshot(self, net_state, net_config):
        """
        Store one snapshot.
        Mirrors TSP add_snapshot(net_state, net_config).
        """
        self.snapshots.append({'state': net_state, 'config': net_config})
        self.energy_history.append(net_state.get('energy',    None))
        self.cut_history.append(   net_state.get('cut_value', None))

    def generate_images(self, W):
        """
        Generate all snapshot images.
        Replaces TSP generate_images(coordinates, distances):
        Max-Cut needs only the weight matrix W, not city coordinates.
        """
        print(f"\nGenerating {len(self.snapshots)} images...")
        for idx, snapshot in enumerate(self.snapshots):
            self._plot_snapshot(idx, snapshot, W)
            print(f"Image {idx+1}/{len(self.snapshots)}", end='\r')
        print("\nImages generated!")

    def generate_video(self, fps=10, video_name='hopfield_maxcut.mp4'):
        """Generate video from images using ffmpeg (mirrors TSP generate_video)."""
        print("\nGenerating video with ffmpeg...")
        video_path    = os.path.join(self.output_dir, video_name)
        image_pattern = os.path.join(self.images_dir, 'img%d.png')

        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-r',      str(fps),
            '-i',      image_pattern,
            '-vframes', str(len(self.snapshots)),
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2'
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
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
            print("✗ ffmpeg not found.  sudo apt-get install ffmpeg")
        except sp.CalledProcessError as e:
            print(f"✗ ffmpeg error: {e}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _plot_snapshot(self, index, snapshot, W):
        """Create one 5-panel figure and save to images_dir/img{index}.png."""
        net_state  = snapshot['state']
        net_config = snapshot['config']
        n          = net_config['n_nodes']

        activations = np.array(net_state['activations'])   # s_i ∈ (-1,+1)
        inputs      = np.array(net_state['inputs'])         # u_i

        colors      = ['tab:blue' if s >= 0 else 'tab:red' for s in activations]
        node_idx    = np.arange(n)

        fig = plt.figure(figsize=(38, 8), dpi=50)
        plt.suptitle(
            f"n={n}  u0={net_config['u0']}  tau={net_config['tau']}  "
            f"timestep={net_config['timestep']}  Snapshot {index}"
        )

        # ---- 1. Spin activations ----
        ax1 = plt.subplot(1, 5, 1)
        ax1.bar(node_idx, activations, color=colors)
        ax1.axhline(0, color='k', linewidth=0.8, linestyle='--')
        ax1.set_ylim(-1.1, 1.1)
        ax1.set_xlabel('Node')
        ax1.set_ylabel('s_i = tanh(u_i / u0)')
        ax1.set_title('Spin Activations')

        # ---- 2. Membrane potentials ----
        ax2 = plt.subplot(1, 5, 2)
        ax2.bar(node_idx, inputs, color=colors)
        ax2.axhline(0, color='k', linewidth=0.8, linestyle='--')
        ax2.set_xlabel('Node')
        ax2.set_ylabel('u_i')
        ax2.set_title('Membrane Potentials')

        # ---- 3. Weight matrix heatmap ----
        ax3 = plt.subplot(1, 5, 3)
        im = ax3.imshow(W, cmap='plasma', interpolation='nearest')
        plt.colorbar(im, ax=ax3)
        ax3.set_title('Weight Matrix W')
        ax3.set_xlabel('Node j')
        ax3.set_ylabel('Node i')

        # ---- 4. Graph partition ----
        ax4 = plt.subplot(1, 5, 4)
        self._draw_partition(ax4, W, activations, n)
        ax4.set_title('Graph Partition\n(blue = A, red = B, green = cut edges)')
        ax4.axis('off')

        # ---- 5. Energy + cut evolution (twin-axis) ----
        ax5 = plt.subplot(1, 5, 5)
        valid_e = [e for e in self.energy_history[:index+1] if e is not None]
        valid_c = [c for c in self.cut_history[:index+1]    if c is not None]
        if valid_e:
            t = list(range(len(valid_e)))
            ax5.plot(t, valid_e, 'b-', linewidth=2, label='Energy')
            ax5.scatter([index], [valid_e[-1]], c='blue', s=60, zorder=5)
            ax5_r = ax5.twinx()
            ax5_r.plot(t, valid_c, color='darkorange', linewidth=2, label='Cut')
            ax5_r.scatter([index], [valid_c[-1]], c='darkorange', s=60, zorder=5)
            ax5_r.set_ylabel('Cut value', color='darkorange')
        ax5.set_xlabel('Snapshot index')
        ax5.set_ylabel('Energy E', color='blue')
        ax5.set_title(
            f"Energy & Cut\n"
            f"E={net_state['energy']:.4f}   Cut={net_state['cut_value']:.2f}"
        )
        ax5.grid(True, alpha=0.3)

        plt.savefig(os.path.join(self.images_dir, f'img{index}.png'))
        plt.close()

    def _draw_partition(self, ax, W, activations, n):
        """
        Draw the graph on a circular layout.
        Nodes coloured blue (side A, s>=0) or red (side B, s<0).
        Cut edges drawn in green; within-partition edges in light grey.
        """
        angles    = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos       = np.column_stack([np.cos(angles), np.sin(angles)])
        partition = np.sign(activations)
        partition[partition == 0] = 1

        for i in range(n):
            for j in range(i + 1, n):
                if W[i, j] > 0:
                    is_cut = (partition[i] != partition[j])
                    ax.plot(
                        [pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        color='green' if is_cut else 'lightgrey',
                        linewidth=1.5 if is_cut else 0.5, zorder=1
                    )

        node_colors = ['tab:blue' if p == 1 else 'tab:red' for p in partition]
        for i in range(n):
            ax.scatter(pos[i, 0], pos[i, 1], s=200,
                       c=node_colors[i], edgecolors='black',
                       linewidths=0.8, zorder=2)
            ax.text(pos[i, 0] * 1.18, pos[i, 1] * 1.18, str(i),
                    ha='center', va='center', fontsize=7)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
