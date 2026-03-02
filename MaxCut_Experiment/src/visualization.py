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

    5-panel layout per snapshot:
        1. Spin activation trajectories  s_i(t) = tanh(u_i/u0) — one line per node
        2. Membrane potential trajectories  u_i(t)              — one line per node
        3. Weight matrix W (heatmap)
        4. Graph partition drawing — circular layout, cut edges in green
        5. Energy and continuous cut value over time (twin-axis)

    Panels 1 & 2 replace the previous bar plots with trajectory lines that
    show how each node evolves from initialisation toward ±1 (or ±∞ for u).
    Line colour: blue  → node converging to s = +1  (side A)
                 red   → node converging to s = -1  (side B)
    """

    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, 'images')

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        self.snapshots = []
        self.energy_history    = []
        self.cut_history       = []

        # ── NEW: full per-snapshot history for trajectory plots ──────────────
        # activation_history[t] = list of n activation values at snapshot t
        # input_history[t]      = list of n membrane potential values at snapshot t
        self.activation_history = []
        self.input_history      = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_snapshot(self, net_state, net_config):
        """
        Store one snapshot and append to the running trajectory histories.
        The two new history lists grow by one row per call, enabling
        trajectory plots that cover all snapshots up to the current one.
        """
        self.snapshots.append({'state': net_state, 'config': net_config})
        self.energy_history.append(net_state.get('energy',    None))
        self.cut_history.append(   net_state.get('cut_value', None))

        # Accumulate per-node trajectories  ← NEW
        self.activation_history.append(list(net_state['activations']))
        self.input_history.append(      list(net_state['inputs']))

    def generate_images(self, W):
        print(f"\nGenerating {len(self.snapshots)} images...")
        for idx, snapshot in enumerate(self.snapshots):
            self._plot_snapshot(idx, snapshot, W)
            print(f"Image {idx+1}/{len(self.snapshots)}", end='\r')
        print("\nImages generated!")

    def generate_video(self, fps=10, video_name='hopfield_maxcut.mp4'):
        print("\nGenerating video with ffmpeg...")
        video_path    = os.path.join(self.output_dir, video_name)
        image_pattern = os.path.join(self.images_dir, 'img%d.png')

        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-r',       str(fps),
            '-i',       image_pattern,
            '-vframes', str(len(self.snapshots)),
            '-vf',      'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            '-vcodec',  'libx264',
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

        # Current activations — used to assign final colour to each trajectory
        current_s = np.array(net_state['activations'])

        # Trajectory arrays up to this snapshot: shape (index+1, n)
        s_traj = np.array(self.activation_history[:index + 1])   # (T, n)
        u_traj = np.array(self.input_history[:index + 1])         # (T, n)
        T      = s_traj.shape[0]
        t_axis = np.arange(T)   # snapshot indices 0 … index

        # Colour each node by its CURRENT sign so the partition is readable
        node_colors = ['tab:blue' if s >= 0 else 'tab:red' for s in current_s]

        fig = plt.figure(figsize=(38, 8), dpi=50)
        plt.suptitle(
            f"n={n}  u0={net_config['u0']}  tau={net_config['tau']}  "
            f"timestep={net_config['timestep']}  Snapshot {index}"
        )

        # ── 1. Spin activation trajectories ──────────────────────────────────
        ax1 = plt.subplot(1, 5, 1)
        for node in range(n):
            ax1.plot(t_axis, s_traj[:, node],
                     color=node_colors[node], linewidth=1.2, alpha=0.85)
            # Dot at current position
            ax1.scatter([index], [s_traj[-1, node]],
                        color=node_colors[node], s=30, zorder=5)

        ax1.axhline( 1.0, color='k', linewidth=0.7, linestyle='--', alpha=0.5)
        ax1.axhline(-1.0, color='k', linewidth=0.7, linestyle='--', alpha=0.5)
        ax1.axhline( 0.0, color='k', linewidth=0.4, linestyle=':',  alpha=0.4)
        ax1.set_ylim(-1.15, 1.15)
        ax1.set_xlabel('Snapshot index')
        ax1.set_ylabel('s_i = tanh(u_i / u0)')
        ax1.set_title('Spin Activation Trajectories')
        ax1.grid(True, alpha=0.25)

        # ── 2. Membrane potential trajectories ───────────────────────────────
        ax2 = plt.subplot(1, 5, 2)
        for node in range(n):
            ax2.plot(t_axis, u_traj[:, node],
                     color=node_colors[node], linewidth=1.2, alpha=0.85)
            ax2.scatter([index], [u_traj[-1, node]],
                        color=node_colors[node], s=30, zorder=5)

        ax2.axhline(0.0, color='k', linewidth=0.4, linestyle=':', alpha=0.4)
        ax2.set_xlabel('Snapshot index')
        ax2.set_ylabel('u_i')
        ax2.set_title('Membrane Potential Trajectories')
        ax2.grid(True, alpha=0.25)

        # ── 3. Weight matrix heatmap ──────────────────────────────────────────
        ax3 = plt.subplot(1, 5, 3)
        im  = ax3.imshow(W, cmap='plasma', interpolation='nearest')
        plt.colorbar(im, ax=ax3)
        ax3.set_title('Weight Matrix W')
        ax3.set_xlabel('Node j')
        ax3.set_ylabel('Node i')

        # ── 4. Graph partition ────────────────────────────────────────────────
        ax4 = plt.subplot(1, 5, 4)
        self._draw_partition(ax4, W, current_s, n)
        ax4.set_title('Graph Partition\n(blue = A, red = B, green = cut edges)')
        ax4.axis('off')

        # ── 5. Energy + cut evolution (twin-axis) ─────────────────────────────
        ax5    = plt.subplot(1, 5, 5)
        valid_e = [e for e in self.energy_history[:index + 1] if e is not None]
        valid_c = [c for c in self.cut_history[:index + 1]    if c is not None]
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
            f"E={net_state['energy']:.4f}  Cut={net_state['cut_value']:.2f}"
        )
        ax5.grid(True, alpha=0.3)

        plt.savefig(os.path.join(self.images_dir, f'img{index}.png'),
                    bbox_inches=None)
        plt.close()

    def _draw_partition(self, ax, W, activations, n):
        """
        Circular graph layout.
        Blue nodes → side A (s >= 0), red → side B (s < 0).
        Green edges → cut, light grey → within partition.
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
