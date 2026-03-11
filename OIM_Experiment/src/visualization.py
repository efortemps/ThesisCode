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
    Snapshot visualizer for the OIM Max-Cut network.

    5-panel layout per snapshot:
      1. Soft-spin trajectories  cos(theta_i)(t) — one line per node
      2. Phase trajectories      theta_i(t)      — one line per node
      3. Weight matrix W (heatmap)
      4. Graph partition drawing — circular layout, cut edges in green
      5. Lyapunov energy and continuous cut value over time (twin-axis)

    Line colour:
      blue → node converging to theta = 0   (spin +1, partition A)
      red  → node converging to theta = pi  (spin -1, partition B)
    """

    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, 'images')

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        self.snapshots      = []
        self.energy_history = []
        self.cut_history    = []

        self.activation_history = []
        self.phase_history      = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_snapshot(self, net_state, net_config):
        """
        Store one snapshot and append to the running trajectory histories.
        Called every `freq` steps from the runner.
        """
        self.snapshots.append({'state': net_state, 'config': net_config})
        self.energy_history.append(net_state.get('energy',    None))
        self.cut_history.append(   net_state.get('cut_value', None))

        self.activation_history.append(list(net_state['activations']))
        self.phase_history.append(     list(net_state['phases']))

    def generate_images(self, W):
        print(f'\nGenerating {len(self.snapshots)} images...')
        for idx, snapshot in enumerate(self.snapshots):
            self._plot_snapshot(idx, snapshot, W)
            print(f'  Image {idx + 1}/{len(self.snapshots)}', end='\r')
        print('\nImages generated!')

    def generate_video(self, fps=10, video_name='oim_maxcut.mp4'):
        print('\nGenerating video with ffmpeg...')
        video_path    = os.path.join(self.output_dir, video_name)
        image_pattern = os.path.join(self.images_dir, 'img%d.png')

        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-r', str(fps),
            '-i', image_pattern,
            '-vframes', str(len(self.snapshots)),
            '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            video_path,
        ]

        try:
            sp.run(cmd, check=True)
            if Path(video_path).is_file():
                print(f'✓ Video created: {video_path}')
                shutil.rmtree(self.images_dir)
            else:
                print('✗ Video creation failed')
        except FileNotFoundError:
            print('✗ ffmpeg not found.  sudo apt-get install ffmpeg')
        except sp.CalledProcessError as e:
            print(f'✗ ffmpeg error: {e}')

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _plot_snapshot(self, index, snapshot, W):
        """Create one 5-panel figure and save to images_dir/img{index}.png."""
        net_state  = snapshot['state']
        net_config = snapshot['config']
        n = net_config['n_nodes']

        # Current soft spins cos(theta) — used to colour each trajectory
        current_s = np.array(net_state['activations'])

        # Trajectory arrays up to this snapshot: shape (T, n)
        s_traj     = np.array(self.activation_history[:index + 1])  # cos(theta)
        theta_traj = np.array(self.phase_history[:index + 1])       # theta (rad)
        T      = s_traj.shape[0]
        t_axis = np.arange(T)

        # Blue → phase converging to 0 (spin +1), red → pi (spin -1)
        node_colors = ['tab:blue' if s >= 0 else 'tab:red' for s in current_s]

        fig = plt.figure(figsize=(38, 8), dpi=50)
        plt.suptitle(
            f'n={n}  K={net_config["K"]}  Ks={net_config["Ks"]}  '
            f'coupling={net_config["coupling"]}  '
            f'timestep={net_config["timestep"]}  Snapshot {index}'
        )

        # ── 1. Soft-spin trajectories  cos(theta_i) ──────────────────────
        ax1 = plt.subplot(1, 4, 1)
        for node in range(n):
            ax1.plot(t_axis, s_traj[:, node],
                     color=node_colors[node], linewidth=1.2, alpha=0.85)
            ax1.scatter([index], [s_traj[-1, node]],
                        color=node_colors[node], s=30, zorder=5)

        ax1.axhline( 1.0, color='k', linewidth=0.7, linestyle='--', alpha=0.5,
                     label='sigma = +1')
        ax1.axhline(-1.0, color='k', linewidth=0.7, linestyle='--', alpha=0.5,
                     label='sigma = -1')
        ax1.axhline( 0.0, color='k', linewidth=0.4, linestyle=':',  alpha=0.4)
        ax1.set_ylim(-1.15, 1.15)
        ax1.set_xlabel('Snapshot index')
        ax1.set_ylabel('s_i = cos(theta_i)')
        ax1.set_title('Soft-Spin Trajectories')
        ax1.grid(True, alpha=0.25)

        # ── 2. Phase trajectories  theta_i ───────────────────────────────
        ax2 = plt.subplot(1, 4, 2)
        for node in range(n):
            ax2.plot(t_axis, theta_traj[:, node],
                     color=node_colors[node], linewidth=1.2, alpha=0.85)
            ax2.scatter([index], [theta_traj[-1, node]],
                        color=node_colors[node], s=30, zorder=5)

        # Reference lines at the two bi-stable fixed points: 0 and pi
        ax2.axhline( 0.0,      color='k', linewidth=0.7, linestyle='--', alpha=0.5,
                     label='theta = 0  (spin +1)')
        ax2.axhline( np.pi,    color='k', linewidth=0.7, linestyle='--', alpha=0.5,
                     label='theta = pi (spin -1)')
        ax2.axhline(-np.pi,    color='k', linewidth=0.4, linestyle=':',  alpha=0.3)
        ax2.axhline( 2*np.pi,  color='k', linewidth=0.4, linestyle=':',  alpha=0.3)
        ax2.set_ylim(-2*np.pi, 2*np.pi)
        ax2.set_xlabel('Snapshot index')
        ax2.set_ylabel('theta_i  (rad)')
        ax2.set_title('Phase Trajectories')
        ax2.grid(True, alpha=0.25)


        # ── 3. Graph partition ────────────────────────────────────────────
        ax4 = plt.subplot(1, 4, 3)
        self._draw_partition(ax4, W, current_s, n)
        ax4.set_title('Graph Partition\n(blue = A, red = B, green = cut edges)')
        ax4.axis('off')

        # ── 4. Lyapunov energy + cut value (twin-axis) ────────────────────
        ax5 = plt.subplot(1, 4, 4)
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
        ax5.set_ylabel('Lyapunov energy L', color='blue')
        ax5.set_title(
            f'Energy & Cut\n'
            f'L={net_state["energy"]:.4f}  Cut={net_state["cut_value"]:.2f}'
        )
        ax5.grid(True, alpha=0.3)

        plt.savefig(os.path.join(self.images_dir, f'img{index}.png'),
                    bbox_inches=None)
        plt.close()

    def _draw_partition(self, ax, W, activations, n):
        """
        Circular graph layout.
        Blue nodes → partition A (cos(theta) >= 0), red → partition B.
        Green edges → cut edges, light grey → within-partition edges.
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
                        linewidth=1.5 if is_cut else 0.5, zorder=1,
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


# =============================================================================
# MuSweepVisualizer — mu-sweep experiment
# =============================================================================

class MuSweepVisualizer:
    """
    Generates a 3-panel summary figure for a mu-sweep experiment.

    Panel 1 — Binarization residual  max_i |sin(theta_i)|  vs mu.
              Green shading marks the binarized region (residual < tol).
    Panel 2 — Final phases theta_i / pi  vs mu  (one line per node).
              Dashed reference lines at 0 and 1 (= pi).
    Panel 3 — Final binary cut value  vs mu.
              Plus minimum Hessian eigenvalue (twin axis) to link
              binarization to the curvature of the energy landscape.
    """

    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_sweep_plot(self, results, W, dataset_name="graph", tol=1e-2):
        """
        Parameters
        ----------
        results      : dict returned by runner_mu_sweep.main()
        W            : weight matrix (for graph metadata)
        dataset_name : string used in the figure title
        tol          : binarization threshold drawn as a reference line

        Returns
        -------
        path : str  path to the saved PNG
        """
        mu_values   = np.array(results["mu"])
        residuals   = np.array(results["binarization_residual"])
        binarized   = np.array(results["is_binarized"])
        phases_arr  = np.array(results["phases"])          # (n_mu, n_nodes)
        cut_values  = np.array(results["binary_cut_value"])
        eigs        = results["hessian_eigenvalues"]       # list of lists
        min_eigs    = np.array([min(e) for e in eigs])

        n_nodes = phases_arr.shape[1]
        n_edges = int(np.sum(W > 0) / 2)

        NODE_COLORS = plt.get_cmap('tab10')(np.linspace(0, 0.9, n_nodes))
        fig, axes = plt.subplots(3, 1, figsize=(11, 13))
        fig.suptitle(
            f"OIM $\\mu$-sweep  —  {dataset_name}  "
            f"({n_nodes} nodes, {n_edges} edges)",
            fontsize=13, fontweight="bold", y=0.98
        )

        # ── Panel 1: binarization residual ──────────────────────────
        ax = axes[0]
        ax.plot(mu_values, residuals, color="#2563EB", lw=2,
                label=r"$\max_i |\sin(\theta_i)|$")
        ax.axhline(tol, color="crimson", ls="--", lw=1.3,
                   label=f"tol = {tol}")
        ax.fill_between(mu_values, 0, residuals.max(),
                        where=binarized, color="#22C55E", alpha=0.18,
                        label="binarized region")
        # mark first binarized mu
        if binarized.any():
            mu_first = mu_values[binarized][0]
            ax.axvline(mu_first, color="#16A34A", ls=":", lw=1.5,
                       label=f"first binarized $\\mu$={mu_first:.2f}")
        ax.set_ylabel("Binarization residual", fontsize=10)
        ax.set_title(r"Binarization residual vs $\mu$", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(mu_values[0], mu_values[-1])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.25)

        # ── Panel 2: final phases ────────────────────────────────────
        ax = axes[1]
        for i in range(n_nodes):
            ax.plot(mu_values, phases_arr[:, i] / np.pi,
                    color=NODE_COLORS[i], lw=1.6,
                    label=f"$\\theta_{{{i+1}}}/\\pi$")
        ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.7)
        ax.axhline(1, color="gray", ls=":", lw=1, alpha=0.7,
                   label="target $\\{0, \\pi\\}$")
        if binarized.any():
            ax.axvline(mu_values[binarized][0], color="#16A34A",
                       ls=":", lw=1.5)
        ax.set_ylabel(r"$\theta_i / \pi$", fontsize=10)
        ax.set_title(r"Final phases vs $\mu$", fontsize=10)
        ncol = max(1, n_nodes // 6)
        ax.legend(fontsize=7, loc="upper right", ncol=ncol)
        ax.set_xlim(mu_values[0], mu_values[-1])
        ax.grid(True, alpha=0.25)

        # ── Panel 3: binary cut value + min Hessian eigenvalue ──────
        ax = axes[2]
        ax.plot(mu_values, cut_values, color="#7C3AED", lw=2,
                label="Binary cut value")
        if binarized.any():
            ax.axvline(mu_values[binarized][0], color="#16A34A",
                       ls=":", lw=1.5)
        ax.set_ylabel("Binary cut value", fontsize=10, color="#7C3AED")
        ax.tick_params(axis="y", labelcolor="#7C3AED")

        ax2 = ax.twinx()
        ax2.plot(mu_values, min_eigs, color="#EA580C", lw=1.5,
                 ls="--", label=r"$\lambda_{\min}(H_\mu)$")
        ax2.axhline(0, color="#EA580C", ls=":", lw=0.8, alpha=0.5)
        ax2.set_ylabel(r"Min Hessian eigenvalue $\lambda_{\min}$",
                       fontsize=10, color="#EA580C")
        ax2.tick_params(axis="y", labelcolor="#EA580C")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=8, loc="lower right")
        ax.set_xlabel(r"$\mu$", fontsize=11)
        ax.set_title(r"Binary cut value and min Hessian eigenvalue vs $\mu$",
                     fontsize=10)
        ax.set_xlim(mu_values[0], mu_values[-1])
        ax.grid(True, alpha=0.25)

        plt.tight_layout(rect=(0, 0, 1, 0.97))
        path = os.path.join(self.output_dir, "mu_sweep.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path
