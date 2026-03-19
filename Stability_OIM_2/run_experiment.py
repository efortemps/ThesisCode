"""
run_experiment.py
=================
Reproduces the three experiments from:

    Bashar, Lin & Shukla, "Stability of Oscillator Ising Machines:
    Not All Solutions Are Created Equal."

Usage
-----
    python run_experiment.py --graph data/8node.txt [OPTIONS]

    # Run only Experiment C (interactive slider):
    python run_experiment.py --graph data/3node.txt --experiments C

    # Full sweep with 100 trials and a finer mu grid:
    python run_experiment.py --graph data/8node.txt --trials 100 --mu-steps 50

Run with -h / --help for the full option list.

Experiments
-----------
A — λ_L vs injection strength μ=2Ks for *all* 2^N spin configurations
    (reproduces Fig. 2 of the paper).

B — Final-energy histograms over N_trials random initialisations at two
    user-specified μ values (reproduces Fig. 4d–e).

C — Interactive slider: λ_L min/max vs energy H; drag the μ slider to
    see stability change in real time (interactive version of Fig. 3).
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from matplotlib.widgets import Slider

# ---------------------------------------------------------------------------
# Allow running both as a flat script and from within a package
# ---------------------------------------------------------------------------
try:
    from OIM_Stability_1 import OIM_Maxcut
    from read_graphs import read_graph_to_J, graph_info
except ModuleNotFoundError:
    # Fallback: try package-style imports (original layout)
    from Stability_OIM_2.OIM_Stability_1 import OIM_Maxcut  # type: ignore
    from Stability_OIM_2.read_graphs import read_graph_to_J, graph_info  # type: ignore


# ===========================================================================
# Helpers
# ===========================================================================

def all_phase_configs(N: int) -> List[np.ndarray]:
    """All 2^N binary phase configurations {0, π}^N."""
    return [np.array(cfg, dtype=float) * np.pi for cfg in product([0, 1], repeat=N)]


def scan_configs(
    oim: OIM_Maxcut,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute energy and largest Lyapunov exponent for every binary
    phase configuration.

    Returns
    -------
    energies : (2^N,) array
    lambdas  : (2^N,) array
    """
    configs = all_phase_configs(oim.N)
    energies = np.array([oim.energy(phi) for phi in configs])
    lambdas  = np.array([oim.largest_lyapunov(phi) for phi in configs])
    return energies, lambdas


def lambda_vs_energy(
    E: np.ndarray,
    lam: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each distinct energy level, return the min and max λ_L.

    Returns
    -------
    unique_E  : sorted distinct energies
    lam_min   : minimum λ_L at each energy
    lam_max   : maximum λ_L at each energy
    """
    unique_E = np.array(sorted(set(np.round(E, 8))))
    lam_min, lam_max = [], []
    for e in unique_E:
        mask = np.isclose(E, e)
        lam_min.append(lam[mask].min())
        lam_max.append(lam[mask].max())
    return unique_E, np.array(lam_min), np.array(lam_max)


def global_min_energy(energies: np.ndarray) -> float:
    """Return the ground-state (minimum) energy."""
    return float(energies.min())


def random_phi0(N: int) -> np.ndarray:
    """Uniform random initial phases in [0, 2π)."""
    return np.random.uniform(0.0, 2.0 * np.pi, size=N)


def binarize(phi: np.ndarray) -> np.ndarray:
    """Snap continuous phases to the nearest of 0 or π."""
    return np.where(phi % (2.0 * np.pi) < np.pi, 0.0, np.pi)


def run_trials(
    oim: OIM_Maxcut,
    n_trials: int = 50,
    t_span: Tuple[float, float] = (0.0, 50.0),
    n_points: int = 500,
) -> np.ndarray:
    """
    Run *n_trials* random initialisations; return array of final binarised
    energies.  Trials where integration fails are skipped.
    """
    energies_final = []
    for _ in range(n_trials):
        phi0 = random_phi0(oim.N)
        sol = oim.simulate(phi0, t_span, n_points)
        if sol is None:
            continue
        phi_final = binarize(sol.y[:, -1])
        energies_final.append(oim.energy(phi_final))
    return np.array(energies_final)


# ===========================================================================
# Experiment A  —  λ_L vs μ for all configurations  (Fig. 2 of the paper)
# ===========================================================================

def experiment_A(
    J: np.ndarray,
    K: float,
    mu_values: np.ndarray,
    *,
    warn_large_N: int = 16,
) -> None:
    """
    Scatter-plot the largest Lyapunov exponent for every binary phase
    configuration across a sweep of injection strengths μ = 2·Ks.

    Globally-optimal configurations are highlighted in a different colour
    so the reader can see when they become stable (λ_L crosses zero).
    """
    N = J.shape[0]
    if N > warn_large_N:
        print(
            f"[Exp A] WARNING: N={N} → 2^N={2**N} configurations. "
            "This may be slow.  Consider a smaller graph or reduce --mu-steps."
        )

    print(f"[Exp A] Scanning {len(mu_values)} μ values × {2**N} configs …")

    # ------------------------------------------------------------------
    # Collect (mu, energies, lambdas) for every mu value
    # BUG FIX: the original code left all_results=[] and never populated it
    # before iterating over it for the plot.
    # ------------------------------------------------------------------
    all_results: List[Tuple[float, np.ndarray, np.ndarray]] = []
    for i, mu in enumerate(mu_values):
        Ks = mu / 2.0
        oim = OIM_Maxcut(J, K=K, Ks=Ks)
        E, lam = scan_configs(oim)
        all_results.append((mu, E, lam))
        if (i + 1) % 5 == 0 or (i + 1) == len(mu_values):
            print(f"  μ step {i+1}/{len(mu_values)} done", end="\r", flush=True)
    print()

    # Determine ground-state energy (same for all mu since J doesn't change)
    E0_ref = global_min_energy(all_results[0][1])

    fig, ax = plt.subplots(figsize=(9, 5), num="Experiment A: λ_L vs μ")

    for mu, E, lam in all_results:
        E0 = global_min_energy(E)
        is_global = np.isclose(E, E0)

        # All configurations (non-global) — light blue
        ax.scatter(
            np.full(np.sum(~is_global), mu),
            lam[~is_global],
            s=6, alpha=0.25, color="steelblue", linewidths=0,
        )
        # Globally optimal configurations — highlighted orange
        if is_global.any():
            ax.scatter(
                np.full(np.sum(is_global), mu),
                lam[is_global],
                s=20, alpha=0.85, color="darkorange",
                zorder=3, linewidths=0,
            )

    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.2, label="λ_L = 0")
    ax.set_xlabel(r"Injection strength  $\mu = 2K_s$", fontsize=12)
    ax.set_ylabel(r"Largest Lyapunov exponent  $\lambda_L$", fontsize=12)
    ax.set_title(
        r"Exp A — Evolution of $\lambda_L$ for all spin configurations"
        f"\n(N={N}, K={K})"
    )

    legend_handles = [
        mpatches.Patch(color="steelblue",   label="Sub-optimal configurations"),
        mpatches.Patch(color="darkorange",  label="Globally optimal configurations"),
        mlines.Line2D([0], [0], color="k", linestyle="--", label=r"$\lambda_L = 0$"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)
    fig.tight_layout()
    print("[Exp A] Close the window to proceed.")
    plt.show()


# ===========================================================================
# Experiment B  —  Energy histograms over random trials  (Fig. 4d–e)
# ===========================================================================

def experiment_B(
    J: np.ndarray,
    K: float,
    mu_values: List[float],
    n_trials: int,
    t_span: Tuple[float, float],
    n_points: int,
) -> None:
    """
    For each μ value run *n_trials* random initialisations and histogram
    the resulting binarised energies.

    When μ is too small (Ks < threshold) the phases may not converge to
    {0,π}; those trials are reported separately.
    """
    N = J.shape[0]
    n = len(mu_values)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), num="Experiment B: Energy Histograms")
    if n == 1:
        axes = [axes]

    for ax, mu in zip(axes, mu_values):
        Ks = mu / 2.0
        oim = OIM_Maxcut(J, K=K, Ks=Ks)
        print(f"[Exp B] μ={mu:.2f} (Ks={Ks:.3f}): running {n_trials} trials …")
        energies_final = run_trials(oim, n_trials=n_trials, t_span=t_span, n_points=n_points)

        if len(energies_final) == 0:
            ax.text(0.5, 0.5, "No trials converged", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            ax.set_title(f"$\\mu = {mu}$  ($K_s = {Ks}$)")
            continue

        E_min = energies_final.min()
        E_max = energies_final.max()

        # One bin per integer energy level
        bins = np.arange(E_min - 0.5, E_max + 1.5, 1.0)
        ax.hist(energies_final, bins=bins, color="steelblue", edgecolor="black",
                alpha=0.85, zorder=2)

        # Mark the ground-state energy with a vertical line
        # (compute it from the config scan — fast for small N)
        E_all, _ = scan_configs(oim)
        E_ground = global_min_energy(E_all)
        ax.axvline(E_ground, color="red", linestyle="--", linewidth=1.5,
                   label=f"Ground state H={E_ground:.1f}", zorder=3)

        ax.set_xlabel("Final energy  H", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"$\\mu = {mu}$  ($K_s = {Ks}$)", fontsize=12)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Exp B — Final energy distribution over {n_trials} trials\n"
        f"(N={N}, K={K})",
        fontsize=12,
    )
    fig.tight_layout()
    print("[Exp B] Close the window to proceed.")
    plt.show()


# ===========================================================================
# Experiment C  —  Interactive slider: λ_L min/max vs H  (Fig. 3)
# ===========================================================================

def experiment_C(
    J: np.ndarray,
    K: float,
    mu_init: float,
    mu_min: float,
    mu_max: float,
    *,
    warn_large_N: int = 16,
) -> None:
    """
    Interactive figure with a μ slider.  Dragging the slider re-computes
    λ_L (min and max) per energy level in real time.

    The red dashed box highlights the globally optimal energy level.
    """
    N = J.shape[0]
    if N > warn_large_N:
        print(
            f"[Exp C] WARNING: N={N} → 2^N={2**N} configs per slider move. "
            "Updates may be slow."
        )

    # ------------------------------------------------------------------
    # Build the initial figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(9, 6), num="Experiment C: Interactive Stability (Fig. 3)")
    # Reserve space at the bottom for the slider
    ax = fig.add_axes((0.10, 0.22, 0.85, 0.68))

    def _compute(mu: float):
        Ks = mu / 2.0
        oim = OIM_Maxcut(J, K=K, Ks=Ks)
        E, lam = scan_configs(oim)
        He, lmin, lmax = lambda_vs_energy(E, lam)
        E_ground = global_min_energy(E)
        return He, lmin, lmax, E_ground

    He0, lmin0, lmax0, E_g0 = _compute(mu_init)

    line_max, = ax.plot(He0, lmax0, "o-", color="darkorange",
                        linewidth=1.8, markersize=6, label=r"max $\lambda_L$")
    line_min, = ax.plot(He0, lmin0, "s-", color="steelblue",
                        linewidth=1.8, markersize=6, label=r"min $\lambda_L$")
    zero_line = ax.axhline(0.0, color="k", linestyle="--", linewidth=1.2)

    # Highlight the globally optimal energy level
    def _draw_ground_box(ax, E_ground, lmin_arr, lmax_arr, He_arr):
        """Draw a red dashed rectangle around the globally optimal energy."""
        for artist in ax.patches:
            artist.remove()
        mask = np.isclose(He_arr, E_ground)
        if not mask.any():
            return
        y_lo = lmin_arr[mask].min() - 0.3
        y_hi = lmax_arr[mask].max() + 0.3
        import matplotlib.patches as mpatches
        rect = mpatches.FancyBboxPatch(
            (E_ground - 0.6, y_lo), 1.2, y_hi - y_lo,
            boxstyle="round,pad=0.1",
            linewidth=1.8, edgecolor="red", facecolor="none",
            linestyle="--", zorder=5,
        )
        ax.add_patch(rect)

    _draw_ground_box(ax, E_g0, lmin0, lmax0, He0)

    ax.set_xlabel("Energy  H", fontsize=12)
    ax.set_ylabel(r"Largest Lyapunov exponent  $\lambda_L$", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title(
        f"Exp C — Stability vs Energy  (μ = {mu_init:.2f},  Ks = {mu_init/2:.3f})"
        f"\n(N={N}, K={K})  |  Red box = globally optimal configurations",
        fontsize=11,
    )

    # Auto-scale on first draw
    ax.set_xlim(He0.min() - 1, He0.max() + 1)
    ax.set_ylim(lmin0.min() - 0.8, lmax0.max() + 0.8)

    # ------------------------------------------------------------------
    # Slider
    # ------------------------------------------------------------------
    ax_slider = fig.add_axes((0.15, 0.07, 0.72, 0.035))
    slider = Slider(
        ax_slider,
        r"$\mu = 2K_s$",
        mu_min,
        mu_max,
        valinit=mu_init,
        valstep=0.05,
        color="steelblue",
    )

    # Label showing current Ks value
    ks_text = fig.text(
        0.89, 0.075,
        f"$K_s$={mu_init/2:.3f}",
        fontsize=9, ha="center", va="center",
    )

    def update(val: float) -> None:
        mu_cur = slider.val
        He, lmin, lmax, E_g = _compute(mu_cur)

        line_max.set_xdata(He)
        line_max.set_ydata(lmax)
        line_min.set_xdata(He)
        line_min.set_ydata(lmin)

        # BUG FIX: also update xlim, not just ylim
        ax.set_xlim(He.min() - 1, He.max() + 1)
        ax.set_ylim(lmin.min() - 0.8, lmax.max() + 0.8)
        ax.set_title(
            f"Exp C — Stability vs Energy  (μ = {mu_cur:.2f},  Ks = {mu_cur/2:.3f})"
            f"\n(N={N}, K={K})  |  Red box = globally optimal configurations",
            fontsize=11,
        )

        _draw_ground_box(ax, E_g, lmin, lmax, He)
        ks_text.set_text(f"$K_s$={mu_cur/2:.3f}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    print("[Exp C] Drag the μ slider to explore stability.  Close the window when done.")
    plt.show()


# ===========================================================================
# CLI argument parsing
# ===========================================================================

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "OIM Stability Experiments — reproduce Figs 2, 3, 4 from "
            "Bashar, Lin & Shukla (2022)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required ---
    parser.add_argument(
        "--graph", "-g",
        required=True,
        metavar="FILE",
        help="Path to graph file (edge-list or DIMACS format).",
    )

    # --- OIM parameters ---
    physics = parser.add_argument_group("OIM parameters")
    physics.add_argument("--K",  type=float, default=1.0,
                         help="Oscillator coupling strength.")
    physics.add_argument("--mu-init", type=float, default=1.6,
                         metavar="MU",
                         help="Initial μ=2Ks value for the interactive slider (Exp C).")

    # --- μ sweep (Experiments A and C slider range) ---
    sweep = parser.add_argument_group("μ sweep (Experiments A and C)")
    sweep.add_argument("--mu-min",   type=float, default=0.1,
                       metavar="MU_MIN",  help="Minimum μ = 2Ks for the sweep.")
    sweep.add_argument("--mu-max",   type=float, default=3.0,
                       metavar="MU_MAX",  help="Maximum μ = 2Ks for the sweep.")
    sweep.add_argument("--mu-steps", type=int,   default=30,
                       metavar="STEPS",   help="Number of μ values in the sweep.")

    # --- Experiment B parameters ---
    expB = parser.add_argument_group("Experiment B parameters")
    expB.add_argument(
        "--mu-hist", nargs="+", type=float, default=[1.6, 3.0],
        metavar="MU",
        help=(
            "μ values at which to run the energy histogram (Exp B).  "
            "Pass multiple values separated by spaces, e.g. --mu-hist 1.6 3.0."
        ),
    )
    expB.add_argument("--trials",   type=int,   default=50,
                      help="Number of random-initialisation trials per histogram.")
    expB.add_argument("--t-end",    type=float, default=50.0,
                      help="Integration end time for dynamic simulations.")
    expB.add_argument("--n-points", type=int,   default=500,
                      help="Number of time-evaluation points in each simulation.")

    # --- Experiment selection ---
    parser.add_argument(
        "--experiments", "-e",
        type=str, default="ABC",
        metavar="EXP",
        help=(
            "Which experiments to run.  Any combination of the letters A, B, C "
            "(case-insensitive), or 'all'.  Examples: --experiments A, "
            "--experiments BC, --experiments all."
        ),
    )

    # --- Misc ---
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (Experiment B).",
    )

    return parser.parse_args(argv)


# ===========================================================================
# Entry point
# ===========================================================================

def main(argv=None) -> None:
    args = parse_args(argv)

    # -----------------------------------------------------------------------
    # Resolve which experiments to run
    # -----------------------------------------------------------------------
    exps = args.experiments.upper()
    if exps in ("ALL", "ABC"):
        run_A = run_B = run_C = True
    else:
        run_A = "A" in exps
        run_B = "B" in exps
        run_C = "C" in exps

    if not (run_A or run_B or run_C):
        print("No experiments selected.  Use --experiments A, B, C or all.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Load graph
    # -----------------------------------------------------------------------
    print(f"Loading graph from: {args.graph}")
    try:
        J = read_graph_to_J(args.graph)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    info = graph_info(J)
    print(
        f"  N={info['N']} nodes, {info['edges']} edges, "
        f"density={info['density']:.3f}, "
        f"degree min/mean/max = {info['degree_min']}/{info['degree_mean']:.1f}/{info['degree_max']}"
    )

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"  Random seed: {args.seed}")

    K = args.K
    mu_values = np.linspace(args.mu_min, args.mu_max, args.mu_steps)

    # -----------------------------------------------------------------------
    # Warn if N is large for exhaustive experiments
    # -----------------------------------------------------------------------
    N = J.shape[0]
    WARN_N = 18
    if N > WARN_N and (run_A or run_C):
        print(
            f"WARNING: N={N} → 2^N = {2**N:,} configurations. "
            f"Experiments A and C enumerate all of them and may be very slow."
        )

    # -----------------------------------------------------------------------
    # Run experiments
    # -----------------------------------------------------------------------
    if run_A:
        print("\n" + "="*60)
        print("EXPERIMENT A — λ_L vs μ  (Fig. 2)")
        print("="*60)
        experiment_A(J, K, mu_values)

    if run_B:
        print("\n" + "="*60)
        print("EXPERIMENT B — Energy histograms  (Fig. 4d–e)")
        print("="*60)
        t_span = (0.0, args.t_end)
        experiment_B(J, K, args.mu_hist, args.trials, t_span, args.n_points)

    if run_C:
        print("\n" + "="*60)
        print("EXPERIMENT C — Interactive stability slider  (Fig. 3)")
        print("="*60)
        experiment_C(J, K, args.mu_init, args.mu_min, args.mu_max)

    print("\nAll selected experiments finished.")


if __name__ == "__main__":
    main()
