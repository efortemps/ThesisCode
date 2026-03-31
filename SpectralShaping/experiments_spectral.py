"""
experiments_spectral.py
========================
Simple, focused experiments on the Spectral Shaping OIM (Mechanism C).

Four experiments, all self-contained:

  Exp 1 — Coupling shape:
      Plot g_k(φ) for several values of k to see how the coupling
      sharpens toward a square wave as k grows.

  Exp 2 — Phase trajectories:
      For a fixed graph, show how θ_i(t) evolves for k = 1, 3, 9.
      Visually confirms that larger k drives faster / cleaner phase splitting.

  Exp 3 — Binarization speed vs k:
      Sweep k over odd integers. For each k, run n_trials random
      initial conditions and record the first time t* at which
          max_i |sin(θ_i(t*)| < binarization_tol.
      Plot: median t* and fraction converged vs k.

  Exp 4 — Cut quality vs k:
      For each k, run n_trials random initialisations and collect the
      cut value of the binarised final state. Compare against the
      optimal (brute-force) cut to see if larger k helps or hurts
      solution quality.

  Exp 5 — Hardness landscape (stability vs cut, k-independent):
      For all 2^N binary equilibria φ* ∈ {0,π}^N compute:
        - λ_max(D(φ*))  (stability indicator — negative means always stable)
        - cut value of φ*
      Scatter plot reveals whether good cuts tend to be naturally stable.
      NOTE: this landscape is INDEPENDENT of k (proven in OIM_SpectralShaping).

Usage
-----
    # From a graph file (recommended):
    python experiments_spectral.py --graph my_graph.txt --out results/

    # Known optimal cut provided explicitly:
    python experiments_spectral.py --graph c8.txt --best-cut 10 --trials 60

    # Best cut read automatically from '# Max cut = 14' comment in file:
    python experiments_spectral.py --graph c10.txt --trials 60

    # Random graph fallback (no file):
    python experiments_spectral.py --N 10 --p 0.7 --seed 42

    # Run only specific experiments:
    python experiments_spectral.py --graph c8.txt --exps 245

Dependencies: numpy, scipy, matplotlib, OIM_SpectralShaping.py (same directory).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from itertools import product as iproduct
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mclors
import numpy as np
from scipy.integrate import solve_ivp

try:
    from SpectralShaping.OIM_SpectralShaping import OIM_SpectralShaping
except ImportError:
    sys.exit(
        "ERROR: OIM_SpectralShaping.py not found.\n"
        "Place it in the same directory and re-run."
    )


# ===========================================================================
# Graph and utility helpers
# ===========================================================================

def random_graph(N: int, p: float = 0.6, seed: int = 42) -> np.ndarray:
    """
    Erdős–Rényi unweighted random graph.
    Returns symmetric W > 0 (adjacency matrix, zero diagonal).
    """
    rng   = np.random.default_rng(seed)
    upper = np.triu((rng.random((N, N)) < p).astype(float), k=1)
    W     = upper + upper.T
    np.fill_diagonal(W, 0.0)
    return W


def load_graph_from_file(path: str) -> Tuple[np.ndarray, int, str]:
    """
    Load a graph from a simple edge-list text file.

    Format
    ------
    Lines starting with '#' are treated as comments and ignored.
    The first non-comment line must be a single integer N (number of nodes).
    Every subsequent non-comment line must contain two integers: "u v"
    (0-indexed node indices), representing an undirected edge.

    Example
    -------
        # 8-node cycle + diagonals, Max cut = 10
        8
        0 1
        1 2
        ...

    Returns
    -------
    W        : np.ndarray, shape (N, N)  — symmetric adjacency matrix
    N        : int                       — number of nodes
    comments : str                       — all comment lines joined (for printing)
    """
    lines    = []
    comments = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                comments.append(line)
            else:
                lines.append(line)

    if not lines:
        raise ValueError(f"Graph file '{path}' contains no data lines.")

    N = int(lines[0])
    W = np.zeros((N, N), dtype=float)

    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        u, v = int(parts[0]), int(parts[1])
        if u < 0 or v < 0 or u >= N or v >= N:
            raise ValueError(
                f"Edge ({u},{v}) out of range for N={N} in file '{path}'."
            )
        W[u, v] = 1.0
        W[v, u] = 1.0

    np.fill_diagonal(W, 0.0)
    return W, N, "\n".join(comments)


def J_from_W(W: np.ndarray) -> np.ndarray:
    """MaxCut sign convention: J = −W."""
    return -W


def cut_from_spins(W: np.ndarray, sigma: np.ndarray) -> float:
    """Cut value: Σ_{i<j} W_ij (1 − σ_i σ_j) / 2."""
    return float(0.25 * np.sum(W * (1.0 - sigma[:, None] * sigma[None, :])))


def spin_from_phase(phi: np.ndarray) -> np.ndarray:
    """θ → σ ∈ {±1}: near 0 → +1, near π → −1."""
    return np.where(phi % (2.0 * np.pi) < np.pi, 1.0, -1.0)


def binarization_residual(phi: np.ndarray) -> float:
    """max_i |sin(θ_i)| — equals 0 at a perfect binary state."""
    return float(np.max(np.abs(np.sin(phi))))


def brute_force_maxcut(W: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Exact MaxCut by exhaustive search over {±1}^N.
    Only feasible for N ≤ 20.
    """
    N, best_cut, best_s = W.shape[0], 0.0, np.ones(W.shape[0])
    for bits in iproduct([-1, 1], repeat=N):
        s   = np.array(bits, dtype=float)
        cut = cut_from_spins(W, s)
        if cut > best_cut:
            best_cut, best_s = cut, s.copy()
    return best_cut, best_s


def run_trial(
    J:        np.ndarray,
    k:        int,
    phi0:     np.ndarray,
    t_span:   Tuple[float, float],
    n_points: int,
    coeffs,
) -> dict:
    """
    Single simulation trial with OIM_SpectralShaping.

    Returns dict with keys:
        sol       : OdeSolution or None
        cut       : float (NaN if not converged)
        residual  : float  (max |sin θ_i| at final time)
        is_binary : bool
    """
    ss  = OIM_SpectralShaping(J, k=k, coeffs=coeffs)
    sol = ss.simulate(phi0.copy(), t_span=t_span, n_points=n_points)
    if sol is None:
        return dict(sol=None, cut=np.nan, residual=np.nan, is_binary=False)
    phi_f    = sol.y[:, -1]
    residual = binarization_residual(phi_f)
    sigma    = spin_from_phase(phi_f)
    W        = -J
    cut      = cut_from_spins(W, sigma)
    return dict(sol=sol, cut=cut, residual=residual, is_binary=residual < 1e-2)


# ===========================================================================
# Experiment 1 — Coupling function shapes
# ===========================================================================

def exp1_coupling_shapes(k_list: List[int], coeffs, out_dir: str) -> None:
    """
    Plot g_k(φ) for several k values on φ ∈ [−π, π].
    Shows how increasing k sharpens the coupling toward sgn(sin φ).
    """
    phi  = np.linspace(-np.pi, np.pi, 2000)
    cmap = plt.get_cmap("plasma", len(k_list))

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, k in enumerate(k_list):
        ss  = OIM_SpectralShaping(np.eye(2), k=k, coeffs=coeffs)
        col = cmap(i / max(len(k_list) - 1, 1))
        ax.plot(phi / np.pi, ss.g_k(phi), color=col, linewidth=2.0, label=f"k = {k}")

    # Target: square wave sgn(sin φ)
    ax.plot(phi / np.pi, np.sign(np.sin(phi + 1e-10)), "k:",
            linewidth=1.2, alpha=0.7, label=r"$\mathrm{sgn}(\sin\varphi)$ ($k\to\infty$)")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"$\varphi \;/\; \pi$",              fontsize=13)
    ax.set_ylabel(r"$g_k(\varphi)$",                   fontsize=13)

    coeff_label = (
        r"square-wave ($c_n = 4/n\pi$)"
        if isinstance(coeffs, str) and coeffs == "square_wave"
        else r"equal weights ($c_n = 1$)"
    )
    ax.set_title(
        f"Experiment 1 — Coupling function $g_k(\\varphi)$\n"
        f"Coefficients: {coeff_label}",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, "exp1_coupling_shapes.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  [Exp 1] Saved → {path}")


# ===========================================================================
# Experiment 2 — Phase trajectories for different k
# ===========================================================================

def exp2_phase_trajectories(
    W:        np.ndarray,
    k_list:   List[int],
    coeffs,
    seed:     int   = 0,
    t_end:    float = 40.0,
    n_points: int   = 400,
    out_dir:  str   = ".",
) -> None:
    """
    One panel per k value: phase trajectories θ_i(t) mod 2π.
    Same initial condition for all k so differences are due to k alone.
    The two binary attractors {0} and {π} are marked as dashed lines.
    """
    rng  = np.random.default_rng(seed)
    N    = W.shape[0]
    J    = J_from_W(W)
    phi0 = rng.uniform(0.0, 2 * np.pi, N)

    n_cols  = len(k_list)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    node_cmap = plt.get_cmap("tab10", N)
    t_span    = (0.0, t_end)

    for ax, k in zip(axes, k_list):
        ss  = OIM_SpectralShaping(J, k=k, coeffs=coeffs)
        sol = ss.simulate(phi0.copy(), t_span=t_span, n_points=n_points)

        if sol is None:
            ax.set_title(f"k = {k}\n(integration failed)")
            continue

        t = sol.t
        for i in range(N):
            col = node_cmap(i / N)
            ax.plot(t, sol.y[i] % (2 * np.pi), color=col, linewidth=1.4, alpha=0.85)

        ax.axhline(0.0,         color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axhline(np.pi,       color="gray",  linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axhline(2 * np.pi,   color="black", linestyle="--", linewidth=1.0, alpha=0.6)

        phi_f    = sol.y[:, -1]
        residual = binarization_residual(phi_f)
        sigma    = spin_from_phase(phi_f)
        W_      = -J
        cut      = cut_from_spins(W_, sigma)

        ax.set_title(
            f"k = {k}\n"
            f"residual={residual:.3f}  cut={cut:.0f}",
            fontsize=10,
        )
        ax.set_xlabel("Time", fontsize=10)
        ax.set_yticks([0, np.pi, 2 * np.pi])
        ax.set_yticklabels(["0", "π", "2π"])
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"$\theta_i(t) mod 2\pi$", fontsize=11)
    fig.suptitle(
        f"Experiment 2 — Phase Trajectories (N={N}, same φ₀ for all k)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    path = os.path.join(out_dir, "exp2_phase_trajectories.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  [Exp 2] Saved → {path}")


# ===========================================================================
# Experiment 3 — Binarization speed vs k
# ===========================================================================

def exp3_binarization_speed(
    W:        np.ndarray,
    k_list:   List[int],
    coeffs,
    n_trials: int   = 30,
    t_end:    float = 150.0,
    n_points: int   = 1500,
    bin_tol:  float = 0.05,
    seed:     int   = 1,
    out_dir:  str   = ".",
) -> None:
    """
    For each k, run n_trials random initial conditions and record:
      (a) first time t* where max_i |sin θ_i| < bin_tol  (NaN if never reached)
      (b) the final residual max_i |sin θ_i(t_end)|      (always available)

    WHY both metrics?
    -----------------
    For certain graph structures (e.g. regular bipartite graphs like C8/C10),
    the optimal binary equilibria are only MARGINALLY stable: λ_max(D) = 0.
    This means convergence is polynomial (~1/t) rather than exponential,
    so the residual decays slowly and may still be above a tight threshold
    at t_end.  Plotting the final residual always gives useful information
    even when the threshold is never crossed.

    bin_tol default is 0.05 (not 0.01) for the same reason: on marginally
    stable graphs a threshold of 0.01 requires very long integration times.
    """
    rng    = np.random.default_rng(seed)
    N      = W.shape[0]
    J      = J_from_W(W)
    t_eval = np.linspace(0.0, t_end, n_points)

    def first_binary_time(trajectory: np.ndarray) -> float:
        """First t such that max|sin θ| < bin_tol.  NaN if never reached."""
        for col_idx in range(trajectory.shape[1]):
            if binarization_residual(trajectory[:, col_idx]) < bin_tol:
                return float(t_eval[col_idx])
        return np.nan

    median_times, frac_converged, all_times  = [], [], []
    median_resid, all_final_resid            = [], []

    for k in k_list:
        times_k  = []
        resids_k = []
        for _ in range(n_trials):
            phi0 = rng.uniform(0.0, 2 * np.pi, N)
            ss   = OIM_SpectralShaping(J, k=k, coeffs=coeffs)
            sol  = ss.simulate(phi0, t_span=(0.0, t_end), n_points=n_points)
            if sol is None:
                times_k.append(np.nan)
                resids_k.append(np.nan)
            else:
                times_k.append(first_binary_time(sol.y))
                resids_k.append(binarization_residual(sol.y[:, -1]))

        times_k  = np.array(times_k)
        resids_k = np.array(resids_k)
        all_times.append(times_k)
        all_final_resid.append(resids_k)
        median_times.append(float(np.nanmedian(times_k)))
        frac_converged.append(float(np.mean(~np.isnan(times_k))))
        median_resid.append(float(np.nanmedian(resids_k)))
        print(f"    k={k:>3}  converged={frac_converged[-1]*100:.0f}%  "
              f"median_t*={median_times[-1]:.1f}  "
              f"median_final_residual={median_resid[-1]:.4f}")

    k_arr  = np.array(k_list)
    med_t  = np.array(median_times)
    frac   = np.array(frac_converged)
    med_r  = np.array(median_resid)

    # Figure 1: speed summary (2 panels)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Experiment 3 — Binarization Speed vs k  "
        f"(N={N}, {n_trials} trials, tol={bin_tol}, T={t_end:.0f})",
        fontsize=12, fontweight="bold",
    )

    # Panel A: fraction converged (always has data)
    ax = axes[0]
    ax.bar(k_arr, frac * 100, color="#6a4fa3", alpha=0.8, width=1.5,
           edgecolor="white", linewidth=0.6)
    ax.axhline(100, color="gray", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Fourier order k", fontsize=12)
    ax.set_ylabel(f"% trials reaching residual < {bin_tol}", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title(f"Binarization rate before T={t_end:.0f}")
    ax.set_xticks(k_arr)
    ax.grid(True, alpha=0.3, axis="y")
    for xi, fi in zip(k_arr, frac):
        ax.text(xi, fi * 100 + 2, f"{fi*100:.0f}%", ha="center",
                va="bottom", fontsize=8, color="#333")

    # Panel B: median final residual vs k (always has data)
    ax = axes[1]
    ax.plot(k_arr, med_r, "o-", color="#e07b39", markersize=7, linewidth=2,
            label="Median final residual")
    ax.axhline(bin_tol, color="purple", linestyle="--", linewidth=1.3,
               label=f"bin_tol = {bin_tol}")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Fourier order k", fontsize=12)
    ax.set_ylabel(r"Median $\max_i |\sin\theta_i|$ at $t_{end}$", fontsize=11)
    ax.set_title(f"Residual at T={t_end:.0f} (lower = more binary)")
    ax.legend(fontsize=9)
    ax.set_xticks(k_arr)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "exp3_binarization_speed.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  [Exp 3] Saved → {path}")

    # Figure 2: final residual box plots (always has data)
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    valid_resids = [r[~np.isnan(r)] for r in all_final_resid]
    positions    = list(range(len(k_list)))
    plot_data = [r if len(r) > 0 else np.array([np.nan]) for r in valid_resids]
    ax2.boxplot(
        plot_data, positions=positions, widths=0.5, patch_artist=True,
        boxprops=dict(facecolor="#c9b8e8", color="#6a4fa3"),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="#6a4fa3"),
        capprops=dict(color="#6a4fa3"),
        flierprops=dict(marker="x", color="#6a4fa3", markersize=4),
    )
    ax2.axhline(bin_tol, color="purple", linestyle="--", linewidth=1.5,
                label=f"bin_tol = {bin_tol}")
    ax2.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
    ax2.set_xticks(positions)
    ax2.set_xticklabels([f"k={k}" for k in k_list])
    ax2.set_xlabel("Fourier order k", fontsize=12)
    ax2.set_ylabel(r"Final residual $\max_i |\sin\theta_i(T)|$", fontsize=11)
    ax2.set_title(
        f"Distribution of final binarization residual vs k  "
        f"(N={N}, {n_trials} trials, T={t_end:.0f})\n"
        f"Below purple line = binarized (tol={bin_tol}).  "
        f"Fraction binarized shown above each box.",
        fontsize=10,
    )
    ax2.legend(fontsize=9)
    y_top = ax2.get_ylim()[1]
    for pos, frac_val in zip(positions, frac_converged):
        ax2.text(pos, y_top * 0.97, f"{frac_val*100:.0f}%",
                 ha="center", va="top", fontsize=9, color="#6a4fa3",
                 fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout()
    path2 = os.path.join(out_dir, "exp3_residual_boxplot.png")
    fig2.savefig(path2, dpi=140)
    plt.close(fig2)
    print(f"  [Exp 3] Saved → {path2}")


# ===========================================================================
# Experiment 4 — Cut quality vs k
# ===========================================================================

def exp4_cut_quality(
    W:        np.ndarray,
    best_cut: float,
    k_list:   List[int],
    coeffs,
    n_trials: int   = 50,
    t_end:    float = 60.0,
    n_points: int   = 600,
    seed:     int   = 2,
    out_dir:  str   = ".",
) -> None:
    """
    For each k, run n_trials random initial conditions.
    Record cut value of the binarised final state.

    Plots:
      - Mean cut / best_cut vs k  (quality)
      - Success probability P(cut ≥ 0.95 · best_cut) vs k
      - Box plots of cut distributions

    Key question: does sharpening the coupling improve or hurt cut quality?
    """
    rng    = np.random.default_rng(seed)
    N      = W.shape[0]
    J      = J_from_W(W)
    t_span = (0.0, t_end)

    all_cuts = []
    mean_ratio, success_prob = [], []

    for k in k_list:
        cuts_k = []
        for _ in range(n_trials):
            phi0   = rng.uniform(0.0, 2 * np.pi, N)
            result = run_trial(J, k, phi0, t_span, n_points, coeffs)
            if not np.isnan(result["cut"]):
                cuts_k.append(result["cut"])
            else:
                cuts_k.append(0.0)   # failed run counts as 0 cut
        cuts_k = np.array(cuts_k)
        all_cuts.append(cuts_k)
        mean_ratio.append(float(np.mean(cuts_k)) / best_cut)
        success_prob.append(float(np.mean(cuts_k >= 0.95 * best_cut)))
        print(f"    k={k:>3}  mean_ratio={mean_ratio[-1]:.3f}  "
              f"P(≥95%)={success_prob[-1]:.2f}")

    k_arr  = np.array(k_list)
    mr_arr = np.array(mean_ratio)
    sp_arr = np.array(success_prob)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Experiment 4 — Cut Quality vs k  "
        f"(N={N}, best_cut={best_cut:.0f}, {n_trials} trials)",
        fontsize=12, fontweight="bold",
    )

    # Panel A: mean cut ratio vs k
    ax = axes[0]
    ax.plot(k_arr, mr_arr, "o-", color="#e07b39", markersize=7, linewidth=2)
    ax.axhline(1.00, color="black", linestyle="--", linewidth=1.2, label="Optimal (1.00)")
    ax.axhline(0.95, color="gray",  linestyle=":",  linewidth=1.2, label="95% threshold")
    ax.set_xlabel("Fourier order k", fontsize=12)
    ax.set_ylabel("Mean cut / best_cut", fontsize=11)
    ax.set_ylim(max(0.0, mr_arr.min() - 0.1), 1.12)
    ax.set_title("Mean cut quality vs k")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_arr)

    # Panel B: success probability vs k
    ax = axes[1]
    ax.plot(k_arr, sp_arr * 100, "s-", color="#d62728", markersize=7, linewidth=2)
    ax.axhline(95, color="gray", linestyle=":", linewidth=1.2)
    ax.set_xlabel("Fourier order k", fontsize=12)
    ax.set_ylabel("P(cut ≥ 95% of best)  [%]", fontsize=11)
    ax.set_ylim(-5, 110)
    ax.set_title("Success probability vs k")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_arr)

    # Panel C: box plot of cut distributions
    ax = axes[2]
    bp = ax.boxplot(
        all_cuts,
        positions=list(range(len(k_list))),
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="#ffd6a5", color="#e07b39"),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="#e07b39"),
        capprops=dict(color="#e07b39"),
        flierprops=dict(marker="x", color="#e07b39", markersize=4),
    )
    ax.axhline(best_cut, color="black", linestyle="--",
               linewidth=1.5, label=f"Best cut = {best_cut:.0f}")
    ax.axhline(0.95 * best_cut, color="gray", linestyle=":",
               linewidth=1.2, label=f"95% = {0.95*best_cut:.1f}")
    ax.set_xticks(list(range(len(k_list))))
    ax.set_xticklabels([f"k={k}" for k in k_list])
    ax.set_xlabel("Fourier order k", fontsize=12)
    ax.set_ylabel("Cut value", fontsize=11)
    ax.set_title("Distribution of cut values vs k")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(out_dir, "exp4_cut_quality.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  [Exp 4] Saved → {path}")


# ===========================================================================
# Experiment 5 — Hardness landscape: stability threshold vs cut quality
# ===========================================================================

def exp5_hardness_landscape(
    W:        np.ndarray,
    best_cut: float,
    out_dir:  str = ".",
) -> None:
    """
    Enumerate all 2^N binary equilibria φ* ∈ {0, π}^N.
    For each compute:
      - λ_D = λ_max(D(φ*)):  stability indicator.
            λ_D ≤ 0  ↔  equilibrium is ALWAYS stable (for any k > 0)
            λ_D > 0  ↔  equilibrium is UNSTABLE without external forcing
      - Cut value of φ*

    The D-matrix formula (same as Cheng et al. 2024, Theorem 2):
        D_ij = J_ij cos(φ_i* − φ_j*) = −W_ij cos(φ_i* − φ_j*)  (i ≠ j)
        D_ii = −Σ_{j≠i} D_ij   (zero row-sum)

    Key result proven in OIM_SpectralShaping: the stability of binary
    equilibria is INDEPENDENT of k (since A(φ*) = α_k · D(φ*) with α_k > 0).
    """
    N = W.shape[0]
    J = J_from_W(W)

    print(f"  [Exp 5] Enumerating 2^{N} = {2**N} binary equilibria ...")

    lambda_D_list, cut_list = [], []
    for bits in iproduct([0, 1], repeat=N):
        phi   = np.array(bits, dtype=float) * np.pi
        sigma = np.where(np.array(bits) == 0, 1.0, -1.0)

        diff  = phi[:, None] - phi[None, :]
        D     = J * np.cos(diff)
        np.fill_diagonal(D, 0.0)
        np.fill_diagonal(D, -D.sum(axis=1))
        lambda_D = float(np.linalg.eigvalsh(D).max())

        cut = cut_from_spins(W, sigma)
        lambda_D_list.append(lambda_D)
        cut_list.append(cut)

    lambda_D = np.array(lambda_D_list)
    cuts     = np.array(cut_list)

    always_stable = lambda_D <= 0.0
    n_stable      = int(always_stable.sum())

    # Use the true maximum cut found by enumeration.
    # The passed-in best_cut is used only for the vertical reference line;
    # it may differ from the enumeration maximum if the graph file comment
    # is wrong (e.g. C10 comment says 14 but actual MaxCut is 15).
    enum_best     = float(cuts.max())
    opt_mask      = np.isclose(cuts, enum_best)

    if not np.isclose(enum_best, best_cut):
        print(f"  [Exp 5] NOTE: enumeration best cut = {enum_best:.0f}, "
              f"but passed best_cut = {best_cut:.0f}. "
              f"Using enumeration value for landscape analysis.")

    print(f"  [Exp 5] Always-stable (λ_D ≤ 0): {n_stable} / {2**N}")
    if opt_mask.any():
        print(f"  [Exp 5] Best-cut equilibria (cut={enum_best:.0f}): "
              f"λ_D ∈ [{lambda_D[opt_mask].min():.3f}, {lambda_D[opt_mask].max():.3f}]")
    else:
        print(f"  [Exp 5] No binary equilibrium exactly achieves cut={enum_best:.0f} "
              f"(floating-point check failed — investigate manually).")

    # ---- Plot ----
    norm = mclors.Normalize(cuts.min(), cuts.max())
    cmap = plt.get_cmap("RdYlGn")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Experiment 5 — Hardness Landscape  (N={N}, all 2^N={2**N} binary equilibria)\n"
        r"$\lambda_D = \lambda_{\max}(D(\phi^*))$ — stability indicator, "
        "independent of $k$",
        fontsize=11, fontweight="bold",
    )

    # Panel A: scatter λ_D vs cut
    ax = axes[0]
    sc = ax.scatter(
        cuts, lambda_D,
        c=cuts, cmap=cmap, norm=norm,
        s=35, alpha=0.75, edgecolors="none", zorder=2,
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.5,
               label=r"$\lambda_D = 0$  (stability boundary)")
    ax.axvline(enum_best, color="purple", linestyle=":", linewidth=1.5,
               label=f"Best cut (enumeration) = {enum_best:.0f}")
    # Highlight best-cut equilibria
    if opt_mask.any():
        ax.scatter(
            cuts[opt_mask], lambda_D[opt_mask],
            s=80, edgecolors="purple", facecolors="none",
            linewidths=1.8, zorder=3, label=f"Best-cut equilibria ({opt_mask.sum()})",
        )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Cut value", fontsize=9)
    ax.set_xlabel("Cut value",                            fontsize=12)
    ax.set_ylabel(r"$\lambda_D = \lambda_{\max}(D(\phi^*))$", fontsize=12)
    ax.set_title(
        "Each dot = one binary equilibrium\n"
        "Below dashed line = naturally stable (no coupling needed)",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: distributions split by stability
    ax2 = axes[1]
    stable_cuts   = cuts[always_stable]
    unstable_cuts = cuts[~always_stable]
    all_cut_vals  = np.unique(np.round(cuts))
    bins = np.arange(cuts.min() - 0.5, cuts.max() + 1.5, 1.0)
    ax2.hist(unstable_cuts, bins=bins, alpha=0.65, color="#d62728",
             label=f"Unstable (λ_D > 0): {(~always_stable).sum()}",
             edgecolor="none")
    ax2.hist(stable_cuts, bins=bins, alpha=0.65, color="#2ca02c",
             label=f"Always stable (λ_D ≤ 0): {n_stable}",
             edgecolor="none")
    ax2.axvline(enum_best, color="black", linestyle="--",
                linewidth=1.8, label=f"Best cut (enumeration) = {enum_best:.0f}")
    ax2.set_xlabel("Cut value",              fontsize=12)
    ax2.set_ylabel("Number of equilibria",   fontsize=11)
    ax2.set_title(
        "Cut quality distribution by stability class\n"
        "Do stable equilibria correspond to good cuts?",
        fontsize=9,
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(out_dir, "exp5_hardness_landscape.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  [Exp 5] Saved → {path}")


# ===========================================================================
# CLI and main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Focused experiments on OIM_SpectralShaping (Mechanism C)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Graph source ───────────────────────────────────────────────────────
    p.add_argument(
        "--graph", type=str, default=None, metavar="FILE",
        help=(
            "Path to an edge-list graph file (see format below). "
            "When given, --N and --p are ignored.\n"
            "File format:\n"
            "  Lines starting with '#' are comments.\n"
            "  First non-comment line: single integer N (node count).\n"
            "  Remaining lines: 'u v' pairs (0-indexed, undirected edges).\n"
            "  Optionally write '# Max cut = X' in a comment to set --best-cut."
        ),
    )
    p.add_argument("--N",   type=int,   default=10,
                   help="Nodes in random Erdős–Rényi graph (ignored if --graph given)")
    p.add_argument("--p",   type=float, default=0.65,
                   help="Edge probability for random graph (ignored if --graph given)")

    # ── Best-cut override ──────────────────────────────────────────────────
    p.add_argument(
        "--best-cut", type=float, default=None, metavar="CUT",
        help=(
            "Known optimal cut (skips brute-force). "
            "If omitted, brute-force runs for N≤18, "
            "or is parsed from a '# Max cut = X' comment in the graph file."
        ),
    )

    # ── Experiment control ─────────────────────────────────────────────────
    p.add_argument("--seed",   type=int,   default=42,
                   help="Random seed for initial conditions (reproducibility)")
    p.add_argument("--trials", type=int,   default=40,
                   help="Trials per parameter value (Exp 3 + 4)")
    p.add_argument("--t-end",  type=float, default=60.0,
                   help="Integration end time")
    p.add_argument("--out",    type=str,   default=".",
                   help="Output directory for saved figures")
    p.add_argument("--exps",   type=str,   default="12345",
                   help="Experiments to run, e.g. '135' runs Exp 1, 3, 5")
    p.add_argument("--coeffs", type=str,   default="equal",
                   choices=["equal", "square_wave"],
                   help="Fourier coefficients: equal (c_n=1) or square_wave (c_n=4/nπ)")
    return p.parse_args()


def _parse_best_cut_from_comments(comments: str) -> Optional[float]:
    """
    Look for a line like '# Max cut = 10' or '# max cut = 14' in the
    comment block and return the value as a float if found.
    """
    import re
    for line in comments.splitlines():
        m = re.search(r"[Mm]ax\s+cut\s*=\s*([0-9]+(?:\.[0-9]*)?)", line)
        if m:
            return float(m.group(1))
    return None


def main():
    args   = parse_args()
    coeffs = None if args.coeffs == "equal" else "square_wave"
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print("=" * 60)
    print("  OIM SpectralShaping — Focused Experiments")
    print("=" * 60)

    # ── Load or generate graph ─────────────────────────────────────────────
    best_cut_hint: Optional[float] = args.best_cut

    if args.graph is not None:
        if not os.path.isfile(args.graph):
            sys.exit(f"ERROR: graph file not found: '{args.graph}'")
        W, N, comments = load_graph_from_file(args.graph)
        n_edges = int(np.sum(W) // 2)
        print(f"\nGraph file : {args.graph}")
        if comments:
            print("  " + "\n  ".join(comments.splitlines()))
        print(f"  N={N}, edges={n_edges}")
        # Try to read best-cut from comments if not given on CLI
        if best_cut_hint is None:
            best_cut_hint = _parse_best_cut_from_comments(comments)
            if best_cut_hint is not None:
                print(f"  Best cut (from file comment) = {best_cut_hint:.0f}")
    else:
        W = random_graph(args.N, p=args.p, seed=args.seed)
        N = args.N
        n_edges = int(np.sum(W) // 2)
        print(f"\nRandom graph: N={N}, edges={n_edges}, p={args.p}, seed={args.seed}")

    # ── Best-cut resolution ────────────────────────────────────────────────
    if best_cut_hint is not None:
        best_cut = best_cut_hint
        print(f"  Using provided optimal cut = {best_cut:.0f}")
    elif N <= 18:
        print("  Computing exact MaxCut (brute force) ...")
        t0 = time.time()
        best_cut, _ = brute_force_maxcut(W)
        print(f"  Optimal cut = {best_cut:.0f}  ({time.time()-t0:.2f}s)")
    else:
        print(f"  N={N} > 18 and no best-cut given — estimating from 50 OIM runs ...")
        rng_bc = np.random.default_rng(args.seed + 99)
        J_bc   = J_from_W(W)
        cuts   = []
        for _ in range(50):
            phi0   = rng_bc.uniform(0, 2 * np.pi, N)
            result = run_trial(J_bc, 9, phi0, (0.0, args.t_end), 600, coeffs)
            if not np.isnan(result["cut"]):
                cuts.append(result["cut"])
        best_cut = float(max(cuts)) if cuts else 1.0
        print(f"  Best cut (estimate) = {best_cut:.0f}")

    # ── k grids (tuned to graph size) ─────────────────────────────────────
    k_sweep = [1, 3, 5, 7, 9, 11, 13]    # Exp 3, 4: parameter sweep
    k_traj  = [1, 3, 7, 13]              # Exp 2: trajectory panels
    k_shape = [1, 3, 5, 9, 15, 25]       # Exp 1: coupling shape plot

    exps = args.exps

    if "1" in exps:
        print("\n── Experiment 1: Coupling shapes ──")
        exp1_coupling_shapes(k_shape, coeffs, args.out)

    if "2" in exps:
        print("\n── Experiment 2: Phase trajectories ──")
        exp2_phase_trajectories(W, k_traj, coeffs,
                                t_end=args.t_end, out_dir=args.out)

    if "3" in exps:
        print(f"\n── Experiment 3: Binarization speed vs k ({args.trials} trials) ──")
        exp3_binarization_speed(W, k_sweep, coeffs,
                                n_trials=args.trials, t_end=args.t_end,
                                seed=args.seed + 1, out_dir=args.out)

    if "4" in exps:
        print(f"\n── Experiment 4: Cut quality vs k ({args.trials} trials) ──")
        exp4_cut_quality(W, best_cut, k_sweep, coeffs,
                         n_trials=args.trials, t_end=args.t_end,
                         seed=args.seed + 2, out_dir=args.out)

    if "5" in exps:
        if N <= 18:
            print("\n── Experiment 5: Hardness landscape ──")
            exp5_hardness_landscape(W, best_cut, args.out)
        else:
            print(f"\n  [Exp 5] Skipped — N={N} > 18 (2^N too large to enumerate)")

    print("\n" + "=" * 60)
    print(f"  Done. Figures saved to: {os.path.abspath(args.out)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
