"""
benchmark_mechanisms.py
=======================
Comparative benchmark of the three binarization mechanisms for MaxCut:

  Mechanism A — Extrinsic Penalty / OIM        (parameter: μ)     [OIM_mu_v2.py]
  Mechanism B — Intrinsic Gain / Hopfield       (parameter: β)     [implemented below]
  Mechanism C — Spectral Shaping / Penalty-Free (parameter: k)     [OIM_SpectralShaping.py]

Research question (thesis):
  "How do different binarization mechanisms affect solution quality,
   runtime, and robustness across instances, and can 'penalty-free'
   binarization via coupling-function design match or improve the
   penalty/gain baselines?"

===========================================================================
FRAMEWORK CONSISTENCY NOTE
===========================================================================
OIM_mu_v2 (Mechanism A):
  - Takes W > 0 (positive adjacency matrix, raw graph weights)
  - Dynamics: dθ_i/dt = Σ_j W_ij sin(θ_i−θ_j) − (μ/2) sin(2θ_i)
  - Stability threshold per equilibrium: μ*_i = λ_max(D(φ*_i))
  - Binarization threshold: μ_bin = min_i μ*_i  (no floor at 0)

OIM_Stability_1 (underlying class for run_experiment.py):
  - Takes J = −W < 0 (negated weights, MaxCut sign convention)
  - Dynamics: dθ_i/dt = −K Σ_j J_ij sin(θ_i−θ_j) − Ks sin(2θ_i)
  - With K=1, Ks=μ/2: IDENTICAL dynamics to OIM_mu_v2 (verified: −J_ij = W_ij)
  - Stability threshold: Ks* = K λ_max(D)/2 = λ_max(D)/2 = μ*/2  ← factor-of-2 naming only

OIM_SpectralShaping (Mechanism C):
  - Takes J = −W (same convention as OIM_Stability_1)
  - Dynamics: dθ_i/dt = Σ_j W_ij g_k(θ_i−θ_j) = −Σ_j J_ij g_k(θ_i−θ_j)
  - At k=1, c_1=1: g_1 = sin → identical to OIM_mu_v2 at μ=0 ✓
  - D-matrix: D_ij = J_ij cos(φ_i−φ_j) = −W_ij cos(φ_i−φ_j), same formula everywhere ✓
  - Stability of binary equilibria is k-independent (A(φ*)=α_k·D(φ*), α_k>0)

Binarization measure (uniform across OIM and Spectral):
  residual = max_i |sin(θ_i)|   → 0 at perfect binary state {0,π}^N

Binarization measure for Hopfield:
  residual = max_i (1 − |tanh(β u_i)|) → 0 when all outputs saturate to ±1

Cut value (all mechanisms):
  cut = Σ_{i<j} W_ij (1 − σ_i σ_j) / 2,   σ_i = sign of binary assignment

===========================================================================
DIFFERENCES BETWEEN OIM_mu_v2 AND OIM_Stability_1 (for reference)
===========================================================================
  1. Weight convention: W>0 vs J=−W. Same dynamics; only sign convention differs.
  2. Threshold naming: μ* = λ_max(D) [mu_v2] vs Ks* = λ_max(D)/2 [Stability_1]
     These are consistent: μ = 2·Ks, so μ* = 2·Ks* ✓
  3. Binarization threshold: mu_v2 allows μ_bin < 0 (some equilibria always stable);
     Stability_1 floors Ks* at 0. The mu_v2 convention is more informative.
  4. Lyapunov energy: Different functional forms but same gradient flow dynamics.
     mu_v2: L = Σ_{i<j} W_ij cos(diff) + (μ/2) Σ sin²(θ_i)
     Stability_1: E = 2·(coupling part) + different injection scaling (≈ 2·L + const)
     Both are valid Lyapunov functions; only magnitudes differ, not the trajectory.

===========================================================================
EXPERIMENTS
===========================================================================
  1. Parameter sweep: binarization residual + cut quality vs control parameter
  2. Success-rate benchmark: P(cut ≥ 0.95 · best_known) over N_RUNS random inits
  3. Hardness landscape: stability threshold vs cut quality for all binary equilibria
  4. Convergence speed: integration time to reach binarization across parameter values
  5. Phase trajectories: visual verification of phase splitting (small N=6 graph)

Usage
-----
  python benchmark_mechanisms.py [--seed SEED] [--N N] [--runs RUNS] [--save]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from itertools import product as iproduct
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # headless rendering — saves figures to disk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
from scipy.integrate import solve_ivp

# ------------------------------------------------------------------
# Try to import the two OIM classes.  Adjust paths as needed.
# ------------------------------------------------------------------
try:
    from OIM_Experiment.src.OIM_mu import OIMMaxCut
except ImportError:
    sys.exit(
        "ERROR: Cannot import OIMMaxCut from OIM_mu_v2.py.\n"
        "Place OIM_mu_v2.py in the same directory and re-run."
    )

try:
    from SpectralShaping.OIM_SpectralShaping import OIM_SpectralShaping
except ImportError:
    sys.exit(
        "ERROR: Cannot import OIM_SpectralShaping from OIM_SpectralShaping.py.\n"
        "Place OIM_SpectralShaping.py in the same directory and re-run."
    )


# ===========================================================================
# Mechanism B — Hopfield Network for MaxCut
# ===========================================================================

class HopfieldMaxCut:
    """
    Hopfield network (Mechanism B) for MaxCut.

    Energy (Eq. A.3):
        E(V) = −(1/2) Σ_ij J_ij V_i V_j + Σ_i (1/R) ∫_0^{V_i} g^{-1}(V) dV
             = (1/2) Σ_ij W_ij V_i V_j + (1/β) Σ_i [V_i arctanh(V_i) − (1/2)log(1−V_i²)]

    Dynamics (Eq. A.4), with J = −W, R=1, C=1:
        du_i/dt = −u_i − Σ_j W_ij tanh(β u_j)

    Activation (Eq. A.5):
        V_i = tanh(β u_i)     [output in (−1, +1)]

    Binarization: V_i → ±1 as β → ∞ or t → ∞.

    Parameters
    ----------
    W    : np.ndarray, shape (N,N), positive symmetric adjacency matrix
    beta : float, activation gain (controls sharpness of tanh)
    """

    def __init__(self, W: np.ndarray, beta: float = 1.0) -> None:
        self.W    = np.asarray(W, dtype=float)
        self.N    = W.shape[0]
        self.beta = float(beta)
        self.u    = np.zeros(self.N)   # internal state (potentials)

    def dynamics(self, t: float, u: np.ndarray) -> np.ndarray:
        """du/dt = −u − W · tanh(β u)  (gradient flow of E w.r.t. u)."""
        V = np.tanh(self.beta * u)
        return -u - self.W @ V

    def simulate(
        self,
        u0:       np.ndarray,
        t_span:   tuple       = (0.0, 50.0),
        n_points: int         = 500,
        rtol:     float       = 1e-6,
        atol:     float       = 1e-8,
    ) -> Optional[object]:
        """Integrate the Hopfield ODE from u0. Returns scipy OdeSolution or None."""
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol    = solve_ivp(
            self.dynamics, t_span, u0,
            t_eval=t_eval, method="RK45", rtol=rtol, atol=atol,
        )
        if not sol.success:
            return None
        self.u = sol.y[:, -1].copy()
        return sol

    def simulate_many(
        self, u0_list: List[np.ndarray], t_span: tuple, n_points: int
    ) -> List[Optional[object]]:
        return [self.simulate(u0, t_span, n_points) for u0 in u0_list]

    def get_output(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """V_i = tanh(β u_i)."""
        if u is None:
            u = self.u
        return np.tanh(self.beta * u)

    def get_spins(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        """σ_i = sign(V_i) = sign(tanh(β u_i)) = sign(u_i)."""
        V = self.get_output(u)
        s = np.sign(V)
        s[s == 0] = 1.0
        return s

    def binarization_residual(self, u: Optional[np.ndarray] = None) -> float:
        """max_i (1 − |V_i|): 0 at perfect binarization (all |V_i|=1)."""
        V = self.get_output(u)
        return float(np.max(1.0 - np.abs(V)))

    def is_binarized(self, u: Optional[np.ndarray] = None, tol: float = 1e-2) -> bool:
        return bool(self.binarization_residual(u) < tol)

    def get_binary_cut_value(self, u: Optional[np.ndarray] = None) -> float:
        """Cut value after hard binarization."""
        sigma = self.get_spins(u)
        return float(0.25 * np.sum(self.W * (1.0 - sigma[:, None] * sigma[None, :])))


# ===========================================================================
# Graph Utilities
# ===========================================================================

def random_graph(N: int, p: float = 0.5, seed: int = 42) -> np.ndarray:
    """Erdős–Rényi random graph (unweighted), returns symmetric W with 0 diagonal."""
    rng = np.random.default_rng(seed)
    upper = (rng.random((N, N)) < p).astype(float)
    upper = np.triu(upper, k=1)
    W     = upper + upper.T
    np.fill_diagonal(W, 0.0)
    return W


def brute_force_maxcut(W: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Exact MaxCut by exhaustive search over all 2^N spin assignments.
    Only feasible for N ≤ 20.  Returns (best_cut, best_sigma).
    """
    N        = W.shape[0]
    best_cut = 0.0
    best_s   = np.ones(N)
    for bits in iproduct([-1, 1], repeat=N):
        s        = np.array(bits, dtype=float)
        cut      = 0.25 * float(np.sum(W * (1.0 - s[:, None] * s[None, :])))
        if cut > best_cut:
            best_cut = cut
            best_s   = s.copy()
    return best_cut, best_s


def cut_from_spins(W: np.ndarray, sigma: np.ndarray) -> float:
    return float(0.25 * np.sum(W * (1.0 - sigma[:, None] * sigma[None, :])))


def w_total(W: np.ndarray) -> float:
    return float(np.sum(W) / 2.0)


def spin_from_phase(phi: np.ndarray) -> np.ndarray:
    """Phase → spin: 0 → +1, π → −1  (standard OIM/SpectralShaping rule)."""
    return np.where(phi % (2.0 * np.pi) < np.pi, 1.0, -1.0)


# ===========================================================================
# Unified trial runners  (one per mechanism)
# ===========================================================================

T_SPAN   = (0.0, 60.0)
N_POINTS = 600


def run_oim_trial(
    W: np.ndarray, mu: float, phi0: np.ndarray
) -> dict:
    """Single OIM (Mechanism A) trial.  W > 0."""
    oim = OIMMaxCut(W, mu=mu, seed=0)
    sol = oim.simulate(phi0.copy(), t_span=T_SPAN, n_points=N_POINTS,
                       rtol=1e-7, atol=1e-7)
    if sol is None:
        return dict(cut=np.nan, residual=np.nan, is_binary=False)
    phi_f    = sol.y[:, -1]
    sigma    = spin_from_phase(phi_f)
    residual = float(np.max(np.abs(np.sin(phi_f))))
    cut      = cut_from_spins(W, sigma)
    return dict(cut=cut, residual=residual, is_binary=residual < 1e-2)


def run_spectral_trial(
    W: np.ndarray, k: int, phi0: np.ndarray
) -> dict:
    """Single Spectral Shaping (Mechanism C) trial.  J = −W."""
    J   = -W
    ss  = OIM_SpectralShaping(J, k=k)
    sol = ss.simulate(phi0.copy(), t_span=T_SPAN, n_points=N_POINTS)
    if sol is None:
        return dict(cut=np.nan, residual=np.nan, is_binary=False)
    phi_f    = sol.y[:, -1]
    sigma    = spin_from_phase(phi_f)
    residual = float(np.max(np.abs(np.sin(phi_f))))
    cut      = cut_from_spins(W, sigma)
    return dict(cut=cut, residual=residual, is_binary=residual < 1e-2)


def run_hopfield_trial(
    W: np.ndarray, beta: float, u0: np.ndarray
) -> dict:
    """Single Hopfield (Mechanism B) trial."""
    hop = HopfieldMaxCut(W, beta=beta)
    sol = hop.simulate(u0.copy(), t_span=T_SPAN, n_points=N_POINTS)
    if sol is None:
        return dict(cut=np.nan, residual=np.nan, is_binary=False)
    u_f      = sol.y[:, -1]
    sigma    = hop.get_spins(u_f)
    residual = hop.binarization_residual(u_f)
    cut      = cut_from_spins(W, sigma)
    return dict(cut=cut, residual=residual, is_binary=residual < 1e-2)


# ===========================================================================
# Experiment 1 — Parameter sweep: binarization quality vs control parameter
# ===========================================================================

def exp1_parameter_sweep(
    W:         np.ndarray,
    best_cut:  float,
    n_runs:    int   = 20,
    seed:      int   = 0,
    out_dir:   str   = ".",
) -> None:
    """
    Sweep the control parameter for each mechanism.
    For each value, run n_runs random initial conditions and record:
      - fraction of runs that binarized
      - mean cut value (over binarized runs only)
      - mean cut ratio = mean_cut / best_cut
    """
    rng = np.random.default_rng(seed)
    N   = W.shape[0]

    # --- Compute OIM binarization threshold (theoretical) ---
    mu_bin_theory = _oim_mu_bin(W)
    print(f"  [Exp1] Theoretical OIM binarization threshold: μ_bin = {mu_bin_theory:.4f}")

    # --- Parameter grids ---
    mu_grid   = np.linspace(0.05, max(3.0, 2.0 * mu_bin_theory + 0.5), 30)
    beta_grid = np.logspace(np.log10(0.3), np.log10(30.0), 20)
    k_grid    = [1, 3, 5, 7, 9, 11, 13, 15]

    def sweep(runner_fn, param_grid, param_name, init_fn):
        frac_bin, mean_cut, mean_ratio = [], [], []
        for p in param_grid:
            cuts, bins = [], []
            for _ in range(n_runs):
                ic     = init_fn(rng, N)
                result = runner_fn(p, ic)
                bins.append(result["is_binary"])
                if result["is_binary"] and not np.isnan(result["cut"]):
                    cuts.append(result["cut"])
            frac_bin.append(float(np.mean(bins)))
            mean_cut.append(float(np.mean(cuts)) if cuts else np.nan)
            mean_ratio.append(
                float(np.mean(cuts)) / best_cut if cuts else np.nan
            )
        return np.array(frac_bin), np.array(mean_cut), np.array(mean_ratio)

    def oim_runner(mu, phi0):
        return run_oim_trial(W, mu, phi0)
    def spec_runner(k, phi0):
        return run_spectral_trial(W, int(k), phi0)
    def hop_runner(beta, u0):
        return run_hopfield_trial(W, beta, u0)

    phase_init = lambda rng, N: rng.uniform(0.0, 2 * np.pi, N)
    small_init = lambda rng, N: rng.uniform(-0.01, 0.01, N)

    print("  [Exp1] Sweeping OIM (A) ...")
    frac_A, cut_A, ratio_A = sweep(oim_runner, mu_grid,   "mu",   phase_init)
    print("  [Exp1] Sweeping Hopfield (B) ...")
    frac_B, cut_B, ratio_B = sweep(hop_runner, beta_grid, "beta", small_init)
    print("  [Exp1] Sweeping Spectral (C) ...")
    frac_C, cut_C, ratio_C = sweep(spec_runner, k_grid,   "k",    phase_init)

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"Experiment 1 — Parameter Sweep (N={N}, best_cut={best_cut:.0f}, "
        f"{n_runs} runs/point)",
        fontsize=13, fontweight="bold",
    )

    col_A, col_B, col_C = "#e07b39", "#4fa3a8", "#6a4fa3"

    # Row 0: fraction binarized
    for ax, grid, frac, col, lbl, thresh in [
        (axes[0, 0], mu_grid,   frac_A, col_A, r"$\mu$  (OIM penalty)",   mu_bin_theory),
        (axes[0, 1], beta_grid, frac_B, col_B, r"$\beta$  (Hopfield gain)", None),
        (axes[0, 2], k_grid,    frac_C, col_C, r"$k$  (Fourier order)",    None),
    ]:
        ax.plot(grid, frac, "o-", color=col, markersize=5, linewidth=1.8)
        if thresh is not None:
            ax.axvline(thresh, color="red", linestyle="--", linewidth=1.2,
                       label=rf"$\mu_{{bin}}^{{theory}}={thresh:.2f}$")
            ax.legend(fontsize=8)
        ax.set_xlabel(lbl, fontsize=11)
        ax.set_ylabel("Fraction binarized", fontsize=10)
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(f"Mechanism {'ABC'[list([axes[0,0],axes[0,1],axes[0,2]]).index(ax)]}: binarization rate")
        ax.grid(True, alpha=0.3)

    # Row 1: mean cut ratio
    for ax, grid, ratio, col, lbl in [
        (axes[1, 0], mu_grid,   ratio_A, col_A, r"$\mu$"),
        (axes[1, 1], beta_grid, ratio_B, col_B, r"$\beta$"),
        (axes[1, 2], k_grid,    ratio_C, col_C, r"$k$"),
    ]:
        mask = ~np.isnan(ratio)
        ax.plot(np.array(grid)[mask], ratio[mask], "s-",
                color=col, markersize=5, linewidth=1.8)
        ax.axhline(0.95, color="gray", linestyle=":", linewidth=1.2,
                   label="95% threshold")
        ax.axhline(1.00, color="black", linestyle="--", linewidth=1.0,
                   label="Optimal")
        ax.set_xlabel(lbl, fontsize=11)
        ax.set_ylabel("Mean cut / best_cut", fontsize=10)
        ax.set_ylim(0.5, 1.15)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_title("Cut quality (binarized runs)")

    fig.tight_layout()
    path = os.path.join(out_dir, "exp1_parameter_sweep.png")
    fig.savefig(path, dpi=130)
    print(f"  [Exp1] Saved → {path}")
    plt.close(fig)


# ===========================================================================
# Experiment 2 — Success-rate benchmark
# ===========================================================================

def exp2_success_rate(
    W:         np.ndarray,
    best_cut:  float,
    n_runs:    int   = 100,
    threshold: float = 0.95,
    seed:      int   = 1,
    out_dir:   str   = ".",
) -> None:
    """
    For each mechanism at its 'best' (above-threshold) parameter, run n_runs
    random initial conditions and measure P(cut ≥ threshold * best_cut).
    """
    rng = np.random.default_rng(seed)
    N   = W.shape[0]

    # Choose parameters: OIM just above mu_bin; Hopfield large β; Spectral k=11
    mu_bin   = _oim_mu_bin(W)
    mu_best  = max(mu_bin * 1.5, mu_bin + 0.3)
    beta_best = 15.0
    k_best    = 11

    print(f"  [Exp2] μ_best={mu_best:.3f}  β_best={beta_best}  k_best={k_best}")
    print(f"  [Exp2] Running {n_runs} trials per mechanism ...")

    # --- Shared initial conditions ---
    phi0s = [rng.uniform(0.0, 2 * np.pi, N) for _ in range(n_runs)]
    u0s   = [rng.uniform(-0.01, 0.01, N)   for _ in range(n_runs)]

    def collect(runner_fn, inits):
        cuts, binarized = [], []
        for ic in inits:
            r = runner_fn(ic)
            binarized.append(r["is_binary"])
            cuts.append(r["cut"] if r["is_binary"] and not np.isnan(r["cut"]) else 0.0)
        return np.array(cuts), np.array(binarized)

    cuts_A, bin_A = collect(lambda ic: run_oim_trial(W, mu_best, ic),     phi0s)
    cuts_B, bin_B = collect(lambda ic: run_hopfield_trial(W, beta_best, ic), u0s)
    cuts_C, bin_C = collect(lambda ic: run_spectral_trial(W, k_best, ic),  phi0s)

    success_A = float(np.mean(cuts_A >= threshold * best_cut))
    success_B = float(np.mean(cuts_B >= threshold * best_cut))
    success_C = float(np.mean(cuts_C >= threshold * best_cut))
    frac_bin_A = float(np.mean(bin_A))
    frac_bin_B = float(np.mean(bin_B))
    frac_bin_C = float(np.mean(bin_C))

    print(f"\n  {'Mechanism':<20} {'μ/β/k':<12} {'P(success)':<14} {'Frac binarized':<16} {'Mean cut'}")
    print("  " + "─" * 72)
    for name, p, succ, fb, cuts in [
        ("A — OIM (penalty)",   f"μ={mu_best:.2f}",  success_A, frac_bin_A, cuts_A),
        ("B — Hopfield (gain)", f"β={beta_best}",   success_B, frac_bin_B, cuts_B),
        ("C — Spectral (k)",    f"k={k_best}",      success_C, frac_bin_C, cuts_C),
    ]:
        mean_c = float(np.mean(cuts[cuts > 0])) if np.any(cuts > 0) else 0.0
        print(f"  {name:<20} {p:<12} {succ:<14.3f} {fb:<16.3f} {mean_c:.2f}")
    print(f"  {'Best known cut':>52} {best_cut:.2f}")

    # --- Plot histograms ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle(
        f"Experiment 2 — Success-Rate Benchmark ({n_runs} runs, "
        f"success ≥ {threshold*100:.0f}% of best={best_cut:.0f}, N={N})",
        fontsize=12, fontweight="bold",
    )
    colors = ["#e07b39", "#4fa3a8", "#6a4fa3"]
    for ax, cuts, frac, name, col, p_str, succ in [
        (axes[0], cuts_A, frac_bin_A, "A — OIM (penalty)",   colors[0], f"μ={mu_best:.2f}",  success_A),
        (axes[1], cuts_B, frac_bin_B, "B — Hopfield (gain)", colors[1], f"β={beta_best}",   success_B),
        (axes[2], cuts_C, frac_bin_C, "C — Spectral (k)",    colors[2], f"k={k_best}",      success_C),
    ]:
        active = cuts[cuts > 0]
        if len(active) > 0:
            bins = np.arange(active.min() - 0.5, active.max() + 1.5, 1.0)
            ax.hist(active, bins=bins, color=col, edgecolor="black",
                    alpha=0.8, zorder=2)
        ax.axvline(best_cut,           color="black", linestyle="--",
                   linewidth=2, label=f"Best known = {best_cut:.0f}", zorder=4)
        ax.axvline(threshold * best_cut, color="red", linestyle=":",
                   linewidth=1.5, label=f"95% threshold = {threshold*best_cut:.1f}", zorder=4)
        ax.set_xlabel("Cut value", fontsize=11)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(
            f"{name}\n{p_str}  |  P(success)={succ:.2f}  |  "
            f"Binarized={frac:.2f}",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(out_dir, "exp2_success_rate.png")
    fig.savefig(path, dpi=130)
    print(f"  [Exp2] Saved → {path}")
    plt.close(fig)


# ===========================================================================
# Experiment 3 — Hardness landscape: stability threshold vs cut quality
# ===========================================================================

def exp3_hardness_landscape(
    W:        np.ndarray,
    best_cut: float,
    out_dir:  str = ".",
    warn_N:   int = 18,
) -> None:
    """
    For every binary equilibrium φ* ∈ {0,π}^N:
      - Compute stability threshold μ*_i = λ_max(D(φ*_i))   [OIM theory, Cheng 2024]
      - Compute corresponding cut value
      - Plot threshold vs cut to reveal the 'hardness landscape'

    Key insight from the paper: better cuts are NOT always more stable.
    Some suboptimal equilibria have lower thresholds (are stabilized first as
    μ increases), while globally optimal solutions may require larger μ to
    be reached reliably.

    NOTE: Mechanism C (Spectral Shaping) produces the SAME D-matrix and
    hence the SAME landscape — the stability of binary equilibria is
    independent of k. What changes with k is the non-binary landscape
    (spurious equilibria), not the binary one.
    """
    N = W.shape[0]
    if N > warn_N:
        print(f"  [Exp3] N={N} > {warn_N}: enumerating 2^{N}={2**N} configs — may be slow.")
    J = -W   # convention for D-matrix

    thresholds, cuts, labels = [], [], []
    for bits in iproduct([0, 1], repeat=N):
        phi  = np.array(bits, dtype=float) * np.pi
        # D-matrix: D_ij = J_ij cos(φ_i - φ_j) = -W_ij cos(φ_i - φ_j)
        diff = phi[:, None] - phi[None, :]
        D    = J * np.cos(diff)
        np.fill_diagonal(D, 0.0)
        np.fill_diagonal(D, -D.sum(axis=1))
        mu_star = float(np.linalg.eigvalsh(D).max())

        sigma = np.where(np.array(bits) == 0, 1.0, -1.0)
        cut   = cut_from_spins(W, sigma)
        thresholds.append(mu_star)
        cuts.append(cut)
        labels.append(bits)

    thresholds = np.array(thresholds)
    cuts       = np.array(cuts)

    # Colour by cut quality
    norm  = mcolors.Normalize(cuts.min(), cuts.max())
    cmap  = plt.get_cmap("RdYlGn")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Experiment 3 — Hardness Landscape (N={N}, "
        f"all 2^N={2**N} binary equilibria)\n"
        r"Stability threshold $\mu^*_i = \lambda_{max}(D(\phi^*_i))$"
        " — independent of mechanism parameter (μ, β, or k)",
        fontsize=11, fontweight="bold",
    )

    # Panel A: scatter threshold vs cut
    ax = axes[0]
    sc = ax.scatter(cuts, thresholds, c=cuts, cmap=cmap, norm=norm,
                    s=30, alpha=0.75, edgecolors="none")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.2,
               label=r"$\mu^*=0$ (always stable)")
    ax.axvline(best_cut, color="purple", linestyle=":", linewidth=1.5,
               label=f"Optimal cut = {best_cut:.0f}")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Cut value", fontsize=9)
    ax.set_xlabel("Cut value", fontsize=11)
    ax.set_ylabel(r"Stability threshold $\mu^*_i$", fontsize=11)
    ax.set_title(
        "Each point = one binary equilibrium\n"
        "Green = better cut;  lower threshold = stabilized at lower μ",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: histogram of thresholds, coloured by cut quality
    ax2 = axes[1]
    # Separate good vs bad cuts (above/below median)
    median_cut = np.median(cuts)
    good_mask  = cuts >= median_cut
    bad_mask   = ~good_mask
    ax2.hist(thresholds[bad_mask],  bins=20, alpha=0.65, color="#d62728",
             label=f"Cut < median ({median_cut:.0f})", edgecolor="none")
    ax2.hist(thresholds[good_mask], bins=20, alpha=0.65, color="#2ca02c",
             label=f"Cut ≥ median ({median_cut:.0f})", edgecolor="none")
    ax2.axvline(0.0, color="k", linestyle="--", linewidth=1.2)
    ax2.set_xlabel(r"Stability threshold $\mu^*_i$", fontsize=11)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title(
        "Distribution of thresholds by cut quality\n"
        "Overlap shows that stability ≠ optimality",
        fontsize=9,
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(out_dir, "exp3_hardness_landscape.png")
    fig.savefig(path, dpi=130)
    print(f"  [Exp3] Saved → {path}")
    plt.close(fig)

    # Print summary table
    opt_mask  = np.isclose(cuts, best_cut)
    sub_mask  = ~opt_mask
    print(f"  [Exp3] Optimal equilibria  (cut={best_cut:.0f}): "
          f"μ* range [{thresholds[opt_mask].min():.3f}, {thresholds[opt_mask].max():.3f}]")
    if sub_mask.any():
        print(f"  [Exp3] Suboptimal equilibria: "
              f"μ* range [{thresholds[sub_mask].min():.3f}, {thresholds[sub_mask].max():.3f}]")
    n_always_stable = int(np.sum(thresholds <= 0))
    print(f"  [Exp3] Always-stable (μ*≤0): {n_always_stable}/{2**N} "
          f"— stable even without SHIL/penalty")


# ===========================================================================
# Experiment 4 — Convergence speed
# ===========================================================================

def exp4_convergence_speed(
    W:        np.ndarray,
    n_runs:   int   = 20,
    bin_tol:  float = 1e-2,
    seed:     int   = 2,
    out_dir:  str   = ".",
) -> None:
    """
    For each mechanism across a range of parameter values, measure the
    'convergence time': the integration time T* at which the trajectory
    first satisfies binarization_residual < bin_tol.

    Uses dense time-point output and searches for the crossing.
    Reports: median T* and fraction of runs that converged before T_max.
    """
    rng    = np.random.default_rng(seed)
    N      = W.shape[0]
    T_MAX  = 60.0
    N_PTS  = 600
    T_EVAL = np.linspace(0, T_MAX, N_PTS)
    J      = -W

    mu_bin  = _oim_mu_bin(W)
    mu_vals = np.linspace(max(0.1, mu_bin * 0.6), mu_bin * 2.5, 12)
    beta_vals = [1.0, 2.0, 4.0, 8.0, 12.0, 20.0]
    k_vals    = [1, 3, 5, 7, 9, 11]

    def first_binary_time(trajectory: np.ndarray, criterion_fn, t_eval) -> float:
        """Return first t where criterion_fn(trajectory[:, t]) < bin_tol, else NaN."""
        for i, t in enumerate(t_eval):
            if criterion_fn(trajectory[:, i]) < bin_tol:
                return float(t)
        return np.nan

    def oim_residual_fn(phi):
        return float(np.max(np.abs(np.sin(phi))))

    def hop_residual_fn(beta):
        def _fn(u):
            return float(np.max(1.0 - np.abs(np.tanh(beta * u))))
        return _fn

    def spectral_residual_fn(phi):
        return float(np.max(np.abs(np.sin(phi))))

    print("  [Exp4] Computing convergence times ...")

    def conv_sweep(runner_fn, param_vals, init_fn, res_fn_maker):
        med_times, frac_conv = [], []
        for p in param_vals:
            times = []
            res_fn = res_fn_maker(p)
            for _ in range(n_runs):
                ic  = init_fn(rng, N)
                sol = runner_fn(p, ic)
                if sol is None:
                    times.append(np.nan)
                    continue
                t_star = first_binary_time(sol.y, res_fn, T_EVAL)
                times.append(t_star)
            times = np.array(times)
            med_times.append(float(np.nanmedian(times)))
            frac_conv.append(float(np.mean(~np.isnan(times))))
        return np.array(med_times), np.array(frac_conv)

    def _oim_runner(mu, phi0):
        oim = OIMMaxCut(W, mu=mu, seed=0)
        return oim.simulate(phi0, t_span=(0, T_MAX), n_points=N_PTS,
                            rtol=1e-7, atol=1e-7)

    def _hop_runner(beta, u0):
        hop = HopfieldMaxCut(W, beta=beta)
        return hop.simulate(u0, t_span=(0, T_MAX), n_points=N_PTS)

    def _spec_runner(k, phi0):
        ss = OIM_SpectralShaping(J, k=int(k))
        return ss.simulate(phi0, t_span=(0, T_MAX), n_points=N_PTS)

    phase_init = lambda rng, N: rng.uniform(0, 2 * np.pi, N)
    small_init = lambda rng, N: rng.uniform(-0.01, 0.01, N)

    med_A, fc_A = conv_sweep(_oim_runner,  mu_vals,   phase_init, lambda mu:   oim_residual_fn)
    med_B, fc_B = conv_sweep(_hop_runner,  beta_vals, small_init, lambda beta: hop_residual_fn(beta))
    med_C, fc_C = conv_sweep(_spec_runner, k_vals,    phase_init, lambda k:    spectral_residual_fn)

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Experiment 4 — Convergence Speed  (N={N}, {n_runs} runs/point, "
        f"binarization tol={bin_tol})",
        fontsize=12, fontweight="bold",
    )
    specs = [
        (axes[0], mu_vals,   med_A, fc_A, "#e07b39", r"$\mu$",    "A — OIM (penalty)"),
        (axes[1], beta_vals, med_B, fc_B, "#4fa3a8", r"$\beta$",  "B — Hopfield (gain)"),
        (axes[2], k_vals,    med_C, fc_C, "#6a4fa3", r"$k$",      "C — Spectral (k)"),
    ]
    for ax, grid, med, fc, col, lbl, title in specs:
        grid = np.array(grid)
        ax2  = ax.twinx()
        l1,  = ax.plot(grid, med, "o-",  color=col,     markersize=6, linewidth=2,
                       label="Median T*")
        l2,  = ax2.plot(grid, fc, "s--", color="gray",  markersize=5, linewidth=1.5,
                        label="Frac. converged")
        ax.set_xlabel(lbl, fontsize=11)
        ax.set_ylabel("Median convergence time T*", fontsize=10, color=col)
        ax2.set_ylabel("Fraction converged", fontsize=10, color="gray")
        ax2.set_ylim(0, 1.15)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(handles=[l1, l2], fontsize=8, loc="upper right")
    fig.tight_layout()
    path = os.path.join(out_dir, "exp4_convergence_speed.png")
    fig.savefig(path, dpi=130)
    print(f"  [Exp4] Saved → {path}")
    plt.close(fig)


# ===========================================================================
# Experiment 5 — Phase trajectories (visual sanity check)
# ===========================================================================

def exp5_phase_trajectories(
    W:       np.ndarray,
    seed:    int = 3,
    out_dir: str = ".",
) -> None:
    """
    For a given graph (preferably small N), show 3 side-by-side panels of
    phase trajectories θ_i(t) for a single representative run — one per
    mechanism at a 'good' parameter value.  Visually confirms phase splitting.

    For Hopfield, shows tanh(β u_i(t)) instead of a raw phase.
    """
    rng    = np.random.default_rng(seed)
    N      = W.shape[0]
    J      = -W
    mu_bin = _oim_mu_bin(W)
    mu     = max(mu_bin * 1.5, mu_bin + 0.5)
    beta   = 12.0
    k      = 9

    phi0 = rng.uniform(0.0, 2 * np.pi, N)
    u0   = rng.uniform(-0.01, 0.01, N)
    T    = 50.0
    NPT  = 500

    # Simulate
    oim  = OIMMaxCut(W, mu=mu, seed=0)
    sol_A = oim.simulate(phi0.copy(), t_span=(0, T), n_points=NPT)
    hop   = HopfieldMaxCut(W, beta=beta)
    sol_B = hop.simulate(u0.copy(), t_span=(0, T), n_points=NPT)
    ss    = OIM_SpectralShaping(J, k=k)
    sol_C = ss.simulate(phi0.copy(), t_span=(0, T), n_points=NPT)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Experiment 5 — Phase Trajectories (N={N})\n"
        f"A: μ={mu:.2f} | B: β={beta} | C: k={k}",
        fontsize=12, fontweight="bold",
    )
    cmap = plt.get_cmap("tab10")

    # Panel A: OIM phases θ_i(t) mod 2π
    ax = axes[0]
    if sol_A is not None:
        t = sol_A.t
        for i in range(N):
            ax.plot(t, sol_A.y[i] % (2 * np.pi), color=cmap(i / N), linewidth=1.2)
        ax.axhline(0,           color="k",    linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(np.pi,       color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(2 * np.pi,   color="k",    linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Time", fontsize=11); ax.set_ylabel(r"$\theta_i \mod 2\pi$", fontsize=11)
    ax.set_title(f"A — OIM (μ={mu:.2f})", fontsize=11)
    ax.set_yticks([0, np.pi, 2 * np.pi]); ax.set_yticklabels(["0", "π", "2π"])
    ax.grid(True, alpha=0.25)

    # Panel B: Hopfield V_i(t) = tanh(β u_i(t))
    ax = axes[1]
    if sol_B is not None:
        t = sol_B.t
        for i in range(N):
            V = np.tanh(beta * sol_B.y[i])
            ax.plot(t, V, color=cmap(i / N), linewidth=1.2)
        ax.axhline( 1, color="k",    linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(-1, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline( 0, color="k",    linestyle=":",  linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Time", fontsize=11); ax.set_ylabel(r"$V_i = \tanh(\beta u_i)$", fontsize=11)
    ax.set_title(f"B — Hopfield (β={beta})", fontsize=11)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.25)

    # Panel C: Spectral phases θ_i(t) mod 2π
    ax = axes[2]
    if sol_C is not None:
        t = sol_C.t
        for i in range(N):
            ax.plot(t, sol_C.y[i] % (2 * np.pi), color=cmap(i / N), linewidth=1.2)
        ax.axhline(0,         color="k",    linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(np.pi,     color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.axhline(2 * np.pi, color="k",    linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Time", fontsize=11); ax.set_ylabel(r"$\theta_i mod 2\pi$", fontsize=11)
    ax.set_title(f"C — Spectral Shaping (k={k})", fontsize=11)
    ax.set_yticks([0, np.pi, 2 * np.pi]); ax.set_yticklabels(["0", "π", "2π"])
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    path = os.path.join(out_dir, "exp5_phase_trajectories.png")
    fig.savefig(path, dpi=130)
    print(f"  [Exp5] Saved → {path}")
    plt.close(fig)


# ===========================================================================
# Experiment 6 — Coupling function shape (qualitative, Mechanism C vs A)
# ===========================================================================

def exp6_coupling_shapes(out_dir: str = ".") -> None:
    """
    Plot the coupling function g_k(φ) for different k values alongside the
    OIM sin(φ) baseline (k=1) and the target square wave sign(sin(φ)).
    Shows how k controls the sharpness of the interaction.
    """
    phi = np.linspace(-np.pi, np.pi, 1000)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Experiment 6 — Coupling Function Shapes\n"
        r"$g_k(\varphi) = \sum_{n\,\mathrm{odd},\,n\leq k} c_n \sin(n\varphi)$",
        fontsize=12, fontweight="bold",
    )

    # Panel A: equal-weight coefficients c_n=1
    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    k_vals = [1, 3, 5, 9, 15]
    for i, k in enumerate(k_vals):
        ss  = OIM_SpectralShaping(np.eye(1), k=k)   # dummy J, only need g_k
        g   = ss.g_k(phi)
        col = cmap(i / len(k_vals))
        ax.plot(phi / np.pi, g, color=col, linewidth=1.8, label=f"k={k}")
    ax.plot(phi / np.pi, np.sign(np.sin(phi)), "k:", linewidth=1.0,
            alpha=0.6, label="sgn(sin) (k→∞)")
    ax.set_xlabel(r"$\varphi / \pi$", fontsize=11)
    ax.set_ylabel(r"$g_k(\varphi)$", fontsize=11)
    ax.set_title("Equal weights ($c_n = 1$)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: square-wave coefficients c_n = 4/(nπ)
    ax = axes[1]
    for i, k in enumerate(k_vals):
        ss  = OIM_SpectralShaping(np.eye(1), k=k, coeffs="square_wave")
        g   = ss.g_k(phi)
        col = cmap(i / len(k_vals))
        ax.plot(phi / np.pi, g, color=col, linewidth=1.8, label=f"k={k}")
    ax.plot(phi / np.pi, np.sign(np.sin(phi)), "k:", linewidth=1.0,
            alpha=0.6, label="sgn(sin) (k→∞)")
    ax.set_xlabel(r"$\varphi / \pi$", fontsize=11)
    ax.set_ylabel(r"$g_k(\varphi)$", fontsize=11)
    ax.set_title(r"Square-wave weights ($c_n = 4/(n\pi)$, Gibbs-optimal)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "exp6_coupling_shapes.png")
    fig.savefig(path, dpi=130)
    print(f"  [Exp6] Saved → {path}")
    plt.close(fig)


# ===========================================================================
# Helper: OIM binarization threshold
# ===========================================================================

def _oim_mu_bin(W: np.ndarray) -> float:
    """
    Compute OIM binarization threshold using D-matrix analysis.
    Matches OIMMaxCut.binarization_threshold()['mu_bin'] from OIM_mu_v2.py.
    mu_bin = min_{φ* ∈ {0,π}^N} λ_max(D(φ*))
    (Note: NOT clamped to 0 — follows the mu_v2 convention.)
    """
    N   = W.shape[0]
    J   = -W
    best = np.inf
    for bits in iproduct([0, 1], repeat=N):
        phi  = np.array(bits, dtype=float) * np.pi
        diff = phi[:, None] - phi[None, :]
        D    = J * np.cos(diff)
        np.fill_diagonal(D, 0.0)
        np.fill_diagonal(D, -D.sum(axis=1))
        lmax = float(np.linalg.eigvalsh(D).max())
        if lmax < best:
            best = lmax
    return best


# ===========================================================================
# Main entry point
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="MaxCut benchmark: OIM vs Hopfield vs Spectral Shaping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--N",      type=int,   default=10,   help="Graph size")
    p.add_argument("--p",      type=float, default=0.7,  help="Edge probability (ER graph)")
    p.add_argument("--seed",   type=int,   default=42,   help="Global random seed")
    p.add_argument("--runs",   type=int,   default=50,
                   help="Trials per parameter value / benchmark")
    p.add_argument("--out",    type=str,   default=".",  help="Output directory for figures")
    p.add_argument("--exps",   type=str,   default="123456",
                   help="Experiments to run, e.g. '125' for Exp 1,2,5")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print("=" * 65)
    print("  MaxCut Benchmark: OIM (A) vs Hopfield (B) vs Spectral (C)")
    print("=" * 65)

    # ── Build graph ────────────────────────────────────────────────
    W = random_graph(args.N, p=args.p, seed=args.seed)
    n_edges = int(np.sum(W) // 2)
    print(f"\nGraph: N={args.N}, edges={n_edges}, p={args.p}, seed={args.seed}")

    # ── Ground truth (brute force for small N) ─────────────────────
    if args.N <= 18:
        print("  Computing exact MaxCut (brute force) ...")
        t0       = time.time()
        best_cut, best_sigma = brute_force_maxcut(W)
        print(f"  Optimal cut = {best_cut:.0f}  (found in {time.time()-t0:.2f}s)")
    else:
        print(f"  N={args.N} > 18: using OIM estimate for best_cut (50 random runs).")
        cuts = []
        for _ in range(50):
            phi0 = np.random.uniform(0, 2 * np.pi, args.N)
            r    = run_oim_trial(W, mu=3.0, phi0=phi0)
            if not np.isnan(r["cut"]):
                cuts.append(r["cut"])
        best_cut = float(max(cuts)) if cuts else 1.0
        print(f"  Best cut found = {best_cut:.0f}")

    exps = args.exps
    n    = args.runs

    if "1" in exps:
        print("\n" + "─" * 45 + "\n  EXPERIMENT 1 — Parameter Sweep\n" + "─" * 45)
        exp1_parameter_sweep(W, best_cut, n_runs=n, seed=args.seed, out_dir=args.out)

    if "2" in exps:
        print("\n" + "─" * 45 + "\n  EXPERIMENT 2 — Success-Rate Benchmark\n" + "─" * 45)
        exp2_success_rate(W, best_cut, n_runs=n, seed=args.seed + 1, out_dir=args.out)

    if "3" in exps and args.N <= 18:
        print("\n" + "─" * 45 + "\n  EXPERIMENT 3 — Hardness Landscape\n" + "─" * 45)
        exp3_hardness_landscape(W, best_cut, out_dir=args.out)
    elif "3" in exps:
        print(f"\n  [Exp3] Skipping (N={args.N} > 18: 2^N too large to enumerate)")

    if "4" in exps:
        print("\n" + "─" * 45 + "\n  EXPERIMENT 4 — Convergence Speed\n" + "─" * 45)
        exp4_convergence_speed(W, n_runs=min(n, 20), seed=args.seed + 2, out_dir=args.out)

    if "5" in exps:
        print("\n" + "─" * 45 + "\n  EXPERIMENT 5 — Phase Trajectories\n" + "─" * 45)
        exp5_phase_trajectories(W, seed=args.seed + 3, out_dir=args.out)

    if "6" in exps:
        print("\n" + "─" * 45 + "\n  EXPERIMENT 6 — Coupling Shapes\n" + "─" * 45)
        exp6_coupling_shapes(out_dir=args.out)

    print("\n" + "=" * 65)
    print("  All experiments done.  Figures saved to:", os.path.abspath(args.out))
    print("=" * 65)


if __name__ == "__main__":
    main()
