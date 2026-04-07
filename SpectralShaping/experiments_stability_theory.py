"""
experiments_stability_theory.py
================================
Numerical experiments investigating WHY almost no binary equilibria are
naturally stable (lambda_D <= 0) and how stability relates to cut quality.

Four focused experiments, all derived from the same core identity:

    trace(D) = 2E - 4C     (E = edges, C = cut value)

Experiment 1 — The trace identity
    Verify numerically that trace(D) = 2E - 4C for every binary equilibrium.
    Plot trace(D) vs cut value: perfect linear relationship, no scatter.

Experiment 2 — Gershgorin bound and the diagonal structure
    For each equilibrium, the diagonal entry D_ii = n_same(i) - n_cut(i).
    By Gershgorin: lambda_max(D) >= max_i D_ii.
    Show that this lower bound already explains most of the trend and that
    lambda_max = 0 requires n_same(i) = 0 for all i.

Experiment 3 — Bipartite vs non-bipartite graphs
    The theorem: for bipartite graphs, exactly the 2 optimal partitions achieve
    lambda_D = 0. For non-bipartite graphs (odd cycles, Petersen, etc.),
    lambda_D > 0 for EVERY binary configuration — the system is never naturally
    stable regardless of solution quality.
    Compare hardness landscapes side by side.

Experiment 4 — Full eigenvalue spectrum as cut quality changes
    Pick 4 equilibria of increasing cut quality from C10.
    Plot the full eigenvalue spectrum of D for each.
    Show the spectrum shifting leftward as cut quality improves.

Usage
-----
    python experiments_stability_theory.py [--out .] [--graph c10.txt]
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import product as iproduct
from typing import Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ===========================================================================
# Graph constructors
# ===========================================================================

def cycle_plus_diagonals(N: int, diagonal_step: int) -> np.ndarray:
    """N-cycle with chords connecting nodes i and i+diagonal_step mod N."""
    W = np.zeros((N, N))
    for i in range(N):
        W[i, (i + 1) % N] = 1
        W[(i + 1) % N, i] = 1
        j = (i + diagonal_step) % N
        if i != j:
            W[i, j] = 1
            W[j, i] = 1
    np.fill_diagonal(W, 0)
    return W


def load_graph_from_file(path: str) -> Tuple[np.ndarray, str]:
    """Load edge-list file: first non-comment line = N, then 'u v' pairs."""
    lines, comments = [], []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                comments.append(line)
            else:
                lines.append(line)
    N = int(lines[0])
    W = np.zeros((N, N))
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 2:
            u, v = int(parts[0]), int(parts[1])
            W[u, v] = W[v, u] = 1.0
    np.fill_diagonal(W, 0)
    return W, "\n".join(comments)


def petersen_graph() -> np.ndarray:
    """
    The Petersen graph: 10 nodes, 15 edges, 3-regular, non-bipartite.
    Outer pentagon: 0-1-2-3-4-0
    Inner pentagram: 5-7-9-6-8-5
    Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    MaxCut = 12 (cannot achieve all 15 edges cut — non-bipartite).
    """
    edges = [
        (0,1),(1,2),(2,3),(3,4),(4,0),   # outer
        (5,7),(7,9),(9,6),(6,8),(8,5),   # inner
        (0,5),(1,6),(2,7),(3,8),(4,9),   # spokes
    ]
    W = np.zeros((10, 10))
    for u, v in edges:
        W[u, v] = W[v, u] = 1.0
    return W


def is_bipartite(W: np.ndarray) -> Tuple[bool, np.ndarray]:
    """
    BFS 2-coloring check. Returns (is_bipartite, coloring).
    Coloring is an array in {0, 1, -1} where -1 = unvisited.
    """
    N = W.shape[0]
    color = -np.ones(N, dtype=int)
    color[0] = 0
    queue = [0]
    while queue:
        u = queue.pop(0)
        for v in np.where(W[u] > 0)[0]:
            if color[v] == -1:
                color[v] = 1 - color[u]
                queue.append(v)
            elif color[v] == color[u]:
                return False, color
    return True, color


def brute_force_maxcut(W: np.ndarray) -> Tuple[float, np.ndarray]:
    N = W.shape[0]
    best_cut, best_s = 0.0, np.ones(N)
    for bits in iproduct([-1, 1], repeat=N):
        s = np.array(bits, dtype=float)
        cut = 0.25 * float(np.sum(W * (1.0 - s[:, None] * s[None, :])))
        if cut > best_cut:
            best_cut, best_s = cut, s.copy()
    return best_cut, best_s


# ===========================================================================
# D-matrix core
# ===========================================================================

def build_D(W: np.ndarray, phi_star: np.ndarray) -> np.ndarray:
    """
    Signed-Laplacian D(phi*) from Cheng et al. 2024 / our implementation.
    J = -W, so D_ij = J_ij cos(phi_i - phi_j) = -W_ij cos(phi_i - phi_j).
    At binary phi*: D_ij = +1 for cut edges, -1 for same-partition edges.
    """
    J    = -W
    diff = phi_star[:, None] - phi_star[None, :]
    D    = J * np.cos(diff)
    np.fill_diagonal(D, 0.0)
    np.fill_diagonal(D, -D.sum(axis=1))
    return D


def enumerate_equilibria(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Enumerate all 2^N binary equilibria. Returns:
        cuts      : (2^N,)  cut values
        lambda_D  : (2^N,)  lambda_max(D)
        traces    : (2^N,)  trace(D)
    """
    N     = W.shape[0]
    E     = int(np.sum(W) // 2)
    cuts, lambdas, traces = [], [], []

    for bits in iproduct([0, 1], repeat=N):
        phi   = np.array(bits, dtype=float) * np.pi
        sigma = np.where(np.array(bits) == 0, 1.0, -1.0)
        cut   = float(0.25 * np.sum(W * (1.0 - sigma[:, None] * sigma[None, :])))
        D     = build_D(W, phi)
        lam   = float(np.linalg.eigvalsh(D).max())
        tr    = float(np.trace(D))
        cuts.append(cut)
        lambdas.append(lam)
        traces.append(tr)

    return np.array(cuts), np.array(lambdas), np.array(traces)


def per_node_analysis(W: np.ndarray, phi_star: np.ndarray) -> np.ndarray:
    """
    For each node i: D_ii = n_same(i) - n_cut(i).
    Returns array of shape (N,).
    """
    N = W.shape[0]
    sigma = np.where(phi_star < np.pi / 2, 1.0, -1.0)
    diag  = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if W[i, j] > 0:
                if sigma[i] == sigma[j]:   # same partition
                    diag[i] += 1
                else:                       # cut
                    diag[i] -= 1
    return diag


# ===========================================================================
# Experiment 1 — Trace identity: trace(D) = 2E - 4C
# ===========================================================================

def exp1_trace_identity(W: np.ndarray, name: str, out_dir: str) -> None:
    """
    Verify numerically that trace(D) = 2E - 4C (perfectly linear in C).
    This is the algebraic backbone explaining the lambda_D vs cut trend.
    """
    N = W.shape[0]
    E = int(np.sum(W) // 2)
    print(f"  [Exp 1] Enumerating {2**N} equilibria for {name} (N={N}, E={E}) ...")

    cuts, lambdas, traces = enumerate_equilibria(W)

    # Theoretical prediction: trace(D) = 2E - 4C
    C_range = np.linspace(cuts.min(), cuts.max(), 200)
    trace_theory = 2 * E - 4 * C_range

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Experiment 1 — Trace identity: trace(D) = 2E − 4C  [{name}, N={N}, E={E}]\n"
        r"This linear shift of the eigenvalue sum is the algebraic driver of the $\lambda_D$–cut trend",
        fontsize=11, fontweight="bold",
    )

    # Panel A: trace(D) vs cut — should be perfectly linear
    ax = axes[0]
    ax.scatter(cuts, traces, s=8, alpha=0.4, color="#7F77DD", label="Enumerated equilibria")
    ax.plot(C_range, trace_theory, "r-", linewidth=2, label=r"Theory: $2E - 4C$")
    ax.set_xlabel("Cut value C", fontsize=12)
    ax.set_ylabel("trace(D(φ*))", fontsize=12)
    ax.set_title("Perfect linear relationship — no scatter", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: lambda_D vs cut, coloured by trace(D)
    ax = axes[1]
    sc = ax.scatter(cuts, lambdas, c=traces, cmap="coolwarm_r",
                    s=12, alpha=0.6, edgecolors="none")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.5,
               label=r"$\lambda_D = 0$ (stability boundary)")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("trace(D)", fontsize=9)
    ax.set_xlabel("Cut value C", fontsize=12)
    ax.set_ylabel(r"$\lambda_{\max}(D)$", fontsize=12)
    ax.set_title(
        r"$\lambda_{\max}(D)$ vs cut, coloured by trace(D)" + "\n"
        "Negative trace (blue) correlates with lower λ_D",
        fontsize=9,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, f"exp1_trace_identity_{name}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  [Exp 1] Saved → {path}")

    # Print the key numbers
    print(f"  [Exp 1] Verification: max |trace(D) - (2E-4C)| = "
          f"{max(abs(tr - (2*E - 4*c)) for tr, c in zip(traces, cuts)):.2e}  (should be ~0)")


# ===========================================================================
# Experiment 2 — Gershgorin bound and diagonal structure
# ===========================================================================

def exp2_gershgorin(W: np.ndarray, name: str, out_dir: str) -> None:
    """
    Show that lambda_max(D) >= max_i D_ii (Gershgorin lower bound).
    Since D_ii = n_same(i) - n_cut(i), a bad cut always has positive D_ii
    for some nodes, guaranteeing lambda_max > 0.

    Also show: lambda_max = 0 requires n_same(i) = 0 for all i,
    i.e. every single edge must be cut.
    """
    N = W.shape[0]
    E = int(np.sum(W) // 2)
    print(f"  [Exp 2] Computing Gershgorin bounds for {name} ...")

    cuts, lambdas, _ = enumerate_equilibria(W)

    # Gershgorin lower bound: max_i D_ii
    gersh_lower = []
    max_same    = []   # max_i n_same(i)
    for bits in iproduct([0, 1], repeat=N):
        phi   = np.array(bits, dtype=float) * np.pi
        d_diag = per_node_analysis(W, phi)
        gersh_lower.append(float(d_diag.max()))
        sigma  = np.where(np.array(bits) == 0, 1.0, -1.0)
        n_same_all = [(W[i] * (sigma == sigma[i])).sum() for i in range(N)]
        max_same.append(float(max(n_same_all)))

    gersh_lower = np.array(gersh_lower)
    max_same    = np.array(max_same)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Experiment 2 — Gershgorin bound and diagonal structure  [{name}]\n"
        r"$D_{ii} = n_{\mathrm{same}}(i) - n_{\mathrm{cut}}(i)$  → "
        r"$\lambda_{\max}(D) \geq \max_i D_{ii}$",
        fontsize=11, fontweight="bold",
    )

    # Panel A: lambda_D vs Gershgorin lower bound
    ax = axes[0]
    vmin = min(lambdas.min(), gersh_lower.min())
    vmax = max(lambdas.max(), gersh_lower.max())
    ax.scatter(gersh_lower, lambdas, s=8, alpha=0.35, color="#D85A30")
    ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1.2, label="y = x  (exact)")
    ax.set_xlabel(r"Gershgorin lower bound: $\max_i D_{ii}$", fontsize=11)
    ax.set_ylabel(r"$\lambda_{\max}(D)$", fontsize=11)
    ax.set_title(r"$\lambda_{\max}(D) \geq \max_i D_{ii}$ always", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel B: max n_same(i) vs lambda_D
    ax = axes[1]
    ax.scatter(max_same, lambdas, s=8, alpha=0.35, color="#1D9E75")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.5,
               label=r"$\lambda_D = 0$")
    ax.axvline(0, color="gray", linestyle=":", linewidth=1.2,
               label=r"$\max n_{\mathrm{same}} = 0$ → all edges cut")
    ax.set_xlabel(r"$\max_i\, n_{\mathrm{same}}(i)$", fontsize=11)
    ax.set_ylabel(r"$\lambda_{\max}(D)$", fontsize=11)
    ax.set_title(
        r"$\lambda_D = 0$ requires $\max_i n_{\mathrm{same}}(i) = 0$" + "\n"
        "i.e. every edge must be cut (only possible if bipartite)",
        fontsize=9,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel C: Gershgorin bound tightness (lambda_D - max_i D_ii)
    ax = axes[2]
    gap = lambdas - gersh_lower
    ax.hist(gap, bins=30, color="#7F77DD", alpha=0.8, edgecolor="none")
    ax.set_xlabel(r"$\lambda_{\max}(D) - \max_i D_{ii}$  (Gershgorin gap)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(
        "How tight is the Gershgorin bound?\n"
        "Gap near 0 → bound is tight (diagonal drives λ_max)",
        fontsize=9,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(out_dir, f"exp2_gershgorin_{name}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  [Exp 2] Saved → {path}")


# ===========================================================================
# Experiment 3 — Bipartite vs non-bipartite
# ===========================================================================

def exp3_bipartite_vs_nonbipartite(out_dir: str) -> None:
    """
    Compare the hardness landscape (lambda_D vs cut) for:
      (a) C10 — bipartite, 3-regular: optimal partition achieves lambda_D = 0
      (b) Petersen graph — non-bipartite, 3-regular: lambda_D > 0 for ALL equilibria

    This is the key structural result: whether ANY equilibrium is naturally
    stable depends entirely on whether the graph is bipartite.
    """
    graphs = {
        "C10 (bipartite, 3-regular)": cycle_plus_diagonals(10, 5),
        "Petersen (non-bipartite, 3-regular)": petersen_graph(),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Experiment 3 — Bipartite vs non-bipartite: structural stability theorem\n"
        r"Bipartite: exactly 2 equilibria reach $\lambda_D = 0$ (optimal).  "
        r"Non-bipartite: $\lambda_D > 0$ for ALL equilibria.",
        fontsize=11, fontweight="bold",
    )

    colors = {"C10 (bipartite, 3-regular)": "#7F77DD",
              "Petersen (non-bipartite, 3-regular)": "#D85A30"}

    for ax, (name, W) in zip(axes, graphs.items()):
        N  = W.shape[0]
        bip, _ = is_bipartite(W)
        best_cut, _ = brute_force_maxcut(W)
        print(f"  [Exp 3] {name}: N={N}, bipartite={bip}, MaxCut={best_cut:.0f}")

        cuts, lambdas, _ = enumerate_equilibria(W)
        opt_mask = np.isclose(cuts, cuts.max())
        col = colors[name]

        sc = ax.scatter(cuts, lambdas, c=cuts, cmap="RdYlGn",
                        vmin=cuts.min(), vmax=cuts.max(),
                        s=20, alpha=0.6, edgecolors="none", zorder=2)
        if opt_mask.any():
            ax.scatter(cuts[opt_mask], lambdas[opt_mask],
                       s=90, edgecolors=col, facecolors="none",
                       linewidths=2, zorder=4,
                       label=f"Optimal (cut={cuts.max():.0f}): {opt_mask.sum()} configs")
        ax.axhline(0, color="black", linestyle="--", linewidth=1.5,
                   label=r"$\lambda_D = 0$")
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("Cut value", fontsize=9)
        ax.set_xlabel("Cut value", fontsize=12)
        ax.set_ylabel(r"$\lambda_{\max}(D)$", fontsize=12)

        n_stable = int((lambdas <= 0).sum())
        bip_str  = "Bipartite" if bip else "Non-bipartite"
        ax.set_title(
            f"{name}\n{bip_str}  |  {n_stable}/{2**N} equilibria have λ_D ≤ 0",
            fontsize=10,
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "exp3_bipartite_vs_nonbipartite.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  [Exp 3] Saved → {path}")


# ===========================================================================
# Experiment 4 — Full eigenvalue spectrum as cut quality changes
# ===========================================================================

def exp4_spectrum_evolution(W: np.ndarray, name: str, out_dir: str) -> None:
    """
    Select 4 representative equilibria of increasing cut quality.
    Plot the full eigenvalue spectrum of D for each.
    Shows the spectrum shifting leftward as cut quality improves.

    Key observation:
    - The zero eigenvalue (from zero row-sum) is always present.
    - The whole spectrum shifts left as trace(D) = 2E - 4C decreases.
    - At optimal (all edges cut): the spectrum is maximally shifted left.
    """
    N = W.shape[0]
    E = int(np.sum(W) // 2)
    best_cut, _ = brute_force_maxcut(W)
    print(f"  [Exp 4] Spectrum analysis for {name} (N={N}, E={E}, MaxCut={best_cut:.0f}) ...")

    # Enumerate all equilibria
    all_bits = list(iproduct([0, 1], repeat=N))
    cuts_all, lambdas_all, spectra_all = [], [], []
    for bits in all_bits:
        phi   = np.array(bits, dtype=float) * np.pi
        sigma = np.where(np.array(bits) == 0, 1.0, -1.0)
        cut   = float(0.25 * np.sum(W * (1.0 - sigma[:, None] * sigma[None, :])))
        D     = build_D(W, phi)
        eigs  = np.linalg.eigvalsh(D)
        cuts_all.append(cut)
        lambdas_all.append(eigs.max())
        spectra_all.append(eigs)

    cuts_all    = np.array(cuts_all)
    lambdas_all = np.array(lambdas_all)

    # Pick 4 target cut levels evenly spaced; find the equilibrium
    # closest to each target and with the lowest lambda_D among ties
    unique_cuts = np.unique(np.round(cuts_all))
    n_panels    = min(4, len(unique_cuts))
    target_cuts = unique_cuts[np.linspace(0, len(unique_cuts)-1, n_panels, dtype=int)]

    selected = []
    for tc in target_cuts:
        mask = np.isclose(cuts_all, tc)
        idxs = np.where(mask)[0]
        best_idx = idxs[np.argmin(lambdas_all[idxs])]
        selected.append(best_idx)

    # ── Plot ──
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(
        f"Experiment 4 — Full D-matrix spectrum at increasing cut quality  [{name}]\n"
        r"Spectrum shifts left as $C$ grows: trace$(D) = 2E - 4C$ decreases",
        fontsize=11, fontweight="bold",
    )

    cmap = plt.cm.RdYlGn
    for ax, idx in zip(axes, selected):
        cut    = cuts_all[idx]
        eigs   = np.sort(spectra_all[idx])
        lam_max = eigs[-1]
        tr     = float(np.trace(build_D(W, np.array(all_bits[idx], dtype=float) * np.pi)))
        frac   = cut / E if E > 0 else 0
        col    = cmap(frac)

        # Eigenvalue lollipop plot
        ax.stem(range(N), eigs, linefmt=f"-", markerfmt="o", basefmt=" ",
                label=None)
        # Colour stems by position
        for line in ax.get_lines():
            line.set_color(col)
        for artist in ax.collections:
            artist.set_color(col)

        ax.axhline(0, color="black", linestyle="--", linewidth=1.3,
                   alpha=0.8)
        ax.set_xlabel("Eigenvalue index", fontsize=11)
        ax.set_xticks(range(N))
        ax.set_title(
            f"Cut = {cut:.0f}  (f={frac:.2f})\n"
            r"$\lambda_{\max}$" + f" = {lam_max:.3f}\n"
            f"trace = {tr:.1f} = 2·{E}−4·{cut:.0f}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_ylabel("Eigenvalue of D(φ*)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, f"exp4_spectrum_evolution_{name}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  [Exp 4] Saved → {path}")


# ===========================================================================
# Experiment 5 — Stability-cut tradeoff for OIM (penalty route)
# ===========================================================================

def exp5_stability_cut_tradeoff(W: np.ndarray, name: str, out_dir: str) -> None:
    """
    For Mechanism A (OIM, penalty mu), show numerically the tradeoff:
      - Small mu: equilibria with low lambda_D (good cuts) are stable,
                  but so are some suboptimal ones.
      - Large mu: all binary equilibria with lambda_D < mu become stable.
                  The binarization threshold mu* = min_phi lambda_D > 0.

    Plots:
      - mu_threshold per equilibrium vs cut value (scatter)
      - Fraction of equilibria stabilized vs mu (sweep)
      - Mean cut value of stable equilibria vs mu
    """
    N = W.shape[0]
    E = int(np.sum(W) // 2)
    print(f"  [Exp 5] Stability-cut tradeoff for {name} ...")

    cuts, lambdas, _ = enumerate_equilibria(W)

    # mu_threshold(phi*) = lambda_D (from Cheng et al. Theorem 2, K=1)
    # Equilibrium phi* is stable iff mu > lambda_D
    mu_thresholds = lambdas.copy()

    # Sweep mu
    mu_vals       = np.linspace(0, max(lambdas) * 1.05, 200)
    frac_stable   = [float(np.mean(mu > mu_thresholds)) for mu in mu_vals]
    mean_cut_stable = []
    for mu in mu_vals:
        mask = mu > mu_thresholds
        mean_cut_stable.append(float(np.mean(cuts[mask])) if mask.any() else np.nan)

    mu_bin = float(np.min(mu_thresholds[mu_thresholds > -1e-9]))   # binarization threshold

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Experiment 5 — Stability-cut tradeoff under penalty μ  [{name}]\n"
        r"As $\mu$ grows, more equilibria stabilize — but mean cut quality of stable set changes",
        fontsize=11, fontweight="bold",
    )

    # Panel A: mu_threshold vs cut (the hardness landscape, now labelled as threshold)
    ax = axes[0]
    norm = plt.Normalize(cuts.min(), cuts.max())
    sc   = ax.scatter(cuts, mu_thresholds, c=cuts, cmap="RdYlGn",
                      norm=norm, s=15, alpha=0.65, edgecolors="none")
    ax.axhline(mu_bin, color="purple", linestyle="--", linewidth=1.5,
               label=rf"$\mu_{{bin}}={mu_bin:.2f}$ (binarization threshold)")
    ax.axhline(0,      color="black",  linestyle=":",  linewidth=1.2)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Cut value", fontsize=9)
    ax.set_xlabel("Cut value", fontsize=12)
    ax.set_ylabel(r"$\mu^* = \lambda_D(\phi^*)$  (stability threshold)", fontsize=11)
    ax.set_title(
        r"Each dot = one equilibrium, threshold = $\lambda_D$" + "\n"
        r"Stable when $\mu > \lambda_D$",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: fraction of equilibria stabilized vs mu
    ax = axes[1]
    ax.plot(mu_vals, np.array(frac_stable) * 100, color="#7F77DD", linewidth=2)
    ax.axvline(mu_bin, color="purple", linestyle="--", linewidth=1.5,
               label=rf"$\mu_{{bin}}={mu_bin:.2f}$")
    ax.set_xlabel(r"Penalty $\mu$", fontsize=12)
    ax.set_ylabel("% equilibria stabilized", fontsize=11)
    ax.set_title(
        r"How many equilibria become stable as $\mu$ grows?" + "\n"
        "Step function: each equilibrium flips at its own threshold",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)

    # Panel C: mean cut of stable equilibria vs mu — THE KEY TRADEOFF
    ax = axes[2]
    valid = ~np.isnan(mean_cut_stable)
    ax.plot(mu_vals[valid], np.array(mean_cut_stable)[valid],
            color="#D85A30", linewidth=2)
    ax.axhline(cuts.max(), color="black", linestyle="--", linewidth=1.3,
               label=f"Optimal cut = {cuts.max():.0f}")
    ax.axhline(np.mean(cuts), color="gray", linestyle=":", linewidth=1.2,
               label=f"Mean over all equil. = {np.mean(cuts):.1f}")
    ax.axvline(mu_bin, color="purple", linestyle="--", linewidth=1.5,
               label=rf"$\mu_{{bin}}={mu_bin:.2f}$")
    ax.set_xlabel(r"Penalty $\mu$", fontsize=12)
    ax.set_ylabel("Mean cut of stable equilibria", fontsize=11)
    ax.set_title(
        r"Cut quality of the stable set vs $\mu$" + "\n"
        "Does larger μ help or hurt? This is the core tradeoff.",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, f"exp5_stability_cut_tradeoff_{name}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  [Exp 5] Saved → {path}")


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Stability theory experiments for OIM / Spectral Shaping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--graph", type=str, default=None, metavar="FILE",
                   help="Graph file (edge-list format). If not given, uses C10 built-in.")
    p.add_argument("--name",  type=str, default=None,
                   help="Graph name label for plots. Inferred from filename if not given.")
    p.add_argument("--out",   type=str, default=".", help="Output directory")
    p.add_argument("--exps",  type=str, default="12345",
                   help="Experiments to run, e.g. '13' for Exp 1 and 3")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=" * 65)
    print("  Stability Theory — D-matrix landscape analysis")
    print("=" * 65)

    # Load or build graph
    if args.graph is not None:
        W, comments = load_graph_from_file(args.graph)
        name = args.name or os.path.splitext(os.path.basename(args.graph))[0]
        if comments:
            print(f"\n  {comments}")
    else:
        W    = cycle_plus_diagonals(10, 5)
        name = "C10"
        print("\n  Using built-in C10 (10-cycle + 5 diagonals)")

    N = W.shape[0]
    bip, _ = is_bipartite(W)
    E      = int(np.sum(W) // 2)
    print(f"  N={N}, E={E}, bipartite={bip}, degrees={W.sum(axis=1).astype(int).tolist()}")

    if N > 18:
        print(f"  WARNING: N={N} > 18, enumerating 2^N={2**N} equilibria will be slow.")
        print("  Consider using a smaller graph for these experiments.")

    exps = args.exps

    if "1" in exps:
        print("\n── Experiment 1: Trace identity ──")
        exp1_trace_identity(W, name, args.out)

    if "2" in exps:
        print("\n── Experiment 2: Gershgorin bound ──")
        exp2_gershgorin(W, name, args.out)

    if "3" in exps:
        print("\n── Experiment 3: Bipartite vs non-bipartite ──")
        exp3_bipartite_vs_nonbipartite(args.out)

    if "4" in exps:
        print("\n── Experiment 4: Full spectrum evolution ──")
        exp4_spectrum_evolution(W, name, args.out)

    if "5" in exps:
        print("\n── Experiment 5: Stability-cut tradeoff under penalty μ ──")
        exp5_stability_cut_tradeoff(W, name, args.out)

    print("\n" + "=" * 65)
    print(f"  Done. Figures saved to: {os.path.abspath(args.out)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
