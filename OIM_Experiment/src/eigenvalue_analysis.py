"""
eigenvalue_analysis.py

Usage:
    python eigenvalue_analysis.py graph.txt [--mu 2.0]

Reads a graph file of the format:
    # optional comment lines
    N                  <- number of nodes
    u v [w]            <- edges (weight w optional, defaults to 1.0)
    ...

Then performs the control-theoretic eigenvalue analysis from:
    Cheng et al., "A control theoretic analysis of oscillator Ising machines"
    Chaos 34, 073103 (2024)
"""

import argparse
import numpy as np
from itertools import product as iproduct
from OIM_Experiment.src.OIM_mu import OIMMaxCut
from OIM_Experiment.src.graph_utils import read_graph

# ──────────────────────────────────────────────
# File parser
# ──────────────────────────────────────────────

def parse_graph(path: str):
    """
    Parse a graph txt file.

    Returns
    -------
    n : int           — number of nodes
    W : np.ndarray    — symmetric N×N weight matrix (W_ij = edge weight, 0 if no edge)
    edges : list      — list of (u, v, w) triples
    """
    edges = []
    n = None

    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            if n is None:
                n = int(tokens[0])
                continue
            if len(tokens) == 2:
                u, v = int(tokens[0]), int(tokens[1])
                w = 1.0
            elif len(tokens) >= 3:
                u, v, w = int(tokens[0]), int(tokens[1]), float(tokens[2])
            else:
                continue
            edges.append((u, v, w))

    if n is None:
        raise ValueError("Could not find node count in file.")

    W = np.zeros((n, n), dtype=float)
    for u, v, w in edges:
        W[u, v] += w
        W[v, u] += w

    return n, W, edges


# ──────────────────────────────────────────────
# Helpers: Hamiltonian and cut value at a binary equilibrium
# ──────────────────────────────────────────────

def hamiltonian_at(W: np.ndarray, phi_star: np.ndarray) -> float:
    """
    Ising Hamiltonian H(σ) for the binary spin assignment induced by φ*.

    σ_i = cos(φ*_i) → +1 if φ*_i = 0, -1 if φ*_i = π.

    H = Σ_{i<j} W_ij σ_i σ_j = 0.5 * Σ_{i,j} W_ij σ_i σ_j

    Relationship to cut:  H = W_total - 2 * cut_value
    Lower H  ↔  better cut (H is minimised at the max-cut solution).
    """
    sigma = np.cos(phi_star)          # exactly ±1 for φ* ∈ {0,π}^N
    sigma = np.sign(sigma)
    sigma[sigma == 0] = 1.0
    return 0.5 * float(np.sum(W * np.outer(sigma, sigma)))


def cut_at(W: np.ndarray, phi_star: np.ndarray) -> float:
    """
    Cut value for the binary spin assignment induced by φ*.

    cut = 0.25 * Σ_{i,j} W_ij (1 - σ_i σ_j)
        = Σ_{(i,j) crossing partition} W_ij
    """
    sigma = np.cos(phi_star)
    sigma = np.sign(sigma)
    sigma[sigma == 0] = 1.0
    return 0.25 * float(np.sum(W * (1.0 - np.outer(sigma, sigma))))


# ──────────────────────────────────────────────
# Eigenvalue analysis
# ──────────────────────────────────────────────

def run_eigenvalue_analysis(oim: OIMMaxCut, verbose: bool = True) -> dict:
    """
    Full eigenvalue analysis over all 2^N Type-I M2 equilibria {0,π}^N.

    For each equilibrium φ* ∈ {0,π}^N:
      - Builds the signed Laplacian D(φ*)          [Eq. (5)]
      - Computes eigenvalues of D(φ*)
      - Computes eigenvalues of the Jacobian A(φ*) at current mu [Theorem 2]
      - Records stability status
      - Computes Ising Hamiltonian H(σ) and cut value

    Global binarisation threshold:
      μ_bin = min_{φ*} λ_max(D(φ*))               [Remark 7]
    """
    n   = oim.n
    mu  = oim.mu
    W   = oim.W
    w_total = 0.5 * float(np.sum(W))
    all_bits = list(iproduct([0, 1], repeat=n))

    rows = []
    for bits in all_bits:
        phi_star = np.array([b * np.pi for b in bits])
        key = "".join(str(b) for b in bits)

        # ── spectral analysis ──────────────────────────────────────
        D      = oim.build_D(phi_star)
        eigs_D = np.linalg.eigvalsh(D)
        lmax_D = float(eigs_D.max())

        A      = _jacobian_silent(oim, phi_star)
        eigs_A = np.linalg.eigvalsh(A)
        lmax_A = float(eigs_A.max())

        mu_threshold = lmax_D          # stable iff mu > lmax_D
        is_stable    = mu > mu_threshold

        # ── solution quality ───────────────────────────────────────
        H   = hamiltonian_at(W, phi_star)
        cut = cut_at(W, phi_star)

        rows.append({
            "bits":         key,
            "phi_star":     phi_star,
            "eigs_D":       eigs_D,
            "lmax_D":       lmax_D,
            "eigs_A":       eigs_A,
            "lmax_A":       lmax_A,
            "mu_threshold": mu_threshold,
            "is_stable":    is_stable,
            "hamiltonian":  H,
            "cut_value":    cut,
        })

    # ── global summary ─────────────────────────────────────────────
    mu_bin   = min(r["mu_threshold"] for r in rows)
    best_cut = max(r["cut_value"]    for r in rows)
    easiest  = min(rows, key=lambda r: r["mu_threshold"])
    hardest  = max(rows, key=lambda r: r["mu_threshold"])
    n_stable = sum(r["is_stable"] for r in rows)

    results = {
        "n":        n,
        "mu":       mu,
        "w_total":  w_total,
        "mu_bin":   mu_bin,
        "Ks_bin":   mu_bin / 2.0,
        "best_cut": best_cut,
        "easiest":  easiest,
        "hardest":  hardest,
        "n_stable": n_stable,
        "total":    len(rows),
        "rows":     rows,
    }

    if verbose:
        _print_report(results)

    return results


def _jacobian_silent(oim: OIMMaxCut, phi_star: np.ndarray) -> np.ndarray:
    """Compute Jacobian without the print statement inside oim.jacobian."""
    D = oim.build_D(phi_star)
    return D - oim.mu * np.diag(np.cos(2.0 * phi_star))


def _print_report(results: dict):
    sep = "─" * 80
    n       = results["n"]
    mu      = results["mu"]
    w_total = results["w_total"]

    print(sep)
    print(f"  EIGENVALUE ANALYSIS  |  N = {n}  |  μ = {mu:.4f}  |  W_total = {w_total:.2f}")
    print(sep)

    # ── binarisation threshold ─────────────────────────────────────
    print(f"\nBINARISATION THRESHOLD")
    print(f"  μ_bin  = {results['mu_bin']:.6f}   (= min_{{φ*}} λ_max(D(φ*)))")
    print(f"  Ks_bin = {results['Ks_bin']:.6f}   (equivalent Ks, K=1)")
    status = "✓ BINARISES" if mu > results["mu_bin"] else "✗ DOES NOT BINARISE"
    print(f"  Current μ = {mu:.4f}  →  {status}")
    print(f"  Stable equilibria : {results['n_stable']} / {results['total']}")
    print(f"  Best cut found    : {results['best_cut']:.2f}  (over all {results['total']} binary assignments)")

    # ── easiest / hardest ──────────────────────────────────────────
    print(f"\nEASIEST EQUILIBRIUM TO STABILISE  (lowest threshold)")
    _print_eq_detail(results["easiest"], mu, w_total)

    print(f"\nHARDEST EQUILIBRIUM TO STABILISE  (highest threshold)")
    _print_eq_detail(results["hardest"], mu, w_total)

    # ── stable equilibria summary ──────────────────────────────────
    stable_rows = [r for r in results["rows"] if r["is_stable"]]
    if stable_rows and n <= 12:
        print(f"\nSTABLE EQUILIBRIA — SORTED BY CUT VALUE (best first)")
        stable_sorted = sorted(stable_rows, key=lambda r: -r["cut_value"])
        col_bits = max(n, 9)
        hdr = (f"  {'φ* (bits)':<{col_bits}}  {'μ_thr':>8}  "
               f"{'λ_max(A)':>10}  {'H':>10}  {'cut':>8}  {'cut/W_tot':>10}")
        print(hdr)
        print("  " + "─" * (col_bits + 8 + 10 + 10 + 8 + 10 + 10))
        for r in stable_sorted:
            ratio = r["cut_value"] / w_total if w_total > 0 else float("nan")
            print(f"  {r['bits']:<{col_bits}}  {r['mu_threshold']:>8.4f}  "
                  f"{r['lmax_A']:>10.4f}  {r['hamiltonian']:>10.4f}  "
                  f"{r['cut_value']:>8.4f}  {ratio:>10.4f}")

    # ── full table for small graphs ────────────────────────────────
    if n <= 6:
        print(f"\nALL 2^N EQUILIBRIA — FULL EIGENVALUE TABLE")
        col_bits = max(n, 9)
        hdr = (f"  {'φ* (bits)':<{col_bits}}  {'λ_max(D)':>10}  {'μ_thr':>8}  "
               f"{'λ_max(A)':>10}  {'H':>10}  {'cut':>8}  stable?")
        print(hdr)
        print("  " + "─" * (col_bits + 10 + 8 + 10 + 10 + 8 + 12))
        for r in sorted(results["rows"], key=lambda x: x["mu_threshold"]):
            s = "✓" if r["is_stable"] else "✗"
            print(f"  {r['bits']:<{col_bits}}  {r['lmax_D']:>10.6f}  {r['mu_threshold']:>8.4f}  "
                  f"{r['lmax_A']:>10.4f}  {r['hamiltonian']:>10.4f}  {r['cut_value']:>8.4f}  {s}")

    elif n <= 12:
        # top-5 easiest + top-5 hardest
        sorted_rows = sorted(results["rows"], key=lambda x: x["mu_threshold"])
        print(f"\nTOP 5 EASIEST + TOP 5 HARDEST EQUILIBRIA (by stability threshold)")
        col_bits = n
        hdr = (f"  {'φ* (bits)':<{col_bits}}  {'μ_thr':>8}  "
               f"{'λ_max(A)':>10}  {'H':>10}  {'cut':>8}  stable?")
        print(hdr)
        print("  " + "─" * (col_bits + 8 + 10 + 10 + 8 + 10))
        for r in sorted_rows[:5] + sorted_rows[-5:]:
            s = "✓" if r["is_stable"] else "✗"
            print(f"  {r['bits']:<{col_bits}}  {r['mu_threshold']:>8.4f}  "
                  f"{r['lmax_A']:>10.4f}  {r['hamiltonian']:>10.4f}  {r['cut_value']:>8.4f}  {s}")

    print(f"\n{sep}\n")


def _print_eq_detail(r: dict, mu: float, w_total: float):
    ratio = r["cut_value"] / w_total if w_total > 0 else float("nan")
    print(f"  φ*          = {r['bits']}")
    print(f"  μ_threshold = {r['mu_threshold']:.6f}  →  "
          f"{'STABLE' if r['is_stable'] else 'UNSTABLE'} at μ={mu:.4f}")
    print(f"  λ(D(φ*))    = {np.round(r['eigs_D'], 6).tolist()}")
    print(f"  λ(A(φ*))    = {np.round(r['eigs_A'], 6).tolist()}")
    print(f"  H(σ)        = {r['hamiltonian']:.4f}  "
          f"(cut = {r['cut_value']:.4f}  =  {ratio:.2%} of W_total)")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OIM eigenvalue analysis")
    parser.add_argument("--graph", help="Path to graph .txt file")
    parser.add_argument("--mu",   type=float, default=2.0,  help="SHIL strength μ (default: 2.0)")
    parser.add_argument("--seed", type=int,   default=42,   help="RNG seed (default: 42)")
    args = parser.parse_args()

    print(f"\nLoading graph from: {args.graph}")
    W = read_graph(args.graph)
    n = W.shape[0]
    edges = [(i, j, W[i, j]) for i in range(n) for j in range(i+1, n) if W[i, j] > 0]
    print(f"Graph: {n} nodes, {len(edges)} edges,  W_total = {W.sum()/2:.2f}")

    oim = OIMMaxCut(weight_matrix=W, mu=args.mu, seed=args.seed)
    run_eigenvalue_analysis(oim, verbose=True)


if __name__ == "__main__":
    main()