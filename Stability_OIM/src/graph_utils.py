import numpy as np
from itertools import product as iproduct


# ── File I/O ─────────────────────────────────────────────────────────────────

def read_graph(filepath: str, default_weight: float = -1.0) -> np.ndarray:
    """
    Read a graph from a text edge-list file and return its J matrix.

    Default weight changed to -1.0 to match the OIM anti-ferromagnetic
    convention (J_ij = -1 for connected spins).

    Parameters
    ----------
    filepath       : path to the .txt graph file
    default_weight : weight used when no third column is present (default -1.0)

    Returns
    -------
    J : np.ndarray, shape (n, n) — symmetric coupling matrix, zero diagonal.
    """
    with open(filepath, 'r') as f:
        lines = [ln.strip() for ln in f
                 if ln.strip() and not ln.strip().startswith('#')]

    n = int(lines[0])
    J = np.zeros((n, n), dtype=float)

    for line in lines[1:]:
        parts = line.split()
        i, j = int(parts[0]), int(parts[1])
        w = float(parts[2]) if len(parts) > 2 else default_weight
        J[i, j] = w
        J[j, i] = w     # enforce symmetry

    return J


def write_graph(J: np.ndarray, filepath: str, comment: str = "") -> None:
    """
    Write a J coupling matrix to a text edge-list file.

    Only the upper triangle is written (the reader symmetrises on load).

    Parameters
    ----------
    J        : (n, n) symmetric coupling matrix
    filepath : output .txt path
    comment  : optional description written as a header comment
    """
    n = J.shape[0]
    with open(filepath, 'w') as f:
        if comment:
            for line in comment.strip().splitlines():
                f.write(f"# {line}\n")
        f.write(f"{n}\n")
        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] != 0.0:
                    f.write(f"{i} {j} {J[i, j]:.6f}\n")


# ── Graph constructors ────────────────────────────────────────────────────────

def build_king_graph(n: int) -> np.ndarray:
    """
    Build the J coupling matrix for an n×n King graph.

    All edges have weight -1.0 (anti-ferromagnetic).
    Node numbering is row-major: i = r*n + c.

    Parameters
    ----------
    n : side length (total nodes = n*n)

    Returns
    -------
    J : (n*n, n*n) symmetric matrix with entries in {-1, 0}.
    """
    N = n * n
    J = np.zeros((N, N), dtype=float)
    for r in range(n):
        for c in range(n):
            i = r * n + c
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr_, nc_ = r + dr, c + dc
                    if 0 <= nr_ < n and 0 <= nc_ < n:
                        j = nr_ * n + nc_
                        J[i, j] = -1.0
    return J


def build_path_graph(n: int) -> np.ndarray:
    """J matrix for a path graph 0-1-2-...(n-1), all edges -1."""
    J = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        J[i, i+1] = J[i+1, i] = -1.0
    return J


def build_complete_graph(n: int) -> np.ndarray:
    """J matrix for a complete graph K_n, all edges -1."""
    J = -1.0 * np.ones((n, n), dtype=float)
    np.fill_diagonal(J, 0.0)
    return J


def random_graph(n: int, edge_prob: float = 0.5,
                 weight: float = -1.0, seed=None) -> np.ndarray:
    """
    Erdős–Rényi random graph G(n, p) with uniform edge weight.

    Parameters
    ----------
    weight : edge weight for all present edges (default -1.0)
    """
    rng   = np.random.default_rng(seed)
    upper = (rng.random((n, n)) < edge_prob).astype(float)
    upper = np.triu(upper, k=1)
    J     = (upper + upper.T) * weight
    np.fill_diagonal(J, 0.0)
    return J


# ── Analysis helpers ──────────────────────────────────────────────────────────

def verify_cut(J: np.ndarray, partition: np.ndarray) -> float:
    """
    Compute the MaxCut value for a given spin partition {-1, +1}.

    Parameters
    ----------
    J         : (n, n) coupling matrix
    partition : (n,) array with values in {-1, +1}

    Returns
    -------
    cut : float — sum of |J_ij| over cut edges (where partition[i] ≠ partition[j])
    """
    n   = len(partition)
    cut = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0.0 and partition[i] != partition[j]:
                cut += abs(J[i, j])
    return cut
