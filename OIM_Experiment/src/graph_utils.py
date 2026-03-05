import numpy as np


def read_graph(filepath):
    """
    Read a weighted undirected graph from a text file (edge-list format).

    File format:
        # Comment lines start with '#'
        n                     ← number of nodes (first non-comment line)
        i  j  [weight]        ← one edge per line; weight defaults to 1.0

    Example:
        # Petersen graph (10 nodes, unweighted)
        10
        0 1
        0 4
        0 5
        ...

    Returns
    -------
    W : np.ndarray, shape (n, n)
        Symmetric adjacency/weight matrix, zeros on the diagonal.
    """
    with open(filepath, 'r') as f:
        lines = [ln.strip() for ln in f
                 if ln.strip() and not ln.strip().startswith('#')]

    n = int(lines[0])
    W = np.zeros((n, n), dtype=float)

    for line in lines[1:]:
        parts = line.split()
        i, j = int(parts[0]), int(parts[1])
        w = float(parts[2]) if len(parts) > 2 else 1.0
        W[i, j] = w
        W[j, i] = w   # enforce symmetry

    return W


def random_graph(n, edge_prob=0.5, seed=None):
    """
    Generate an unweighted Erdős–Rényi random graph G(n, p).

    Returns
    -------
    W : np.ndarray, shape (n, n)
        Symmetric 0/1 adjacency matrix, zeros on the diagonal.
    """
    rng = np.random.default_rng(seed)
    upper = (rng.random((n, n)) < edge_prob).astype(float)
    upper = np.triu(upper, k=1)          # keep upper triangle only
    W     = upper + upper.T              # symmetrize
    np.fill_diagonal(W, 0.0)
    return W


def verify_cut(W, partition):
    """
    Compute the exact cut weight by direct edge enumeration.
    Use this as a ground-truth check against the formula-based value.

    Parameters
    ----------
    W         : np.ndarray (n, n), symmetric weight matrix
    partition : np.ndarray (n,)   with values in {-1, +1}

    Returns
    -------
    cut : float
    """
    n   = len(partition)
    cut = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0 and partition[i] != partition[j]:
                cut += W[i, j]
    return cut
