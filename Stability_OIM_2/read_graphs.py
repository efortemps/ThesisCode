"""
read_graphs.py
==============
Reads a simple unweighted graph file with the following format:

    # comment lines (ignored)
    N              <- number of nodes (single integer on its own line)
    u v            <- one undirected edge per line (0-indexed integers)
    u v
    ...

For MaxCut the coupling is set antiferromagnetic: J[u,v] = J[v,u] = -1,
so that minimising the OIM energy is equivalent to maximising the cut.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path


def read_graph_to_J(filepath: str | Path) -> np.ndarray:
    """
    Parse a graph file and return the NxN coupling matrix J.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    J : np.ndarray, shape (N, N)
        Symmetric coupling matrix; J[i,j] = -1 for each edge, 0 otherwise.

    Raises
    ------
    FileNotFoundError : if the file does not exist.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")

    # Strip comments and blank lines
    data_lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    # First data line: N (number of nodes)
    N = int(data_lines[0])
    J = np.zeros((N, N), dtype=float)

    # Remaining lines: edges "u v"
    for line in data_lines[1:]:
        u, v = map(int, line.split())
        J[u, v] = -1.0
        J[v, u] = -1.0

    return J


def graph_info(J: np.ndarray) -> dict:
    """Return basic graph statistics derived from J."""
    N = J.shape[0]
    edges = int((J != 0).sum() // 2)
    degrees = (J != 0).sum(axis=1)
    return {
        "N": N,
        "edges": edges,
        "density": 2 * edges / max(N * (N - 1), 1),
        "degree_min": int(degrees.min()),
        "degree_max": int(degrees.max()),
        "degree_mean": float(degrees.mean()),
    }
