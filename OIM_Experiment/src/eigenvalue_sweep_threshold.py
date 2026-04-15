#!/usr/bin/env python3
"""
eigenvalue_sweep_threshold.py
──────────────────────────────────────────────────────────────────────────────
Eigenvalue sweep & bifurcation analysis on THRESHOLD GRAPHS.

A threshold graph is built by starting from a single vertex and repeatedly
adding either:
  • An ISOLATED vertex  (bit 0): no edges to any existing vertex
  • A DOMINATING vertex (bit 1): connected to ALL existing vertices

Construction sequence: s[0] s[1] … s[N-1], with s[0] = 0 by convention.
Edge rule:  edge (i, j), i < j  ⟺  s[j] = 1.

The analysis mirrors eigenvalue_sweep_analysis.py exactly.  Added on top:

Figure 0 — Threshold graph structure
  Top-left  | NetworkX spring layout: isolated (□) vs dominating (●) nodes
  Top-right | Adjacency-matrix heatmap
  Bottom    | Construction table + key graph metrics

Figures 1–3 — identical to eigenvalue_sweep_analysis.py
  Fig 1 | Phase dynamics + convergence table
  Fig 2 | Bifurcation analysis (λ_max bar + bifurcation diagram)
  Fig 3 | Quality analysis (H vs cut scatter + D spectrum)

Usage
─────
# Explicit sequence (first bit is always 0 = seed vertex)
python eigenvalue_sweep_threshold.py --sequence 0110100

# Random threshold graph of size N
python eigenvalue_sweep_threshold.py --n 7 --seed 42

# Override μ and save all figures
python eigenvalue_sweep_threshold.py --sequence 011010 --mu 3.5 --save

Mathematical background (Cheng et al., Chaos 34, 073103, 2024)
────────────────────────────────────────────────────────────────
A(φ*, μ) = D(φ*) − μ·I_N  →  λ_k(A) = λ_k(D) − μ  (slope −1 in μ)
Stability:  μ > λ_max(D(φ*))
μ_bin      = min_{φ*∈{0,π}^N} λ_max(D(φ*))          [Remark 7]
H(σ)       = Σ_{i<j} W_ij σ_i σ_j = W_tot − 2·cut
"""

import argparse
import sys
import time
from collections import defaultdict
from itertools import product as iproduct
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# optional — used only for Figure 0 graph layout
try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("[warn] networkx not found — Figure 0 graph panel will be skipped")

from OIM_Experiment.src.OIM_mu import OIMMaxCut

# ── TikZ / PGFPlots global style ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.edgecolor":    "black",
    "axes.linewidth":    0.8,
    "xtick.color":       "black",
    "ytick.color":       "black",
    "text.color":        "black",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

# ── Colour palette ─────────────────────────────────────────────────────────────
WHITE     = "#ffffff"
BLACK     = "#000000"
GRAY      = "#b0b0b0"
LIGHT     = "#e6e6e6"
C_STABLE  = "#4C72B0"   # blue  → stable
C_UNSTABLE= "#DD8452"   # orange → unstable
C_BPART   = "#4C72B0"
C_FERRO   = "#DD8452"
C_MIXED   = "#55A868"
C_MU_LINE = "#ffb74d"
C_ISO     = "#5588cc"   # isolated vertex fill
C_DOM     = "#cc4444"   # dominating vertex fill

_TYPE_COL = {
    "M2-binary":   "#d4eac8",
    "M1-half":     "#fde8c8",
    "M1-mixed":    "#fde8c8",
    "Type-III":    "#e8c8de",
    "not-converged":"#e8e8e8",
}

# ══════════════════════════════════════════════════════════════════════════════
# Threshold graph utilities
# ══════════════════════════════════════════════════════════════════════════════

def build_threshold_graph(sequence: str) -> tuple[np.ndarray, str]:
    """
    Build the (unweighted, symmetric) adjacency matrix for a threshold graph
    defined by the binary construction sequence.

    Parameters
    ----------
    sequence : str
        Binary string of 0s and 1s.  First character must be '0' (seed vertex).
        '0' = add isolated vertex, '1' = add dominating vertex.

    Returns
    -------
    W : np.ndarray, shape (N, N)   — symmetric adjacency / weight matrix
    seq : str                      — normalised sequence (forced s[0]='0')
    """
    seq = list(sequence)
    if seq[0] != '0':
        seq[0] = '0'   # enforce convention
    seq_str = "".join(seq)
    N = len(seq)
    W = np.zeros((N, N), dtype=float)
    # edge (i, j), i < j  ⟺  seq[j] == '1'
    for j in range(1, N):
        if seq[j] == '1':
            for i in range(j):
                W[i, j] = W[j, i] = 1.0
    return W, seq_str


def random_threshold_sequence(n: int, seed: int = 42) -> str:
    """Generate a random threshold sequence of length n (s[0]='0' always)."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n).tolist()
    bits[0] = 0
    return "".join(str(b) for b in bits)


def threshold_graph_info(W: np.ndarray, seq: str) -> dict:
    """Compute basic structural properties of the threshold graph."""
    N = W.shape[0]
    degrees  = W.sum(axis=1).astype(int)
    n_dom    = seq.count('1')
    n_iso    = seq.count('0')
    n_edges  = int(W.sum()) // 2
    density  = n_edges / max(N * (N - 1) // 2, 1)
    # connected iff at least one dominating vertex exists
    connected = (n_dom > 0)
    return dict(N=N, degrees=degrees, n_dom=n_dom, n_iso=n_iso,
                n_edges=n_edges, density=density, connected=connected,
                seq=seq)


# ══════════════════════════════════════════════════════════════════════════════
# Jacobian helper
# ══════════════════════════════════════════════════════════════════════════════

def _jacobian(oim: OIMMaxCut, phi_star: np.ndarray) -> np.ndarray:
    D = oim.build_D(phi_star)
    return D - oim.mu * np.diag(np.cos(2.0 * phi_star))


# ══════════════════════════════════════════════════════════════════════════════
# Equilibrium analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyse_equilibria(oim: OIMMaxCut) -> dict:
    n, mu, W = oim.n, oim.mu, oim.W
    w_total  = float(np.sum(W)) / 2.0
    rows = []
    for bits in iproduct([0, 1], repeat=n):
        phi   = np.array([b * np.pi for b in bits], dtype=float)
        D     = oim.build_D(phi)
        ev_D  = np.sort(np.linalg.eigvalsh(D))
        lmax  = float(ev_D[-1])
        A     = _jacobian(oim, phi)
        ev_A  = np.sort(np.linalg.eigvalsh(A))

        sigma = np.sign(np.cos(phi)); sigma[sigma == 0] = 1.0
        H     = 0.5  * float(np.sum(W * np.outer(sigma, sigma)))
        cut   = 0.25 * float(np.sum(W * (1.0 - np.outer(sigma, sigma))))

        rows.append(dict(bits=bits, phi=phi, D=D,
                         ev_D=ev_D, ev_A=ev_A, lmax_D=lmax,
                         lmax_A=float(ev_A[-1]), mu_thr=lmax,
                         stable=(mu > lmax), H=H, cut=cut))

    mu_bin   = min(r["lmax_D"] for r in rows)
    best_cut = max(r["cut"]    for r in rows)
    n_stable = sum(r["stable"] for r in rows)
    return dict(rows=rows, mu_bin=mu_bin, best_cut=best_cut, w_total=w_total,
                easiest=min(rows, key=lambda r: r["lmax_D"]),
                hardest=max(rows, key=lambda r: r["lmax_D"]),
                n_stable=n_stable, total=len(rows), n=n, mu=mu)


# ══════════════════════════════════════════════════════════════════════════════
# μ sweep
# ══════════════════════════════════════════════════════════════════════════════

def mu_sweep(oim: OIMMaxCut, eq_data: dict, mu_vals: np.ndarray) -> dict:
    rows = eq_data["rows"]
    paths, bif_pts = {}, []
    for r in rows:
        key = r["bits"]
        paths[key] = r["ev_D"][None, :] - mu_vals[:, None]
        mu_star = r["lmax_D"]
        if mu_vals[0] <= mu_star <= mu_vals[-1]:
            bif_pts.append((mu_star, key))

    lmax_D_arr   = np.array([r["lmax_D"] for r in rows])
    lmax_A_mat   = lmax_D_arr[:, None] - mu_vals[None, :]
    n_stable_mu  = np.sum(lmax_A_mat < 0, axis=0)
    return dict(mu_vals=mu_vals, paths=paths, n_stable_mu=n_stable_mu,
                lmax_A_min_mu=lmax_A_mat.min(axis=0),
                lmax_A_max_mu=lmax_A_mat.max(axis=0),
                bifurcation_pts=bif_pts)


# ══════════════════════════════════════════════════════════════════════════════
# 3 representative equilibria
# ══════════════════════════════════════════════════════════════════════════════

def pick_representatives(rows: list):
    sr   = sorted(rows, key=lambda r: r["lmax_D"])
    low  = sr[0]
    mid  = sr[len(sr) // 2]
    high = sr[-1]

    def _lbl(tag, r):
        b = "".join(str(x) for x in r["bits"])
        return (f"{tag}: $[{b}]$ "
                f"$\\lambda_{{\\max}}={r['lmax_D']:.3f}$ "
                f"$H={r['H']:.1f}$ cut$={r['cut']:.1f}$")

    return [
        (low,  _lbl("Easiest", low),  C_BPART,  "-"),
        (mid,  _lbl("Median",  mid),  C_MIXED,  "--"),
        (high, _lbl("Hardest", high), C_FERRO,  ":"),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Convergence identification
# ══════════════════════════════════════════════════════════════════════════════

def _atom(theta_i: float, tol: float = 0.05) -> str:
    s, c = np.sin(theta_i), np.cos(theta_i)
    if abs(s) < tol:
        return "zero" if c > 0 else "pi"
    if abs(abs(s) - 1.0) < tol:
        return "half"
    return "other"


def identify_convergence(sol, W, binary_equilibria, bin_tol=0.05):
    theta = sol.y[:, -1].copy()
    n = len(theta)
    sigma = np.sign(np.cos(theta)); sigma[sigma == 0] = 1.0
    bits  = tuple(0 if s > 0 else 1 for s in sigma)
    H     = 0.5  * float(np.sum(W * np.outer(sigma, sigma)))
    cut   = 0.25 * float(np.sum(W * (1.0 - np.outer(sigma, sigma))))
    residual = float(np.max(np.abs(np.sin(theta))))

    atom_types = [_atom(th, bin_tol) for th in theta]
    n_zero = atom_types.count("zero"); n_pi   = atom_types.count("pi")
    n_half = atom_types.count("half"); n_othe = atom_types.count("other")

    if n_zero + n_pi == n:
        state_type = "M2-binary"
    elif n_half == n:
        state_type = "M1-half"
    elif n_half > 0 and n_othe == 0 and n_zero + n_pi + n_half == n:
        state_type = "M1-mixed"
    elif n_othe > 0:
        state_type = "Type-III"
    else:
        state_type = "not-converged"

    nearest_eq, min_dist = None, np.inf
    for r in binary_equilibria:
        diff = (theta - r["phi"] + np.pi) % (2 * np.pi) - np.pi
        dist = float(np.linalg.norm(diff))
        if dist < min_dist:
            min_dist = dist
            nearest_eq = dict(bits=r["bits"], phi=r["phi"].copy(),
                              H=r["H"], cut=r["cut"],
                              mu_thr=r["mu_thr"], stable=r["stable"],
                              dist_L2=dist)
    return dict(theta_end=theta, bits=bits, H=H, cut=cut,
                residual=residual, is_binary=(state_type=="M2-binary"),
                state_type=state_type, atom_types=atom_types,
                nearest_eq=nearest_eq)


# ══════════════════════════════════════════════════════════════════════════════
# Axis style helper
# ══════════════════════════════════════════════════════════════════════════════

def _ax_style(ax, title="", xlabel="", ylabel="", titlesize=12):
    ax.set_facecolor(WHITE)
    for sp in ax.spines.values():
        sp.set_edgecolor(BLACK); sp.set_linewidth(0.8)
    ax.tick_params(colors=BLACK, labelsize=9)
    ax.grid(True, color=LIGHT, linewidth=0.6, alpha=1.0)
    if title:  ax.set_title(title,  color=BLACK, fontsize=titlesize,
                            fontweight="bold", pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=BLACK, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, color=BLACK, fontsize=10)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 0 — Threshold graph structure
# ══════════════════════════════════════════════════════════════════════════════

def make_figure0(W: np.ndarray, seq: str, info: dict, mu_bin: float):
    """
    Three-panel figure:
      Left   — Graph visualisation (NetworkX spring layout)
      Centre — Adjacency matrix heatmap
      Right  — Construction table + key properties
    """
    N = info["N"]

    fig = plt.figure(figsize=(20, 7), facecolor=WHITE)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.30,
                            left=0.04, right=0.97, top=0.88, bottom=0.10)
    ax_graph = fig.add_subplot(gs[0, 0])
    ax_heat  = fig.add_subplot(gs[0, 1])
    ax_tbl   = fig.add_subplot(gs[0, 2])

    # ── Left: graph layout ───────────────────────────────────────────────────
    ax_graph.set_facecolor(WHITE)
    ax_graph.axis("off")

    if HAS_NX:
        G = nx.from_numpy_array(W)
        pos = nx.spring_layout(G, seed=0, k=2.5 / max(np.sqrt(N), 1))

        node_colors = [C_DOM if seq[i] == '1' else C_ISO for i in range(N)]
        node_shapes_dom = [i for i in range(N) if seq[i] == '1']
        node_shapes_iso = [i for i in range(N) if seq[i] == '0']

        nx.draw_networkx_edges(G, pos, ax=ax_graph,
                               edge_color=GRAY, width=1.4, alpha=0.7)
        nx.draw_networkx_nodes(G, pos, nodelist=node_shapes_dom,
                               node_color=C_DOM, node_shape='o',
                               node_size=500, ax=ax_graph,
                               edgecolors=BLACK, linewidths=1.0)
        nx.draw_networkx_nodes(G, pos, nodelist=node_shapes_iso,
                               node_color=C_ISO, node_shape='s',
                               node_size=400, ax=ax_graph,
                               edgecolors=BLACK, linewidths=1.0)
        nx.draw_networkx_labels(G, pos, ax=ax_graph,
                                font_size=9, font_color=WHITE,
                                font_weight="bold")
        legend_handles = [
            mpatches.Patch(facecolor=C_DOM, edgecolor=BLACK,
                           label="Dominating (1)"),
            mpatches.Patch(facecolor=C_ISO, edgecolor=BLACK,
                           label="Isolated (0)"),
        ]
        ax_graph.legend(handles=legend_handles, loc="lower center",
                        fontsize=8.5, facecolor=WHITE, edgecolor=GRAY,
                        labelcolor=BLACK, framealpha=0.92)
    else:
        ax_graph.text(0.5, 0.5,
                      "Install networkx\nfor graph layout\n(pip install networkx)",
                      ha="center", va="center", fontsize=11, color=GRAY,
                      transform=ax_graph.transAxes)

    ax_graph.set_title(
        f"Threshold graph  $N={N}$  $|E|={info['n_edges']}$\n"
        f"sequence: $\\tt{{{seq}}}$\n"
        f"{'connected' if info['connected'] else 'disconnected'}  |  "
        f"density $= {info['density']:.2f}$",
        color=BLACK, fontsize=10, fontweight="bold")

    # ── Centre: adjacency matrix heatmap ────────────────────────────────────
    im = ax_heat.imshow(W, cmap="Blues", vmin=0, vmax=1,
                        interpolation="nearest", aspect="auto")
    ax_heat.set_xticks(range(N)); ax_heat.set_yticks(range(N))
    ax_heat.set_xticklabels([f"v{i}" for i in range(N)], fontsize=8,
                             rotation=45, ha="right")
    ax_heat.set_yticklabels([f"v{i}" for i in range(N)], fontsize=8)
    for i in range(N):
        for j in range(N):
            if W[i, j] > 0:
                ax_heat.text(j, i, "1", ha="center", va="center",
                             fontsize=7, color=WHITE if W[i,j]>0.5 else BLACK)
    plt.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02,
                 label="edge weight")
    _ax_style(ax_heat,
              title="Adjacency matrix $W$\n"
                    r"$W_{ij}=1 iff s[j]=1$, $i<j$",
              xlabel="vertex $j$", ylabel="vertex $i$")

    # ── Right: construction table ────────────────────────────────────────────
    ax_tbl.axis("off")
    ax_tbl.set_facecolor(WHITE)

    col_labels = ["vertex", "type", "bit s[i]", "degree", "new edges"]
    table_data = []
    degrees = info["degrees"]
    for i in range(N):
        vtype   = "dominating" if seq[i] == '1' else "isolated"
        new_e   = i if seq[i] == '1' else 0        # connects to all i prev
        table_data.append([
            f"v{i}",
            vtype,
            seq[i],
            str(int(degrees[i])),
            str(new_e),
        ])

    tbl = ax_tbl.table(
        cellText=table_data, colLabels=col_labels,
        bbox=[0.0, 0.30, 1.0, 0.65],
        cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor(LIGHT)
        tbl[0, j].set_text_props(fontweight="bold", color=BLACK)
    for i, row in enumerate(table_data, start=1):
        col = C_DOM + "44" if row[1] == "dominating" else C_ISO + "44"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(col)
            tbl[i, j].set_edgecolor(GRAY)
            tbl[i, j].set_linewidth(0.4)

    # Key properties text
    props = (
        f"$N = {N}$  (vertices)\n"
        f"$|E| = {info['n_edges']}$  (edges)\n"
        f"Dominating: {info['n_dom']}  |  Isolated: {info['n_iso']}\n"
        f"Max degree: {degrees.max():.0f}  |  Min: {degrees.min():.0f}\n"
        f"Degree sequence: {sorted(degrees.tolist(), reverse=True)}\n"
        f"Density: {info['density']:.3f}\n"
        f"Connected: {'yes' if info['connected'] else 'no'}\n\n"
        f"$\\mu_{{\\rm bin}} = {mu_bin:.4f}$\n"
        f"(binarisation threshold, Remark 7)"
    )
    ax_tbl.text(0.5, 0.27, props, transform=ax_tbl.transAxes,
                ha="center", va="top", fontsize=9.5, color=BLACK,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=LIGHT,
                          edgecolor=GRAY, alpha=0.95),
                linespacing=1.6)

    ax_tbl.set_title("Construction sequence table\n"
                     r"edge $(i,j)$ exists $ iff s[j]=1$",
                     color=BLACK, fontsize=10, fontweight="bold")

    fig.suptitle(
        f"Threshold Graph — OIM Eigenvalue Sweep  |  "
        f"sequence: $\\tt{{{seq}}}$  |  "
        f"$N={N}$, $|E|={info['n_edges']}$  |  "
        f"$\\mu_{{\\rm bin}}={mu_bin:.4f}$",
        color=BLACK, fontsize=12, fontweight="bold")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Phase dynamics + convergence table
# ══════════════════════════════════════════════════════════════════════════════

def _draw_convergence_tables(ax, conv_results, n_init):
    per_cols = ["#", "state type", "bits", "H", "cut", "residual",
                "nearest M2", "dist", "stable?"]
    per_rows = []
    for i, c in enumerate(conv_results):
        bits_s = "".join(str(b) for b in c["bits"])
        neq = c["nearest_eq"]
        nb  = "".join(str(b) for b in neq["bits"]) if neq else "—"
        nd  = f"{neq['dist_L2']:.3f}" if neq else "—"
        ns  = ("✓" if neq["stable"] else "✗") if neq else "—"
        per_rows.append([str(i), c["state_type"], bits_s,
                         f"{c['H']:.3f}", f"{c['cut']:.3f}",
                         f"{c['residual']:.4f}", nb, nd, ns])

    sum_cols = ["state type", "bits", "H", "cut", "count", "%",
                "mean res.", "nearest M2", "dist", "stable?"]
    summary_map = defaultdict(lambda: {"state_type": None, "H": None,
                                       "cut": None, "residuals": [],
                                       "nearest_eq": None})
    for c in conv_results:
        key = (c["state_type"], c["bits"])
        sm  = summary_map[key]
        sm.update(state_type=c["state_type"], H=c["H"], cut=c["cut"],
                  nearest_eq=c["nearest_eq"])
        sm["residuals"].append(c["residual"])

    sum_rows = []
    for (stype, bits), sm in sorted(summary_map.items(),
                                    key=lambda x: -x[1]["cut"]):
        bits_s = "".join(str(b) for b in bits)
        neq = sm["nearest_eq"]
        nb  = "".join(str(b) for b in neq["bits"]) if neq else "—"
        nd  = f"{neq['dist_L2']:.3f}" if neq else "—"
        ns  = ("✓" if neq["stable"] else "✗") if neq else "—"
        cnt = len(sm["residuals"])
        sum_rows.append([stype, bits_s,
                         f"{sm['H']:.3f}", f"{sm['cut']:.3f}",
                         str(cnt), f"{100*cnt/n_init:.1f}%",
                         f"{np.mean(sm['residuals']):.4f}", nb, nd, ns])

    rh = 0.028; hh = 0.032; th = 0.042; gp = 0.055
    scale = min(1.0, 0.97 / max(
        th + hh + len(per_rows)*rh + gp + th + hh + len(sum_rows)*rh, 0.01))
    rh *= scale; hh *= scale; th *= scale; gp *= scale
    y = 0.99

    def _put_table(y_top, col_labels, data_rows, title_txt):
        ax.text(0.5, y_top, title_txt, ha="center", va="top",
                fontsize=9.5, fontweight="bold", color=BLACK,
                transform=ax.transAxes)
        y_top -= th
        nc, nr = len(col_labels), len(data_rows)
        bbox   = [0.0, y_top - hh - nr*rh, 1.0, hh + nr*rh]
        tbl    = ax.table(cellText=data_rows, colLabels=col_labels,
                          bbox=bbox, cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(7.2)
        for j in range(nc):
            tbl[0, j].set_facecolor(LIGHT)
            tbl[0, j].set_text_props(fontweight="bold", color=BLACK)
        for i, row in enumerate(data_rows, start=1):
            stype = row[1] if col_labels[0] == "#" else row[0]
            base_col = _TYPE_COL.get(stype, WHITE)
            for j in range(nc):
                tbl[i, j].set_facecolor(base_col + "28")
                tbl[i, j].set_edgecolor(GRAY)
                tbl[i, j].set_linewidth(0.4)
            type_col_idx = 1 if col_labels[0] == "#" else 0
            tbl[i, type_col_idx].set_facecolor(base_col + "80")
        return y_top - hh - nr*rh

    y = _put_table(y, per_cols,  per_rows,  "Per-trajectory convergence")
    y -= gp
    _put_table(y, sum_cols, sum_rows, "Summary — unique terminal states")

    handles = [mpatches.Patch(facecolor=v, edgecolor=GRAY, label=k)
               for k, v in _TYPE_COL.items()]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
              ncol=3, fontsize=7.5, facecolor=WHITE, edgecolor=GRAY,
              framealpha=0.92, labelcolor=BLACK)


def make_figure1(graph_name, n, W, seq, eq_data, sols, MU_SIM, conv_results,
                 n_init):
    mu_bin   = eq_data["mu_bin"]
    w_total  = eq_data["w_total"]
    best_cut = eq_data["best_cut"]

    fig = plt.figure(figsize=(26, max(9, 2.0 + 0.40*len(sols))),
                     facecolor=WHITE)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.06,
                            width_ratios=[1.55, 1.0],
                            left=0.04, right=0.99, top=0.88, bottom=0.09)
    ax_phase = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[0, 1])
    ax_table.axis("off")

    SPIN_COLS = plt.get_cmap("tab20")(np.linspace(0, 1, max(n, 2)))
    PI_TICKS  = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    PI_LABELS = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]

    t = sols[0].t
    for sol in sols:
        for spin in range(n):
            ax_phase.plot(t, sol.y[spin], color=SPIN_COLS[spin % 20],
                          alpha=0.42, linewidth=1.0)
    for yref, lw_r in [(np.pi, 1.1), (np.pi/2, 0.7), (0.0, 1.4),
                       (-np.pi/2, 0.7), (-np.pi, 0.9)]:
        ax_phase.axhline(yref, color=GRAY, linestyle="--",
                         linewidth=lw_r, alpha=0.75)
        if abs(abs(yref) - np.pi/2) < 1e-9:
            lbl = "+π/2" if yref > 0 else "−π/2"
            ax_phase.text(t[-1]*0.995, yref+0.10, lbl, ha="right",
                          va="bottom", fontsize=7.5, color=GRAY)

    ax_phase.set_yticks(PI_TICKS)
    ax_phase.set_yticklabels(PI_LABELS, fontsize=10, color=BLACK)
    ax_phase.set_ylim(-4.2, 4.2); ax_phase.set_xlim(t[0], t[-1])

    n_binary  = sum(1 for c in conv_results if c["is_binary"])
    n_m1      = sum(1 for c in conv_results
                    if c["state_type"] in ("M1-half", "M1-mixed"))
    n_type3   = sum(1 for c in conv_results if c["state_type"] == "Type-III")
    n_notconv = sum(1 for c in conv_results
                    if c["state_type"] == "not-converged")
    is_all_bin = (n_binary == len(conv_results))
    status_str = (
        "BINARISED ✓" if is_all_bin else
        f"NOT YET BINARISED ✗  (M2: {n_binary} | ±π/2: {n_m1} | "
        f"TypeIII: {n_type3} | no-conv: {n_notconv})")
    ax_phase.text(0.98, 0.97, status_str, transform=ax_phase.transAxes,
                  ha="right", va="top", fontsize=9, fontweight="bold",
                  color=C_STABLE if is_all_bin else C_UNSTABLE,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor=WHITE,
                            edgecolor=GRAY, alpha=0.95))
    ax_phase.text(0.01, 0.97,
                  f"$\\mu={MU_SIM:.4f}$ | $\\mu_{{\\rm bin}}={mu_bin:.4f}$ | "
                  f"$W_{{\\rm tot}}={w_total:.1f}$ | Best cut $={best_cut:.1f}$",
                  transform=ax_phase.transAxes, ha="left", va="top",
                  fontsize=9.5,
                  bbox=dict(boxstyle="round,pad=0.28", facecolor=WHITE,
                            edgecolor=GRAY, alpha=0.93))

    spin_patches = [mpatches.Patch(color=SPIN_COLS[s % 20],
                                   label=f"spin {s}") for s in range(n)]
    ax_phase.legend(handles=spin_patches, loc="lower right", fontsize=8,
                    ncol=max(1, n // 5), framealpha=0.90)
    _ax_style(ax_phase,
              title=(f"Phase dynamics  $\\mu={MU_SIM:.4f}$ "
                     f"({'above' if MU_SIM > mu_bin else 'below'} "
                     f"$\\mu_{{\\rm bin}}={mu_bin:.4f}$) | "
                     f"{n_init} initial conditions "
                     r"[uniform $(-\pi,\,\pi)$]"),
              xlabel="time $t$", ylabel="phase $\\theta_i(t)$ (rad)")
    _draw_convergence_tables(ax_table, conv_results, n_init)

    fig.suptitle(
        f"OIM Phase Dynamics | Threshold graph  $\\tt{{{seq}}}$  |  "
        f"$N={n}$, $\\mu_{{\\rm bin}}={mu_bin:.4f}$ | "
        f"Best cut $={best_cut:.1f}$, $W_{{\\rm tot}}={w_total:.1f}$",
        color=BLACK, fontsize=12, fontweight="bold")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Bifurcation analysis
# ══════════════════════════════════════════════════════════════════════════════

def make_figure2(graph_name, n, seq, eq_data, sweep_data):
    rows       = eq_data["rows"]
    mu_bin     = eq_data["mu_bin"]
    w_total    = eq_data["w_total"]
    best_cut   = eq_data["best_cut"]
    mu_vals    = sweep_data["mu_vals"]
    n_eq       = eq_data["total"]
    current_mu = eq_data["mu"]

    lmax_D_all   = np.array([r["lmax_D"] for r in rows])
    H_all        = np.array([r["H"]      for r in rows])
    cut_all      = np.array([r["cut"]    for r in rows])
    unique_lmax, inverse, counts_lmax = np.unique(
        np.round(lmax_D_all, 5), return_inverse=True, return_counts=True)
    avg_H_per_lmax   = np.array([H_all  [inverse == k].mean()
                                 for k in range(len(unique_lmax))])
    avg_cut_per_lmax = np.array([cut_all[inverse == k].mean()
                                 for k in range(len(unique_lmax))])

    cmap_cut = plt.get_cmap("RdYlGn")
    norm_cut = mcolors.Normalize(vmin=0, vmax=best_cut)

    fig = plt.figure(figsize=(18, 12), facecolor=WHITE)
    gs  = gridspec.GridSpec(2, 1, figure=fig,
                            height_ratios=[0.70, 1.30], hspace=0.45,
                            left=0.06, right=0.96, top=0.92, bottom=0.08)
    ax_lmbar = fig.add_subplot(gs[0])
    ax_bif   = fig.add_subplot(gs[1])

    # ── bar chart ─────────────────────────────────────────────────────────────
    sorted_rows  = sorted(rows, key=lambda r: r["lmax_D"])
    lmax_arr     = np.array([r["lmax_D"] for r in sorted_rows])
    cut_arr_s    = np.array([r["cut"]    for r in sorted_rows])
    bar_cols     = [cmap_cut(norm_cut(c)) for c in cut_arr_s]

    ax_lmbar.bar(np.arange(n_eq), lmax_arr, color=bar_cols,
                 width=1.0, edgecolor="none", zorder=2)
    ax_lmbar.axhline(0,          color=BLACK,    linewidth=0.9, zorder=3)
    ax_lmbar.axhline(mu_bin,     color=C_STABLE, linewidth=2.0,
                     linestyle="--", zorder=5,
                     label=f"$\\mu_{{\\rm bin}} = {mu_bin:.3f}$")
    ax_lmbar.axhline(current_mu, color=C_MU_LINE, linewidth=2.0,
                     linestyle="--", zorder=6,
                     label=f"current $\\mu = {current_mu:.3f}$")

    sm_bar = plt.cm.ScalarMappable(cmap=cmap_cut, norm=norm_cut)
    sm_bar.set_array([])
    cb_bar = fig.colorbar(sm_bar, ax=ax_lmbar, fraction=0.015, pad=0.01)
    cb_bar.set_label("Cut value", fontsize=10)
    cb_bar.ax.tick_params(labelsize=9); cb_bar.outline.set_edgecolor(BLACK)

    ax_lmbar.set_xlim(-1, n_eq); ax_lmbar.legend(fontsize=10, loc="upper left")
    _ax_style(ax_lmbar,
              title=(f"$\\lambda_{{\\max}}(D(\\phi^*))$ for all $2^N={n_eq}$ "
                     f"equilibria (sorted) | bar colour = cut quality | "
                     f"stable at current $\\mu$: {eq_data['n_stable']}/{n_eq}"),
              xlabel="Equilibrium index",
              ylabel="$\\lambda_{\\max}(D(\\phi^*))$")

    # ── bifurcation diagram ───────────────────────────────────────────────────
    bif_lo = unique_lmax.min() - mu_vals.max() - 0.5
    bif_hi = unique_lmax.max() + 0.5

    ax_bif.fill_between(mu_vals,
                        np.minimum(sweep_data["lmax_A_min_mu"], 0), 0,
                        color=C_STABLE, alpha=0.12, zorder=0)
    ax_bif.fill_between(mu_vals, 0,
                        np.maximum(sweep_data["lmax_A_max_mu"], 0),
                        color=C_UNSTABLE, alpha=0.10, zorder=0)
    ax_bif.axhline(0,          color=BLACK,    linewidth=1.5, zorder=5,
                   label="$\\lambda_{\\max}(A)=0$ (stability boundary)")
    ax_bif.axvline(current_mu, color=C_MU_LINE, linewidth=1.8,
                   linestyle="--", zorder=6,
                   label=f"current $\\mu={current_mu:.3f}$")
    ax_bif.axvline(mu_bin,     color=BLACK,    linewidth=1.8,
                   linestyle=":", zorder=7,
                   label=f"$\\mu_{{\\rm bin}}={mu_bin:.3f}$")

    ann_y_vals = np.linspace(bif_hi*0.95, bif_hi*0.05, len(unique_lmax))
    for idx, (lm, cnt, H_mean, cut_mean) in enumerate(
            zip(unique_lmax, counts_lmax, avg_H_per_lmax, avg_cut_per_lmax)):
        c  = cmap_cut(norm_cut(cut_mean))
        lw = 0.8 + 0.55 * np.log1p(cnt / 2.0)
        ax_bif.plot(mu_vals, lm - mu_vals, color=c, linewidth=lw, alpha=0.85)
        mu_star = lm
        if mu_vals[0] <= mu_star <= mu_vals[-1]:
            ax_bif.scatter([mu_star], [0.0], color=c, s=55, zorder=7,
                           edgecolors=BLACK, linewidths=0.7)
            ax_bif.annotate(
                f"$\\mu^*={mu_star:.2f}$\n"
                f"$\\bar{{H}}={H_mean:.1f}$\n"
                f"cut$={cut_mean:.1f}$ $\\times{cnt}$",
                xy=(mu_star, 0.0),
                xytext=(mu_star + (mu_vals[-1]-mu_vals[0])*0.012,
                        ann_y_vals[idx]),
                fontsize=7, color=c, zorder=8,
                arrowprops=dict(arrowstyle="->", color=c, lw=0.65,
                                shrinkA=2, shrinkB=2))

    sm_bif = plt.cm.ScalarMappable(cmap=cmap_cut, norm=norm_cut)
    sm_bif.set_array([])
    cb_bif = fig.colorbar(sm_bif, ax=ax_bif, fraction=0.012, pad=0.01)
    cb_bif.set_label("Mean cut at $\\mu^*$", fontsize=10)
    cb_bif.ax.tick_params(labelsize=9); cb_bif.outline.set_edgecolor(BLACK)

    ax_bif.set_xlim(mu_vals[0], mu_vals[-1])
    ax_bif.set_ylim(bif_lo, bif_hi)
    ax_bif.legend(fontsize=10, loc="upper right", framealpha=0.93)
    ax_bif.text(0.01, 0.04, "← stable",   transform=ax_bif.transAxes,
                ha="left", fontsize=10, color=C_STABLE)
    ax_bif.text(0.01, 0.96, "← unstable", transform=ax_bif.transAxes,
                ha="left", va="top", fontsize=10, color=C_UNSTABLE)
    _ax_style(ax_bif,
              title=("Bifurcation diagram: $\\lambda_{\\max}(D)-\\mu$ vs "
                     "$\\mu$ | line colour = cut quality | "
                     "dots = stability transitions"),
              xlabel="$\\mu$",
              ylabel="$\\lambda_{\\max}(A)=\\lambda_{\\max}(D)-\\mu$")

    fig.suptitle(
        f"OIM Bifurcation Analysis | Threshold graph $\\tt{{{seq}}}$ | "
        f"$N={n}$, $2^N={n_eq}$ equilibria | "
        f"$\\mu_{{\\rm bin}}={mu_bin:.4f}$ | "
        f"Best cut $={best_cut:.1f}$, $W_{{\\rm tot}}={w_total:.1f}$",
        color=BLACK, fontsize=12, fontweight="bold")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Quality analysis
# ══════════════════════════════════════════════════════════════════════════════

def make_figure3(graph_name, n, W, seq, eq_data, conv_results):
    rows       = eq_data["rows"]
    best_cut   = eq_data["best_cut"]
    w_total    = eq_data["w_total"]
    mu_bin     = eq_data["mu_bin"]
    current_mu = eq_data["mu"]
    reps       = pick_representatives(rows)

    H_all      = np.array([r["H"]     for r in rows])
    cut_all    = np.array([r["cut"]   for r in rows])
    stable_mask= np.array([r["stable"] for r in rows])

    fig, (ax_hcut, ax_dspec) = plt.subplots(
        1, 2, figsize=(16, 7), facecolor=WHITE)
    fig.subplots_adjust(wspace=0.32, left=0.07, right=0.97,
                        top=0.88, bottom=0.12)

    # ── H vs cut scatter ──────────────────────────────────────────────────────
    ax_hcut.scatter(cut_all[~stable_mask], H_all[~stable_mask],
                    c=C_UNSTABLE, s=20, alpha=0.45, zorder=2,
                    label=f"unstable at $\\mu={current_mu:.2f}$")
    ax_hcut.scatter(cut_all[stable_mask], H_all[stable_mask],
                    c=C_STABLE, s=45, alpha=0.85, zorder=3,
                    edgecolors=BLACK, linewidths=0.5,
                    label=f"stable at $\\mu={current_mu:.2f}$")

    cut_line = np.linspace(cut_all.min()-0.3, cut_all.max()+0.3, 300)
    ax_hcut.plot(cut_line, w_total - 2.0*cut_line, color=GRAY,
                 linewidth=1.3, linestyle="-", alpha=0.65, zorder=1,
                 label=f"$H = {w_total:.0f} - 2\\cdot$cut")
    ax_hcut.axvline(best_cut, color=C_BPART, linewidth=1.6, linestyle="--",
                    alpha=0.8, label=f"best cut $={best_cut:.1f}$")
    ax_hcut.axhline(H_all.min(), color=C_FERRO, linewidth=1.6, linestyle="--",
                    alpha=0.8, label=f"min $H={H_all.min():.1f}$")

    seen = set()
    for c in conv_results:
        if c["bits"] in seen: continue
        seen.add(c["bits"])
        col = _TYPE_COL.get(c["state_type"], GRAY)
        mk  = "*" if c["is_binary"] else "^"
        ax_hcut.scatter([c["cut"]], [c["H"]], marker=mk,
                        s=190 if c["is_binary"] else 120,
                        color=col, edgecolors=BLACK, linewidths=0.8, zorder=6)
        b_str = "".join(str(b) for b in c["bits"])
        ax_hcut.annotate(
            f"$[{b_str}]$\n{c['state_type']}",
            xy=(c["cut"], c["H"]),
            xytext=(c["cut"] + best_cut*0.025,
                    c["H"] - (H_all.max()-H_all.min())*0.04),
            fontsize=8, color=col,
            arrowprops=dict(arrowstyle="->", color=col, lw=0.7,
                            shrinkA=2, shrinkB=2))

    hnd, lbl = ax_hcut.get_legend_handles_labels()
    type_patches = [mpatches.Patch(color=v, label=k)
                    for k, v in _TYPE_COL.items()]
    ax_hcut.legend(handles=hnd + type_patches, fontsize=8.5, loc="upper right")
    _ax_style(ax_hcut,
              title=(f"Ising Hamiltonian $H$ vs cut — all $2^N={len(rows)}$ "
                     f"equilibria\n$H=W_{{\\rm tot}}-2\\cdot$cut | "
                     f"$W_{{\\rm tot}}={w_total:.1f}$ | "
                     f"best cut$={best_cut:.1f}$ | "
                     f"$\\mu_{{\\rm bin}}={mu_bin:.3f}$"),
              xlabel="Cut value", ylabel="$H(\\sigma)$", titlesize=11)

    # ── D(φ*) eigenvalue spectrum ─────────────────────────────────────────────
    xidx = np.arange(1, n+1)
    ev_all_reps = np.concatenate([eq["ev_D"] for eq, _, _, _ in reps])
    ev_lo, ev_hi = ev_all_reps.min(), ev_all_reps.max()
    margin = max(0.06*(ev_hi - ev_lo), 0.4)

    ax_dspec.fill_between([0.5, n+0.5], ev_lo-margin, 0,
                          color=C_STABLE,   alpha=0.10, zorder=0)
    ax_dspec.fill_between([0.5, n+0.5], 0, ev_hi+margin,
                          color=C_UNSTABLE, alpha=0.10, zorder=0)
    ax_dspec.axhline(0, color=BLACK, linewidth=1.0, linestyle="--",
                     alpha=0.55, zorder=2)

    for k_r, (eq, lbl, col, ls) in enumerate(reps):
        ev_desc = eq["ev_D"][::-1]
        ax_dspec.plot(xidx, ev_desc, color=col, linestyle=ls,
                      linewidth=2.2, marker="o", markersize=8,
                      label=lbl, zorder=3)
        dy = margin*(0.6 + 0.35*k_r)
        ax_dspec.annotate(
            f"$\\lambda_{{\\max}}={eq['lmax_D']:.3f}$\n"
            f"$H={eq['H']:.1f}$ cut$={eq['cut']:.1f}$",
            xy=(1, ev_desc[0]),
            xytext=(1.6, ev_desc[0]+dy),
            fontsize=9, color=col, zorder=5,
            arrowprops=dict(arrowstyle="->", color=col, lw=0.9,
                            shrinkA=2, shrinkB=2))

    ax_dspec.set_xticks(xidx)
    ax_dspec.set_xlim(0.5, n+0.5)
    ax_dspec.set_ylim(ev_lo-2*margin, ev_hi+3.5*margin)
    ax_dspec.legend(fontsize=8.5, loc="lower left", ncol=1)
    ax_dspec.text(0.98, 0.98,
                  "$A(\\phi^*,\\mu)=D(\\phi^*)-\\mu I$\n"
                  "$\\lambda_k(A)=\\lambda_k(D)-\\mu$\n"
                  "Stable iff $\\mu>\\lambda_{\\max}(D)$",
                  transform=ax_dspec.transAxes, ha="right", va="top",
                  fontsize=9.5,
                  bbox=dict(boxstyle="round,pad=0.35", facecolor=WHITE,
                            edgecolor=GRAY, alpha=0.94))
    _ax_style(ax_dspec,
              title=("$D(\\phi^*)$ eigenvalue spectrum — 3 representative "
                     "equilibria\nBlue: $\\lambda<0$ | Orange: $\\lambda>0$ | "
                     "Annotated: $\\lambda_{\\max}$, $H$, cut"),
              xlabel="Eigenvalue rank $k$ (largest first)",
              ylabel="$\\lambda_k\\left(D(\\phi^*)\\right)$",
              titlesize=11)

    fig.suptitle(
        f"OIM Quality Analysis | Threshold graph $\\tt{{{seq}}}$ | "
        f"$N={n}$, $2^N={len(rows)}$ equilibria | "
        f"$\\mu_{{\\rm bin}}={mu_bin:.4f}$ | "
        f"Best cut $={best_cut:.1f}$, $W_{{\\rm tot}}={w_total:.1f}$",
        color=BLACK, fontsize=12, fontweight="bold")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="OIM eigenvalue sweep on threshold graphs — 4 figures")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--sequence", type=str,
                     help="Binary construction sequence, e.g. '0110100'")
    grp.add_argument("--n", type=int,
                     help="Random threshold graph of size N")
    parser.add_argument("--mu_min",  type=float, default=None)
    parser.add_argument("--mu_max",  type=float, default=None)
    parser.add_argument("--n_mu",    type=int,   default=300)
    parser.add_argument("--mu",      type=float, default=None,
                        help="Reference μ for equilibrium analysis (default: auto)")
    parser.add_argument("--n_init",  type=int,   default=12)
    parser.add_argument("--t_end",   type=float, default=80.0)
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--save",    action="store_true",
                        help="Save all 4 figures as PDF+PNG")
    args = parser.parse_args()

    # ── Build threshold graph ─────────────────────────────────────────────────
    if args.sequence is not None:
        raw_seq = args.sequence.strip()
        if not all(c in "01" for c in raw_seq):
            sys.exit("--sequence must contain only '0' and '1'")
    else:
        raw_seq = random_threshold_sequence(args.n, seed=args.seed)
        print(f"Random threshold sequence (N={args.n}, seed={args.seed}): "
              f"{raw_seq}")

    W, seq = build_threshold_graph(raw_seq)
    n = W.shape[0]
    info = threshold_graph_info(W, seq)
    graph_name = f"threshold_{seq}"

    print("=" * 65)
    print("OIM EIGENVALUE SWEEP — Threshold Graph")
    print(f"  Sequence : {seq}")
    print(f"  N = {n}   |E| = {info['n_edges']}   density = {info['density']:.3f}")
    print(f"  Dominating: {info['n_dom']}   Isolated: {info['n_iso']}")
    print(f"  Degree sequence: {sorted(info['degrees'].tolist(), reverse=True)}")
    print("=" * 65)

    if n > 18:
        print(f"[warn] N={n} > 18 → 2^N={2**n} — may be very slow")

    # ── λ_max range scan (all 2^N equilibria) ─────────────────────────────────
    oim_scan = OIMMaxCut(W, mu=1.0, seed=args.seed)
    print(f"\nScanning all 2^{n}={2**n} equilibria...")
    t0 = time.time()
    lmax_scan = [
        float(np.linalg.eigvalsh(oim_scan.build_D(
            np.array([b*np.pi for b in bits], dtype=float))).max())
        for bits in iproduct([0, 1], repeat=n)
    ]
    print(f"  Done in {time.time()-t0:.1f}s")
    global_lmax_min = min(lmax_scan)
    global_lmax_max = max(lmax_scan)
    print(f"  λ_max(D) range: [{global_lmax_min:.4f}, {global_lmax_max:.4f}]")
    print(f"  μ_bin estimate : {global_lmax_min:.4f}")

    mu_min_eff = (args.mu_min if args.mu_min is not None
                  else min(0.0, global_lmax_min - 0.5))
    mu_max_eff = (args.mu_max if args.mu_max is not None
                  else global_lmax_max * 1.30)
    mu_ref     = (args.mu     if args.mu     is not None
                  else (mu_min_eff + mu_max_eff) / 2.0)

    print(f"  μ sweep : [{mu_min_eff:.4f}, {mu_max_eff:.4f}]  ({args.n_mu} steps)")
    print(f"  μ ref   : {mu_ref:.4f}")

    oim = OIMMaxCut(W, mu=mu_ref, seed=args.seed)
    print(f"\nEquilibrium analysis at μ={mu_ref:.4f}...")
    eq_data = analyse_equilibria(oim)
    print(f"  μ_bin={eq_data['mu_bin']:.4f}  stable: "
          f"{eq_data['n_stable']}/{eq_data['total']}  "
          f"best cut: {eq_data['best_cut']:.1f}")

    mu_vals    = np.linspace(mu_min_eff, mu_max_eff, args.n_mu)
    sweep_data = mu_sweep(oim, eq_data, mu_vals)
    print(f"  Bifurcation points: {len(sweep_data['bifurcation_pts'])}")

    # ── Simulation ────────────────────────────────────────────────────────────
    MU_SIM = max(global_lmax_min + 0.01, mu_min_eff + 0.02)
    rng    = np.random.default_rng(args.seed)
    phi0s  = [rng.uniform(-np.pi, np.pi, n) for _ in range(args.n_init)]
    oim_sim = OIMMaxCut(W, mu=MU_SIM, seed=args.seed)
    print(f"\nSimulating {args.n_init} trajectories at "
          f"μ={MU_SIM:.4f} (t=0..{args.t_end})...")
    t0   = time.time()
    sols = oim_sim.simulate_many(phi0s, t_span=(0., args.t_end), n_points=500)
    print(f"  Done in {time.time()-t0:.1f}s")

    conv_results = [
        identify_convergence(sol, W, eq_data["rows"], bin_tol=0.05)
        for sol in sols
    ]

    # console report
    cw = max(n, 8)
    print(f"\n  {'#':>3} {'type':<16} {'bits':<{cw}} {'H':>8} "
          f"{'cut':>8} {'residual':>9}")
    print("  " + "─" * (3 + 16 + cw + 8 + 8 + 9))
    for i, c in enumerate(conv_results):
        b = "".join(str(x) for x in c["bits"])
        print(f"  {i:>3} {c['state_type']:<16} {b:<{cw}} "
              f"{c['H']:>8.3f} {c['cut']:>8.3f} {c['residual']:>9.5f}")

    summary = defaultdict(lambda: {"count": 0, "residuals": []})
    for c in conv_results:
        key = (c["state_type"], c["bits"])
        summary[key]["count"] += 1
        summary[key]["state_type"] = c["state_type"]
        summary[key]["residuals"].append(c["residual"])
    print(f"\n  Unique terminal states: {len(summary)}")
    for (stype, bits), sm in sorted(summary.items(),
                                    key=lambda x: -x[1]["count"]):
        b = "".join(str(x) for x in bits)
        print(f"  type={stype:<16} bits={b} count={sm['count']}/{args.n_init} "
              f"mean_res={np.mean(sm['residuals']):.5f}")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    fig0 = make_figure0(W, seq, info, eq_data["mu_bin"])
    fig1 = make_figure1(graph_name, n, W, seq, eq_data, sols,
                        MU_SIM, conv_results, args.n_init)
    fig2 = make_figure2(graph_name, n, seq, eq_data, sweep_data)
    fig3 = make_figure3(graph_name, n, W, seq, eq_data, conv_results)

    if args.save:
        stem = f"threshold_{seq}"
        for tag, fig in [("structure", fig0), ("dynamics", fig1),
                         ("bifurcation", fig2), ("quality", fig3)]:
            for ext in ("pdf", "png"):
                fname = f"oim_{tag}_{stem}.{ext}"
                fig.savefig(fname, bbox_inches="tight", dpi=150)
                print(f"  Saved: {fname}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
