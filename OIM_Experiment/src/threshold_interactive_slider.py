#!/usr/bin/env python3
"""
threshold_interactive_slider.py
──────────────────────────────────────────────────────────────────────────────
Interactive eigenvalue explorer for threshold graphs parameterised by N.

All computations are pre-computed for every N in [n_min, n_max] before the
window opens. Moving the slider to N=k instantly shows the full eigenvalue /
bifurcation analysis for a threshold graph on k nodes.

Six panels (3 rows × 3 columns, row-1 wide panel spans all 3 cols)
────────────────────────────────────────────────────────────────────
[0,0] Graph topology (NetworkX spring layout)
[0,1] Adjacency matrix heatmap
[0,2] Properties text + construction table
[1,0:3] Jacobian eigenvalue paths λ_k(A) = λ_k(D) − μ vs μ  (full width)
[2,0] D-matrix eigenvalue spectrum — 3 representative equilibria
[2,1] λ_max(D) bar chart — all 2^N equilibria, sorted, coloured by cut
[2,2] Bifurcation diagram λ_max(D) − μ vs μ
       right axis: number of stable equilibria vs μ

Mathematical background (Cheng et al., Chaos 34, 073103, 2024)
────────────────────────────────────────────────────────────────
A(φ*, μ) = D(φ*) − μ·I_N → λ_k(A) = λ_k(D) − μ (slope −1 in μ)
Stability: μ > λ_max(D(φ*))
μ_bin = min_{φ*∈{0,π}^N} λ_max(D(φ*)) [Remark 7]

Usage
─────
python threshold_interactive_slider.py [options]

--n_min INT   smallest N (default: 3)
--n_max INT   largest N (default: 10)
--n_mu  INT   μ-grid points per N (default: 300)
--seed  INT   RNG seed for random threshold sequences (default: 42)
--sequence STR  comma-separated explicit sequences, one per N
              e.g. "010,0110,01010" (overrides random generation)
──────────────────────────────────────────────────────────────────────────────
"""

import argparse
import sys
import time
from itertools import product as iproduct

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("[warn] networkx not found — graph-topology panel will be blank")

from OIM_Experiment.src.OIM_mu import OIMMaxCut

# ── TikZ-like global style ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.edgecolor":     "black",
    "axes.linewidth":     0.8,
    "xtick.color":        "black",
    "ytick.color":        "black",
    "text.color":         "black",
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#b0b0b0",
    "legend.facecolor":   "white",
    "legend.labelcolor":  "black",
})

WHITE    = "#ffffff"
BLACK    = "#000000"
GRAY     = "#b0b0b0"
LIGHT    = "#e6e6e6"
C_STABLE   = "#4C72B0"
C_UNSTABLE = "#DD8452"
C_BPART  = "#4C72B0"
C_FERRO  = "#DD8452"
C_MIXED  = "#55A868"
C_MU_LINE = "#ffb74d"
C_ISO    = "#5588cc"
C_DOM    = "#cc4444"
C_MUBIN  = "#c44e52"


def _ax_style(ax, title="", xlabel="", ylabel="", titlesize=9):
    ax.set_facecolor(WHITE)
    for sp in ax.spines.values():
        sp.set_edgecolor(BLACK)
        sp.set_linewidth(0.8)
    ax.tick_params(colors=BLACK, labelsize=8)
    ax.grid(True, color=LIGHT, linewidth=0.5, alpha=1.0)
    if title:  ax.set_title(title,  color=BLACK, fontsize=titlesize,
                            fontweight="bold", pad=4)
    if xlabel: ax.set_xlabel(xlabel, color=BLACK, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=BLACK, fontsize=9)


# ═══════════════════════════════════════════════════════════════════════════════
# Threshold graph utilities
# ═══════════════════════════════════════════════════════════════════════════════

def build_threshold_graph(sequence: str):
    seq = list(sequence)
    if seq[0] != '0':
        seq[0] = '0'
    seq_str = "".join(seq)
    N = len(seq)
    W = np.zeros((N, N), dtype=float)
    for j in range(1, N):
        if seq[j] == '1':
            for i in range(j):
                W[i, j] = W[j, i] = 1.0
    return W, seq_str


def random_threshold_sequence(n: int, seed: int = 42) -> str:
    rng = np.random.default_rng(seed + n)
    bits = rng.integers(0, 2, size=n).tolist()
    bits[0] = 0
    return "".join(str(b) for b in bits)


def threshold_graph_info(W: np.ndarray, seq: str) -> dict:
    N = W.shape[0]
    degrees = W.sum(axis=1).astype(int)
    n_dom = seq.count('1')
    n_iso = seq.count('0')
    n_edges = int(W.sum()) // 2
    density = n_edges / max(N * (N - 1) // 2, 1)
    return dict(N=N, degrees=degrees, n_dom=n_dom, n_iso=n_iso,
                n_edges=n_edges, density=density,
                connected=(n_dom > 0), seq=seq)


# ═══════════════════════════════════════════════════════════════════════════════
# Eigenvalue analysis helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _jacobian(oim, phi):
    D = oim.build_D(phi)
    return D - oim.mu * np.diag(np.cos(2.0 * phi))


def analyse_equilibria(oim):
    n, mu, W = oim.n, oim.mu, oim.W
    w_total = float(np.sum(W)) / 2.0
    rows = []
    for bits in iproduct([0, 1], repeat=n):
        phi = np.array([b * np.pi for b in bits], dtype=float)
        D = oim.build_D(phi)
        ev_D = np.sort(np.linalg.eigvalsh(D))
        lmax = float(ev_D[-1])
        A = _jacobian(oim, phi)
        ev_A = np.sort(np.linalg.eigvalsh(A))
        sigma = np.sign(np.cos(phi)); sigma[sigma == 0] = 1.0
        H   = 0.5 * float(np.sum(W * np.outer(sigma, sigma)))
        cut = 0.25 * float(np.sum(W * (1.0 - np.outer(sigma, sigma))))
        rows.append(dict(bits=bits, phi=phi, D=D, ev_D=ev_D, ev_A=ev_A,
                         lmax_D=lmax, lmax_A=float(ev_A[-1]), mu_thr=lmax,
                         stable=(mu > lmax), H=H, cut=cut))
    mu_bin   = min(r["lmax_D"] for r in rows)
    best_cut = max(r["cut"]    for r in rows)
    n_stable = sum(r["stable"] for r in rows)
    return dict(rows=rows, mu_bin=mu_bin, best_cut=best_cut, w_total=w_total,
                n_stable=n_stable, total=len(rows), n=n, mu=mu)


def mu_sweep(oim, eq_data, mu_vals):
    rows = eq_data["rows"]
    paths, bif_pts = {}, []
    for r in rows:
        key = r["bits"]
        paths[key] = r["ev_D"][None, :] - mu_vals[:, None]
        if mu_vals[0] <= r["lmax_D"] <= mu_vals[-1]:
            bif_pts.append((r["lmax_D"], key))
    lmax_D_arr  = np.array([r["lmax_D"] for r in rows])
    lmax_A_mat  = lmax_D_arr[:, None] - mu_vals[None, :]
    n_stable_mu = np.sum(lmax_A_mat < 0, axis=0)
    return dict(mu_vals=mu_vals, paths=paths, n_stable_mu=n_stable_mu,
                lmax_A_min_mu=lmax_A_mat.min(axis=0),
                lmax_A_max_mu=lmax_A_mat.max(axis=0),
                bifurcation_pts=bif_pts)


def pick_representatives(rows):
    sr  = sorted(rows, key=lambda r: r["lmax_D"])
    low = sr[0]; mid = sr[len(sr) // 2]; high = sr[-1]

    def _lbl(tag, r):
        b = "".join(str(x) for x in r["bits"])
        return (f"{tag}: $[{b}]$ "
                f"$\\lambda_{{\\max}}={r['lmax_D']:.3f}$ "
                f"cut$={r['cut']:.1f}$")

    return [(low,  _lbl("Easy",   low),  C_BPART, "-"),
            (mid,  _lbl("Median", mid),  C_MIXED, "--"),
            (high, _lbl("Hard",   high), C_FERRO, ":")]


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-computation
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_all(n_min, n_max, n_mu, seed, explicit_sequences=None):
    all_data = {}
    bar_w = 36

    print(f"\n{'='*65}")
    print(f" Pre-computing threshold-graph eigenvalue data")
    print(f" N ∈ [{n_min}, {n_max}]  n_mu={n_mu}  seed={seed}")
    print(f"{'='*65}")

    for n in range(n_min, n_max + 1):
        t0 = time.perf_counter()

        # ── build graph ───────────────────────────────────────────────────
        if explicit_sequences and (n - n_min) < len(explicit_sequences):
            raw_seq = explicit_sequences[n - n_min]
            if len(raw_seq) != n:
                print(f" [warn] explicit sequence '{raw_seq}' has wrong "
                      f"length for N={n}, falling back to random")
                raw_seq = random_threshold_sequence(n, seed)
        else:
            raw_seq = random_threshold_sequence(n, seed)

        W, seq = build_threshold_graph(raw_seq)
        info   = threshold_graph_info(W, seq)

        # ── scan all 2^N equilibria to find μ range ───────────────────────
        oim_scan = OIMMaxCut(W, mu=1.0, seed=seed)
        lmax_scan = [
            float(np.linalg.eigvalsh(
                oim_scan.build_D(
                    np.array([b * np.pi for b in bits], dtype=float)
                )).max())
            for bits in iproduct([0, 1], repeat=n)
        ]

        lmax_min = min(lmax_scan)
        lmax_max = max(lmax_scan)

        mu_min_eff = min(0.0, lmax_min - 0.5)
        mu_max_eff = lmax_max * 1.30
        mu_ref     = (mu_min_eff + mu_max_eff) / 2.0
        mu_vals    = np.linspace(mu_min_eff, mu_max_eff, n_mu)

        oim     = OIMMaxCut(W, mu=mu_ref, seed=seed)
        eq_data = analyse_equilibria(oim)
        sw_data = mu_sweep(oim, eq_data, mu_vals)
        reps    = pick_representatives(eq_data["rows"])

        all_data[n] = dict(
            W=W, seq=seq, info=info,
            eq_data=eq_data, sweep_data=sw_data,
            mu_vals=mu_vals,
            lmax_min=lmax_min, lmax_max=lmax_max,
            reps=reps,
        )

        elapsed = time.perf_counter() - t0
        pct     = 100 * (n - n_min + 1) / (n_max - n_min + 1)
        filled  = int(bar_w * pct / 100)
        bar     = "█" * filled + "░" * (bar_w - filled)
        print(f" [{bar}] N={n:>2}  seq={seq}  "
              f"2^N={2**n:>4}  μ_bin={eq_data['mu_bin']:>7.4f}  "
              f"({elapsed:.1f}s)")

    print(f"\n All done — {n_max - n_min + 1} graphs precomputed.\n")
    return all_data


# ═══════════════════════════════════════════════════════════════════════════════
# Panel draw functions
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_graph(ax, W, seq, info):
    ax.cla()
    ax.axis("off")
    ax.set_facecolor(WHITE)
    N = info["N"]

    if HAS_NX:
        G   = nx.from_numpy_array(W)
        pos = nx.spring_layout(G, seed=0, k=2.2 / max(np.sqrt(N), 1))
        nc_dom = [i for i in range(N) if seq[i] == '1']
        nc_iso = [i for i in range(N) if seq[i] == '0']
        nx.draw_networkx_edges(G, pos, ax=ax,
                               edge_color=GRAY, width=1.4, alpha=0.7)
        nx.draw_networkx_nodes(G, pos, nodelist=nc_dom, ax=ax,
                               node_color=C_DOM, node_shape='o',
                               node_size=480, edgecolors=BLACK, linewidths=1.0)
        nx.draw_networkx_nodes(G, pos, nodelist=nc_iso, ax=ax,
                               node_color=C_ISO, node_shape='s',
                               node_size=380, edgecolors=BLACK, linewidths=1.0)
        nx.draw_networkx_labels(G, pos, ax=ax,
                                font_size=8, font_color=WHITE,
                                font_weight="bold")
        ax.legend(handles=[
            mpatches.Patch(facecolor=C_DOM, edgecolor=BLACK,
                           label="Dominating (1)"),
            mpatches.Patch(facecolor=C_ISO, edgecolor=BLACK,
                           label="Isolated (0)"),
        ], fontsize=7.5, loc="lower center",
           facecolor=WHITE, edgecolor=GRAY, labelcolor=BLACK)
    else:
        ax.text(0.5, 0.5, f"N={N}\nseq={seq}\n(networkx missing)",
                ha="center", va="center", fontsize=9, color=GRAY,
                transform=ax.transAxes)

    ax.set_title(
        f"Threshold graph $N={N}$  $|E|={info['n_edges']}$\n"
        f"seq: $\\tt{{{seq}}}$  "
        f"{'connected' if info['connected'] else 'disconnected'}  "
        f"density $={info['density']:.2f}$",
        color=BLACK, fontsize=8.5, fontweight="bold", pad=4)


def _draw_heatmap(ax, W, seq, fig):
    ax.cla()
    ax.set_facecolor(WHITE)
    N = W.shape[0]
    ax.imshow(W, cmap="Blues", vmin=0, vmax=1,
              interpolation="nearest", aspect="auto")
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels([f"v{i}" for i in range(N)],
                       fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels([f"v{i}" for i in range(N)], fontsize=7)
    ax.tick_params(colors=BLACK, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(BLACK); sp.set_linewidth(0.8)
    for i in range(N):
        for j in range(N):
            if W[i, j] > 0:
                ax.text(j, i, "1", ha="center", va="center",
                        fontsize=6, color=WHITE)
    ax.set_title("Adjacency matrix $W$", color=BLACK,
                 fontsize=8.5, fontweight="bold", pad=4)


def _draw_props(ax, info, eq_data):
    ax.cla()
    ax.axis("off")
    ax.set_facecolor(WHITE)
    seq  = info["seq"]
    deg  = info["degrees"]
    mu_b = eq_data["mu_bin"]
    bc   = eq_data["best_cut"]
    wt   = eq_data["w_total"]
    ns   = eq_data["n_stable"]
    tot  = eq_data["total"]
    mu_r = eq_data["mu"]

    col_labels = ["v", "type", "s[i]", "deg", "+edges"]
    table_data = []
    for i in range(info["N"]):
        vtype = "dom" if seq[i] == '1' else "iso"
        new_e = i if seq[i] == '1' else 0
        table_data.append([f"v{i}", vtype, seq[i],
                           str(int(deg[i])), str(new_e)])

    tbl = ax.table(
        cellText=table_data, colLabels=col_labels,
        bbox=[0.0, 0.45, 1.0, 0.52], cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor(LIGHT)
        tbl[0, j].set_text_props(fontweight="bold", color=BLACK)
    for i, row in enumerate(table_data, start=1):
        c = (C_DOM + "44") if row[1] == "dom" else (C_ISO + "44")
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(c)
            tbl[i, j].set_edgecolor(GRAY)
            tbl[i, j].set_linewidth(0.4)

    props = (
        f"$N={info['N']}$ vertices\n"
        f"$|E|={info['n_edges']}$ edges\n"
        f"Dom: {info['n_dom']}  Iso: {info['n_iso']}\n"
        f"Density: {info['density']:.3f}\n"
        f"Max deg: {int(deg.max())}  Min: {int(deg.min())}\n"
        f"Deg seq: {sorted(deg.tolist(), reverse=True)}\n\n"
        f"$2^N = {tot}$ equilibria\n"
        f"$\\mu_{{\\rm bin}} = {mu_b:.4f}$ (Remark 7)\n"
        f"$\\mu_{{\\rm ref}} = {mu_r:.4f}$\n"
        f"Stable at $\\mu_{{\\rm ref}}$: {ns}/{tot}\n"
        f"Best cut = {bc:.1f},  $W_{{\\rm tot}} = {wt:.1f}$"
    )

    ax.text(0.5, 0.42, props, transform=ax.transAxes,
            ha="center", va="top", fontsize=8, color=BLACK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHT,
                      edgecolor=GRAY, alpha=0.95),
            linespacing=1.55)
    ax.set_title("Construction table + key properties",
                 color=BLACK, fontsize=8.5, fontweight="bold", pad=4)


def _draw_jac_paths(ax, eq_data, sweep_data, reps, n):
    """Full-width panel: all λ_k(A) = λ_k(D) − μ paths for 3 representative eq."""
    ax.cla()
    ax.set_facecolor(WHITE)

    mu_vals  = sweep_data["mu_vals"]
    _all_ev  = np.concatenate([r["ev_D"] for r in eq_data["rows"]])
    jac_lo   = _all_ev.min() - mu_vals.max() - 0.2
    jac_hi   = _all_ev.max() + 0.4

    ax.fill_between(mu_vals, jac_lo, 0,
                    color=C_STABLE, alpha=0.08, zorder=0)
    ax.fill_between(mu_vals, 0, jac_hi,
                    color=C_UNSTABLE, alpha=0.07, zorder=0)
    ax.axhline(0, color=BLACK, linewidth=1.2, zorder=5,
               label="$\\lambda=0$ (stability boundary)")

    mu_range = mu_vals[-1] - mu_vals[0]

    for k_r, (eq, lbl, col, ls) in enumerate(reps):
        paths = eq["ev_D"][None, :] - mu_vals[:, None]   # (M, N)
        for k in range(n):
            is_top = (k == n - 1)
            ax.plot(mu_vals, paths[:, k],
                    color=col, linestyle=ls,
                    linewidth=1.6 if is_top else 0.5,
                    alpha=1.0 if is_top else 0.28,
                    zorder=4 if is_top else 2,
                    label=(lbl if is_top else None))

        mu_star = eq["lmax_D"]
        ax.axvline(mu_star, color=col, linewidth=0.9,
                   linestyle="-.", alpha=0.75, zorder=3)
        ypos = jac_hi * (0.88 - k_r * 0.14)
        ax.text(mu_star + mu_range * 0.01, ypos,
                f"$\\mu^*={mu_star:.2f}$",
                fontsize=7, color=col, fontweight="bold")

    # global μ_bin
    mu_bin = eq_data["mu_bin"]
    ax.axvline(mu_bin, color=BLACK, linewidth=1.4,
               linestyle=":", zorder=7,
               label=f"$\\mu_{{\\rm bin}}={mu_bin:.3f}$")

    ax.set_xlim(mu_vals[0], mu_vals[-1])
    ax.set_ylim(jac_lo, jac_hi)
    _ax_style(ax,
              title=("Jacobian eigenvalue paths "
                     "$\\lambda_k(A(\\phi^*,\\mu))=\\lambda_k(D(\\phi^*))-\\mu$ "
                     "vs $\\mu$\n"
                     "Bold = $\\lambda_{\\max}(A)$; thin = all $N$ eigenvalue paths; "
                     "dash-dot = bifurcation $\\mu^*$ per equilibrium"),
              xlabel="$\\mu$",
              ylabel="$\\lambda_k(A)=\\lambda_k(D)-\\mu$",
              titlesize=8.5)
    ax.legend(fontsize=7.5, loc="upper right", ncol=4, framealpha=0.92)
    ax.text(0.01, 0.03,
            "All paths are straight lines with slope $-1$. "
            "Stability switches ON when the bold path crosses $\\lambda=0$.",
            transform=ax.transAxes, fontsize=7, color=BLACK,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=WHITE,
                      edgecolor=GRAY, alpha=0.90))


def _draw_d_spectrum(ax, eq_data, reps, n):
    """D-matrix eigenvalue spectrum for 3 representative equilibria."""
    ax.cla()
    ax.set_facecolor(WHITE)

    xidx = np.arange(1, n + 1)
    ev_all_reps = np.concatenate([eq["ev_D"] for eq, _, _, _ in reps])
    ev_lo, ev_hi = ev_all_reps.min(), ev_all_reps.max()
    margin = max(0.06 * (ev_hi - ev_lo), 0.3)

    ax.fill_between([0.5, n + 0.5], ev_lo - margin, 0,
                    color=C_STABLE, alpha=0.10, zorder=0)
    ax.fill_between([0.5, n + 0.5], 0, ev_hi + margin,
                    color=C_UNSTABLE, alpha=0.10, zorder=0)
    ax.axhline(0, color=BLACK, linewidth=0.9, linestyle="--",
               alpha=0.55, zorder=2)

    for k_r, (eq, lbl, col, ls) in enumerate(reps):
        ev_desc = eq["ev_D"][::-1]
        ax.plot(xidx, ev_desc, color=col, linestyle=ls,
                linewidth=1.6, marker="o", markersize=5,
                label=lbl, zorder=3)
        dy = margin * (0.55 + 0.30 * k_r)
        ax.annotate(
            f"$\\lambda_{{\\max}}={eq['lmax_D']:.2f}$",
            xy=(1, ev_desc[0]),
            xytext=(1.5, ev_desc[0] + dy),
            fontsize=6.5, color=col, zorder=5,
            arrowprops=dict(arrowstyle="->", color=col, lw=0.7,
                            shrinkA=2, shrinkB=2))

    ax.set_xticks(xidx)
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(ev_lo - 2 * margin, ev_hi + 3 * margin)
    _ax_style(ax,
              title=("$D(\\phi^*)$ spectrum — 3 representative eq.\n"
                     "Blue: $\\lambda<0$ | Orange: $\\lambda>0$"),
              xlabel="Eigenvalue rank $k$ (largest first)",
              ylabel="$\\lambda_k(D(\\phi^*))$",
              titlesize=8.5)
    ax.legend(fontsize=6.5, loc="lower left", ncol=1)


def _draw_lmax_bar(ax, eq_data):
    """λ_max(D) bar chart for all 2^N equilibria, sorted, coloured by cut."""
    ax.cla()
    ax.set_facecolor(WHITE)

    rows       = eq_data["rows"]
    mu_bin     = eq_data["mu_bin"]
    best_cut   = eq_data["best_cut"]
    current_mu = eq_data["mu"]
    n_eq       = eq_data["total"]

    sorted_rows = sorted(rows, key=lambda r: r["lmax_D"])
    lmax_arr    = np.array([r["lmax_D"] for r in sorted_rows])
    cut_arr     = np.array([r["cut"]    for r in sorted_rows])

    cmap_cut = plt.get_cmap("RdYlGn")
    norm_cut = mcolors.Normalize(vmin=0, vmax=max(best_cut, 1e-9))
    bar_cols = [cmap_cut(norm_cut(c)) for c in cut_arr]

    ax.bar(np.arange(n_eq), lmax_arr, color=bar_cols,
           width=1.0, edgecolor="none", zorder=2)
    ax.axhline(0, color=BLACK, linewidth=0.8, zorder=3)
    ax.axhline(mu_bin, color=C_MUBIN, linewidth=1.6,
               linestyle="--", zorder=5,
               label=f"$\\mu_{{\\rm bin}}={mu_bin:.3f}$")
    ax.axhline(current_mu, color=C_MU_LINE, linewidth=1.4,
               linestyle="--", zorder=6,
               label=f"$\\mu_{{\\rm ref}}={current_mu:.3f}$")

    ax.set_xlim(-1, n_eq)
    _ax_style(ax,
              title=(f"$\\lambda_{{\\max}}(D(\\phi^*))$ — all $2^N={n_eq}$ eq. (sorted)\n"
                     f"Colour = cut quality | "
                     f"stable at $\\mu_{{\\rm ref}}$: "
                     f"{eq_data['n_stable']}/{n_eq}"),
              xlabel="Equilibrium index",
              ylabel="$\\lambda_{\\max}(D(\\phi^*))$",
              titlesize=8.5)
    ax.legend(fontsize=7.5, loc="upper left")


def _draw_bifurcation(ax, eq_data, sweep_data):
    """Bifurcation diagram + n_stable(μ) on right axis."""
    ax.cla()
    ax.set_facecolor(WHITE)

    rows    = eq_data["rows"]
    mu_vals = sweep_data["mu_vals"]
    n_eq    = eq_data["total"]
    mu_bin  = eq_data["mu_bin"]

    lmax_D  = np.array([r["lmax_D"] for r in rows])
    cut_arr = np.array([r["cut"]    for r in rows])
    best_cut = eq_data["best_cut"]
    unique_lmax, inverse, counts_lmax = np.unique(
        np.round(lmax_D, 5), return_inverse=True, return_counts=True)
    avg_cut = np.array([cut_arr[inverse == k].mean()
                        for k in range(len(unique_lmax))])

    bif_lo = unique_lmax.min() - mu_vals.max() - 0.3
    bif_hi = unique_lmax.max() + 0.4

    ax.fill_between(mu_vals,
                    np.minimum(sweep_data["lmax_A_min_mu"], 0), 0,
                    color=C_STABLE, alpha=0.12, zorder=0)
    ax.fill_between(mu_vals, 0,
                    np.maximum(sweep_data["lmax_A_max_mu"], 0),
                    color=C_UNSTABLE, alpha=0.10, zorder=0)
    ax.axhline(0,      color=BLACK,  linewidth=1.2, zorder=5)
    ax.axvline(mu_bin, color=C_MUBIN, linewidth=1.5,
               linestyle=":", zorder=7,
               label=f"$\\mu_{{\\rm bin}}={mu_bin:.3f}$")

    cmap_bif = plt.get_cmap("RdYlGn")
    norm_bif = mcolors.Normalize(vmin=0, vmax=max(best_cut, 1e-9))
    mu_range = mu_vals[-1] - mu_vals[0]
    ann_ys   = np.linspace(bif_hi * 0.92, bif_hi * 0.08, len(unique_lmax))

    for idx, (lm, cnt, cut_mean) in enumerate(
            zip(unique_lmax, counts_lmax, avg_cut)):
        c  = cmap_bif(norm_bif(cut_mean))
        lw = 0.7 + 0.45 * np.log1p(cnt / 2.0)
        ax.plot(mu_vals, lm - mu_vals, color=c, linewidth=lw, alpha=0.88)
        if mu_vals[0] <= lm <= mu_vals[-1]:
            ax.scatter([lm], [0.0], color=c, s=28, zorder=7,
                       edgecolors=BLACK, linewidths=0.5)
            ax.annotate(
                f"$\\mu^*={lm:.2f}$ ×{cnt}",
                xy=(lm, 0.0),
                xytext=(lm + mu_range * 0.025, ann_ys[idx]),
                fontsize=5.8, color=c, zorder=8,
                arrowprops=dict(arrowstyle="->", color=c, lw=0.55,
                                shrinkA=2, shrinkB=2))

    ax.set_xlim(mu_vals[0], mu_vals[-1])
    ax.set_ylim(bif_lo, bif_hi)

    # right axis: n_stable(μ)
    ax2 = ax.twinx()
    ax2.plot(mu_vals, sweep_data["n_stable_mu"],
             color=C_STABLE, linewidth=1.8, linestyle="-", alpha=0.85,
             zorder=4, label="# stable eq.")
    ax2.set_ylabel("# stable equilibria", color=C_STABLE, fontsize=8)
    ax2.tick_params(axis="y", colors=C_STABLE, labelsize=7)
    ax2.set_ylim(-0.5, n_eq + 0.5)
    for sp in ax2.spines.values():
        sp.set_edgecolor(BLACK); sp.set_linewidth(0.8)
    ax2.legend(fontsize=7, loc="center right")

    _ax_style(ax,
              title=("Bifurcation diagram "
                     "$\\lambda_{\\max}(D)-\\mu$ vs $\\mu$\n"
                     "Dots = stability transitions | "
                     "right axis = # stable eq."),
              xlabel="$\\mu$",
              ylabel="$\\lambda_{\\max}(A)=\\lambda_{\\max}(D)-\\mu$",
              titlesize=8.5)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.text(0.97, 0.06, "← stable",
            transform=ax.transAxes, ha="right", fontsize=7, color=C_STABLE)
    ax.text(0.97, 0.94, "← unstable",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=C_UNSTABLE)


# ═══════════════════════════════════════════════════════════════════════════════
# Master redraw — calls all six panels
# ═══════════════════════════════════════════════════════════════════════════════

def redraw_all(n, data, axes, fig, n_min, n_max):
    """Redraw all six panels for the given N."""
    ax_graph, ax_heat, ax_props, ax_jac, ax_spec, ax_bar, ax_bif = axes

    W       = data["W"]
    seq     = data["seq"]
    info    = data["info"]
    eq_data = data["eq_data"]
    sw_data = data["sweep_data"]
    reps    = data["reps"]

    _draw_graph(ax_graph, W, seq, info)
    _draw_heatmap(ax_heat, W, seq, fig)
    _draw_props(ax_props, info, eq_data)
    _draw_jac_paths(ax_jac, eq_data, sw_data, reps, n)
    _draw_d_spectrum(ax_spec, eq_data, reps, n)
    _draw_lmax_bar(ax_bar, eq_data)
    _draw_bifurcation(ax_bif, eq_data, sw_data)

    fig.suptitle(
        f"OIM Threshold-Graph Eigenvalue Explorer | "
        f"$N = {n}$ (slider: {n_min} → {n_max}) | "
        f"seq: $\\tt{{{seq}}}$ | "
        f"$2^N = {2**n}$ equilibria | "
        f"$\\mu_{{\\rm bin}} = {eq_data['mu_bin']:.4f}$ | "
        f"best cut $= {eq_data['best_cut']:.1f}$",
        color=BLACK, fontsize=11, fontweight="bold", y=1.002)

    fig.canvas.draw_idle()


# ═══════════════════════════════════════════════════════════════════════════════
# Interactive figure
# ═══════════════════════════════════════════════════════════════════════════════

def make_interactive(all_data, n_min, n_max):
    n_arr = np.arange(n_min, n_max + 1)

    def _snap_n(val):
        """Convert slider values to a valid integer N."""
        return int(np.clip(int(round(float(val))), n_min, n_max))

    fig = plt.figure(figsize=(22, 15), facecolor=WHITE)

    # ── GridSpec: 3 rows × 3 cols ─────────────────────────────────────────
    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.52, wspace=0.38,
        left=0.05, right=0.97, top=0.95, bottom=0.10)

    ax_graph = fig.add_subplot(gs[0, 0])
    ax_heat  = fig.add_subplot(gs[0, 1])
    ax_props = fig.add_subplot(gs[0, 2])
    ax_jac   = fig.add_subplot(gs[1, :])   # full-width row 1 (all 3 cols)
    ax_spec  = fig.add_subplot(gs[2, 0])
    ax_bar   = fig.add_subplot(gs[2, 1])
    ax_bif   = fig.add_subplot(gs[2, 2])

    axes = (ax_graph, ax_heat, ax_props,
            ax_jac,
            ax_spec, ax_bar, ax_bif)

    # ── Slider ────────────────────────────────────────────────────────────
    ax_slider = fig.add_axes((0.15, 0.025, 0.70, 0.025))
    slider = Slider(
        ax_slider, "$N$",
        valmin=n_min, valmax=n_max,
        valinit=n_min,
        valstep=1,              # one step per integer N
        color=C_STABLE, track_color=LIGHT)
    ax_slider.set_facecolor(WHITE)
    slider.label.set_color(BLACK)
    slider.valtext.set_color(BLACK)
    slider.valtext.set_fontsize(11)

    # ◀ / ▶ step buttons
    ax_btn_l = fig.add_axes((0.08, 0.020, 0.04, 0.035))
    ax_btn_r = fig.add_axes((0.88, 0.020, 0.04, 0.035))
    btn_l = Button(ax_btn_l, "◀", color=LIGHT, hovercolor=LIGHT)
    btn_r = Button(ax_btn_r, "▶", color=LIGHT, hovercolor=LIGHT)
    for b in (btn_l, btn_r):
        b.label.set_color(BLACK)
        b.label.set_fontsize(12)

    # Tick labels on the slider axis
    ax_slider.set_xticks(n_arr)
    ax_slider.set_xticklabels([str(n) for n in n_arr],
                               fontsize=9, color=BLACK)

    # ── Callbacks ─────────────────────────────────────────────────────────
    def update(val):
        n = _snap_n(val)
        redraw_all(n, all_data[n], axes, fig, n_min, n_max)

    def step_left(_):
        slider.set_val(max(n_min, _snap_n(slider.val) - 1))

    def step_right(_):
        slider.set_val(min(n_max, _snap_n(slider.val) + 1))

    slider.on_changed(update)
    btn_l.on_clicked(step_left)
    btn_r.on_clicked(step_right)

    # Initial render
    update(n_min)
    # Prevent garbage collection of widgets
    fig._widgets = [slider, btn_l, btn_r]
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Interactive N-slider eigenvalue explorer "
                    "for threshold graphs")
    parser.add_argument("--n_min",     type=int, default=3,
                        help="Smallest N (default: 3)")
    parser.add_argument("--n_max",     type=int, default=10,
                        help="Largest N (default: 10)")
    parser.add_argument("--n_mu",      type=int, default=300,
                        help="μ-grid points per N (default: 300)")
    parser.add_argument("--seed",      type=int, default=42,
                        help="RNG seed for random sequences (default: 42)")
    parser.add_argument("--sequence",  type=str, default=None,
                        help="Comma-separated explicit sequences, one per N. "
                             "E.g. '010,0110,01010' (length must match N)")
    args = parser.parse_args()

    if args.n_min < 2:
        sys.exit("--n_min must be ≥ 2")
    if args.n_max > 20:
        print(f"[warn] N={args.n_max} > 20 → 2^N={2**args.n_max} — "
              "equilibrium scan will be slow")
    if args.n_max < args.n_min:
        sys.exit("--n_max must be ≥ --n_min")

    explicit_seqs = None
    if args.sequence is not None:
        explicit_seqs = [s.strip() for s in args.sequence.split(",")]

    all_data = precompute_all(args.n_min, args.n_max, args.n_mu,
                              args.seed, explicit_seqs)

    print(" Launching interactive window …")
    print(" Use the slider or ◀ / ▶ buttons to switch N.\n")
    fig = make_interactive(all_data, args.n_min, args.n_max)
    plt.show()


if __name__ == "__main__":
    main()
