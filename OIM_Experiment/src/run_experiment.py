"""
run_experiment.py
-----------------
OIM Stability Experiments — A, B (interactive), C (interactive).
Uses OIMMaxCut (OIM_mu_v2) exclusively. TikZ / PGFPlots style throughout.

Experiments
-----------
  A : Scatter — per-equilibrium stability threshold mu*_i = lambda_max(D(phi*_i))
      vs mu, for all 2^N binary equilibria.
      Orange = globally optimal; Blue = sub-optimal.

  B : Interactive histogram — mu slider pre-computes the Ising H distribution
      over n_trials random restarts.  The histogram, binarisation fraction,
      and best-cut annotation update live as the slider moves.

  C : Interactive two-panel figure (mu slider).
      LEFT  : lambda_max(Jacobian A(phi*)) min/max band vs OIM Lyapunov energy.
              Uses the FULL Jacobian A = D - mu*diag(cos(2*phi*)), whose
              eigenvalues shift linearly with mu (= true Lyapunov exponents).
              Zero-crossing marks the stability transition — this is what
              makes the left panel animate as mu changes.
      RIGHT : Pre-computed mean Ising H +/- 1 std vs K_s.  Static sweep with
              live slider cursor, theory K_s* line, empirical onset, and
              clickable binarised dots.
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from matplotlib.widgets import Slider

# ── TikZ / PGFPlots global style ───────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "serif",
    "font.size"        : 10,
    "axes.edgecolor"   : "black",
    "axes.linewidth"   : 0.8,
    "xtick.color"      : "black",
    "ytick.color"      : "black",
    "text.color"       : "black",
    "figure.facecolor" : "white",
    "axes.facecolor"   : "white",
})

WHITE  = "#ffffff"
BLACK  = "#000000"
GRAY   = "#b0b0b0"
LIGHT  = "#e6e6e6"
BLUE   = "#4C72B0"
ORANGE = "#DD8452"
ACCENT = "#1f77b4"

# ── Import OIMMaxCut ────────────────────────────────────────────────────────────
try:
    from OIM_Experiment.src.OIM_mu import OIMMaxCut
except ModuleNotFoundError:
    from OIM_Experiment.src.OIM_mu import OIMMaxCut

# ── Graph reader ────────────────────────────────────────────────────────────────
def _load_graph(path: str) -> np.ndarray:
    """Return positive weight matrix W (OIMMaxCut convention)."""
    try:
        try:
            from graph_utils import read_graph
        except ModuleNotFoundError:
            from OIM_Experiment.src.graph_utils import read_graph
        return np.array(read_graph(path), dtype=float)
    except (ModuleNotFoundError, ImportError):
        pass
    try:
        try:
            from graph_utils import read_graph
        except ModuleNotFoundError:
            from OIM_Experiment.src.graph_utils import read_graph
        return -np.array(read_graph(path), dtype=float)   # W = -J
    except (ModuleNotFoundError, ImportError):
        raise ImportError("Cannot find a graph reader (graph_utils or read_graphs).")


def _graph_info(W: np.ndarray) -> dict:
    N = W.shape[0]
    edges = int(np.sum(W > 0)) // 2
    return {"N": N, "edges": edges, "density": 2 * edges / max(N * (N - 1), 1)}


# ===========================================================================
# Shared computation helpers
# ===========================================================================

def all_phase_configs(N: int) -> List[np.ndarray]:
    """All 2^N binary phase configurations in {0, pi}^N."""
    return [np.array(cfg, dtype=float) * np.pi
            for cfg in product([0, 1], repeat=N)]


def scan_configs_D(oim: OIMMaxCut) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (OIM Lyapunov energy, mu*_i = lambda_max(D(phi*))) for every
    2^N binary equilibrium.  These values are mu-INDEPENDENT: D(phi*) depends
    only on W and phi*, not on the current mu.
    Used in Experiment A (static scatter).
    """
    configs = all_phase_configs(oim.n)
    energies, lambdas = [], []
    for phi in configs:
        oim.theta = phi.copy()
        energies.append(oim.get_energy())
        lambdas.append(oim.stability_threshold(phi))   # lambda_max(D)
    return np.array(energies), np.array(lambdas)


def scan_configs_jacobian(oim: OIMMaxCut) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (OIM Lyapunov energy, lambda_max(Jacobian A(phi*))) for every
    2^N binary equilibrium.

    The Jacobian is:
        A(phi*) = D(phi*) - mu * diag(cos(2 * phi*_i))

    At binary phi* in {0, pi}:  cos(2 * phi*_i) = 1  for all i, so
        A(phi*) = D(phi*) - mu * I
        lambda_max(A) = lambda_max(D) - mu

    This quantity is mu-DEPENDENT: it shifts linearly downward as mu increases,
    crossing zero at the equilibrium's stability threshold mu*_i.
    This is the TRUE Lyapunov exponent — negative means stable.
    Used in Experiment C (live left panel).
    """
    configs = all_phase_configs(oim.n)
    energies, lambdas = [], []
    for phi in configs:
        oim.theta = phi.copy()
        energies.append(oim.get_energy())
        A = oim.jacobian(phi)                                 # D - mu*diag(cos(2*phi*))
        lambdas.append(float(np.linalg.eigvalsh(A).max()))   # true Lyapunov exponent
    return np.array(energies), np.array(lambdas)


def ground_ising_energy(oim: OIMMaxCut) -> float:
    """Minimum Ising H over all 2^N binary configs (mu-independent)."""
    return float(min(oim.get_hamiltonian(theta=phi)
                     for phi in all_phase_configs(oim.n)))


def lambda_vs_energy(
    E: np.ndarray, lam: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group equilibria by energy level; return min/max lambda per level."""
    unique_E = np.array(sorted(set(np.round(E, 8))))
    lam_min, lam_max = [], []
    for e in unique_E:
        mask = np.isclose(E, e)
        lam_min.append(lam[mask].min())
        lam_max.append(lam[mask].max())
    return unique_E, np.array(lam_min), np.array(lam_max)


def global_min_energy(energies: np.ndarray) -> float:
    return float(energies.min())


def random_phi0(N: int) -> np.ndarray:
    return np.random.uniform(0.0, 2.0 * np.pi, size=N)


def run_trials(oim: OIMMaxCut, n_trials: int = 50,
               t_span: Tuple[float, float] = (0.0, 50.0),
               n_points: int = 500) -> Tuple[np.ndarray, int]:
    """
    Run n_trials random simulations.
    Returns (H_finals array, count naturally binarised).
    H is computed via get_hamiltonian(theta=...) which binarises internally.
    """
    H_finals, n_bin = [], 0
    for _ in range(n_trials):
        sol = oim.simulate(random_phi0(oim.n), t_span, n_points)
        if sol is None:
            continue
        phi_f     = sol.y[:, -1]
        H_finals.append(oim.get_hamiltonian(theta=phi_f))
        oim.theta = phi_f
        if oim.is_binarized(tol=0.1):
            n_bin += 1
    return np.array(H_finals), n_bin


# ── Axis styling helpers ───────────────────────────────────────────────────────
def _style_ax(ax, grid_axis: str = "both") -> None:
    ax.set_facecolor(WHITE)
    ax.tick_params(colors=BLACK, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(BLACK)
        sp.set_linewidth(0.8)
    ax.grid(True, color=LIGHT, linewidth=0.6, axis=grid_axis)


def _tikz_bbox() -> dict:
    return dict(boxstyle="round,pad=0.4", facecolor=WHITE,
                edgecolor=GRAY, alpha=1.0)


def _style_slider_ax(ax_sl) -> None:
    ax_sl.set_facecolor(WHITE)
    for sp in ax_sl.spines.values():
        sp.set_edgecolor(GRAY)


# ===========================================================================
# Experiment A — lambda_max(D) scatter vs mu  (static)
# ===========================================================================

def experiment_A(W: np.ndarray, mu_values: np.ndarray,
                 *, warn_large_N: int = 16) -> None:
    """
    For each mu in mu_values, compute the per-equilibrium stability threshold
        mu*_i = lambda_max( D(phi*_i) )
    for all 2^N binary equilibria and scatter them.

    Note: these thresholds are mu-INDEPENDENT (D doesn't depend on mu).
    The scatter therefore shows fixed threshold marks across all mu steps.
    The meaningful reading is: at each mu (X-axis), which equilibria have
    mu*_i < mu (already stable) vs mu*_i > mu (still unstable).

    Theorem 2:  phi*_i  stable  <=>  mu > mu*_i = lambda_max(D(phi*_i))
    Remark 7:   mu_bin = min_i mu*_i  (first threshold crossed = easiest eq.)
    """
    N = W.shape[0]
    if N > warn_large_N:
        print(f"[Exp A] WARNING: N={N} -> 2^N={2**N} configs. May be slow.")
    print(f"[Exp A] Scanning {len(mu_values)} mu values x {2**N} configs ...")

    all_results = []
    for i, mu in enumerate(mu_values):
        oim = OIMMaxCut(W, mu=mu)
        E, lam = scan_configs_D(oim)
        all_results.append((mu, E, lam))
        if (i + 1) % 5 == 0 or (i + 1) == len(mu_values):
            print(f"  step {i+1}/{len(mu_values)}", end="\r", flush=True)
    print()

    fig, ax = plt.subplots(figsize=(9, 5),
                           num="Experiment A: lambda_max(D) vs mu")
    fig.patch.set_facecolor(WHITE)

    for mu, E, lam in all_results:
        E0        = global_min_energy(E)
        is_global = np.isclose(E, E0)
        ax.scatter(np.full((~is_global).sum(), mu), lam[~is_global],
                   s=6, alpha=0.30, color=BLUE, linewidths=0)
        if is_global.any():
            ax.scatter(np.full(is_global.sum(), mu), lam[is_global],
                       s=22, alpha=0.88, color=ORANGE, zorder=3, linewidths=0)

    ax.axhline(0.0, color=BLACK, linestyle="--", linewidth=1.2)
    _style_ax(ax)
    ax.set_xlabel(r"Injection strength  $\mu = 2K_s$", fontsize=12, color=BLACK)
    ax.set_ylabel(r"Threshold  $\mu^*_i = \lambda_{\max}(D(\phi^*_i))$",
                  fontsize=12, color=BLACK)
    ax.set_title(
        r"Exp A — Stability threshold $\lambda_{\max}(D)$ for all binary"
        r" equilibria vs $\mu$"
        f"\n(N={N},  K=1 fixed — thresholds are mu-independent; "
        r"equilibrium stable when $\mu > \mu^*_i$)",
        color=BLACK,
    )
    ax.legend(handles=[
        mpatches.Patch(color=BLUE,   label="Sub-optimal equilibria"),
        mpatches.Patch(color=ORANGE, label="Globally optimal equilibria"),
        mlines.Line2D([0], [0], color=BLACK, linestyle="--",
                      label=r"$\mu^*_i = 0$"),
    ], fontsize=9, facecolor=WHITE, edgecolor=GRAY, labelcolor=BLACK)
    fig.tight_layout()
    print("[Exp A] Close the window to proceed.")
    plt.show()


# ===========================================================================
# Experiment B — Interactive histogram with mu slider
# ===========================================================================

def experiment_B(W: np.ndarray, mu_min: float, mu_max: float,
                 n_mu_steps: int, n_trials: int,
                 t_span: Tuple[float, float], n_points: int) -> None:
    """
    Interactive Ising-H histogram driven by a mu slider.

    ALL data is pre-computed upfront.  A single shared bin grid is built
    from the union of all H values across every mu step, so the axes are
    completely fixed — only bar heights change as the slider moves.

    Moving the slider updates bar heights in-place (no redraw, no axis
    rescaling), so the animation is smooth and stable.

    Each frame shows:
      Blue bars  : H count distribution at the current mu (fixed bins)
      Orange --  : Ground-state H (best possible cut, mu-independent)
      Black -.-  : mu_bin theory threshold (Remark 7)
      Annotation : mu, K_s, binarisation %, best cut at this mu
    """
    N = W.shape[0]
    mu_values = np.linspace(mu_min, mu_max, n_mu_steps)

    # ── Theory threshold ──────────────────────────────────────────────────
    print("[Exp B] Computing binarisation threshold (Remark 7) ...")
    oim_ref  = OIMMaxCut(W, mu=1.0)
    theory   = oim_ref.binarization_threshold()
    mu_bin   = theory["mu_bin"]
    H_ground = ground_ising_energy(oim_ref)
    W_total  = oim_ref.get_w_total()
    print(f"  mu_bin = {mu_bin:.4f}   Ground state H = {H_ground:.4f}")

    # ── Pre-compute all trial data ─────────────────────────────────────────
    print(f"[Exp B] Pre-computing {n_mu_steps} mu steps x {n_trials} trials ...")
    all_H_data   = []
    all_frac_bin = []
    all_best_cut = []

    for i, mu in enumerate(mu_values):
        oim = OIMMaxCut(W, mu=mu)
        H_arr, n_bin = run_trials(oim, n_trials, t_span, n_points)
        all_H_data.append(H_arr)
        frac     = n_bin / max(len(H_arr), 1)
        best_cut = float((W_total - H_arr.min()) / 2.0) if len(H_arr) else float("nan")
        all_frac_bin.append(frac)
        all_best_cut.append(best_cut)
        pct = 100 * (i + 1) / n_mu_steps
        prog = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  [{prog}] {pct:5.1f}%  mu={mu:.3f}  "
              f"binarised={frac*100:.0f}%  best_cut={best_cut:.1f}",
              end="\r", flush=True)
    print()

    # ── Build ONE global fixed bin grid from all H values ─────────────────
    # This is the key: bins never change, so axes never jump.
    all_H_concat = np.concatenate([H for H in all_H_data if len(H)])
    H_lo = int(np.floor(min(all_H_concat.min(), H_ground))) - 1
    H_hi = int(np.ceil( max(all_H_concat.max(), H_ground))) + 1
    global_bins  = np.arange(H_lo, H_hi + 1, 1.0)       # integer-spaced bins
    bin_centers  = 0.5 * (global_bins[:-1] + global_bins[1:])
    n_bins       = len(bin_centers)

    # Pre-compute counts for every mu step using the fixed bin grid
    all_counts = np.zeros((n_mu_steps, n_bins), dtype=float)
    for i, H_arr in enumerate(all_H_data):
        if len(H_arr):
            counts, _ = np.histogram(H_arr, bins=global_bins)
            all_counts[i] = counts

    # Fixed y-axis: tallest bar across the entire sweep + 15% headroom
    y_max = all_counts.max() * 1.18
    bar_width = 0.80   # slightly narrower than bin width so edges show

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 6), num="Experiment B: Interactive Histogram",
                     facecolor=WHITE)
    ax_hist = fig.add_axes((0.09, 0.20, 0.87, 0.68))
    ax_sl   = fig.add_axes((0.15, 0.06, 0.70, 0.03))

    _style_ax(ax_hist)
    _style_slider_ax(ax_sl)

    # Draw bars once — we will only update their heights later
    init_idx   = n_mu_steps // 2
    bar_rects  = ax_hist.bar(
        bin_centers, all_counts[init_idx],
        width=bar_width, color=BLUE,
        edgecolor=BLACK, linewidth=0.5, alpha=0.85, zorder=2,
    )

    # Fixed axes — never change
    ax_hist.set_xlim(H_lo - 0.5, H_hi + 0.5)
    ax_hist.set_ylim(0, y_max)
    ax_hist.set_xlabel("Ising Hamiltonian H  (hard-binarised final state)",
                       fontsize=12, color=BLACK)
    ax_hist.set_ylabel(f"Count  (out of {n_trials} trials)",
                       fontsize=11, color=BLACK)

    # Live annotation text
    annot = ax_hist.text(
        0.98, 0.97, "", transform=ax_hist.transAxes,
        ha="right", va="top", fontsize=9, color=BLACK,
        bbox=_tikz_bbox(), zorder=5,
    )

    # Title text object (updated on slider move)
    title_obj = ax_hist.set_title("", color=BLACK)

    def _refresh(idx: int) -> None:
        """Update bar heights and annotation for mu index idx."""
        mu   = mu_values[idx]
        Ks   = mu / 2.0
        frac = all_frac_bin[idx]
        bcut = all_best_cut[idx]

        # Update each bar height in place — zero-height bars stay invisible
        for rect, h in zip(bar_rects, all_counts[idx]):
            rect.set_height(h)

        status       = "BINARISED ✓" if frac >= 0.5 else "not binarised ✗"
        status_color = BLUE if frac >= 0.5 else ORANGE
        annot.set_text(
            f"$\\mu$ = {mu:.4f}   $K_s$ = {Ks:.4f}\n"
            f"Binarised: {frac*100:.0f}% of trials   {status}\n"
            f"Best cut found = {bcut:.1f}   "
            f"($\\mu_{{\\rm bin}}$ = {mu_bin:.4f})"
        )
        annot.set_color(status_color)

        title_obj.set_text(
            fr"Ising H distribution   $\mu$ = {mu:.4f}   "
            fr"$K_s$ = {Ks:.4f}   (N={N},  {n_trials} trials)"
            "\n")

    _refresh(init_idx)

    # ── Slider ────────────────────────────────────────────────────────────
    slider = Slider(ax_sl, r"$\mu$", mu_min, mu_max,
                    valinit=mu_values[init_idx],
                    valstep=float(mu_values[1] - mu_values[0]),
                    color=ACCENT, track_color=LIGHT)
    slider.label.set_color(BLACK)
    slider.valtext.set_color(BLACK)

    def update(val: float) -> None:
        idx = int(np.argmin(np.abs(mu_values - slider.val)))
        _refresh(idx)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    print("[Exp B] Drag the slider to explore. Close when done.")
    plt.show()


# ===========================================================================
# Experiment C — interactive two-panel figure (mu slider)
# ===========================================================================

def experiment_C(W: np.ndarray, mu_init: float, mu_min: float, mu_max: float,
                 *, warn_large_N: int = 16,
                 n_trials_sweep: int = 30, t_end_sweep: float = 50.0,
                 n_points_sweep: int = 500, n_ks_steps: int = 40) -> None:
    """
    Two-panel interactive figure driven by a mu slider.

    LEFT panel — lambda_max(Jacobian A(phi*)) min/max vs OIM Lyapunov energy.

      Key: uses the FULL Jacobian  A = D(phi*) - mu * diag(cos(2*phi*_i)).
      At binary phi*: cos(2*phi*_i) = 1, so A = D - mu*I, which gives
          lambda_max(A) = lambda_max(D) - mu
      This is the TRUE Lyapunov exponent and changes linearly with mu:
        - Negative => equilibrium is stable
        - Positive => equilibrium is unstable
        - Zero crossing at mu = mu*_i (the per-equilibrium threshold)
      Moving the slider shifts all lines downward uniformly, showing
      which equilibria flip from unstable (>0) to stable (<0).

    RIGHT panel — Pre-computed mean Ising H +/- 1 std vs K_s.
      Static sweep; slider drives the live cursor line and info box only.
      Markers:
        Black -.-  : K_s* = mu_bin / 2   (Remark 7 theory)
        Orange --  : Empirical K_s* (first point >= 50% binarised)
        Blue  |    : Current slider K_s (moves with slider)
        Orange dots: Binarised grid points (clickable tooltip)
    """
    N = W.shape[0]
    if N > warn_large_N:
        print(f"[Exp C] WARNING: N={N} -> 2^N={2**N} configs per slider move.")

    # ── Theory threshold ──────────────────────────────────────────────────
    print(f"[Exp C] Computing D-matrix thresholds for {2**N} equilibria ...")
    oim_theory = OIMMaxCut(W, mu=1.0)
    theory     = oim_theory.binarization_threshold()
    mu_bin     = theory["mu_bin"]
    Ks_theory  = theory["Ks_bin"]
    per_eq     = theory["all_thresholds"]
    print(f"  mu_bin (Remark 7) = {mu_bin:.4f}   K_s* = {Ks_theory:.4f}")
    print(f"  Easiest eq: {theory['easiest_eq']}  thr={min(per_eq.values()):.4f}")
    print(f"  Hardest eq: {theory['hardest_eq']}  thr={max(per_eq.values()):.4f}")

    H_ground = ground_ising_energy(oim_theory)
    print(f"  Ground state Ising H = {H_ground:.4f}")

    # ── Pre-compute K_s empirical sweep (RIGHT panel, static) ─────────────
    ks_grid      = np.linspace(mu_min / 2.0, mu_max / 2.0, n_ks_steps)
    mean_H_arr   = np.full(n_ks_steps, np.nan)
    std_H_arr    = np.full(n_ks_steps, np.nan)
    frac_bin_arr = np.zeros(n_ks_steps)

    print(f"[Exp C] Pre-computing {n_ks_steps} pts x {n_trials_sweep} trials ...")
    for idx, Ks in enumerate(ks_grid):
        oim = OIMMaxCut(W, mu=2.0 * Ks)
        H_arr, n_bin = run_trials(oim, n_trials_sweep,
                                  (0.0, t_end_sweep), n_points_sweep)
        if len(H_arr):
            mean_H_arr[idx]   = float(H_arr.mean())
            std_H_arr[idx]    = float(H_arr.std())
            frac_bin_arr[idx] = n_bin / len(H_arr)
        if (idx + 1) % 10 == 0 or (idx + 1) == n_ks_steps:
            print(f"  Ks step {idx+1}/{n_ks_steps}", end="\r", flush=True)
    print()

    bin_mask          = frac_bin_arr >= 0.5
    ks_star_empirical = float(ks_grid[bin_mask][0]) if bin_mask.any() else None
    valid             = ~np.isnan(mean_H_arr)

    # ── Figure layout ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 7),
                     num="Experiment C: Interactive Stability",
                     facecolor=WHITE)
    ax_left  = fig.add_axes((0.06, 0.20, 0.41, 0.70))
    ax_right = fig.add_axes((0.55, 0.20, 0.42, 0.70))
    ax_sl    = fig.add_axes((0.15, 0.06, 0.70, 0.03))

    _style_ax(ax_left)
    _style_ax(ax_right)
    _style_slider_ax(ax_sl)

    # ── LEFT PANEL — Jacobian eigenvalue min/max vs Lyapunov energy ────────
    def _compute_lam(mu: float):
        """
        Build OIMMaxCut at current mu and compute lambda_max(Jacobian A)
        for all 2^N binary equilibria.  Returns (energy levels, lmin, lmax,
        global_min_energy, raw lambda array).
        """
        oim    = OIMMaxCut(W, mu=mu)
        E, lam = scan_configs_jacobian(oim)   # uses full Jacobian — mu-dependent
        He, lmin, lmax = lambda_vs_energy(E, lam)
        return He, lmin, lmax, global_min_energy(E), lam

    He0, lmin0, lmax0, Eg0, lam0 = _compute_lam(mu_init)

    line_min, = ax_left.plot(He0, lmin0, "s-", color=BLUE, lw=1.8, ms=6,
                             label=r"min $\lambda_{\max}(A)$ per energy level")
    ref_zero,  = ax_left.plot([], [], color=BLACK, linestyle="--", lw=1.2)
    ax_left.axhline(0.0, color=BLACK, linestyle="--", lw=1.2,
                    label=r"$\lambda_{\max}(A) = 0$  (stability boundary)")

    ax_left.set_xlabel("OIM Lyapunov Energy E", fontsize=12, color=BLACK)
    ax_left.set_ylabel(r"$\lambda_{\max}(A(\phi^*))$  [Jacobian eigenvalue]",
                       fontsize=12, color=BLACK)
    ax_left.legend(fontsize=9, loc="upper right",
                   facecolor=WHITE, edgecolor=GRAY, labelcolor=BLACK)

    def _left_xlim(He):
        pad = max((He.max() - He.min()) * 0.1, 0.8)
        return He.min() - pad, He.max() + pad

    def _left_ylim(lmin_a, lmax_a):
        pad = max((lmax_a.max() - lmin_a.min()) * 0.15, 0.8)
        return lmin_a.min() - pad, lmax_a.max() + pad

    ax_left.set_xlim(*_left_xlim(He0))
    ax_left.set_ylim(*_left_ylim(lmin0, lmax0))
    ax_left.set_title(
        fr"Stability vs OIM Energy  ($\mu$={mu_init:.2f},  "
        fr"$K_s$={mu_init/2:.3f})"
        f"\n(N={N})  |  orange box = globally optimal energy level",
        color=BLACK,
    )

    def _eig_str(lam_all, mu):
        return (
            f"$\\lambda_{{\\max}}(A)$ min = {lam_all.min():.4f}\n"
            f"$\\lambda_{{\\max}}(A)$ max = {lam_all.max():.4f}\n"
            f"$\\mu_{{\\rm bin}}$ = {mu_bin:.4f}   "
            f"$K_s^*$ = {Ks_theory:.4f}\n"
            f"Easiest: {theory['easiest_eq']}\n"
            f"Hardest: {theory['hardest_eq']}"
        )

    eig_text = ax_left.text(
        0.02, 0.97, _eig_str(lam0, mu_init),
        transform=ax_left.transAxes,
        fontsize=8.0, va="top", ha="left",
        bbox=_tikz_bbox(), color=BLACK,
    )

    # ── RIGHT PANEL — Ising H vs K_s (static pre-computed) ────────────────
    y_lo = ((mean_H_arr[valid] - std_H_arr[valid]).min() - 0.3
            if valid.any() else H_ground - 2)
    y_hi = ((mean_H_arr[valid] + std_H_arr[valid]).max() + 0.3
            if valid.any() else H_ground + 2)

    if valid.any():
        ax_right.fill_between(
            ks_grid[valid],
            (mean_H_arr - std_H_arr)[valid],
            (mean_H_arr + std_H_arr)[valid],
            alpha=0.20, color=BLUE, label=r"$\pm$1 std",
        )
        ax_right.plot(ks_grid[valid], mean_H_arr[valid],
                      color=BLUE, lw=2.0, label="Mean H (binarised)")

    ax_right.axhline(H_ground, color=GRAY, linestyle=":", lw=1.5,
                     label=f"Ground state H = {H_ground:.1f}", zorder=3)
    ax_right.axvline(Ks_theory, color=BLACK, linestyle="-.", lw=2.0, zorder=6,
                     label=fr"$K_s^*$ theory = {Ks_theory:.4f}")
    if ks_star_empirical is not None:
        ax_right.axvline(ks_star_empirical, color=ORANGE, linestyle="--",
                         lw=1.6, zorder=5,
                         label=fr"Empirical $K_s^*$ = {ks_star_empirical:.4f}")

    ks_init   = mu_init / 2.0
    vline_cur = ax_right.axvline(ks_init, color=BLUE, lw=1.8,
                                  label=r"Current $K_s$", zorder=7)
    _mean_init = (float(np.interp(ks_init, ks_grid[valid], mean_H_arr[valid]))
                  if valid.any() else np.nan)
    dot_cur,  = ax_right.plot([ks_init], [_mean_init], "o",
                               color=BLUE, ms=8, zorder=8)

    annot = ax_right.annotate(
        "", xy=(0, 0), xytext=(18, 18), textcoords="offset points",
        bbox=_tikz_bbox(),
        arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.2),
        color=BLACK, fontsize=8.5, zorder=11,
    )
    annot.set_visible(False)

    ax_right.set_xlabel(r"$K_s$  (= $\mu / 2$)", fontsize=12, color=BLACK)
    ax_right.set_ylabel("Ising Hamiltonian H  (binarised final state)",
                        fontsize=11, color=BLACK)
    ax_right.set_title(
        f"Ising H vs $K_s$ — {N}-node graph\n"
        f"({n_trials_sweep} trials/pt,  "
        fr"$t_{{\rm end}}={t_end_sweep:.0f}$)"
        f"  |  orange = naturally binarised",
        color=BLACK,
    )
    ax_right.legend(fontsize=8.5, loc="lower right",
                    facecolor=WHITE, edgecolor=GRAY, labelcolor=BLACK)
    ax_right.set_xlim(ks_grid[0], ks_grid[-1])
    if valid.any():
        ax_right.set_ylim(y_lo, y_hi)

    _idx0  = int(np.argmin(np.abs(ks_grid - ks_init)))
    _is_bin = frac_bin_arr[_idx0] >= 0.5

    def _info_str(ks, mean_h, is_bin):
        return (
            f"$K_s$ = {ks:.4f}   ($\\mu$ = {2*ks:.4f})\n"
            f"Mean H = {mean_h:.3f}\n"
            f"($\\mu_{{\\rm bin}}$ = {mu_bin:.4f})\n"
            + ("BINARISED ✓" if is_bin else "NOT BINARISED ✗")
        )

    info_text = ax_right.text(
        0.03, 0.07, _info_str(ks_init, _mean_init, _is_bin),
        transform=ax_right.transAxes, fontsize=9,
        va="bottom", ha="left", bbox=_tikz_bbox(), color=BLACK,
    )

    # ── Slider ────────────────────────────────────────────────────────────
    slider = Slider(ax_sl, r"$\mu = 2K_s$", mu_min, mu_max,
                    valinit=mu_init, valstep=0.05,
                    color=ACCENT, track_color=LIGHT)
    slider.label.set_color(BLACK)
    slider.valtext.set_color(BLACK)

    # ── Update callback — LEFT panel is fully live ─────────────────────────
    def update(val: float) -> None:
        mu_cur = slider.val
        ks_cur = mu_cur / 2.0

        # ── Recompute left panel using Jacobian eigenvalues ────────────────
        He, lmin, lmax, Eg, lam_all = _compute_lam(mu_cur)

        # Update line data
        line_min.set_xdata(He)
        line_min.set_ydata(lmin)

        # Rescale axes properly
        ax_left.relim()
        ax_left.set_xlim(*_left_xlim(He))
        ax_left.set_ylim(*_left_ylim(lmin, lmax))

        # Update title and annotation
        ax_left.set_title(
            fr"Stability vs OIM Energy  ($\mu$={mu_cur:.2f},  "
            fr"$K_s$={ks_cur:.3f})"
            f"\n(N={N})  |  orange box = globally optimal energy level",
            color=BLACK,
        )
        eig_text.set_text(_eig_str(lam_all, mu_cur))

        # ── Update right panel cursor ──────────────────────────────────────
        vline_cur.set_xdata([ks_cur, ks_cur])
        mean_h = (float(np.interp(ks_cur, ks_grid[valid], mean_H_arr[valid]))
                  if valid.any() else np.nan)
        dot_cur.set_xdata([ks_cur])
        dot_cur.set_ydata([mean_h])
        idx_near = int(np.argmin(np.abs(ks_grid - ks_cur)))
        info_text.set_text(_info_str(ks_cur, mean_h,
                                     frac_bin_arr[idx_near] >= 0.5))
        fig.canvas.draw_idle()

    slider.on_changed(update)

    fig.suptitle(
        f"OIM Stability Explorer — N={N}   |   mu parametrisation   "
        f"|   Drag slider to explore",
        fontsize=12, fontweight="bold", color=BLACK, y=0.99,
    )
    print("[Exp C] Drag the mu slider to explore stability. Close when done.")
    plt.show()


# ===========================================================================
# CLI
# ===========================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="OIM Stability Experiments  (OIMMaxCut / mu parametrisation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--graph",       "-g", required=True, metavar="FILE",
                        help="Path to graph file")
    parser.add_argument("--experiments", "-e", type=str,  default="ABC",
                        help="Which experiments to run: A, B, C, AB, ...")
    parser.add_argument("--seed",              type=int,  default=None,
                        help="Global random seed")

    sweep = parser.add_argument_group("mu sweep  [all experiments]")
    sweep.add_argument("--mu_init",   type=float, default=1.6,
                       help="Initial mu for sliders")
    sweep.add_argument("--mu_min",    type=float, default=0.1,
                       help="Lower bound of mu sweep")
    sweep.add_argument("--mu_max",    type=float, default=3.0,
                       help="Upper bound of mu sweep")
    sweep.add_argument("--n_mu",  type=int,   default=30,
                       help="Number of mu steps (Exp A scatter + Exp B slider)")

    expB = parser.add_argument_group("Experiment B  [interactive histogram]")
    expB.add_argument("--trials",   type=int,   default=50,
                      help="Number of random trials per mu step")
    expB.add_argument("--t-end",    type=float, default=50.0,
                      help="Integration end time")
    expB.add_argument("--n-points", type=int,   default=500,
                      help="Time points saved per trajectory")

    expC = parser.add_argument_group("Experiment C  [interactive two-panel]")
    expC.add_argument("--n-ks-steps", type=int,   default=40,
                      help="K_s grid points for right-panel empirical sweep")
    expC.add_argument("--c-trials",   type=int,   default=30,
                      help="Trials per K_s grid point")
    expC.add_argument("--c-t-end",    type=float, default=50.0,
                      help="Integration end time for Exp C")
    expC.add_argument("--c-n-points", type=int,   default=500,
                      help="Time points per trajectory in Exp C")

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    exps = args.experiments.upper()
    run_A = "A" in exps
    run_B = "B" in exps
    run_C = "C" in exps

    if not (run_A or run_B or run_C):
        print("No experiments selected.")
        sys.exit(1)

    print(f"Loading graph: {args.graph}")
    try:
        W = _load_graph(args.graph)
    except (FileNotFoundError, ImportError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    info = _graph_info(W)
    print(f"  N={info['N']} nodes, {info['edges']} edges, "
          f"density={info['density']:.3f}")

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"  Random seed: {args.seed}")

    mu_values = np.linspace(args.mu_min, args.mu_max, args.n_mu)

    if run_A:
        print("\n" + "=" * 60 + "\nEXPERIMENT A\n" + "=" * 60)
        experiment_A(W, mu_values)

    if run_B:
        print("\n" + "=" * 60 + "\nEXPERIMENT B\n" + "=" * 60)
        experiment_B(
            W, args.mu_min, args.mu_max, args.n_mu,
            args.trials, (0.0, args.t_end), args.n_points,
        )

    if run_C:
        print("\n" + "=" * 60 + "\nEXPERIMENT C\n" + "=" * 60)
        experiment_C(
            W, args.mu_init, args.mu_min, args.mu_max,
            n_ks_steps=args.n_ks_steps,
            n_trials_sweep=args.c_trials,
            t_end_sweep=args.c_t_end,
            n_points_sweep=args.c_n_points,
        )

    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()
