"""
run_experiment.py
-----------------
OIM Stability Experiments — Experiments A, B, C.
Refactored to use OIMMaxCut (OIM_mu_v2) exclusively.
TikZ / PGFPlots visual style throughout.

Experiments
-----------
  A : lambda_max(D) vs mu — scatter of per-equilibrium stability threshold
      for all 2^N binary configurations, across a sweep of mu values.
      Orange = globally optimal (lowest H) configurations.
      Blue   = sub-optimal configurations.
      Horizontal dashed line at 0 marks the stability boundary.

  B : Ising Hamiltonian histograms — for one or more fixed mu values,
      run n_trials random initial conditions, hard-binarise the final
      phase vector, and histogram the resulting H values.
      Shows how solution quality distributes across random restarts.

  C : Interactive two-panel figure (mu slider).
      LEFT  : lambda_max(D) min/max band vs OIM Lyapunov energy for all
              binary equilibria at the current mu. Orange dashed box
              highlights the globally optimal energy level.
      RIGHT : Mean Ising H (hard-binarised) +/- 1 std vs K_s,
              pre-computed over a K_s grid. Vertical markers show the
              theoretical K_s* (Remark 7) and empirical binarisation
              onset. Orange dots are clickable for per-point details.
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from typing import List, Optional, Tuple

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

# TikZ colour palette
WHITE  = "#ffffff"
BLACK  = "#000000"
GRAY   = "#b0b0b0"
LIGHT  = "#e6e6e6"
BLUE   = "#4C72B0"   # stable / optimal
ORANGE = "#DD8452"   # unstable / sub-optimal
ACCENT = "#1f77b4"

# ── Import OIMMaxCut ────────────────────────────────────────────────────────────
try:
    from OIM_mu_v2 import OIMMaxCut
except ModuleNotFoundError:
    from OIM_Experiment.src.OIM_mu_v2 import OIMMaxCut

# ── Graph reading ───────────────────────────────────────────────────────────────
# OIMMaxCut expects W_ij >= 0 (positive weight matrix, NOT negated).
# We try read_graph first; if unavailable we negate J from read_graph_to_J.
def _load_graph(path: str) -> np.ndarray:
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
        J = read_graph(path)
        return -np.array(J, dtype=float)   # W = -J  (Ising sign convention)
    except (ModuleNotFoundError, ImportError):
        raise ImportError(
            "Cannot find a graph reader.  "
            "Ensure graph_utils.py or read_graphs.py is on the Python path."
        )


def _graph_info(W: np.ndarray) -> dict:
    N       = W.shape[0]
    edges   = int(np.sum(W > 0)) // 2
    density = 2 * edges / max(N * (N - 1), 1)
    return {"N": N, "edges": edges, "density": density}


# ===========================================================================
# Shared helpers (all use OIMMaxCut API)
# ===========================================================================

def all_phase_configs(N: int) -> List[np.ndarray]:
    """All 2^N binary phase configurations in {0, pi}^N."""
    return [np.array(cfg, dtype=float) * np.pi
            for cfg in product([0, 1], repeat=N)]


def scan_configs(oim: OIMMaxCut) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (OIM Lyapunov energy, stability threshold mu*_i) for every
    2^N binary equilibrium.

    Old API   : oim.energy(phi),           oim.largest_lyapunov(phi)
    New API   : oim.get_energy()  [reads self.theta],
                oim.stability_threshold(phi) = lambda_max(D(phi*))
    """
    configs  = all_phase_configs(oim.n)
    energies, lambdas = [], []
    for phi in configs:
        oim.theta = phi.copy()
        energies.append(oim.get_energy())
        lambdas.append(oim.stability_threshold(phi))
    return np.array(energies), np.array(lambdas)


def ground_ising_energy(oim: OIMMaxCut) -> float:
    """
    Minimum Ising Hamiltonian H over all 2^N binary configs.
    mu-independent — reflects the true optimal cut.

    Old API : oim.ising_hamiltonian(phi)
    New API : oim.get_hamiltonian(theta=phi)
    """
    configs = all_phase_configs(oim.n)
    return float(min(oim.get_hamiltonian(theta=phi) for phi in configs))


def lambda_vs_energy(
    E: np.ndarray, lam: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group equilibria by energy level; return min/max lambda per level."""
    unique_E         = np.array(sorted(set(np.round(E, 8))))
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


def run_trials(
    oim:      OIMMaxCut,
    n_trials: int                 = 50,
    t_span:   Tuple[float, float] = (0.0, 50.0),
    n_points: int                 = 500,
) -> np.ndarray:
    """
    Run n_trials random simulations.  Returns the Ising H of the
    hard-binarised final phase state for each trial.

    Old API : oim.ising_hamiltonian(sol.y[:, -1])
    New API : oim.get_hamiltonian(theta=sol.y[:, -1])
              (get_hamiltonian internally calls get_spins() for binarisation)
    """
    H_finals = []
    for _ in range(n_trials):
        phi0 = random_phi0(oim.n)
        sol  = oim.simulate(phi0, t_span, n_points)
        if sol is None:
            continue
        H_finals.append(oim.get_hamiltonian(theta=sol.y[:, -1]))
    return np.array(H_finals)


# ── Axis styling helper ────────────────────────────────────────────────────────
def _style_ax(ax, grid_axis: str = "both") -> None:
    """Apply TikZ spine / tick / grid style to an Axes object."""
    ax.set_facecolor(WHITE)
    ax.tick_params(colors=BLACK, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(BLACK)
        sp.set_linewidth(0.8)
    ax.grid(True, color=LIGHT, linewidth=0.6, axis=grid_axis)


def _tikz_bbox() -> dict:
    """Standard white annotation box with gray border (TikZ style)."""
    return dict(boxstyle="round,pad=0.4", facecolor=WHITE,
                edgecolor=GRAY, alpha=1.0)


# ===========================================================================
# Experiment A — stability threshold lambda_max(D) vs mu
# ===========================================================================

def experiment_A(W: np.ndarray, mu_values: np.ndarray,
                 *, warn_large_N: int = 16) -> None:
    """
    For each mu in mu_values, instantiate OIMMaxCut(W, mu) and compute
    the per-equilibrium stability threshold
        mu*_i = lambda_max( D(phi*_i) )
    for all 2^N binary equilibria.

    Plot (scatter):
      X-axis : mu
      Y-axis : mu*_i  (= lambda_max of the signed Laplacian D at that eq.)
      Orange : globally optimal equilibria (lowest Lyapunov energy = max cut)
      Blue   : sub-optimal equilibria
      Dashed : mu*_i = 0 boundary (above = stable, below = unstable)

    Theory connection — Theorem 2:
      phi*_i is asymptotically stable  <=>  mu > mu*_i
    The binarisation threshold mu_bin (Remark 7) is the smallest mu*_i
    over all equilibria, i.e. the mu at which the easiest equilibrium
    first becomes stable.
    """
    N = W.shape[0]
    if N > warn_large_N:
        print(f"[Exp A] WARNING: N={N} -> 2^N={2**N} configs. May be slow.")
    print(f"[Exp A] Scanning {len(mu_values)} mu values x {2**N} configs ...")

    all_results = []
    for i, mu in enumerate(mu_values):
        oim = OIMMaxCut(W, mu=mu)
        E, lam = scan_configs(oim)
        all_results.append((mu, E, lam))
        if (i + 1) % 5 == 0 or (i + 1) == len(mu_values):
            print(f"  mu step {i+1}/{len(mu_values)} done", end="\r", flush=True)
    print()

    fig, ax = plt.subplots(figsize=(9, 5),
                           num="Experiment A: lambda_max(D) vs mu")
    fig.patch.set_facecolor(WHITE)

    for mu, E, lam in all_results:
        E0        = global_min_energy(E)
        is_global = np.isclose(E, E0)
        ax.scatter(np.full(np.sum(~is_global), mu), lam[~is_global],
                   s=6, alpha=0.30, color=BLUE, linewidths=0)
        if is_global.any():
            ax.scatter(np.full(np.sum(is_global), mu), lam[is_global],
                       s=22, alpha=0.88, color=ORANGE, zorder=3, linewidths=0)

    ax.axhline(0.0, color=BLACK, linestyle="--", linewidth=1.2)
    _style_ax(ax)
    ax.set_xlabel(r"Injection strength  $\mu = 2K_s$", fontsize=12, color=BLACK)
    ax.set_ylabel(
        r"Stability threshold  $\mu^*_i = \lambda_{\max}(D(\phi^*_i))$",
        fontsize=12, color=BLACK)
    ax.set_title(
        r"Exp A — Per-equilibrium stability threshold $\lambda_{\max}(D)$ vs $\mu$"
        f"\n(N={N},  K=1 fixed in OIMMaxCut)",
        color=BLACK,
    )
    ax.legend(handles=[
        mpatches.Patch(color=BLUE,   label="Sub-optimal equilibria"),
        mpatches.Patch(color=ORANGE, label="Globally optimal equilibria"),
        mlines.Line2D([0], [0], color=BLACK, linestyle="--",
                      label=r"$\mu^*_i = 0$  (stability boundary)"),
    ], fontsize=9, facecolor=WHITE, edgecolor=GRAY, labelcolor=BLACK)
    fig.tight_layout()
    print("[Exp A] Close the window to proceed.")
    plt.show()


# ===========================================================================
# Experiment B — Ising Hamiltonian histograms
# ===========================================================================

def experiment_B(W: np.ndarray, mu_values: list, n_trials: int,
                 t_span: Tuple[float, float], n_points: int) -> None:
    """
    For each mu in mu_values, run n_trials simulations from random phi0
    using OIMMaxCut(W, mu).  The final phase is hard-binarised via
    get_spins() inside get_hamiltonian(); the resulting Ising H is
    recorded.

    Histogram per mu panel:
      Blue bars  : H distribution across trials
      Orange --  : ground-state H (minimum H over all 2^N binary configs,
                   mu-independent; lower H <=> higher cut value)

    As mu increases past mu_bin, trajectories should cluster near the
    ground state (= maximum cut).

    H = W_total - 2 * cut_value, so:
      lower (more negative) H  <=>  better cut.
    """
    N = W.shape[0]
    n = len(mu_values)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4),
                             num="Experiment B: Ising Hamiltonian Histograms")
    fig.patch.set_facecolor(WHITE)
    if n == 1:
        axes = [axes]

    oim_ref  = OIMMaxCut(W, mu=1.0)      # mu irrelevant for H_ground
    H_ground = ground_ising_energy(oim_ref)

    for ax, mu in zip(axes, mu_values):
        Ks  = mu / 2.0
        oim = OIMMaxCut(W, mu=mu)
        print(f"[Exp B] mu={mu:.2f}  (Ks={Ks:.3f}): running {n_trials} trials ...")
        H_finals = run_trials(oim, n_trials=n_trials,
                              t_span=t_span, n_points=n_points)

        if len(H_finals) == 0:
            ax.text(0.5, 0.5, "No trials converged",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color=BLACK)
            ax.set_title(f"mu={mu}  (Ks={Ks:.3f})", color=BLACK)
            _style_ax(ax)
            continue

        bins = np.arange(H_finals.min() - 0.5, H_finals.max() + 1.5, 1.0)
        ax.hist(H_finals, bins=bins, color=BLUE,
                edgecolor=BLACK, linewidth=0.6, alpha=0.85, zorder=2)
        ax.axvline(H_ground, color=ORANGE, linestyle="--", linewidth=1.8,
                   label=f"Ground state H = {H_ground:.1f}", zorder=3)
        _style_ax(ax)
        ax.set_xlabel("Ising Hamiltonian H  (hard-binarised final state)",
                      fontsize=11, color=BLACK)
        ax.set_ylabel("Count", fontsize=11, color=BLACK)
        ax.set_title(fr"$\mu = {mu}$  ($K_s = {Ks:.3f}$)",
                     fontsize=12, color=BLACK)
        ax.legend(fontsize=8, facecolor=WHITE, edgecolor=GRAY,
                  labelcolor=BLACK)

    fig.suptitle(
        f"Exp B — Ising H distribution over {n_trials} trials per mu\n"
        f"(N={N},  H = hard binarisation of final phase state)",
        fontsize=12, color=BLACK,
    )
    fig.tight_layout()
    print("[Exp B] Close the window to proceed.")
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

    LEFT panel — lambda_max(D) min/max band vs OIM Lyapunov energy.
      For every unique Lyapunov energy level among the 2^N binary
      equilibria, plots the minimum and maximum stability threshold
      mu*_i = lambda_max(D(phi*_i)) at that level.
      Orange box highlights the globally optimal energy level.
      Annotation text reports eigenvalue extremes and the theoretical
      binarisation threshold mu_bin (Remark 7).
      Updates live as the slider moves.

    RIGHT panel — Mean Ising H +/- 1 std vs K_s.
      Pre-computed (static) empirical sweep over a K_s grid.
      The slider cursor (blue vertical line) indicates the current K_s;
      the info box at bottom-left updates accordingly.
      Markers:
        Black -.-  : K_s* = mu_bin / 2  (Remark 7 theory)
        Orange --  : empirical binarisation onset (first K_s with
                     >= 50% of trials binarised)
        Blue  |    : current slider K_s
        Orange dots: K_s grid points where >= 50% of trials binarised
                     (click for a tooltip with per-point details)
    """
    N = W.shape[0]
    if N > warn_large_N:
        print(f"[Exp C] WARNING: N={N} -> 2^N={2**N} configs per slider move.")

    # ── Theory: D-matrix binarisation threshold (Remark 7) ───────────────
    print(f"[Exp C] Computing D-matrix thresholds for {2**N} equilibria ...")
    oim_theory = OIMMaxCut(W, mu=1.0)
    theory     = oim_theory.binarization_threshold()

    # OIMMaxCut API keys: mu_bin, Ks_bin, all_thresholds, easiest_eq, hardest_eq
    mu_bin    = theory["mu_bin"]
    Ks_theory = theory["Ks_bin"]           # = mu_bin / 2
    per_eq    = theory["all_thresholds"]   # {spin_pattern_str: mu*_i}
    print(f"  mu_bin (Remark 7) = {mu_bin:.4f}   K_s* = {Ks_theory:.4f}")
    print(f"  Easiest eq : {theory['easiest_eq']}   "
          f"thr = {min(per_eq.values()):.4f}")
    print(f"  Hardest eq : {theory['hardest_eq']}   "
          f"thr = {max(per_eq.values()):.4f}")

    # Ground-state Ising H (mu-independent)
    H_ground = ground_ising_energy(oim_theory)
    print(f"  Ground state Ising H = {H_ground:.4f}")

    # ── Pre-compute empirical H mean / std over K_s grid ─────────────────
    ks_grid      = np.linspace(mu_min / 2.0, mu_max / 2.0, n_ks_steps)
    mean_H_arr   = np.full(n_ks_steps, np.nan)
    std_H_arr    = np.full(n_ks_steps, np.nan)
    frac_bin_arr = np.zeros(n_ks_steps)

    print(f"[Exp C] Pre-computing {n_ks_steps} pts x {n_trials_sweep} trials ...")
    for idx, Ks in enumerate(ks_grid):
        mu_cur   = 2.0 * Ks
        oim      = OIMMaxCut(W, mu=mu_cur)
        H_trials = []
        n_bin    = 0
        for _ in range(n_trials_sweep):
            phi0  = random_phi0(N)
            sol   = oim.simulate(phi0, (0.0, t_end_sweep), n_points_sweep)
            if sol is None:
                continue
            phi_f = sol.y[:, -1]
            # get_hamiltonian binarises internally via get_spins()
            H_trials.append(oim.get_hamiltonian(theta=phi_f))
            # Use OIMMaxCut's built-in binarisation check
            oim.theta = phi_f
            if oim.is_binarized(tol=0.1):
                n_bin += 1
        if H_trials:
            mean_H_arr[idx]   = float(np.mean(H_trials))
            std_H_arr[idx]    = float(np.std(H_trials))
            frac_bin_arr[idx] = n_bin / len(H_trials)
        if (idx + 1) % 10 == 0 or (idx + 1) == n_ks_steps:
            print(f"  Ks step {idx+1}/{n_ks_steps}", end="\r", flush=True)
    print()

    bin_mask          = frac_bin_arr >= 0.5
    ks_star_empirical = float(ks_grid[bin_mask][0]) if bin_mask.any() else None
    valid             = ~np.isnan(mean_H_arr)

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 7),
                     num="Experiment C: Interactive Stability",
                     facecolor=WHITE)
    ax_left  = fig.add_axes((0.06, 0.20, 0.41, 0.70))
    ax_right = fig.add_axes((0.55, 0.20, 0.42, 0.70))
    ax_sl    = fig.add_axes((0.15, 0.06, 0.70, 0.03))

    _style_ax(ax_left)
    _style_ax(ax_right)
    ax_sl.set_facecolor(WHITE)
    for sp in ax_sl.spines.values():
        sp.set_edgecolor(GRAY)

    # ── LEFT PANEL — lambda_max(D) min/max vs OIM Lyapunov energy ─────────
    def _compute_lam(mu: float):
        oim    = OIMMaxCut(W, mu=mu)
        E, lam = scan_configs(oim)
        He, lmin, lmax = lambda_vs_energy(E, lam)
        return He, lmin, lmax, global_min_energy(E), lam

    He0, lmin0, lmax0, Eg0, lam0 = _compute_lam(mu_init)

    line_max, = ax_left.plot(He0, lmax0, "o-", color=ORANGE,
                             lw=1.8, ms=6,
                             label=r"max $\lambda_{\max}(D)$ per energy level")
    line_min, = ax_left.plot(He0, lmin0, "s-", color=BLUE,
                             lw=1.8, ms=6,
                             label=r"min $\lambda_{\max}(D)$ per energy level")
    ax_left.axhline(0.0, color=BLACK, linestyle="--", lw=1.2,
                    label=r"$\lambda_{\max}(D) = 0$  (stability boundary)")

    def _draw_ground_box(ax, Eg, lmin_a, lmax_a, He_a):
        for p in list(ax.patches):
            p.remove()
        mask = np.isclose(He_a, Eg)
        if not mask.any():
            return
        ax.add_patch(mpatches.FancyBboxPatch(
            (Eg - 0.6, lmin_a[mask].min() - 0.3),
            1.2, (lmax_a[mask].max() - lmin_a[mask].min()) + 0.6,
            boxstyle="round,pad=0.1", linewidth=1.8,
            edgecolor=ORANGE, facecolor="none", linestyle="--", zorder=5,
        ))

    _draw_ground_box(ax_left, Eg0, lmin0, lmax0, He0)
    ax_left.set_xlabel("OIM Lyapunov Energy E", fontsize=12, color=BLACK)
    ax_left.set_ylabel(r"$\lambda_{\max}(D(\phi^*))$", fontsize=12, color=BLACK)
    ax_left.legend(fontsize=9, loc="upper right",
                   facecolor=WHITE, edgecolor=GRAY, labelcolor=BLACK)
    ax_left.set_xlim(He0.min() - 1, He0.max() + 1)
    ax_left.set_ylim(lmin0.min() - 0.8, lmax0.max() + 0.8)
    ax_left.set_title(
        fr"Stability vs OIM Energy  ($\mu$={mu_init:.2f},  $K_s$={mu_init/2:.3f})"
        f"\n(N={N})  |  orange box = globally optimal energy level",
        color=BLACK,
    )

    def _eig_str(lam_all):
        return (
            f"$\\lambda_{{\\max}}$ min = {lam_all.min():.4f}\n"
            f"$\\lambda_{{\\max}}$ max = {lam_all.max():.4f}\n"
            f"— D-matrix theory (Remark 7) —\n"
            f"$\\mu_{{\\rm bin}}$ = {mu_bin:.4f}   "
            f"$K_s^*$ = {Ks_theory:.4f}\n"
            f"Easiest: {theory['easiest_eq']}\n"
            f"Hardest: {theory['hardest_eq']}"
        )

    eig_text = ax_left.text(
        0.02, 0.97, _eig_str(lam0),
        transform=ax_left.transAxes,
        fontsize=8.5, va="top", ha="left",
        bbox=_tikz_bbox(),
        color=BLACK,
    )

    # ── RIGHT PANEL — Ising H vs K_s ──────────────────────────────────────
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

    # Theory K_s*
    ax_right.axvline(Ks_theory, color=BLACK, linestyle="-.", lw=2.0, zorder=6,
                     label=fr"$K_s^*$ theory (Rem. 7) = {Ks_theory:.4f}")

    # Empirical onset
    if ks_star_empirical is not None:
        ax_right.axvline(ks_star_empirical, color=ORANGE, linestyle="--",
                         lw=1.6, zorder=5,
                         label=fr"Empirical $K_s^*$ = {ks_star_empirical:.4f}")

    # Current K_s: live cursor
    ks_init   = mu_init / 2.0
    vline_cur = ax_right.axvline(ks_init, color=BLUE, lw=1.8,
                                  label=r"Current $K_s$", zorder=7)
    _mean_init = (float(np.interp(ks_init, ks_grid[valid], mean_H_arr[valid]))
                  if valid.any() else np.nan)
    dot_cur,  = ax_right.plot([ks_init], [_mean_init], "o",
                               color=BLUE, ms=8, zorder=8)

    # Binarised dots — clickable
    bin_valid = bin_mask & valid
    bin_xs    = ks_grid[bin_valid]
    bin_ys    = mean_H_arr[bin_valid]
    bin_idxs  = np.where(bin_valid)[0]

    bin_dots = ax_right.scatter(
        bin_xs, bin_ys,
        color=ORANGE, s=70, zorder=9, picker=6,
        edgecolors=WHITE, linewidths=0.8,
        label="Naturally binarised (click)",
    )

    annot = ax_right.annotate(
        "", xy=(0, 0), xytext=(18, 18),
        textcoords="offset points",
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
        f"  |  orange dots = naturally binarised",
        color=BLACK,
    )
    ax_right.legend(fontsize=8.5, loc="lower right",
                    facecolor=WHITE, edgecolor=GRAY, labelcolor=BLACK)
    ax_right.set_xlim(ks_grid[0], ks_grid[-1])
    if valid.any():
        ax_right.set_ylim(y_lo, y_hi)

    _idx0   = int(np.argmin(np.abs(ks_grid - ks_init)))
    _is_bin = frac_bin_arr[_idx0] >= 0.5

    def _info_str(ks, mean_h, is_bin):
        return (
            f"$K_s$ = {ks:.4f}   ($\\mu$ = {2*ks:.4f})\n"
            f"Mean H (binarised) = {mean_h:.3f}\n"
            f"$K_s^*$ theory = {Ks_theory:.4f}   "
            f"($\\mu_{{\\rm bin}}$ = {mu_bin:.4f})\n"
            + ("BINARISED \u2713" if is_bin else "NOT BINARISED \u2717")
        )

    info_text = ax_right.text(
        0.03, 0.07, _info_str(ks_init, _mean_init, _is_bin),
        transform=ax_right.transAxes,
        fontsize=9, va="bottom", ha="left",
        bbox=_tikz_bbox(),
        color=BLACK,
    )

    # ── Slider ────────────────────────────────────────────────────────────
    slider = Slider(ax_sl, r"$\mu = 2K_s$", mu_min, mu_max,
                    valinit=mu_init, valstep=0.05,
                    color=ACCENT, track_color=LIGHT)
    slider.label.set_color(BLACK)
    slider.valtext.set_color(BLACK)

    # ── Update callback ───────────────────────────────────────────────────
    def update(val: float) -> None:
        mu_cur = slider.val
        ks_cur = mu_cur / 2.0

        He, lmin, lmax, Eg, lam_all = _compute_lam(mu_cur)
        line_max.set_xdata(He); line_max.set_ydata(lmax)
        line_min.set_xdata(He); line_min.set_ydata(lmin)
        ax_left.set_xlim(He.min() - 1, He.max() + 1)
        ax_left.set_ylim(lmin.min() - 0.8, lmax.max() + 0.8)
        ax_left.set_title(
            fr"Stability vs OIM Energy  ($\mu$={mu_cur:.2f},  "
            fr"$K_s$={ks_cur:.3f})"
            f"\n(N={N})  |  orange box = globally optimal energy level",
            color=BLACK,
        )
        _draw_ground_box(ax_left, Eg, lmin, lmax, He)
        eig_text.set_text(_eig_str(lam_all))

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

    # ── Click handler — tooltip on binarised dots ─────────────────────────
    def on_pick(event):
        if event.artist is not bin_dots:
            annot.set_visible(False)
            fig.canvas.draw_idle()
            return
        i         = event.ind[0]
        grid_idx  = bin_idxs[i]
        ks_val    = float(ks_grid[grid_idx])
        h_val     = float(mean_H_arr[grid_idx])
        frac      = float(frac_bin_arr[grid_idx])
        vs_theory = "above" if ks_val >= Ks_theory else "below"
        annot.xy  = (ks_val, h_val)
        annot.set_text(
            f"Ks  = {ks_val:.4f}   (mu = {2*ks_val:.4f})\n"
            f"Mean H (binarised)  = {h_val:.3f}\n"
            f"Ground state H      = {H_ground:.3f}\n"
            f"Binarised: {frac*100:.0f}% of trials\n"
            f"{vs_theory} K_s* theory ({Ks_theory:.4f})"
        )
        annot.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("button_press_event", lambda e: (
        annot.set_visible(False), fig.canvas.draw_idle()
    ) if e.inaxes is ax_right else None)

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
                        help="Path to graph file (edge list or adjacency matrix)")
    parser.add_argument("--experiments", "-e", type=str,  default="ABC",
                        help="Which experiments to run: A, B, C, AB, ABC, ...")
    parser.add_argument("--seed",              type=int,  default=None,
                        help="Global random seed")

    sweep = parser.add_argument_group("mu sweep  [Exp A and C]")
    sweep.add_argument("--mu-init",   type=float, default=1.6, metavar="MU",
                       help="Initial mu for the slider / Exp A reference")
    sweep.add_argument("--mu-min",    type=float, default=0.1,
                       help="Lower bound of mu sweep")
    sweep.add_argument("--mu-max",    type=float, default=3.0,
                       help="Upper bound of mu sweep")
    sweep.add_argument("--mu-steps",  type=int,   default=30,
                       help="Number of mu values in Exp A scatter")

    expB = parser.add_argument_group("Experiment B  [Ising H histograms]")
    expB.add_argument("--mu-hist",  nargs="+", type=float, default=[1.6, 3.0],
                      metavar="MU",
                      help="One or more mu values for the H histograms")
    expB.add_argument("--trials",   type=int,   default=50,
                      help="Number of random trials per histogram")
    expB.add_argument("--t-end",    type=float, default=50.0,
                      help="Integration end time")
    expB.add_argument("--n-points", type=int,   default=500,
                      help="Time points saved per trajectory")

    expC = parser.add_argument_group("Experiment C  [interactive slider]")
    expC.add_argument("--n-ks-steps", type=int,   default=40,
                      help="Number of K_s grid points in empirical sweep")
    expC.add_argument("--c-trials",   type=int,   default=30,
                      help="Random trials per K_s grid point")
    expC.add_argument("--c-t-end",    type=float, default=50.0,
                      help="Integration end time for Exp C trials")
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

    mu_values = np.linspace(args.mu_min, args.mu_max, args.mu_steps)

    if run_A:
        print("\n" + "=" * 60 + "\nEXPERIMENT A\n" + "=" * 60)
        experiment_A(W, mu_values)

    if run_B:
        print("\n" + "=" * 60 + "\nEXPERIMENT B\n" + "=" * 60)
        experiment_B(W, args.mu_hist, args.trials,
                     (0.0, args.t_end), args.n_points)

    if run_C:
        print("\n" + "=" * 60 + "\nEXPERIMENT C\n" + "=" * 60)
        experiment_C(W, args.mu_init, args.mu_min, args.mu_max,
                     n_ks_steps=args.n_ks_steps,
                     n_trials_sweep=args.c_trials,
                     t_end_sweep=args.c_t_end,
                     n_points_sweep=args.c_n_points)

    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()
