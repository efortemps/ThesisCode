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

try:
    from OIM_Stability_1 import OIM_Maxcut
    from read_graphs import read_graph_to_J, graph_info
except ModuleNotFoundError:
    from Stability_OIM_2.OIM_Stability_1 import OIM_Maxcut
    from Stability_OIM_2.read_graphs import read_graph_to_J, graph_info


# ===========================================================================
# Helpers
# ===========================================================================

def all_phase_configs(N: int) -> List[np.ndarray]:
    """All 2^N binary phase configurations {0, pi}^N."""
    return [np.array(cfg, dtype=float) * np.pi
            for cfg in product([0, 1], repeat=N)]


def scan_configs(oim: OIM_Maxcut) -> Tuple[np.ndarray, np.ndarray]:
    """Return (OIM_energies, lambda_L) for all 2^N binary configs."""
    configs  = all_phase_configs(oim.N)
    energies = np.array([oim.energy(phi)           for phi in configs])
    lambdas  = np.array([oim.largest_lyapunov(phi) for phi in configs])
    return energies, lambdas


def ground_ising_energy(oim: OIM_Maxcut) -> float:
    """
    Minimum Ising Hamiltonian over all 2^N binary configurations.
    Independent of Ks — reflects the true ground state cut quality.
    """
    configs = all_phase_configs(oim.N)
    return float(min(oim.ising_hamiltonian(phi) for phi in configs))


def lambda_vs_energy(
    E: np.ndarray, lam: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def binarize(phi: np.ndarray) -> np.ndarray:
    return np.where(phi % (2.0 * np.pi) < np.pi, 0.0, np.pi)


def run_trials(
    oim:      OIM_Maxcut,
    n_trials: int                 = 50,
    t_span:   Tuple[float, float] = (0.0, 50.0),
    n_points: int                 = 500,
) -> np.ndarray:
    """
    Run n_trials random simulations and return the Ising Hamiltonian
    of the rounded final state for each trial.
    Uses ising_hamiltonian() so values are independent of Ks.
    """
    H_finals = []
    for _ in range(n_trials):
        phi0 = random_phi0(oim.N)
        sol  = oim.simulate(phi0, t_span, n_points)
        if sol is None:
            continue
        H_finals.append(oim.ising_hamiltonian(sol.y[:, -1]))
    return np.array(H_finals)


# ===========================================================================
# Experiment A — lambda_L vs mu
# ===========================================================================

def experiment_A(J, K, mu_values, *, warn_large_N=16):
    N = J.shape[0]
    if N > warn_large_N:
        print(f"[Exp A] WARNING: N={N} -> 2^N={2**N} configs. May be slow.")
    print(f"[Exp A] Scanning {len(mu_values)} mu values x {2**N} configs ...")

    all_results = []
    for i, mu in enumerate(mu_values):
        oim = OIM_Maxcut(J, K=K, Ks=mu / 2.0)
        E, lam = scan_configs(oim)
        all_results.append((mu, E, lam))
        if (i + 1) % 5 == 0 or (i + 1) == len(mu_values):
            print(f"  mu step {i+1}/{len(mu_values)} done", end="\r", flush=True)
    print()

    fig, ax = plt.subplots(figsize=(9, 5), num="Experiment A: lambda_L vs mu")
    for mu, E, lam in all_results:
        E0        = global_min_energy(E)
        is_global = np.isclose(E, E0)
        ax.scatter(np.full(np.sum(~is_global), mu), lam[~is_global],
                   s=6, alpha=0.25, color="steelblue", linewidths=0)
        if is_global.any():
            ax.scatter(np.full(np.sum(is_global), mu), lam[is_global],
                       s=20, alpha=0.85, color="darkorange", zorder=3, linewidths=0)

    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"Injection strength $\mu = 2K_s$", fontsize=12)
    ax.set_ylabel(r"Largest Lyapunov exponent $\lambda_L$", fontsize=12)
    ax.set_title(
        r"Exp A -- Evolution of $\lambda_L$ for all spin configurations"
        f"\n(N={N}, K={K})"
    )
    ax.legend(handles=[
        mpatches.Patch(color="steelblue",  label="Sub-optimal configurations"),
        mpatches.Patch(color="darkorange", label="Globally optimal configurations"),
        mlines.Line2D([0], [0], color="k", linestyle="--",
                      label=r"$\lambda_L = 0$"),
    ], fontsize=9)
    fig.tight_layout()
    print("[Exp A] Close the window to proceed.")
    plt.show()


# ===========================================================================
# Experiment B — Ising Hamiltonian histograms
# ===========================================================================

def experiment_B(J, K, mu_values, n_trials, t_span, n_points):
    N = J.shape[0]
    n = len(mu_values)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4),
                             num="Experiment B: Ising Hamiltonian Histograms")
    if n == 1:
        axes = [axes]

    # Ground state is Ks-independent — compute once
    oim_ref  = OIM_Maxcut(J, K=K, Ks=0.0)
    H_ground = ground_ising_energy(oim_ref)

    for ax, mu in zip(axes, mu_values):
        Ks  = mu / 2.0
        oim = OIM_Maxcut(J, K=K, Ks=Ks)
        print(f"[Exp B] mu={mu:.2f} (Ks={Ks:.3f}): running {n_trials} trials ...")
        H_finals = run_trials(oim, n_trials=n_trials,
                              t_span=t_span, n_points=n_points)

        if len(H_finals) == 0:
            ax.text(0.5, 0.5, "No trials converged", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            ax.set_title(f"mu={mu} (Ks={Ks})")
            continue

        bins = np.arange(H_finals.min() - 0.5, H_finals.max() + 1.5, 1.0)
        ax.hist(H_finals, bins=bins, color="steelblue",
                edgecolor="black", alpha=0.85, zorder=2)
        ax.axvline(H_ground, color="red", linestyle="--", linewidth=1.5,
                   label=f"Ground state H = {H_ground:.1f}", zorder=3)
        ax.set_xlabel("Ising Hamiltonian H  (after rounding)", fontsize=11)
        ax.set_ylabel("Count",                                  fontsize=11)
        ax.set_title(f"$\\mu = {mu}$  ($K_s = {Ks}$)", fontsize=12)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Exp B -- Ising Hamiltonian distribution over {n_trials} trials\n"
        f"(N={N}, K={K},  H rounded from final phase state)",
        fontsize=12,
    )
    fig.tight_layout()
    print("[Exp B] Close the window to proceed.")
    plt.show()


# ===========================================================================
# Experiment C — Interactive slider
# ===========================================================================

def experiment_C(J, K, mu_init, mu_min, mu_max, *,
                 warn_large_N=16,
                 n_trials_sweep=30, t_end_sweep=50.0,
                 n_points_sweep=500, n_ks_steps=40):
    """
    Two-panel interactive figure with mu slider.

    LEFT  -- lambda_L min/max vs OIM Lyapunov energy (Fig. 3 of the paper).
             Annotation box shows live lambda extremes AND the fixed
             D-matrix theory thresholds (Theorem 2 / Remark 7).

    RIGHT -- mean Ising Hamiltonian H(final, rounded) +/- 1 std vs Ks.
             Red dots  : grid points where >= 50% of trials converged
                         naturally to a binary state (clickable tooltip).
             Black -.- : K_s* theory (Remark 7).
             Gold  |   : per-equilibrium thresholds (Theorem 2).
             Orange -- : empirical K_s* (first point >= 50% binarized).
             Red line  : current slider K_s.
    """
    N = J.shape[0]
    if N > warn_large_N:
        print(f"[Exp C] WARNING: N={N} -> 2^N={2**N} configs per move.")

    def _is_binarized(phi: np.ndarray, tol: float = 0.1) -> bool:
        """True if every phase is within tol rad of 0 or pi (mod 2pi)."""
        phi_mod = phi % (2.0 * np.pi)
        return bool(np.all((phi_mod < tol) | (np.abs(phi_mod - np.pi) < tol)))

    # ── Theory: D-matrix binarisation threshold (Remark 7) ───────────────
    print(f"[Exp C] Computing D-matrix thresholds (Remark 7) for {2**N} equilibria ...")
    oim_theory = OIM_Maxcut(J, K=K, Ks=0.0)
    theory     = oim_theory.binarization_threshold()
    Ks_theory  = theory["Ks_star"]
    print(f"  K_s* (theory, Remark 7) = {Ks_theory:.4f}   mu* = {2*Ks_theory:.4f}")
    print(f"  Easiest eq: {theory['easiest_eq']}   "
          f"thr = {min(theory['per_eq'].values()):.4f}")
    print(f"  Hardest eq: {theory['hardest_eq']}   "
          f"thr = {max(theory['per_eq'].values()):.4f}")

    # Unique (clamped) per-equilibrium thresholds for rug marks
    unique_thr = sorted(set(max(0.0, v) for v in theory["per_eq"].values()))

    # ── Ground state Ising Hamiltonian (Ks-independent) ───────────────────
    H_ground = ground_ising_energy(oim_theory)
    print(f"  Ground state Ising H = {H_ground:.4f}")

    # ── Pre-compute empirical H_ising mean / std over Ks grid ─────────────
    ks_grid      = np.linspace(mu_min / 2.0, mu_max / 2.0, n_ks_steps)
    mean_H_arr   = np.full(n_ks_steps, np.nan)
    std_H_arr    = np.full(n_ks_steps, np.nan)
    frac_bin_arr = np.zeros(n_ks_steps)

    print(f"[Exp C] Pre-computing {n_ks_steps} pts x {n_trials_sweep} trials ...")
    for idx, Ks in enumerate(ks_grid):
        oim      = OIM_Maxcut(J, K=K, Ks=Ks)
        H_trials = []
        n_bin    = 0
        for _ in range(n_trials_sweep):
            phi0  = random_phi0(N)
            sol   = oim.simulate(phi0, (0.0, t_end_sweep), n_points_sweep)
            if sol is None:
                continue
            phi_f = sol.y[:, -1]
            # Always record Ising H after rounding, regardless of binarization
            H_trials.append(oim.ising_hamiltonian(phi_f))
            if _is_binarized(phi_f):
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
    fig      = plt.figure(figsize=(16, 7), num="Experiment C: Interactive Stability")
    ax_left  = fig.add_axes((0.06, 0.20, 0.41, 0.70))
    ax_right = fig.add_axes((0.55, 0.20, 0.42, 0.70))
    ax_sl    = fig.add_axes((0.15, 0.06, 0.70, 0.03))

    # ── LEFT PANEL — lambda_L vs OIM Lyapunov energy ──────────────────────
    def _compute_lam(mu: float):
        oim    = OIM_Maxcut(J, K=K, Ks=mu / 2.0)
        E, lam = scan_configs(oim)
        He, lmin, lmax = lambda_vs_energy(E, lam)
        return He, lmin, lmax, global_min_energy(E), lam

    He0, lmin0, lmax0, Eg0, lam0 = _compute_lam(mu_init)

    line_max, = ax_left.plot(He0, lmax0, "o-", color="darkorange",
                             lw=1.8, ms=6, label=r"max $\lambda_L$ per level")
    line_min, = ax_left.plot(He0, lmin0, "s-", color="steelblue",
                             lw=1.8, ms=6, label=r"min $\lambda_L$ per level")
    ax_left.axhline(0.0, color="k", linestyle="--", lw=1.2)

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
            edgecolor="red", facecolor="none", linestyle="--", zorder=5,
        ))

    _draw_ground_box(ax_left, Eg0, lmin0, lmax0, He0)
    ax_left.set_xlabel("OIM Lyapunov Energy E",                        fontsize=12)
    ax_left.set_ylabel(r"Largest Lyapunov exponent $\lambda_L$",       fontsize=12)
    ax_left.legend(fontsize=9, loc="upper right")
    ax_left.set_xlim(He0.min() - 1, He0.max() + 1)
    ax_left.set_ylim(lmin0.min() - 0.8, lmax0.max() + 0.8)
    ax_left.set_title(
        fr"Stability vs OIM Energy ($\mu$={mu_init:.2f}, $K_s$={mu_init/2:.3f})"
        f"\n(N={N}, K={K})  |  red box = globally optimal",
        fontsize=11,
    )

    def _eig_str(lam_all):
        return (
            f"Global $\\lambda_L$ min = {lam_all.min():.4f}\n"
            f"Global $\\lambda_L$ max = {lam_all.max():.4f}\n"
            f"-- D-matrix theory (Remark 7) --\n"
            f"$K_s^*$ = {Ks_theory:.4f}   $\\mu^*$ = {2*Ks_theory:.4f}\n"
            f"Easiest: {theory['easiest_eq']}   Hardest: {theory['hardest_eq']}"
        )

    eig_text = ax_left.text(
        0.02, 0.97, _eig_str(lam0),
        transform=ax_left.transAxes,
        fontsize=8.5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="gray", alpha=0.9),
    )

    # ── RIGHT PANEL — Ising H(final, rounded) vs Ks ───────────────────────
    y_lo = ((mean_H_arr[valid] - std_H_arr[valid]).min() - 0.3
            if valid.any() else H_ground - 2)
    y_hi = ((mean_H_arr[valid] + std_H_arr[valid]).max() + 0.3
            if valid.any() else H_ground + 2)

    if valid.any():
        ax_right.fill_between(
            ks_grid[valid],
            (mean_H_arr - std_H_arr)[valid],
            (mean_H_arr + std_H_arr)[valid],
            alpha=0.25, color="steelblue", label="+/-1 std",
        )
        ax_right.plot(ks_grid[valid], mean_H_arr[valid],
                      color="steelblue", lw=2.0, label="mean H (rounded)")

    ax_right.axhline(H_ground, color="limegreen", linestyle=":", lw=1.5,
                     label=f"Ground state H = {H_ground:.1f}", zorder=3)

    # Theory binarisation threshold K_s* (Remark 7)
    ax_right.axvline(Ks_theory, color="black", linestyle="-.", lw=2.2, zorder=6,
                     label=f"$K_s^*$ theory (Rem.7) = {Ks_theory:.4f}")

    # Current K_s: moveable line + dot
    ks_init    = mu_init / 2.0
    vline_cur  = ax_right.axvline(ks_init, color="red", lw=1.8,
                                   label=r"Current $K_s$", zorder=7)
    _mean_init = (float(np.interp(ks_init, ks_grid[valid], mean_H_arr[valid]))
                  if valid.any() else np.nan)
    dot_cur,   = ax_right.plot([ks_init], [_mean_init], "o",
                                color="red", ms=8, zorder=8)

    # ── Binarized grid points — clickable red dots ────────────────────────
    bin_valid = bin_mask & valid
    bin_xs    = ks_grid[bin_valid]
    bin_ys    = mean_H_arr[bin_valid]
    bin_idxs  = np.where(bin_valid)[0]

    bin_dots  = ax_right.scatter(
        bin_xs, bin_ys,
        color="red", s=70, zorder=9, picker=6,
        edgecolors="white", linewidths=0.8,
        label="Naturally binarized (click)",
    )

    annot = ax_right.annotate(
        "", xy=(0, 0), xytext=(18, 18),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#1e1e2e",
                  edgecolor="red", alpha=0.93),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
        color="white", fontsize=8.5, zorder=11,
    )
    annot.set_visible(False)

    ax_right.set_xlabel(r"$K_s$",                                        fontsize=12)
    ax_right.set_ylabel("Ising Hamiltonian H  (rounded final state)",    fontsize=11)
    ax_right.set_title(
        f"Ising H(final, rounded) vs $K_s$ -- {N}-node graph\n"
        f"({n_trials_sweep} trials/pt,  t_end={t_end_sweep:.0f})  "
        f"| red dots = naturally binarized",
        fontsize=11,
    )
    ax_right.legend(fontsize=8.5, loc="lower right")
    ax_right.set_xlim(ks_grid[0], ks_grid[-1])
    if valid.any():
        ax_right.set_ylim(y_lo, y_hi)

    _idx0   = int(np.argmin(np.abs(ks_grid - ks_init)))
    _is_bin = frac_bin_arr[_idx0] >= 0.5

    def _info_str(ks, mean_h, is_bin):
        return (
            f"$K_s$ = {ks:.4f}\n"
            f"mean H (rounded) = {mean_h:.3f}\n"
            f"K_s* theory = {Ks_theory:.4f}\n"
            + ("BINARIZED" if is_bin else "NOT BINARIZED")
        )

    info_text = ax_right.text(
        0.03, 0.07, _info_str(ks_init, _mean_init, _is_bin),
        transform=ax_right.transAxes,
        fontsize=9, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="#1e1e2e", edgecolor="gray", alpha=0.88),
        color="white",
    )

    # ── Slider ────────────────────────────────────────────────────────────
    slider = Slider(ax_sl, r"$\mu = 2K_s$", mu_min, mu_max,
                    valinit=mu_init, valstep=0.05, color="steelblue")

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
            fr"Stability vs OIM Energy ($\mu$={mu_cur:.2f}, $K_s$={ks_cur:.3f})"
            f"\n(N={N}, K={K})  |  red box = globally optimal",
            fontsize=11,
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

    # ── Pick handler — fires when user clicks a binarized dot ────────────
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
            f"Ks  = {ks_val:.4f}\n"
            f"mean H (rounded) = {h_val:.3f}\n"
            f"ground state H   = {H_ground:.3f}\n"
            f"naturally binarized: {frac*100:.0f}% of trials\n"
            f"{vs_theory} K_s* theory ({Ks_theory:.4f})"
        )
        annot.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)
    fig.canvas.mpl_connect("button_press_event", lambda e:
        (annot.set_visible(False), fig.canvas.draw_idle())
        if e.inaxes is ax_right else None
    )

    print("[Exp C] Drag the mu slider to explore stability. Close when done.")
    plt.show()


# ===========================================================================
# CLI
# ===========================================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="OIM Stability Experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── General ───────────────────────────────────────────────────────────
    parser.add_argument("--graph",       "-g", required=True, metavar="FILE",
                        help="Path to graph file")
    parser.add_argument("--experiments", "-e", type=str,  default="ABC",
                        help="Which experiments to run, e.g. A, B, C, AB, ABC")
    parser.add_argument("--seed",              type=int,  default=None,
                        help="Global random seed")

    # ── OIM physics ───────────────────────────────────────────────────────
    oim = parser.add_argument_group("OIM physics")
    oim.add_argument("--K",       type=float, default=1.0,
                     help="Oscillator coupling strength K")
    oim.add_argument("--mu-init", type=float, default=1.6, metavar="MU",
                     help="Initial mu = 2*Ks for slider / reference")

    # ── mu sweep (shared by A and C) ──────────────────────────────────────
    sweep = parser.add_argument_group("mu sweep  [Exp A and C]")
    sweep.add_argument("--mu-min",   type=float, default=0.1,
                       help="Lower bound of mu sweep")
    sweep.add_argument("--mu-max",   type=float, default=3.0,
                       help="Upper bound of mu sweep")
    sweep.add_argument("--mu-steps", type=int,   default=30,
                       help="Number of mu values in Exp A scatter")

    # ── Experiment B ──────────────────────────────────────────────────────
    expB = parser.add_argument_group("Experiment B  [Ising H histograms]")
    expB.add_argument("--mu-hist",  nargs="+", type=float, default=[1.6, 3.0],
                      metavar="MU",
                      help="One or more mu values to histogram")
    expB.add_argument("--trials",   type=int,   default=50,
                      help="Number of random trials per histogram")
    expB.add_argument("--t-end",    type=float, default=50.0,
                      help="Integration end time")
    expB.add_argument("--n-points", type=int,   default=500,
                      help="Number of time points saved per trajectory")

    # ── Experiment C ──────────────────────────────────────────────────────
    expC = parser.add_argument_group("Experiment C  [interactive slider]")
    expC.add_argument("--n-ks-steps",    type=int,   default=40,
                      help="Number of Ks grid points in the empirical sweep")
    expC.add_argument("--c-trials",      type=int,   default=30,
                      help="Random trials per Ks grid point")
    expC.add_argument("--c-t-end",       type=float, default=50.0,
                      help="Integration end time for Exp C trials")
    expC.add_argument("--c-n-points",    type=int,   default=500,
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
        J = read_graph_to_J(args.graph)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    info = graph_info(J)
    print(f"  N={info['N']} nodes, {info['edges']} edges, "
          f"density={info['density']:.3f}")

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"  Random seed: {args.seed}")

    K         = args.K
    mu_values = np.linspace(args.mu_min, args.mu_max, args.mu_steps)

    if run_A:
        print("\n" + "="*60 + "\nEXPERIMENT A\n" + "="*60)
        experiment_A(J, K, mu_values)

    if run_B:
        print("\n" + "="*60 + "\nEXPERIMENT B\n" + "="*60)
        experiment_B(J, K, args.mu_hist, args.trials,
                     (0.0, args.t_end), args.n_points)

    if run_C:
        print("\n" + "="*60 + "\nEXPERIMENT C\n" + "="*60)
        experiment_C(J, K, args.mu_init, args.mu_min, args.mu_max,
                     n_ks_steps=args.n_ks_steps,
                     n_trials_sweep=args.c_trials,
                     t_end_sweep=args.c_t_end,
                     n_points_sweep=args.c_n_points)

    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()