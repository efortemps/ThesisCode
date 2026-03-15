import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import matplotlib.lines as mlines
from Stability_OIM.src.oim import OscillatorIsingMachine

# ─────────────────────────────────────────────────────────────────────────────
# Problem definition (two coupled spins, unweighted graph, Section IV‑A)
# ─────────────────────────────────────────────────────────────────────────────
K = 1.0
J = np.array([[0., -1.],
              [-1., 0.]])   # J_12 = J_21 = -1 (anti-ferromagnetic edge)

equilibria = [
    (r"phi*=(0,0)",         np.array([0.,      0.      ])),
    (r"phi*=(0,pi)",        np.array([0.,      np.pi   ])),
    (r"phi*=(pi,0)",        np.array([np.pi,   0.      ])),
    (r"phi*=(pi,pi)",       np.array([np.pi,   np.pi   ])),
    (r"phi*=(pi/2,pi/2)",   np.array([np.pi/2, np.pi/2 ])),
]

# Two parameter combinations (matching Figs. 6 & 7 in the paper)
param_sets = [
    (1.5, "K=1, Ks=1.5  [Ks > K  →  all M2 equilibria stable]"),
    (0.5, "K=1, Ks=0.5  [Ks < K  →  (0,0) and (pi,pi) unstable]"),
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. Stability analysis table
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("EXPERIMENT 1 — Stability of five Type‑I equilibria (N=2 spins)")
print("=" * 70)

for Ks, label in param_sets:
    oim = OscillatorIsingMachine(J, K, Ks)
    print(f"\n  {label}")
    print(f"  {'Equilibrium':<22} {'Type':<16} {'Threshold Ks*':<16} Verdict")
    print(f"  {'-'*65}")
    for name, phi_star in equilibria:
        res = oim.stability_analysis(phi_star, verbose=False)
        print(f"  {name:<22} {res['eq_type']:<16} {res['threshold']:<16.4f} {res['verdict']}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Phase portraits
# ─────────────────────────────────────────────────────────────────────────────
N_GRID   = 16          # grid density for initial conditions
T_SPAN   = (0., 12.)  # integration horizon
N_POINTS = 300         # time‑steps recorded per trajectory

phi1_vals = np.linspace(-1.8, 4.8, N_GRID)
phi2_vals = np.linspace(-1.8, 4.8, N_GRID)
phi0_list = [np.array([p1, p2]) for p1 in phi1_vals for p2 in phi2_vals]

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
fig.suptitle("Experiment 1 — Phase portraits, OIM with N=2 coupled spins",
             fontsize=13, fontweight="bold", y=1.01)

cmap = plt.get_cmap('rainbow')
norm = Normalize(vmin=0, vmax=len(phi0_list) - 1)

for ax, (Ks, subtitle) in zip(axes, param_sets):
    oim  = OscillatorIsingMachine(J, K, Ks)
    sols = oim.simulate_many(phi0_list, t_span=T_SPAN, n_points=N_POINTS)

    # — trajectories
    for idx, sol in enumerate(sols):
        c = cmap(norm(idx))
        ax.plot(sol.y[0], sol.y[1], color=c, alpha=0.35, linewidth=0.75)
        if sol.y.shape[1] > 6:
            ax.annotate("", xy=(sol.y[0, 5], sol.y[1, 5]),
                        xytext=(sol.y[0, 0], sol.y[1, 0]),
                        arrowprops=dict(arrowstyle="->", color=c,
                                        lw=0.6, mutation_scale=8))

    # — equilibria + domain‑of‑attraction circles (Theorem 7: radius = pi/2)
    for name, phi_star in equilibria:
        res    = oim.stability_analysis(phi_star, verbose=False)
        stable = res["is_stable"]
        color  = "steelblue" if stable is True else ("firebrick" if stable is False else "orange")
        marker = "o"         if stable is True else ("X"         if stable is False else "D")
        ax.scatter(phi_star[0], phi_star[1],
                   c=color, s=130, zorder=6, marker=marker,
                   edgecolors="white", linewidths=0.8)
        if stable is True:
            ax.add_patch(mpatches.Circle((phi_star[0], phi_star[1]), np.pi / 2,
                                         fill=False, color="red",
                                         linestyle="--", linewidth=1.5, zorder=5))

    # — axes formatting
    pi_ticks  = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    pi_labels = ["0", "π/2", "π", "3π/2", "2π"]
    ax.set_xticks(pi_ticks);  ax.set_xticklabels(pi_labels, fontsize=9)
    ax.set_yticks(pi_ticks);  ax.set_yticklabels(pi_labels, fontsize=9)
    ax.set_xlim(-1.8, 4.8);   ax.set_ylim(-1.8, 4.8)
    ax.set_xlabel(r"$\phi_1$", fontsize=11)
    ax.set_ylabel(r"$\phi_2$", fontsize=11)
    ax.set_title(subtitle, fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linewidth=0.5)

# — shared legend
fig.legend(handles=[
    mpatches.Patch(facecolor="steelblue", label="Asymptotically stable eq."),
    mpatches.Patch(facecolor="firebrick", label="Unstable eq."),
    mpatches.Patch(facecolor="orange",    label="Critical / inconclusive"),
    mlines.Line2D([0],[0], color="red", linestyle="--", lw=1.5,
               label=r"Domain‑of‑attraction est. ($r=\pi/2$)"),
], loc="lower center", ncol=4, fontsize=9,
   bbox_to_anchor=(0.5, -0.06), framealpha=0.9)

plt.tight_layout()
plt.savefig("experiment1_phase_portraits.png", dpi=150, bbox_inches="tight")
print("\nFigure saved → experiment1_phase_portraits.png")
plt.show()
