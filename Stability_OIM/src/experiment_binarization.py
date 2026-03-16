import numpy as np
import matplotlib.pyplot as plt
from itertools import product as iproduct
from Stability_OIM.src.OIM_Stability import OscillatorIsingMachine
from Stability_OIM.src.graph_utils import read_graph

# ── 1. Load King graph ─────────────────────────────────────────────────────────
J = read_graph("Stability_OIM/data/king.txt")          # loads from the text file
K = 1.0
N = J.shape[0]                         # 9 spins

# ── 2. Compute binarization threshold (Remark 7) ──────────────────────────────
print("Computing binarization threshold over all 2^9 = 512 Type I equilibria…")
oim_ref    = OscillatorIsingMachine(J, K, 1.0)
thresholds = [oim_ref.stability_threshold(np.array([b * np.pi for b in bits]))
              for bits in iproduct([0, 1], repeat=N)]
bin_thresh = min(thresholds)
print(f"  Ks* = {bin_thresh:.6f}  (paper: 0.276400)\n")

# ── 3. Simulate for 4 Ks values ───────────────────────────────────────────────
KS_VALUES = [0.0100, 0.2400, 0.2770, 0.7900]
N_INIT    = 30           # random initial conditions per Ks value
T_SPAN    = (0., 400.)   # long integration to observe convergence
N_POINTS  = 1500
rng       = np.random.default_rng(42)

phi0_list = [rng.uniform(-np.pi, np.pi, N) for _ in range(N_INIT)]

print("Simulating…")
results = {}
for Ks in KS_VALUES:
    oim  = OscillatorIsingMachine(J, K, Ks)
    sols = oim.simulate_many(phi0_list, t_span=T_SPAN, n_points=N_POINTS)

    H_finals = []
    for sol in sols:
        phi_final = sol.y[:, -1]
        s = np.cos(phi_final)
        H_finals.append(float(-0.5 * (s @ J @ s)))

    results[Ks] = {"sols": sols, "H_finals": H_finals}
    print(f"  Ks = {Ks:.4f}  |  mean H(T) = {np.mean(H_finals):.4f}  "
          f"std = {np.std(H_finals):.4f}  "
          f"{'[BINARIZED]' if Ks > bin_thresh else '[NOT binarized]'}")

# ── 4. Plot phase evolution (Figs 11–14 reproduction) ─────────────────────────
BG, PANEL, WHITE = "#111827", "#1e293b", "#f1f5f9"
SPIN_COLORS      = plt.get_cmap("tab10")(np.linspace(0, 1, N))

labels = [
    f"Ks = 0.0100  (substantially below Ks* = {bin_thresh:.4f})",
    f"Ks = 0.2400  (slightly below Ks*)",
    f"Ks = 0.2770  (slightly above Ks*)  →  binarized",
    f"Ks = 0.7900  (substantially above Ks*)  →  binarized",
]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor(BG)
fig.suptitle(
    f"Binarization Experiment — 3×3 King Graph  (K = 1,  Ks* ≈ {bin_thresh:.4f})",
    color=WHITE, fontsize=13, fontweight="bold", y=1.01
)

for ax, Ks, label in zip(axes.flat, KS_VALUES, labels):
    ax.set_facecolor(PANEL)
    sols     = results[Ks]["sols"]
    H_finals = results[Ks]["H_finals"]
    t        = sols[0].t

    for sol in sols:
        for spin in range(N):
            ax.plot(t, sol.y[spin], color=SPIN_COLORS[spin],
                    alpha=0.22, linewidth=0.65)

    # Reference lines at 0 and ±π
    ax.axhline(y= np.pi, color="#ef9a9a", linestyle="--", linewidth=1.2,
               alpha=0.8, label="π")
    ax.axhline(y=0,      color="#90caf9", linestyle="--", linewidth=1.2,
               alpha=0.8, label="0")
    ax.axhline(y=-np.pi, color="#ef9a9a", linestyle="--", linewidth=1.2,
               alpha=0.5)

    # Status badge
    is_binarized = Ks > bin_thresh
    ax.text(0.97, 0.94,
            "BINARIZED ✓" if is_binarized else "NOT binarized ✗",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, fontweight="bold",
            color="#4caf50" if is_binarized else "#ef5350",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f172a", alpha=0.9))

    # Mean Hamiltonian
    ax.text(0.97, 0.06,
            f"mean H(final) = {np.mean(H_finals):.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=WHITE,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f172a", alpha=0.85))

    # Formatting
    ax.set_title(label, color=WHITE, fontsize=10, pad=6)
    ax.set_xlabel("time  t", color=WHITE, fontsize=10)
    ax.set_ylabel("phase  φ  (rad)", color=WHITE, fontsize=10)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-4.5, 4.5)
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels(["-3.14", "-1.57", "0", "1.57", "3.14"],
                       color=WHITE, fontsize=8.5)
    t_ticks = np.linspace(t[0], t[-1], 5)
    ax.set_xticks(t_ticks)
    ax.set_xticklabels([f"{v:.0f}" for v in t_ticks], color=WHITE, fontsize=8.5)
    ax.tick_params(colors=WHITE)
    ax.grid(True, alpha=0.15, color=WHITE, linewidth=0.5)
    ax.legend(loc="upper left", fontsize=8.5,
              facecolor="#0f172a", labelcolor=WHITE, framealpha=0.9)
    for sp in ax.spines.values():
        sp.set_edgecolor("#334155")

plt.tight_layout()
plt.savefig("experiment_binarization.png", dpi=145,
            bbox_inches="tight", facecolor=BG)
print("\nFigure saved → experiment_binarization.png")
plt.show()
