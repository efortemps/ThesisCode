import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
from itertools import product as iproduct
import time

from Stability_OIM.src.OIM_Stability import OscillatorIsingMachine
from Stability_OIM.src.graph_utils import read_graph

# ── Problem setup ──────────────────────────────────────────────────────────────
K       = 1.0
J       = read_graph("Stability_OIM/data/king.txt")
N_SPINS = J.shape[0]   # 9

# Binarization threshold  (Remark 7: min over all 2^N Type I equilibria)
print("Computing binarization threshold…")
oim_ref    = OscillatorIsingMachine(J, K, 1.0)
bin_thresh = min(
    oim_ref.stability_threshold(np.array([b * np.pi for b in bits]))
    for bits in iproduct([0, 1], repeat=N_SPINS)
)
print(f"  Ks* = {bin_thresh:.6f}  (paper: 0.276400)\n")

# ── Pre-computation ────────────────────────────────────────────────────────────
KS_MIN, KS_MAX, N_KS = 0.05, 0.80, 200
Ks_values = np.linspace(KS_MIN, KS_MAX, N_KS)
KS_STEP   = float(Ks_values[1] - Ks_values[0])

N_INIT   = 12
T_SPAN   = (0., 200.)
N_POINTS = 500
rng      = np.random.default_rng(42)

phi0_list = [rng.uniform(-np.pi, np.pi, N_SPINS) for _ in range(N_INIT)]

print(f"Pre-computing {N_KS} × {N_INIT} trajectories — please wait…")
t0 = time.time()

all_trajs = []    # list[list[sol]]  — indexed by integer, NO float key bug
all_H     = []    # list[np.array]   — H(final) values per Ks

for k_idx, Ks in enumerate(Ks_values):
    oim  = OscillatorIsingMachine(J, K, Ks)
    sols = oim.simulate_many(phi0_list, t_span=T_SPAN, n_points=N_POINTS)
    H_finals = []
    for sol in sols:
        s = np.cos(sol.y[:, -1])
        H_finals.append(float(-0.5 * (s @ J @ s)))
    all_trajs.append(sols)
    all_H.append(np.array(H_finals))
    pct = 100 * (k_idx + 1) / N_KS
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    print(f"  [{bar}] {pct:5.1f}%  Ks={Ks:.3f}  "
          f"mean H={np.mean(H_finals):.3f}  "
          f"{'[BINARIZED]' if Ks > bin_thresh else ''}",
          flush=True)

mean_H_all = np.array([np.mean(h) for h in all_H])
std_H_all  = np.array([np.std(h)  for h in all_H])
print(f"\nReady in {time.time()-t0:.1f} s — opening interactive window…\n")

# ── Theme ──────────────────────────────────────────────────────────────────────
BG, PANEL, WHITE = "#111827", "#1e293b", "#f1f5f9"
ACCENT           = "#e94560"
SPIN_COLORS      = plt.get_cmap("tab10")(np.linspace(0, 1, N_SPINS))

# ── Figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 7.5), facecolor=BG)
ax_phase  = fig.add_axes((0.04, 0.13, 0.44, 0.79))   # left:  phase evolution
ax_hcurve = fig.add_axes((0.54, 0.13, 0.44, 0.79))   # right: H(Ks) curve
ax_slider = fig.add_axes((0.15, 0.03, 0.70, 0.035))  # bottom: slider

for ax in (ax_phase, ax_hcurve):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE)
    for sp in ax.spines.values():
        sp.set_edgecolor("#334155")

# ── Static right panel: H(Ks) curve ───────────────────────────────────────────
ax_hcurve.fill_between(Ks_values,
                        mean_H_all - std_H_all,
                        mean_H_all + std_H_all,
                        color="#90caf9", alpha=0.18)
ax_hcurve.plot(Ks_values, mean_H_all,
               color="#90caf9", linewidth=2.5, label="mean H(final)")
ax_hcurve.plot(Ks_values, mean_H_all - std_H_all,
               color="#90caf9", linewidth=0.8, linestyle="--", alpha=0.5)
ax_hcurve.plot(Ks_values, mean_H_all + std_H_all,
               color="#90caf9", linewidth=0.8, linestyle="--", alpha=0.5,
               label="± 1 std")

# Threshold & optimum reference lines
ax_hcurve.axvline(bin_thresh, color="#ffb74d", linestyle="--",
                  linewidth=2.0, label=f"Ks* = {bin_thresh:.4f}")
ax_hcurve.axhline(-8.0, color="#4caf50", linestyle=":",
                  linewidth=1.5, label="H = −8 (binarized optimum)")

# Shaded stability regions
ax_hcurve.axvspan(KS_MIN,     bin_thresh, alpha=0.06, color="#ef5350")
ax_hcurve.axvspan(bin_thresh, KS_MAX,     alpha=0.06, color="#4caf50")

# Region labels
ax_hcurve.text((KS_MIN + bin_thresh) / 2, -5.4,
               "not binarized", color="#ef9a9a",
               fontsize=8.5, ha="center", style="italic")
ax_hcurve.text((bin_thresh + KS_MAX) / 2, -5.4,
               "binarized", color="#81c784",
               fontsize=8.5, ha="center", style="italic")

# Dynamic cursor — moves with slider
vline_h = ax_hcurve.axvline(x=Ks_values[N_KS // 2], color=ACCENT,
                              linestyle="-", linewidth=2.5, zorder=10,
                              label="Current Ks")

# Scatter dot for current Ks mean H — updated by slider
h_dot, = ax_hcurve.plot([], [], "o", color=ACCENT, markersize=9,
                         zorder=11)

ax_hcurve.set_xlim(KS_MIN, KS_MAX)
ax_hcurve.set_ylim(-9.5, -5.0)
yt = np.arange(-9.5, -4.5, 0.5)
ax_hcurve.set_yticks(yt)
ax_hcurve.set_yticklabels([f"{v:.1f}" for v in yt], color=WHITE, fontsize=8.5)
xt = np.round(np.linspace(KS_MIN, KS_MAX, 8), 3)
ax_hcurve.set_xticks(xt)
ax_hcurve.set_xticklabels([f"{v:.2f}" for v in xt], color=WHITE, fontsize=8.5)
ax_hcurve.set_xlabel("Ks", color=WHITE, fontsize=11)
ax_hcurve.set_ylabel("H  (Ising Hamiltonian at convergence)",
                      color=WHITE, fontsize=10)
ax_hcurve.set_title("H(final) vs Ks — 3×3 King graph  (Theorem 2 / Remark 7)",
                     color=WHITE, fontsize=11)
ax_hcurve.legend(fontsize=8.5, facecolor="#0f172a", labelcolor=WHITE,
                  loc="lower right", framealpha=0.90)
ax_hcurve.grid(True, alpha=0.18, color=WHITE)

# Status text in right panel
status_txt = ax_hcurve.text(
    0.02, 0.06, "", transform=ax_hcurve.transAxes,
    ha="left", va="bottom", fontsize=9, color=WHITE,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f172a", alpha=0.85))

# ── Slider ─────────────────────────────────────────────────────────────────────
slider = Slider(ax_slider, "  Ks  ", KS_MIN, KS_MAX,
                valinit=Ks_values[N_KS // 2], valstep=KS_STEP,
                color=ACCENT, track_color="#0f172a")
ax_slider.set_facecolor("#0f172a")
slider.label.set_color(WHITE); slider.valtext.set_color(WHITE)

# ── Phase evolution drawing ────────────────────────────────────────────────────
PI_TICKS  = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
PI_LABELS = ["-π", "-π/2", "0", "π/2", "π"]

def draw_phase(idx):
    """Redraw the left panel for pre-computed index idx."""
    ax_phase.clear()
    ax_phase.set_facecolor(PANEL)

    Ks   = Ks_values[idx]
    sols = all_trajs[idx]
    h    = all_H[idx]
    t    = sols[0].t

    for sol in sols:
        for spin in range(N_SPINS):
            ax_phase.plot(t, sol.y[spin],
                          color=SPIN_COLORS[spin],
                          alpha=0.22, linewidth=0.65)

    ax_phase.axhline( np.pi, color="#ef9a9a", linestyle="--",
                      linewidth=1.2, alpha=0.85, label="π")
    ax_phase.axhline(0,      color="#90caf9", linestyle="--",
                      linewidth=1.2, alpha=0.85, label="0")
    ax_phase.axhline(-np.pi, color="#ef9a9a", linestyle="--",
                      linewidth=1.2, alpha=0.50)

    # Binarization badge
    binarized = Ks > bin_thresh
    ax_phase.text(
        0.97, 0.95,
        "BINARIZED ✓" if binarized else "NOT binarized ✗",
        transform=ax_phase.transAxes, ha="right", va="top",
        fontsize=10, fontweight="bold",
        color="#4caf50" if binarized else "#ef5350",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a", alpha=0.90))

    # Hamiltonian annotation
    ax_phase.text(
        0.97, 0.07,
        f"mean H(final) = {np.mean(h):.3f}\nstd = {np.std(h):.4f}",
        transform=ax_phase.transAxes, ha="right", va="bottom",
        fontsize=9, color=WHITE,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a", alpha=0.85))

    ax_phase.set_xlim(t[0], t[-1])
    ax_phase.set_ylim(-4.5, 4.5)
    ax_phase.set_yticks(PI_TICKS)
    ax_phase.set_yticklabels(PI_LABELS, color=WHITE, fontsize=9)
    xt = np.linspace(t[0], t[-1], 5)
    ax_phase.set_xticks(xt)
    ax_phase.set_xticklabels([f"{v:.0f}" for v in xt], color=WHITE, fontsize=9)
    ax_phase.set_xlabel("time  t", color=WHITE, fontsize=11)
    ax_phase.set_ylabel("phase  φ  (rad)", color=WHITE, fontsize=11)
    ax_phase.set_title(
        f"Phase evolution — K = {K},  Ks = {Ks:.4f}  "
        f"({'above' if binarized else 'below'} Ks* = {bin_thresh:.4f})",
        color=WHITE, fontsize=11)
    ax_phase.tick_params(colors=WHITE)
    ax_phase.grid(True, alpha=0.14, color=WHITE, linewidth=0.5)
    ax_phase.legend(loc="upper left", fontsize=9,
                    facecolor="#0f172a", labelcolor=WHITE, framealpha=0.90)
    for sp in ax_phase.spines.values():
        sp.set_edgecolor("#334155")

    # Colour bar legend for spins (small patches)
    patches = [mpatches.Patch(color=SPIN_COLORS[i], label=f"spin {i}")
               for i in range(N_SPINS)]
    ax_phase.legend(handles=patches, loc="lower left", fontsize=7,
                    facecolor="#0f172a", labelcolor=WHITE, framealpha=0.85,
                    ncol=3)


def update(val):
    idx = int(np.argmin(np.abs(Ks_values - slider.val)))
    draw_phase(idx)
    Ks = Ks_values[idx]
    vline_h.set_xdata([Ks, Ks])
    h_dot.set_data([Ks], [np.mean(all_H[idx])])
    status_txt.set_text(
        f"Ks = {Ks:.4f}\n"
        f"mean H = {np.mean(all_H[idx]):.3f}\n"
        f"{'▲ BINARIZED' if Ks > bin_thresh else '▼ not binarized'}")
    fig.canvas.draw_idle()


slider.on_changed(update)
fig.suptitle(
    "Binarization Experiment — Interactive Explorer   "
    "3×3 King Graph  (K = 1)",
    color=WHITE, fontsize=13, fontweight="bold", y=1.003)

# Initial draw at the middle Ks value
update(Ks_values[N_KS // 2])
plt.show()
