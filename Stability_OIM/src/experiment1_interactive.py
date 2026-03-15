import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider
import matplotlib.lines as mlines
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import time
from Stability_OIM.src.oim import OscillatorIsingMachine

# ── Problem setup ─────────────────────────────────────────────────────────────
K = 1.0
J = np.array([[0., -1.],
              [-1.,  0.]])

equilibria = [
    (r"$\phi^*=(0,0)$",           np.array([0.,      0.      ])),
    (r"$\phi^*=(0,\pi)$",         np.array([0.,      np.pi   ])),
    (r"$\phi^*=(\pi,0)$",         np.array([np.pi,   0.      ])),
    (r"$\phi^*=(\pi,\pi)$",       np.array([np.pi,   np.pi   ])),
    (r"$\phi^*=(\pi/2,\pi/2)$",   np.array([np.pi/2, np.pi/2 ])),
]

EQ_COLORS_THRESH = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

# ── Pre-computation ────────────────────────────────────────────────────────────
KS_MIN, KS_MAX, N_KS = 0.5, 1.5, 21
Ks_values = np.linspace(KS_MIN, KS_MAX, N_KS)
KS_STEP   = float(Ks_values[1] - Ks_values[0])

N_GRID   = 12
phi1_vals = np.linspace(-1.8, 4.8, N_GRID)
phi2_vals = np.linspace(-1.8, 4.8, N_GRID)
phi0_list = [np.array([p1, p2]) for p1 in phi1_vals for p2 in phi2_vals]
T_SPAN, N_POINTS = (0., 10.), 200

print("Pre-computing trajectories for 21 Ks values — please wait…")
t0 = time.time()
all_trajs, all_stab = {}, {}

for Ks in Ks_values:
    oim  = OscillatorIsingMachine(J, K, Ks)
    sols = oim.simulate_many(phi0_list, t_span=T_SPAN, n_points=N_POINTS)
    idx = int(np.searchsorted(Ks_values, Ks))   # integer index as key
    all_trajs[idx] = sols
    all_stab[idx]  = {name: oim.stability_analysis(phi_star, verbose=False)
                  for name, phi_star in equilibria}

    pct = 100 * (np.searchsorted(Ks_values, Ks) + 1) / N_KS
    print(f"  [{pct:5.1f}%]  Ks = {Ks:.3f}  ✓", flush=True)

# Stability thresholds are fixed (depend only on J, K, phi*)
oim_ref    = OscillatorIsingMachine(J, K, 1.0)
thresholds = {name: oim_ref.stability_analysis(phi_star, verbose=False)["threshold"]
              for name, phi_star in equilibria}
print(f"\nReady in {time.time()-t0:.1f} s  —  opening interactive window…\n")

# ── Figure layout (dark theme) ─────────────────────────────────────────────────
BG      = "#111827"
PANEL   = "#1e293b"
ACCENT  = "#e94560"
WHITE   = "#f1f5f9"

fig = plt.figure(figsize=(17, 7.5), facecolor=BG)
ax_phase  = fig.add_axes((0.04, 0.13, 0.44, 0.79))   # left: phase portrait
ax_thresh = fig.add_axes((0.54, 0.13, 0.44, 0.79))   # right: threshold diagram
ax_slider = fig.add_axes((0.15, 0.03, 0.70, 0.035))  # bottom: slider

for ax in (ax_phase, ax_thresh):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE)
    for sp in ax.spines.values():
        sp.set_edgecolor("#334155")

# ── Static right panel: threshold diagram ─────────────────────────────────────
for i, (name, _) in enumerate(equilibria):
    thr = thresholds[name]
    ax_thresh.axhline(y=thr, color=EQ_COLORS_THRESH[i],
                      linewidth=2.5, label=f"{name}   Ks* = {thr:.2f}")

# Diagonal Ks = Ks* line
ax_thresh.plot([KS_MIN, KS_MAX], [KS_MIN, KS_MAX],
               color=WHITE, linestyle="--", linewidth=1.2, alpha=0.45,
               label="Ks = Ks*  (bifurcation)")

# Shaded stability/instability region (for threshold = K = 1)
ax_thresh.axhspan(KS_MIN, K,      alpha=0.06, color="#ef5350")
ax_thresh.axhspan(K,      KS_MAX, alpha=0.06, color="#4caf50")
ax_thresh.text(1.02, K + 0.04, "stable above →", color="#81c784", fontsize=7.5, ha="left")
ax_thresh.text(1.02, K - 0.12, "← unstable below", color="#ef9a9a", fontsize=7.5, ha="left")

# Dynamic vertical cursor (updated by slider)
vline = ax_thresh.axvline(x=1.0, color=ACCENT, linestyle="-", linewidth=2.5, zorder=10,
                           label="Current Ks")

ax_thresh.set_xlim(KS_MIN, KS_MAX)
ax_thresh.set_ylim(-0.25, 1.8)
ax_thresh.set_xlabel("Ks", color=WHITE, fontsize=11)
ax_thresh.set_ylabel("Stability threshold  Ks*  (= K λmax(D) / 2)", color=WHITE, fontsize=10)
ax_thresh.set_title("Stability thresholds — Theorem 2  (K = 1)", color=WHITE, fontsize=11)
ax_thresh.legend(fontsize=8.5, facecolor="#0f172a", labelcolor=WHITE,
                 loc="upper left", framealpha=0.85)
ax_thresh.grid(True, alpha=0.18, color=WHITE)

# Annotation box showing stability verdict at current Ks
verdict_txt = ax_thresh.text(
    0.98, 0.04, "", transform=ax_thresh.transAxes,
    ha="right", va="bottom", fontsize=9, color=WHITE,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f172a", alpha=0.85))

# ── Slider ─────────────────────────────────────────────────────────────────────
slider = Slider(ax_slider, "  Ks  ", KS_MIN, KS_MAX,
                valinit=1.0, valstep=KS_STEP,
                color=ACCENT, track_color="#0f172a")
ax_slider.set_facecolor("#0f172a")
slider.label.set_color(WHITE);  slider.valtext.set_color(WHITE)

# ── Phase portrait drawing ──────────────────────────────────────────────────────
cmap = plt.get_cmap('rainbow')
norm_traj  = Normalize(vmin=0, vmax=len(phi0_list) - 1)
PI_TICKS   = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
PI_LABELS  = ["0", "π/2", "π", "3π/2", "2π"]

def draw_phase(Ks):
    ax_phase.clear()
    ax_phase.set_facecolor(PANEL)

    idx  = int(np.argmin(np.abs(Ks_values - Ks))) 
    sols = all_trajs[idx]
    stab = all_stab[idx]
    key  = Ks_values[idx]                            

    # Trajectories
    for idx, sol in enumerate(sols):
        c = cmap(norm_traj(idx))
        ax_phase.plot(sol.y[0], sol.y[1], color=c, alpha=0.28, linewidth=0.7)
        if sol.y.shape[1] > 8:
            ax_phase.annotate(
                "", xy=(sol.y[0, 7], sol.y[1, 7]),
                xytext=(sol.y[0, 0], sol.y[1, 0]),
                arrowprops=dict(arrowstyle="->", color=c, lw=0.5, mutation_scale=7))

    # Equilibria + domain-of-attraction circles
    stable_list, unstable_list = [], []
    for name, phi_star in equilibria:
        res    = stab[name]
        stable = res["is_stable"]
        ec     = "#4fc3f7" if stable is True else ("#ef5350" if stable is False else "#ffb74d")
        mk     = "o"       if stable is True else ("X"       if stable is False else "D")
        ax_phase.scatter(phi_star[0], phi_star[1],
                         c=ec, s=140, zorder=6, marker=mk,
                         edgecolors=WHITE, linewidths=0.8)
        if stable is True:
            ax_phase.add_patch(mpatches.Circle(
                (phi_star[0], phi_star[1]), np.pi / 2,
                fill=False, color="#ef9a9a", linestyle="--", linewidth=1.5, zorder=5))
            stable_list.append(name.replace("$", "").replace("\\", ""))
        elif stable is False:
            unstable_list.append(name.replace("$", "").replace("\\", ""))

    # Formatting
    ax_phase.set_xticks(PI_TICKS);  ax_phase.set_xticklabels(PI_LABELS, fontsize=9, color=WHITE)
    ax_phase.set_yticks(PI_TICKS);  ax_phase.set_yticklabels(PI_LABELS, fontsize=9, color=WHITE)
    ax_phase.set_xlim(-1.8, 4.8);   ax_phase.set_ylim(-1.8, 4.8)
    ax_phase.set_xlabel(r"$\phi_1$", fontsize=12, color=WHITE)
    ax_phase.set_ylabel(r"$\phi_2$", fontsize=12, color=WHITE)
    ax_phase.set_title(f"Phase portrait   K = {K},   Ks = {key:.3f}", color=WHITE, fontsize=11)
    ax_phase.set_aspect("equal")
    ax_phase.grid(True, alpha=0.14, color=WHITE)
    ax_phase.tick_params(colors=WHITE)
    for sp in ax_phase.spines.values():
        sp.set_edgecolor("#334155")

    # Mini-legend inside phase portrait
    legend_elems = [
        mpatches.Patch(facecolor="#4fc3f7", label="Asympt. stable"),
        mpatches.Patch(facecolor="#ef5350", label="Unstable"),
        mpatches.Patch(facecolor="#ffb74d", label="Critical"),
        mlines.Line2D([0],[0], color="#ef9a9a", lw=1.2, linestyle="--",
                   label=r"DoA est. ($r=\pi/2$)"),
    ]
    ax_phase.legend(handles=legend_elems, loc="upper right", fontsize=7.5,
                    facecolor="#0f172a", labelcolor=WHITE, framealpha=0.85)

    # Update verdict box in right panel
    verdict_txt.set_text(
        f"Stable:    {', '.join(stable_list) or '—'}\n"
        f"Unstable: {', '.join(unstable_list) or '—'}")


def update(val):
    Ks = slider.val
    draw_phase(Ks)
    vline.set_xdata([Ks, Ks])
    fig.canvas.draw_idle()

slider.on_changed(update)
fig.suptitle(
    "Experiment 1 — Interactive OIM Phase Portraits   (N = 2 coupled spins,  K = 1)",
    color=WHITE, fontsize=13, fontweight="bold", y=1.003)

draw_phase(1.0)
plt.show()
