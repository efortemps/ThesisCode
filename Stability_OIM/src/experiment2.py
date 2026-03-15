import numpy as np
import matplotlib.pyplot as plt
from Stability_OIM.src.oim import OscillatorIsingMachine

# ── 1. Setup matching Experiment 2 ───────────────────────────────────────────
K = 1.0
Ks = 3.0

# 3-node path graph (Fig 3 topology: 1-2 and 2-3 are connected)
J = np.array([
    [ 0., -1.,  0.],
    [-1.,  0., -1.],
    [ 0., -1.,  0.]
])

oim = OscillatorIsingMachine(J, K, Ks)

# The two equilibria studied
phi_star_M  = np.array([0., np.pi, 0.])   # Belongs to set M (unfrustrated)
phi_star_M_bar = np.array([0., 0., 0.])   # Belongs to M_bar (frustrated)

# ── 2. Compute Radii using Theorem 7 ──────────────────────────────────────────
R1 = oim.estimate_domain_of_attraction(phi_star_M)
R2 = oim.estimate_domain_of_attraction(phi_star_M_bar)

print("=" * 60)
print("EXPERIMENT 2 — Domain of Attraction Estimates")
print("=" * 60)
print(f"Equilibrium (0, pi, 0):")
print(f"  Estimated Radius = {R1:.4f}")
print(f"  Expected approx  = {np.pi/2:.4f} (pi/2)")
print()
print(f"Equilibrium (0, 0, 0):")
print(f"  Estimated Radius = {R2:.4f}")
print(f"  Expected approx  = {0.6033*np.pi/2:.4f} (0.6033 * pi/2)")
print("=" * 60)

# ── 3. Plot 3D Spheres (Figure 8 recreation) ──────────────────────────────────
fig = plt.figure(figsize=(13, 6))
fig.suptitle("Experiment 2: Domains of Attraction (N=3 chain, K=1, Ks=3)", 
             fontsize=14, fontweight='bold')

for idx, (phi_star, R, title) in enumerate([
    (phi_star_M, R1, r"$\phi^* = (0, \pi, 0)$   [ Set $\mathcal{M}$ ]"),
    (phi_star_M_bar, R2, r"$\phi^* = (0, 0, 0)$   [ Set $\overline{\mathcal{M}}$ ]")
]):
    ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
    
    # Generate sphere coordinates
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    x = phi_star[0] + R * np.cos(u) * np.sin(v)
    y = phi_star[1] + R * np.sin(u) * np.sin(v)
    z = phi_star[2] + R * np.cos(v)
    
    # Plot transparent sphere
    ax.plot_surface(x, y, z, color='skyblue', alpha=0.3, edgecolor='w', linewidth=0.3)
    
    # Plot the equilibrium at the center
    ax.scatter(*phi_star, color='red', s=80, label="Equilibrium", zorder=5)

    # Add a vector field (arrows) strictly ON the sphere surface pointing inwards
    u_vec, v_vec = np.mgrid[0:2*np.pi:8j, 0:np.pi:6j]
    xs = (phi_star[0] + R * np.cos(u_vec) * np.sin(v_vec)).flatten()
    ys = (phi_star[1] + R * np.sin(u_vec) * np.sin(v_vec)).flatten()
    zs = (phi_star[2] + R * np.cos(v_vec)).flatten()

    dx, dy, dz = np.zeros_like(xs), np.zeros_like(ys), np.zeros_like(zs)
    
    for i in range(len(xs)):
        dot_phi = oim.phase_dynamics(0, np.array([xs[i], ys[i], zs[i]]))
        dx[i], dy[i], dz[i] = dot_phi

    # Normalize vectors for neat rendering
    norms = np.sqrt(dx**2 + dy**2 + dz**2)
    norms[norms == 0] = 1.0
    dx, dy, dz = dx/norms, dy/norms, dz/norms
    
    arrow_len = R * 0.25
    ax.quiver(xs, ys, zs, dx, dy, dz, length=arrow_len, color='navy', 
          arrow_length_ratio=0.4, alpha=0.6)

    # Formatting
    ax.set_title(f"{title}\nRadius = {R:.4f}", pad=10)
    ax.set_xlabel(r"$\phi_1$"); ax.set_ylabel(r"$\phi_2$"); ax.set_zlabel(r"$\phi_3$")
    
    # Make aspect ratio equal (prevents the sphere from looking like an ellipsoid)
    ax.set_box_aspect((1, 1, 1))

    # Clean limits to center the sphere nicely
    ax.set_xlim([phi_star[0]-1.7, phi_star[0]+1.7])
    ax.set_ylim([phi_star[1]-1.7, phi_star[1]+1.7])
    ax.set_zlim([phi_star[2]-1.7, phi_star[2]+1.7])

plt.tight_layout()
plt.show()
