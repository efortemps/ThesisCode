"""
OIM_Stability_1.py
==================
Core Oscillator Ising Machine (OIM) implementation for MaxCut.

Reference
---------
Bashar, Lin & Shukla, "Stability of Oscillator Ising Machines:
Not All Solutions Are Created Equal."

Dynamics (Eq. 2):
    dtheta_i/dt = -K sum_j W_ij sin(theta_i - theta_j) - Ks sin(2*theta_i)

Lyapunov / energy (Eq. 1):
    E(theta) = -K sum_{i!=j} W_ij cos(theta_i - theta_j) - Ks sum_i cos(2*theta_i)

Stability at a fixed point is determined by the eigenvalues of the Jacobian
(Eq. 3). A configuration is stable iff all eigenvalues are <= 0.

D-matrix (Theorem 2 / Remark 7):
    D(phi*)_ij = J_ij cos(phi_i* - phi_j*)     i != j
    D(phi*)_ii = -sum_{j!=i} D_ij
    Stability threshold for phi*: Ks > K * lambda_max(D(phi*)) / 2
    Binarisation threshold (Remark 7): Ks* = min over all Type-I M2 equilibria
"""

from __future__ import annotations
from typing import List, Optional
from itertools import product

import numpy as np
from scipy.integrate import solve_ivp


class OIM_Maxcut:
    """
    Oscillator Ising Machine for the MaxCut problem.

    Parameters
    ----------
    J  : np.ndarray, shape (N, N)
         Symmetric coupling matrix. For MaxCut use J[i,j] = -1 for each edge.
    K  : float   oscillator-to-oscillator coupling strength
    Ks : float   second-harmonic injection strength
    """

    def __init__(self, J: np.ndarray, K: float, Ks: float) -> None:
        if J.ndim != 2 or J.shape[0] != J.shape[1]:
            raise ValueError("J must be a square 2-D array.")
        self.J  = np.asarray(J, dtype=float)
        self.K  = float(K)
        self.Ks = float(Ks)
        self.N  = J.shape[0]

    def __repr__(self) -> str:
        return (
            f"OIM_Maxcut(N={self.N}, K={self.K}, Ks={self.Ks})\n"
            f"J =\n{self.J}"
        )

    # ------------------------------------------------------------------
    # Core physics
    # ------------------------------------------------------------------

    def dynamics(self, t: float, phi: np.ndarray) -> np.ndarray:
        """RHS of the ODE: dphi/dt  (Eq. 2)."""
        diff     = phi[:, None] - phi[None, :]
        coupling = (self.J * np.sin(diff)).sum(axis=1)
        return -self.K * coupling - self.Ks * np.sin(2.0 * phi)

    def energy(self, phi: np.ndarray) -> float:
        """
        Full OIM Lyapunov energy E(phi)  (Eq. 1).

        Includes SYNC term — depends on Ks.
        Use ising_hamiltonian() to evaluate pure cut quality.
        """
        diff        = phi[:, None] - phi[None, :]
        E_coupling  = -self.K  * (self.J * np.cos(diff)).sum()
        E_injection = -self.Ks * np.cos(2.0 * phi).sum()
        return float(E_coupling + E_injection)

    def energy_dynamics(self, phi: np.ndarray) -> float:
        """Alias kept for backwards compatibility."""
        return self.energy(phi)

    def ising_hamiltonian(self, phi: np.ndarray) -> float:
        """
        Pure Ising Hamiltonian after rounding phi to the nearest binary state:

            H = -sum_{i<j} J_ij * s_i * s_j,    s_i in {+1, -1}

        where  s_i = +1  if  phi_i mod 2pi < pi   (phase near 0)
               s_i = -1  if  phi_i mod 2pi >= pi  (phase near pi)

        This is INDEPENDENT of K and Ks and directly reflects cut quality.
        A lower value = better cut.
        """
        phi_bin = np.where(phi % (2.0 * np.pi) < np.pi, 0.0, np.pi)
        s       = np.where(phi_bin < 0.5, 1.0, -1.0)   # 0 -> +1,  pi -> -1
        # s^T J s = 2 * sum_{i<j} J_ij s_i s_j  (J symmetric, zero diagonal)
        return float(-0.5 * (s @ self.J @ s))

    def jacobian(self, phi: np.ndarray) -> np.ndarray:
        """
        Jacobian matrix at phase configuration phi  (Eq. 3).

            A_ij = -K J_ij cos(phi_i - phi_j)                        i != j
            A_ii = -K sum_{j!=i} J_ij cos(phi_i - phi_j) - 2Ks cos(2 phi_i)
        """
        diff = phi[:, None] - phi[None, :]
        Jmat = -self.K * self.J * np.cos(diff)
        diag = Jmat.sum(axis=1) - 2.0 * self.Ks * np.cos(2.0 * phi)
        np.fill_diagonal(Jmat, diag)
        return Jmat

    def largest_lyapunov(self, phi: np.ndarray) -> float:
        """lambda_L = max real eigenvalue of the Jacobian. Stable iff <= 0."""
        return float(np.linalg.eigvals(self.jacobian(phi)).real.max())

    # ------------------------------------------------------------------
    # D-matrix analysis  (Theorem 2 / Remark 7 — Cheng et al. 2024)
    # ------------------------------------------------------------------

    def build_D(self, phi_star: np.ndarray) -> np.ndarray:
        """
        Signed-Laplacian D(phi*) from Theorem 2, eq. (5).

            Off-diagonal:  D_ij = J_ij * cos(phi_i* - phi_j*)   (i != j)
            Diagonal:      D_ii = -sum_{j!=i} D_ij

        D is real symmetric; lambda_max(D) governs the stability threshold.
        """
        diff = phi_star[:, None] - phi_star[None, :]
        D    = self.J * np.cos(diff)
        np.fill_diagonal(D, 0.0)
        np.fill_diagonal(D, -D.sum(axis=1))
        return D

    def stability_threshold(self, phi_star: np.ndarray) -> float:
        """
        Per-equilibrium stability threshold (Theorem 2):

            threshold(phi*) = K * lambda_max(D(phi*)) / 2

            Ks > threshold  ->  phi* is asymptotically stable
            Ks < threshold  ->  phi* is unstable
        """
        return self.K * float(np.linalg.eigvalsh(self.build_D(phi_star)).max()) / 2.0

    def binarization_threshold(self) -> dict:
        """
        Global binarisation threshold from Remark 7:

            Ks* = min_{phi* in {0,pi}^N} stability_threshold(phi*)

        Iterates over all 2^N Type-I M2 equilibria.
        Individual negative thresholds (always-stable equilibria) are
        clamped to 0 before the global minimum is taken.

        Returns
        -------
        dict with keys:
            Ks_star    : float   global binarisation threshold (>= 0)
            mu_star    : float   = 2 * Ks_star
            per_eq     : {label: threshold}  for every binary equilibrium
            easiest_eq : label of equilibrium with lowest  threshold
            hardest_eq : label of equilibrium with highest threshold
        """
        configs = [
            np.array(cfg, dtype=float) * np.pi
            for cfg in product([0, 1], repeat=self.N)
        ]
        per_eq = {}
        for phi in configs:
            label         = "".join("1" if p > 0.5 else "0" for p in phi / np.pi)
            per_eq[label] = self.stability_threshold(phi)

        Ks_star = max(0.0, min(per_eq.values()))
        return {
            "Ks_star"    : Ks_star,
            "mu_star"    : 2.0 * Ks_star,
            "per_eq"     : per_eq,
            "easiest_eq" : min(per_eq, key=per_eq.get),
            "hardest_eq" : max(per_eq, key=per_eq.get),
        }

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        phi0:     np.ndarray,
        t_span:   tuple,
        n_points: int,
        method:   str   = "RK45",
        rtol:     float = 1e-6,
        atol:     float = 1e-8,
    ) -> Optional[object]:
        """Integrate the ODE from phi0. Returns scipy OdeResult or None."""
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol    = solve_ivp(
            self.dynamics, t_span, phi0,
            t_eval=t_eval, method=method, rtol=rtol, atol=atol,
        )
        if not sol.success:
            print(f"[OIM] Simulation failed: {sol.message}")
            return None
        return sol

    def simulate_many(
        self,
        phi0_list: List[np.ndarray],
        t_span:    tuple,
        n_points:  int,
    ) -> List[Optional[object]]:
        """Run simulate() for each initial condition in phi0_list."""
        return [self.simulate(phi0, t_span, n_points) for phi0 in phi0_list]