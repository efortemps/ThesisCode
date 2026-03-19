"""
OIM_Stability_1.py
==================
Core Oscillator Ising Machine (OIM) implementation for MaxCut.

Reference
---------
Bashar, Lin & Shukla, "Stability of Oscillator Ising Machines:
Not All Solutions Are Created Equal."

The system dynamics follow Eq. (2) of the paper:

    dθ_i/dt = -K Σ_j W_ij sin(θ_i - θ_j) - Ks sin(2θ_i)

and the Lyapunov (energy) function from Eq. (1):

    E(θ) = -K Σ_{i≠j} W_ij cos(θ_i - θ_j) - Ks Σ_i cos(2θ_i)

Stability of a fixed point is determined by the eigenvalues of the
Jacobian matrix (Eq. 3).  A configuration is stable iff all eigenvalues
(Lyapunov exponents) are ≤ 0; we track only the *largest* one λ_L.
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np
from scipy.integrate import solve_ivp


class OIM_Maxcut:
    """
    Oscillator Ising Machine for the MaxCut problem.

    Parameters
    ----------
    J : np.ndarray, shape (N, N)
        Symmetric coupling matrix W.  For MaxCut use J[i,j] = -1 for each
        edge (antiferromagnetic); see ``read_graphs.read_graph_to_J``.
    K : float
        Oscillator-to-oscillator coupling strength.
    Ks : float
        Second-harmonic injection strength.
    """

    def __init__(self, J: np.ndarray, K: float, Ks: float) -> None:
        if J.ndim != 2 or J.shape[0] != J.shape[1]:
            raise ValueError("J must be a square 2-D array.")
        self.J = np.asarray(J, dtype=float)
        self.K = float(K)
        self.Ks = float(Ks)
        self.N = J.shape[0]

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"OIM_Maxcut(N={self.N}, K={self.K}, Ks={self.Ks})\n"
            f"J =\n{self.J}"
        )

    # ------------------------------------------------------------------
    # Core physics  (fully vectorised — no Python-level loops)
    # ------------------------------------------------------------------

    def dynamics(self, t: float, phi: np.ndarray) -> np.ndarray:
        """
        RHS of the ODE: dφ/dt.

        Parameters
        ----------
        t   : float        — time (unused; required by solve_ivp API)
        phi : (N,) ndarray — current oscillator phases

        Returns
        -------
        dphi_dt : (N,) ndarray
        """
        # diff[i,j] = φ_i - φ_j
        diff = phi[:, None] - phi[None, :]           # (N, N)
        # Sum over j: Σ_j W_ij sin(φ_i − φ_j)
        coupling = (self.J * np.sin(diff)).sum(axis=1)  # (N,)
        return -self.K * coupling - self.Ks * np.sin(2.0 * phi)

    def energy(self, phi: np.ndarray) -> float:
        """
        Lyapunov / energy function E(φ).

        Parameters
        ----------
        phi : (N,) ndarray — oscillator phases (need not be binarised)

        Returns
        -------
        E : float
        """
        diff = phi[:, None] - phi[None, :]
        # Double-sum counts each pair twice; divide by 2.
        E_coupling = -self.K * (self.J * np.cos(diff)).sum() / 2.0
        E_injection = -self.Ks * np.cos(2.0 * phi).sum()
        return float(E_coupling + E_injection)

    # Keep old name for backwards compatibility with run_experiment.py
    def energy_dynamics(self, phi: np.ndarray) -> float:
        return self.energy(phi)

    def jacobian(self, phi: np.ndarray) -> np.ndarray:
        """
        Jacobian matrix at phase configuration *phi* (Eq. 3 of the paper).

        J_mat[i,j] = -K W_ij cos(φ_i − φ_j)          for i ≠ j
        J_mat[i,i] = -K Σ_{j≠i} W_ij cos(φ_i − φ_j)
                     − 2 Ks cos(2φ_i)

        Parameters
        ----------
        phi : (N,) ndarray

        Returns
        -------
        Jmat : (N, N) ndarray  (real, symmetric when J is symmetric)
        """
        diff = phi[:, None] - phi[None, :]    # (N, N)
        cos_diff = np.cos(diff)               # (N, N)

        # Build off-diagonal part: -K * W * cos(diff); diagonal = 0 here
        # because W[i,i] = 0 (no self-loops).
        Jmat = -self.K * self.J * cos_diff    # (N, N)

        # Diagonal: row-sum of off-diagonal entries − 2Ks cos(2φ_i)
        # Row sum already equals the j≠i sum when W[i,i]=0.
        diag = Jmat.sum(axis=1) - 2.0 * self.Ks * np.cos(2.0 * phi)
        np.fill_diagonal(Jmat, diag)

        return Jmat

    def largest_lyapunov(self, phi: np.ndarray) -> float:
        """
        Largest Lyapunov exponent λ_L = max eigenvalue of the Jacobian.

        A configuration is *stable* iff λ_L ≤ 0.

        Parameters
        ----------
        phi : (N,) ndarray — phase configuration (typically binarised to {0,π})

        Returns
        -------
        lambda_L : float
        """
        Jmat = self.jacobian(phi)
        eigvals = np.linalg.eigvals(Jmat)
        return float(eigvals.real.max())

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        phi0: np.ndarray,
        t_span: tuple[float, float],
        n_points: int,
        method: str = "RK45",
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ) -> Optional[object]:
        """
        Integrate the ODE from initial condition *phi0*.

        Returns
        -------
        sol : scipy OdeResult or None if integration failed.
        """
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(
            self.dynamics,
            t_span,
            phi0,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            print(f"[OIM] Simulation failed: {sol.message}")
            return None
        return sol

    def simulate_many(
        self,
        phi0_list: List[np.ndarray],
        t_span: tuple[float, float],
        n_points: int,
    ) -> List[Optional[object]]:
        """Run ``simulate`` for each initial condition in *phi0_list*."""
        return [self.simulate(phi0, t_span, n_points) for phi0 in phi0_list]
