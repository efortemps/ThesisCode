"""
OIM_SpectralShaping.py
======================
Spectral Shaping (Mechanism C) implementation for MaxCut.

References
----------
Thesis appendix A.3 — Spectral Shaping / Penalty-Free binarization.
Steinerberger, "Max-cut via Kuramoto-type oscillators,"
    SIAM J. Appl. Dyn. Syst. 22(2), 730-743, 2023.
Tong & Muehlebach, "A dynamical systems perspective on discrete optimization,"
    L4DC, PMLR 211, 2023.

Overview
--------
Instead of penalizing non-binary phases via a second-harmonic injection term
(OIM, Mechanism A), binarization is here enforced purely by the *shape* of
the coupling function g_k.  The coupling is a truncated odd Fourier series
in the phase difference; the Fourier order k acts as the annealing parameter.

  k = 1   ->  g_1(phi) = sin(phi)         [standard Kuramoto / OIM at Ks=0]
  k -> inf -> g_k -> square-wave sgn(sin) [hard binary, enforces {0,pi}]

Dynamics (Eq. A.7 of thesis):
    dtheta_i/dt = sum_j J_ij g_k(theta_j - theta_i)
                = -sum_j J_ij g_k(theta_i - theta_j)   [g_k is odd]

Energy (Eq. A.6 of thesis):
    E(theta) = -1/2 * sum_ij J_ij G_k(theta_i - theta_j)

Coupling functions (odd harmonics only):
    g_k(phi) = sum_{n in odd, 1<=n<=k} c_n      sin(n phi)
    G_k(phi) = sum_{n in odd, 1<=n<=k} c_n / n  cos(n phi)   [G_k' = g_k]

Default coefficients c_n = 1  (equal-weight truncated Fourier series).
Alternative: c_n = 4/(n*pi) gives the standard partial-sum Fourier
approximation to the square wave.

Stability at binary equilibria
-------------------------------
At phi* in {0, pi}^N the Jacobian simplifies to

    A(phi*) = alpha_k * D(phi*)

where  D(phi*)  is the same signed-Laplacian used in OIM_Maxcut (Theorem 2 /
Remark 7 of Cheng et al. 2024) and

    alpha_k = sum_{n odd, n<=k} n * c_n   (> 0 for c_n > 0).

Consequently, a binary equilibrium is stable iff lambda_max(D(phi*)) <= 0,
independent of k.  The role of k is therefore NOT to change which binary
equilibria are stable, but to *eliminate* spurious non-binary equilibria that
would otherwise attract trajectories.
"""

from __future__ import annotations
from typing import List, Optional
from itertools import product

import numpy as np
from scipy.integrate import solve_ivp


class OIM_SpectralShaping:
    """
    Oscillator Ising Machine with Spectral Shaping coupling for MaxCut.

    Parameters
    ----------
    J      : np.ndarray, shape (N, N)
             Symmetric coupling matrix. For MaxCut use J[i,j] = -1 per edge.
    k      : int
             Fourier order.  Only odd harmonics 1, 3, 5, ..., <= k are used.
             k=1 recovers standard Kuramoto coupling (g_1 = sin).
             Increase k to sharpen the coupling toward a square wave.
    coeffs : array-like or None
             Fourier coefficients c_n, one per odd harmonic n <= k
             (length = number of odd integers in [1, k]).
             None  ->  c_n = 1 for all harmonics  (default, equal-weight).
             'square_wave'  ->  c_n = 4/(n*pi), Gibbs-optimal approximation.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        J:      np.ndarray,
        k:      int,
        coeffs: Optional[object] = None,
    ) -> None:
        if J.ndim != 2 or J.shape[0] != J.shape[1]:
            raise ValueError("J must be a square 2-D array.")
        self.J = np.asarray(J, dtype=float)
        self.N = J.shape[0]
        self.k = int(k)

        # Odd harmonics: 1, 3, 5, ..., <= k
        self._harmonics = np.array(
            [n for n in range(1, self.k + 1, 2)], dtype=float
        )
        n_harmonics = len(self._harmonics)

        if coeffs is None:
            self._coeffs = np.ones(n_harmonics)
        elif isinstance(coeffs, str) and coeffs == "square_wave":
            # c_n = 4/(n*pi): Fourier series of sgn(sin(phi))
            self._coeffs = 4.0 / (self._harmonics * np.pi)
        else:
            c = np.asarray(coeffs, dtype=float)
            if c.shape != (n_harmonics,):
                raise ValueError(
                    f"coeffs must have length {n_harmonics} "
                    f"(odd harmonics up to k={k}: {self._harmonics.astype(int).tolist()})."
                )
            self._coeffs = c

        # alpha_k = sum_n n * c_n  (scales the Jacobian at binary equilibria)
        self._alpha_k = float((self._harmonics * self._coeffs).sum())

    def __repr__(self) -> str:
        harm = self._harmonics.astype(int).tolist()
        coef = np.round(self._coeffs, 4).tolist()
        return (
            f"OIM_SpectralShaping(N={self.N}, k={self.k})\n"
            f"  harmonics : {harm}\n"
            f"  coeffs    : {coef}\n"
            f"  alpha_k   : {self._alpha_k:.4f}\n"
            f"J =\n{self.J}"
        )

    # ------------------------------------------------------------------
    # Coupling functions  (scalar and vectorised)
    # ------------------------------------------------------------------

    def g_k(self, phi: np.ndarray) -> np.ndarray:
        """
        Sharpened coupling function:
            g_k(phi) = sum_{n odd, n<=k} c_n sin(n phi)

        Properties: odd (g_k(-phi) = -g_k(phi)), zeros at phi = 0 and pi.
        At k=1, c_1=1: g_1 = sin  (standard OIM coupling, Ks=0 limit).
        """
        out = np.zeros_like(phi, dtype=float)
        for n, c in zip(self._harmonics, self._coeffs):
            out += c * np.sin(n * phi)
        return out

    def G_k(self, phi: np.ndarray) -> np.ndarray:
        """
        Energy coupling potential (antiderivative of g_k):
            G_k(phi) = sum_{n odd, n<=k} c_n/n cos(n phi)

        Satisfies  dG_k/dphi = g_k(phi).
        E(theta) = -1/2 sum_ij J_ij G_k(theta_i - theta_j).
        """
        out = np.zeros_like(phi, dtype=float)
        for n, c in zip(self._harmonics, self._coeffs):
            out += (c / n) * np.cos(n * phi)
        return out

    def g_k_prime(self, phi: np.ndarray) -> np.ndarray:
        """
        Derivative of the coupling function:
            g_k'(phi) = sum_{n odd, n<=k} n c_n cos(n phi)

        Used in the Jacobian.
        At binary equilibria (phi in {0, pi, -pi}):
            g_k'(0)  = +alpha_k
            g_k'(pi) = -alpha_k       [since cos(n*pi) = -1 for odd n]
        Hence  g_k'(phi_i* - phi_j*) = alpha_k * cos(phi_i* - phi_j*).
        """
        out = np.zeros_like(phi, dtype=float)
        for n, c in zip(self._harmonics, self._coeffs):
            out += n * c * np.cos(n * phi)
        return out

    # ------------------------------------------------------------------
    # Core physics
    # ------------------------------------------------------------------

    def dynamics(self, t: float, phi: np.ndarray) -> np.ndarray:
        """
        RHS of the ODE (Eq. A.7):
            dphi_i/dt = sum_j J_ij g_k(phi_j - phi_i)
                      = -sum_j J_ij g_k(phi_i - phi_j)   [g_k odd]

        Gradient flow of E: dphi/dt = -grad_phi E.
        """
        diff = phi[:, None] - phi[None, :]          # diff[i,j] = phi_i - phi_j
        return -(self.J * self.g_k(diff)).sum(axis=1)

    def energy(self, phi: np.ndarray) -> float:
        """
        Lyapunov energy E(phi) (Eq. A.6):
            E(phi) = -1/2 sum_ij J_ij G_k(phi_i - phi_j)

        Monotonically non-increasing along trajectories (gradient flow).
        Use ising_hamiltonian() for pure cut-quality evaluation.
        """
        diff = phi[:, None] - phi[None, :]
        return float(-0.5 * (self.J * self.G_k(diff)).sum())

    def ising_hamiltonian(self, phi: np.ndarray) -> float:
        """
        Pure Ising Hamiltonian after rounding phi to the nearest binary state:

            H = -sum_{i<j} J_ij s_i s_j,    s_i in {+1, -1}

        Binarization rule (same as OIM_Maxcut):
            s_i = +1  if  phi_i mod 2pi < pi   (phase near 0)
            s_i = -1  if  phi_i mod 2pi >= pi  (phase near pi)

        Independent of k; directly reflects cut quality.
        Lower value = better MaxCut.
        """
        s = np.where(phi % (2.0 * np.pi) < np.pi, 1.0, -1.0)
        return float(-0.5 * (s @ self.J @ s))

    def jacobian(self, phi: np.ndarray) -> np.ndarray:
        """
        Jacobian of the dynamics at phase configuration phi.

        From dphi_i/dt = -sum_j J_ij g_k(phi_i - phi_j):

            A_ij = J_ij  g_k'(phi_i - phi_j)           for i != j
            A_ii = -sum_{j!=i} J_ij g_k'(phi_i - phi_j)

        The matrix has zero row-sums by construction (like a Laplacian).
        Stable fixed point iff all eigenvalues of A are <= 0.

        At binary equilibria phi* in {0,pi}^N this simplifies to:
            A(phi*) = alpha_k * D(phi*)
        where D is the signed-Laplacian of build_D().
        """
        diff = phi[:, None] - phi[None, :]
        Amat = self.J * self.g_k_prime(diff)   # A[i,j] = J_ij * g_k'(phi_i-phi_j)
        # Diagonal: zero row-sum (J diagonal is 0, so Amat diagonal is 0 before this)
        np.fill_diagonal(Amat, -Amat.sum(axis=1))
        return Amat

    def largest_lyapunov(self, phi: np.ndarray) -> float:
        """
        lambda_L = max real eigenvalue of the Jacobian at phi.
        Stable fixed point iff lambda_L <= 0.
        """
        return float(np.linalg.eigvals(self.jacobian(phi)).real.max())

    # ------------------------------------------------------------------
    # D-matrix and stability analysis at binary equilibria
    # ------------------------------------------------------------------

    def build_D(self, phi_star: np.ndarray) -> np.ndarray:
        """
        Signed-Laplacian D(phi*) — identical in form to OIM_Maxcut.build_D().

            Off-diagonal:  D_ij = J_ij * cos(phi_i* - phi_j*)   (i != j)
            Diagonal:      D_ii = -sum_{j!=i} D_ij

        At a binary equilibrium phi* the Jacobian satisfies:
            A(phi*) = alpha_k * D(phi*)
        where alpha_k = sum_{n odd, n<=k} n * c_n > 0.

        D is graph-structure-dependent (through J) and independent of k.
        """
        diff = phi_star[:, None] - phi_star[None, :]
        D    = self.J * np.cos(diff)
        np.fill_diagonal(D, 0.0)
        np.fill_diagonal(D, -D.sum(axis=1))
        return D

    def stability_at_binary(self, phi_star: np.ndarray) -> dict:
        """
        Stability of a binary equilibrium phi* in {0, pi}^N.

        Since A(phi*) = alpha_k * D(phi*) with alpha_k > 0, the equilibrium
        is stable iff lambda_max(D(phi*)) <= 0.  This is INDEPENDENT of k.

        Parameters
        ----------
        phi_star : array of 0s and pis, shape (N,)

        Returns
        -------
        dict with keys:
            lambda_D   : float  lambda_max(D(phi*))
            is_stable  : bool   True iff lambda_D <= 0
            lambda_A   : float  lambda_max(A(phi*)) = alpha_k * lambda_D
                                (eigenvalue of the actual Jacobian)
        """
        D         = self.build_D(phi_star)
        lambda_D  = float(np.linalg.eigvalsh(D).max())
        lambda_A  = self._alpha_k * lambda_D
        return {
            "lambda_D"  : lambda_D,
            "is_stable" : lambda_D <= 0.0,
            "lambda_A"  : lambda_A,
        }

    def binary_stability_survey(self) -> dict:
        """
        Enumerate all 2^N binary equilibria and classify their stability.

        For Mechanism C, stability of binary equilibria is determined solely
        by the graph structure (via D) and is independent of k.

        Returns
        -------
        dict with keys:
            per_eq       : {label: {"lambda_D", "is_stable", "lambda_A"}}
            stable_eqs   : list of labels that are stable
            unstable_eqs : list of labels that are unstable
            n_stable     : int
            hardest_eq   : label with highest lambda_D  (hardest to stabilize)
            easiest_eq   : label with lowest  lambda_D  (most naturally stable)
        """
        configs = [
            np.array(cfg, dtype=float) * np.pi
            for cfg in product([0, 1], repeat=self.N)
        ]
        per_eq = {}
        for phi in configs:
            label         = "".join("1" if p > 0.5 else "0" for p in phi / np.pi)
            per_eq[label] = self.stability_at_binary(phi)

        stable_eqs   = [lb for lb, v in per_eq.items() if v["is_stable"]]
        unstable_eqs = [lb for lb, v in per_eq.items() if not v["is_stable"]]

        return {
            "per_eq"       : per_eq,
            "stable_eqs"   : stable_eqs,
            "unstable_eqs" : unstable_eqs,
            "n_stable"     : len(stable_eqs),
            "hardest_eq"   : max(per_eq, key=lambda lb: per_eq[lb]["lambda_D"]),
            "easiest_eq"   : min(per_eq, key=lambda lb: per_eq[lb]["lambda_D"]),
        }

    def alpha_k(self) -> float:
        """
        alpha_k = sum_{n odd, n<=k} n * c_n.

        Scales the Jacobian eigenvalues at binary equilibria:
            lambda_A = alpha_k * lambda_D.
        Increases with k (for c_n > 0), making stable equilibria more
        stable and unstable ones more unstable — i.e. sharpening the
        attraction basins as k grows.
        """
        return self._alpha_k

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
            print(f"[OIM_SpectralShaping] Simulation failed: {sol.message}")
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
