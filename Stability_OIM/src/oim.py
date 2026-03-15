import numpy as np
from scipy.integrate import solve_ivp
from itertools import product
from typing import List, Optional


class OscillatorIsingMachine:
    """
    Oscillator Ising Machine (OIM) analysis and simulation framework.

    Parameters
    ----------
    J   : (N, N) symmetric coupling matrix, J_ij in {-1, 0, 1}, zero diagonal.
    K   : Coupling strength (positive scalar).
    K_s : SYNC (second-harmonic injection locking) strength (non-negative scalar).
    """

    def __init__(self, J: np.ndarray, K: float, K_s: float):
        J = np.asarray(J, dtype=float)
        assert J.ndim == 2 and J.shape[0] == J.shape[1], "J must be square"
        assert np.allclose(J, J.T, atol=1e-10),          "J must be symmetric"
        assert np.allclose(np.diag(J), 0, atol=1e-10),   "J must have zero diagonal"
        assert K > 0 and K_s >= 0,                        "K must be >0, K_s >= 0"

        self.J   = J
        self.N   = J.shape[0]
        self.K   = float(K)
        self.K_s = float(K_s)

    def __repr__(self):
        return (f"OscillatorIsingMachine(N={self.N}, K={self.K}, K_s={self.K_s})\n"
                f"J =\n{self.J}")

    # ------------------------------------------------------------------
    #  Energy & dynamics
    # ------------------------------------------------------------------

    def energy(self, phi: np.ndarray) -> float:
        """
        E(phi) = -K sum_{i<j} J_ij cos(phi_i-phi_j) - K_s sum_i cos(2 phi_i)
        """
        phi      = np.asarray(phi, dtype=float)
        diff     = phi[:, None] - phi[None, :]          # diff[i,j] = phi_i - phi_j
        coupling = np.sum(self.J * np.cos(diff)) 
        sync     = np.sum(np.cos(2.0 * phi))
        return float(-self.K * coupling - self.K_s * sync)

    def phase_dynamics(self, t, phi: np.ndarray) -> np.ndarray:
        """
        phi_dot_i = -K sum_{j≠i} J_ij sin(phi_i - phi_j) - K_s sin(2 phi_i)

        """
        phi  = np.asarray(phi, dtype=float)
        diff = phi[:, None] - phi[None, :]
        return ( -self.K   * np.sum(self.J * np.sin(diff), axis=1)
                 -self.K_s * np.sin(2.0 * phi) )

    def dE_dt(self, phi: np.ndarray) -> float:
        """
        Rate of change of energy along a trajectory: dE/dt = -||f(phi)||^2 <= 0.
        Confirms global descent property regardless of equilibrium stability.
        """
        dphi = self.phase_dynamics(None, phi)
        return float(-0.5*np.dot(dphi, dphi))

    # ------------------------------------------------------------------
    #  Structural matrices
    # ------------------------------------------------------------------

    def build_D(self, phi_star: np.ndarray) -> np.ndarray:
        """
        Signed-Laplacian matrix D(phi*) from Theorem 2, equation (5).

            Off-diagonal:  D_ij  =  J_ij * cos(phi_i* - phi_j*)   (i != j)
            Diagonal:      D_ii  = -sum_{j!=i} D_ij

        Properties:
          - D is symmetric (since J is symmetric and cos is even).
          - D * 1_N = 0  (zero row sum), so 0 is always an eigenvalue of D.
          - lambda_max(D) determines the K_s stability threshold (Theorem 2).
        """
        phi_star = np.asarray(phi_star, dtype=float)
        diff = phi_star[:, None] - phi_star[None, :]
        D    = self.J * np.cos(diff)
        np.fill_diagonal(D, 0.0)
        np.fill_diagonal(D, -D.sum(axis=1))
        return D

    def build_D_bar(self, phi_star: np.ndarray) -> np.ndarray:
        """
        Modified Laplacian D_bar(phi*) for Type III equilibria (Theorem 4).

            D_bar = D(phi*) + delta * diag(varphi_i)
            varphi_i = -sin^2(phi_i*)
            delta    = -4 K_s / K      (derived via equilibrium condition, Remark 6)

        Key derivation:
            cos(2x) = 1 - 2sin^2(x)                          (trig identity)
            A(phi*) = K D - 2Ks diag(cos(2phi_i*))
                    = K D - 2Ks diag(1 - 2sin^2(phi_i*))
                    = K D - 2Ks I + 4Ks diag(sin^2(phi_i*))
                    = K [D + (-4Ks/K) diag(-sin^2(phi_i*))]  - 2Ks I
                    = K D_bar                                 - 2Ks I

        So the Jacobian has the same form as in the Type I case, allowing the
        same eigenvalue stability criterion to be applied (Theorem 4 proof).
        """
        phi_star = np.asarray(phi_star, dtype=float)
        D     = self.build_D(phi_star)
        delta = -4.0 * self.K_s / self.K
        varphi_i = -np.sin(phi_star) ** 2
        return D + delta * np.diag(varphi_i)

    def jacobian(self, phi_star: np.ndarray) -> np.ndarray:
        """
        Jacobian A(phi*) of the vector field f at equilibrium phi*.
        Valid for all equilibrium types (I, II, III).

            A(phi*) = K D(phi*) - 2 K_s diag(cos(2 phi_i*))

        Derivation from f_i(phi) = -K sum_{j!=i} J_ij sin(phi_i-phi_j) - Ks sin(2phi_i):
            df_i/dphi_j  =  K J_ij cos(phi_i* - phi_j*)        (i != j)
            df_i/dphi_i  = -K sum_{j!=i} J_ij cos(phi_i*-phi_j*) - 2Ks cos(2phi_i*)
                         = K D_ii - 2Ks cos(2phi_i*)
        """
        phi_star = np.asarray(phi_star, dtype=float)
        D        = self.build_D(phi_star)
        diag_cos = np.diag(np.cos(2.0 * phi_star))
        return self.K * D - 2.0 * self.K_s * diag_cos

    # ------------------------------------------------------------------
    #  Equilibrium classification  (Definitions 1–3, Theorem 1)
    # ------------------------------------------------------------------

    def classify_equilibrium(self, phi_star: np.ndarray, tol: float = 1e-7) -> str:
        """
        Classify phi* as one of:
            "Type I (M2)" : all phi_i* in {0, pi}    (mod 2pi) → cos(2phi_i*) = +1
            "Type I (M1)" : all phi_i* in {±pi/2}    (mod 2pi) → cos(2phi_i*) = -1
            "Type II"     : all phi_i* in {k*pi/2} but mixed M1/M2 (structurally unstable)
            "Type III"    : at least one phi_i* not in {k*pi/2}

        Logic (from Theorem 1):
          Step 1: Check sin(2phi_i*) ≈ 0 for all i.  If not → Type III.
          Step 2: Check the sign of cos(2phi_i*) for all i.
                  All +1 → Type I M2   (structurally stable, may be stable/unstable)
                  All -1 → Type I M1   (always unstable, Theorem 2)
                  Mixed  → Type II     (always unstable, Theorem 3)
        """
        phi_star = np.asarray(phi_star, dtype=float)

        if np.any(np.abs(np.sin(2.0 * phi_star)) > tol):
            return "Type III"

        # ── Step 2: Type I vs Type II ─────────────────────────────────
        diff    = phi_star[:, None] - phi_star[None, :]    # diff[i,j] = phi_i*-phi_j*
        offdiag = ~np.eye(self.N, dtype=bool)
        eq4_all_J = np.all(np.abs(np.sin(diff[offdiag])) < tol)

        if not eq4_all_J:
            return "Type II"
        
        # ── Step 3: M1 / M2 ──────────────────────────────────────────
        cos2 = np.cos(2.0 * phi_star)
        if np.all(cos2 >  1.0 - tol): return "Type I (M2)"
        if np.all(cos2 < -1.0 + tol): return "Type I (M1)"

        return "Type I (mixed — check tolerance)" # unlikely due to Theorem 1, but possible if tol is too large

    def is_equilibrium(self, phi_star: np.ndarray, tol: float = 1e-7) -> bool:
        """Return True if ||f(phi*)|| < tol  (phi* is approximately a fixed point)."""
        return float(np.linalg.norm(self.phase_dynamics(None,phi_star))) < tol

    # ------------------------------------------------------------------
    #  Stability analysis
    # ------------------------------------------------------------------

    def stability_analysis(self, phi_star: np.ndarray,
                           verbose: bool = True) -> dict:
        """
        Full stability analysis at equilibrium phi* via Lemma 1 (linearisation).

        Steps:
            1. Compute D(phi*) and its eigenvalues; record lambda_max(D).
            2. Compute Jacobian A(phi*).
            3. Examine real parts of eigenvalues of A (Lemma 1 / Hartman–Grobman):
                 All Re(lambda_i) < 0  →  asymptotically stable
                 Any Re(lambda_i) > 0  →  unstable
                 All Re(lambda_i) <= 0 →  inconclusive (centre manifold needed)
            4. Compute K_s stability threshold from Theorem 2:
                 K_s* = K * lambda_max(D) / 2

        Returns
        -------
        dict with keys:
            phi_star, energy, eq_type, D, eigenvalues_D, lambda_max_D,
            K_s_threshold, A, eigenvalues_A, real_parts_A, verdict, is_stable
        """
        phi_star = np.asarray(phi_star, dtype=float)

        eq_type       = self.classify_equilibrium(phi_star)
        E_val         = self.energy(phi_star)


        D             = self.build_D(phi_star)
        eigvals_D     = np.linalg.eigvalsh(D)
        lambda_max_D  = float(eigvals_D.max())
        threshold = self.K * lambda_max_D / 2.0

        A          = self.jacobian(phi_star)
        eigvals_A  = np.linalg.eigvals(A)
        
        max_re_eig_A = float(np.real(eigvals_A).max())

        if   np.all(max_re_eig_A < -1e-10): verdict, is_stable = "Asymptotically stable",          True
        elif np.any(max_re_eig_A >  1e-10): verdict, is_stable = "Unstable",                        False
        else:                             verdict, is_stable = "Critical (centre manifold needed)", None

        result = dict(
            phi_star     = phi_star,
            energy       = E_val,
            eq_type      = eq_type,
            D            = D,
            eigenvalues_D= eigvals_D,
            lambda_max_D = lambda_max_D,
            threshold    = threshold,
            A            = A,
            eigenvalues_A= eigvals_A,
            max_re_eig_A = max_re_eig_A,
            verdict      = verdict,
            is_stable    = is_stable,
        )

        if verbose:
            self._print_stability_report(result)
        return result

    def _print_stability_report(self, r: dict) -> None:
        phi_pi = np.round(r["phi_star"] / np.pi, 4)
        sep    = "─" * 58
        print(sep)
        print(f"  phi*                    = {phi_pi} x pi")
        print(f"  Type                    = {r['eq_type']}")
        print(f"  E(phi*)                 = {r['energy']:.6f}")
        print(f"  lambda_max(D(phi*))     = {r['lambda_max_D']:.6f}")
        print(f"  K * lambda_max(D) / 2   = {r['threshold']:.6f}")
        print(f"  max Re(eig(A))          = {r['max_re_eig_A']:.6f}")
        print(f"  Verdict  -->  {r['verdict']}")
        print(sep)


    def stability_threshold(self, phi_star: np.ndarray) -> float:
        """
        Compute the stability threshold for a single equilibrium phi*.

        From Theorem 2 (screenshot):

            threshold(phi*) = K * lambda_N(D(phi*)) / 2

        where lambda_N denotes the maximum eigenvalue of the symmetric
        signed-Laplacian D(phi*).  Stability criterion for phi* in M2:

            K_s > threshold(phi*)  ->  asymptotically stable  (Lemma 1)
            K_s < threshold(phi*)  ->  unstable               (Lemma 1)

        No comparison to self.K_s is made here.  Typical usage:

            thresholds = [oim.stability_threshold(phi) for phi in equilibria]
            # then pass thresholds to your plotting routine

        Parameters
        ----------
        phi_star : equilibrium phase vector (N,)

        Returns
        -------
        float : K * lambda_max(D(phi*)) / 2
        """
        phi_star     = np.asarray(phi_star, dtype=float)
        D            = self.build_D(phi_star)
        lambda_max_D = float(np.linalg.eigvalsh(D).max())
        return self.K * lambda_max_D / 2.0

    # ------------------------------------------------------------------
    #  Simulation
    # ------------------------------------------------------------------

    def simulate(self,
                 phi0:     np.ndarray,
                 t_span:   tuple,
                 n_points: int = 2000,
                 **ode_kwargs) -> object:
        """
        Integrate the OIM ODE from initial condition phi0 using RK45.

        Parameters
        ----------
        phi0     : Initial phase vector (N,).
        t_span   : (t0, tf) integration interval.
        n_points : Number of time steps at which to record the solution.
        **ode_kwargs : Passed directly to scipy.integrate.solve_ivp.

        Returns
        -------
        sol : scipy OdeSolution object, extended with:
              sol.energy  (ndarray) — energy E(phi(t)) along the trajectory.
        """
        phi0   = np.asarray(phi0, dtype=float)
        assert len(phi0) == self.N, f"phi0 must have length N={self.N}"
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol    = solve_ivp(
            self.phase_dynamics, t_span, phi0,
            t_eval=t_eval, method="RK45",
            rtol=1e-9, atol=1e-12, dense_output=False,
            **ode_kwargs
        )
        sol.energy = np.array([self.energy(sol.y[:, k])
                               for k in range(sol.y.shape[1])])
        return sol

    def simulate_many(self,
                      phi0_list: List[np.ndarray],
                      t_span:    tuple,
                      n_points:  int = 500) -> List:
        """
        Simulate from multiple initial conditions.

        Returns
        -------
        List of sol objects (one per initial condition).
        """
        return [self.simulate(phi0, t_span, n_points) for phi0 in phi0_list]