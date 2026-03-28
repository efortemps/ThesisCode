import numpy as np
import random
from scipy.integrate import solve_ivp
from itertools import product as iproduct
from typing import List, Optional


class OIMMaxCut:
    """
    Oscillator Ising Machine for the Max-Cut problem.

    Uses the mu-parametrisation where

        mu = 2 * Ks / K   (equivalently, K = 1 and Ks = mu/2)

    so that a single scalar mu controls both the coupling strength
    and the SHIL (second-harmonic injection locking) strength.

    The phase dynamics are:

        d(theta_i)/dt = sum_{j!=i} W_ij sin(theta_i - theta_j) - (mu/2) sin(2*theta_i)

    where W_ij = a_ij >= 0 are the original graph weights (positive).
    This is equivalent to the oim.py convention with J = -W, K = 1, Ks = mu/2.

    Parameters
    ----------
    weight_matrix : symmetric N x N matrix with non-negative entries.
                    W_ij = a_ij  (raw edge weight, NOT negated).
    mu            : SHIL strength parameter (mu > 0 required for binarisation).
                    Binarisation is guaranteed when mu > binarization_threshold().
    seed          : RNG seed for reproducibility.
    init_mode     : 'small_random' — phases initialised as tiny noise around 0.
    """

    def __init__(self, weight_matrix, mu: float = 2.0,
                 seed: int = 42, init_mode: str = "small_random"):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.W         = np.array(weight_matrix, dtype=float)
        self.n         = self.W.shape[0]
        self.mu        = float(mu)
        self.timestep  = 1e-2
        self.init_mode = init_mode
        self.theta     = self._init_phases()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_phases(self) -> np.ndarray:
        """Initial phase vector (small noise around 0)."""
        if self.init_mode == "small_random":
            return np.random.uniform(-1e-2, 1e-2, self.n)
        raise ValueError(f"Unknown init_mode '{self.init_mode}'.")

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def _phase_rhs(self, t, theta: np.ndarray) -> np.ndarray:
        """
        Right-hand side of the ODE — compatible with scipy.integrate.solve_ivp.

        d(theta_i)/dt = sum_j W_ij sin(theta_i - theta_j) - (mu/2) sin(2*theta_i)

        In oim.py notation (J = -W, K = 1, Ks = mu/2):
            = -K sum_j J_ij sin(theta_i - theta_j) - Ks sin(2*theta_i)
        """
        diff          = theta[:, None] - theta[None, :]
        coupling_term = np.sum(self.W * np.sin(diff), axis=1)
        shil_term     = (self.mu / 2.0) * np.sin(2.0 * theta)
        return coupling_term - shil_term

    # ------------------------------------------------------------------
    # SciPy integration
    # ------------------------------------------------------------------

    def simulate(self, phi0: Optional[np.ndarray] = None,
                 t_span: tuple = (0., 50.),
                 n_points: int = 500,
                 rtol: float = 1e-9,
                 atol: float = 1e-9):
        """
        Integrate the phase dynamics with SciPy RK45.

        Parameters
        ----------
        phi0       : Initial condition (N,). If None, uses self.theta.
        t_span     : (t_start, t_end) integration window.
        n_points   : Number of time points stored in the output.
        rtol, atol : Solver tolerances.

        Returns
        -------
        sol : scipy OdeSolution  (sol.t = times, sol.y = phases (N, n_points))

        Side-effect: self.theta updated to sol.y[:, -1].
        """
        if phi0 is None:
            phi0 = self.theta.copy()
        phi0   = np.asarray(phi0, dtype=float)
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(
            self._phase_rhs, t_span, phi0,
            method="RK45", t_eval=t_eval,
            rtol=rtol, atol=atol, dense_output=False
        )
        self.theta = sol.y[:, -1].copy()
        return sol

    def simulate_many(self, phi0_list: List[np.ndarray],
                      t_span: tuple = (0., 50.),
                      n_points: int = 500,
                      rtol: float = 1e-9,
                      atol: float = 1e-9) -> List:
        """
        Run simulate() from multiple initial conditions.
        self.theta is restored after the call (non-destructive).

        Returns list of scipy OdeSolution objects.
        """
        theta_backup = self.theta.copy()
        solutions = [
            self.simulate(phi0, t_span=t_span, n_points=n_points,
                          rtol=rtol, atol=atol)
            for phi0 in phi0_list
        ]
        self.theta = theta_backup
        return solutions

    # ------------------------------------------------------------------
    # Structural matrices  (paper notation — Theorem 2, Eq. 5)
    # ------------------------------------------------------------------

    def build_D(self, phi_star: np.ndarray) -> np.ndarray:
        """
        Signed Laplacian D(phi*) — equation (5) of the paper.

            D_ij = J_ij * cos(phi_i* - phi_j*) = -W_ij * cos(phi_i* - phi_j*)  (i != j)
            D_ii = -sum_{j!=i} D_ij   (zero row-sum)

        lambda_max(D) is the per-equilibrium stability threshold (Theorem 2).
        """
        phi_star = np.asarray(phi_star, dtype=float)
        diff     = phi_star[:, None] - phi_star[None, :]
        D        = -self.W * np.cos(diff)
        np.fill_diagonal(D, 0.0)
        np.fill_diagonal(D, -D.sum(axis=1))
        # print(f"[OIM] Building D(phi*) for phi* = {phi_star}")
        # print(f"[OIM] D(phi*) =\n{D}")
        return D

    def jacobian(self, phi_star: np.ndarray) -> np.ndarray:
        """
        Jacobian A(phi*) of the dynamics at equilibrium phi*.

            A = D(phi*) - mu * diag(cos(2*phi_i*))   (K=1, Ks=mu/2)

        A = -Hessian(L). Eigenvalues of A determine local stability (Lemma 1).
        """
        phi_star = np.asarray(phi_star, dtype=float)
        D        = self.build_D(phi_star)
        return D - self.mu * np.diag(np.cos(2.0 * phi_star))

    # ------------------------------------------------------------------
    # Stability threshold  (per equilibrium)
    # ------------------------------------------------------------------

    def stability_threshold(self, phi_star: np.ndarray) -> float:
        """
        Per-equilibrium stability threshold  mu*_i  (Theorem 2).

        Definition
        ----------
            mu*_i = lambda_max( D(phi*_i) )

        Meaning
        -------
        The equilibrium phi*_i is:
          - Asymptotically STABLE  if  mu  > mu*_i
          - UNSTABLE               if  mu  < mu*_i

        This is the threshold for ONE specific equilibrium phi*_i.
        To know whether the whole system binarises, use binarization_threshold().

        Relationship to oim.py (K=1):
            mu*_i = 2 * Ks*_i   where  Ks*_i = K * lambda_max(D) / 2
        """
        D    = self.build_D(phi_star)
        lmax = float(np.linalg.eigvalsh(D).max())
        return lmax

    # ------------------------------------------------------------------
    # Binarisation threshold  (global, over all equilibria)
    # ------------------------------------------------------------------

    def binarization_threshold(self) -> dict:
        """
        Global binarisation threshold  mu_bin  (Remark 7 / screenshot formula).

        Definition  (from the paper)
        ----------------------------
            mu_bin = min_{phi*}  lambda_max( D(phi*) )

        where the minimum is taken over ALL 2^N Type-I M2 equilibria
        (every binary assignment of N spins to {0, pi}).

        Meaning
        -------
          mu > mu_bin  =>  at least one Type-I M2 equilibrium is
                           asymptotically stable  =>  the system BINARISES.
          mu < mu_bin  =>  NO Type-I M2 equilibrium is stable  =>
                           the system does NOT binarise.

        This is the KEY difference from stability_threshold(phi*):
          - stability_threshold(phi*)  answers: "is THIS specific equilibrium stable?"
          - binarization_threshold()   answers: "will the system binarise AT ALL?"

        Returns
        -------
        dict with keys:
            mu_bin          : float — the global binarisation threshold
            Ks_bin          : float — equivalent Ks threshold (= mu_bin / 2, K=1)
            all_thresholds  : dict  — {spin_pattern_str: mu*_i} for every equilibrium
            easiest_eq      : list  — spin pattern (bits) of the easiest-to-stabilise eq
            hardest_eq      : list  — spin pattern (bits) of the hardest-to-stabilise eq
        """
        all_bits   = list(iproduct([0, 1], repeat=self.n))
        results    = {}
        for bits in all_bits:
            phi_star = np.array([b * np.pi for b in bits])
            key      = str(list(bits))
            results[key] = self.stability_threshold(phi_star)

        mu_bin     = min(results.values())
        easiest_key = min(results.keys(), key=lambda k: results[k])
        hardest_key = max(results.keys(), key=lambda k: results[k])

        return {
            "mu_bin"        : float(mu_bin),
            "Ks_bin"        : float(mu_bin / 2.0),
            "all_thresholds": results,
            "easiest_eq"    : easiest_key,
            "hardest_eq"    : hardest_key,
        }

    # ------------------------------------------------------------------
    # Gradient of L  (= -d(theta)/dt)
    # ------------------------------------------------------------------

    def get_hamiltonian(self, theta: Optional[np.ndarray] = None) -> float:
        """
        Ising Hamiltonian H(sigma) evaluated at the binarised spin configuration.

        Definition
        ----------
            H = sum_{i<j} W_ij * sigma_i * sigma_j
              = (1/2) * sigma^T W sigma

        This equals the Lyapunov energy L = get_energy() evaluated at a
        perfectly binarised state (where all sin^2(theta_i) = 0).

        Properties
        ----------
        - The OIM MINIMISES H  <=>  MAXIMISES the Max-Cut value.
        - Exact relationship:   H = W_total - 2 * cut_value
          where  W_total = sum_{i<j} W_ij  =  get_w_total().
          => lower (more negative) H means a BETTER (higher) cut.
        - Minimum possible H  = W_total - 2 * OPT_CUT  (OPT_CUT = optimum).

        Parameters
        ----------
        theta : optional phase vector (N,). If None, uses self.theta.
                Allows evaluating H without mutating the object state.

        Returns
        -------
        float : Hamiltonian value H(sigma(theta)).
        """
        if theta is not None:
            _theta_save = self.theta.copy()
            self.theta  = np.asarray(theta, dtype=float)
        sigma = self.get_spins()
        H     = 0.5 * float(np.sum(self.W * (sigma[:, None] * sigma[None, :])))
        if theta is not None:
            self.theta = _theta_save
        return H

    def get_w_total(self) -> float:
        """Sum of all edge weights: W_total = sum_{i<j} W_ij = np.sum(W) / 2."""
        return float(np.sum(self.W) / 2.0)

    def get_gradient(self) -> np.ndarray:
        """grad L(theta) = -d(theta)/dt evaluated at self.theta."""
        return -self._phase_rhs(None, self.theta)

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def get_energy(self) -> float:
        """
        Lyapunov energy  L(theta; W, mu).

        L = sum_{i<j} W_ij cos(theta_i - theta_j) + (mu/2) sum_i sin^2(theta_i)

        dL/dt = -||d(theta)/dt||^2 <= 0  (global descent).
        """
        diff     = self.theta[:, None] - self.theta[None, :]
        coupling = np.sum(self.W * np.cos(diff))
        penalty  = (self.mu / 2.0) * np.sum(np.sin(self.theta) ** 2)
        return float(coupling + penalty)

    # ------------------------------------------------------------------
    # Binarisation diagnostics
    # ------------------------------------------------------------------

    def binarization_residual(self) -> float:
        """max_i |sin(theta_i)| — equals 0 at perfect binarisation."""
        return float(np.max(np.abs(np.sin(self.theta))))

    def is_binarized(self, tol: float = 1e-2) -> bool:
        """True iff all |sin(theta_i)| < tol."""
        return bool(np.all(np.abs(np.sin(self.theta)) < tol))

    # ------------------------------------------------------------------
    # Cut utilities
    # ------------------------------------------------------------------

    def activation(self, theta: np.ndarray) -> np.ndarray:
        """Soft-spin projection: cos(theta_i) in [-1, +1]."""
        return np.cos(theta)

    def activations(self) -> list:
        return self.activation(self.theta).tolist()

    def get_spins(self) -> np.ndarray:
        """Hard binarisation: sigma_i = sign(cos(theta_i)) in {-1, +1}."""
        sigma             = np.sign(np.cos(self.theta))
        sigma[sigma == 0] = 1.0
        return sigma

    def get_cut_value(self) -> float:
        """Continuous relaxed cut value from current phases."""
        diff = self.theta[:, None] - self.theta[None, :]
        return float(0.25 * (np.sum(self.W) - np.sum(self.W * np.cos(diff))) / 2.0)

    def get_binary_cut_value(self) -> float:
        """Cut value after hard binarisation of current phases."""
        sigma = self.get_spins()
        return float(0.25 * np.sum(self.W * (1.0 - sigma[:, None] * sigma[None, :])))

    def phases(self) -> list:
        return self.theta.tolist()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_net_configuration(self) -> dict:
        return {
            "n_nodes"  : self.n,
            "mu"       : self.mu,
            "Ks_equiv" : self.mu / 2.0,
            "K_equiv"  : 1.0,
            "coupling" : "RK45_scipy",
            "timestep" : self.timestep,
            "init_mode": self.init_mode,
        }

    def get_net_state(self) -> dict:
        return {
            "phases"                : self.phases(),
            "activations"           : self.activations(),
            "energy"                : self.get_energy(),
            "cut_value"             : self.get_cut_value(),
            "binary_cut_value"      : self.get_binary_cut_value(),
            "binarization_residual" : self.binarization_residual(),
            "is_binarized"          : self.is_binarized(),
        }
