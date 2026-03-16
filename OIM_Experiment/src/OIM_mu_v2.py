import numpy as np
import random
from scipy.integrate import solve_ivp
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
                    Binarisation occurs when mu > mu*  (see stability_threshold()).
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
        self.timestep  = 1e-2          # kept for Euler update() back-compat only
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

    def update(self):
        """
        Forward Euler step (kept for backward compatibility).
        Prefer simulate() / simulate_many() for accurate integration.
        """
        self.theta += self.timestep * self._phase_rhs(None, self.theta)

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
        phi0     : Initial condition (N,).  If None, uses self.theta.
        t_span   : (t_start, t_end) integration window.
        n_points : Number of time points stored in the output.
        rtol, atol : Solver tolerances.

        Returns
        -------
        sol : scipy OdeSolution object.
              sol.t        -- time vector  (n_points,)
              sol.y        -- phase matrix (N, n_points)
              sol.y[:, -1] -- final phases

        Side-effect
        -----------
        self.theta is updated to sol.y[:, -1] after integration.
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

        Parameters
        ----------
        phi0_list : list of (N,) initial phase vectors.
        t_span    : shared integration window.
        n_points  : shared number of time points.
        rtol, atol : solver tolerances.

        Returns
        -------
        List of scipy OdeSolution objects (one per initial condition).

        Note
        ----
        self.theta is restored to its original value — simulate_many() is non-destructive.
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
    # Structural matrices (paper notation — Theorem 2, Eq. 5)
    # ------------------------------------------------------------------

    def build_D(self, phi_star: np.ndarray) -> np.ndarray:
        """
        Signed Laplacian D(phi*) from Theorem 2, equation (5) of the paper.

        In the mu-framework J = -W, so:

            D_ij  = J_ij * cos(phi_i* - phi_j*)
                  = -W_ij * cos(phi_i* - phi_j*)   (i != j)
            D_ii  = -sum_{j!=i} D_ij               (zero row-sum)

        Properties
        ----------
        - Symmetric, zero row-sums  =>  0 is always an eigenvalue.
        - lambda_max(D) sets the binarisation threshold (Theorem 2):
              binarised  iff  mu > lambda_max(D)   [since mu = 2*Ks with K=1]
        - Identical result to oim.py's build_D() when called with J = -W.
        """
        phi_star = np.asarray(phi_star, dtype=float)
        diff     = phi_star[:, None] - phi_star[None, :]
        D        = -self.W * np.cos(diff)     # J*cos = -W*cos
        np.fill_diagonal(D, 0.0)
        
        np.fill_diagonal(D, -D.sum(axis=1))  # enforce zero row-sum
        return D

    def jacobian(self, phi_star: np.ndarray) -> np.ndarray:
        """
        Jacobian A(phi*) of the dynamics at equilibrium phi*.

            A = K * D(phi*) - 2*Ks * diag(cos(2*phi_i*))
              =     D(phi*) -  mu  * diag(cos(2*phi_i*))   (K=1, Ks=mu/2)

        Note: A = -Hessian(L), consistent with d(theta)/dt = -grad L.
        Eigenvalues of A determine local stability (Lemma 1 / Theorem 2).
        """
        phi_star = np.asarray(phi_star, dtype=float)
        D        = self.build_D(phi_star)
        return D - self.mu * np.diag(np.cos(2.0 * phi_star))

    def stability_threshold(self, phi_star: np.ndarray) -> float:
        """
        mu-threshold above which phi* is asymptotically stable (Theorem 2).

            mu* = lambda_max(D(phi*))

        Relationship to oim.py convention (K=1):
            mu* = 2 * Ks*   where Ks* = K * lambda_max(D) / 2

        Parameters
        ----------
        phi_star : Type I M2 equilibrium point (all phases in {0, pi}).

        Returns
        -------
        float : mu* — binarisation occurs iff self.mu > mu*.
        """
        D    = self.build_D(phi_star)
        lmax = float(np.linalg.eigvalsh(D).max())
        return lmax

    # ------------------------------------------------------------------
    # Gradient of L  (= -d(theta)/dt)
    # ------------------------------------------------------------------

    def get_gradient(self) -> np.ndarray:
        """grad L(theta) = -d(theta)/dt evaluated at self.theta."""
        return -self._phase_rhs(None, self.theta)

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def get_energy(self) -> float:
        """
        Lyapunov energy L(theta; W, mu).

        L = sum_{i<j} W_ij cos(theta_i - theta_j) + (mu/2) sum_i sin^2(theta_i)

        dL/dt = grad(L) . d(theta)/dt = -||d(theta)/dt||^2 <= 0  (global descent).
        """
        diff     = self.theta[:, None] - self.theta[None, :]
        coupling = 0.5 * np.sum(self.W * np.cos(diff))
        penalty  = (self.mu / 2.0) * np.sum(np.sin(self.theta) ** 2)
        return float(coupling + penalty)

    # ------------------------------------------------------------------
    # Binarisation diagnostics
    # ------------------------------------------------------------------

    def binarization_residual(self) -> float:
        """max_i |sin(theta_i)| -- equals 0 at perfect binarisation."""
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
