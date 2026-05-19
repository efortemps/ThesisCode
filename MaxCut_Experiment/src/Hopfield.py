import numpy as np
import random
from scipy.integrate import solve_ivp


class HopfieldNetMaxCut:
    """
    Continuous Hopfield-Tank network for the Max-Cut problem.

    State: 1-D membrane potential vector u ∈ R^n (one per graph node).
    ODE  : tau * du_i/dt = -u_i - sum_j w_ij * tanh(λ * u_j)

    The gain parameter is λ = 1/u0 (steepness of tanh).
    Internally u0 = 1/λ is kept for the math; the public API uses λ.
    """

    def __init__(self, weight_matrix, seed=42, lam=20.0,
                 init_mode='small_random', integration_method='euler'):
        """
        :param weight_matrix:     Symmetric n×n adjacency/weight matrix.
        :param seed:              RNG seed for reproducibility.
        :param lam:               Gain parameter λ = 1/u0 (steepness of tanh).
        :param init_mode:         Initialisation strategy.
        :param integration_method: 'euler' or 'RK45'.
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.n   = len(weight_matrix)
        self.W   = np.array(weight_matrix, dtype=float)
        self.lam = lam
        self.u0  = 1.0 / lam   # kept internally for energy/Hessian formulae
        self.tau = 1.0
        self.timestep = 1e-5
        self.init_mode = init_mode
        self.integration_method = integration_method
        self.u = self._init_inputs()

        # Track simulation time for RK45
        self.t = 0.0

    def _init_inputs(self):
        """
        Return an initial membrane-potential vector u ∈ R^n.
        Saturation level: s0 = tanh(λ u) = 0.45  =>  u = arctanh(s0) / λ = arctanh(s0) * u0
        """
        s0  = 0.45
        SAT = np.arctanh(s0)   # u-space saturation = SAT * u0 = SAT / λ

        if self.init_mode == 'small_random':
            return np.array([(random.random() - 0.5) / 10_000.0
                             for _ in range(self.n)])
        elif self.init_mode == 'large_random':
            signs = np.random.choice([-1.0, 1.0], size=self.n)
            return signs * SAT * self.u0
        elif self.init_mode == 'bad_partition':
            signs = np.array([1.0 if i < self.n // 2 else -1.0
                              for i in range(self.n)])
            return signs * SAT * self.u0
        elif self.init_mode == 'ferromagnetic':
            return np.ones(self.n) * SAT * self.u0
        elif self.init_mode == 'min_eigenvec':
            eigvals, eigvecs = np.linalg.eigh(self.W)
            v_min = eigvecs[:, 0]
            return v_min * SAT * self.u0
        else:
            raise ValueError(f"Unknown init_mode '{self.init_mode}'.")

    # ------------------------------------------------------------------
    # Core dynamics
    # ------------------------------------------------------------------

    def activation(self, u):
        """
        f(u) = tanh(λ u) = tanh(u / u0),  output in (-1, +1).
        """
        return np.tanh(self.lam * u)

    def _ode_func(self, t, u):
        """
        ODE right-hand side for solve_ivp.
        du/dt = (-u - W s) / tau,   s = tanh(λ u)
        """
        s     = np.tanh(self.lam * u)
        drive = self.W @ s
        return (-u - drive) / self.tau

    def update(self):
        """
        Steps the simulation forward by `self.timestep`.
        """
        if self.integration_method == 'euler':
            s     = self.activation(self.u)
            drive = self.W @ s
            du_dt = (-self.u - drive) / self.tau
            self.u += self.timestep * du_dt
            self.t += self.timestep

        elif self.integration_method == 'RK45':
            sol = solve_ivp(
                fun=self._ode_func,
                t_span=(self.t, self.t + self.timestep),
                y0=self.u,
                method='RK45',
                t_eval=[self.t + self.timestep],
            )
            self.u  = sol.y[:, -1]
            self.t += self.timestep
        else:
            raise ValueError("Unknown integration_method. Use 'euler' or 'RK45'.")

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def get_hessian(self):
        """
        Computes the Hessian of the Lyapunov energy E(s) w.r.t. s.

        Energy (written in terms of λ = 1/u0):
            E(s) = (1/2) s^T W s + (1/λ) * Σ_i [s_i arctanh(s_i)
                                                  + (1/2) log(1 - s_i^2)]

        Gradient :  dE/ds_i = (W s)_i + (1/λ) arctanh(s_i)
        Hessian  :  d²E/(ds_i ds_j) = W_ij + δ_ij * (1/λ) / (1 - s_i^2)
                                     = W_ij + δ_ij *  u0   / (1 - s_i^2)
        """
        s   = self.activation(self.u)
        eps = 1e-10
        s_c = np.clip(s, -1.0 + eps, 1.0 - eps)

        diag_terms = self.u0 / (1.0 - s_c**2)   # u0 = 1/λ
        H = self.W + np.diag(diag_terms)
        return H

    def get_energy(self):
        s   = self.activation(self.u)
        E_quad = 0.5 * (s @ self.W @ s)

        eps = 1e-10
        s_c = np.clip(s, -1.0 + eps, 1.0 - eps)
        E_int = self.u0 * np.sum(
            s_c * np.arctanh(s_c) + 0.5 * np.log(1.0 - s_c**2)
        )
        return E_quad + E_int

    def get_cut_value(self):
        s = self.activation(self.u)
        return 0.25 * (np.sum(self.W) - s @ self.W @ s)

    def get_binary_cut_value(self):
        s = np.sign(self.activation(self.u))
        s[s == 0] = 1
        return 0.25 * (np.sum(self.W) - s @ self.W @ s)

    def get_partition(self):
        s = np.sign(self.activation(self.u))
        s[s == 0] = 1
        return s

    def activations(self):
        return self.activation(self.u).tolist()

    def get_net_configuration(self):
        return {
            "n_nodes":               self.n,
            "lambda":                self.lam,
            "u0 (= 1/lambda)":       self.u0,
            "tau":                   self.tau,
            "timestep":              self.timestep,
            "initialisation method": self.init_mode,
            "integration_method":    self.integration_method,
        }

    def get_net_state(self):
        return {
            "activations":      self.activations(),
            "inputs":           self.u.tolist(),
            "energy":           self.get_energy(),
            "cut_value":        self.get_cut_value(),
            "binary_cut_value": self.get_binary_cut_value(),
            "hessian":          self.get_hessian().tolist(),
        }
