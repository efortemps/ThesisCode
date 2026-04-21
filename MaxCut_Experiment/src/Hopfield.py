import numpy as np
import random
from scipy.integrate import solve_ivp

class HopfieldNetMaxCut:
    """
    Continuous Hopfield-Tank network for the Max-Cut problem.

    State: 1-D membrane potential vector u ∈ R^n (one per graph node).
    ODE : tau * du_i/dt = -u_i - sum_j w_ij * tanh(u_j / u0)
    """
    def __init__(self, weight_matrix, seed=42, u0=0.05, init_mode='small_random', integration_method='euler'):
        """
        :param weight_matrix: Symmetric n×n adjacency/weight matrix.
        :param seed: RNG seed for reproducibility.
        :param u0: Gain parameter (steepness of tanh).
        :param init_mode: Initialisation strategy.
        :param integration_method: 'euler' or 'RK45'.
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.n = len(weight_matrix)
        self.W = np.array(weight_matrix, dtype=float)
        self.u0 = u0
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
        """
        s0 = 0.45
        SAT = np.arctanh(s0)

        if self.init_mode == 'small_random':
            return np.array([(random.random() - 0.5) / 10_000.0 for _ in range(self.n)])
        elif self.init_mode == 'large_random':
            signs = np.random.choice([-1.0, 1.0], size=self.n)
            return signs * SAT * self.u0
        elif self.init_mode == 'bad_partition':
            signs = np.array([1.0 if i < self.n // 2 else -1.0 for i in range(self.n)])
            return signs * SAT * self.u0
        elif self.init_mode == 'ferromagnetic':
            return np.ones(self.n) * SAT * self.u0
        elif self.init_mode == 'min_eigenvec':
            eigvals, eigvecs = np.linalg.eigh(self.W)
            v_min = eigvecs[:, 0]
            return v_min * SAT * self.u0
        else:
            raise ValueError(f"Unknown init_mode '{self.init_mode}'.")

    def activation(self, u):
        """
        f(u) = tanh(u / u0), output in (-1, +1).
        """
        return np.tanh(u / self.u0)

    def _ode_func(self, t, u):
        """
        ODE right-hand side for solve_ivp.
        du/dt = (-u - W * s) / tau
        """
        s = np.tanh(u / self.u0)
        drive = self.W @ s
        return (-u - drive) / self.tau

    def update(self):
        """
        Steps the simulation forward by `self.timestep`.
        """
        if self.integration_method == 'euler':
            # Forward Euler
            s = self.activation(self.u)
            drive = self.W @ s
            du_dt = (-self.u - drive) / self.tau
            self.u += self.timestep * du_dt
            self.t += self.timestep
            
        elif self.integration_method == 'RK45':
            # Runge-Kutta 4(5) using solve_ivp
            sol = solve_ivp(
                fun=self._ode_func,
                t_span=(self.t, self.t + self.timestep),
                y0=self.u,
                method='RK45',
                t_eval=[self.t + self.timestep]
            )
            # Update state and time
            self.u = sol.y[:, -1]
            self.t += self.timestep
        else:
            raise ValueError("Unknown integration_method. Use 'euler' or 'RK45'.")

    def get_hessian(self):
        """
        Computes the Hessian matrix of the Lyapunov energy function E(s)
        with respect to the activation vector s.
        
        The energy function is:
        E(s) = 0.5 * s^T W s + u0 * sum_i [s_i * arctanh(s_i) + 0.5 * log(1 - s_i^2)]
        
        Gradient dE/ds_i = (W s)_i + u0 * arctanh(s_i)
        Hessian d^2 E / (ds_i ds_j) = W_ij + delta_ij * (u0 / (1 - s_i^2))
        """
        s = self.activation(self.u)
        
        # To avoid division by zero if s is exactly 1 or -1
        eps = 1e-10
        s_c = np.clip(s, -1.0 + eps, 1.0 - eps)
        
        # Second derivative of the integral term w.r.t s_i
        diag_terms = self.u0 / (1.0 - s_c**2)
        
        # Hessian H = W + diag(u0 / (1 - s^2))
        H = self.W + np.diag(diag_terms)
        return H

    def get_energy(self):
        s = self.activation(self.u)
        E_quad = 0.5 * (s @ self.W @ s)
        
        eps = 1e-10
        s_c = np.clip(s, -1.0 + eps, 1.0 - eps)
        E_int = self.u0 * np.sum(s_c * np.arctanh(s_c) + 0.5 * np.log(1.0 - s_c ** 2))
        
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
            "n_nodes": self.n,
            "u0": self.u0,
            "tau": self.tau,
            "timestep": self.timestep,
            "initialisation method" : self.init_mode,
            "integration_method": self.integration_method
        }

    def get_net_state(self):
        return {
            "activations": self.activations(),
            "inputs": self.u.tolist(),
            "energy": self.get_energy(),
            "cut_value": self.get_cut_value(),
            "binary_cut_value": self.get_binary_cut_value(),
            "hessian": self.get_hessian().tolist()
        }