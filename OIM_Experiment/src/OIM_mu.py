import numpy as np
import random
import matplotlib.pyplot as plt


class OIMMaxCut:

    def __init__(self, weight_matrix, mu=2.0, seed=42, init_mode="small_random"):
        """
        Parameters
        ----------
        weight_matrix : symmetric n x n adjacency / weight matrix.
        mu            : penalty coefficient (Eq. 5).  Controls bi-stability.
                        Equivalent to Ks/K = mu/2 in the original OIM notation.
        seed          : RNG seed for reproducibility.
        init_mode     : 'small_random' — tiny uniform noise around theta=0.
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.n = len(weight_matrix)
        self.W = np.array(weight_matrix, dtype=float)
        self.mu = mu
        self.timestep = 1e-2
        self.init_mode = init_mode
        self.theta = self._init_phases()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_phases(self):
        """Initial phase vector theta in R^n (small noise around 0)."""
        noise_scale = 1e-2
        if self.init_mode == "small_random":
            return np.random.uniform(-noise_scale, noise_scale, self.n)
        raise ValueError(f"Unknown init_mode '{self.init_mode}'.")

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def _get_state_change(self):
        """
        Compute dtheta/dt = -grad L for all nodes simultaneously.

        dtheta_i/dt = sum_{j!=i} W_ij sin(theta_i - theta_j) - (mu/2) sin(2 theta_i)
        """
        diff = self.theta[:, None] - self.theta[None, :]         
        coupling_term = np.sum(self.W * np.sin(diff), axis=1)    
        shil_term = (self.mu / 2.0) * np.sin(2.0 * self.theta)   
        return coupling_term - shil_term

    def update(self):
        """Forward Euler step: theta <- theta + dt * dtheta/dt."""
        self.theta += self.timestep * self._get_state_change()

    # ------------------------------------------------------------------
    # Gradient of L  (= -dtheta/dt)
    # ------------------------------------------------------------------

    def get_gradient(self):
        """
        Return grad L(theta; W, mu) as a vector of length n.

        grad_k L = -sum_{j!=k} W_kj sin(theta_k - theta_j) + (mu/2) sin(2 theta_k)
        """
        return -self._get_state_change()

    # ------------------------------------------------------------------
    # Hessian of L
    # ------------------------------------------------------------------

    def get_hessian(self):
        """
        Return the Hessian H

        Derived by differentiating grad_k L a second time:

          Off-diagonal (k != l):
            H_kl = d/d_theta_l [grad_k L]
                 = W_kl * cos(theta_k - theta_l)

          Diagonal:
            H_kk = d/d_theta_k [grad_k L]
                 = -sum_{j!=k} W_kj cos(theta_k - theta_j)
                   + mu * cos(2 theta_k)
        """
        theta = self.theta
        diff = theta[:, None] - theta[None, :]          
        C = np.cos(diff)                                 
        H = self.W * C      
        diag_coupling = -np.sum(self.W * C, axis=1) 
        diag_penalty = self.mu * np.cos(2.0 * theta)   
        np.fill_diagonal(H, diag_coupling + diag_penalty)
        return H

    # ------------------------------------------------------------------
    # Binarisation diagnostics
    # ------------------------------------------------------------------

    def binarization_residual(self):
        """
        Return max_i |sin(theta_i)|.

        At a perfectly binarised state (theta_i in {0, pi}), sin(theta_i)=0
        for all i, so this residual equals 0.
        """
        return float(np.max(np.abs(np.sin(self.theta))))

    def is_binarized(self, tol=1e-2):
        """True if all phases are within tol of {0, pi} (i.e. |sin(theta_i)| < tol)."""
        return bool(np.all(np.abs(np.sin(self.theta)) < tol))

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def get_energy(self):
        """
        Lyapunov energy L(theta; W, mu).

        L = sum_{i<j} W_ij cos(theta_i - theta_j) + (mu/2) sum_i sin^2(theta_i)
        """
        diff = self.theta[:, None] - self.theta[None, :]
        coupling = 0.5 * np.sum(self.W * np.cos(diff))
        penalty = (self.mu / 2.0) * np.sum(np.sin(self.theta) ** 2)
        return coupling + penalty

    # ------------------------------------------------------------------
    # Cut utilities
    # ------------------------------------------------------------------

    def activation(self, theta):
        """Soft-spin projection: cos(theta_i) in [-1, +1]."""
        return np.cos(theta)

    def activations(self):
        return self.activation(self.theta).tolist()

    def get_spins(self):
        """Hard binarisation: sigma_i = sign(cos(theta_i)) in {-1, +1}."""
        sigma = np.sign(np.cos(self.theta))
        sigma[sigma == 0] = 1.0
        return sigma

    def get_cut_value(self):
        """Continuous relaxed cut value from current phases."""
        diff = self.theta[:, None] - self.theta[None, :]
        return 0.25 * (np.sum(self.W) - np.sum(self.W * np.cos(diff))) / 2.0

    def get_binary_cut_value(self):
        """Cut value after hard binarisation."""
        sigma = self.get_spins()
        return 0.25 * np.sum(self.W * (1.0 - sigma[:, None] * sigma[None, :]))

    def phases(self):
        return self.theta.tolist()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_net_configuration(self):
        return {
            "n_nodes": self.n,
            "mu": self.mu,
            "Ks_equiv": self.mu / 2.0,
            "coupling": "cosine_shil",
            "timestep": self.timestep,
            "init_mode": self.init_mode,
        }

    def get_net_state(self):
        return {
            "phases": self.phases(),
            "activations": self.activations(),
            "energy": self.get_energy(),
            "cut_value": self.get_cut_value(),
            "binary_cut_value": self.get_binary_cut_value(),
            "binarization_residual": self.binarization_residual(),
            "is_binarized": self.is_binarized(),
        }