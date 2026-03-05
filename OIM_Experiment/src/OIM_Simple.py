import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.integrate import quad


class OIMMaxCut:
    """
    Oscillator-based Ising Machine (OIM) for the Max-Cut problem.

    State: phase vector theta in R^n  (one oscillator phase per graph node).
    """

    def __init__(self, weight_matrix, seed=42, K=1.0, Ks=2.0,
                 coupling='cosine_shil', init_mode='small_random'):
        """
        Parameters
        ----------
        weight_matrix : symmetric n x n adjacency / weight matrix.
        seed          : RNG seed for reproducibility.
        K             : oscillator coupling strength (default 1.0).
        Ks            : Sub-Harmonic Injection Locking strength (default 2.0).
                        Penalty coefficient lambda = Ks / K.
                        Only used in 'cosine_shil' coupling mode.
        coupling      : 'cosine_shil'  -- standard OIM with SHIL (Eq. 8).
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.n = len(weight_matrix)
        self.W = np.array(weight_matrix, dtype=float)
        self.K = K
        self.Ks = Ks
        self.coupling = coupling
        self.timestep = 1e-2
        self.init_mode = init_mode
        self.theta = self._init_phases()


    def _init_phases(self):
        """
        Return an initial phase vector theta in R^n.

        The two Ising spin states map to phases 0 (spin +1) and pi (spin -1).
        Small Gaussian noise is added to all modes to break exact symmetry.
        """
        noise_scale = 1e-2 

        if self.init_mode == 'small_random':
            # Tiny uniform noise around theta=0.
            # The SHIL and coupling terms discover the partition from scratch.
            return np.random.uniform(-noise_scale, noise_scale, self.n)

        else:
            raise ValueError(
                f"Unknown init_mode '{self.init_mode}'. "
                "Choose from: small_random, large_random, bad_partition, "
                "ferromagnetic, min_eigenvec."
            )

    def _get_state_change(self):
        """
        Compute d theta_i / dt for all nodes simultaneously.

        cosine_shil:
          d theta_i/dt = K * sum_j W_ij * sin(theta_i - theta_j)
                       - Ks * sin(2 * theta_i)

          The coupling term pushes adjacent oscillators toward anti-alignment
          (phase difference pi), minimising the Ising energy.
          The SHIL term penalises phases away from {0, pi}, enforcing bi-stability.

        """
        diff = self.theta[:, None] - self.theta[None, :] 

        if self.coupling == 'cosine_shil':
            coupling_term = self.K * np.sum(self.W * np.sin(diff), axis=1)
            shil_term     = self.Ks * np.sin(2.0 * self.theta)
            return coupling_term - shil_term
        else : 
            raise ValueError("Wrong method we are actually in simple framework")

    def update(self):
        """
        Forward Euler step:  theta <- theta + dt * d theta/dt
        """
        self.theta += self.timestep * self._get_state_change()


    def activation(self, theta):
        """
        Soft-spin projection of a phase vector onto [-1, +1].

        In the rank-2 relaxation, each oscillator is a unit vector
        u_i = (cos theta_i, sin theta_i).  The natural scalar projection
        onto the binarisation axis is cos(theta_i):
          cos(0)  = +1  (spin up)
          cos(pi) = -1  (spin down)

        """
        return np.cos(theta)

    def activations(self):
        """Return current soft-spin values cos(theta) as a Python list."""
        return self.activation(self.theta).tolist()

    def get_energy(self):
        """
        Lyapunov energy of the OIM

        cosine_shil:
          L = sum_{i<j} W_ij cos(theta_i - theta_j)
            + (Ks/K) * sum_i sin^2(theta_i)

          First term: Ising coupling energy (minimised when adjacent nodes
          are anti-aligned, i.e., theta_i - theta_j = pi).
          Second term: angle penalty, zero only when theta_i in {0, pi}.

        generalized:
          L = sum_{i<j} W_ij * g2(theta_i - theta_j)

        Monotonically non-increasing: dL/dt = -||d theta/dt||^2 <= 0.
        The global minimum of L corresponds to a max-cut solution.
        """
        diff = self.theta[:, None] - self.theta[None, :]

        ising_term  = np.sum(self.W * np.cos(diff))
        penalty = (self.Ks / self.K) * np.sum(np.sin(self.theta) ** 2)
        return ising_term + penalty

    # ------------------------------------------------------------------
    # Cut value
    # ------------------------------------------------------------------

    def get_cut_value(self):
        """
        Continuous relaxation of the cut weight using current phases.

        In the rank-2 relaxation, sigma_i * sigma_j -> u_i . u_j = cos(theta_i - theta_j).

          Cut(theta) = 0.25 * (sum(W) - sum_{i,j} W_ij cos(theta_i - theta_j))
                     = sum_{i<j} W_ij * sin^2((theta_i - theta_j) / 2)

        This matches the exact integer cut when all theta_i in {0, pi},
        and mirrors the formula used in the Hopfield network:
          Cut(s) = 0.25 * (sum(W) - s^T W s)  with s_i = tanh(u_i/u0).
        """
        diff = self.theta[:, None] - self.theta[None, :]
        return 0.25 * (np.sum(self.W) - np.sum(self.W * np.cos(diff)))

    def get_binary_cut_value(self):
        """
        Binarise the phases and return the exact integer cut weight.
        Use this to evaluate the quality of the solution after convergence.
        """
        sigma = self.get_partition()
        return 0.25 * (np.sum(self.W) - float(sigma @ self.W @ sigma))

    # ------------------------------------------------------------------
    # Solution extraction
    # ------------------------------------------------------------------

    def get_partition(self):
        """
        Return the binary spin vector sigma in {+1, -1}^n.
          sigma_i = +1  (phase near 0)   -> node i belongs to partition A
          sigma_i = -1  (phase near pi)  -> node i belongs to partition B
        """
        sigma = np.sign(np.cos(self.theta))
        sigma[sigma == 0] = 1.0
        return sigma

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_net_configuration(self):
        return {
            "n_nodes":   self.n,
            "K":         self.K,
            "Ks":        self.Ks,
            "coupling":  self.coupling,
            "timestep":  self.timestep,
            "init_mode": self.init_mode,
        }

    def get_net_state(self):
        return {
            "phases":           self.phases(),
            "activations":      self.activations(),
            "energy":           self.get_energy(),
            "cut_value":        self.get_cut_value(),
            "binary_cut_value": self.get_binary_cut_value(),
        }

    def phases(self):
        """Return current phase vector as a Python list."""
        return self.theta.tolist()
