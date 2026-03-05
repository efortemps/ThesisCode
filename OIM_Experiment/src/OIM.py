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
                        'generalized'  -- optimal g2 coupling (Theorem 6).
        init_mode     : one of:
                        'small_random'  small noise near theta=0 (baseline).
                        'large_random'  each phase randomly pre-committed to 0 or pi.
                        'bad_partition' fixed first-half/second-half phase split.
                        'ferromagnetic' all phases at 0 (zero cut, worst case).
                        'min_eigenvec'  phases from sign of smallest eigenvector of W.
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

        if coupling == 'generalized':
            self._g2_coeffs = self._compute_g2_fourier_coefficients(num_terms=10)


    def _init_phases(self):
        """
        Return an initial phase vector theta in R^n.

        The two Ising spin states map to phases 0 (spin +1) and pi (spin -1).
        Small Gaussian noise is added to all modes to break exact symmetry.
        """
        noise_scale = 5e-2

        if self.init_mode == 'small_random':
            # Tiny perturbations near theta=0.  The network discovers the
            # partition from an essentially unbiased starting point.
            return np.array(
                [(random.random() - 0.5) * 10_000.0 for _ in range(self.n)]
            )

        elif self.init_mode == 'large_random':
            # Each oscillator is randomly pre-committed to 0 or pi.
            # Equivalent to starting from a random partition; the dynamics must
            # decide whether to stay or escape to a better configuration.
            phases = np.random.choice([0.0, np.pi], size=self.n)
            return phases + np.random.uniform(-noise_scale, noise_scale, self.n)

        elif self.init_mode == 'bad_partition':
            # Fixed split: nodes 0..n//2-1 at phase 0, rest at phase pi.
            # Analogous to the first-half / second-half partition in Hopfield.
            # For most graphs this is a sub-optimal cut; tests escape ability.
            phases = np.array(
                [0.0 if i < self.n // 2 else np.pi for i in range(self.n)]
            )
            return phases + np.random.uniform(-noise_scale, noise_scale, self.n)

        elif self.init_mode == 'ferromagnetic':
            # All oscillators at phase 0: every node on the same side -> zero cut.
            # The ferromagnetic state is a fixed point of the coupling term
            # (all sin(theta_i - theta_j) = 0), but the SHIL term sin(2*theta_i)
            # also vanishes at theta=0, making this an unstable equilibrium.
            # Tiny noise allows the dynamics to escape, but convergence is slow.
            return np.zeros(self.n) + np.random.uniform(-1e-4, 1e-4, self.n)

        elif self.init_mode == 'min_eigenvec':
            # Assign phases from the sign of the smallest eigenvector of W.
            # The smallest eigenvector of the adjacency matrix encodes the
            # best anti-ferromagnetic split: connected nodes tend to have
            # opposite signs, giving a good initial guess for the max-cut.
            eigvals, eigvecs = np.linalg.eigh(self.W)
            v_min = eigvecs[:, 0]
            phases = np.where(v_min >= 0, 0.0, np.pi)
            return phases + np.random.uniform(-noise_scale, noise_scale, self.n)

        else:
            raise ValueError(
                f"Unknown init_mode '{self.init_mode}'. "
                "Choose from: small_random, large_random, bad_partition, "
                "ferromagnetic, min_eigenvec."
            )

    @staticmethod
    def _compute_g2_fourier_coefficients(num_terms=10):
        """
        Fourier cosine series of g2(x) = 1 - 2x^2/pi^2 on [-pi, pi].

        Properties:
          g2(0)  = +1  (maximum, drives aligned phases apart)
          g2(pi) = -1  (minimum, stabilises anti-aligned phases)
          g2 is even and 2pi-periodic

        A Fourier expansion is used rather than the closed-form g2 to ensure
        global differentiability, as required by the ODE integrator.

        g2(x) ≈ a_0 + sum_{k=1}^{N} a_k * cos(k*x)
        g2'(x) ≈      sum_{k=1}^{N} -k * a_k * sin(k*x)
        """
        def f(x):
            return 1.0 - 2.0 * x ** 2 / np.pi ** 2

        a0_int, _ = quad(f, -np.pi, np.pi)
        coeffs = [a0_int / (2.0 * np.pi)]
        for k in range(1, num_terms + 1):
            ak_int, _ = quad(lambda x: f(x) * np.cos(k * x), -np.pi, np.pi)
            coeffs.append(ak_int / np.pi)
        return np.array(coeffs)

    def _g2_prime(self, x):
        """
        Derivative of the Fourier expansion of g2:
          g2'(x) ≈ -sum_{k=1}^{N} k * a_k * sin(k * x)
        """
        result = np.zeros_like(x, dtype=float)
        for k in range(1, len(self._g2_coeffs)):
            result -= k * self._g2_coeffs[k] * np.sin(k * x)
        return result

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

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

        else:   
            return -np.sum(self.W * self._g2_prime(diff), axis=1)

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

        if self.coupling == 'cosine_shil':
            ising_term  = 0.5 * np.sum(self.W * np.cos(diff))
            penalty     = (self.Ks / self.K) * np.sum(np.sin(self.theta) ** 2)
            return ising_term + penalty

        else:   # generalized
            g2_vals = self._g2_coeffs[0] * np.ones_like(diff)
            for k in range(1, len(self._g2_coeffs)):
                g2_vals += self._g2_coeffs[k] * np.cos(k * diff)
            return 0.5 * np.sum(self.W * g2_vals)

    # ------------------------------------------------------------------
    # Cut value
    # ------------------------------------------------------------------

    def get_cut_value(self):
        """
        Continuous relaxation of the cut weight using current phases.

        In the rank-2 relaxation, sigma_i * sigma_j -> u_i . u_j = cos(theta_i - theta_j).
        The continuous cut analog (Sec. 3.3) is therefore:

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
