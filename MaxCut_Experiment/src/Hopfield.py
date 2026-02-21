import numpy as np
import random
import matplotlib.pyplot as plt


class HopfieldNetMaxCut:
    """
    Continuous Hopfield-Tank network for the Max-Cut problem.

    State: 1-D membrane potential vector  u ∈ R^n  (one per graph node).

    ODE : tau * du_i/dt = -u_i + sum_j  w_ij * tanh(u_j / u0)

    """

    def __init__(self, weight_matrix, seed=42, u0 = 0.05):
        """
        :param weight_matrix: Symmetric n×n weight/adjacency matrix of the graph.
                              w_ij >= 0, w_ii = 0.
        :param seed:          RNG seed for reproducibility.
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.n = len(weight_matrix)
        self.W = np.array(weight_matrix, dtype=float)
        self.u0 = u0
        self.tau = 1.0
        self.timestep = 1e-5
        self.u = self._init_inputs()


    def _init_inputs(self):
        """
        Initialise u near zero with a tiny random perturbation.
        Mirrors TSP init_inputs(): centred value ± small noise to break symmetry.
        Without noise, all nodes would evolve identically -> trivial partition.
        """
        return np.array([(random.random() - 0.5) / 10000.0 for _ in range(self.n)])

    def activation(self, u):
        """
        f(u) = tanh(u / u0),   output in (-1, +1).
        """
        return np.tanh(u / self.u0)

    def _get_state_change(self):
        """
        Compute du_i/dt for all nodes.

            du_i/dt = ( -u_i + sum_j w_ij * s_j ) / tau

        """
        s = self.activation(self.u) 
        drive = self.W @ s            
        return (-self.u - drive) / self.tau

    def update(self):
        """
        Forward Euler:  u  <-  u + dt * du/dt

        """
        self.u += self.timestep * self._get_state_change()


    def get_energy(self):
        """
        E_quad = -1/2 * s^T W s
            Quadratic Max-Cut term. Minimising E_quad maximises the cut.

        E_int  = u0 * sum_i [ s_i * arctanh(s_i) + 1/2 * log(1 - s_i^2) ]

        """
        s = self.activation(self.u)
        E_quad = 0.5 * (s @ self.W @ s)

        eps = 1e-10 
        s_c = np.clip(s, -1.0 + eps, 1.0 - eps)
        E_int = self.u0 * np.sum(
            s_c * np.arctanh(s_c) + 0.5 * np.log(1.0 - s_c ** 2)
        )
        return E_quad + E_int

    def get_cut_value(self):
        """
        Continuous relaxation of the cut weight using current activations.

            Cut(s) = 1/2 * (sum(W) - s^T W s)

        Valid for s_i in (-1,+1); equals the exact integer cut when s_i in {-1,+1}.
        """
        s = self.activation(self.u)
        return 0.25 * (np.sum(self.W) - s @ self.W @ s)

    def get_binary_cut_value(self):
        """
        Binarise the activations with sign() and return the exact integer cut weight.
        Use this to evaluate the quality of the solution after convergence.
        """
        s = np.sign(self.activation(self.u))
        s[s == 0] = 1 
        return 0.25 * (np.sum(self.W) - s @ self.W @ s)

    def get_partition(self):
        """
        Return the binary partition vector: s_i in {-1, +1}.
            s_i = +1  ->  node i belongs to side A
            s_i = -1  ->  node i belongs to side B
        """
        s = np.sign(self.activation(self.u))
        s[s == 0] = 1
        return s

    def activations(self):
        return self.activation(self.u).tolist()

    def get_net_configuration(self):
        return {
            "n_nodes":  self.n,
            "u0":       self.u0,
            "tau":      self.tau,
            "timestep": self.timestep,
        }

    def get_net_state(self):
        return {
            "activations":      self.activations(),
            "inputs":           self.u.tolist(),
            "energy":           self.get_energy(),
            "cut_value":        self.get_cut_value(),
            "binary_cut_value": self.get_binary_cut_value(),
        }