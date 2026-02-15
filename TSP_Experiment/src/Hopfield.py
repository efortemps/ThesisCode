import matplotlib as plt 
import numpy as np 
import scipy.integrate as int
import random 

class HopfieldNet:
    def __init__(self, distances, seed, sigma, method):
        """
        Function that initializes all the parameters of our Hopfield net 
        
        :param self: Refers to the object itself of the Hopfield net, the specific instance of the class being created.  
        :param distances: Matrix containing all the distances between the cities. Note that this matrix is Symmetric. 
        :param seed: Fixed seed of the random number generator. 
        :param size_adj: Description
        """
        self.seed = seed
        random.seed(self.seed)
        self.method = method

        self.size = len(distances)

        self.inputs_change = np.zeros([self.size, self.size], float)
        self.a = 100
        self.b = 100
        self.c = 90
        self.d = 110

        # alpha is the gain here
        self.alpha = 50
        self.tau = 1
        self.timestep = 1e-5
        self.distances = distances

        self.sigma = sigma

        self.inputs = self.init_inputs()

    def init_inputs(self):
        """
        Function will initialize the activation value of each unit to 1/N plus or minus a small random perturbation (in this way
        the sum of the initial activations is approximately equal to N). See page 16 of the document. 
        
        :param self: Refers to the object itself of the Hopfield net, the specific instance of the class being created. 
        """
        base = np.ones([self.size, self.size], float)
        base /= self.size ** 2
        for x in range(0, self.size):
            for y in range(0, self.size):
                base[x][y] += ((random.random() - 0.5) / 10000)
        return base
    
    def init_inputs2(self):
        """
        Function will initialize the activation value of each unit to 1/N plus or minus a small random perturbation (in this way
        the sum of the initial activations is approximately equal to N). See page 16 of the document. 
        
        :param self: Refers to the object itself of the Hopfield net, the specific instance of the class being created. 
        """
        base = np.ones([self.size, self.size], float)
        base /= self.size
        for x in range(0, self.size):
            for y in range(0, self.size):
                base[x][y] += random.uniform(0, 0.03)
        return base
     

    def activation(self, single_input):
        return 0.5 * (1 + np.tanh(single_input * self.alpha))

    def get_a_update(self, city, position):
        """
        Computes the A term that will take importance in the update equation, in extent, it will compute the activations of all the possible positions in which a city can be in
        given the inputs and then it will compute the sum over all these possible positions. 
        (all neurons along the row of said city representing the possible positions in which the city can be along the Hamiltonian cycle)
        
        :param self: Refers to the object itself of the Hopfield net, the specific instance of the class being created. 
        :param city: Row of the matrix of neurons of our Hopfield net representing all the possible positions in which the city can find itself in.  
        :param position: Column representing the neurons/cities that can find themselves along one position. 
        """
        sum = np.sum(self.activation(self.inputs[city, :]))
        # The reason we perform this substraction is because in the sum we have sum_{j≠position} v_city,j so we need to remove from the np.sum that 
        # took it into account. 
        sum -= self.activation(self.inputs[city, position])
        return sum * self.a
    
    def get_b_update(self, main_city, position):
        """
        Computes the B term that will take importance in the update equation, in extent, it will compute the activations of all the cities that can find themselves in 
        said position given the inputs and then it will compute the sum over all these possible column. 
        (all neurons along the row of said city representing the possible positions in which the city can be along the Hamiltonian cycle)
        
        :param self: Refers to the object itself of the Hopfield net, the specific instance of the class being created. 
        :param city: Row of the matrix of neurons of our Hopfield net representing all the possible positions in which the city can find itself in.  
        :param position: Column representing the neurons/cities that can find themselves along one position. 
        """
        sum = np.sum(self.activation(self.inputs[:, position]))
        sum -= self.activation(self.inputs[main_city][position])
        return sum * self.b

    def get_c_update(self):
        """
        Computes the C term that will matter in the update term. It will sum the activation over all neurons of the network and ensures in the same 
        way that the at least (and at most) N neurons of the network are activated (making sure that we visit at least N cities to avoid trivial solutions)
        """
        sum = np.sum(self.activation(self.inputs[:, :]))
        sum -= self.size + self.sigma
        return sum * self.c
    
    def get_neighbours_weights(self, main_city, position):
        """
        Compute the *neighbor-distance weighted activation sum* used in the D-term contribution
        to the Hopfield(-Tank) TSP update.

        Implementation details
        ----------------------
        1) It loops over all candidate neighbor cities y (variable name `city` in the loop).
        2) It reads the *activations* (outputs) v_{y,i+1} and v_{y,i-1} by applying the sigmoid
        `activation(·)` to the corresponding input potentials u. (In your code:
        `self.inputs` stores u, `activation(self.inputs[...])` gives v.)
        3) It weights these two activations by the distance d_{x,y} from the distance matrix.
        4) It sums over y.

        Indexing and wrap-around:
        - (position + 1) % self.size implements i+1 with wrap-around, so if i is the last
        position, i+1 wraps back to 0 (closed tour). 
        - (position - 1) in Python already wraps for negative indices (i=0 → -1 means last
        column), so this also implements i-1 with wrap-around.

        Parameters
        ----------
        main_city : int
            The city index x (row in the V matrix).
        position : int
            The tour-position index i (column in the V matrix).

        Returns
        -------
        float
            The scalar Σ_y d_{main_city,y} * (v_{y,position+1} + v_{y,position-1}).
            This is the D-term “neighbor pressure” acting on neuron (main_city, position),
            before multiplying by D.
        """
        sum = 0.0
        for city in range(0, self.size):
            preceding = self.activation(self.inputs[city, (position + 1) % self.size])
            following = self.activation(self.inputs[city, (position - 1)])
            sum += self.distances[main_city][city] * (preceding + following)
        return sum

    def get_d_update(self, main_city, position):
        return self.get_neighbours_weights(main_city, position) * self.d

    def get_states_change_classical(self, city, pos):
        new_state = -self.inputs[city][pos]
        new_state -= self.get_a_update(city, pos)
        new_state -= self.get_b_update(city, pos)
        new_state -= self.get_c_update()
        new_state -= self.get_d_update(city, pos)
        return new_state
    
    def get_states_change_mandzukic(self, city, pos):
        """
        Same function as above the updates the state as in the classical hopfield network but it is said that it leads
        to more feasible tours unlike the first one. 
        """
        new_state = 0
        new_state -= self.get_a_update(city, pos)
        new_state -= self.get_b_update(city, pos)
        new_state -= self.get_c_update()
        new_state -= self.get_d_update(city, pos)
        return new_state

    def update(self):
        """
        Update the network inputs `self.inputs`.

        - If method == "Classical": Euler step 
        u <- u + dt * du/dt

        - If method == "Mandzukic": overwrite u 

        By default this uses the PART strategy (random permutation of all neurons),
        which is one of the strategies described in the paper.
        """
        n = self.size

        if self.method == "Mandzukic":
            # Mańdziuk update
            # Mandziuk performs an asynchronous update that we won't consider here. 
            old_inputs = self.inputs.copy()

            indices = [(x, i) for x in range(n) for i in range(n)]
            random.shuffle(indices)

            for (city, pos) in indices:
                self.inputs[city, pos] = self.get_states_change_mandzukic(city, pos)

            # Optional: store change for plotting/diagnostics
            self.inputs_change = self.inputs - old_inputs

        else:
            # Classical Hopfield–Tank: Euler step 
            self.inputs_change = np.zeros((n, n), float)
            for city in range(n):
                for pos in range(n):
                    self.inputs_change[city, pos] = self.timestep * self.get_states_change_classical(city, pos)

            self.inputs += self.inputs_change
            pass

    def get_a_energy(self):
        """
        Computes the A term that will take importance in the energy function, in extent, it will compute the activations of all the possible positions in which a city can be in
        given the inputs and then it will compute the sum over all these possible positions. 
        (all neurons along the row of said city representing the possible positions in which the city can be along the Hamiltonian cycle)
        
        :param self: Refers to the object itself of the Hopfield net, the specific instance of the class being created.  
        """

        V = self.activation(self.inputs)
        row_sums = np.sum(V, axis=1)
        row_sq_sums = np.sum(V * V, axis=1)
        E_A = np.sum(row_sums**2 - row_sq_sums)
        return 0.5 * self.a * E_A
    
    def get_b_energy(self):
        """
        Computes the B term that will take importance in the energy function, in extent, it will compute the activations of all the possible cities that can find themselves along
        in the given position given the inputs and then it will compute the sum over all these cities. 
        (all neurons along the column of said position representing the possible cities that can find themselves in that position)
        
        :param self: Refers to the object itself of the Hopfield net, the specific instance of the class being created. 
        """

        V = self.activation(self.inputs)
        col_sums = np.sum(V, axis=0)
        col_sq_sums = np.sum(V * V, axis=0)
        E_B = np.sum(col_sums**2 - col_sq_sums)
        return 0.5 * self.b * E_B
    
    def get_c_energy(self): 
        """
        Computes the C term that will take importance in the energy function, in extent, it makes sure that the 
        all the cities are visited in order to avoid trivial solutions. 
        (all neurons along the column of said position representing the possible cities that can find themselves in that position)
        
        """
        sum = np.sum(self.activation(self.inputs[:, :]))
        sum -= self.size + self.sigma
        return sum**2 * (self.c/2)
    
    def get_e2_energy(self):
        """
        Vectorized E2 energy (same as Mańdziuk eq. 5.2 / Hopfield-Tank distance term)
        """
        V = self.activation(self.inputs)   
        n = self.size           
        V_next = np.roll(V, shift=-1, axis=1)  
        V_prev = np.roll(V, shift=+1, axis=1) 

        E_next = 0.0
        E_prev = 0.0
        for i in range(n):
            vi = V[:, i]
            E_next += vi @ self.distances @ V_next[:, i]
            E_prev += vi @ self.distances @ V_prev[:, i]

        return 0.5 * self.d * (E_next + E_prev)
    
    def get_energy(self): 
        return self.get_a_energy() + self.get_b_energy() + self.get_c_energy() + self.get_e2_energy()

    def activations(self):
        return self.activation(self.inputs)

    def get_net_configuration(self):
        return {"a": self.a, "b": self.b, "c": self.c, "d": self.d, "alpha": self.alpha,
                "sigma": self.sigma, "timestep": self.timestep}

    def get_net_state(self):
        return {"activations": self.activations().tolist(),
                "inputs": self.inputs.tolist(),
                "inputsChange": self.inputs_change.tolist(), 
                "energy": self.get_energy()}
