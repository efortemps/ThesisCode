import matplotlib as plt 
import numpy as np 
import scipy.integrate as int
import random 

# Main city is X in our implementation

class HopfieldNet:
    def __init__(self, distances, seed, sigma, tau, method):
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
        self.u0 = 0.02
        self.tau = 1
        self.timestep = 1e-5
        self.distances = distances

        self.u_Xi = self.init_inputs()

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
        return 0.5 * (1 + np.tanh(single_input / self.u0))

    def get_A_update(self, city, position):
        """
        Computes the A term that will take importance in the update equation, in extent, it will compute the activations of all the possible positions in which a city can be in
        given the inputs and then it will compute the sum over all these possible positions. 
        (all neurons along the row of said city representing the possible positions in which the city can be along the Hamiltonian cycle)
        
        :param self: Refers to the object itself of the Hopfield net, the specific instance of the class being created. 
        :param city: Row of the matrix of neurons of our Hopfield net representing all the possible positions in which the city can find itself in.  
        :param position: Column representing the neurons/cities that can find themselves along one position. 
        """
        V_Xj = self.activation(self.u_Xi[city, :])
        sum = np.sum(V_Xj)
        # The reason we perform this substraction is because in the sum we have sum_{j≠position} v_city,j so we need to remove from the np.sum that 
        # took it into account. 
        sum -= self.activation(self.u_Xi[city, position])
        return sum * self.a
    
    def get_B_update(self, city, position):
        """
        Computes the B term that will take importance in the update equation, in extent, it will compute the activations of all the cities that can find themselves in 
        said position given the inputs and then it will compute the sum over all these possible column. 
        (all neurons along the row of said city representing the possible positions in which the city can be along the Hamiltonian cycle)
        
        :param self: Refers to the object itself of the Hopfield net, the specific instance of the class being created. 
        :param city: Row of the matrix of neurons of our Hopfield net representing all the possible positions in which the city can find itself in.  
        :param position: Column representing the neurons/cities that can find themselves along one position. 
        """
        V_Yi = self.activation(self.u_Xi[:, position])
        sum = np.sum(V_Yi)
        sum -= self.activation(self.u_Xi[city, position])
        return sum * self.b

    def get_C_update(self):
        """
        Computes the C term that will matter in the update term. It will sum the activation over all neurons of the network and ensures in the same 
        way that the at least (and at most) N neurons of the network are activated (making sure that we visit at least N cities to avoid trivial solutions)
        """
        sum = np.sum(self.activation(self.u_Xi[:, :]))
        sum -= self.size
        return sum * self.c
    
    def get_D_update(self, main_city, position):
        """
        Compute the *neighbor-distance weighted activation sum* used in the D-term contribution
        to the Hopfield(-Tank) TSP update.
        """
        sum = 0.0
        for city in range(0, self.size):
            preceding = self.activation(self.u_Xi[city, (position + 1) % self.size])
            following = self.activation(self.u_Xi[city, (position - 1)])
            sum += self.distances[main_city][city] * (preceding + following)
        return sum * self.d

    def get_states_change_classical(self, city, pos):
        new_state = -self.u_Xi[city][pos] / self.tau
        new_state -= self.get_A_update(city, pos)
        new_state -= self.get_B_update(city, pos)
        new_state -= self.get_C_update()
        new_state -= self.get_D_update(city, pos)
        return new_state
    
    def update(self):
        """
        Update the network states.
        """
        n = self.size
        self.increment = np.zeros((n, n), float)
        for city in range(n):
            for pos in range(n):
                self.increment[city, pos] = self.timestep * self.get_states_change_classical(city, pos)

        self.u_Xi += self.increment
        pass

    def get_a_energy(self):
        """
        Computes the A term that will take importance in the energy function, in extent, it will compute the activations of all the possible positions in which a city can be in
        given the inputs and then it will compute the sum over all these possible positions. 
        (all neurons along the row of said city representing the possible positions in which the city can be along the Hamiltonian cycle)
        
        :param self: Refers to the object itself of the Hopfield net, the specific instance of the class being created.  
        """

        V = self.activation(self.u_Xi)
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

        V = self.activation(self.u_Xi)
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
        sum = np.sum(self.activation(self.u_Xi[:, :]))
        sum -= self.size + self.sigma
        return sum**2 * (self.c/2)
    
    def get_e2_energy(self):
        """
        Vectorized E2 energy (same as Mańdziuk eq. 5.2 / Hopfield-Tank distance term)
        """
        V = self.activation(self.u_Xi)   
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
        return self.activation(self.u_Xi)

    def get_net_configuration(self):
        return {"a": self.a, "b": self.b, "c": self.c, "d": self.d, "alpha": self.alpha,
                "sigma": self.sigma, "timestep": self.timestep}

    def get_net_state(self):
        return {"activations": self.activations().tolist(),
                "inputs": self.u_Xi.tolist(),
                "inputsChange": self.inputs_change.tolist(), 
                "energy": self.get_energy()}
