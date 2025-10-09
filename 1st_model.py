import numpy as np
import matplotlib as plt

def isSymmetric(matrix):
    rows = matrix.shape[0]
    for i in range(rows):
        for j in range(rows):
            if (matrix[i][j] != matrix[j][i]): 
                return False
    return True 

def WeightMatComputation(patterns):
    """
    Function is used to train the Hopfield to memorize the patterns mentioned above. When the number of memories to be contained
    in T is not too large and are uncorrelated (strong assumption), the way to tune the weight matrix is by Hebbian learning

    This is typically defined by the outer product. 

    @Input

    patterns : <np.array> Matrix containing all the memory patterns that we will identify as stable states

    @Returns 

    WeightMatrix : <np.array> Weight Matrix containing all the synapses connections between the different neurons
    """ 

    numberOfPatterns = patterns.shape[0]
    numberOfUnits = patterns.shape[1]
    WeightMatrix = np.zeros((numberOfUnits, numberOfUnits))

    for i in range (numberOfPatterns): 
        WeightMatrix += np.outer(patterns[i], patterns[i])

    for i in range(numberOfUnits): 
        WeightMatrix[i][i] = 0

    WeightMatrix /= numberOfUnits    
    if isSymmetric(WeightMatrix) == False:
        print ("Matrix is not Symmetric !! Error in the code")
    print("Weight Matrix is : ")
    print(WeightMatrix)
    return WeightMatrix

def neuronsUpdate(state, WeightMatrix) : 
    """
    Defines the update of the neurons values according to the weights

    @Inputs 

    state : <np.array> input State vector of all the units in the NN, those will be binary at this stage. They will be updated
    using the update equation of a neuron depending on all his inputs and using a sign function. 
    
    @Returns 

    state : <np.array> updated state vector
    """
    state = np.sign(WeightMatrix @ state)
    print("Current State Vectors")
    print(state)
    return state 

def IterationProcedure(WeightMatrix, noisyInput, steps = 10):
    state = noisyInput.copy()
    for i in range(steps): 
        print("Iteration number : ", i)
        state = neuronsUpdate(state, WeightMatrix)
    return state

def energy(W, state, bias=None):
    if bias is None:
        bias = np.zeros(W.shape[0])
    E = -0.5 * state @ W @ state - bias @ state
    return E

def main(): 
    steps = 5
    patterns = np.array([[-1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1], [-1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    WeightMatrix = WeightMatComputation(patterns=patterns)
    noisy_input = np.array([-1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    cleaned_input = IterationProcedure(WeightMatrix, noisy_input, steps)
    print("Noisy Input : ", noisy_input)
    print("Cleaned : ", cleaned_input)

if __name__ == '__main__':
    main()
