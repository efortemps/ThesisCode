import numpy as np

def EnergyFunction(x,y,z,n): 
    return -(-x-y-z)**n-(-x+y+z)**n-(-x-y+z)**n-(x+y-z)**n

def relu_pow(u, n):
    """
    Rectified polynomial: [u]_+^n = (u^n) if u >= 0 else 0
    Works for scalars or numpy arrays.
    """
    u = np.asarray(u)
    # compute u**n only where u >= 0, else 0
    return np.where(u >= 0, u.astype(float) ** n, 0.0)

def RectifiedPolynomial(x, y, z, n):
    """
    Rectified polynomial energy:
    E_n_rect = - ( [t1]_+^n + [t2]_+^n + [t3]_+^n + [t4]_+^n )
    """
    t1 = (-x - y - z)
    t2 = (-x + y + z)
    t3 = ( x - y + z)
    t4 = ( x + y - z)
    return - (relu_pow(t1, n) + relu_pow(t2, n) + relu_pow(t3, n) + relu_pow(t4, n))

def PredictedOutput(x,y,n, exp_type): 
    if exp_type == "standard":
        E_off = EnergyFunction(x, y, -1, n)
        E_on = EnergyFunction(x, y, 1, n)
        return np.sign(E_off - E_on), E_off-E_on
    elif exp_type == "rectified":
        print("Using Rectified Polynomial")
        E_off = RectifiedPolynomial(x, y, -1, n)
        E_on = RectifiedPolynomial(x, y, 1, n)
        return np.sign(E_off - E_on), E_off-E_on
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")
    
if __name__ == "__main__":
    print("Available experiment types: 'standard', 'rectified'")
    print("Type 'exit' to quit the program")
    while True : 
        exp_type = input("\nEnter experiment type : ").strip().lower()
        if exp_type == "exit": 
            print("Done Experiment,Program Done")
            break
        if exp_type not in ["standard", "rectified"]:
           print("Error: Please enter 'standard', 'rectified', or 'exit'")
           continue
        if exp_type == "standard": 
            for n in range(4): 
                print("====================")
                print("Experiment for n=", n)
                print("====================")
                patterns = np.array([[-1,-1,-1],[-1,1,1],[1,-1,1],[1,1,-1]])
                for pattern in patterns :
                    x_pattern = pattern[0]
                    y_pattern = pattern[1]
                    z = PredictedOutput(x_pattern, y_pattern, n, exp_type=exp_type)
                    PredictedPattern = np.array([x_pattern,y_pattern,z])
                    print('-----------------------------------------')
                    print("Predicted Output : ", PredictedPattern)
                    print("True Output : ", pattern)
                    print("Are they equal : ", z==pattern[2])
                    print('-----------------------------------------')
                    continue
        if exp_type == "rectified": 
            for n in range(6): 
                print("====================")
                print("Experiment for n=", n)
                print("====================")
                patterns = np.array([[-1,-1,-1],[-1,1,1],[1,-1,1],[1,1,-1]])
                for pattern in patterns :
                    x_pattern = pattern[0]
                    y_pattern = pattern[1]
                    z = PredictedOutput(x_pattern, y_pattern, n, exp_type=exp_type)
                    PredictedPattern = np.array([x_pattern,y_pattern,z])
                    print('-----------------------------------------')
                    print("Predicted Output : ", PredictedPattern)
                    print("True Output : ", pattern)
                    print("Are they equal : ", z==pattern[2])
                    print('-----------------------------------------')
                    continue
