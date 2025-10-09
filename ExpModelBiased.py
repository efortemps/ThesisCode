import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def sgn(x):
    # Tie-breaking: sign(0) -> +1
    return np.where(x >= 0, 1, -1)

class Hopfield:
    def __init__(self, nbrNeurons, bias):
        self.nbrNeurons = nbrNeurons
        self.WeightMatrix = np.zeros((nbrNeurons, nbrNeurons), dtype=float)
        self.SensorBias = np.ones((nbrNeurons))*bias

    def train_hebb(self, patterns):
        """
        patterns: iterable of 1D numpy arrays of shape (n,)
                  values must be +1 or -1
        """
        P = 0
        WeightMatrix = np.zeros((self.nbrNeurons, self.nbrNeurons), dtype=float)
        for p in patterns:
            p = p.reshape(self.nbrNeurons)
            WeightMatrix += np.outer(p, p)
            P += 1
        WeightMatrix = WeightMatrix / self.nbrNeurons       # classic normalization
        np.fill_diagonal(WeightMatrix, 0.0)
        self.WeightMatrix = WeightMatrix

    def energy(self, statePotential):
        statePotential = statePotential.reshape(self.nbrNeurons)
        return -0.5 * float(statePotential @ self.WeightMatrix @ statePotential) - self.SensorBias@statePotential

    def recall(self, InitialState, pattern_ref=None, max_sweeps=50, synchronous=False, show = False):
        """
        Returns:
            s_final: final state
            energies: list of energy values per iteration
            overlaps: list of overlaps per iteration (if pattern_ref provided)
        """
        statePotential = InitialState.copy().reshape(self.nbrNeurons)
        energies = []
        overlaps = []

        for sweeps in range(max_sweeps):
            energies.append(self.energy(statePotential))
            if pattern_ref is not None:
                overlaps.append(self.overlap(statePotential, pattern_ref))

            prev = statePotential.copy()
            if synchronous:
                statePotential = sgn(self.WeightMatrix @ statePotential + self.SensorBias)
            else:
                for i in np.random.permutation(self.nbrNeurons):
                    h = float(self.WeightMatrix[i] @ statePotential + self.SensorBias[i])
                    statePotential[i] = 1 if h >= 0 else -1

            # Check if we have converged towards a value
            if np.array_equal(statePotential, prev):
                # record final values once more
                energies.append(self.energy(statePotential))
                if pattern_ref is not None:
                    overlaps.append(self.overlap(statePotential, pattern_ref))
                break
        iterations = np.arange(len(energies))
        if show : 
            plt.figure(figsize=(6,4))
            plt.plot(iterations, energies,'ro-', linewidth=2, markersize=8)
            plt.xlabel("Number of iterations")
            plt.ylabel("Energy level")
            plt.title("Energy level vs Number of Stored Patterns")
            plt.grid()
            plt.show()

        return statePotential, energies, overlaps
    

    def overlap(self, a, b):
        """Normalized overlap between two patterns in [-1,1]"""
        return float((a.reshape(self.nbrNeurons) * b.reshape(self.nbrNeurons)).mean())
    
def experiment_vary_patterns(SensitiveBias, N=100, pattern_counts=[1,2,3,5,8,10],  noise_level=0.3, max_iterations=50, synchronous = False):
    rng = np.random.RandomState(42)
    if synchronous: 
        print("Synchronous update setting")
    overlaps_final = np.zeros(len(pattern_counts))
    energies_final = np.zeros(len(pattern_counts))
    iterations = np.zeros(len(pattern_counts))
    i = 0
    for P in pattern_counts:
        if P == pattern_counts[-1]:
            show = True
        else : 
            show = False
        # 1. Create P random patterns
        patterns = [rng.choice([1, -1], size=N) for _ in range(P)]

        # 2. Train Hopfield
        net = Hopfield(N, SensitiveBias)
        net.train_hebb(patterns)

        # 3. Create noisy version of first pattern
        p0 = patterns[0]
        p0_noisy = p0.copy()
        flip_idx = rng.choice(N, size=int(noise_level * N), replace=False)
        p0_noisy[flip_idx] *= -1

        # 4. Recall
        s_final, energies, overlaps = net.recall(p0_noisy, pattern_ref=p0, max_sweeps=max_iterations, synchronous= synchronous, show=show)

        # 5. Record final overlap
        overlaps_final[i] = overlaps[-1]
        energies_final[i] = energies[-1]
        iterations[i] = len(overlaps)
        i += 1

        print(f"P={P}: final overlap={overlaps[-1]:.3f}, iterations={len(overlaps)}")

        

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    ax1.plot(pattern_counts, overlaps_final, marker='o', color='tab:blue')
    ax1.set_xlabel("Number of stored patterns (P)")
    ax1.set_ylabel("Final overlap with target pattern")
    ax1.set_title("Hopfield Recall vs Number of Stored Patterns")
    ax1.grid()

    ax2.plot(pattern_counts, energies_final, marker='o', color='tab:red')
    ax2.set_xlabel("Number of stored patterns (P)")
    ax2.set_ylabel("Final energy level")
    ax2.set_title("Energy level vs Number of Stored Patterns")
    ax2.grid()

    ax3.plot(pattern_counts, iterations, 'g-')
    ax3.set_xlabel("Number of stored patterns (P)")
    ax3.set_ylabel("Number of full networks sweeps")
    ax3.set_title("Full network sweeps vs Number of stored patterns")
    ax3.grid()

    plt.tight_layout()
    plt.show()

def precisionRelation(SensitiveBias, noise_level, max_iterations, synchronous): 
    PatternNumber = 150
    NbrNeurons = np.arange(104, 1024, 20)
    N_max = NbrNeurons[-1]
    rng = np.random.RandomState(42)
    base_patterns = [rng.choice([1, -1], size=N_max) for _ in range(PatternNumber)]
    overlaps_values = np.zeros(len(NbrNeurons))
    i = 0
    for Neurons in (NbrNeurons): 
        pattern = [p[:Neurons] for p in base_patterns]
        net = Hopfield(Neurons, SensitiveBias)
        net.train_hebb(pattern)

        p0 = pattern[0]
        p0_noisy = p0.copy()
        flip_idx = rng.choice(Neurons, size=int(noise_level * Neurons), replace=False)
        p0_noisy[flip_idx] *= -1

        # 4. Recall
        _ , _ , overlaps = net.recall(p0_noisy, pattern_ref=p0, max_sweeps=max_iterations, synchronous= synchronous)
        overlaps_values[i] = overlaps[-1]
        i += 1
    
    # Linear regression fit
    X = NbrNeurons.reshape(-1, 1)
    y = overlaps_values.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    slope = model.coef_[0][0]
    intercept = model.intercept_[0]

    print("Linear regression results:")
    print(f"  Overlap ≈ {slope:.5f} * N + {intercept:.3f}")
    print(f"  R² = {r2:.4f}")
    
    # 6️⃣ Plot how overlap evolves with N
    plt.figure(figsize=(6,4))
    plt.plot(NbrNeurons, overlaps_values, 'o-', lw=2, label='Observed overlap values')
    plt.plot(NbrNeurons, y_pred, 'r--', label=f'Linear fit (R²={r2:.3f})')
    plt.xlabel("Number of neurons (N)")
    plt.ylabel("Final overlap with target pattern")
    plt.title("Recall accuracy vs number of neurons (fixed underlying pattern)")
    plt.grid(True)
    plt.legend()
    plt.show()

    return NbrNeurons, overlaps_values

    
# --- Demo usage ---
if __name__ == "__main__":

    # By default experiment Asynchronous update rule
    # experiment_vary_patterns()

    NbrNeurons = 100
    FinalNbrPatterns = 200
    max_rounds = 100
    noise_level = 0.3
    Nbrpatterns = np.arange(2, FinalNbrPatterns+20, 20)
    SensorBias = 0.2

    start = time.perf_counter()
    experiment_vary_patterns(SensitiveBias=SensorBias,N=NbrNeurons,pattern_counts=Nbrpatterns, noise_level = noise_level, max_iterations=max_rounds, synchronous=True)
    end = time.perf_counter()

    runTime = end - start

    print("Total Run time is : ", runTime)

    # precisionRelation(SensorBias, noise_level=noise_level, max_iterations=max_rounds,synchronous=False)

    # Synchronous update using same amount of parameters