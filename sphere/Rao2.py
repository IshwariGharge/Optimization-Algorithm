import random
import math
import numpy as np
import pandas as pd

# Number of independent runs
run, Runs = 0, 30
best_val = np.zeros(Runs)  # Store best value from each run

# Loop over runs
while run < Runs:
    maxfes = 10000
    dim = 30
    pop_size = 10
    max_iter = math.floor(maxfes / pop_size)

    lb = -100 * np.ones(dim)
    ub = 100 * np.ones(dim)

    def fitness(particle):
        return sum(particle[i] ** 2 for i in range(dim))  # Sphere function

    Positions = np.zeros((pop_size, dim))
    best_pos = np.zeros(dim)
    worst_pos = np.zeros(dim)
    finval = np.zeros(max_iter)
    f1 = np.zeros(pop_size)
    f2 = np.zeros(pop_size)

    # Initialize population
    for i in range(dim):
        Positions[:, i] = np.random.uniform(0, 1, pop_size) * (ub[i] - lb[i]) + lb[i]

    for k in range(max_iter):
        best_score = float("inf")
        worst_score = float("-inf")

        for i in range(pop_size):
            for j in range(dim):
                Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])

            f1[i] = fitness(Positions[i, :])

            if f1[i] < best_score:
                best_score = f1[i].copy()
                best_pos = Positions[i, :].copy()
            if f1[i] > worst_score:
                worst_score = f1[i].copy()
                worst_pos = Positions[i, :].copy()

        finval[k] = best_score

        if (k + 1) % 500 == 0:
            print(f"For run {run+1}, best solution: {best_score} at iteration {k+1}")

        Positioncopy = Positions.copy()

        for i in range(pop_size):
            r = np.random.randint(pop_size)
            while r == i:
                r = np.random.randint(pop_size)

            if f1[i] < f1[r]:
                for j in range(dim):
                    r1 = random.random()
                    r2 = random.random()
                    Positions[i, j] = Positioncopy[i, j] + \
                                      r1 * (best_pos[j] - worst_pos[j]) + \
                                      r2 * (abs(Positioncopy[i, j]) - abs(Positioncopy[r, j]))
                    Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])
            else:
                for j in range(dim):
                    r1 = random.random()
                    r2 = random.random()
                    Positions[i, j] = Positioncopy[i, j] + \
                                      r1 * (best_pos[j] - worst_pos[j]) + \
                                      r2 * (abs(Positioncopy[r, j]) - abs(Positioncopy[i, j]))
                    Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])

            f2[i] = fitness(Positions[i, :])

        # Accept new positions if better
        for i in range(pop_size):
            if f1[i] < f2[i]:
                Positions[i, :] = Positioncopy[i, :]

    best_score = np.min(finval)
    print(f"The best solution for run {run+1} is: {best_score}")
    best_val[run] = best_score
    run += 1

# Final Results
print("\nFinal Results after all runs:")
best_result = np.min(best_val)
worst_result = np.max(best_val)
mean_result = np.mean(best_val)
std_result = np.std(best_val)

print("The Best solution is:", best_result)
print("The Worst solution is:", worst_result)
print("The Mean is:", mean_result)
print("The Standard Deviation is:", std_result)

# Store to Excel using pandas
data = {
    "Run": np.arange(1, Runs + 1),
    "Best Value": best_val
}

summary = {
    "Run": ["Best", "Worst", "Mean", "Std Dev"],
    "Best Value": [best_result, worst_result, mean_result, std_result]
}

df_results = pd.DataFrame(data)
df_summary = pd.DataFrame(summary)

with pd.ExcelWriter("Rao2_Sphere_results.xlsx", engine='openpyxl') as writer:
    df_results.to_excel(writer, index=False, sheet_name='Run Results')
    df_summary.to_excel(writer, index=False, sheet_name='Summary')
