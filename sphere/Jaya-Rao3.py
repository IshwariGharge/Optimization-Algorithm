

import random
import math
import numpy as np
import pandas as pd

run, Runs = 0, 30  # Number of runs
best_val = np.zeros(Runs)  # Stores best value from each run

while run < Runs:
    maxfes = 10000
    dim = 30
    pop_size = 10
    max_iter = math.floor(maxfes / pop_size)

    lb = -100 * np.ones(dim)
    ub = 100 * np.ones(dim)

    def fitness(particle):
        return np.sum(particle ** 2)  # Sphere function

    Positions = np.zeros((pop_size, dim))
    best_pos = np.zeros(dim)
    worst_pos = np.zeros(dim)
    finval = np.zeros(max_iter)
    f1 = np.zeros(pop_size)
    f2 = np.zeros(pop_size)

    for i in range(dim):
        Positions[:, i] = np.random.uniform(0, 1, pop_size) * (ub[i] - lb[i]) + lb[i]

    for k in range(max_iter):
        best_score = float("inf")
        worst_score = float("-inf")

        for i in range(pop_size):
            Positions[i, :] = np.clip(Positions[i, :], lb, ub)
            f1[i] = fitness(Positions[i, :])

            if f1[i] < best_score:
                best_score = f1[i]
                best_pos = Positions[i, :].copy()
            if f1[i] > worst_score:
                worst_score = f1[i]
                worst_pos = Positions[i, :].copy()

        finval[k] = best_score

        if (k + 1) % 500 == 0:
            print(f"For run {run + 1}, the best solution is: {best_score} in iteration {k + 1}")

        Positioncopy = Positions.copy()

        for i in range(pop_size):
            r = np.random.randint(pop_size)
            while r == i:
                r = np.random.randint(pop_size)

            if f1[i] < f1[r]:
                for j in range(dim):
                    r1 = random.random()
                    r2 = random.random()
                    Positions[i, j] = (
                        Positioncopy[i, j]
                         + r1 * (best_pos[j] - abs(Positioncopy[i, j]))
                        + r2 * (abs(Positioncopy[i, j]) - Positioncopy[r, j])
                    )
            else:
                for j in range(dim):
                    r1 = random.random()
                    r2 = random.random()
                    Positions[i, j] = (
                        Positioncopy[i, j]
                         + r1 * (best_pos[j] - abs(Positioncopy[i, j]))
                        + r2 * (abs(Positioncopy[r, j]) - Positioncopy[i, j])
                    )

            Positions[i, :] = np.clip(Positions[i, :], lb, ub)
            f2[i] = fitness(Positions[i, :])

        for i in range(pop_size):
            if f1[i] < f2[i]:
                Positions[i, :] = Positioncopy[i, :]

    best_score = np.min(finval)
    print(f"The best solution for run {run + 1} is: {best_score}")
    best_val[run] = best_score
    run += 1

# Final results
overall_best = np.min(best_val)
overall_worst = np.max(best_val)
overall_mean = np.mean(best_val)
overall_std = np.std(best_val)

print("\nFinal Results after all runs:")
print("The Best solution is:", overall_best)
print("The Worst solution is:", overall_worst)
print("The Mean is:", overall_mean)
print("The Standard Deviation is:", overall_std)

# Save to Excel
results_df = pd.DataFrame({
    'Run': np.arange(1, Runs + 1),
    'Best Solution': best_val
})

summary_df = pd.DataFrame({
    'Metric': ['Best', 'Worst', 'Mean', 'Standard Deviation'],
    'Value': [overall_best, overall_worst, overall_mean, overall_std]
})

with pd.ExcelWriter("Jaya-Rao3_Sphere_results.xlsx") as writer:
    results_df.to_excel(writer, sheet_name='Run Results', index=False)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)

print("\n Results successfully saved to 'Jaya-Rao3_Sphere_results.xlsx'")
