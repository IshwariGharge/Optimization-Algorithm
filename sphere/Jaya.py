import random
import math
import numpy as np
import pandas as pd  # Add pandas for Excel writing

run, Runs = 0, 30
best_val = np.zeros(Runs)

while run < Runs:
    maxfes = 10000
    dim = 30
    pop_size = 10
    max_iter = math.floor(maxfes / pop_size)

    lb = -100 * np.ones(dim)
    ub = 100 * np.ones(dim)

    def fitness(particle):
        return np.sum(particle ** 2)

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
            print("For run", run + 1, "the best solution is:", best_score, "in iteration number:", k + 1)

        Positioncopy = Positions.copy()

        for i in range(pop_size):
            for j in range(dim):
                r = np.random.rand(2)
                Positions[i, j] = Positions[i, j] + r[0] * (best_pos[j] - abs(Positions[i, j])) - r[1] * (worst_pos[j] - abs(Positions[i, j]))
                Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])
            f2[i] = fitness(Positions[i, :])

        for i in range(pop_size):
            if f1[i] < f2[i]:
                Positions[i, :] = Positioncopy[i, :]

    best_score = np.amin(finval)
    print("The best solution for run", run + 1, "is:", best_score)
    best_val[run] = best_score
    run += 1

# Final Summary
summary = {
    "Best": np.min(best_val),
    "Worst": np.max(best_val),
    "Mean": np.mean(best_val),
    "StdDev": np.std(best_val)
}

# Create a DataFrame for run-wise results and summary
df_runs = pd.DataFrame({
    "Run": list(range(1, Runs + 1)),
    "Best_Solution": best_val
})

df_summary = pd.DataFrame([summary])

# Write to Excel
with pd.ExcelWriter("Jaya_Sphere_results.xlsx", engine='openpyxl') as writer:
    df_runs.to_excel(writer, sheet_name='Run_Results', index=False)
    df_summary.to_excel(writer, sheet_name='Summary', index=False)

print("\nResults saved to 'Jaya_Sphere_results.xlsx'")
