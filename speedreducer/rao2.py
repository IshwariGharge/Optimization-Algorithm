import random
import math
import numpy as np
import pandas as pd

# Number of independent runs
run, Runs = 0, 30
best_val = np.zeros(Runs)  # Store best value from each run

# SRDP Bounds (For 7 variables as per the problem)
lb = np.array([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0])  # Lower bounds for SRDP
ub = np.array([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5])  # Upper bounds for SRDP

# Constraints function (e.g., SRDP constraints; must be respected)
def constraints(x):
    x1, x2, x3, x4, x5, x6, x7 = x
    g = [
        27 / (x1 * x2**2 * x3) - 1,
        397.5 / (x1 * x2**2 * x3**2) - 1,
        (1.93 * x4**3) / (x2 * x3 * x6**4) - 1,
        (1.93 * x5**3) / (x2 * x3 * x7**4) - 1,
        np.sqrt((745 * x4 / (x2 * x3 * x6**4))) - 1,
        np.sqrt((745 * x5 / (x2 * x3 * x7**4))) - 1
    ]
    return g

# Objective function (for SRDP; here using a simplified version, you can modify it as needed)
def fitness(x):
    # You would define the exact objective here (using SRDP specifications).
    # Placeholder objective: Minimize the sum of the variables (this should be adjusted).
    return np.sum(x**2)

# Loop over runs
while run < Runs:
    maxfes = 10000
    dim = 7  # SRDP has 7 variables
    pop_size = 10
    max_iter = math.floor(maxfes / pop_size)

    Positions = np.zeros((pop_size, dim))
    best_pos = np.zeros(dim)
    worst_pos = np.zeros(dim)
    finval = np.zeros(max_iter)
    f1 = np.zeros(pop_size)
    f2 = np.zeros(pop_size)

    # Initialize population within bounds
    for i in range(dim):
        Positions[:, i] = np.random.uniform(lb[i], ub[i], pop_size)

    for k in range(max_iter):
        best_score = float("inf")
        worst_score = float("-inf")

        for i in range(pop_size):
            # Ensure the positions respect bounds and constraints
            for j in range(dim):
                Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])

            # Apply constraints
            g = constraints(Positions[i, :])
            if np.all(np.array(g) <= 0):  # Feasible solution
                f1[i] = fitness(Positions[i, :])
            else:
                f1[i] = float("inf")  # Penalize infeasible solutions

            # Update the best and worst scores and positions
            if f1[i] < best_score:
                best_score = f1[i].copy()
                best_pos = Positions[i, :].copy()

            if f1[i] > worst_score:
                worst_score = f1[i].copy()
                worst_pos = Positions[i, :].copy()

        finval[k] = best_score

        # Print progress
        if (k + 1) % 500 == 0:
            print(f"For run {run + 1}, best solution: {best_score} at iteration {k + 1}")

        Positioncopy = Positions.copy()

        # Update positions
        for i in range(pop_size):
            r = np.random.randint(pop_size)
            while r == i:
                r = np.random.randint(pop_size)

            if f1[i] < f1[r]:  # Compare current solution with randomly selected solution
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

        # Accept new positions if better (based on fitness)
        for i in range(pop_size):
            if f1[i] < f2[i]:
                Positions[i, :] = Positioncopy[i, :]

    # Store best score from this run
    best_score = np.min(finval)
    print(f"The best solution for run {run + 1} is: {best_score}")
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

with pd.ExcelWriter("Rao2_SRDP_results.xlsx", engine='openpyxl') as writer:
    df_results.to_excel(writer, index=False, sheet_name='Run Results')
    df_summary.to_excel(writer, index=False, sheet_name='Summary')
