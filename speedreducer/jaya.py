import numpy as np
import pandas as pd
import math

Runs = 30
best_val = np.zeros(Runs)
best_solutions = []

# Variable bounds for SRDP
bounds = np.array([
    [2.6, 3.6],   # x1 (b)
    [0.7, 0.8],   # x2 (m)
    [17, 28],     # x3 (z)
    [7.3, 8.3],   # x4 (l1)
    [7.3, 8.3],   # x5 (l2)
    [2.9, 3.9],   # x6 (d1)
    [5.0, 5.5]    # x7 (d2)
])

def constraints(x):
    x1, x2, x3, x4, x5, x6, x7 = x
    g = [
        27 / (x1 * x2**2 * x3) - 1,
        397.5 / (x1 * x2**2 * x3**2) - 1,
        (1.93 * x4**3) / (x2 * x3 * x6**4) - 1,
        (1.93 * x5**3) / (x2 * x3 * x7**4) - 1,
        np.sqrt((745 * x4 / (x2 * x3))**2 + 1.69e7) / (110 * x6**3) - 1,
        np.sqrt((745 * x5 / (x2 * x3))**2 + 1.575e8) / (85 * x7**3) - 1,
        x2 * x3 / 40 - 1,
        x1 / x2 - 5,
        1.5 * x6 + 1.9 - x4,
        1.1 * x7 + 1.9 - x5,
        x4 - x5
    ]
    return np.array(g)

def objective(x):
    x1, x2, x3, x4, x5, x6, x7 = x
    return (0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
            - 1.508 * x1 * (x6**2 + x7**2)
            + 7.4777 * (x6**3 + x7**3)
            + 0.7854 * (x4 * x6**2 + x5 * x7**2))

def fitness(x):
    g = constraints(x)
    penalty = np.sum(np.maximum(g, 0)) * 1e6
    return objective(x) + penalty

# Main ARSCA Loop
for run in range(Runs):
    dim = 7
    pop_size = 10
    maxfes = 10000
    max_iter = math.floor(maxfes / pop_size)

    lb = bounds[:, 0]
    ub = bounds[:, 1]

    Positions = np.random.uniform(lb, ub, size=(pop_size, dim))
    best_score = float("inf")
    worst_score = float("-inf")
    best_pos = np.zeros(dim)
    worst_pos = np.zeros(dim)

    for iter in range(max_iter):
        f1 = np.array([fitness(ind) for ind in Positions])

        best_idx = np.argmin(f1)
        worst_idx = np.argmax(f1)
        best_score = f1[best_idx]
        worst_score = f1[worst_idx]
        best_pos = Positions[best_idx].copy()
        worst_pos = Positions[worst_idx].copy()

        Positioncopy = Positions.copy()
        f2 = np.zeros(pop_size)

        for i in range(pop_size):
            for j in range(dim):
                r = np.random.rand(2)
                Positions[i, j] = Positions[i, j] + r[0] * (best_pos[j] - abs(Positions[i, j])) - r[1] * (worst_pos[j] - abs(Positions[i, j]))
                Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])
            f2[i] = fitness(Positions[i, :])

        for i in range(pop_size):
            if f1[i] < f2[i]:
                Positions[i, :] = Positioncopy[i, :]

    # After iterations, record if feasible
    best_candidate = Positions[np.argmin([fitness(ind) for ind in Positions])]
    if np.all(constraints(best_candidate) <= 0):
        final_fit = objective(best_candidate)
        best_val[run] = final_fit
        best_solutions.append([final_fit] + list(best_candidate))
        print(f"✅ Feasible solution for run {run + 1}: {final_fit}")
    else:
        best_val[run] = np.nan
        print(f"❌ No feasible solution for run {run + 1}")

# Save to Excel
columns = ["Weight", "x1 (b)", "x2 (m)", "x3 (z)", "x4 (l1)", "x5 (l2)", "x6 (d1)", "x7 (d2)"]
df = pd.DataFrame(best_solutions, columns=columns)

summary = {
    "Best": np.nanmin(best_val),
    "Worst": np.nanmax(best_val),
    "Mean": np.nanmean(best_val),
    "StdDev": np.nanstd(best_val)
}
df_summary = pd.DataFrame([summary])

with pd.ExcelWriter("Jaya_SpeedReducer_Updated.xlsx", engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name="Feasible_Solutions", index=False)
    df_summary.to_excel(writer, sheet_name="Summary", index=False)

print("\n📦 Results saved to 'Jaya_SpeedReducer_Updated.xlsx'")
