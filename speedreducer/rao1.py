import numpy as np
import pandas as pd

# Define variable bounds
bounds = np.array([
    [2.6, 3.6],   # x1 (b)
    [0.7, 0.8],   # x2 (m)
    [17, 28],     # x3 (z)
    [7.3, 8.3],   # x4 (l1)
    [7.3, 8.3],   # x5 (l2)
    [2.9, 3.9],   # x6 (d1)
    [5.0, 5.5]    # x7 (d2)
])

def objective(x):
    x1, x2, x3, x4, x5, x6, x7 = x
    return (
        0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
        - 1.508 * x1 * (x6**2 + x7**2)
        + 7.4777 * (x6**3 + x7**3)
        + 0.7854 * (x4 * x6**2 + x5 * x7**2)
    )

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

def fitness(x):
    penalty = np.sum(np.maximum(constraints(x), 0)) * 1e6
    return objective(x) + penalty

def initialize_population(pop_size, dim, bounds):
    return np.random.uniform(bounds[:, 0], bounds[:, 1], size=(pop_size, dim))

def rao1_update(population, fitness_values):
    best = population[np.argmin(fitness_values)]
    worst = population[np.argmax(fitness_values)]
    new_pop = np.copy(population)

    for i in range(len(population)):
        r = np.random.rand(population.shape[1])

        new_pop[i] = population[i] + r * (best - np.abs(worst))
        
        new_pop[i] = np.clip(new_pop[i], bounds[:, 0], bounds[:, 1])

    return new_pop

def run_rao1_srdp(runs=30, iterations=1000, pop_size=10):
    dim = 7
    results = []

    for run in range(runs):
        population = initialize_population(pop_size, dim, bounds)
        fitness_values = np.array([fitness(ind) for ind in population])

        for _ in range(iterations):
            new_population = rao1_update(population, fitness_values)
            new_fitness = np.array([fitness(ind) for ind in new_population])

            for i in range(pop_size):
                if new_fitness[i] < fitness_values[i]:
                    population[i] = new_population[i]
                    fitness_values[i] = new_fitness[i]

        best_idx = np.argmin(fitness_values)
        best_sol = population[best_idx]

        if np.all(constraints(best_sol) <= 0):
            weight = objective(best_sol)
            results.append([weight] + list(best_sol))

        print(f"Run {run+1}/{runs} completed.")

    # Save to Excel
    df = pd.DataFrame(results, columns=[
        "Weight", "x1 (b)", "x2 (m)", "x3 (z)", "x4 (l1)", "x5 (l2)", "x6 (d1)", "x7 (d2)"
    ])
    df.to_excel("Rao1_SpeedReducer.xlsx", index=False)
    print("\n✅ All feasible run results saved to 'Rao1_SpeedReducer.xlsx'.")

    if not df.empty:
        best_row = df.loc[df["Weight"].idxmin()]
        print("\n📌 Best Overall Feasible Solution:")
        print(best_row.to_string(index=True))
    else:
        print("❌ No feasible solutions found.")

# Execute the algorithm
if __name__ == "__main__":
    run_rao1_srdp()
