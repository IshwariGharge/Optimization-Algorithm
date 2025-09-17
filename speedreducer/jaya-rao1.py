import numpy as np
import pandas as pd

def jaya_speed_reducer():
    # Population size
    pop = 10
    # Number of design variables
    var = 7
    # Maximum function evaluations
    max_fes = 30000
    # Maximum number of iterations
    max_gen = max_fes // pop

    # Variable bounds
    lower_bounds = np.array([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0])
    upper_bounds = np.array([3.6, 0.8, 28, 28, 8.3, 3.9, 5.5])

    # Initialize population
    x = lower_bounds + (upper_bounds - lower_bounds) * np.random.rand(pop, var)

    # Calculate initial objective values
    f = objective(x)
    gen = 1

    fopt = []
    xopt = []
    results = []

    while gen <= max_gen:
        best_value_gen = np.inf
        best_solution_gen = None

        for _ in range(30):  # Repeat 30 times per iteration
            xnew = update_population_jaya(x, f)
            xnew = trimr(lower_bounds, upper_bounds, xnew)

            # Evaluate new population and constraints
            fnew = objective(xnew)
            gnew = constraints(xnew)

            # Apply constraint handling (penalty method)
            penalty = np.sum(np.maximum(0, gnew), axis=1)
            fnew = fnew + 1e6 * penalty  # Penalize infeasible solutions

            for i in range(pop):
                if fnew[i] < f[i]:
                    x[i, :] = xnew[i, :]
                    f[i] = fnew[i]

            best_value = np.min(f)
            if best_value < best_value_gen:
                best_value_gen = best_value
                best_solution_gen = x[np.argmin(f), :]

        fopt.append(best_value_gen)
        xopt.append(best_solution_gen)
        results.append([gen, best_value_gen] + list(best_solution_gen))

        print(f"Iteration No. = {gen}")
        print("Best Value = ", best_value_gen)

        gen += 1

    # Find the overall best solution
    val = np.min(fopt)
    ind = np.argmin(fopt)
    fes = pop * (ind + 1)
    print(f"Optimum value = {val:.50f}")
    print(f"Function evaluations = {fes}")

    # Save results to Excel
    columns = ["Iteration", "Best Value"] + [f"Var_{i+1}" for i in range(var)]
    df = pd.DataFrame(results, columns=columns)
    df.to_excel("Jaya-rao1_SpeedReducer_Results.xlsx", index=False)
    print("Results saved to 'Jaya-rao1_SpeedReducer_Results.xlsx'.")

def objective(x):
    """Objective function: Minimize weight of the speed reducer"""
    x1, x2, x3, x4, x5, x6, x7 = x.T
    return (
        0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
        - 1.508 * x1 * (x6**2 + x7**2)
        + 7.4777 * (x6**2 + x7**2)
        + 0.7854 * (x4 * x6**2 + x5 * x7**2)
    )

def constraints(x):
    """Constraint functions for the Speed Reducer Problem"""
    x1, x2, x3, x4, x5, x6, x7 = x.T

    g1  = 27 / (x1 * x3 * x2**2) - 1
    g2  = 397.5 / (x1 * x2**2 * x3**2) - 1
    g3  = (1.93 * x4**3) / (x2 * x6**4 * x3) - 1
    g4  = (1.93 * x5**3) / (x2 * x7**4 * x3) - 1
    g5  = np.sqrt((745 * (x4 / (x2 * x3)))**2 + 16.9e6) / (110 * x6**3) - 1
    g6  = np.sqrt((745 * (x5 / (x2 * x3)))**2 + 157.5e6) / (85 * x7**3) - 1
    g7  = (x2 * x3) / 40 - 1
    g8  = 5 * x2 / x1 - 1
    g9  = x1 / (12 * x2) - 1
    g10 = (1.5 * x6 + 1.9) / x4 - 1
    g11 = (1.1 * x7 + 1.9) / x5 - 1

    return np.column_stack([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])

def update_population_jaya(x, f):
    """Update population using Jaya algorithm."""
    pop, var = x.shape
    best = x[np.argmin(f), :]
    worst = x[np.argmax(f), :]
    xnew = np.zeros((pop, var))

    for i in range(pop):
        for j in range(var):
            xnew[i, j] = x[i, j] + np.random.rand() * (best[j] - abs(x[i, j])) + np.random.rand() * (best[j] - worst[j])

    return xnew

def trimr(mini, maxi, x):
    """Trim the population to ensure variables stay within bounds."""
    return np.clip(x, mini, maxi)

if __name__ == "__main__":
    jaya_speed_reducer()
