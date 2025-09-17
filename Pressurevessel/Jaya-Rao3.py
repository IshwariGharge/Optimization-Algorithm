import numpy as np
import pandas as pd
import math

def Jayarao_3_updated():
    # Population size
    pop = 10
    # Number of design variables
    var = 4  # Pressure vessel design has 4 variables
    # Maximum function evaluations
    max_fes = 30000
    # Maximum number of iterations
    max_gen = max_fes // pop

    # Variable bounds
    mini = [0, 0, 0, 0]  # Lower bounds for the design variables
    maxi = [100, 100, 100, 100]  # Upper bounds for the design variables

    # Initialize population
    x = np.zeros((pop, var))
    for i in range(var):
        x[:, i] = mini[i] + (maxi[i] - mini[i]) * np.random.rand(pop)

    # Calculate initial objective values
    f = objective(x)
    gen = 1

    fopt = []
    xopt = []

    # Store results for Excel
    results = []

    while gen <= max_gen:
        # Run the generation 30 times for each iteration
        best_value_gen = np.inf
        best_solution_gen = None

        for _ in range(30):  # Repeat 30 times for each iteration
            # Update population using Rao-3 logic
            xnew = update_population(x, f)
            # Trim population within bounds
            xnew = trimr(mini, maxi, xnew)
            # Evaluate new population
            fnew = objective(xnew)

            # Selection: update population with better individuals
            for i in range(pop):
                if fnew[i] < f[i]:
                    x[i, :] = xnew[i, :]
                    f[i] = fnew[i]

            # Track the best value of this run
            best_value = np.min(f)
            if best_value < best_value_gen:
                best_value_gen = best_value
                best_solution_gen = x[np.argmin(f), :]

        # Store best solution after 30 runs
        fopt.append(best_value_gen)
        xopt.append(best_solution_gen)

        # Store results in the list
        results.append([gen, best_value_gen] + list(best_solution_gen))

        # Display progress
        print(f"Iteration No. = {gen}")
        print("Best Value = ", best_value_gen)

        gen += 1

    # Find the overall best solution
    val = np.min(fopt)
    ind = np.argmin(fopt)
    fes = pop * (ind + 1)
    print(f"Optimum value = {val:.10f}")
    print(f"Function evaluations = {fes}")

    # Save results to Excel
    columns = ["Iteration", "Best Value"] + [f"Var_{i+1}" for i in range(var)]
    df = pd.DataFrame(results, columns=columns)
    df.to_excel("JayaRao3_Results_Updated_30Runs_PV.xlsx", index=False)
    print("Results saved to 'JayaRao3_Results_Updated_30Runs_PV.xlsx'.")


def objective(x):
    """Objective function for the pressure vessel design problem."""
    return np.array([pressure_vessel(sol) for sol in x])


def pressure_vessel(solution):
    """Objective function for the pressure vessel design problem."""
    x1, x2, x3, x4 = solution

    # Constraints
    g1 = -x1 + 0.0193 * x3
    g2 = -x2 + 0.00954 * x3
    g3 = 1296000 - (4 / 3) * math.pi * (x3 ** 3) - math.pi * (x3 ** 2) * x4
    g4 = x4 - 240

    # Penalize solutions that violate constraints
    if g1 <= 0 and g2 <= 0 and g3 <= 0 and g4 <= 0:
        # Objective function: Minimize the cost of the pressure vessel
        return 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3**2 + 3.1661 * x1**2 * x4 + 19.84 * x1**2 * x3
    else:
        return 1e10  # Penalize solutions that violate constraints


def update_population(x, f):
    """Update population using Rao-3 logic."""
    pop, var = x.shape
    best = x[np.argmin(f), :]
    worst = x[np.argmax(f), :]
    xnew = np.zeros((pop, var))

    for i in range(pop):
        # Randomly select a different individual
        k = np.random.randint(pop)
        while k == i:
            k = np.random.randint(pop)

        # Check fitness relationship
        if f[i] < f[k]:
            for j in range(var):
                r = np.random.rand(2)
                xnew[i, j] = (
                    x[i, j]
                    + r[0] * (best[j] - abs(x[i, j]))
                    + r[1] * (abs(x[i, j]) - x[k, j])
                )
        else:
            for j in range(var):
                r = np.random.rand(2)
                xnew[i, j] = (
                    x[i, j]
                    + r[0] * (best[j] - abs(x[i, j]))
                    + r[1] * (abs(x[k, j]) - x[i, j])
                )

    return xnew


def trimr(mini, maxi, x):
    """Trim the population to ensure variables stay within bounds."""
    return np.clip(x, mini, maxi)


# Run the algorithm
if __name__ == "__main__":
    Jayarao_3_updated()
