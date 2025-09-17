import numpy as np
import pandas as pd

def rao_1():
    # Population size
    pop = 30
    # Number of design variables
    var = 4  # Number of variables for the Gear Train problem
    # Maximum function evaluations
    max_fes = 30000
    # Maximum number of iterations
    max_gen = max_fes // pop

    # Variable bounds (adjusted for the Gear Train problem)
    mini = np.array([12, 12, 12, 12])  # Lower bounds for the number of teeth on gears
    maxi = np.array([60, 60, 60, 60])  # Upper bounds for the number of teeth on gears

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
        # Run 30 times for each iteration
        best_value = float('inf')
        best_solution = None
        for _ in range(30):
            # Update population
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

            # Record best solution in this run
            run_best_value = np.min(f)
            run_best_solution = x[np.argmin(f), :]

            # Track the best solution of the 30 runs
            if run_best_value < best_value:
                best_value = run_best_value
                best_solution = run_best_solution

        fopt.append(best_value)
        xopt.append(best_solution)

        # Store results in the list
        results.append([gen, best_value] + list(best_solution))

        # Display progress
        print(f"Iteration No. = {gen}")
        print("Best Value = ", best_value)

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
    df.to_excel("Rao1_Results_Gear_Train_30_Runs.xlsx", index=False)
    print("Results saved to 'Rao1_Results_Gear_Train_30_Runs.xlsx'.")


def objective(x):
    """Objective function: Gear Train problem."""
    return np.apply_along_axis(gear_train, 1, x)


def gear_train(solution):
    x1, x2, x3, x4 = solution
    result = ((1/6.931) - (x3*x2)/(x1*x4))**2
    return result


def update_population(x, f):
    """Update population using Rao-1 logic."""
    pop, var = x.shape
    best = x[np.argmin(f), :]
    worst = x[np.argmax(f), :]
    xnew = np.zeros((pop, var))

    for i in range(pop):
        for j in range(var):
            xnew[i, j] = x[i, j] + np.random.rand() * (best[j] - worst[j])

    return xnew


def trimr(mini, maxi, x):
    """Trim the population to ensure variables stay within bounds."""
    return np.clip(x, mini, maxi)


# Run the algorithm
if __name__ == "__main__":
    rao_1()
