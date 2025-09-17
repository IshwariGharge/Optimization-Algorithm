import numpy as np
import pandas as pd

def Jaya_updated():
    # Population size
    pop = 10
    # Number of design variables
    var = 4  # Gear Train has 4 variables
    # Maximum function evaluations
    max_fes = 30000
    # Maximum number of iterations
    max_gen = max_fes // pop

    # Variable bounds for gear train problem
    mini = np.array([12, 12, 12, 12])  # Minimum gear sizes
    maxi = np.array([60, 60, 60, 60])  # Maximum gear sizes

    # Initialize population
    x = np.zeros((pop, var))
    for i in range(var):
        x[:, i] = mini[i] + (maxi[i] - mini[i]) * np.random.rand(pop)

    # Calculate initial objective values
    f = gear_train(x)
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
            # Update population using Jaya logic
            xnew = update_population(x, f)
            # Trim population within bounds
            xnew = trimr(mini, maxi, xnew)
            # Evaluate new population
            fnew = gear_train(xnew)

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
    print(f"Optimum value = {val:.50f}")
    print(f"Function evaluations = {fes}")

    # Save results to Excel
    columns = ["Iteration", "Best Value"] + [f"Var_{i+1}" for i in range(var)]
    df = pd.DataFrame(results, columns=columns)
    df.to_excel("Jaya_Results_GearTrain.xlsx", index=False)
    print("Results saved to 'Jaya_Results_GearTrain.xlsx'.")


def gear_train(x):
    """Objective function: Gear Train problem."""
    return ((1 / 6.931) - (x[:, 2] * x[:, 1]) / (x[:, 0] * x[:, 3])) ** 2


def update_population(x, f):
    """Update population using Jaya logic."""
    pop, var = x.shape
    best = x[np.argmin(f), :]
    worst = x[np.argmax(f), :]
    xnew = np.zeros((pop, var))

    for i in range(pop):
        # Check fitness relationship
        for j in range(var):
            r = np.random.rand(2)
            xnew[i, j] = (
                x[i, j]
                + r[0] * (best[j] - abs(x[i, j]))
                - r[1] * (worst[j] - abs(x[i, j]))
            )

    return xnew


def trimr(mini, maxi, x):
    """Trim the population to ensure variables stay within bounds."""
    return np.clip(x, mini, maxi)


# Run the algorithm
if __name__ == "__main__":
    Jaya_updated()
