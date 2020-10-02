import numpy as np

def linearSystemSolver(augmentedMatrix):
    n = len(augmentedMatrix)

    # Making numpy array of n size and initializing
    # to zero for storing solution vector
    x = np.zeros(n)

    # Applying Gauss Elimination
    for i in range(n):
        if augmentedMatrix[i][i] == 0.0:
            raise ZeroDivisionError

        for j in range(i + 1, n):
            ratio = augmentedMatrix[j][i] / augmentedMatrix[i][i]

            for k in range(n + 1):
                augmentedMatrix[j][k] = augmentedMatrix[j][k] - ratio * augmentedMatrix[i][k]

    # Back Substitution
    x[n - 1] = augmentedMatrix[n - 1][n] / augmentedMatrix[n - 1][n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = augmentedMatrix[i][n]

        for j in range(i + 1, n):
            x[i] = x[i] - augmentedMatrix[i][j] * x[j]

        x[i] = x[i] / augmentedMatrix[i][i]

    return x
