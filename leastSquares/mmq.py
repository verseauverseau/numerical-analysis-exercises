import numpy as np

from leastSquares.linearSystemSolver import linearSystemSolver

def DiscreteScalarProduct(x, y, fx, fy):
    try:
        if len(x) == len(y):
            scalarProduct = 0

            for i in range(len(x)):
                scalarProduct += fx(x[i]) * fy(y[i])

            return scalarProduct
        else:
            raise ArithmeticError('x e y devem ter mesmo tamanho')
    except:
        raise RuntimeError('Algo deu errado')


def arrangeNormalLinearSystem(x_i, y_i, base, objectiveFunction):
    try:
        n = len(base)
        extended_matrix = np.zeros((n, n+1))

        for i in range(n):
            for j in range(n):
                extended_matrix[i][j] = DiscreteScalarProduct(x_i, x_i, base[i], base[j])

        for i in range(n):
            extended_matrix[i][n] = DiscreteScalarProduct(x_i, y_i, base[i], objectiveFunction)

        return extended_matrix
    except:
        raise RuntimeError('Algo deu errado')


def returnLeastSquaresApproximationParams(x_i, y_i, base, g_y):
    normalSystem = arrangeNormalLinearSystem(x_i, y_i, base, g_y)

    return linearSystemSolver(normalSystem)
