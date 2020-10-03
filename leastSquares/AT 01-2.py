import numpy as np

import matplotlib.pyplot as plt
import pylab

x_i = [0.882, -6.857, -2.735, -0.563, -1.574, -8.743, -8.841, -7.266, 3.462, 0.551, -7.417, -0.373]
y_i = [5.202, 5.701, 3.353, 0.809, 1.822, 7.792, 7.933, 6.323, -2.95, 5.159, 4.317, 0.682]

def ident(t):
    return t


def squared(t):
    return t * t


def g_y(y):
    return 1


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


def arrangeNormalLinearSystem(y_i, base, objectiveFunction):
    try:
        n = len(base)
        A = np.zeros((n, n))
        b = [None] * n

        for i in range(n):
            for j in range(n):
                A[i][j] = DiscreteScalarProduct(base[i][1], base[j][1], base[i][0], base[j][0])

        for i in range(n):
            b[i] = DiscreteScalarProduct(base[i][1], y_i, base[i][0], objectiveFunction)

        return A, b
    except:
        raise RuntimeError('Algo deu errado')


def solveAdjustmentUsingLeastSquares():
    xy = [None] * len(x_i)
    for i in range(len(x_i)):
        xy[i] = x_i[i] * y_i[i]

    base = [[squared, x_i], [ident, xy], [squared, y_i], [ident, x_i], [ident, y_i]]
    A, b = arrangeNormalLinearSystem(y_i, base, g_y)

    return np.linalg.solve(A, b)


def plot(alpha, beta, gamma, csi, eta):
    # Criando uma partição regular do intervalo [xI,XF] para X
    # A partição será de m subintervalos. Portanto são m+1 pontos, para incluir os extremos
    # O tamanho de cada subintervalo é (xF-xI)/m
    xI = -10
    xF = 4
    m = 100
    XPoints = np.linspace(xI, xF, m + 1)

    # Criando uma partição regular do intervalo [yI,yF] para Y
    # A partição será de n subintervalos. Portanto são n+1 pontos, para incluir os extremos
    # O tamanho de cada subintervalo é (yF-yI)/n
    # Atenção: vamos criar no sentido contrário, por causa da visualização de matrizes
    # (as primeiras linhas são as mais no alto; sim, o Y vai corresponder às linhas da matriz)
    yI = -4
    yF = 6
    n = 50
    YPoints = np.linspace(yF, yI, n + 1)

    ZPoints = np.ndarray((n + 1, m + 1))

    for x in range(0, len(XPoints)):
        for y in range(0, len(YPoints)):
            ZPoints[y][x] = alpha * squared(XPoints[x]) + beta * XPoints[x] * YPoints[y] + gamma * squared(YPoints[y]) + csi * XPoints[x] + eta * YPoints[y]

    # Limites de x e y no gráfico
    pylab.xlim([xI, xF])
    pylab.ylim([yI, yF])

    # Título
    plt.title('Ajuste e pontos dados')

    # Set x axis label for the contour plot
    plt.xlabel('X')
    # Set y axis label for the contour plot
    plt.ylabel('Y')

    contours = plt.contour(XPoints, YPoints, ZPoints, [1])

    for i in range(len(x_i)):
        plt.plot(x_i[i], y_i[i], 'bo')

    # Display the contour plot
    plt.show()


def main():
    coefficients = solveAdjustmentUsingLeastSquares()

    alpha = coefficients[0]
    beta = coefficients[1]
    gamma = coefficients[2]
    csi = coefficients[3]
    eta = coefficients[4]

    plot(alpha, beta, gamma, csi, eta)

main()
