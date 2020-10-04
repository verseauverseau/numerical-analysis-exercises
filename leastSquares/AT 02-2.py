import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab


def f_1(t):
    return t * t

def f_2(t):
    return t

def f_3(t):
    return 1

def ident(y):
    return y

def readData():
    dataStartDecimalDate = 1979.542
    dataEndDecimalData = 1983.793

    wholeDataSet = pd.read_csv('co2_mm_mlo.csv')
    decimalDateColumn = wholeDataSet['decimal date']

    filteredData = wholeDataSet[(decimalDateColumn >= dataStartDecimalDate) & (decimalDateColumn <= dataEndDecimalData)]
    dataToAnalyze = filteredData[['decimal date', 'trend']]

    return list(dataToAnalyze['decimal date']), list(dataToAnalyze['trend'])

def plot(x_i, y_i, a, b, c):
    xI = 1979
    xF = 1984
    m = 100
    XPoints = np.linspace(xI, xF, m + 1)

    yI = 330
    yF = 350
    n = 100

    YPoints = np.ndarray((n + 1, m + 1))

    for i in range(0, len(XPoints)):
        x = XPoints[i]
        YPoints[i] = a * f_1(x) + b * f_2(x) + c * f_3(x)

    # Limites de x e y no gráfico
    pylab.xlim([xI, xF])
    pylab.ylim([yI, yF])

    # Título
    plt.title('Ajuste a polinômio de grau 2 e pontos dados em destaque')

    # Set x axis label for the contour plot
    plt.xlabel('X')
    # Set y axis label for the contour plot
    plt.ylabel('Y')

    plt.plot(XPoints, YPoints)

    for i in range(len(x_i)):
        plt.plot(x_i[i], y_i[i], 'bo')

    # Display the contour plot
    plt.show()

def discreteScalarProduct(x, y, fx, fy):
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
        A = np.zeros((n, n))
        b = np.zeros((n, 1))

        for i in range(n):
            for j in range(n):
                A[i][j] = discreteScalarProduct(x_i, x_i, base[i], base[j])

        for i in range(n):
            b[i][0] = discreteScalarProduct(x_i, y_i, base[i], objectiveFunction)

        return A, b
    except:
        raise RuntimeError('Algo deu errado')

def calculateR2(y_i, f_i):
    SS_tot = 0
    SS_res = 0
    y_mean = sum(y_i) / len(y_i)

    for i in range(len(y_i)):
        SS_tot += (y_i[i] - y_mean) ** 2
        SS_res += (y_i[i] - f_i[i]) ** 2

    return 1 - (SS_res / SS_tot)

def adjustDataToParabola(x_i, y_i):
    n = len(x_i)

    f_i = [None] * n
    base = [f_1, f_2, f_3]

    A, b = arrangeNormalLinearSystem(x_i, y_i, base, ident)
    coefficients = np.linalg.solve(A, b)

    a, b, c = coefficients[0], coefficients[1], coefficients[2]
    print('a = ', a)
    print('b = ', b)
    print('c = ', c)

    for i in range(n):
        x = x_i[i]
        f_i[i] = a * f_1(x) + b * f_2(x) + c * f_3(x)

    print('R² do ajuste com 0 harmônicos: ', calculateR2(y_i, f_i))

    plot(x_i, y_i, a, b, c)

    return

def main():
    x_i, y_i = readData()

    adjustDataToParabola(x_i, y_i)

    return

main()
