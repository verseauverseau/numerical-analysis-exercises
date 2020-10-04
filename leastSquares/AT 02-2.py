import math

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

def f_4(x):
    return math.cos(2*math.pi*x)

def f_5(x):
    return math.sin(2*math.pi*x)

def f_6(x):
    return math.cos(4*math.pi*x)

def f_7(x):
    return math.sin(4*math.pi*x)

def f_8(x):
    return math.cos(6*math.pi*x)

def f_9(x):
    return math.sin(6*math.pi*x)

def f_10(x):
    return math.cos(8*math.pi*x)

def f_11(x):
    return math.sin(8*math.pi*x)

def f_12(x):
    return math.cos(10*math.pi*x)

def f_13(x):
    return math.sin(10*math.pi*x)

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

xI = 1979
xF = 1984
m = 100
XPoints = np.linspace(xI, xF, m + 1)

yI = 330
yF = 350
n = 100

YPoints = [np.ndarray((n + 1, m + 1))] * 6

def plot(x_i, y_i, x, y, k):
    # Limites de x e y no gráfico
    pylab.xlim([xI, xF])
    pylab.ylim([yI, yF])

    # Título
    plt.title('Ajuste a polinômio de grau 2 com ' + str(k) + ' harmônicos e pontos dados em destaque')

    # Set x axis label for the contour plot
    plt.xlabel('X')
    # Set y axis label for the contour plot
    plt.ylabel('Y')

    plt.plot(x, y)

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
        size = len(base)
        A = np.zeros((size, size))
        b = np.zeros((size, 1))

        for i in range(size):
            for j in range(size):
                A[i][j] = discreteScalarProduct(x_i, x_i, base[i], base[j])

        for i in range(size):
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

def adjustDataToParabolaPlusKHarmonics(x_i, y_i):
    n = len(x_i)

    #Com k = 0
    base0 = [f_1, f_2, f_3]
    #Com k = 1
    base1 = [f_1, f_2, f_3, f_4, f_5]
    #Com k = 2
    base2 = [f_1, f_2, f_3, f_4, f_5, f_6, f_7]
    #Com k = 3
    base3 = [f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9]
    #Com k = 3
    base4 = [f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9, f_10, f_11]
    #Com k = 3
    base5 = [f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9, f_10, f_11, f_12, f_13]

    bases = [base0, base1, base2, base3, base4, base5]

    for k in range(len(bases)):
        f_i = [0] * n
        A, b = arrangeNormalLinearSystem(x_i, y_i, bases[k], ident)
        coefficients = np.linalg.solve(A, b)

        n_harms = (len(bases[k]) - 3) / 2

        print('Coeficientes (a, b, c, A_k, B_k, A_k+1, ...): ', coefficients)

        for i in range(n):
            x = x_i[i]
            for j in range(len(bases[k])):
                f_i[i] += coefficients[j] * bases[k][j](x)

        for i in range(len(XPoints)):
            x = XPoints[i]
            y = 0

            for j in range(len(bases[k])):
                y += coefficients[j] * bases[k][j](x)

            YPoints[k][i] = y

        plot(x_i, y_i, XPoints, YPoints[k], n_harms)
        print('R² do ajuste com ' + str(n_harms) + ' harmônicos: ', calculateR2(y_i, f_i))

    return

def main():
    x_i, y_i = readData()

    # adjustDataToParabola(x_i, y_i)
    adjustDataToParabolaPlusKHarmonics(x_i, y_i)

    return

main()
