import math
import numpy as np
import matplotlib.pyplot as plt
import pylab

t_i = [0.2, 1.1, 5.6, 9.1, 11.2, 11.7, 12.3, 18.5]
T_i = [1.3, 6.9, 28.7, 41.4, 47.0, 48.5, 50.1, 61.7]

Tm = []
Delta0 = []

# Valores de alpha calculados levando em consideração a relação de meia vida e alpha e estimando intervalos de tempo
# onde a diferença T - Tm parece ter caído pela metade
# Intervalos de t utilizados: [1.1 -> 5.6, 5.6 -> 11.2, 9.1 -> 12.3]
# alpha = ln 2 / (tf - ti)
alphas = [0.1540327, 0.1237763, 0.2166085]

Qs = []

def ident(t):
    return t

def f_1(t):
    return 1

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

def calculateQAndR2(T_m, Delta_0, alpha):
    Q = 0
    Q_mean = 0

    T_mean = sum(T_i) / len(T_i)

    for i in range(len(t_i)):
        Q += (T_i[i] - (T_m + Delta_0 * math.exp(-alpha * t_i[i]))) ** 2
        Q_mean += (T_i[i] - T_mean) ** 2

    Qs.append(Q)

    return 1 - (Q / Q_mean)

def adjustDataUsingLeastSquares():
    currentAlpha = alphas[0]

    def f_2(t):
        return math.exp(-currentAlpha * t)

    base = [f_1, f_2]

    for alpha in alphas:
        currentAlpha = alpha
        calculateAdjustment(alpha, base)

    # Dado que os dois primeiros valores estimados para alfa foram os com melhor R², portanto melhor ajuste,
    # vamos varrer o intervalo entre eles para tentar encontrar valores ainda melhores para alfa.
    # Além disso, também vale analisar alguns valores de alpha entre o segundo e o terceiro iniciais.
    diff1 = abs(alphas[0] - alphas[1])
    diff2 = abs(alphas[1] - alphas[2])
    betterAlphas = [alphas[0] + (diff1 / 3), alphas[0] + (2 * diff1 / 3), alphas[1] + (diff2 / 5), alphas[1] + (2 * diff2 / 5), alphas[1] + (3 * diff2 / 5), alphas[1] + (4 * diff2 / 5)]

    alphas.append(betterAlphas)

    for alpha in betterAlphas:
        currentAlpha = alpha
        calculateAdjustment(alpha, base)

def calculateAdjustment(alpha, base):
    A, b = arrangeNormalLinearSystem(t_i, T_i, base, ident)
    coefficients = np.linalg.solve(A, b)

    T_m, Delta_0 = coefficients[0], coefficients[1]

    Tm.append(T_m)
    Delta0.append(Delta_0)

    R2 = calculateQAndR2(T_m, Delta_0, alpha)
    print('R² com alpha = ' + str(alpha) + ' => ' + str(R2))


def plot(x_i, y_i, T_m, Delta_0, alpha):
    xI = 0
    xF = 20
    m = 100
    XPoints = np.linspace(xI, xF, m + 1)

    yI = -1
    yF = 70
    n = 100
    YPoints = np.zeros((n + 1))

    for i in range(len(XPoints)):
        x = XPoints[i]

        YPoints[i] = (T_m + Delta_0 * math.exp(-alpha * x))

    # Limites de x e y no gráfico
    pylab.xlim([xI, xF])
    pylab.ylim([yI, yF])

    # Título
    plt.title('Ajuste e pontos dados em destaque, alpha = ' + str(alpha))

    # Set x axis label for the contour plot
    plt.xlabel('X')
    # Set y axis label for the contour plot
    plt.ylabel('Y')

    plt.plot(XPoints, YPoints)

    for i in range(len(x_i)):
        plt.plot(x_i[i], y_i[i], 'bo')

    # Display the contour plot
    plt.show()

def main():
    adjustDataUsingLeastSquares()

    # print('Q', Qs)
    # print('T_m', Tm)
    # print('Delta0', Delta0)

    best_fit = int(np.argmin(Qs))

    #Plotar melhor ajuste
    plot(t_i, T_i, Tm[best_fit], Delta0[best_fit], alphas[best_fit])

    #Resultados
    print('Melhor ajuste de T_m: ' + str(Tm[best_fit]))
    print('Melhor ajuste de Delta_0 (em módulo): ' + str(abs(Delta0[best_fit])))
    print('Melhor alpha: ' + str(alphas[best_fit]))

main()
