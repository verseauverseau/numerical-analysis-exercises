
import matplotlib.pyplot as plt
import numpy as np
from pip._vendor.msgpack.fallback import xrange

def main():
    x_i = [-4, -2, -1, 2, 4]
    f_x_i = [-3, 2, -4, -2, -1]
    targetPoints = [-5, -3, 0, 1, 3, 5]
    givenPoints = []
    estimatedPoints = []

    func, targetInterpolation = lagrange(x_i, f_x_i, targetPoints)
    print("Interpolação de " + str(targetPoints) + " : " + str(targetInterpolation))

    for k in range(len(targetPoints)):
        estimatedPoints.append((targetPoints[k], targetInterpolation[k]))
    for k in range(len(x_i)):
        givenPoints.append((x_i, f_x_i))

    plot(func, estimatedPoints, givenPoints)


def plot(f, points, givenPoints):
    plt.axis([-5.5, 5.5, -45, 15])

    x = np.linspace(-6, 6, 100)
    y = f(x)

    plt.plot(x, y, 'r')

    x_list = []
    y_list = []
    for x_p, y_p in points:
        x_list.append(x_p)
        y_list.append(y_p)

    given_x_list = []
    given_y_list = []
    for x_p, y_p in givenPoints:
        given_x_list.append(x_p)
        given_y_list.append(y_p)

    plt.plot(x_list, y_list, 'ro')
    plt.plot(given_x_list, given_y_list, 'bo')

    plt.show()


def lagrange(x_i, f_x_i, target):
    targetInterpolation = []

    def P(x):
        total = 0
        n = len(x_i)
        for i in range(n):
            xi, yi = x_i[i], f_x_i[i]

            def g(i, n):
                tot_mul = 1
                for j in xrange(n):
                    if i == j:
                        continue
                    xj, yj = x_i[j], f_x_i[j]
                    tot_mul *= (x - xj) / float(xi - xj)

                return tot_mul

            total += yi * g(i, n)
        return total

    for point in target:
        targetInterpolation.append(P(point))

    return P, targetInterpolation

main()
