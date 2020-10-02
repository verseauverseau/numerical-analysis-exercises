
from leastSquares.mmq import returnLeastSquaresApproximationParams

def testFunction(x):
    return 2*x + 1

def f1(x):
    return 1

def f2(x):
    return x

def ident(y):
    return y

def testLeastSquaresApproximation():
    epsilon = 0.0000001

    n = 15

    testPoints = [None]*n
    testImages = [None]*n

    for i in range(n):
        x = i/10
        testPoints[i] = x
        testImages[i] = testFunction(x)

    base = [f1, f2]

    result = returnLeastSquaresApproximationParams(testPoints, testImages, base, ident)
    assert abs(result[0] - 1.0) <= epsilon
    assert abs(result[1] - 2.0) <= epsilon


testLeastSquaresApproximation()
