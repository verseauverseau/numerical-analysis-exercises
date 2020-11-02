import math

def newton(f, df, x0, eps, itmax):
    L = range(1, itmax + 1)

    iteracao = 0
    a = x0

    raiz = None
    erro = 0

    for i in L:
        raiz = a

        if df(raiz) != 0:
            raiz = raiz - f(raiz) / df(raiz)
            erro = raiz - a
            a = raiz
            iteracao = i
        else:
            iteracao = itmax + 1
            break
        if abs(erro) <= eps:
            break

    if iteracao > itmax:
        iteracao = 0.25
    elif iteracao == itmax:
        iteracao = 0.75

    return [raiz, erro, iteracao]

def At4Q2():
    eps = 10 ** -6
    max_it = 1000

    def f(beta):
        return (1 / math.pi) * (beta - 0.5 * math.sin(2*beta)) - 0.3

    def df_dbeta(beta):
        return (1 / math.pi) * (1 - math.cos(2*beta))

    def h(beta):
        if 0 < beta < math.pi/2:
            return 0.5 * (1 - math.cos(beta))

    # Nossa missão é achar raízes de f usando o método de Newton
    # f' tem raízes em 0 e pi
    # Além disso, no intervalo [0, pi/2], f" é sempre positiva, portanto a concavidade é positiva
    # Existe raiz em [0, pi/2], pois f(0) < 0 e f(pi/2) > 0. Escolhemos beta_0 como pi/2 pois |f(pi/2) - 0| < |f(0) - 0|
    beta_0 = math.pi / 2
    resultados = newton(f, df_dbeta, beta_0, eps, max_it)

    print("Valor de beta que satisfaz f = 0: " + str(resultados[0]) + " obtido em " + str(resultados[2]) + " iterações com erro " + str(resultados[1]))
    print("Valor de h correspondente a 30% do volume do tanque: ", h(resultados[0]))

At4Q2()
