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


def At4Q1():
    A = 3.99
    B = -0.88
    C = -0.77
    D = 5.04
    E = -5.57

    eps = 10 ** (-6)
    max_int = 1000

    def g(t):
        return A * math.exp(B * t) + C * (t ** 2) + D * t + E

    def dg_dt(t):
        return A * B * math.exp(B * t) + 2 * C * t + D

    def d2g_dt2(t):
        return A * (B ** 2) * math.exp(B * t) + 2 * C

    def d3g_dt3(t):
        return A * (B ** 3) * math.exp(B * t)

    x_0 = []

    # Encontrar raiz da segunda derivada, ponto crítico da primeira derivada
    # A segunda derivada de g (f') tem raiz entre 0 e 1, pois f'(0) > 0 e f'(1) < 0
    resultados_d2g_dt2 = newton(d2g_dt2, d3g_dt3, 0.5, eps, max_int)
    print("Resultados f' = g'' (raiz, erro, qtde_de_iteracoes): " + str(resultados_d2g_dt2))

    c = resultados_d2g_dt2[0]

    # A concavidade de f (g') é voltada para baixo, pois f'' é sempre negativa em R
    if dg_dt(c) > 0:
        x_0.append(c - 1)
        x_0.append(c + 1)
    elif dg_dt(c) == 0:
        x_0.append(c)
    else:
        print("f não tem raiz real")

    # Achar as raízes de f = g'
    raizes_dg_dt = []
    for x in x_0:
        resultados_dg_dt = newton(dg_dt, d2g_dt2, x, eps, max_int)
        raizes_dg_dt.append(resultados_dg_dt[0])
        print("Raiz de f = g': " + str(resultados_dg_dt[0]) + " obtida em " + str(
            resultados_dg_dt[2]) + " iterações com erro de " + str(resultados_dg_dt[1]))

    print('\n\n\n')

    # Achar raízes de g
    # Antes de c, g tem concavidade para cima e depois, para baixo
    # raizes_dg_dt são pontos críticos de g,  digamos x_1 e x_2
    x_1, x_2 = raizes_dg_dt[0], raizes_dg_dt[1]

    x_0_g = []

    # Trecho com concavidade para cima
    if g(x_1) < 0:
        x_0_g.append(x_1 - 1)
        x_0_g.append(x_1 + 1)
    elif dg_dt(c) == 0:
        x_0.append(x_1)
    else:
        print("g não tem raiz real nesse intervalo")

    # Trecho com concavidade para baixo
    if g(x_2) > 0:
        x_0_g.append(x_2 - 1)
        x_0_g.append(x_2 + 1)
    elif dg_dt(c) == 0:
        x_0.append(x_2)
    else:
        print("g não tem raiz real nesse intervalo")

    # Achar as raízes de g
    raizes_g = []
    for x in x_0_g:
        resultados_g = newton(g, dg_dt, x, eps, max_int)
        raizes_g.append(resultados_g[0])
        print("Raiz de g: " + str(resultados_g[0]) + " obtida em " + str(
            resultados_g[2]) + " iterações com erro de " + str(resultados_g[1]))

    for item in raizes_dg_dt:
        print("Ponto crítico de g: ", item)

    print("Ponto de inflexão de g (mudança de concavidade): ", resultados_d2g_dt2[0])

At4Q1()
