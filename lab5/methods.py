import numpy as np 
from functools import reduce
from math import factorial
import matplotlib.pyplot as plt


def lagrange_polynomial(xs, ys, n):
    return lambda x: sum([
        ys[i] * reduce(
            lambda a, b: a * b,
                        [(x - xs[j]) / (xs[i] - xs[j])
            for j in range(n) if i != j])
        for i in range(n)])


def divided_differences(x, y):
    n = len(y)
    A = np.copy(y).astype(float)
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            A[i] = (A[i] - A[i-1]) / (x[i] - x[i-j])
    return A 



def newton_divided_difference_polynomial(xs, ys, n):
    A = divided_differences(xs, ys)
    return lambda x: ys[0] + sum([
        A[k] * reduce(lambda a, b: a * b, [x - xs[j] for j in range(k)]) for k in range(1, n)
    ])


def finite_differences(y):
    n = len(y)
    delta_y = np.zeros((n, n))
    delta_y[:,0] = y
    for j in range(1, n):
        for i in range(n-j):
            delta_y[i,j] = delta_y[i+1,j-1] - delta_y[i,j-1]
    return delta_y


def newton_finite_difference_polynomial(xs, ys, n):
    h = xs[1] - xs[0]

    delta_y = finite_differences(ys)
    return lambda x: ys[0] + sum([
        reduce(lambda a, b: a * b, [(x - xs[0]) / h - j for j in range(k)]) * delta_y[k, 0] / factorial(k) for k in range(1, n)
    ])


def stirling_polynomial(xs, ys, n):
    n = len(xs) - 1
    alpha = n // 2
    fin_difs = []
    fin_difs.append(ys[:])

    for k in range(1, n + 1):
        last = fin_difs[-1][:]
        fin_difs.append(
            [last[i + 1] - last[i] for i in range(n - k + 1)])

    h = xs[1] - xs[0]
    dts1 = [0, -1, 1, -2, 2, -3, 3, -4, 4]

    f1 = lambda x: ys[alpha] + sum([
        reduce(lambda a, b: a * b,
               [(x - xs[alpha]) / h + dts1[j] for j in range(k)])
        * fin_difs[k][len(fin_difs[k]) // 2] / factorial(k)
        for k in range(1, n + 1)])

    f2 = lambda x: ys[alpha] + sum([
        reduce(lambda a, b: a * b,
               [(x - xs[alpha]) / h - dts1[j] for j in range(k)])
        * fin_difs[k][len(fin_difs[k]) // 2 - (1 - len(fin_difs[k]) % 2)] / factorial(k)
        for k in range(1, n + 1)])

    return lambda x: (f1(x) + f2(x)) / 2


def bessel_polynomial(xs, ys, n):
    n = len(xs) - 1
    alpha = n // 2
    fin_difs = []
    fin_difs.append(ys[:])

    for k in range(1, n + 1):
        last = fin_difs[-1][:]
        fin_difs.append(
            [last[i + 1] - last[i] for i in range(n - k + 1)])

    h = xs[1] - xs[0]
    dts1 = [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]

    return lambda x: (ys[alpha] + ys[alpha]) / 2 + sum([
        reduce(lambda a, b: a * b,
               [(x - xs[alpha]) / h + dts1[j] for j in range(k)])
        * fin_difs[k][len(fin_difs[k]) // 2] / factorial(2 * k) +

        ((x - xs[alpha]) / h - 1 / 2) *
        reduce(lambda a, b: a * b,
               [(x - xs[alpha]) / h + dts1[j] for j in range(k)])
        * fin_difs[k][len(fin_difs[k]) // 2] / factorial(2 * k + 1)

        for k in range(1, n + 1)])


def plot_function(func, a, b, name, dx = 0.001):
    xs, fs = [], []
    a -= dx
    b += dx 
    x = a 
    while x <= b:
        xs.append(x)
        fs.append(func(x))
        x += dx 
    plt.plot(xs, fs, 'g', label = name)


def generate_functions(xs, ys, x, n):
    methods = [
        ('Многочлен Лагранжа', lagrange_polynomial),
        ('Многочлен Ньютона с разделёнными разностями', newton_divided_difference_polynomial),
        ('Многочлен Ньютона с конечными разностями', newton_finite_difference_polynomial),
        ('Многочлен Стирлинга', stirling_polynomial),
        ('Многочлен Бесселя', bessel_polynomial)
    ]

    for name, method in methods:
        if method is bessel_polynomial and len(xs) % 2 == 1:
            print(f"{name} неприменим для нечетного количества точек.")
            continue

        print(name)
        P = method(xs, ys, n)
        print(f'P({x}) = {P(x)}')

        plt.title(name)
        plot_function(P, xs[0], xs[-1], name)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.scatter(x, P(x), c='r')
        for i in range(len(xs)):
            plt.scatter(xs[i], ys[i], c='b')

        plt.show()

