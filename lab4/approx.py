import numpy as np
from main import output

def linear_approx(x, y):
    n = len(x)
    A = np.array([[np.sum(x**2), np.sum(x)],
                  [np.sum(x), n]])
    B = np.array([np.sum(x * y), np.sum(y)])
    a, b = np.linalg.solve(A, B)
    print('Линейная функция:')
    print(f'Коэффициенты: {a}, {b}')
    return a, b

def poly_2_approx(x, y):
    n = len(x)
    A = np.array([[np.sum(x**4), np.sum(x**3), np.sum(x**2)],
                  [np.sum(x**3), np.sum(x**2), np.sum(x)],
                  [np.sum(x**2), np.sum(x), n]])
    B = np.array([np.sum(y * x**2), np.sum(y * x), np.sum(y)])
    a, b, c = np.linalg.solve(A, B)
    print('Полином 2-й степени:')
    print(f'Коэффициенты: {a}, {b}, {c}')
    return a, b, c

def poly_3_approx(x, y):
    n = len(x)
    A = np.array([[np.sum(x**6), np.sum(x**5), np.sum(x**4), np.sum(x**3)],
                  [np.sum(x**5), np.sum(x**4), np.sum(x**3), np.sum(x**2)],
                  [np.sum(x**4), np.sum(x**3), np.sum(x**2), np.sum(x)],
                  [np.sum(x**3), np.sum(x**2), np.sum(x), n]])
    B = np.array([np.sum(y * x**3), np.sum(y * x**2), np.sum(y * x), np.sum(y)])
    a, b, c, d = np.linalg.solve(A, B)
    print('Полином 3-й степени:')
    print(f'Коэффициенты: {a}, {b}, {c}, {d}')
    return a, b, c, d

def exp_approx(x, y):
    y = np.log(y)
    a, b = linear_approx(x, y)
    print('Экспоненциальная функция:')
    print(f'Коэффициенты: {a}, {b}')
    return np.exp(b), a

def log_approx(x, y):
    x = np.log(x)
    a, b = linear_approx(x, y)
    print('Логарифмическая функция:')
    print(f'Коэффициенты: {a}, {b}')
    return a, b

def degree_approx(x, y):
    x = np.log(x)
    y = np.log(y)
    a, b = linear_approx(x, y)
    print('Степенная функция:')
    print(f'Коэффициенты: {a}, {b}')
    return np.exp(b), a
