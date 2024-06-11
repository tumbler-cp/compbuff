import numpy as np 
import matplotlib.pyplot as plt
from tabulate import tabulate
from math import exp

def f1(x, y):
    return y + (1 + x) * (y**2)

def f2(x, y):
    return x + y

def f3(x, y):
    return y / x

def f11(x, x0, y0):
    return -exp(x) / (x*exp(x) - (x0*exp(x0)*y0 + exp(x0)) / y0)

def f22(x, x0, y0):
    return exp(x - x0) * (y0 + x0 + 1) - x - 1

def f33(x, x0, y0):
    return (x*y0) / x0

def exact(f, x0, y0, xn, h):
    x = np.arange(x0, xn + h, h)
    y = np.zeros(len(x))
    
    for i in range(0, len(x) - 1):
        y[i] = f(x[i], x0, y0)

    return x, y

def euler_method(f, x0, y0, xn, h):
    x = np.arange(x0, xn + h, h)
    y = np.zeros(len(x))
    y[0] = y0

    for i in range(1, len(x)):
        y[i] = y[i - 1] + h * f(x[i - 1], y[i - 1])
    
    return x, y


def runge_kutta_4(f, x0, y0, xn, h):
    x = np.arange(x0, xn + h, h)
    y = np.zeros(len(x))
    y[0] = y0

    for i in range(1, len(x)):
        k1 = h * f(x[i-1], y[i-1])
        k2 = h * f(x[i-1] + h/2, y[i-1] + k1/2)
        k3 = h * f(x[i-1] + h/2, y[i-1] + k2/2)
        k4 = h * f(x[i-1] + h, y[i-1] + k3)
        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return x, y 


def adams(f, x0, y0, xn, h, checker):
    x = np.arange(x0, xn + h, h)
    y = np.zeros(len(x))
    y[0] = y0
    
    k1 = h * f(x[0], y[0])
    k2 = h * f(x[0] + h/2, y[0] + k1/2)
    k3 = h * f(x[0] + h/2, y[0] + k2/2)
    k4 = h * f(x[0] + h, y[0] + k3)
    y[1] = y[0] + (k1 + 2*k2 + 2*k3 + k4) / 6

    k1 = h * f(x[1], y[1])
    k2 = h * f(x[1] + h/2, y[1] + k1/2)
    k3 = h * f(x[1] + h/2, y[1] + k2/2)
    k4 = h * f(x[1] + h, y[1] + k3)
    y[2] = y[1] + (k1 + 2*k2 + 2*k3 + k4) / 6

    k1 = h * f(x[2], y[2])
    k2 = h * f(x[2] + h/2, y[2] + k1/2)
    k3 = h * f(x[2] + h/2, y[2] + k2/2)
    k4 = h * f(x[2] + h, y[2] + k3)
    y[3] = y[2] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    for i in range(3, len(x) - 1):
        d = f(x[i], y[i]) - f(x[i - 1], y[i - 1])
        d2 = f(x[i], y[i]) - 2 * f(x[i - 1], y[i - 1]) + f(x[i - 2], y[i - 2])
        d3 = f(x[i], y[i]) - 3 * f(x[i - 1], y[i - 1]) + 3 * f(x[i - 2], y[i - 2]) - f(x[i - 3], y[i - 3])
        y[i + 1] = y[i] + h * f(x[i], y[i]) + ((h**2)/2)*d + ((5*h**3)/12)*d2 + ((3*h**4)/8)*d3
    
    if checker:
        return x, y

    x_p, y_p = adams(f, x0, y0, xn, h/2, True)

    if not abs(y[0] - y_p[0]) / 15 <= 0.001:
        return adams(f, x0, y0, xn, h / 2, False)

    return x, y
    
