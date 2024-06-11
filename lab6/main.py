import matplotlib.pyplot as plt 
from methods import *
import numpy as np
from tabulate import tabulate


def main():
    print('Выберите уравнение:')
    print("1. y' = x + y")
    print("2. y' = x * y")
    print("3. y' = x^2 + y + 2")
    choice = int(input('Ваш выбор (1 - 3): '))

    if choice == 1:
        f = f1 
        fe = f11 
    elif choice == 2:
        f = f2 
        fe = f22 
    elif choice == 3:
        f = f3 
        fe = f33
    else:
        print("Неправильный выбор!")
        return

    x0 = float(input("Введите начальное значение x0: "))
    y0 = float(input("Введите начальное значение y0: "))
    xn = float(input("Введите конечное значение xn: "))
    h = float(input("Введите шаг h: "))

    x_1, y_1 = euler_method(f, x0, y0, xn, h)
    x_2, y_2 = runge_kutta_4(f, x0, y0, xn, h)
    x_3, y_3 = adams(f, x0, y0, xn, h, False)
    x_4, y_4 = exact(fe, x0, y0, xn, h)

    table = []
    x = np.arange(x0, xn + h, h)
    
    for i in range(0, len(x) - 1):
        table.append([i, x[i], y_1[i], y_2[i], y_3[i], y_4[i]])
    print(tabulate(table, headers=['i', 'x_i', 'Метод Эйлера', 'Метод Рунге-Кутта 4-го порядка', 'Метод Адамса', 'Точное решение']))

    plt.figure(figsize=(10, 6))
    plt.plot(x_1, y_1, label="Метод Эйлера", color='red')
    plt.plot(x_2, y_2, label="Метод Рунге-Кутта", color='green')
    plt.plot(x_3, y_3, label="Метод Адамса", color='blue')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show() 


if __name__ == '__main__':
    main()
