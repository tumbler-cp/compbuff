import yaml
import numpy as np
import sys
from methods import generate_functions


def file_input(filename):
    try:
        with open(filename, 'r') as fin:
            data = yaml.safe_load(fin)
            xs = data['xs']
            ys = data['ys']
            x = data['x']
    except:
        print('Неправильный конфигурационный файл')
    return xs, ys, x


def fun_input():
    func = input("Выберите функцию (sin, cos, exp): ")
    start = float(input("Введите начало интервала: "))
    end = float(input("Введите конец интервала: "))
    num_points = int(input("Введите количество точек: "))
    x0 = float(input('Введите точку интерполяции: '))
    x = np.linspace(start, end, num_points)
    if func == 'sin':
        y = np.sin(x)
    elif func == 'cos':
        y = np.cos(x)
    elif func == 'exp':
        y = np.exp(x)
    else:
        raise ValueError("Неизвестная функция")
    return x, y, x0


def hand_input():
    x = float(input("Введите точку интерполяции: "))
    str = ''
    xs = []
    ys = []
    print("Введите 'quit', чтобы закончить ввод.")
    print("Введите узлы интерполяции:")
    while str != 'quit':
        str = input()
        point = str.strip().split()
        if len(point) == 2:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
        else:
            if str != 'quit':
                print("! Неправильный ввод. Введенная точка не будет использована.")
    return xs, ys, x


def main():
    x = []
    y = []
    x0 = 0
    if len(sys.argv) > 1:
        x, y, x0 = file_input(sys.argv[1])
    else:
        while True:
            print('Введите способ ввода')
            print('1. Из клавиатуры')
            print('2. Из функции')
            choice = input('Выберите способ ввода(1-2): ')
            if choice == '1':
                x, y, x0 = hand_input()
                break
            elif choice == '2':
                x, y, x0 = fun_input()
                break
            else:
                print('Выберите число 1 или 2')
    generate_functions(x, y, x0, len(x))


if __name__ == '__main__':
    main()