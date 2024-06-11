import numpy as np
from approx import *
import matplotlib.pyplot as plt
import sys

output = []

def squared_error(y_t, y_c):
    return np.sqrt(np.mean((y_t - y_c)**2))


def generate_functions(x, y):
    A = {}
    y_generated = {}
    y_for_plot = {}
    x_s = np.linspace(min(x) - 1, max(x) + 1, 100)

    a, b = linear_approx(x, y)
    A['linear'] = (a, b)
    y_generated['linear'] = a * x + b 
    y_for_plot['linear'] = a * x_s + b 

    a, b, c = poly_2_approx(x, y)
    A['poly2'] = (a, b, c)
    y_generated['poly2'] = a * x ** 2 + b * x + c 
    y_for_plot['poly2'] = a * x_s ** 2 + b * x_s + c 

    a, b, c, d = poly_3_approx(x, y)
    A['poly3'] = (a, b, c, d) 
    y_generated['poly3'] = a * x ** 3 + b * x ** 2 + c * x + d 
    y_for_plot['poly3'] = a * x_s ** 3 + b * x_s ** 2 + c * x_s + d 

    a, b = exp_approx(x, y)
    A['exp'] = (a, b)
    y_generated['exp'] = a * np.exp(b * x)
    y_for_plot['exp'] = a * np.exp(b * x_s)

    a, b = log_approx(x, y)
    A['log'] = (a, b)
    y_generated['log'] = a * np.log(x) + b 
    y_for_plot['log'] = a * np.log(x_s) + b 

    a, b = degree_approx(x, y)
    A['degree'] = (a, b)
    y_generated['degree'] = a * x ** b 
    y_for_plot['degree'] = a * x_s ** b 

    sq_error = {
        key: squared_error(y, y_c) for key, y_c in y_generated.items()
            }
    return A, y_generated, sq_error, y_for_plot


def plot_functions(x, y, y_generated, best_func):
    plt.scatter(x, y, label='Исходные числа', color='black')
    x_s = np.linspace(min(x) - 1, max(x) + 1, 100)

    for func, y_gen in y_generated.items():
        plt.plot(x_s, y_gen, label=func)

    plt.plot(x_s, y_generated[best_func], label=f'Лучшая аппроксимация: {best_func}', color='red', linewidth=2)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.subplots_adjust(right=0.75)
    plt.show()


def comp_pearson_correlation(x, y):
    x_ = np.mean(x)
    y_ = np.mean(y)
    return sum((x - x_) * (y - y_) for x, y in zip(x, y)) / \
            np.sqrt(sum((x - x_) ** 2 for x in x) * sum((y - y_) ** 2 for y in y))

def cout():
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'w') as fin:
            for s in output:
                fin.write(s)
                fin.write('\n')
    else:
        for s in output:
            print(s)


def main():
    x = np.array([3.63, 3.57, 3.5, 3.44, 3.38, 3.33, 3.27, 3.22])
    y = np.array([6.44, 6.23, 6.03, 5.84, 5.66, 5.45, 5.23, 5.06])

    A, y_generated, sq_error, y_for_plot = generate_functions(x, y)

    best_func = min(sq_error)

    pearson_corr = comp_pearson_correlation(x, y)

    ss_res = np.sum((y - y_generated[best_func])**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f"\nКоэффициент корреляции Пирсона для линейной функции: {pearson_corr}")

    print(f"\nКоэффициент детерминации для наилучшей аппроксимирующей функции ({best_func}): {r_squared}")

    plot_functions(x, y, y_for_plot, best_func)

    cout()
    print(5 * '<>')
    print(output)


if __name__ == '__main__':
    main()
    
