import numpy as np
from numba import prange, njit

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


@njit(fastmath=True)
def func(z: complex, c: complex) -> complex:
    """
    Вычисляет значение функции в точке.

    Parameters:
        z (complex): Точка, в которой происходит вычисление.
        c (complex): Комплексная константа (очень важна для множества Жюлиа, кстати).

    Returns:
        complex: Комплексное значение.
    """
    return z ** 3 + c


@njit(fastmath=True)
def check_point(c: complex, r: float, max_iter: int, z: complex) -> int:
    """
    Проверяет точку на хаотичность.

    Parameters:
        c (complex): Комплексная константа.
        r (float): Радиус сходимости.
        max_iter (int): Максимальное количество итераций.
        z (complex): Проверяемая комплексная точка на хаотичность.

    Returns:
        int: Количество итераций, сделанных для того, чтобы точка ушла на бесконечность. Если оно равно max_it,
    то будем считать, что точка не хаотична.
    """
    res = z
    for i in prange(max_iter):
        if abs(res) > r:
            return i
        res = func(res, c)
    return max_iter


@njit(parallel=True)
def set_calculate(c: complex,
                  min_x: float = -1, max_x: float = 1,
                  min_y: float = -1, max_y: float = 1,
                  nx: int = 1000,
                  ny: int = 1000,
                  r: float = 2,
                  max_iter: int = 200,
                  ) -> np.ndarray:
    """
       Проверяет каждую точку области на хаотичность.

       Parameters:
           c (complex): Комплексная константа.
           min_x: (float): Минимальное значение абсциссы области.
           max_x: (float): Максимальное значение абсциссы области.
           min_y: (float): Минимальное значение ординаты области.
           max_y: (float): Максимальное значение абсциссы области.
           nx: (int): Количество проверяемых точек по абсциссе.
           ny: (int): Количество проверяемых точек по ординате.
           r (float): Радиус сходимости.
           max_iter (int): Максимальное количество итераций.

       Returns:
           np.ndarray: Возвращает массив, в котором для каждой точке на комплексной решетке вычислено значение
       функции check_point. Таким образом мы понимаем, как будет выглядeть множество Жюлиа.
    """

    dx = (max_x - min_x) / nx
    dy = (max_y - min_y) / ny

    result = np.empty((ny + 1, nx + 1), dtype=np.uint16)
    for iy in prange(ny + 1):
        for ix in prange(nx + 1):
            x = min_x + ix * dx
            y = min_y + iy * dy
            it = check_point(c, r, max_iter, complex(x, y))
            result[iy, ix] = it

    return result


def make_plot(res: np.ndarray,
              ind: int = 0) -> plt.plot:
    """
        Визуализирует множество Жюлиа.

        Parameters:
           res: (np.ndarray): Полученный массив с количеством итераций для каждой точки области.
           ind: (int): Индекс сохраняемой фотографии.

        Returns:
           plot: На выходе получаем визуализацию множества Жюлиа при помощи pcolormesh.
    """
    plt.figure(figsize=(8, 8), dpi=200)
    plt.pcolormesh(np.log(res + 1), cmap='twilight')
    plt.savefig(f"./images/image{ind}.png")
    plt.axis('equal')
    plt.show()
    plt.close()


if __name__ == "__main__":
    Jul_1 = set_calculate(complex(-0.4800, -0.5950))
    Jul_2 = set_calculate(complex(0.5200, -0.1250))
    Jul_3 = set_calculate(complex(0.4950, -0.6000))
    Jul_4 = set_calculate(complex(-0.5900, -0.6100))
    Jul_5 = set_calculate(complex(-0.5750, 0.6000))
    Jul_6 = set_calculate(complex(-0.3600, -0.6850))
    Jul_7 = set_calculate(complex(0.0850, -0.7850))

    make_plot(Jul_1, 1)
    make_plot(Jul_2, 2)
    make_plot(Jul_3, 3)
    make_plot(Jul_4, 4)
    make_plot(Jul_5, 5)
    make_plot(Jul_6, 6)
    make_plot(Jul_7, 7)
