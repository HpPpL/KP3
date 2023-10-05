import numpy as np
from numba import prange, njit

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

@njit
def func(z, c):
    return z ** 3 + c

@njit
def check_point(c, r, max_it, z: complex) -> int:
    """Данная функция предназначена для проверки принадлежности отдельной точки множеству Жюлиа. Функция
    возвращает целое число, являющиеся количеством выполненных итераций в проверке (то есть λ) """
    func = lambda z: z ** 2 + c
    res = func(z)
    for i in prange(max_it):
        if np.abs(res) > r:
            return i
        res = func(res)

    return max_it


@njit(parallel=True)
def set_calculate(c: complex,
                  minx=-1,
                  miny=-1,
                  maxx=1,
                  maxy=1,
                  nx: int = 1000,
                  ny: int = 1000,
                  r: float = 2,
                  max_it: int = 200,
                  ) -> np.ndarray:
    dx = (maxx - minx) / nx
    dy = (maxy - miny) / ny

    result = np.empty((ny + 1, nx + 1), dtype=np.uint16)
    for iy in prange(ny + 1):
        for ix in prange(nx + 1):
            x = minx + ix * dx
            y = miny + iy * dy
            it = check_point(c, r, max_it, complex(x, y))
            result[iy, ix] = it

    return result


def make_plot(res):
    plt.figure(figsize=(8, 8), dpi=200)
    plt.pcolormesh(np.log(res + 1), cmap='twilight')
    plt.axis('equal')
    plt.show()
    plt.close()


pi = set_calculate(complex(0.3341, 0.3966))
make_plot(pi)
