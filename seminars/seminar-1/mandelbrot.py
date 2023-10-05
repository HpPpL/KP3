import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def escape_proto(c: complex,
                 r: float = 2,
                 max_it=200) -> int:
    z = 0.0 + 0.0j
    i = 0
    for i in range(max_it):
        z = z ** 2 + c
        if abs(z) > r:
            return i

    return max_it


def mandelbrot_proto(minx: float,
                     maxx: float,
                     miny: float,
                     maxy: float,
                     nx: int = 200,
                     ny: int = 200,
                     r: float = 2,
                     max_it: int = 200) -> list[list[int]]:
    dx = (maxx - minx) / nx
    dy = (maxy - miny) / ny

    # result_2 = [[minx + x * dx, miny + y * dy] for y in range(ny + 1) for x in range(nx + 1)]
    # print(result_2)
    result = []
    for iy in range(ny + 1):
        result.append([])
        for ix in range(nx + 1):
            x = minx + ix * dx
            y = miny + iy * dy
            c = x + y * 1j
            it = escape_proto(c, r=r, max_it=max_it)
            result[-1].append(it)

    return result


arr_proto = np.array(mandelbrot_proto(
    -1.5, 0.5,
    -1.0, 1.0
))

plt.figure()
plt.pcolormesh(np.log(arr_proto + 1), cmap='twilight')
plt.axis('equal')
# plt.show()

#  --- - - - - -- - - -
# Теперь сделаем через numba.
from numba import njit, prange


@njit
def escape_njit(c: complex,
                r: float = 2,
                max_it=200) -> int:
    z = 0.0 + 0.0j
    i = 0
    for i in range(max_it):
        z = z ** 2 + c
        if abs(z) > r:
            return i

    return max_it


@njit(parallel=True)
def mandelbrot_njit(minx: float,
                    maxx: float,
                    miny: float,
                    maxy: float,
                    nx: int = 200,
                    ny: int = 200,
                    r: float = 2,
                    max_it: int = 200) -> np.ndarray:
    dx = (maxx - minx) / nx
    dy = (maxy - miny) / ny

    result = np.empty((ny + 1, nx + 1), dtype=np.uint16)
    for iy in prange(ny + 1):
        # result.append([])
        for ix in range(nx + 1):
            x = minx + ix * dx
            y = miny + iy * dy
            c = x + y * 1j
            it = escape_njit(c, r=r, max_it=max_it)
            # result[-1].append(it)
            result[iy, ix] = it

    return result


arr_njit = np.array(mandelbrot_njit(
    -1.5, 0.5,
    -1.0, 1.0
))

print(escape_proto(1.0) == escape_njit(1.0))

# Дальше замерим время и поймем, что njit выполняется в 40 раз быстрее (800 ms против 20 ms)..
# Потом заменили range на range, и получилось (8 ms)