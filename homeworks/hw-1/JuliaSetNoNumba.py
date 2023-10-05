# Вариант 5. Множество Жюлиа. Функция: z_(n+1) = z^3_n + c.
# Прежде всего, необходимо понять, что является множеством Жюлиа - множество комлексных точек, для которых итерирование
# функции приведет к уходу на бесконечность.

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class JuliaSet:
    def __init__(self,
                 c: complex,
                 p1: tuple = (-1, -1),
                 p2: tuple = (1, 1),
                 nx: int = 1000,
                 ny: int = 1000,
                 r: float = 2,
                 max_it: int = 200,
                 ):
        self.func = lambda z: z ** 2 + c
        self.minx = p1[0]
        self.miny = p1[1]
        self.maxx = p2[0]
        self.maxy = p2[1]
        self.nx = nx
        self.ny = ny
        self.r = r
        self.max_it = max_it

    def check_point(self, z: complex):
        res = self.func(z)
        for i in range(self.max_it):
            if abs(res) > self.r:
                return i
            res = self.func(res)

        return self.max_it

    def set_calculate(self):
        dx = (self.maxx - self.minx) / self.nx
        dy = (self.maxy - self.miny) / self.ny

        result = []
        for iy in range(self.ny + 1):
            result.append([])
            for ix in range(self.nx + 1):
                x = self.minx + ix * dx
                y = self.miny + iy * dy
                it = self.check_point(complex(x, y))
                result[-1].append(it)

        return result

    def make_plot(self):
        res = np.array(self.set_calculate())
        plt.figure(figsize=(8, 8), dpi=200)
        plt.pcolormesh(np.log(res + 1), cmap='twilight')
        plt.axis('equal')
        plt.show()
        plt.close()


pi = JuliaSet(complex(0.3341, 0.3966))
pi.make_plot()
