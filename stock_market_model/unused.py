from stock_market_model.simulation import Model
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def animate(model: Model, t: int = 100, steps_per_frame: int = 1):
    figure = plt.figure()
    ca_plot = plt.imshow(model.matrix, cmap='seismic')

    def animation_func(i):
        for i in range(steps_per_frame):
            model.step()
        ca_plot.set_data(model.matrix.filled())
        return ca_plot

    plt.colorbar(ca_plot)
    return FuncAnimation(figure, animation_func, frames=int(t / steps_per_frame))


def can_r(a):
    n = len(a)
    m = len(a[0])
    result = np.zeros((n, m), dtype=np.bool)
    for i in range(n):
        for j in range(m):
            if (a[i][j]) and (j < (m - 1)) and (not a[i][j + 1]):
                result[i][j] = True
    return result


def can_l(a):
    n = len(a)
    m = len(a[0])
    result = np.zeros((n, m), dtype=np.bool)
    for i in range(n):
        for j in range(m):
            if (a[i][j]) and (j > 0) and (not a[i][j - 1]):
                result[i][j] = True
    return result


def can_t(a):
    n = len(a)
    m = len(a[0])
    result = np.zeros((n, m), dtype=np.bool)
    for i in range(n):
        for j in range(m):
            if (a[i][j]) and (i > 0) and (not a[i - 1][j]):
                result[i][j] = True
    return result


def can_b(a):
    n = len(a)
    m = len(a[0])
    result = np.zeros((n, m), dtype=np.bool)
    for i in range(n):
        for j in range(m):
            if (a[i][j]) and (i < n - 1) and (not a[i + 1][j]):
                result[i][j] = True
    return result
