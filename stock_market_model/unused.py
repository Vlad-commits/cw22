from stock_market_model.model import Model
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def animate(model: Model, t: int = 100, steps_per_frame: int = 1):
    figure = plt.figure()
    ca_plot = plt.imshow(model.matrix, cmap='seismic')

    def animation_func(i):
        for i in range(steps_per_frame):
            model.step()
        ca_plot.set_data(model.matrix)
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


def cells_have_inactive_neighbours(activeness_mask):
    n = len(activeness_mask)
    m = len(activeness_mask[0])
    result = np.ndarray((n, m), dtype=np.bool)
    for i in range(n):
        for j in range(m):
            r = False
            if i != 0 and not (activeness_mask[i - 1][j]):
                r = True
            if j != 0 and not (activeness_mask[i][j - 1]):
                r = True
            if i != n - 1 and not activeness_mask[i + 1][j]:
                r = True
            if j != m - 1 and not activeness_mask[i][j + 1]:
                r = True
            result[i][j] = r
    return result


ani = animate(Model(128, 512, 0.0493,initial_active_freq=0.2), t=2000, steps_per_frame=10)
# simulate_and_plot([0.0493], [0.2], 1000)
# asd = simulate_and_plots(models=[Model(p_h=0.051), Model(p_h=0.0485)], ts=range(200), labels=["f", "l"])
plt.show()
