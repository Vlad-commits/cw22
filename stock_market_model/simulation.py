import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from scipy.ndimage import convolve


class Model:
    def __init__(self,
                 n=512,
                 m=128,
                 # probability that an active trader can turn one of his inactive neighbors into an active one
                 p_h=0.0485,
                 # probability to diffuse and become inactive when having at least one inactive neighbour
                 p_d=0.05,
                 # probability to spontaneously enter market dynamics
                 p_e=0.0001,
                 #
                 initial_active_freq=0.1
                 ):
        self.p_h = p_h
        self.p_e = p_e
        self.p_d = p_d
        self.n = n
        self.m = m

        self.ones = np.ones((self.n, self.m), dtype=np.byte)
        self.fours = 4 * self.ones

        self.activeness_mask = np.random.random(self.n * self.m).reshape(
            (self.n, self.m)) < (self.ones * initial_active_freq)

        self.matrix = ma.masked_array(np.zeros((n, m), dtype=np.byte), mask=self.activeness_mask,
                                      fill_value=1)

        self.not_h_matrix = self.ones * (1 - p_h)
        self.e_matrix = self.ones * p_e
        self.d_matrix = self.ones * p_d
        self.convolution_kernel = np.array([[0, 1, 0],
                                            [1, 0, 1],
                                            [0, 1, 0]], dtype=np.byte)

    def step(self):
        n_active_neighbours = convolve(np.array(self.activeness_mask, dtype=np.byte),
                                       self.convolution_kernel)

        # p_to_be_activated = self.ones - self.not_h_matrix ** n_active_neighbours  * (self.ones - self.e_matrix)
        p_to_be_activated = self.ones - self.not_h_matrix ** (n_active_neighbours > 0) * (self.ones - self.e_matrix)

        to_be_activated = np.random.binomial(1, p_to_be_activated, (self.n, self.m)) == self.ones

        to_survive_deactivation = (np.random.binomial(1, self.d_matrix, (self.n, self.m)) == 0) | (
                n_active_neighbours == self.fours)

        self.activeness_mask = ~self.activeness_mask & to_be_activated | self.activeness_mask & to_survive_deactivation
        self.matrix.mask = self.activeness_mask

    def get_active_count(self):
        return np.count_nonzero(self.activeness_mask)


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


def simulate_and_plot(p_hs: list, max_t):
    ts = range(1, max_t)
    active_count_series_list = []
    for p_h in p_hs:
        active_count_series_list.append(simulate(ts, Model(p_h=p_h)))

    for index, active_count_series in enumerate(active_count_series_list):
        plt.plot(ts, active_count_series, label=p_hs[index])
    plt.legend(loc='best')


def simulate(ts, model: Model):
    active_counts = []
    for t in ts:
        model.step()
        active_count = model.get_active_count()
        active_counts.append(active_count)
    return active_counts
