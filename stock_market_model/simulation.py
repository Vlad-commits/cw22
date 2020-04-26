import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage import convolve


class Model:
    def __init__(self,
                 n=512,
                 m=128,
                 # probability that an active trader can turn one of his inactive neighbors into an active one
                 p_h=0.0485,
                 # probability to diffuse and become inactive when having at least one inactive neighbour
                 p_d=0.049,
                 # probability to spontaneously enter market dynamics
                 p_e=0.0001,
                 A=1.8,
                 h=0,
                 #
                 initial_active_freq=0.2
                 ):
        self.p_h = p_h
        self.p_e = p_e
        self.p_d = p_d
        self.n = n
        self.m = m

        self.ones = np.ones((self.n, self.m), dtype=np.byte)

        initialy_active = (np.random.random(self.n * self.m).reshape(
            (self.n, self.m)) < (self.ones * initial_active_freq))
        coin_flip = self.flip_coin()
        self.matrix = initialy_active * (-1 * coin_flip) + initialy_active * (~coin_flip)

        self.not_h_matrix = self.ones.astype(dtype=np.float64) * (1 - p_h)
        self.h_matrix = self.ones.astype(dtype=np.float64) * p_h
        self.not_e_matrix = self.ones.astype(dtype=np.float64) * (1 - p_e)
        self.e_matrix = self.ones.astype(dtype=np.float64) * p_e
        self.d_matrix = self.ones.astype(dtype=np.float64) * p_d

        self.A = A
        self.a = 2 * A
        self.h = h

    @staticmethod
    def ksi(cluster_numbers):
        max_number = max(cluster_numbers)
        values = np.random.default_rng().uniform(-1, 1, max_number + 1)
        return np.array([values[i] if (i in cluster_numbers) else 0 for i in range(max_number + 1)])

    @staticmethod
    def eta(cluster_size):
        return np.random.default_rng().uniform(-1, 1, (cluster_size, cluster_size))

    @staticmethod
    def zeta(cluster_numbers):
        max_number = max(cluster_numbers)
        values = np.random.default_rng().uniform(-1, 1, max_number + 1)
        return np.array([values[i] if (i in cluster_numbers) else 0 for i in range(max_number + 1)])

    def step(self):
        activeness_mask = self.matrix != 0

        to_be_activated = self.get_activated_by_neighbours(activeness_mask)

        coin_flip = self.flip_coin()
        to_be_activated = to_be_activated * ((-1) * coin_flip) + to_be_activated * (~coin_flip)
        to_be_activated = to_be_activated * (~activeness_mask)

        has_inactive_neighbours = Model.get_cells_having_inactive_neighbours(activeness_mask)
        to_survive_deactivation = (np.random.default_rng().binomial(1, self.d_matrix,
                                                                    (self.n, self.m)) == 0) | (
                                      ~has_inactive_neighbours)

        self.matrix = self.matrix * (activeness_mask & to_survive_deactivation) + to_be_activated

        cluster_numbers_for_cells, cluster_numbers = Model.get_cluster_numbers(self.matrix != 0, self.n, self.m)
        cluster_sizes = Model.get_cluster_sizes(cluster_numbers_for_cells, cluster_numbers)

        ksis = Model.ksi(cluster_numbers)
        zetas = Model.zeta(cluster_numbers)

        for cluster_number in cluster_numbers:
            current_cluster = self.matrix[cluster_numbers_for_cells == cluster_number]
            current_cluster_size = cluster_sizes[cluster_number]

            ksi = ksis[cluster_number]
            eta = Model.eta(current_cluster_size)
            A = eta * self.a + self.A * ksi
            h = self.h * zetas[cluster_number]

            I_for_current_cluster_cells = np.matmul(A, current_cluster) / cluster_sizes[cluster_number] + h

            p_for_current_cluster_cells = 1 / (1 + np.exp(-2 * I_for_current_cluster_cells))

            new_spins = np.random.default_rng().binomial(1, p_for_current_cluster_cells, current_cluster_size)
            new_spins[new_spins == 0] = -1

            self.matrix[cluster_numbers_for_cells == cluster_number] = new_spins
        return self.matrix

    def get_activated_by_neighbours(self, activeness_mask):
        can_activate_bot = Model.get_cells_can_activate_bot(activeness_mask)
        can_activate_top = Model.get_cells_can_activate_top(activeness_mask)
        can_activate_left = Model.get_cells_can_activate_left(activeness_mask)
        can_activate_right = Model.get_cells_can_activate_right(activeness_mask)
        can_activate_n = can_activate_bot.astype(np.byte) \
                         + can_activate_left.astype(np.byte) \
                         + can_activate_right.astype(np.byte) \
                         + can_activate_top.astype(np.byte)
        p = 1 / can_activate_n
        p[p == np.inf] = 0
        will_try_activate_bot = can_activate_bot \
                                & (np.random.default_rng().binomial(1, p, (self.n, self.m)) == self.ones)
        can_activate_n = can_activate_left.astype(np.byte) \
                         + can_activate_right.astype(np.byte) \
                         + can_activate_top.astype(np.byte)
        p = 1 / can_activate_n
        p[p == np.inf] = 0
        will_try_activate_left = can_activate_left \
                                 & ~will_try_activate_bot \
                                 & (np.random.default_rng().binomial(1, p, (self.n, self.m)) == self.ones)
        can_activate_n = can_activate_right.astype(np.byte) \
                         + can_activate_top.astype(np.byte)
        p = 1 / can_activate_n
        p[p == np.inf] = 0
        will_try_activate_right = can_activate_right \
                                  & ~will_try_activate_bot \
                                  & ~will_try_activate_left \
                                  & (np.random.default_rng().binomial(1, p, (self.n, self.m)) == self.ones)
        will_try_activate_top = can_activate_top \
                                & ~will_try_activate_right \
                                & ~will_try_activate_bot \
                                & ~will_try_activate_left
        maybe_activated_by = np.roll(will_try_activate_bot, self.m).astype(np.byte) \
                             + np.roll(will_try_activate_top, -self.m).astype(np.byte) \
                             + np.roll(will_try_activate_left, -1).astype(np.byte) \
                             + np.roll(will_try_activate_right, 1).astype(np.byte)
        p_to_be_activated = self.ones - (self.not_h_matrix ** maybe_activated_by) * self.not_e_matrix
        to_be_activated = np.random.default_rng().binomial(1, p_to_be_activated, (self.n, self.m)) == self.ones
        return to_be_activated

    def flip_coin(self):
        return np.random.default_rng().binomial(1, 0.5, (self.n, self.m)) == 0

    @staticmethod
    def get_cells_having_inactive_neighbours(activeness_mask):
        return convolve(~activeness_mask, Model.convolution_kernel_neighbours, mode='constant')

    def cluster_sizes(self):
        activeness_mask = self.matrix != 0
        cluster_numbers_for_cells, cluster_numbers = Model.get_cluster_numbers(activeness_mask, self.n, self.m)
        return Model.get_cluster_sizes(cluster_numbers_for_cells, cluster_numbers)

    @staticmethod
    def get_cluster_numbers(activeness_mask, n, m):
        next_cluster_number = 1
        cluster_numbers = []
        cluster_numbers_for_cells = np.zeros((n, m), dtype=int)
        for i in range(n):
            for j in range(m):
                if activeness_mask[i][j]:
                    left_number = 0
                    top_number = 0
                    if i != 0:
                        left_number = cluster_numbers_for_cells[i - 1][j]
                    if j != 0:
                        top_number = cluster_numbers_for_cells[i][j - 1]

                    left = left_number != 0
                    top = top_number != 0
                    if left & top:
                        cluster_numbers_for_cells[i][j] = left_number
                        if left_number != top_number:
                            cluster_numbers_for_cells[cluster_numbers_for_cells == left_number] = top_number
                            cluster_numbers.remove(left_number)
                        pass
                    elif left:
                        cluster_numbers_for_cells[i][j] = left_number
                    elif top:
                        cluster_numbers_for_cells[i][j] = top_number
                    else:
                        cluster_numbers_for_cells[i][j] = next_cluster_number
                        cluster_numbers.append(next_cluster_number)
                        next_cluster_number += 1

        return (cluster_numbers_for_cells, cluster_numbers)

    @staticmethod
    def get_cluster_sizes(cluster_numbers_for_cells, cluster_numbers):
        max_number = max(cluster_numbers)

        return np.array([np.count_nonzero(cluster_numbers_for_cells == i) if (i in cluster_numbers) else 0 for i in
                         range(max_number + 1)])

    @staticmethod
    def get_cells_can_activate_right(activeness_mask, already_activated=None):
        if already_activated is None:
            r = convolve(activeness_mask.astype(np.byte), Model.convolution_kernel_right)
            can_activate_right = (r == 2)
            can_activate_right[:, len(activeness_mask[0]) - 1] = False
            return can_activate_right
        else:
            return Model.get_cells_can_activate_right(activeness_mask) \
                   & Model.get_cells_can_activate_right(activeness_mask | already_activated)

    @staticmethod
    def get_cells_can_activate_left(activeness_mask, already_activated=None):
        if already_activated is None:
            l = convolve(activeness_mask.astype(np.byte), Model.convolution_kernel_left)
            can_activate_left = (l == 2)
            can_activate_left[:, 0] = False
            return can_activate_left
        else:
            return Model.get_cells_can_activate_left(activeness_mask) \
                   & Model.get_cells_can_activate_left(activeness_mask | already_activated)

    @staticmethod
    def get_cells_can_activate_top(activeness_mask, already_activated=None):
        if already_activated is None:
            t = convolve(activeness_mask.astype(np.byte), Model.convolution_kernel_top)
            can_activate_top = (t == 2)
            can_activate_top[0, :] = False
            return can_activate_top
        else:
            return Model.get_cells_can_activate_top(activeness_mask) \
                   & Model.get_cells_can_activate_top(activeness_mask | already_activated)

    @staticmethod
    def get_cells_can_activate_bot(activeness_mask, already_activated=None):
        if already_activated is None:
            b = convolve(activeness_mask.astype(np.byte), Model.convolution_kernel_bottom)
            can_activate_bot = (b == 2)
            can_activate_bot[len(activeness_mask) - 1, :] = False
            return can_activate_bot
        else:
            return Model.get_cells_can_activate_bot(activeness_mask) \
                   & Model.get_cells_can_activate_bot(activeness_mask | already_activated)

    def get_active_count(self):
        return np.count_nonzero(self.matrix)


Model.convolution_kernel_right = np.array([[0, 0, 0],
                                           [1, 2, 0],
                                           [0, 0, 0]], dtype=np.byte)
Model.convolution_kernel_left = np.array([[0, 0, 0],
                                          [0, 2, 1],
                                          [0, 0, 0]], dtype=np.byte)
Model.convolution_kernel_top = np.array([[0, 0, 0],
                                         [0, 2, 0],
                                         [0, 1, 0]], dtype=np.byte)
Model.convolution_kernel_bottom = np.array([[0, 1, 0],
                                            [0, 2, 0],
                                            [0, 0, 0]], dtype=np.byte)
Model.convolution_kernel_neighbours = np.array([[0, 1, 0],
                                                [1, 0, 1],
                                                [0, 1, 0]], dtype=np.byte)


def simulate_and_plot(p_hs: list, initial_acitv_freqs: list, max_t, p_d=0.049, p_e=0.0001):
    ts = range(1, max_t)
    assert len(p_hs) == len(initial_acitv_freqs)
    models = [Model(p_h=p_hs[i], p_d=p_d, p_e=p_e, initial_active_freq=initial_acitv_freqs[i]) for i in
              range(len(p_hs))]
    labels = p_hs

    return simulate_and_plots(labels, models, ts)


def simulate_and_plots(labels, models, ts):
    active_count_series_list = []
    clusters_sizes_list = []
    for model in models:
        active_count_series, clusters_sizes = simulate(ts, model)
        active_count_series_list.append(active_count_series)
        clusters_sizes_list.append(clusters_sizes)
    return plot(active_count_series_list, clusters_sizes_list, labels, ts)


def plot(active_count_series_list, clusters_sizes_list, labels, ts):
    plt.subplot(121)
    for index, active_count_series in enumerate(active_count_series_list):
        plt.plot(ts, active_count_series, label=labels[index])
    plt.legend(loc='best')
    plt.subplot(122)
    for index, clusters_sizes in enumerate(clusters_sizes_list):
        sns.distplot(clusters_sizes, label=labels[index])
    return plt.legend(loc='best')


def simulate(ts, model: Model):
    active_counts = []
    for t in ts:
        model.step()
        active_count = model.get_active_count()
        active_counts.append(active_count)
    clusters_sizes = model.cluster_sizes()
    return active_counts, clusters_sizes[clusters_sizes != 0]


def simulate_and_write(model: Model, max_time, path):
    with path.open("ab") as f:
        for t in range(max_time):
            matrix = model.step()
            np.save(f, matrix)


def read_and_invoke(p, callback):
    with p.open('rb') as f:
        fsz = os.fstat(f.fileno()).st_size
        out = np.load(f)
        callback(out)
        while f.tell() < fsz:
            out = np.load(f)
            callback(out)


def read_and_compute_his(p, n=512, m=128):
    his = []

    def callback(matrix):
        cluster_numbers_for_cells, cluster_numbers = Model.get_cluster_numbers(matrix != 0, n, m)
        cluster_sizes = Model.get_cluster_sizes(cluster_numbers_for_cells, cluster_numbers)

        temp = np.zeros((n, m))
        for cluster_number in cluster_numbers:
            temp[cluster_numbers_for_cells == cluster_number] = cluster_sizes[cluster_number]
        hi = np.sum(temp * matrix) / (np.sum(temp))
        his.append(hi)

    read_and_invoke(p, callback)
    return np.array(his)


p = Path("2_ph0493t9000.npy")
his_saved = Path("2_his.npy")

st = time.time()

simulate_and_write(Model(p_h=0.0493), 9000, p)

writed = time.time()
print(writed - st)

his = read_and_compute_his(p)
with his_saved.open("ab") as f:
    np.save(f, his)
print(writed - time.time())
