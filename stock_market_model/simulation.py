import time

import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.ndimage.interpolation import shift


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

        self.not_h_matrix = self.ones.astype(dtype=np.float) * (1 - p_h)
        self.not_e_matrix = self.ones.astype(dtype=np.float) * (1 - p_e)
        self.d_matrix = self.ones.astype(dtype=np.float) * p_d
        self.convolution_kernel_neighbours = np.array([[0, 1, 0],
                                                       [1, 0, 1],
                                                       [0, 1, 0]])
        self.convolution_kernel_right = np.array([[0, 0, 0],
                                                  [1, 2, 0],
                                                  [0, 0, 0]], dtype=np.byte)
        self.convolution_kernel_left = np.array([[0, 0, 0],
                                                 [0, 2, 1],
                                                 [0, 0, 0]], dtype=np.byte)
        self.convolution_kernel_top = np.array([[0, 0, 0],
                                                [0, 2, 0],
                                                [0, 1, 0]], dtype=np.byte)
        self.convolution_kernel_bottom = np.array([[0, 1, 0],
                                                   [0, 2, 0],
                                                   [0, 0, 0]], dtype=np.byte)

    def step(self):
        am = self.activeness_mask.astype(np.byte)
        b = convolve(am, self.convolution_kernel_bottom)
        t = convolve(am, self.convolution_kernel_top)
        l = convolve(am, self.convolution_kernel_left)
        r = convolve(am, self.convolution_kernel_right)

        can_activate_bot = (b == 2)
        can_activate_bot[self.n - 1, :] = False
        can_activate_top = (t == 2)
        can_activate_top[0, :] = False
        can_activate_left = (l == 2)
        can_activate_left[:, 0] = False
        can_activate_right = (r == 2)
        can_activate_right[:, self.m - 1] = False

        can_activate_n = can_activate_bot.astype(np.byte) \
                         + can_activate_left.astype(np.byte) \
                         + can_activate_right.astype(np.byte) \
                         + can_activate_top.astype(np.byte)
        p = 1 / can_activate_n
        p[p == np.inf] = 0
        will_try_activate_bot = can_activate_bot \
                                & (np.random.binomial(1, p, (self.n, self.m)) == self.ones)

        can_activate_n = can_activate_left.astype(np.byte) \
                         + can_activate_right.astype(np.byte) \
                         + can_activate_top.astype(np.byte)
        p = 1 / can_activate_n
        p[p == np.inf] = 0
        will_try_activate_left = can_activate_left \
                                 & ~will_try_activate_bot \
                                 & (np.random.binomial(1, p, (self.n, self.m)) == self.ones)

        can_activate_n = can_activate_right.astype(np.byte) \
                         + can_activate_top.astype(np.byte)
        p = 1 / can_activate_n
        p[p == np.inf] = 0
        will_try_activate_right = can_activate_right \
                                  & ~will_try_activate_bot \
                                  & ~will_try_activate_left \
                                  & (np.random.binomial(1, p, (self.n, self.m)) == self.ones)

        will_try_activate_top = can_activate_top \
                                & ~will_try_activate_right \
                                & ~will_try_activate_bot \
                                & ~will_try_activate_left

        maybe_activated_by = np.roll(will_try_activate_bot, self.m).astype(np.byte) \
                             + np.roll(will_try_activate_top, -self.m).astype(np.byte) \
                             + np.roll(will_try_activate_left, -1).astype(np.byte) \
                             + np.roll(will_try_activate_right, 1).astype(np.byte)

        p_to_be_activated = self.ones - (self.not_h_matrix ** maybe_activated_by) * self.not_e_matrix

        to_be_activated = np.random.binomial(1, p_to_be_activated, (self.n, self.m)) == self.ones

        has_inactive_neighbours = convolve(~self.activeness_mask, self.convolution_kernel_neighbours)
        to_survive_deactivation = (np.random.binomial(1, self.d_matrix, (self.n, self.m)) == 0) | (
            ~has_inactive_neighbours)

        self.activeness_mask = ~self.activeness_mask & to_be_activated | self.activeness_mask & to_survive_deactivation
        self.matrix.mask = self.activeness_mask

    def get_active_count(self):
        return np.count_nonzero(self.activeness_mask)


def simulate_and_plot(p_hs: list, initial_acitv_freqs: list, max_t):
    ts = range(1, max_t)
    active_count_series_list = []
    assert len(p_hs) == len(initial_acitv_freqs)
    for i in range(len(p_hs)):
        active_count_series_list.append(simulate(ts, Model(p_h=p_hs[i],
                                                           initial_active_freq=initial_acitv_freqs[i])))

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
