import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from stock_market_model.model import Model


# def simulate_and_plot(p_hs: list, initial_acitv_freqs: list, max_t, p_d=0.049, p_e=0.0001):
#     ts = range(1, max_t)
#     assert len(p_hs) == len(initial_acitv_freqs)
#     models = [Model(p_h=p_hs[i], p_d=p_d, p_e=p_e, initial_active_freq=initial_acitv_freqs[i]) for i in
#               range(len(p_hs))]
#     labels = p_hs
#
#     return simulate_and_plots(labels, models, ts)
#
#
# def simulate_and_plots(labels, models, ts):
#     active_count_series_list = []
#     clusters_sizes_list = []
#     for model in models:
#         active_count_series, clusters_sizes = simulate(ts, model)
#         active_count_series_list.append(active_count_series)
#         clusters_sizes_list.append(clusters_sizes)
#     return plot(active_count_series_list, clusters_sizes_list, labels, ts)





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


def calculate_log_returns(simulation_file_path, path_to_save):
    his = read_and_compute_his(simulation_file_path)
    with path_to_save.open("ab") as f:
        np.save(f, his)


def read_and_compute_spins(p, n=512, m=128):
    spins = []

    def callback(matrix):
        t2 = (np.count_nonzero(matrix == -1), np.count_nonzero(matrix == 1))
        spins.append(t2)

    read_and_invoke(p, callback)
    return spins

def calculate_spins_dynamics(simulation_file_path, path_to_save):
    his = read_and_compute_spins(simulation_file_path)
    with path_to_save.open("ab") as f:
        np.save(f, his)