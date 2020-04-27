from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_active_cells_count_over_time(ts, active_count_series_list, labels):
    for index, active_count_series in enumerate(active_count_series_list):
        plt.plot(ts, active_count_series, label=labels[index])
    plt.legend(loc='best')
    return plt


def plot_cluster_sizes(clusters_sizes_list, labels):
    for index, clusters_sizes in enumerate(clusters_sizes_list):
        sns.distplot(clusters_sizes, label=labels[index])
    plt.legend(loc='best')
    return plt


def plot_trading_dynamic_over_time(spins):
    plt.plot(range(len(spins)), [pos / (pos + neg) for (pos, neg) in spins])


def qq_plot(model, real):
    percs = np.linspace(0, 100, 100)
    qn_a = np.percentile(model, percs)
    qn_b = np.percentile(real, percs)
    plt.xlabel("model")
    plt.ylabel("real")
    plt.title("Q-Q plot")
    plt.plot(qn_a, qn_b, ls="", marker="o")
    x = np.linspace(np.min((qn_a.min(), qn_b.min())), np.max((qn_a.max(), qn_b.max())))
    plt.plot(x, x, color="k", ls="--")
    return plt


def log_returns_over_time(model, real):
    plt.title("Time series of returns reproduced with the simulation")
    plt.subplot("211")
    plt.ylim((-10, 10))
    plt.plot(range(len(model)), model)

    plt.title("Normalized logarithmic returns for the S&P500.")
    plt.subplot("212")
    plt.ylim((-10, 10))
    plt.plot(range(len(real)), real)


def densities(model, real):
    kde_kws = {'cumulative': False}
    sns.distplot(model, label="model", kde_kws=kde_kws)
    sns.distplot(real, label="real", kde_kws=kde_kws)
    sns.distplot(np.random.normal(size=10000), label="normal", kde_kws=kde_kws)
    plt.legend(loc="best")
    return plt
