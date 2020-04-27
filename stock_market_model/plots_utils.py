import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_active_cells_count_over_time(spins):
    plt.title("Amount of active traders over time")
    plt.xlabel("t")
    plt.ylabel("Amount of active traders")
    plt.plot(range(len(spins)), [pos + neg for (pos, neg) in spins])
    return plt


def plot_cluster_sizes(clusters_sizes_over_time):
    percs = [50, 95, 100]

    data = np.array([np.percentile(clusters_sizes_atm[clusters_sizes_atm != 0], percs) for clusters_sizes_atm in
            clusters_sizes_over_time])
    plt.title("Cluster sizes percentiles over time")
    plt.xlabel("t")
    plt.ylabel("Cluster size")
    for i, perc in enumerate(percs):
        plt.plot(range(len(clusters_sizes_over_time)), data[:,i],label=str(perc)+" percentile")
    plt.legend(loc='best')

    return plt


def plot_trading_dynamic_over_time(spins):
    plt.title("Grid trading dynamics over time")
    plt.xlabel("t")
    plt.ylabel("Fraction of purchasing traders")
    plt.plot(range(len(spins)), [pos / (pos + neg) for (pos, neg) in spins])
    return plt


def qq_plot(model, real):
    percs = np.linspace(0, 100, 100)
    qn_a = np.percentile(model, percs)
    qn_b = np.percentile(real, percs)

    plt.title("Q-Q plot")
    plt.xlabel("model")
    plt.ylabel("real")
    plt.plot(qn_a, qn_b, ls="", marker="o")

    x = np.linspace(np.min((qn_a.min(), qn_b.min())), np.max((qn_a.max(), qn_b.max())))
    plt.plot(x, x, color="k", ls="--")
    return plt


def log_returns_over_time(model, real):
    plt.subplot("211")
    plt.title("Time series of logarithmic returns reproduced with the simulation")
    plt.ylabel("$R$")
    plt.xlabel("$t$")
    plt.ylim((-10, 10))
    plt.plot(range(len(model)), model)

    plt.subplot("212")
    plt.title("Normalized logarithmic returns for the S&P500.",y=1.08)
    plt.ylabel("$R$")
    plt.xlabel("$t$")
    plt.ylim((-10, 10))
    plt.plot(range(len(real)), real)
    plt.tight_layout()


def densities(model, real):
    kde_kws = {'cumulative': False}
    plt.title("pdfs")
    plt.xlabel("$R$")
    plt.ylabel("$f(R)$")
    sns.distplot(model, label="model", kde_kws=kde_kws)
    sns.distplot(real, label="real", kde_kws=kde_kws)
    sns.distplot(np.random.normal(size=10000), label="normal", kde_kws=kde_kws)
    plt.legend(loc="best")
    return plt
