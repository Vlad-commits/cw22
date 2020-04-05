import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import plots
import tests
from mcmc_sampler import MCMCSampler

multivariate_normal1 = stats.multivariate_normal([0], [[0.5]])
multivariate_normal2 = stats.multivariate_normal([-4], [[0.5]])
bernoulli = stats.bernoulli(0.4)


def naive_rvs(size):
    ber = bernoulli.rvs(size)
    n1 = multivariate_normal1.rvs(size)
    n2 = multivariate_normal2.rvs(size)
    return [n1[i] if (ber[i] == 1) else n2[i] for i in range(size)]


def pdf(x): return 0.4 * multivariate_normal1.pdf(x) + 0.6 * multivariate_normal2.pdf(x)


def cdf(x): return 0.4 * multivariate_normal1.cdf(x) + 0.6 * multivariate_normal2.cdf(x)


def create_sample_from_random_walk_proposal_fun(D):
    normal = stats.multivariate_normal([0], [[D]])
    return lambda x: normal.rvs() + x


sample_size = 10000
sampler = MCMCSampler(pdf)


def n_samples(d, x_0, sample_size, n, discard_first=0):
    return [sampler.sample(x_0, create_sample_from_random_walk_proposal_fun(d), sample_size,
                           discard_first=discard_first) for i in range(n)]


def ks_test(samples, cdf, ks_test_points):
    return [tests.kstest(sample, cdf, ks_test_points) for sample in samples]


def lr_test(samples, cdf, ks_test_points):
    return [tests.lrtest_1dim(sample, cdf, ks_test_points) for sample in samples]


def sample_and_plot(ds, use_mean_of=20, discard_first=0, plot_ecdfs=False, plot_histograms=False) -> List[List[float]]:
    test_points = np.unique(np.logspace(0, np.log10(sample_size - 1), num=20, dtype=int))

    samples = []
    ks_statistics = []
    lr_statistics = []
    for d in ds:
        samples_for_current_proposal = n_samples(d, 0, sample_size, use_mean_of, discard_first=discard_first)
        ks_statistics_for_current_proposal = ks_test(samples_for_current_proposal, cdf, test_points)
        lr_statistics_for_current_proposal = lr_test(samples_for_current_proposal, naive_rvs(sample_size), test_points)

        samples.append(samples_for_current_proposal)
        ks_statistics.append(np.average(ks_statistics_for_current_proposal, axis=0))
        lr_statistics.append(np.average(lr_statistics_for_current_proposal, axis=0))

    labels = ["D=" + str(i) for i in ds]
    if plot_ecdfs:
        plots.plot_cdf_and_ecdfs([s[0] for s in samples], cdf, labels)
    if plot_histograms:
        plots.plot_pdf_and_histograms([s[0] for s in samples], pdf, labels)
    plots.plot_ks(test_points, ks_statistics, labels)
    plots.plot_lr(test_points, lr_statistics, labels)
    plt.show()
    return samples


sample_and_plot([0.1, 1, 10, 100, 1000], plot_ecdfs=True, plot_histograms=True)
