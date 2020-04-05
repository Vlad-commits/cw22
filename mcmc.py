from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import plots
import tests
from mcmc_sampler import MCMCSampler

multivariate_normal1 = stats.multivariate_normal([0], [[0.5]])
multivariate_normal2 = stats.multivariate_normal([-4], [[0.5]])


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


def sample_and_plot(ds, use_mean_of=10, discard_first=0, plot_ecdfs=False, plot_histograms=False) -> List[List[float]]:
    ks_test_points = range(sample_size)

    samples = []
    ks_statistics = []
    for d in ds:
        samples_for_current_proposal = n_samples(d, 0, sample_size, use_mean_of, discard_first=discard_first)
        ks_statistics_for_current_proposal = ks_test(samples_for_current_proposal, cdf, ks_test_points)

        samples.append(samples_for_current_proposal)
        ks_statistics.append(np.average(ks_statistics_for_current_proposal, axis=0))

    if plot_ecdfs:
        plots.plot_cdf_and_ecdfs([s[0] for s in samples], cdf)
    if plot_histograms:
        plots.plot_pdf_and_histograms([s[0] for s in samples], pdf)
    plots.plot_ks(ks_test_points, ks_statistics, [str(i) for i in ds])
    plt.show()
    return samples


sample_and_plot([0.1, 1, 10, 100, 1000], plot_ecdfs=True, plot_histograms=True)
