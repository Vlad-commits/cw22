from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

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


def sample_and_test(d, discard_first, ks_test_points, use_mean_of):
    samples_for_current_proposal = []
    ks_statistics_for_current_proposal = []
    for i in range(use_mean_of):
        sample = sampler.sample(0, create_sample_from_random_walk_proposal_fun(d), sample_size,
                                discard_first=discard_first)
        samples_for_current_proposal.append(sample)

        dn = tests.kstest(sample, cdf, ks_test_points)
        ks_statistics_for_current_proposal.append(dn)
    return ks_statistics_for_current_proposal, samples_for_current_proposal


def sample_and_plot(ds, use_mean_of=10, discard_first=0) -> List[List[float]]:
    ks_test_points = range(sample_size)

    samples = []
    ks_statistics = []
    for d in ds:
        ks_statistics_for_current_proposal, samples_for_current_proposal = \
            sample_and_test(d, discard_first, ks_test_points, use_mean_of)

        samples.append(samples_for_current_proposal)
        ks_statistics.append(np.average(ks_statistics_for_current_proposal, axis=0))

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("n")
    plt.ylabel("D_n")

    for index, ks in enumerate(ks_statistics):
        plt.plot(ks_test_points, ks, label=str(ds[index]));
    plt.legend(loc='best')
    plt.show()
    return samples


sample_and_plot([0.1, 1, 10, 100, 1000])
# sample_and_plot([1, 5, 10, 50, 100])
# sample_and_plot([5, 10])
