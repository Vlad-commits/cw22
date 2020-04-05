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


def sample_and_plot(ds, use_mean_of=10, discard_first=0):
    samples = []
    for d in ds:
        samples_for_current_d = []
        for i in range(use_mean_of):
            sample = sampler.sample(0, create_sample_from_random_walk_proposal_fun(d), sample_size,
                                    discard_first=discard_first)
            samples_for_current_d.append(sample)
        samples.append(samples_for_current_d)

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("n")
    plt.ylabel("D_n")

    ks_test_points = range(sample_size)
    for index, samples_for_d in enumerate(samples):
        dns = []
        for sample in samples_for_d:
            dn = tests.kstest(sample, cdf, ks_test_points)
            dns.append(dn)
        avg_dn = np.average(dns)
        plt.plot(ks_test_points, avg_dn, label=str(ds[index]))
        plt.legend(loc='best')
    plt.show()


sample_and_plot([0.1, 1, 10, 100, 1000])
# sample_and_plot([1, 5, 10, 50, 100])
# sample_and_plot([5, 10])
