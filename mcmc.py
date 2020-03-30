import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.ndimage.filters import gaussian_filter1d

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

samples = []
# ds = [0.1, 1, 10, 100, 1000]
# ds = [1, 5, 10, 50, 100]
ds = [5, 10]

for d in ds:
    sample = sampler.sample(0, create_sample_from_random_walk_proposal_fun(d), sample_size)
    samples.append(sample)

ks_test_points = range(sample_size)
plt.yscale("log")
plt.xscale("log")
for index, sample in enumerate(samples):
    dn = tests.kstest(sample, cdf, ks_test_points)
    smooth_dn = gaussian_filter1d(dn, sigma=50)
    plt.plot(ks_test_points, smooth_dn, label=str(ds[index]))
    plt.legend(loc='best')
plt.show()
