import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
import seaborn as sns

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
ds = [0.1, 1, 10, 100, 1000]

for d in ds:
    sample = sampler.sample(0, create_sample_from_random_walk_proposal_fun(d), sample_size)
    samples.append(sample)

plt.figure(1)
plt.subplot(121)
X = np.linspace(-8, 5, 10000)
plt.plot(X, [cdf(x) for x in X], label="real")
for i, sample in enumerate(samples):
    sns.distplot(sample, hist=False, label="sample " + str(i) + " histogram",
                 kde_kws={'cumulative': True})

plt.subplot(122)
ks_test_points = range(sample_size)
plt.yscale("log")
plt.xscale("log", basex=2)
for index, sample in enumerate(samples):
    dn2 = tests.kstest(sample, cdf, ks_test_points)
    plt.plot(ks_test_points, dn2, label=str(ds[index]))
    plt.legend(loc='best')
plt.show()
