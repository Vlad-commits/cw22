import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
import seaborn as sns

from mcmc_sampler import MCMCSampler

multivariate_normal1 = stats.multivariate_normal([0], [[0.5]])
multivariate_normal2 = stats.multivariate_normal([-4], [[0.5]])


def pdf(x): return 0.4 * multivariate_normal1.pdf(x) + 0.6 * multivariate_normal2.pdf(x)


def cdf(x): return 0.4 * multivariate_normal1.cdf(x) + 0.6 * multivariate_normal2.cdf(x)


def create_sample_from_random_walk_proposal_fun(D):
    normal = stats.multivariate_normal([0], [[D]])
    return lambda x: normal.rvs() + x


sample_size = 1000
sampler = MCMCSampler(pdf)

samples = []

for d in [3, 50, 200, 1000]:
    sample = sampler.sample(0, create_sample_from_random_walk_proposal_fun(d), sample_size)
    samples.append(sample)

plt.figure(1)
plt.subplot(121)
X = np.linspace(-8, 5, 10000)
plt.plot(X, [pdf(x) for x in X], label="density")
for i, sample in enumerate(samples):
    sns.distplot(sample, hist=False, label="sample " + str(i) + " histogram")

plt.subplot(122)
ks_test_points = np.geomspace(1, sample_size, num=20, dtype=int)
plt.yscale("log")
plt.xscale("log")
for i, sample in enumerate(samples):
    ds = [stats.kstest(sample[:i], cdf)[0] for i in ks_test_points]
    plt.plot(ks_test_points, ds)

plt.show()
