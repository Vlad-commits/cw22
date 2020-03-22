import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as stats
import seaborn as sns

from mcmc_sampler import MCMCSampler

multivariate_normal1 = stats.multivariate_normal([0], [[0.5]])
multivariate_normal2 = stats.multivariate_normal([-4], [[0.5]])


def pdf(x): return 0.4 * multivariate_normal1.pdf(x) + 0.6 * multivariate_normal2.pdf(x)


Q = stats.multivariate_normal([0], [[3]])


def sample_from_random_walk_proposal(x):
    return Q.rvs() + x


def sample(target_pdf, x_0, sample_from_proposal_fun, n_samples):
    uniform = stats.uniform()
    x_prev = x_0
    result = [x_prev]
    for i in range(0, n_samples):
        x_prime = sample_from_proposal_fun(x_prev)
        acceptance_probability = min(target_pdf(x_prime) / target_pdf(x_prev), 1)
        if acceptance_probability >= uniform.rvs():
            x_prev = x_prime
            result.append(x_prev)
    return result


sampler = MCMCSampler(pdf)
(samples, accepted) = sampler.sample(0, sample_from_random_walk_proposal, 10000)

print(accepted/len(samples))


X = np.linspace(-8, 5, 10000)
plt.plot(X, [pdf(x) for x in X], color="r", label="density")
ax = sns.distplot(samples, color="y", hist=False, label="sample histogram")
plt.show()
