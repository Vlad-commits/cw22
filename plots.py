from pandas import np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d


def plot_cdf_and_ecdfs(samples, cdf):
    X = np.linspace(-8, 5, 10000)
    plt.plot(X, [cdf(x) for x in X], label="real")
    for i, sample in enumerate(samples):
        sns.distplot(sample, hist=False, label="sample " + str(i) + " histogram",
                     kde_kws={'cumulative': True})


def plot_ks_tests(samples,ks_test_points,labels):
    plt.yscale("log")
    plt.xscale("log")
    for index, sample in enumerate(samples):
        smooth_dn = gaussian_filter1d(dn, sigma=50)
        plt.plot(ks_test_points, smooth_dn, label=labels[index])
        plt.legend(loc='best')
    plt.show()
