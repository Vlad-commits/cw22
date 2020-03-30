from pandas import np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_cdf_and_ecdfs(samples, cdf):
    X = np.linspace(-8, 5, 10000)
    plt.plot(X, [cdf(x) for x in X], label="real")
    for i, sample in enumerate(samples):
        sns.distplot(sample, hist=False, label="sample " + str(i) + " histogram",
                     kde_kws={'cumulative': True})
