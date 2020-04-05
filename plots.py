from pandas import np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_cdf_and_ecdfs(samples, cdf, left=-8, right=5):
    X = np.linspace(left, right, 10000)
    plt.plot(X, [cdf(x) for x in X], label="real")
    for i, sample in enumerate(samples):
        sns.distplot(sample, hist=False, label="sample " + str(i) + " ECDF",
                     kde_kws={'cumulative': True})


def plot_pdf_and_histograms(samples, pdf, left=-8, right=5):
    X = np.linspace(left, right, 10000)
    plt.plot(X, [pdf(x) for x in X], label="real")
    for i, sample in enumerate(samples):
        sns.distplot(sample, hist=False, label="sample " + str(i) + " histogram")
