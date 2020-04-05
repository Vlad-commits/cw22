from pandas import np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_cdf_and_ecdfs(samples, cdf, left=-8, right=5):
    plt.figure(1)
    X = np.linspace(left, right, 10000)
    plt.plot(X, [cdf(x) for x in X], label="real")
    for i, sample in enumerate(samples):
        sns.distplot(sample, hist=False, label="sample " + str(i) + " ECDF",
                     kde_kws={'cumulative': True})


def plot_pdf_and_histograms(samples, pdf, left=-8, right=5):
    plt.figure(2)

    X = np.linspace(left, right, 10000)
    plt.plot(X, [pdf(x) for x in X], label="real")
    for i, sample in enumerate(samples):
        sns.distplot(sample, hist=False, label="sample " + str(i) + " histogram")


def plot_ks(ks_test_points, ks_statistics, labels):
    plt.figure(3)

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("n")
    plt.ylabel("D_n")
    for i in range(len(ks_statistics)):
        ks = ks_statistics[i]
        plt.plot(ks_test_points, ks, label=labels[i]);
    plt.legend(loc='best')
