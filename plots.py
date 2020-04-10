import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_cdf_and_ecdfs(samples, cdf, labels, left=-8, right=5):
    plt.figure(1)
    plt.title("Cumulative distribution function and empirical cumulative distribution functions")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    X = np.linspace(left, right, 10000)
    plt.plot(X, [cdf(x) for x in X], label="CDF")
    for i, sample in enumerate(samples):
        sns.distplot(sample, hist=False, label=labels[i],
                     kde_kws={'cumulative': True})


def plot_pdf_and_histograms(samples, pdf, labels, left=-8, right=5):
    plt.figure(2)
    plt.title("Probability density function and histograms.")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    X = np.linspace(left, right, 10000)
    plt.plot(X, [pdf(x) for x in X], label="PDF")
    for i, sample in enumerate(samples):
        sns.distplot(sample, hist=False, label=labels[i])


def plot_ks(test_points, ks_statistics, labels):
    plt.figure(3)
    plt.title("Kolmogorov-Smirnov test.")

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("n")
    plt.ylabel("$D_n$")
    for i in range(len(ks_statistics)):
        ks = ks_statistics[i]
        plt.plot(test_points, ks, label=labels[i])
    plt.legend(loc='best')


def plot_lr(test_points, lr_statistics, labels):
    plt.figure(4)
    plt.title("Lehmann-Rosenblatt test.")

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("n")
    plt.ylabel("$\\omega^2_{n,n}$")
    for i in range(len(lr_statistics)):
        lr = lr_statistics[i]
        plt.plot(test_points, lr, label=labels[i])
    plt.legend(loc='best')


def plot_acceptance_rates(acceptance_rates, labels):
    plt.figure(5)
    plt.title("Acceptance rates")
    plt.boxplot(acceptance_rates, labels=labels)
