from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import stock_market_model.plots as plots



def analyze(n_try = 3):
    df = pd.read_csv("data/real/sp500_paper.csv")
    log_returns_series_path = Path("data/" + str(n_try) + "/log_returns.npy")
    spins_paths = Path("data/" + str(n_try) + "/spins.npy")
    cluster_sizes_paths = Path("data/" + str(n_try) + "/cluster_sizes.npy")
    with log_returns_series_path.open('rb') as f:
        model_log_retruns = np.load(f)
    normalized_model_log_retruns = (model_log_retruns - model_log_retruns.mean()) / model_log_retruns.std()
    with spins_paths.open('rb') as f:
        spins = np.load(f)
    with cluster_sizes_paths.open('rb') as f:
        cluster_sizes = np.load(f, allow_pickle=True)
    sp_log_returns = np.log(df["Close"][1:]) - np.log(df["Close"].shift()[1:])
    normalized_sp_log_returns = (sp_log_returns - sp_log_returns.mean()) / sp_log_returns.std()
    plt.figure("1")
    plots.log_returns_over_time(normalized_model_log_retruns, normalized_sp_log_returns)
    plt.figure("2")
    plots.densities(normalized_model_log_retruns, normalized_sp_log_returns)
    plt.figure("3")
    plots.qq_plot(normalized_model_log_retruns, normalized_sp_log_returns)
    plt.figure("4")
    plots.plot_trading_dynamic_over_time(spins)
    plt.figure("5")
    plots.plot_active_cells_count_over_time(spins)
    plt.figure("6")
    plots.plot_cluster_sizes(cluster_sizes)


