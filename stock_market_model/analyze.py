from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import stock_market_model.plots as plots

df = pd.read_csv("data/real/sp500_paper.csv")
log_returns_series_path = Path("data/2/log_returns.npy")

sp_log_returns = np.log(df["Close"][1:]) - np.log(df["Close"].shift()[1:])
normalized_sp_log_returns = (sp_log_returns - sp_log_returns.mean()) / sp_log_returns.std()

with log_returns_series_path.open('rb') as f:
    model_log_retruns = np.load(f)

normalized_model_log_retruns = (model_log_retruns - model_log_retruns.mean()) / model_log_retruns.std()

plt.figure("1")
plots.log_returns_over_time(normalized_model_log_retruns,normalized_sp_log_returns)

plt.figure("2")
plots.densities(normalized_model_log_retruns,normalized_sp_log_returns)

plt.figure("3")
plots.qq_plot(normalized_model_log_retruns,normalized_sp_log_returns)
plt.show()
