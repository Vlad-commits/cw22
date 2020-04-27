from pathlib import Path
from stock_market_model import utils

n_try = 5

p = Path("data/" + str(n_try) + "/matrix_over_time.npy")
his_saved = Path("data/" + str(n_try) + "/log_returns.npy")
spins_saved = Path("data/" + str(n_try) + "/spins.npy")
cluster_sizes_saved = Path("data/" + str(n_try) + "/cluster_sizes.npy")

utils.calculate_cluster_sizes(p, cluster_sizes_saved)
utils.calculate_spins_dynamics(p,spins_saved)
utils.calculate_log_returns(p, his_saved)
