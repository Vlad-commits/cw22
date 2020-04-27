from pathlib import Path
from stock_market_model import utils

n_try = 4

p = Path("data" + str(n_try) + "/maxtrix_over_time.npy")
his_saved = Path("data" + str(n_try) + "/log_returns.npy")

spins = utils.read_and_compute_spins(p)

# utils.calculate_log_returns(p, his_saved)
