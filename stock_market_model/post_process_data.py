from pathlib import Path
from stock_market_model import utils


n_try = 4

p = Path("data/" + str(n_try) + "/matrix_over_time.npy")
his_saved = Path("data/" + str(n_try) + "/log_returns.npy")
spins_saved = Path("data/" + str(n_try) + "/spins.npy")

spins = utils.calculate_spins_dynamics(p,spins_saved)
# utils.calculate_log_returns(p, his_saved)
