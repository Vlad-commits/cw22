from pathlib import Path

import stock_market_model.model as model

n_try = 7
p = Path("data/" + str(n_try) + "/matrix_over_time.npy")
model.simulate_and_write(model.Model(p_h=0.0498, A=2.8), 10000, p)
