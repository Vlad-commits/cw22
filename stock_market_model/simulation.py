from pathlib import Path

import stock_market_model.model as model

n_try = 4
p = Path("data/" + str(n_try) + "/matrix_over_time.npy")
model.simulate_and_write(model.Model(p_h=0.0493, A=5), 9000, p)