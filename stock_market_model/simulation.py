from pathlib import Path

from stock_market_model import utils
from stock_market_model.model import Model

p = Path("3_ph0493t9000a2dot5.npy")
utils.simulate_and_write(Model(p_h=0.0493), 9000, p)