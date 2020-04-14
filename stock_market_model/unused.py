from stock_market_model.simulation import Model
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate(model: Model, t: int = 100, steps_per_frame: int = 1):
    figure = plt.figure()
    ca_plot = plt.imshow(model.matrix, cmap='seismic')

    def animation_func(i):
        for i in range(steps_per_frame):
            model.step()
        ca_plot.set_data(model.matrix.filled())
        return ca_plot

    plt.colorbar(ca_plot)
    return FuncAnimation(figure, animation_func, frames=int(t / steps_per_frame))
