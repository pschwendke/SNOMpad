import numpy as np
from bokeh.plotting import ColumnDataSource


class PlottingBuffer:
    def __init__(self, n: int, names: list):
        """ simple wrapper around a bokeh ColumnDataSource for easier access
        """
        init_data = {h: np.ones(n) for h in names}
        init_data.update({'t': np.arange(-n + 1, 1)})
        self.buffer = ColumnDataSource(init_data)
        self.names = names

    def put(self, data: np.ndarray):
        for n, key in enumerate(self.names):
            channel = self.buffer.data[key]
            channel[:-1] = channel[1:]
            channel[-1] = data[n]
            self.buffer.data[key] = channel

    def avg(self, key):
        return self.buffer.data[key].mean()

    def std(self, key):
        return self.buffer.data[key].std()
