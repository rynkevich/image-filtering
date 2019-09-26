import numpy as np

from filters.WindowBasedFilter import WindowBasedFilter


class MovingAverageFilter(WindowBasedFilter):
    def __init__(self, window_size):
        super().__init__(window_size)

    def _window_filter(self, image, current_x, current_y):
        accumulator = np.zeros(3)
        for x in range(current_x - self._half_window_size, current_x + self._half_window_size + 1):
            for y in range(current_y - self._half_window_size, current_y + self._half_window_size + 1):
                accumulator += image[x][y]

        return accumulator // (self._window_size ** 2)
