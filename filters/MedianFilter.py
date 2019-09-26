import numpy as np

from filters.WindowBasedFilter import WindowBasedFilter


class MedianFilter(WindowBasedFilter):
    def __init__(self, window_size):
        super().__init__(window_size)

    def _window_filter(self, image, current_x, current_y):
        accumulator = []
        for x in range(current_x - self._half_window_size, current_x + self._half_window_size + 1):
            for y in range(current_y - self._half_window_size, current_y + self._half_window_size + 1):
                accumulator.append(image[x][y])

        accumulator = np.array(accumulator)
        accumulator.sort(0)
        return accumulator[accumulator.shape[0] // 2]
