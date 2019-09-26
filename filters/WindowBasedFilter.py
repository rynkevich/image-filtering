from abc import ABC, abstractmethod

import numpy as np


class WindowBasedFilter(ABC):
    def __init__(self, window_size):
        self._validate_window(window_size)
        self._window_size = window_size
        self._half_window_size = self._window_size // 2

    @property
    def _edge(self):
        return None

    def apply(self, image):
        filtered_image = np.empty(image.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                filtered_image[i][j] = self._apply_to_window(image, i, j)

        return np.array(filtered_image, dtype=np.uint8)

    def _apply_to_window(self, image, current_x, current_y):
        if not self._window_fits(current_x, current_y, image.shape):
            return self._edge if self._edge is not None else image[current_x][current_y]
        return self._window_filter(image, current_x, current_y)

    @abstractmethod
    def _window_filter(self, image, current_x, current_y):
        raise NotImplementedError()

    def _window_fits(self, x, y, image_shape):
        return x in range(self._half_window_size, image_shape[0] - self._half_window_size) \
               and y in range(self._half_window_size, image_shape[1] - self._half_window_size)

    @staticmethod
    def _validate_window(window_size):
        if window_size % 2 != 1:
            raise ValueError('Window size must be an odd number')
        if window_size <= 1:
            raise ValueError('Window must be at least 3x3 in size')
