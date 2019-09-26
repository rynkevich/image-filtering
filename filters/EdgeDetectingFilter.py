from filters.WindowBasedFilter import WindowBasedFilter


class EdgeDetectingFilter(WindowBasedFilter):
    _WINDOW_SIZE = 3
    _SOBEL_MATRIX_X = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]
    _SOBEL_MATRIX_Y = [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ]
    _PREWITT_MATRIX_X = [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ]
    _PREWITT_MATRIX_Y = [
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ]

    def __init__(self, use_prewitt=False):
        super().__init__(self._WINDOW_SIZE)
        self._use_prewitt = use_prewitt

    @property
    def _edge(self):
        return 0

    def _window_filter(self, image, current_x, current_y):
        accumulator_x, accumulator_y = 0, 0
        for x_index, x in enumerate(range(current_x - self._half_window_size, current_x + self._half_window_size + 1)):
            for y_index, y in enumerate(range(current_y - self._half_window_size, current_y + self._half_window_size + 1)):
                if self._use_prewitt:
                    accumulator_x += image[x][y] * self._PREWITT_MATRIX_X[x_index][y_index]
                    accumulator_y += image[x][y] * self._PREWITT_MATRIX_Y[x_index][y_index]
                else:
                    accumulator_x += image[x][y] * self._SOBEL_MATRIX_X[x_index][y_index]
                    accumulator_y += image[x][y] * self._SOBEL_MATRIX_Y[x_index][y_index]

        magnitude = (accumulator_x ** 2 + accumulator_y ** 2) ** (1 / 2)

        return magnitude if magnitude < 255 else 255
