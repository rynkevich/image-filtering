import sys

import matplotlib.pyplot as plt
import imageio
import numpy as np

from filters import MovingAverageFilter, MedianFilter
from filters.EdgeDetectingFilter import EdgeDetectingFilter

VALID_ARGC = 3
WINDOW_SIZE = 3


def main():
    if len(sys.argv) < VALID_ARGC:
        print('Usage: main.py <path_to_image> (average|median|sobel|prewitt)')
        return

    original_img = imageio.imread(sys.argv[1])
    img_to_filter = original_img

    is_grayscale = False
    image_filter = None
    filter_name = str.lower(sys.argv[2])
    if filter_name == 'average':
        image_filter = MovingAverageFilter(WINDOW_SIZE)
    elif filter_name == 'median':
        image_filter = MedianFilter(WINDOW_SIZE)
    elif filter_name == 'sobel' or filter_name == 'prewitt':
        is_grayscale = True
        img_to_filter = rgb_to_grayscale(original_img)
        image_filter = EdgeDetectingFilter(use_prewitt=filter_name == 'prewitt')
    else:
        raise NotImplementedError(f'Filter "{filter_name}" is not supported')

    filtered_img = image_filter.apply(img_to_filter)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    fig.edgecolor = 'black'
    fig.canvas.set_window_title('Image Filtering')

    show_img(axes[0], original_img, 'Original', is_grayscale)
    show_img(axes[1], filtered_img, 'Filtered', is_grayscale)

    plt.tight_layout()
    plt.show()


def rgb_to_grayscale(image):
    return np.dot(image[..., :3], [0.3, 0.587, 0.114])


def show_img(ax, img, title, is_grayscale):
    ax.set_title(title)
    ax.imshow(img, cmap=(plt.get_cmap('gray') if is_grayscale else None))
    ax.axis('off')


if __name__ == '__main__':
    main()
