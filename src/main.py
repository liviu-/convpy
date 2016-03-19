#!/bin/env/python2
import math

import numpy as np
import cv2
import matplotlib.pyplot as plt
from IPython import embed

from process_args import get_args

KERNEL_SIZE = 3

def extend_image(img, cols, rows):
    return img
    

def gaussian_kernel(sigma, size):
    center = math.floor(int(size/2))
    kernel = np.zeros([size, size])
    sigma = float(sigma)
    for i in range(size):
        for j in range(size):
            kernel[i,j] = math.exp(-(pow((i - center), 2)/(2 * sigma * sigma) +
                                     pow((j - center), 2)/(2 * sigma * sigma)))

    return np.around(kernel)/sum(sum(np.around(kernel)))


def edges(img, val=None):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.int32)
    # Sobel
    #kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float64)
    return convolve(img, kernel)


def gaussian_blur(img, val):
    kernel_size = KERNEL_SIZE
    kernel = gaussian_kernel(val, kernel_size)
    return convolve(img, kernel)


def convolve(img, kernel):
    """Discrete convolution"""
    cols, rows, d = img.shape
    k_cols, k_rows = kernel.shape
    extend_k_cols, extend_k_rows = (int(math.floor(k_cols/2)),
                                    int(math.floor(k_rows/2)))
    extended_img = extend_image(
        img, extend_k_cols, extend_k_rows)

    new_img = np.array(
        [
         [sum(
             (img[col + i, row + j, dim] * 
              kernel[i + extend_k_cols, j + extend_k_rows]) 
             for i in range(-extend_k_cols, extend_k_cols + 1)
             for j in range(-extend_k_rows, extend_k_rows + 1)
             )
          for dim in range(d)
          ]
        for col in range(extend_k_cols, cols - extend_k_cols) 
        for row in range(extend_k_rows, rows - extend_k_rows)
        ],
        dtype=np.int32).reshape(cols - extend_k_cols - 1,
                                rows - extend_k_rows - 1, d)
    return new_img


def brighten(img, val):
    g = np.vectorize(lambda x: min(x + int(val), 255), otypes=[np.uint8])
    return g(img)


def greyscale(img, val=None):
    cols, rows, d = img.shape
    return np.array(
        [sum(img[col, row])/d
        for col in range(cols) for row in range(rows)],
        dtype=np.uint8).reshape(cols, rows)


def process(img, args):
    new_img = {}
    for key, val in vars(args).items():
        if key != 'image' and val:
            new_img[key] = globals()[key](img, val)

    return (new_img or img)
    


def read_image(img_filename):
    img_BGR = cv2.imread(img_filename)
    return cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)


def normalize_for_display(img):
    img = img - img.min()
    img = img.astype(np.float32)
    img /= img.max()/254.0
    return img


def display_img(img):
    if img.items:
        for key, val in img.items():
            plt.axis("off")
            # Color image
            img = normalize_for_display(val)
            if len(img.shape) > 2:
                plt.imshow(img.astype(np.uint8))
            else:
                plt.imshow(img, cmap="Greys_r")
            plt.show()
    else:
        plt.imshow(img)
        plt.show()


def main():
    args = get_args()
    img = read_image(args.image)
    new_img = process(img, args)
    display_img(new_img)


if __name__ == '__main__':
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    main()
