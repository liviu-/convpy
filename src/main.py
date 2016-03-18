import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
from IPython import embed

KERNEL = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.int32)

def get_args():
    parser = argparse.ArgumentParser(
        description="Script to process images")
    parser.add_argument(
            "image", type=str, nargs='?',
            help="Image path to image to be processed")

    args = parser.parse_args()

    return args.image

def process(img, kernel):
    g = np.vectorize(lambda x: min(x + 100, 255), otypes=[np.uint8])
    return g(img)


def read_image(img_filename):
    return cv2.imread(img_filename)


def display_img(img):
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def main():
    img_filename = get_args()
    img = read_image(img_filename)
    new_img = process(img, KERNEL)
    display_img(new_img)


if __name__ == '__main__':
    main()
