#!/bin/env/python2
import numpy as np
import cv2
import matplotlib.pyplot as plt
from IPython import embed

from process_args import get_args

KERNEL = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.int32)

def brighten(img, val):
    g = np.vectorize(lambda x: min(x + int(val), 255), otypes=[np.uint8])
    return g(img)

def process(img, args):
    new_img = {}
    for key, val in vars(args).items():
        if key != 'image' and val:
            new_img[key] = globals()[key](img, val)

    return (new_img or img)
    


def read_image(img_filename):
    return cv2.imread(img_filename)


def display_img(img):
    for key, val in img.items():
        plt.axis("off")
        plt.imshow(cv2.cvtColor(val, cv2.COLOR_BGR2RGB))
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
