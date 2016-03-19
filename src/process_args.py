import argparse 

def get_args():
    parser = argparse.ArgumentParser(
        description="Script to process images")
    parser.add_argument(
            "image", type=str,
            help="Image path to image to be processed")
    parser.add_argument("-c", "--convolve", help="apply convolution to the image", action="store_true")
    parser.add_argument("-bw", "--greyscale", help="turn the image to greyscale", action="store_true")
    parser.add_argument("-b", "--brighten", help="brighten the image")
    parser.add_argument("-gb", "--gaussian_blur", help="blur the image using a Gaussian filter mask with a custom sigma")
    parser.add_argument("-e", "--edges", help="detect edges", action="store_true")

    args = parser.parse_args()

    return args


