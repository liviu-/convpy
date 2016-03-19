import argparse 

def get_args():
    parser = argparse.ArgumentParser(
        description="Script to process images")
    parser.add_argument(
            "image", type=str,
            help="Image path to image to be processed")
    parser.add_argument("-c", "--convolve", help="apply convolution to the image", action="store_true")
    parser.add_argument("-b", "--brighten", help="brighten the image")

    args = parser.parse_args()

    return args


