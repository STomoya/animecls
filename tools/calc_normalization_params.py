
import argparse
import os
import glob

import cv2
from joblib import Parallel, delayed
import numpy as np
from storch.dataset.utils import is_image_file


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Folder to images.')
    return parser.parse_args()


def main():
    args = get_args()
    image_files = glob.glob(os.path.join(args.input, '**', '*'), recursive=True)
    image_files = [file for file in image_files if os.path.isfile(file) and is_image_file(file)]

    def process(image_file):
        image = cv2.imread(image_file, cv2.IMREAD_COLOR).astype(np.float32)
        image /= 255
        image_mean = image.mean(axis=(0, 1))
        image_square = (image ** 2).mean(axis=(0, 1))
        return image_mean, image_square

    stats = Parallel(n_jobs=-1, verbose=3)(delayed(process)(image_file) for image_file in image_files)
    means, squares = list(zip(*stats))
    mean_sum, square_sum = sum(means), sum(squares)
    mean = mean_sum / len(image_files)
    std = np.sqrt((square_sum / len(image_files)) - mean ** 2)
    print(f'mean: {mean}\nstd:  {std}')

if __name__=='__main__':
    main()
