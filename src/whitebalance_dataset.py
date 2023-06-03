from argparse import ArgumentParser
from patmlkit.image import read_rgb_image, write_rgb_image
from patmlkit.coco import COCO
import os
import os.path as pp
import numpy as np
import cv2 as cv
from utils import adjust_rgb_wb


def preprocess_dataset(source_coco, dest_dir):
    coco = COCO.from_json(source_coco)
    source_dir = pp.dirname(source_coco)
    source_name = pp.basename(source_coco)
    os.makedirs(dest_dir, exist_ok=True)
    os.link(pp.join(source_coco), pp.join(dest_dir, source_name))

    for image in coco.images:
        source_image_path = pp.join(source_dir, image.file_name)
        dest_image_path = pp.join(dest_dir, image.file_name)
        img = read_rgb_image(source_image_path)
        img = adjust_rgb_wb(img)
        write_rgb_image(dest_image_path, img)


def main():
    parser = ArgumentParser("whitebalance_dataset.py")
    parser.add_argument("--source_coco", type=str, help="Source COCO json")
    parser.add_argument("--dest_dir", type=str, help="Destination directory")
    args = parser.parse_args()

    preprocess_dataset(args.source_coco, args.dest_dir)


if __name__ == "__main__":
    main()
