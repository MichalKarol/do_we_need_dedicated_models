from collections import defaultdict
from matplotlib.pyplot import annotate
from math import ceil
from typing import DefaultDict, Dict, List, Tuple, NewType
import cv2 as cv
from os import listdir, getcwd
from os.path import isfile, join, splitext, abspath
import json
from dataclasses import dataclass
import numpy as np
from argparse import ArgumentParser

@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int


@dataclass
class AImage:
    id: int
    path: str
    height: int
    width: int


@dataclass
class Annotation:
    id: int
    image: AImage
    bbox: BoundingBox
    category: int
    segmentation: List[int]
    area: int

def read_coco_file(coco_path: str) -> List[Annotation]:
    with open(coco_path) as f:
        data = json.load(f)
        images = data["images"]
        annotations = data["annotations"]
        categories = data["categories"]

        image_map: Dict[str, tuple] = {
            str(image["id"]): (
                image["id"],
                image["file_name"],
                image["height"],
                image["width"],
            )
            for image in images
        }
        category_map: Dict[int, int] = {
            category["id"]: idx for idx, category in enumerate(categories)
        }

        output_annotations: List[Annotation] = []
        for annotation in annotations:
            ann_id = annotation["id"]
            image = image_map[str(annotation["image_id"])]
            category = category_map[annotation["category_id"]]
            if "bbox" in annotation:
                [x, y, w, h] = annotation["bbox"]
                
            if "point" in annotation:
                xx, yy = annotation["point"]
                [x, y, w, h] = [xx-1, yy-1, 2, 2]
            
            x = max([x, 0])
            y = max([y, 0])
            

            output_annotations.append(
                Annotation(
                    ann_id,
                    AImage(*image),
                    BoundingBox(x, y, w, h),
                    category,
                    None,
                    None,
                )
            )
        return output_annotations

def main():
    parser = ArgumentParser()
    parser.add_argument("coco_dir_path", type=str)
    args = parser.parse_args()

    for dirr in ["test", "train"]:
        filepath = join(args.coco_dir_path, dirr, f"{dirr}.json")
        annotations = read_coco_file(filepath)

        file_dict = defaultdict(list)
        class_map = {1: 1, 0: 2, 2: 3}
        for ann in annotations:
            x_min = ann.bbox.x
            y_min = ann.bbox.y
            w = ann.bbox.w
            h = ann.bbox.h
            
            x_cen = (x_min + w //2)
            y_cen = (y_min + h //2)

            file_dict[ann.image.path].append({
                "label_id": class_map[ann.category],
                "x": x_cen,
                "y": y_cen
            })

        curr_dir = getcwd()
        file_new_path = abspath(join(curr_dir, args.coco_dir_path, dirr))

        for path, values in file_dict.items():
            extensionsless_path = splitext(path)[0]
            new_path = join(file_new_path, extensionsless_path)

            with open(f"{new_path}.json", "w") as file:
                json.dump(values, file)


if __name__ == "__main__":
    main()
