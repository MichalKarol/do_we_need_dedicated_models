from typing import List
from pytorch_lightning.core.datamodule import LightningDataModule
import torch
from torch.nn.functional import interpolate
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, random_split
from torchvision import transforms
import cv2 as cv
from os import listdir
from os.path import isfile, join
from torch import Tensor, nn
from torchvision.transforms.functional import to_tensor
from pycocotools.coco import COCO
import numpy as np
import argparse

def _collate_fn(batch: List[Tensor]) -> tuple:
    return tuple(zip(*batch))

class ShidcAnnoatatedDataset(Dataset):
    def __init__(self, path, is_test, transformations):
        self.base_path = join(path, "test") if is_test else join(path, "train")
        ann_file = join(self.base_path, "test.json" if is_test else "train.json")
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transformations = transformations

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return to_tensor((cv.imread(join(self.base_path, path))) / 255).to(torch.float)

    def _load_target(self, id: int):
        def convert_bbox(bbox: List[int]):
            x, y, w, h = bbox
            return [x, y, min(x+w, 511), min(y+h, 511)]
        def check_bbox(ann):
            x, y, w, h = ann["bbox"]
            return w > 20 and h > 20
        def generateMask(segmentations: List[int]):
            polygon = [
                (segmentations[i * 2], segmentations[i * 2 + 1])
                for i in range(len(segmentations) // 2)
            ]
            return cv.fillConvexPoly(np.zeros((512, 512)), np.int32([polygon]), 1)
            
            
        anns = [ann for ann in self.coco.loadAnns(self.coco.getAnnIds(id)) if check_bbox(ann)]

        return {
            "boxes": torch.FloatTensor([convert_bbox(ann["bbox"]) for ann in anns]),
            "labels": torch.LongTensor([ann["category_id"] for ann in anns]),
            "masks":  torch.ByteTensor(np.array([generateMask(ann["segmentation"]) for ann in anns]))
        }

    def get_raw_data(self, index: int):
        id = self.ids[index]
        path = self.coco.loadImgs(id)[0]["file_name"]
        image = cv.imread(join(self.base_path, path))
        target = self.coco.loadAnns(self.coco.getAnnIds(id)) 
        return image, target

    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transformations is not None:
            image, target = self.transformations(image), target

        return image, target

    def __len__(self) -> int:
        return len(self.ids)


class ShidcAnnotatedDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        enable_transformations: bool
    ):
        super().__init__()
        self.data_dir = data_dir if not enable_transformations else f'{data_dir}_transformed'
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transforms = None
        self.num_classes = 1


    def setup(self, stage=None):
        self.train_loader = DataLoader(
            ShidcAnnoatatedDataset(
                self.data_dir,
                is_test=False,
                transformations=self.transforms
            ),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            ShidcAnnoatatedDataset(
                self.data_dir,
                is_test=True,
                transformations=None
            ),
            batch_size=self.batch_size,
            collate_fn=_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.test_loader

    def test_dataloader(self):
        return self.test_loader

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataloader options")
        parser.add_argument("--ds_data_dir", type=str, required=True, help="Required")
        parser.add_argument("--ds_batch_size", type=int, default=16)
        parser.add_argument("--ds_num_workers", type=int, default=8)
        parser.add_argument("--ds_enable_transformations", action=argparse.BooleanOptionalAction)
        return parent_parser
    
    @staticmethod
    def from_argparse_args(args):
        keys = ["ds_data_dir", "ds_batch_size", "ds_num_workers", "ds_enable_transformations"]
        dm_args = {key[3:]: getattr(args, key) for key in keys}
        for key in keys:
            delattr(args, key)
            
        return ShidcAnnotatedDataModule(**dm_args)