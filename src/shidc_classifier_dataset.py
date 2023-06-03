from functools import lru_cache
from typing import List
from pytorch_lightning.core.datamodule import LightningDataModule
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import cv2 as cv
from os.path import join
from torch import FloatTensor, Tensor
from torchvision.transforms.functional import to_tensor
from pycocotools.coco import COCO
from utils import adjust_rgb_wb, pad_to_square, rotate_image
import numpy as np

# def _collate_fn(batch: List[Tensor]) -> tuple:
#     return tuple(zip(*batch))


class ShidcClassifierDataset(Dataset):
    def __init__(self, path, is_test, transformations, ki_neg_cls, ki_lym_cls):
        self.base_path = join(path, "test") if is_test else join(path, "train")
        ann_file = join(self.base_path, "test.json" if is_test else "train.json")
        self.coco = COCO(ann_file)
        self.ki_neg_cls = ki_neg_cls
        self.ki_lym_cls = ki_lym_cls
        rng = np.random.default_rng()

        def check_ann(ann):
            label = ann["category_id"]
            x, y, w, h = ann["bbox"]
            return (
                w > 20
                and h > 20
                and (label == self.ki_neg_cls or label == self.ki_lym_cls)
            )

        self.ids = np.array(
            [idd for idd in self.coco.anns.keys() if check_ann(self.coco.anns[idd])]
        )
        labels = np.array([self.coco.anns[idd]["category_id"] for idd in self.ids])
        neg_cls = np.sum(labels == ki_neg_cls)
        lym_cls = np.sum(labels == ki_lym_cls)

        if lym_cls < neg_cls:
            bias = (neg_cls / lym_cls) - 1
            lym_ids = self.ids[labels == ki_lym_cls]
            self.ids = np.hstack([self.ids, np.repeat(lym_ids, bias)])
            rng.shuffle(self.ids)
        else:
            bias = (lym_cls / neg_cls) - 1
            neg_ids = self.ids[labels == ki_neg_cls]
            self.ids = np.hstack([self.ids, np.repeat(neg_ids, bias)])
            rng.shuffle(self.ids)

        self.transformations = transformations
        rng = np.random.default_rng()
        self.rotations = rng.uniform(0, 360, self.ids.shape)
        self.saturations = rng.uniform(0.9, 1.1, self.ids.shape)

        labels = np.array([self.coco.anns[idd]["category_id"] for idd in self.ids])
        neg_cls = np.sum(labels == ki_neg_cls)
        lym_cls = np.sum(labels == ki_lym_cls)
        print(neg_cls, lym_cls)

    @lru_cache(maxsize=5)
    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return adjust_rgb_wb(cv.imread(join(self.base_path, path)))

    def __getitem__(self, index: int):
        id = self.ids[index]
        rotation = self.rotations[index]
        saturation = self.saturations[index]
        ann = self.coco.anns[id]
        label = 0 if ann["category_id"] == self.ki_lym_cls else 1
        x, y, w, h = ann["bbox"]
        hw = w // 2
        hh = h // 2
        cx = x + hw
        cy = y + hh

        image = self._load_image(ann["image_id"])
        ih, iw, _ = image.shape

        xmin = max(0, int(cx) - hw)
        xmax = min(iw, int(cx) + hw)
        ymin = max(0, int(cy) - hh)
        ymax = min(ih, int(cy) + hh)

        image = image[ymin:ymax, xmin:xmax]
        # image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        # image[:, :, 1] = np.clip(image[:, :, 1] * saturation, 0, 180)
        # image = cv.cvtColor(image, cv.COLOR_HSV2BGR)

        image = rotate_image(
            image, rotation, channels_first=False, borderValue=(255, 255, 255)
        )
        image = pad_to_square(image, 256, False, 255)
        image = to_tensor(image / 255).to(torch.float)

        if self.transformations is not None:
            image = self.transformations(image)
        # return image, label, w*h

        return (
            image,
            FloatTensor([w / 256]),
            FloatTensor([h / 256]),
            FloatTensor([w * h / (256 * 256)]),
        ), label

    def __len__(self) -> int:
        return len(self.ids)


class ShidcClassifierDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        ki_neg_cls: int,
        ki_lym_cls: int,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transforms = None
        self.num_classes = 2
        self.ki_neg_cls = ki_neg_cls
        self.ki_lym_cls = ki_lym_cls

    def setup(self, stage=None):
        self.train_loader = DataLoader(
            ShidcClassifierDataset(
                self.data_dir,
                is_test=False,
                transformations=self.transforms,
                ki_neg_cls=self.ki_neg_cls,
                ki_lym_cls=self.ki_lym_cls,
            ),
            batch_size=self.batch_size,
            shuffle=True,
            # collate_fn=_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        self.test_loader = DataLoader(
            ShidcClassifierDataset(
                self.data_dir,
                is_test=True,
                transformations=None,
                ki_neg_cls=self.ki_neg_cls,
                ki_lym_cls=self.ki_lym_cls,
            ),
            batch_size=self.batch_size,
            # collate_fn=_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
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
        parser.add_argument("--ds_ki_neg_cls", type=int, default=1)
        parser.add_argument("--ds_ki_lym_cls", type=int, default=3)
        parser.add_argument("--ds_batch_size", type=int, default=16)
        parser.add_argument("--ds_num_workers", type=int, default=8)
        return parent_parser

    @staticmethod
    def from_argparse_args(args):
        keys = [
            "ds_data_dir",
            "ds_batch_size",
            "ds_num_workers",
            "ds_ki_neg_cls",
            "ds_ki_lym_cls",
        ]
        dm_args = {key[3:]: getattr(args, key) for key in keys}
        for key in keys:
            delattr(args, key)

        return ShidcClassifierDataModule(**dm_args)
