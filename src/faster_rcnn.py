from argparse import ArgumentParser
import shutil
from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.utilities.seed import seed_everything
import torch
import os
from torch import nn
from pytorch_lightning import LightningModule
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN as torchvision_FasterRCNN,
)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from model_checkpoint_wrapper import ModelCheckpointWrapper
from early_stopping_wrapper import EarlyStoppingWrapper
from utils import evaluate_iou
from shidc_annoatated_dataset import ShidcAnnotatedDataModule
from wandb_logger_wrapper import WandbLoggerWrapper
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import torch.multiprocessing
import torch.backends.cudnn
import torch.autograd

torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)


def create_fasterrcnn_backbone(
    backbone: str, trainable_backbone_layers: int = 3, **kwargs: Any
) -> nn.Module:
    """
    Args:
        backbone:
            Supported backones are: "resnet18", "resnet34","resnet50", "resnet101", "resnet152",
            "resnext50_32x4d", "resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2",
            as resnets with fpn backbones.
        trainable_backbone_layers: number of trainable resnet layers starting from final block.
    """

    return resnet_fpn_backbone(
        backbone,
        pretrained=True,
        trainable_layers=trainable_backbone_layers,
        **kwargs,
    )


class FasterRCNN(LightningModule):
    """PyTorch Lightning implementation of `Faster R-CNN: Towards Real-Time Object Detection with Region Proposal
    Networks <https://arxiv.org/abs/1506.01497>`_.
    Paper authors: Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
    Model implemented by:
        - `Teddy Koker <https://github.com/teddykoker>`
    During training, the model expects both the input tensors, as well as targets (list of dictionary), containing:
        - boxes (`FloatTensor[N, 4]`): the ground truth boxes in `[x1, y1, x2, y2]` format.
        - labels (`Int64Tensor[N]`): the class label for each ground truth box
    """

    def __init__(
        self,
        learning_rate: float = 0.0001,
        num_classes: int = 4,
        backbone: str = "resnet101",
        trainable_backbone_layers: int = 5,
        **kwargs: Any,
    ):
        """
        Args:
            learning_rate: the learning rate
            num_classes: number of detection classes (including background)
            backbone: Pretained backbone CNN architecture or torch.nn.Module instance.
            trainable_backbone_layers: number of trainable resnet layers starting from final block
        """
        super().__init__()

        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.backbone = backbone
        backbone_model = create_fasterrcnn_backbone(
            self.backbone,
            trainable_backbone_layers,
            **kwargs,
        )

        self.model = torchvision_FasterRCNN(
            backbone_model, num_classes=num_classes, **kwargs
        )

    def forward(self, x):
        with torch.no_grad():
            self.model.eval()
            return self.model(x)

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        images, targets = batch

        # fasterrcnn takes both images and targets for training, returns
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())
        output_data = {"loss": loss, **{k: v.detach() for k, v in loss_dict.items()}}
        return output_data

    def training_epoch_end(self, outputs):
        keys = outputs[0].keys()
        averages = {
            f"avg_{k}": torch.stack([output[k] for output in outputs]).mean()
            for k in keys
        }
        self.log_dict(averages)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            images, targets = batch
            outs = self.model(images)
            return outs, targets

    def validation_epoch_end(self, outputs):
        flat_map = lambda xs: [y for ys in xs for y in ys]
        with torch.no_grad():
            metric = MeanAveragePrecision(
                max_detection_thresholds=[100],
                iou_thresholds=[0.5, 0.75],
                rec_thresholds=[],
            )
            outs, targets = list(zip(*outputs))
            outs = flat_map(outs)
            targets = flat_map(targets)
            metric.update(outs, targets)
            metric_dict = metric.compute()
            iou = torch.stack(
                [evaluate_iou(t, o) for t, o in zip(targets, outs)]
            ).mean()
            output_data = {
                **{f"avg_{k}": v for k, v in metric_dict.items()},
                "avg_iou": iou,
            }
            self.log_dict(output_data)
            return output_data

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=0.005,
        )

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Model options")
        parser.add_argument("--backbone", type=str, required=True, help="Required")
        return parent_parser

    @staticmethod
    def from_argparse_args(args):
        model_args = {key: getattr(args, key) for key in ["backbone"]}
        for key in ["backbone"]:
            delattr(args, key)

        return FasterRCNN(**model_args)


def main():
    parser = ArgumentParser()
    parser = WandbLoggerWrapper.add_argparse_args(parser)
    parser = ModelCheckpointWrapper.add_argparse_args(parser)
    parser = EarlyStoppingWrapper.add_argparse_args(parser)
    parser = FasterRCNN.add_argparse_args(parser)
    parser = ShidcAnnotatedDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    seed_everything(21334, workers=True)
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    dirpath = args.dirpath

    wandb_logger = WandbLoggerWrapper.from_argparse_args(args)
    checkpoint_callback = ModelCheckpointWrapper.from_argparse_args(args)
    progress_bar = TQDMProgressBar(refresh_rate=5)
    early_stopping = EarlyStoppingWrapper.from_argparse_args(args)
    dataloader = ShidcAnnotatedDataModule.from_argparse_args(args)
    model = FasterRCNN.from_argparse_args(args)

    if not args.max_epochs:
        args.max_epochs = 100
    if not args.precision:
        args.precision = 16
    args.logger = wandb_logger
    args.callbacks = [progress_bar, checkpoint_callback, early_stopping]
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dataloader)
    shutil.copy(checkpoint_callback.best_model_path, os.path.join(dirpath, "best.ckpt"))
    print(checkpoint_callback.best_model_path)


if __name__ == "__main__":
    main()
