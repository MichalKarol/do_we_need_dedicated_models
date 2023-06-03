from argparse import ArgumentParser
from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.utilities.seed import seed_everything
import torch
import os
from pytorch_lightning import LightningModule
from torchvision.models.detection.ssd import SSD as torchvision_SSD, _vgg_extractor
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from shidc_annoatated_dataset import ShidcAnnotatedDataModule
from wandb_logger_wrapper import WandbLoggerWrapper
from model_checkpoint_wrapper import ModelCheckpointWrapper
from early_stopping_wrapper import EarlyStoppingWrapper
from utils import evaluate_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import shutil
from torchvision.models.vgg import vgg16, vgg19



import torch.multiprocessing
import torch.backends.cudnn
import torch.autograd

torch.multiprocessing.set_sharing_strategy("file_system")
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)


class SSD(LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.0001,
        num_classes: int = 4,
        backbone: str = "vgg16",
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
        backbone_model = vgg16() if self.backbone == "vgg16" else vgg19()
        backbone_model = _vgg_extractor(backbone_model, False, trainable_backbone_layers)

        anchor_generator = DefaultBoxGenerator(
            [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
            steps=[8, 16, 32, 64, 100, 300],
        )

        self.model = torchvision_SSD(
            backbone_model,
            anchor_generator,
            (512, 512),
            num_classes=num_classes,
            **kwargs,
        )

    def forward(self, x):
        with torch.no_grad():
            self.model.eval()
            return self.model(x)

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        images, targets = batch
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

        return SSD(**model_args)


def main():
    parser = ArgumentParser()
    parser = WandbLoggerWrapper.add_argparse_args(parser)
    parser = ModelCheckpointWrapper.add_argparse_args(parser)
    parser = EarlyStoppingWrapper.add_argparse_args(parser)
    parser = SSD.add_argparse_args(parser)
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
    model = SSD.from_argparse_args(args)

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
