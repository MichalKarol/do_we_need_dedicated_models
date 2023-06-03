from argparse import ArgumentParser
import types
from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.utilities.seed import seed_everything
import torch
import os
from torch import nn
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from shidc_classifier_dataset import ShidcClassifierDataModule
from wandb_logger_wrapper import WandbLoggerWrapper
from model_checkpoint_wrapper import ModelCheckpointWrapper
from early_stopping_wrapper import EarlyStoppingWrapper
import torchvision.models as models
import torch.nn.functional as F
import torch.multiprocessing
import torch.backends.cudnn
import torch.autograd
from pytorch_lightning.plugins import DDPPlugin
import shutil

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

class ResNetClassifier(LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.0001,
        checkpoint: str = None
    ):
        """
        Args:
            learning_rate: the learning rate
            num_classes: number of detection classes (including background)
        """
        super().__init__()
        if not checkpoint:
            self.learning_rate = learning_rate
            self.conv = nn.Sequential(
                nn.Conv2d(3, 64, 3),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 3),
                nn.ReLU(True),
                nn.Conv2d(64, 128, 3),
                nn.ReLU(True),
                nn.Conv2d(128, 128, 3),
                nn.ReLU(True),
                nn.AdaptiveAvgPool2d((7, 7)),
            )
            self.fc = nn.Sequential(
                nn.Linear(128 * 7 *7 + 3, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 2048),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(2048, 2),
            )
            self.loss_fn =  nn.BCEWithLogitsLoss()
        else:
            self.load_state_dict(torch.load(checkpoint)["state_dict"])

    def forward(self, x):
        torch.cuda.empty_cache()
        (images, weights, heights, areas) = x 
        x = self.conv(images)
        x = torch.flatten(x, 1)
        x = torch.hstack((x, weights, heights, areas))
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        data, labels = batch
        preds = self.forward(data)
        labels = F.one_hot(labels, num_classes=2)
        loss = self.loss_fn(preds, labels.to(torch.float))
        return loss

    def training_epoch_end(self, outputs):
        averages = {f"avg_loss": torch.stack([output["loss"] for output in outputs]).mean()}
        self.log_dict(averages)

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        with torch.no_grad():
            data, labels = batch
            preds = self.forward(data)
            return preds, labels

    def validation_epoch_end(self, outputs):
        torch.cuda.empty_cache()
        flat_map = lambda xs: [y for ys in xs for y in ys]
        with torch.no_grad():
            metrics = {
                "accuracy": accuracy_score,
                "precision": precision_score,
                "recall": recall_score,
                "mcc": matthews_corrcoef,
                "f1": f1_score,
            }
            outs, labels = list(zip(*outputs))
            outs = torch.argmax(torch.vstack(flat_map(outs)), 1).to("cpu")
            labels = torch.vstack(flat_map(labels)).ravel().to("cpu")
            print(outs, "outs")
            print(labels, "labels")
            output_data = {**{f"avg_{k}": m(labels, outs) for k, m in metrics.items()}}
            print(output_data)
            self.log_dict(output_data)
            return output_data

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
        )
    
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Model options")
        parser.add_argument("--checkpoint", type=str)
        return parent_parser
    
    @staticmethod
    def from_argparse_args(args):
        model_args = {key: getattr(args, key) for key in ["checkpoint"]}
        for key in ["checkpoint"]:
            delattr(args, key)
            
        return ResNetClassifier(**model_args)

def main():
    parser = ArgumentParser()
    parser = WandbLoggerWrapper.add_argparse_args(parser)
    parser = ModelCheckpointWrapper.add_argparse_args(parser)
    parser = EarlyStoppingWrapper.add_argparse_args(parser)
    parser = ResNetClassifier.add_argparse_args(parser)
    parser = ShidcClassifierDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    seed_everything(21334, workers=True)
    os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    dirpath = args.dirpath

    wandb_logger = WandbLoggerWrapper.from_argparse_args(args)
    checkpoint_callback = ModelCheckpointWrapper.from_argparse_args(args)
    progress_bar = TQDMProgressBar(refresh_rate=2)
    early_stopping = EarlyStoppingWrapper.from_argparse_args(args)
    dataloader = ShidcClassifierDataModule.from_argparse_args(args)
    model = ResNetClassifier.from_argparse_args(args)

    if not args.max_epochs:
        args.max_epochs = 100
    if not args.precision:
        args.precision = 16
    args.logger = wandb_logger
    args.callbacks = [progress_bar, checkpoint_callback, early_stopping]
    args.strategy=DDPPlugin(find_unused_parameters=False)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dataloader)
    shutil.copy(checkpoint_callback.best_model_path, os.path.join(dirpath, "best.ckpt"))
    print(checkpoint_callback.best_model_path)

if __name__ == "__main__":
        main()  