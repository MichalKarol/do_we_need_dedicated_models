from pytorch_lightning.callbacks import ModelCheckpoint


class ModelCheckpointWrapper:
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group(title="Model checkpoint options")
        parser.add_argument("--monitor", type=str, default="avg_loss")
        parser.add_argument("--mode", type=str, default="min")
        parser.add_argument("--dirpath", type=str, required=True, help="Required")
        parser.add_argument("--save_last", type=bool, default=True)
        parser.add_argument("--save_top_k", type=int, default=5)
        parser.add_argument("--filename", type=str, default="{epoch:02d}-{step:04d}-{loss:.4f}")
        return parent_parser

    @staticmethod
    def from_argparse_args(args):
        keys = ["monitor", "mode", "dirpath", "save_last", "save_top_k", "filename"]
        model_checkpoint_args = {key: getattr(args, key) for key in keys}
        for key in keys:
            delattr(args, key)

        return ModelCheckpoint(**model_checkpoint_args)
