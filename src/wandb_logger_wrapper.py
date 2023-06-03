from pytorch_lightning.loggers import WandbLogger


class WandbLoggerWrapper:
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group(title="Wandb options")
        parser.add_argument("--project", type=str, required=True, help="Required")
        parser.add_argument("--group", type=str, required=True, help="Required")
        return parent_parser

    @staticmethod
    def from_argparse_args(args):
        wandb_args = {key: getattr(args, key) for key in ["project", "group"]}
        for key in ["project", "group"]:
            delattr(args, key)

        return WandbLogger(**wandb_args)
