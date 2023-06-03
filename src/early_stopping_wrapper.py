from pytorch_lightning.callbacks import EarlyStopping


class EarlyStoppingWrapper:
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group(title="Early stopping options")
        parser.add_argument("--es_monitor", type=str, default="avg_loss")
        parser.add_argument("--es_mode", type=str, default="min")
        parser.add_argument("--es_patience", type=int, default=5)
        parser.add_argument("--es_min_delta", type=float, default=0.0)
        return parent_parser

    @staticmethod
    def from_argparse_args(args):
        keys = ["es_monitor", "es_mode", "es_patience", "es_min_delta"]
        early_stopping_args = {key[3:]: getattr(args, key) for key in keys}
        early_stopping_args["verbose"] = True
        for key in keys:
            delattr(args, key)

        return EarlyStopping(**early_stopping_args)
