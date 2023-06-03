from argparse import ArgumentParser
import os
import os.path as pp
import json
import pandas as pd
from patmlkit.json import read_json_file

def main():
    parser = ArgumentParser()
    parser.add_argument("detections", type=str)
    args, unknown = parser.parse_known_args()

    results = dict()

    for key, _, files in sorted(os.walk(args.detections), key=lambda key: key[0]):
        for file in sorted(files):
            if ".json" not in file:
                continue

            _ ,dirname = pp.split(key)
            data = read_json_file(pp.join(key, file))
            results[dirname] = {
                "ki_neg_accuracy": data["ki_neg"]["accuracy"],
                "ki_pos_accuracy": data["ki_pos"]["accuracy"],
                "ki_neg_precision": data["ki_neg"]["precision"],
                "ki_pos_precision": data["ki_pos"]["precision"],
                "ki_neg_recall": data["ki_neg"]["recall"],
                "ki_pos_recall": data["ki_pos"]["recall"],
                "ki_neg_f1": data["ki_neg"]["f1"],
                "ki_pos_f1": data["ki_pos"]["f1"],
                "ki_neg_support_pred": data["ki_neg"]["support_pred"],
                "ki_pos_support_pred": data["ki_pos"]["support_pred"],
                "r2": data["r2"],
                "mse": data["mse"],
                "mae": data["mae"],
            }

    df = pd.DataFrame(results)
    # df.to_json(pp.join(args.detections, "results.json"))
    df.to_csv(pp.join(args.detections, "results.csv"))


if __name__ == "__main__":
    main()