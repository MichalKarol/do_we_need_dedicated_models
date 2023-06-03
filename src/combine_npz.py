from argparse import ArgumentParser, BooleanOptionalAction
import os
import numpy as np
from results import TILE_SIZE, OVERLAP, COLORS
import cv2 as cv
from torchvision.ops import nms
from os import path as pp
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, matthews_corrcoef, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import json
import torch
from utils import read_zoom_map, open_image_with_padding, mask_image
import datetime
from collections import defaultdict

KI_NEG_CATEGORY = 1
KI_POS_CATEGORY = 2
KI_LYM_CATEGORY = 3

np.random.seed(21334)

def calculate_metrics(y_true, y_pred, average=False):
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    metrics = {
        "accuracy": accuracy,
        "mcc": mcc,
    }

    averages = ["binary"] if not average else ["micro", "macro", "weighted"]
    for avg in averages:
        precision = precision_score(y_true, y_pred, average=avg, zero_division=0)
        recall = recall_score(y_true, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
        postfix = "" if avg == "binary" else f"_{avg}"
        metrics[f"precision{postfix}"] = precision
        metrics[f"recall{postfix}"] = recall
        metrics[f"f1{postfix}"] = f1

    if not average:
        metrics[f"support"] = int(np.sum(y_true))
        metrics[f"support_pred"] = int(np.sum(y_pred))
    else:
        indicies = (y_true == KI_NEG_CATEGORY) | (y_true == KI_POS_CATEGORY) | (y_pred == KI_NEG_CATEGORY) | (y_pred == KI_POS_CATEGORY)
        ki_pos_neg_y_true = y_true[indicies].copy()
        ki_pos_neg_y_pred = y_pred[indicies].copy()
        metrics[f"mcc_ki_pos_neg"] = matthews_corrcoef(
            ki_pos_neg_y_true, 
            ki_pos_neg_y_pred
        )

    return metrics


def generate_image(points, detections, distances, datapath, img_dir, img_name, masked, rescale_factor, iou_threshold, min_threshold, tile_size, overlap, imaged, white_balance, pathonet_balance):
    if imaged != "" and img_name not in imaged.split(","):
        return

    zoom_map = read_zoom_map(img_dir)
    zoom = zoom_map.get(img_name, 1) * rescale_factor
    img_path = pp.join(img_dir, img_name)
    pad_img = open_image_with_padding(img_path, zoom, tile_size, overlap, white_balance, pathonet_balance)

    if masked:
        pad_img = mask_image(pad_img, points)

    for idx, (xmin, ymin, xmax, ymax, label, _) in enumerate(detections):
        if masked and distances is not None and not np.any(distances[idx]):
            continue

        color = COLORS[int(label)]
        pad_img = cv.rectangle(
            pad_img, np.int32([xmin, ymin]), np.int32([xmax, ymax]), color, 3
        )

    for x, y, label in points:
        color = COLORS[int(label)]
        pad_img = cv.circle(pad_img, (int(x), int(y)), 15, color, -1)

    output_name = f"{iou_threshold}_{min_threshold}_{img_name}"
    pad_img_path = pp.join(datapath, output_name)
    pad_img = cv.resize(pad_img, None, fx=0.25, fy=0.25, interpolation=cv.INTER_CUBIC)
    cv.imwrite(pad_img_path, pad_img)


def run_options(args, npfiles, npfpostfix, sorted_iou_thresholds, sorted_min_thresholds, search_best=False):
    start = datetime.datetime.now()
    imaged = args.imaged
    args.imaged = None if search_best or args.save_images else args.imaged

    files = [np.load(pp.join(args.datapath, npf)) for npf in npfiles]
    points_all = [arrcol["arr_0"] for arrcol in files]
    joined_detections_all = [arrcol["arr_3"] for arrcol in files]

    overall_results = defaultdict(lambda: defaultdict(lambda: defaultdict(None)))
    overall_detection_matches = defaultdict(lambda: defaultdict(list))
    overall_detections_distances = defaultdict(lambda: defaultdict(list))
    overall_detections = defaultdict(lambda: defaultdict(list))

    all_len = len(npfiles) * len(sorted_iou_thresholds) * len(sorted_min_thresholds)
    all_idx = 0

    for npidx, npf in enumerate(npfiles):
        img_name = npf[5:-1*len(npfpostfix)]
        joined_detections = joined_detections_all[npidx]
        score_sorted_idx = np.argsort(-joined_detections[:, 5])
        sorted_detections = joined_detections[score_sorted_idx]
        points = points_all[npidx]
        if len(points.shape) == 1:
            points = np.empty((0, 3))

        kdtree = cKDTree(points[:, :2])
        points_len = points.shape[0]
        series_n_idxes = np.empty((sorted_detections.shape[0], 3))
        bound_checks_all = np.empty((sorted_detections.shape[0], 3))
        distance_checks_all = np.empty((sorted_detections.shape[0], 3))
        centers = np.empty((sorted_detections.shape[0], 2))

        if points.shape[0] != 0:
            xmins = sorted_detections[:, 0].astype(int)
            xmaxes = sorted_detections[:, 2].astype(int)
            ymins = sorted_detections[:, 1].astype(int)
            ymaxes = sorted_detections[:, 3].astype(int)

            widths = xmaxes - xmins
            heights = ymaxes - ymins

            centers_x = (widths // 2 + xmins).reshape(-1, 1)
            centers_y = (heights // 2 + ymins).reshape(-1, 1)
            centers = np.hstack([centers_x, centers_y])

            _, series_n_idxes = kdtree.query(centers, min(points.shape[0], 3))
            if points.shape[0] == 1:
                series_n_idxes = series_n_idxes.reshape(-1, 1)


            if series_n_idxes.shape[0] != 0:
                xx = points[series_n_idxes][:, :, 0]
                yy = points[series_n_idxes][:, :, 1]
                bound_checks_x_all = np.logical_and(sorted_detections[:, 0, None] <= xx, xx <= sorted_detections[:, 2, None])
                bound_checks_y_all = np.logical_and(sorted_detections[:, 1, None] <= yy, yy <= sorted_detections[:, 3, None])
                bound_checks_all = np.logical_and(bound_checks_x_all, bound_checks_y_all)
                distance_checks_all = np.sqrt(np.subtract(xx, centers_x)**2 + np.subtract(yy, centers_y)**2) < 100

        min_filtered_detections = sorted_detections
        min_filtered_series_n_idxes = series_n_idxes
        min_filtered_bound_checks = bound_checks_all
        min_filtered_distance_checks_all = distance_checks_all
        min_filtered_centers = centers

        for p_thr in sorted_min_thresholds:
            min_filtered_detections_idx = min_filtered_detections[:, 5] > p_thr
            has_min_idx = len(min_filtered_detections_idx) > 0
            min_filtered_detections = min_filtered_detections[min_filtered_detections_idx] if has_min_idx else np.empty((0, 6))
            min_filtered_series_n_idxes = min_filtered_series_n_idxes[min_filtered_detections_idx] if has_min_idx else np.empty((0, 3))
            min_filtered_bound_checks = min_filtered_bound_checks[min_filtered_detections_idx] if has_min_idx else np.empty((0, 3))
            min_filtered_distance_checks_all = min_filtered_distance_checks_all[min_filtered_detections_idx] if has_min_idx else np.empty((0, 3))
            min_filtered_centers = min_filtered_centers[min_filtered_detections_idx] if has_min_idx else np.empty((0, 2))

            nmsed_detections = min_filtered_detections
            nmsed_series_n_idxes = min_filtered_series_n_idxes
            nmsed_bound_checks = min_filtered_bound_checks
            nmsed_distance_checks_all = min_filtered_distance_checks_all
            nmsed_centers = min_filtered_centers

            for iou_thr in sorted_iou_thresholds[::-1]:
                with torch.no_grad():
                    nmsed_detections_idx = nms(torch.as_tensor(nmsed_detections[:, 0:4]), torch.as_tensor(nmsed_detections[:, 5]), iou_thr)

                has_nms_idx = len(nmsed_detections_idx) > 0
                nmsed_detections = nmsed_detections[nmsed_detections_idx] if has_nms_idx else np.empty((0, 6))
                nmsed_series_n_idxes = nmsed_series_n_idxes[nmsed_detections_idx] if has_nms_idx else np.empty((0, 3))
                nmsed_bound_checks = nmsed_bound_checks[nmsed_detections_idx] if has_nms_idx else np.empty((0, 3))
                nmsed_distance_checks_all = nmsed_distance_checks_all[nmsed_detections_idx] if has_nms_idx else np.empty((0, 3))
                nmsed_centers = nmsed_centers[nmsed_detections_idx] if has_nms_idx else np.empty((0, 2))

                visited = set()
                aligned = set()
                matches = []

                if len(nmsed_detections.shape) == 1:
                    nmsed_detections = nmsed_detections.reshape(1, 6)
                    minShape = nmsed_series_n_idxes.shape[0]
                    nmsed_series_n_idxes = nmsed_series_n_idxes.reshape(1, min(minShape, 3))
                    nmsed_bound_checks = nmsed_bound_checks.reshape(1, min(minShape, 3))
                    nmsed_distance_checks_all = nmsed_distance_checks_all.reshape(1, min(minShape, 3))
                    nmsed_centers = nmsed_centers.reshape(1, 2)
                if nmsed_centers.shape[0] != 0:
                    reclassTree = cKDTree(nmsed_centers)

                    ki_neg_nmsed_detections_idx = nmsed_detections[:, 4] == KI_NEG_CATEGORY
                    mapping = np.nonzero(ki_neg_nmsed_detections_idx)[0]

                    _, reclass_indexes = reclassTree.query(nmsed_centers[ki_neg_nmsed_detections_idx], min(nmsed_centers.shape[0], 10 + 1))

                    if len(reclass_indexes.shape) == 1:
                        reclass_indexes = reclass_indexes.reshape(-1, min(nmsed_centers.shape[0], 10 + 1))

                    reclass_indexes = reclass_indexes[:, 1:]

                    PASSES = 30
                    nmsed_detections_new = []
                    for _ in range(PASSES):
                        nmsed_detections_new = nmsed_detections.copy()
                        for det_idx in range(sum(ki_neg_nmsed_detections_idx)):
                            detection_neighbourhood = reclass_indexes[det_idx]
                            diff_x = nmsed_centers[mapping[det_idx], 0] - nmsed_centers[detection_neighbourhood, 0]
                            diff_y = nmsed_centers[mapping[det_idx], 1] - nmsed_centers[detection_neighbourhood, 1]
                            distances = np.sqrt(diff_x**2 + diff_y**2).ravel()
                            non_zero_distances_idx = distances >= 1
                            ki_neg_idx_in_neighbours = nmsed_detections[detection_neighbourhood[non_zero_distances_idx], 4] == KI_NEG_CATEGORY
                            ki_lym_idx_in_neighbours = nmsed_detections[detection_neighbourhood[non_zero_distances_idx], 4] == KI_LYM_CATEGORY

                            ki_neg_score = np.sum(1/(distances[non_zero_distances_idx][ki_neg_idx_in_neighbours]))
                            ki_lym_score = np.sum(1/(distances[non_zero_distances_idx][ki_lym_idx_in_neighbours]))

                            if ki_lym_score > ki_neg_score:
                                nmsed_detections_new[mapping[det_idx], 4] = KI_LYM_CATEGORY

                        lym_num_old = sum(nmsed_detections[:, 4] == KI_LYM_CATEGORY)
                        lym_num_new = sum(nmsed_detections_new[:, 4] == KI_LYM_CATEGORY)

                        nmsed_detections = nmsed_detections_new

                        if lym_num_old == lym_num_new:
                            break

                if points.shape[0] != 0 and nmsed_series_n_idxes.shape[0] != 0:
                    for det_idx, label in enumerate(nmsed_detections[:, 4]):
                        n_idxes = nmsed_series_n_idxes[det_idx]
                        bound_checks = nmsed_bound_checks[det_idx]

                        for idx, n_idx in enumerate(n_idxes):
                            if n_idx in visited or n_idx >= points_len:
                                continue

                            if not bound_checks[idx]:
                                continue

                            visited.add(n_idx)
                            aligned.add(det_idx)
                            matches.append([points[n_idxes][idx, 2], label])
                            break

                for p_idx, c in enumerate(points[:, 2]):
                    if p_idx not in visited:
                        matches.append([c, 0])

                for det_idx, label in enumerate(nmsed_detections[:, 4]):
                    if det_idx not in aligned:
                        if not args.masked or (args.masked and np.any(nmsed_distance_checks_all[det_idx])):
                            matches.append([0, label])

                points_match = np.array(matches) if matches else np.empty((0,2))

                base_classes = points_match[:, 0]
                detected_classes = points_match[:, 1]
                true_divisor = np.sum(points[:, 2] == KI_POS_CATEGORY) + np.sum(points[:, 2] == KI_NEG_CATEGORY)
                pred_divisor = np.sum(detected_classes == KI_POS_CATEGORY) + np.sum(detected_classes == KI_NEG_CATEGORY)

                overall_results[iou_thr][p_thr][img_name] = {
                    "ki_neg": calculate_metrics((base_classes == KI_NEG_CATEGORY), (detected_classes == KI_NEG_CATEGORY), False),
                    "ki_pos": calculate_metrics((base_classes == KI_POS_CATEGORY), (detected_classes == KI_POS_CATEGORY), False),
                    "ki_lym": calculate_metrics((base_classes == KI_LYM_CATEGORY), (detected_classes == KI_LYM_CATEGORY), False),
                    "avg": calculate_metrics(base_classes, detected_classes, True),
                    "true_ki": 0 if true_divisor == 0 else np.sum(points[:, 2] == KI_POS_CATEGORY) / true_divisor,
                    "pred_ki": 0 if pred_divisor == 0 else np.sum(detected_classes == KI_POS_CATEGORY) / pred_divisor,
                }

                if args.imaged is not None:
                    generate_image(points, nmsed_detections, nmsed_distance_checks_all, args.datapath, args.img_path, img_name, args.masked, args.rescale_factor, iou_thr, p_thr, args.tile_size, args.overlap, args.imaged, args.white_balance, args.pathonet_balance)
                overall_detections[iou_thr][p_thr].append(nmsed_detections)
                overall_detections_distances[iou_thr][p_thr].append(nmsed_distance_checks_all)
                overall_detection_matches[iou_thr][p_thr].append(points_match)
                percent = all_idx * 100 / all_len
                all_idx = all_idx + 1
                print(f"Done {all_idx}/{all_len}({percent}%)")

    points_all_vs = np.vstack([points for points in points_all if points.shape[0] != 0])
    best_score = 0
    best_score_params = (0, 0)
    option_metrics_dict = defaultdict(lambda: defaultdict(None))

    for iou_thr in sorted_iou_thresholds:
        for p_thr in sorted_min_thresholds:
            option_images_results = overall_results[iou_thr][p_thr].values()
            points_all_vs = np.vstack([itm[:, 0].reshape(-1, 1) for itm in overall_detection_matches[iou_thr][p_thr]])
            points_detection_all_vs = np.vstack([itm[:, 1].reshape(-1, 1) for itm in overall_detection_matches[iou_thr][p_thr]])
            option_metrics = {
                "ki_neg": calculate_metrics((points_all_vs == KI_NEG_CATEGORY), (points_detection_all_vs == KI_NEG_CATEGORY), False),
                "ki_pos": calculate_metrics((points_all_vs == KI_POS_CATEGORY), (points_detection_all_vs == KI_POS_CATEGORY), False),
                "ki_lym": calculate_metrics((points_all_vs == KI_LYM_CATEGORY), (points_detection_all_vs == KI_LYM_CATEGORY), False),
                "avg": calculate_metrics(points_all_vs, points_detection_all_vs, True),
                "r2": r2_score([res["true_ki"] for res in option_images_results], [res["pred_ki"] for res in option_images_results]),
                "mse": mean_squared_error([res["true_ki"] for res in option_images_results], [res["pred_ki"] for res in option_images_results]),
                "mae": mean_absolute_error([res["true_ki"] for res in option_images_results], [res["pred_ki"] for res in option_images_results]),
            }
            result_name = f"{iou_thr}_{p_thr}_result.json"
            with open(pp.join(args.datapath, result_name), "w") as jf:
                json.dump(option_metrics, jf)
            names = ["ki_neg", "ki_pos"]
            score = sum([option_metrics[name]["f1"]*option_metrics[name]["support"] for name in names]) / sum([option_metrics[name]["support"] for name in names])
            if score > best_score:
                best_score = score
                best_score_params = (iou_thr, p_thr)
                print(f"New max with {best_score} is {best_score_params}")
                print(iou_thr, p_thr, option_metrics)
            option_metrics_dict[iou_thr][p_thr] = {
                "metrics": option_metrics,
                **overall_results[iou_thr][p_thr]
            }
    end = datetime.datetime.now()


    print("-"*24)
    print(sorted_iou_thresholds, sorted_min_thresholds)
    print(end - start)
    print(f"Best max with {best_score} is {best_score_params}!")
    print("-"*24)

    if search_best:
        best_iou_thr, best_p_thr = best_score_params
        results = option_metrics_dict[best_iou_thr][best_p_thr]
        with open(pp.join(args.datapath, "best.json"), "w") as jf:
            json.dump(option_metrics_dict[best_iou_thr][best_p_thr], jf)
        with open(pp.join(args.datapath, "best.txt"), "w") as jf:
            jf.write(f"""
{results["metrics"]["ki_neg"]["accuracy"]},
{results["metrics"]["ki_pos"]["accuracy"]},
{results["metrics"]["ki_lym"]["accuracy"]},
{results["metrics"]["ki_neg"]["precision"]},
{results["metrics"]["ki_pos"]["precision"]},
{results["metrics"]["ki_lym"]["precision"]},
{results["metrics"]["ki_neg"]["recall"]},
{results["metrics"]["ki_pos"]["recall"]},
{results["metrics"]["ki_lym"]["recall"]},
{results["metrics"]["ki_neg"]["f1"]},
{results["metrics"]["ki_pos"]["f1"]},
{results["metrics"]["ki_lym"]["f1"]},
{results["metrics"]["ki_neg"]["mcc"]},
{results["metrics"]["ki_pos"]["mcc"]},
{results["metrics"]["ki_lym"]["mcc"]},
{results["metrics"]["ki_neg"]["support"]},
{results["metrics"]["ki_pos"]["support"]},
{results["metrics"]["ki_lym"]["support"]},
{results["metrics"]["r2"]},
{results["metrics"]["mse"]},
{results["metrics"]["mae"]},
            """.replace("\n", ""))
            args.imaged = imaged if imaged is not None else ""
            if args.imaged is not None:
                for npidx, npf in enumerate(npfiles):
                    img_name = npf[5:-1*len(npfpostfix)]
                    points = points_all[npidx]
                    matches = overall_detections[best_iou_thr][best_p_thr][npidx]
                    distances = overall_detections_distances[best_iou_thr][best_p_thr][npidx]
                    generate_image(points, matches, distances, args.datapath, args.img_path, img_name, args.masked, args.rescale_factor, best_iou_thr, best_p_thr, args.tile_size, args.overlap, args.imaged, args.white_balance, args.pathonet_balance)


def main():
    parser = ArgumentParser()
    parser.add_argument("datapath", type=str)#, default="/home/mkarol/data/pytorch-gan/10_5_detections/retinanet_resnet101/")
    parser.add_argument("img_path", type=str)#, default="/home/mkarol/data/dvc_datasets/ki_67/Fulawka_new_point_dataset")
    parser.add_argument("--iou_threshold", type=float)
    parser.add_argument("--min_threshold", type=float)
    parser.add_argument("--masked", action=BooleanOptionalAction)
    parser.add_argument("--imaged", type=str)
    parser.add_argument("--save_images", action=BooleanOptionalAction)
    parser.add_argument("--rescale_factor", type=float, help="Rescale factor", default=1.0)
    parser.add_argument("--tile_size", type=float, help="Tile size", default=TILE_SIZE)
    parser.add_argument("--overlap", type=float, help="Overlap", default=OVERLAP)
    parser.add_argument("--white_balance", action=BooleanOptionalAction)
    parser.add_argument("--pathonet_balance", action=BooleanOptionalAction)
    parser.add_argument("--rerun", type=str, help="Rerun those paths")
    parser.add_argument("--recalc", action=BooleanOptionalAction)
    args, unknown = parser.parse_known_args()

    npfiles = sorted([f for f in os.listdir(args.datapath) if f.endswith(".npz") and not f.endswith("combined.npz") and not f.endswith("recalc.npz")])
    npfpostfix = ".npz"

    if args.recalc:
        npfiles = sorted([f for f in os.listdir(args.datapath) if f.endswith("recalc.npz") and not f.endswith("combined.npz")])
        npfpostfix = "_recalc.npz"

    if args.rerun:
        rerunned_paths = set(args.rerun.split(","))
        npfiles = [f for f in npfiles if any([rerunpath for rerunpath in rerunned_paths if rerunpath in f])]


    if args.iou_threshold is not None and args.min_threshold is not None:
        run_options(args, npfiles, npfpostfix, [args.iou_threshold], [args.min_threshold])
        return

    run_options(args, npfiles, npfpostfix, np.arange(0.1, 1.1, 0.1), np.arange(0.0, 1.0, 0.1), True)

if __name__ == "__main__":
    main()