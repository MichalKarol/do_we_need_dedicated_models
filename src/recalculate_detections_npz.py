from argparse import ArgumentParser, BooleanOptionalAction
import os
import numpy as np
from neg_lym_classifier import ResNetClassifier
from results import TILE_SIZE, OVERLAP
from os import path as pp
import torch
from utils import pad_to_square, read_zoom_map, open_image_with_padding, mask_image
import datetime
from torchvision.transforms.functional import to_tensor
from scipy.spatial import cKDTree


KI_NEG_CATEGORY = 1
KI_POS_CATEGORY = 2
KI_LYM_CATEGORY = 3

np.random.seed(21334)

def run_options(args, npfiles):
    start = datetime.datetime.now()

    files = [np.load(pp.join(args.datapath, npf)) for npf in npfiles]
    points_all = [arrcol["arr_0"] for arrcol in files]
    joined_detections_all = [arrcol["arr_3"] for arrcol in files]

    zoom_map = read_zoom_map(args.img_path)

    location = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    if args.checkpoint:
        with torch.no_grad():
            model = ResNetClassifier()
            model.load_state_dict(torch.load(args.checkpoint, map_location=location)["state_dict"])
            model.to(location)
    
    for npidx, npf in enumerate(npfiles):
        img_name = npf[5:-4]
        joined_detections = joined_detections_all[npidx]
        score_sorted_idx = np.argsort(-joined_detections[:, 5])
        sorted_detections = joined_detections[score_sorted_idx]
        points = points_all[npidx]
        if len(points.shape) == 1:
            points = np.empty((0, 3))      

        if points.shape[0] != 0 and sorted_detections.shape[0] != 0:
            xmins = sorted_detections[:, 0].astype(int)
            xmaxes = sorted_detections[:, 2].astype(int)
            ymins = sorted_detections[:, 1].astype(int)
            ymaxes = sorted_detections[:, 3].astype(int)
            kiNegs = sorted_detections[:, 4] == KI_NEG_CATEGORY

            widths = xmaxes - xmins
            heights = ymaxes - ymins
            areas = widths * heights
        
            zoom = zoom_map.get(img_name, 1) * args.rescale_factor
            img_path = pp.join(args.img_path, img_name)
            pad_img = open_image_with_padding(img_path, zoom, args.tile_size, args.overlap, args.white_balance, args.pathonet_balance)
            if args.masked:
                pad_img = mask_image(pad_img, points)
            pad_height, pad_width, _ = pad_img.shape
            pad_img = to_tensor(pad_img / 255).to(torch.float)
            print("Tensored", pad_img.shape)

            def get_image(idxs):
                xmin, xmax, ymin, ymax = idxs
                hw = min((xmax - xmin) // 2, 128)
                hh = min((ymax - ymin) // 2, 128)

                cx = (xmax - xmin) // 2 + xmin
                cy = (ymax - ymin) // 2 + ymin

                bxmin = max(0, int(cx)-hw)
                bxmax = min(pad_width, int(cx)+hw)
                bymin = max(0, int(cy)-hh)
                bymax = min(pad_height, int(cy)+hh)

                image = pad_img[
                    :,
                    int(bymin):int(bymax),
                    int(bxmin):int(bxmax)
                ]
                image = pad_to_square(image, 256, True, 1)
                return image

            print(f"!!!{sum(sorted_detections[:, 4] == KI_LYM_CATEGORY)} !!!")

            xd_start = datetime.datetime.now()
            print(np.vstack([xmins[kiNegs], xmaxes[kiNegs], ymins[kiNegs], ymaxes[kiNegs]]).shape)

            if model:
                images = np.rollaxis(np.apply_along_axis(get_image, 0, np.vstack([xmins[kiNegs], xmaxes[kiNegs], ymins[kiNegs], ymaxes[kiNegs]])), 3, 0)
                images = torch.as_tensor(images)
                mapping = np.nonzero(kiNegs)[0]
                kiNegWidths = torch.as_tensor(widths[kiNegs] / 256, dtype=torch.float).reshape(-1, 1)
                kiNegHeights = torch.as_tensor(heights[kiNegs] / 256, dtype=torch.float).reshape(-1, 1)
                kiNegArea = torch.as_tensor(areas[kiNegs] / (256 * 256), dtype=torch.float).reshape(-1, 1)

                batch_size = 4
                for batch_idx in range(0, images.shape[0], batch_size):
                    bstart = batch_idx
                    bend = min(batch_idx + batch_size, images.shape[0])
                    bKiNegs = slice(bstart, bend)

                    with torch.no_grad():
                        preds = torch.argmax(model((
                            images[bKiNegs].to(location),
                            kiNegWidths[bKiNegs].to(location),
                            kiNegHeights[bKiNegs].to(location),
                            kiNegArea[bKiNegs].to(location)
                        )), 1)
                        sorted_detections[mapping[bKiNegs], 4] = torch.where(preds == 0, KI_LYM_CATEGORY, KI_NEG_CATEGORY).to("cpu")
                        # print(f"{mapping[bKiNegs][-1]}/{kiNegs.shape[0]} {mapping[bKiNegs][-1] * 100 / kiNegs.shape[0]}%")

            
            
                            
            print(f"!!!{sum(sorted_detections[:, 4] == KI_LYM_CATEGORY)} !!!")
            xd_end = datetime.datetime.now()
            print(xd_end - xd_start)
            print(f"{npidx} / {len(npfiles)} {100*npidx/len(npfiles)}")
            recalc_path_path = pp.join(args.datapath, f"data_{img_name}_recalc.npz")
            np.savez(recalc_path_path, points, np.zeros((0, 0)), np.zeros((0, 0)), sorted_detections)

    end = datetime.datetime.now()
    print(end - start)
    

def main():
    parser = ArgumentParser()
    parser.add_argument("datapath", type=str)#, default="/home/mkarol/data/pytorch-gan/10_5_detections/retinanet_resnet101/")
    parser.add_argument("img_path", type=str)#, default="/home/mkarol/data/dvc_datasets/ki_67/Fulawka_new_point_dataset")
    parser.add_argument("--masked", action=BooleanOptionalAction)
    parser.add_argument("--rescale_factor", type=float, help="Rescale factor", default=1.0)
    parser.add_argument("--tile_size", type=float, help="Tile size", default=TILE_SIZE)
    parser.add_argument("--overlap", type=float, help="Overlap", default=OVERLAP)
    parser.add_argument("--white_balance", action=BooleanOptionalAction, default=True)
    parser.add_argument("--pathonet_balance", action=BooleanOptionalAction)
    parser.add_argument("--rerun", type=str, help="Rerun those paths")
    parser.add_argument("--checkpoint", type=str)
    args, unknown = parser.parse_known_args()
    
    npfiles = sorted([f for f in os.listdir(args.datapath) if f.endswith(".npz") and not f.endswith("combined.npz") and not f.endswith("recalc.npz")])

    if args.rerun:
        rerunned_paths = set(args.rerun.split(",")) 
        npfiles = [f for f in npfiles if any([rerunpath for rerunpath in rerunned_paths if rerunpath in f])]
    
    run_options(args, npfiles)

if __name__ == "__main__":
    main()