from argparse import ArgumentParser, BooleanOptionalAction
from ctypes import ArgumentError
import itertools
import shutil
import pytorch_lightning as pl
import torch
import os
from retinanet import RetinaNet
from faster_rcnn import FasterRCNN
from mask_rcnn import MaskRCNN
from ssd import SSD
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ChainDataset
import cv2 as cv
from pycocotools.coco import COCO
from os import path as pp
import tempfile
from torch.utils.data.dataset import IterableDataset
from torchvision.transforms.functional import to_tensor
from shutil import rmtree
import subprocess
import json
from utils import read_zoom_map, open_image_with_padding, mask_image, calculate_image_sizes
from datetime import datetime

import torch.multiprocessing
import torch.backends.cudnn
import torch.autograd
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

NUM_WORKERS = 0
COLORS = {
	1: (0, 128, 0),
	2: (0, 0, 255),
	3: (128, 0, 128),
}
TILE_SIZE = 512
OVERLAP = 10

class PyTorchPredictor:
	def __init__(self, model, trainer, checkpoint_path):
		location = "cuda" if torch.cuda.is_available() else "cpu"
		model.load_state_dict(torch.load(checkpoint_path, map_location=location)["state_dict"])
		self.model = model
		self.trainer = trainer
	
	def predict(self, dataset):
		with torch.no_grad():
			return self.trainer.predict(self.model, dataset)


class YoloPredictor:
	def __init__(self, chckpoint_path, tile_size):
		self.chckpoint_path = chckpoint_path
		self.tile_size = tile_size
	
	def predict(self, dataset):
		with tempfile.TemporaryDirectory() as tmp_dir:
			with tempfile.NamedTemporaryFile() as tp:
				for batch_idx, batch in enumerate(iter(dataset)):
					for img_idx, img in enumerate(batch):
						img_path = pp.join(tmp_dir, f"{batch_idx}_{img_idx}.png")
						tp.write(f"{img_path}\n".encode("utf-8"))
						cv.imwrite(img_path, (img * 255).permute(1, 2, 0).numpy())
				tp.flush()

				results = []
				with tempfile.NamedTemporaryFile() as fp:
					p1 = subprocess.Popen(["cat", tp.name], stdout=subprocess.PIPE)
					cwd = "src/darknet"
					model_path = pp.abspath(self.chckpoint_path)
					p = subprocess.Popen(["./darknet", "detector", "test", "data/obj.data.transformed", "cfg/yolo-obj-detection.cfg", model_path, "-ext_output", "-dont_show", "-out", fp.name], cwd=cwd, stdin=p1.stdout)
					p.communicate()
					print(p.stdout)
					rets = json.load(fp)
					print(json.dumps(rets))
					for ret in rets:
						objs = ret["objects"]
						filtered_objs = [obj for obj in objs if 
								obj["relative_coordinates"]["center_x"] < 1 and 
								obj["relative_coordinates"]["center_y"] < 1 and 
								obj["relative_coordinates"]["width"] < 1 and 
								obj["relative_coordinates"]["height"] < 1]
						
						boxes = np.empty((len(filtered_objs), 4))
						for i, obj in enumerate(filtered_objs):
							boxes[i][0] = (obj["relative_coordinates"]["center_x"] - (obj["relative_coordinates"]["width"] / 2))   * self.tile_size
							boxes[i][1] = (obj["relative_coordinates"]["center_y"] - (obj["relative_coordinates"]["height"] / 2))   * self.tile_size
							boxes[i][2] = (obj["relative_coordinates"]["center_x"] + (obj["relative_coordinates"]["width"] / 2))   * self.tile_size
							boxes[i][3] = (obj["relative_coordinates"]["center_y"] + (obj["relative_coordinates"]["height"] / 2))   * self.tile_size

						labels = np.array([
							obj["class_id"] + 1 for obj in filtered_objs
						])

						scores = np.array([
							obj["confidence"] for obj in filtered_objs
						])

						results.append(np.array([{
							"boxes": torch.as_tensor(boxes),
							"labels": torch.as_tensor(labels),
							"scores": torch.as_tensor(scores),
						}]))
		return np.array(results)


class PathoNetPredictor:
	def __init__(self, chckpoint_path, tile_size):
		self.chckpoint_path = chckpoint_path
		self.tile_size = tile_size

	def predict(self, dataset):
		with tempfile.TemporaryDirectory() as tmp_dir:
			with tempfile.NamedTemporaryFile() as tp:
				for batch_idx, batch in enumerate(iter(dataset)):
					for img_idx, img in enumerate(batch):
						img_path = pp.join(tmp_dir, f"{batch_idx:05d}_{img_idx:05d}.png")
						cv.imwrite(img_path, (img * 255).permute(1, 2, 0).numpy())

				cwd = "src/PathoNet"
				model_path = pp.abspath(self.chckpoint_path)
				results_path = f"{tp.name}.npz"
				p = subprocess.Popen(["python", "evaluation.py", "--inputPath" , tmp_dir, "--configPath", "configs/eval.json", "--modelPath", model_path, "--imageSize", str(self.tile_size), "--savePath", results_path], cwd=cwd, env=os.environ.copy())
				p.communicate()
				npyobj = np.load(results_path, allow_pickle=True)
				results = npyobj["arr_0"]
				for detections_list in results:
					for detections in detections_list:
						detections["boxes"] = torch.as_tensor(detections["boxes"])
						detections["labels"] = torch.as_tensor(detections["labels"])
						detections["scores"] = torch.as_tensor(detections["scores"])
				return results



class TilesDataset(IterableDataset):	
	def __init__(self, img, points, tile_size, overlap, masked, dlen, white_balance, pathonet_balance):
		self.img = img
		self.points = points
		self.tile_size = tile_size
		self.overlap = overlap
		self.masked = masked
		self.dlen = dlen
		self.white_balance = white_balance
		self.pathonet_balance = pathonet_balance

	
	def __iter__(self):
		worker_info = torch.utils.data.get_worker_info()
		img_path, zoom = self.img
		img = open_image_with_padding(img_path, zoom, self.tile_size, self.overlap, self.white_balance, self.pathonet_balance)
		_, detection_size, _, _, w_tiles, h_tiles = calculate_image_sizes(img_path, zoom, self.tile_size, self.overlap)
		if self.masked:
			img = mask_image(img, self.points)

		img = torch.from_numpy(img.transpose((2, 0, 1))).contiguous() / 255.0

		worker_indexes = enumerate(itertools.product(range(w_tiles), range(h_tiles)))

		if worker_info:
			worker_indexes = [(idx, tile) for idx, tile in worker_indexes if idx % worker_info.num_workers == worker_info.id]

		for _, (x, y), in worker_indexes:
			res = img[:, x*detection_size:x*detection_size+self.tile_size, y*detection_size:y*detection_size+self.tile_size]
			yield res

	def __len__(self):
		return self.dlen
		

def process_output(output, batch_size, detection_size, h_tiles, img_name, tmp_dir, points, done, date_start, items):
	annotations = []
	for widx, detections_list in enumerate(output):
		for idx, detections in enumerate(detections_list):
			fixed_index = widx * batch_size + idx

			x_offset = (fixed_index % h_tiles) * detection_size
			y_offset = (fixed_index // h_tiles) * detection_size


			boxes = detections["boxes"].numpy()

			boxes[:, 0] += x_offset
			boxes[:, 2] += x_offset
			boxes[:, 1] += y_offset
			boxes[:, 3] += y_offset
			labels = detections["labels"].numpy()
			scores = detections["scores"].numpy()
			annotations.append(np.hstack([boxes, labels[:, None], scores[:, None]]))
	
	joined_detections_tensor = np.vstack(annotations)

	tmp_path = pp.join(tmp_dir, f"data_{img_name}.npz")
	np.savez(tmp_path, points, np.zeros((0, 0)), np.zeros((0, 0)), joined_detections_tensor)
	done["status"] += len(annotations)

	date_now = datetime.now()
	take = (date_now - date_start) * (1 - (done["status"]/items)) / (done["status"]/items)
	print(f"{done['status']}/{items} {done['status']/items*100}% Running for: {date_now-date_start} ETA: {take}")



def main():
	parser = ArgumentParser()
	parser.add_argument("model", type=str, help="retinanet")
	parser.add_argument("backbone", type=str, help="resnet101")
	parser.add_argument("checkpoint", type=str)
	parser.add_argument("dataset", type=str, help="../dvc_dataset/Fulawka_new_point_dataset/")
	parser.add_argument("batch_size", type=int, help="8")
	parser.add_argument("--transformed", action=BooleanOptionalAction)
	parser.add_argument("--masked", action=BooleanOptionalAction)
	parser.add_argument("--rerun", type=str, help="Rerun those paths")
	parser.add_argument("--rescale_factor", type=float, help="Rescale factor", default=1.0)
	parser.add_argument("--tile_size", type=int, help="Tile size", default=TILE_SIZE)
	parser.add_argument("--overlap", type=int, help="Overlap", default=OVERLAP)
	parser.add_argument("--detections_dir", type=str, help="detections", default="detections")
	parser.add_argument("--white_balance", action=BooleanOptionalAction)
	parser.add_argument("--pathonet_balance", action=BooleanOptionalAction)
	parser = pl.Trainer.add_argparse_args(parser)
	args = parser.parse_args()

	trainer = pl.Trainer.from_argparse_args(args)
	predictor = None

	tile_size = args.tile_size
	overlap = args.overlap
	padding = int(tile_size * overlap / 100)
	detection_size = int(tile_size - (2 * padding))

	if args.model == "retinanet":
		predictor = PyTorchPredictor(RetinaNet(num_classes=4, backbone=args.backbone), trainer, args.checkpoint)
	if args.model == "faster_rcnn":
		predictor = PyTorchPredictor(FasterRCNN(num_classes=4, backbone=args.backbone), trainer, args.checkpoint)
	if args.model == "mask_rcnn":
		predictor = PyTorchPredictor(MaskRCNN(num_classes=4, backbone=args.backbone), trainer, args.checkpoint)
	if args.model == "ssd":
		predictor = PyTorchPredictor(SSD(num_classes=4, backbone=args.backbone), trainer, args.checkpoint)
	if args.model == "yolo":
		args.batch_size = 1
		predictor = YoloPredictor(args.checkpoint, tile_size)
	if args.model == "pathonet":
		args.batch_size = 1
		predictor = PathoNetPredictor(args.checkpoint, tile_size)

	if predictor == None:
		raise ArgumentError("Missing model")

	path_to_dataset_json = pp.join(args.dataset, "output.json")
	dataset_coco = COCO(path_to_dataset_json)
	image_ids = list(sorted(dataset_coco.imgs.keys()))

	transformed_status = "_transformed" if args.transformed else ""
	masked_status = "_masked" if args.masked else ""
	output_path = f"./{args.detections_dir}/{args.model}_{args.backbone}{transformed_status}{masked_status}"
	
	if not args.rerun:
		if pp.exists(output_path):
			rmtree(output_path)
		os.makedirs(output_path, exist_ok=True)

	rerunned_paths = None if not args.rerun else set(args.rerun.split(",")) 

	image_names = ",".join([dataset_coco.loadImgs(img_id)[0]["file_name"] for img_id in image_ids])
	print(image_names)

	zoom_map = read_zoom_map(args.dataset)
	print(zoom_map)
	imgs = []
	img_names = []
	shape_list = []
	points_list = []

	for img_id in image_ids:
		img_name = dataset_coco.loadImgs(img_id)[0]["file_name"]
		if rerunned_paths and img_name not in rerunned_paths:
			continue
		print(f"Running {img_name}")
		zoom = zoom_map.get(img_name, 1) * args.rescale_factor
		img_path = pp.join(args.dataset, img_name)

		_, _, _, _, w_tiles, h_tiles = calculate_image_sizes(img_path, zoom, tile_size, overlap)

		def convert_annotation(ann, zoom):
			if "point" in ann:
				x, y = ann["point"]
			else:
				xmin, ymin, w, h = ann["bbox"]
				x = xmin + w //2
				y = ymin + h //2
			label = ann["category_id"]
			return int(x * zoom + padding), int(y * zoom + padding), label

		points = np.array([convert_annotation(ann, zoom) for ann in dataset_coco.loadAnns(dataset_coco.getAnnIds(img_id))])

		
		imgs.append((img_path, zoom))
		shape_list.append([w_tiles, h_tiles])
		points_list.append(points)
		img_names.append(img_name)

	items = np.sum([shape[0] * shape[1] for shape in shape_list])
	done = {"status": 0}
	date_start = datetime.now()

	

	if args.model in ["pathonet", "yolo"]:
		datasets = [TilesDataset(img, points, tile_size, overlap, args.masked, w_tiles * h_tiles, args.white_balance, args.pathonet_balance) for points, img, (w_tiles, h_tiles) in zip(points_list, imgs, shape_list)]
		output_data = [(w_tiles, h_tiles, img_name, points) for img_name, points, (w_tiles, h_tiles) in zip(img_names, points_list, shape_list)]
		dataloader = DataLoader(
			ChainDataset(datasets),
			num_workers=NUM_WORKERS,
			batch_size=args.batch_size,
			pin_memory=True
		)
		
		output = predictor.predict(dataloader)
		start = 0
		for w_tiles, h_tiles, img_name, points in output_data:
			end = start + w_tiles * h_tiles
			process_output(output[start:end], args.batch_size, detection_size, h_tiles, img_name, output_path, points, done, date_start, items)
			start = end
		return

	with tempfile.TemporaryDirectory() as tmp_dir:
		for img_name, points, img, (w_tiles, h_tiles) in zip(img_names, points_list, imgs, shape_list):
			dataloader = DataLoader(
				TilesDataset(img, points, tile_size, overlap, args.masked, w_tiles * h_tiles, args.white_balance, args.pathonet_balance),
				num_workers=NUM_WORKERS,
				batch_size=args.batch_size,
				pin_memory=True
			)
			
			output = predictor.predict(dataloader)
			process_output(output, args.batch_size, detection_size, h_tiles, img_name, tmp_dir, points, done, date_start, items)

			
		data_end = datetime.now()
		print(f"Finished in {data_end-date_start}")


		for img_name in img_names:
			print(f"Saving data_{img_name}.npz")
			tmp_path = pp.join(tmp_dir, f"data_{img_name}.npz")
			data_path = pp.join(output_path, f"data_{img_name}.npz")
			shutil.move(tmp_path, data_path) 


if __name__ == "__main__":
	main()
