import torch
from torchvision.ops.boxes import _box_inter_union
import imagesize
import os.path as pp
import numpy as np
import json
import cv2 as cv
import math


def giou_loss(input_boxes, target_boxes, eps=1e-7):
    """
    Args:
        input_boxes: Tensor of shape (N, 4) or (4,).
        target_boxes: Tensor of shape (N, 4) or (4,).
        eps (float): small number to prevent division by zero
    """
    inter, union = _box_inter_union(input_boxes, target_boxes)
    iou = inter / union

    # area of the smallest enclosing box
    min_box = torch.min(input_boxes, target_boxes)
    max_box = torch.max(input_boxes, target_boxes)
    area_c = (max_box[:, 2] - min_box[:, 0]) * (max_box[:, 3] - min_box[:, 1])

    giou = iou - ((area_c - union) / (area_c + eps))

    loss = 1 - giou

    return loss.sum()


def evaluate_iou(target, pred):
    target_data = target["boxes"].int()
    pred_data = pred["boxes"].int()
    min_shape = min(pred_data.shape[0], target_data.shape[0])
    resized_target_data = target_data.narrow(0, 0, min_shape)
    resized_pred_data = pred_data.narrow(0, 0, min_shape)
    return giou_loss(resized_pred_data, resized_target_data)


def read_zoom_map(path: str):
    zomm_map_path = pp.join(path, "zoom.json")
    if not pp.exists(zomm_map_path):
        return {}
    with open(zomm_map_path, "r") as file_handler:
        return json.load(file_handler)
    return {}


def get_dominant_color(image):
    a2D = image.reshape(-1, image.shape[-1]).astype(np.uint32)
    col_range = (256, 256, 256)
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)


def mask_image(image, annotations):
    width, height, _ = image.shape
    out_mask = np.zeros((width, height), np.float32)

    for ann in annotations:
        x, y, _ = ann
        out_mask = cv.circle(out_mask, (x, y), 100, color=1, thickness=-1)

    out_image = image * out_mask[:, :, np.newaxis]
    color = get_dominant_color(image)

    black_mask = np.logical_and(
        np.logical_and(out_image[:, :, 0] == 0, out_image[:, :, 1] == 0),
        out_image[:, :, 2] == 0,
    )
    out_image[black_mask] = color
    return out_image


def calculate_image_sizes(image_path: str, zoom: float, tile_size: int, overlap: int):
    padding = int(tile_size * overlap / 100)
    detection_size = int(tile_size - (2 * padding))
    hb, wb = imagesize.get(image_path)
    h = int(hb * zoom)
    w = int(wb * zoom)
    w_tiles = math.ceil(w / detection_size)
    h_tiles = math.ceil(h / detection_size)
    return padding, detection_size, w, h, w_tiles, h_tiles


def open_image_with_padding(
    image_path: str,
    zoom: float,
    tile_size: int,
    overlap: int,
    white_balance: bool,
    pathonet_balance: bool,
):
    padding, detection_size, w, h, w_tiles, h_tiles = calculate_image_sizes(
        image_path, zoom, tile_size, overlap
    )
    print("OPEN", padding, detection_size, w, h)
    img = cv.imread(image_path).astype(np.float32)
    if white_balance:
        img = adjust_rgb_wb(img)
    if pathonet_balance:
        img = adjust_pathonet_wb(img)
    img = cv.resize(img, None, fx=zoom, fy=zoom, interpolation=cv.INTER_CUBIC)
    extra_pad_w = w_tiles * detection_size - w
    extra_pad_h = h_tiles * detection_size - h
    img = np.pad(
        img,
        ((padding, padding + extra_pad_w), (padding, padding + extra_pad_h), (0, 0)),
        "constant",
        constant_values=255,
    )
    return img


def wb(channel, perc=0.05):
    mi, ma = (np.percentile(channel, perc), np.percentile(channel, 100.0 - perc))
    channel = np.uint8(np.clip((channel - mi) * 255.0 / (ma - mi), 0, 255))
    return channel


def adjust_rgb_wb(image):
    return np.dstack([wb(channel, 0.05) for channel in cv.split(image)])


def adjust_pathonet_wb(image):
    mi_avg = [43, 49, 86]
    ma_avg = [213, 236, 240]

    def pathonet_balance(channel, mi, ma):
        return (channel * (ma - mi)) / 255.0 + mi

    return np.dstack(
        [
            pathonet_balance(channel, mi_avg[ch_idx], ma_avg[ch_idx])
            for ch_idx, channel in enumerate(cv.split(image))
        ]
    )


def pad_to_square(img, size: int, channels_first: bool = False, pad_value: float = 255):
    a, b, c = img.shape
    h, w = (a, b) if not channels_first else (b, c)
    pad_up = (size - h) // 2
    pad_down = size - h - pad_up
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left

    kwargs = (
        {
            "pad_width": [(pad_up, pad_down), (pad_left, pad_right), (0, 0)],
            "constant_values": pad_value,
        }
        if not channels_first
        else {
            "pad_width": [(0, 0), (pad_up, pad_down), (pad_left, pad_right)],
            "constant_values": pad_value,
        }
    )

    return np.pad(img, **kwargs)


def rotate_image(image, angle, channels_first: bool, borderValue=(0, 0, 0)):
    a, b, c = image.shape
    cx, cy = (a // 2, b // 2) if not channels_first else (b // 2, c // 2)
    rot_mat = cv.getRotationMatrix2D((cy, cx), angle, 1.0)
    result = cv.warpAffine(
        image,
        rot_mat,
        image.shape[1::-1],
        flags=cv.INTER_CUBIC,
        borderValue=borderValue,
    )
    return result
