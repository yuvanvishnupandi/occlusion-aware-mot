"""
Bounding Box Utilities for Multi-Object Tracking.
Provides coordinate transformations, vectorized IoU, and distance calculations.
"""

from typing import Tuple, Union
import numpy as np


def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    """
    Convert bounding box from [x1, y1, x2, y2] to center-based [cx, cy, w, h].
    """
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return np.array([cx, cy, w, h], dtype=np.float32)


def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    """
    Convert bounding box from [cx, cy, w, h] to [x1, y1, x2, y2].
    """
    cx, cy, w, h = box[0], box[1], box[2], box[3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def xyxy_to_tlwh(box: np.ndarray) -> np.ndarray:
    """
    Convert bounding box from [x1, y1, x2, y2] to top-left [top_left_x, top_left_y, w, h].
    """
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return np.array([x1, y1, w, h], dtype=np.float32)


def tlwh_to_xyxy(box: np.ndarray) -> np.ndarray:
    """
    Convert bounding box from [top_left_x, top_left_y, w, h] to [x1, y1, x2, y2].
    """
    x, y, w, h = box[0], box[1], box[2], box[3]
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between two [x1, y1, x2, y2] boxes.
    """
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of IoU matrix between N boxes_a and M boxes_b.
    Each box is in [x1, y1, x2, y2] format.
    Returns:
        np.ndarray of shape (N, M) with values in range [0, 1].
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    # Coordinates of intersection
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    areas_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    areas_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = areas_a[:, None] + areas_b[None, :] - intersection
    union = np.maximum(union, 1e-6)

    return np.ascontiguousarray(intersection / union, dtype=np.float32)


def compute_center_distance_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance between box centers.
    Returns:
        np.ndarray of shape (N, M).
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    centers_a = np.column_stack([(boxes_a[:, 0] + boxes_a[:, 2]) / 2.0, (boxes_a[:, 1] + boxes_a[:, 3]) / 2.0])
    centers_b = np.column_stack([(boxes_b[:, 0] + boxes_b[:, 2]) / 2.0, (boxes_b[:, 1] + boxes_b[:, 3]) / 2.0])

    diff = centers_a[:, None, :] - centers_b[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1)).astype(np.float32)


def clip_boxes(boxes: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Clip boxes to image boundaries [0, height] and [0, width].
    image_shape: (height, width)
    """
    if len(boxes) == 0:
        return boxes
    h, w = image_shape[:2]
    clipped = boxes.copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, w)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, h)
    return clipped
