"""
Detection package for person detection and bounding box utilities.
"""

from detection.box_utils import (
    xyxy_to_xywh,
    xywh_to_xyxy,
    xyxy_to_tlwh,
    tlwh_to_xyxy,
    calculate_iou,
    compute_iou_matrix,
    compute_center_distance_matrix,
    clip_boxes,
)
from detection.detector import Detection, YOLOv8Detector

__all__ = [
    "xyxy_to_xywh",
    "xywh_to_xyxy",
    "xyxy_to_tlwh",
    "tlwh_to_xyxy",
    "calculate_iou",
    "compute_iou_matrix",
    "compute_center_distance_matrix",
    "clip_boxes",
    "Detection",
    "YOLOv8Detector",
]
