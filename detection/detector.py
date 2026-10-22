"""
Person Detector using YOLOv8 ONNX Runtime.
Provides real-time person detection with Non-Maximum Suppression and coordinate mapping.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
import cv2
import numpy as np
import onnxruntime as ort

from detection.box_utils import clip_boxes


@dataclass
class Detection:
    """Represents a single detected person object."""
    box_xyxy: np.ndarray      # [x1, y1, x2, y2]
    confidence: float         # Detection confidence score [0.0 - 1.0]
    class_id: int = 0         # Class 0 corresponds to Person in COCO dataset
    class_name: str = "person"


class YOLOv8Detector:
    """
    High-performance, CPU/GPU accelerated YOLOv8 ONNX Person Detector.
    Uses letterboxing, onnxruntime C++ inference, and cv2.dnn.NMSBoxes.
    """

    def __init__(
        self,
        model_path: str = "models/yolov8n.onnx",
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
        device: str = "CPU"
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.device = device

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at: {self.model_path}. "
                "Please ensure models/yolov8n.onnx exists."
            )

        providers = ["CPUExecutionProvider"]
        if device.upper() == "CUDA" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Resize image with letterbox padding to maintain aspect ratio.
        Returns:
            processed_tensor: normalized tensor shape (1, 3, 640, 640)
            scale: scaling ratio applied
            pad: (pad_w, pad_h)
        """
        orig_h, orig_w = image.shape[:2]
        target_w, target_h = self.input_size

        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))

        pad_w = (target_w - new_w) / 2.0
        pad_h = (target_h - new_h) / 2.0

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # Convert BGR to RGB, normalize to [0, 1], and transpose to (1, 3, H, W)
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        return np.ascontiguousarray(blob), scale, (pad_w, pad_h)

    def postprocess(
        self,
        raw_output: np.ndarray,
        orig_shape: Tuple[int, int],
        scale: float,
        pad: Tuple[float, float]
    ) -> List[Detection]:
        """
        Decode YOLOv8 raw output (1, 84, 8400) to bounding boxes in original image coordinates.
        Filter for Person (class_id = 0) and apply NMS.
        """
        # Shape: (8400, 84)
        predictions = np.squeeze(raw_output).T
        pad_w, pad_h = pad

        # Column 4 is person score in standard COCO 80-classes
        person_scores = predictions[:, 4]
        mask = person_scores >= self.confidence_threshold

        filtered_preds = predictions[mask]
        filtered_scores = person_scores[mask]

        if len(filtered_preds) == 0:
            return []

        # Extract box centers and dimensions: cx, cy, w, h
        cx = filtered_preds[:, 0]
        cy = filtered_preds[:, 1]
        w = filtered_preds[:, 2]
        h = filtered_preds[:, 3]

        # Map back from letterbox to original image space
        x1 = (cx - w / 2.0 - pad_w) / scale
        y1 = (cy - h / 2.0 - pad_h) / scale
        x2 = (cx + w / 2.0 - pad_w) / scale
        y2 = (cy + h / 2.0 - pad_h) / scale

        boxes = np.column_stack([x1, y1, x2, y2])
        boxes = clip_boxes(boxes, orig_shape)

        # Convert to cv2 format [x, y, w, h] for NMSBoxes
        cv2_boxes = []
        scores_list = []
        for box, score in zip(boxes, filtered_scores):
            bx1, by1, bx2, by2 = box
            bw = max(0.0, bx2 - bx1)
            bh = max(0.0, by2 - by1)
            if bw > 5.0 and bh > 10.0:  # Minimum viable person size
                cv2_boxes.append([int(bx1), int(by1), int(bw), int(bh)])
                scores_list.append(float(score))

        if not cv2_boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            cv2_boxes,
            scores_list,
            score_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold
        )

        detections: List[Detection] = []
        if len(indices) > 0:
            for idx in indices.flatten():
                bx, by, bw, bh = cv2_boxes[idx]
                box_xyxy = np.array([bx, by, bx + bw, by + bh], dtype=np.float32)
                detections.append(
                    Detection(
                        box_xyxy=box_xyxy,
                        confidence=scores_list[idx],
                        class_id=0,
                        class_name="person"
                    )
                )

        return detections

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run end-to-end detection on a single BGR image.
        Returns:
            List of Detection objects.
        """
        orig_shape = image.shape[:2]
        blob, scale, pad = self.preprocess(image)
        outputs = self.session.run([self.output_name], {self.input_name: blob})
        detections = self.postprocess(outputs[0], orig_shape, scale, pad)
        return detections
