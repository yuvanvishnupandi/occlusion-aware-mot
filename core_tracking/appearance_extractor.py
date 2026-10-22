"""
Deep Appearance Feature Extractor for Person Re-Identification.
Uses pre-trained OSNet-x0.25 ONNX model trained on MSMT17 to produce 512-D L2-normalized embeddings.
Includes batch-size alignment and a multi-region color histogram fallback.
"""

from typing import List, Optional
import os
import cv2
import numpy as np
import onnxruntime as ort


class AppearanceFeatureExtractor:
    """
    Extracts discriminative appearance feature embeddings from bounding box crops.
    """

    def __init__(
        self,
        model_path: str = "models/osnet_x0_25.onnx",
        crop_size: tuple = (128, 256),  # (width, height) standard Re-ID aspect
        device: str = "CPU"
    ):
        self.model_path = model_path
        self.crop_size = crop_size
        self.device = device
        self.use_onnx = False

        if os.path.exists(self.model_path):
            try:
                providers = ["CPUExecutionProvider"]
                if device.upper() == "CUDA" and "CUDAExecutionProvider" in ort.get_available_providers():
                    providers.insert(0, "CUDAExecutionProvider")

                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session_options.intra_op_num_threads = 2

                self.session = ort.InferenceSession(
                    self.model_path,
                    sess_options=session_options,
                    providers=providers
                )
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                self.use_onnx = True
            except Exception as e:
                print(f"Warning: Could not initialize OSNet ONNX: {e}. Falling back to spatial color extractor.")
                self.use_onnx = False
        else:
            print(f"Notice: {self.model_path} not found. Using spatial color histogram fallback.")
            self.use_onnx = False

        # ImageNet normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def _extract_crop(self, frame: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
        """Safely crop bounding box from image with boundary checks."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            crop = np.zeros((self.crop_size[1], self.crop_size[0], 3), dtype=np.uint8)
        return crop

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """Resize to 256x128, RGB convert, ImageNet normalize, transpose to (3, H, W)."""
        resized = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.mean) / self.std
        tensor = np.transpose(normalized, (2, 0, 1))
        return tensor

    def _color_histogram_features(self, crop: np.ndarray) -> np.ndarray:
        """
        Fast fallback appearance descriptor: 3-part spatial grid (head, torso, legs)
        with HSV color histograms. Produces 512-D L2-normalized vector.
        """
        resized = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # Split vertically into 3 spatial regions (Head 25%, Torso 45%, Legs 30%)
        h, w = hsv.shape[:2]
        r1 = hsv[0:int(h * 0.25), :]
        r2 = hsv[int(h * 0.25):int(h * 0.70), :]
        r3 = hsv[int(h * 0.70):h, :]

        feature_parts = []
        for region in [r1, r2, r3]:
            # 16 Hue bins, 8 Saturation bins, 8 Value bins
            hist_h = cv2.calcHist([region], [0], None, [16], [0, 180]).flatten()
            hist_s = cv2.calcHist([region], [1], None, [8], [0, 256]).flatten()
            hist_v = cv2.calcHist([region], [2], None, [8], [0, 256]).flatten()
            feature_parts.extend([hist_h, hist_s, hist_v])

        feat = np.concatenate(feature_parts).astype(np.float32)
        # Pad or interpolate to 512 dimensions for consistency with OSNet
        if len(feat) < 512:
            feat = np.pad(feat, (0, 512 - len(feat)), mode="constant")
        else:
            feat = feat[:512]

        norm = np.linalg.norm(feat)
        if norm > 1e-6:
            feat /= norm
        return feat

    def extract(self, frame: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """
        Extract 512-D normalized appearance embeddings for a batch of bounding boxes.
        boxes_xyxy: shape (N, 4)
        Returns:
            features: shape (N, 512)
        """
        num_boxes = len(boxes_xyxy)
        if num_boxes == 0:
            return np.empty((0, 512), dtype=np.float32)

        crops = [self._extract_crop(frame, box) for box in boxes_xyxy]

        if not self.use_onnx:
            feats = [self._color_histogram_features(c) for c in crops]
            return np.array(feats, dtype=np.float32)

        # OSNet ONNX model requires input batch size of exactly 16
        processed = [self._preprocess_crop(c) for c in crops]
        batch_size = len(processed)
        embeddings_list = []

        try:
            for chunk_start in range(0, batch_size, 16):
                chunk = processed[chunk_start:chunk_start + 16]
                actual_len = len(chunk)
                while len(chunk) < 16:
                    chunk.append(np.zeros_like(processed[0]))
                chunk_array = np.stack(chunk, axis=0).astype(np.float32)
                outputs = self.session.run([self.output_name], {self.input_name: chunk_array})
                embeddings_list.append(outputs[0][:actual_len])

            embeddings = np.concatenate(embeddings_list, axis=0)

            # L2 Normalize embeddings along feature axis
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            normalized_embeddings = embeddings / norms
            return normalized_embeddings.astype(np.float32)

        except Exception as e:
            print(f"Warning: ONNX inference failed ({e}), falling back to color histograms.")
            feats = [self._color_histogram_features(c) for c in crops]
            return np.array(feats, dtype=np.float32)
