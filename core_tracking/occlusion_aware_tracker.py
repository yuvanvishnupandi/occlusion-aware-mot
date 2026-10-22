"""
Master Occlusion-Aware Multi-Object Tracker.
Integrates Adaptive Kalman Filter, Cascaded Hungarian Association,
EMA Appearance Memory Bank, and Anti-Drift Gating.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np

from core_tracking.kalman_filter import AdaptiveKalmanFilter
from core_tracking.track_state import Track, TrackStatus
from core_tracking.association_engine import CascadedAssociationEngine
from detection.box_utils import xyxy_to_xywh, compute_iou_matrix


class OcclusionAwareTracker:
    """
    Online causal Multi-Object Tracker with specialized occlusion robustness.
    """

    def __init__(
        self,
        max_age: int = 50,
        n_init: int = 3,
        fused_cost_threshold: float = 0.45,
        reid_cost_threshold: float = 0.42,
        iou_cost_threshold: float = 0.60,
        ema_alpha: float = 0.90,
        mutual_occlusion_iou_thresh: float = 0.25
    ):
        self.max_age = max_age
        self.n_init = n_init
        self.mutual_occlusion_iou_thresh = mutual_occlusion_iou_thresh

        self.kf = AdaptiveKalmanFilter()
        self.association_engine = CascadedAssociationEngine(
            fused_cost_threshold=fused_cost_threshold,
            reid_cost_threshold=reid_cost_threshold,
            iou_cost_threshold=iou_cost_threshold
        )
        self.ema_alpha = ema_alpha

        self.tracks: List[Track] = []
        self.frame_count = 0
        self.cumulative_reassociations = 0

    @property
    def total_identities(self) -> int:
        return Track._id_counter

    @property
    def next_id(self) -> int:
        return Track._id_counter + 1

    def reset(self):
        """Reset tracker state and monotonic ID counter."""
        self.tracks.clear()
        self.frame_count = 0
        self.cumulative_reassociations = 0
        Track.reset_counter()

    def _detect_mutual_occlusions(self):
        """
        Scan active tracks for mutual spatial overlap.
        If two tracks overlap by IoU > mutual_occlusion_iou_thresh,
        flag both tracks as occluded so their appearance updates freeze.
        """
        active = [t for t in self.tracks if t.status in [TrackStatus.CONFIRMED, TrackStatus.OCCLUDED]]
        if len(active) < 2:
            for t in active:
                t.mark_occluded(False, 0.0)
            return

        boxes = np.array([t.to_xyxy() for t in active], dtype=np.float32)
        iou_mat = compute_iou_matrix(boxes, boxes)
        # Clear diagonal
        np.fill_diagonal(iou_mat, 0.0)

        for i, track in enumerate(active):
            max_overlap = float(np.max(iou_mat[i]))
            if max_overlap >= self.mutual_occlusion_iou_thresh:
                track.mark_occluded(True, score=max_overlap)
            else:
                track.mark_occluded(False, score=0.0)

    def predict(self):
        """Advance motion models and detect pre-association occlusions."""
        self.frame_count += 1
        for track in self.tracks:
            track.predict(self.kf)
        self._detect_mutual_occlusions()

    def update(
        self,
        detection_boxes_xyxy: np.ndarray,
        detection_features: Optional[np.ndarray] = None,
        low_conf_boxes_xyxy: Optional[np.ndarray] = None
    ) -> List[Track]:
        """
        Process new frame detections.
        Arguments:
            detection_boxes_xyxy: Array of shape (N, 4) in [x1, y1, x2, y2]
            detection_features: Array of shape (N, 512) normalized embeddings
            low_conf_boxes_xyxy: Optional array of low-confidence detections for recovery
        Returns:
            List of currently confirmed active Track objects.
        """
        if detection_boxes_xyxy is None or len(detection_boxes_xyxy) == 0:
            detection_boxes_xyxy = np.empty((0, 4), dtype=np.float32)
        if detection_features is None or len(detection_features) == 0:
            detection_features = np.empty((0, 512), dtype=np.float32)
        if low_conf_boxes_xyxy is None or len(low_conf_boxes_xyxy) == 0:
            low_conf_boxes_xyxy = np.empty((0, 4), dtype=np.float32)

        # Split tracks into confirmed, unconfirmed (tentative), and lost buffer
        confirmed_tracks = [t for t in self.tracks if t.status in [TrackStatus.CONFIRMED, TrackStatus.OCCLUDED]]
        unconfirmed_tracks = [t for t in self.tracks if t.status == TrackStatus.TENTATIVE]
        lost_tracks = [t for t in self.tracks if t.status == TrackStatus.LOST]

        # Run cascaded data association
        matches, unmatched_tracks, unmatched_dets_idx = self.association_engine.associate(
            kf=self.kf,
            confirmed_tracks=confirmed_tracks,
            unconfirmed_tracks=unconfirmed_tracks,
            lost_tracks=lost_tracks,
            detection_boxes=detection_boxes_xyxy,
            detection_features=detection_features,
            low_conf_boxes=low_conf_boxes_xyxy
        )

        # Update matched tracks
        for track, det_idx, is_reassociated in matches:
            det_box = detection_boxes_xyxy[det_idx]
            det_feat = detection_features[det_idx] if len(detection_features) > det_idx else None

            # Convert to measurement format [cx, cy, a, h]
            cx, cy, w, h = xyxy_to_xywh(det_box)
            measurement = np.array([cx, cy, w / max(1.0, h), h], dtype=np.float32)

            track.update(
                kf=self.kf,
                measurement=measurement,
                feature=det_feat,
                is_reassociated=is_reassociated
            )

            if is_reassociated:
                self.cumulative_reassociations += 1

        # Initialize new tracks for unmatched high-confidence detections
        for det_idx in unmatched_dets_idx:
            det_box = detection_boxes_xyxy[det_idx]
            det_feat = detection_features[det_idx] if len(detection_features) > det_idx else None

            cx, cy, w, h = xyxy_to_xywh(det_box)
            measurement = np.array([cx, cy, w / max(1.0, h), h], dtype=np.float32)
            mean, covariance = self.kf.initiate(measurement)

            new_track = Track(
                mean=mean,
                covariance=covariance,
                feature=det_feat,
                n_init=self.n_init,
                max_age=self.max_age,
                ema_alpha=self.ema_alpha
            )
            self.tracks.append(new_track)

        # Remove terminated tracks
        self.tracks = [t for t in self.tracks if not t.is_terminated()]

        # Return confirmed and active tentative tracks for immediate rendering
        return [t for t in self.tracks if (t.is_confirmed() or t.hits >= 1) and t.time_since_update == 0]

    def get_stats(self) -> Dict[str, int]:
        """Return real-time tracker operational statistics."""
        confirmed_count = sum(1 for t in self.tracks if t.status == TrackStatus.CONFIRMED)
        occluded_count = sum(1 for t in self.tracks if t.is_occluded)
        lost_count = sum(1 for t in self.tracks if t.status == TrackStatus.LOST)
        return {
            "frame": self.frame_count,
            "active_tracks": confirmed_count,
            "occluded_tracks": occluded_count,
            "buffered_lost_tracks": lost_count,
            "cumulative_reassociations": self.cumulative_reassociations
        }
