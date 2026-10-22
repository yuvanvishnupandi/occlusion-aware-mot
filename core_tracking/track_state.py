"""
Track State Representation and Lifecycle Machine.
Manages track progression across TENTATIVE, CONFIRMED, OCCLUDED, LOST, and TERMINATED states.
Implements EMA appearance feature banks with occlusion-aware update freezing.
"""

from collections import deque
from enum import Enum
from typing import Deque, List, Optional, Tuple
import numpy as np

from core_tracking.kalman_filter import AdaptiveKalmanFilter
from detection.box_utils import xywh_to_xyxy


class TrackStatus(Enum):
    """Lifecycle states of a tracked object."""
    TENTATIVE = 1     # Spawned, awaiting confirmation (default: 3 hits)
    CONFIRMED = 2     # Actively tracked target
    OCCLUDED = 3      # Currently undergoing dense mutual occlusion
    LOST = 4          # Temporarily unobserved, preserved in re-association buffer
    TERMINATED = 5    # Permanently expired, ready for garbage collection


class Track:
    """
    Stateful entity representing an individual tracked identity over time.
    """

    # Global monotonic identity counter
    _id_counter = 0

    @classmethod
    def reset_counter(cls):
        cls._id_counter = 0

    @classmethod
    def next_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter

    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        feature: Optional[np.ndarray] = None,
        n_init: int = 3,
        max_age: int = 60,
        ema_alpha: float = 0.90
    ):
        self.track_id = Track.next_id()
        self.mean = mean
        self.covariance = covariance
        self.status = TrackStatus.TENTATIVE

        self.n_init = n_init
        self.max_age = max_age
        self.ema_alpha = ema_alpha

        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        # If n_init is 1, immediately start as CONFIRMED
        self.status = TrackStatus.CONFIRMED if n_init <= 1 else TrackStatus.TENTATIVE

        # Occlusion tracking metrics
        self.is_occluded = False
        self.occlusion_score = 0.0
        self.reassociated_count = 0

        # Appearance feature memory
        self.smooth_feature: Optional[np.ndarray] = None
        self.feature_bank: List[np.ndarray] = []
        if feature is not None:
            norm = np.linalg.norm(feature)
            if norm > 1e-6:
                feature = feature / norm
            self.smooth_feature = feature.copy()
            self.feature_bank.append(feature.copy())

        # Trajectory history: recent box centers for trail visualization and velocity
        self.trajectory: Deque[Tuple[float, float]] = deque(maxlen=45)
        cx, cy = self.mean[0], self.mean[1]
        self.trajectory.append((float(cx), float(cy)))

    def to_tlwh(self) -> np.ndarray:
        """Get current estimated bounding box as [top_left_x, top_left_y, w, h]."""
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # w = aspect_ratio * h
        ret[:2] -= ret[2:] / 2.0
        return ret

    def to_xyxy(self) -> np.ndarray:
        """Get current estimated bounding box as [x1, y1, x2, y2]."""
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return xywh_to_xyxy(np.array([cx, cy, w, h], dtype=np.float32))

    def predict(self, kf: AdaptiveKalmanFilter):
        """Advance Kalman state prediction to the next frame."""
        self.mean, self.covariance = kf.predict(
            self.mean,
            self.covariance,
            is_occluded=self.is_occluded or (self.status in [TrackStatus.OCCLUDED, TrackStatus.LOST]),
            time_since_update=self.time_since_update
        )
        self.age += 1
        self.time_since_update += 1

        # Check for state transitions upon missed detections
        if self.time_since_update > 1:
            if self.status == TrackStatus.CONFIRMED:
                self.status = TrackStatus.OCCLUDED if self.is_occluded else TrackStatus.LOST
            elif self.status == TrackStatus.TENTATIVE:
                self.status = TrackStatus.TERMINATED

        if self.time_since_update > self.max_age:
            self.status = TrackStatus.TERMINATED

    def update(
        self,
        kf: AdaptiveKalmanFilter,
        measurement: np.ndarray,
        feature: Optional[np.ndarray] = None,
        is_reassociated: bool = False
    ):
        """
        Update Kalman state with matched detection measurement [cx, cy, a, h].
        Updates appearance feature bank with occlusion-aware gating.
        """
        self.mean, self.covariance = kf.update(self.mean, self.covariance, measurement)
        self.hits += 1
        self.time_since_update = 0

        # Trajectory record
        cx, cy = self.mean[0], self.mean[1]
        self.trajectory.append((float(cx), float(cy)))

        # Lifecycle confirmation
        if self.status == TrackStatus.TENTATIVE and self.hits >= self.n_init:
            self.status = TrackStatus.CONFIRMED
        elif self.status in [TrackStatus.OCCLUDED, TrackStatus.LOST]:
            self.status = TrackStatus.CONFIRMED
            if is_reassociated:
                self.reassociated_count += 1

        # Occlusion-Aware Appearance Feature Update
        if feature is not None:
            norm = np.linalg.norm(feature)
            if norm > 1e-6:
                feature = feature / norm

            # CRITICAL INNOVATION:
            # If the person is under severe occlusion, freeze the feature update.
            # Do NOT update with corrupted pixels from the occluder.
            if not self.is_occluded:
                if self.smooth_feature is None:
                    self.smooth_feature = feature.copy()
                else:
                    # Exponential Moving Average update
                    self.smooth_feature = (
                        self.ema_alpha * self.smooth_feature + (1.0 - self.ema_alpha) * feature
                    )
                    norm = np.linalg.norm(self.smooth_feature)
                    if norm > 1e-6:
                        self.smooth_feature /= norm

                # Maintain history bank up to 10 reliable appearance samples
                self.feature_bank.append(feature.copy())
                if len(self.feature_bank) > 10:
                    self.feature_bank.pop(0)

    def mark_occluded(self, is_occluded: bool, score: float = 0.0):
        """Set mutual occlusion state and severity score."""
        self.is_occluded = is_occluded
        self.occlusion_score = score
        if is_occluded and self.status == TrackStatus.CONFIRMED:
            self.status = TrackStatus.OCCLUDED
        elif not is_occluded and self.status == TrackStatus.OCCLUDED and self.time_since_update == 0:
            self.status = TrackStatus.CONFIRMED

    def is_confirmed(self) -> bool:
        return self.status == TrackStatus.CONFIRMED

    def is_active(self) -> bool:
        return self.status in [TrackStatus.CONFIRMED, TrackStatus.OCCLUDED]

    def is_reassociable(self) -> bool:
        return self.status in [TrackStatus.OCCLUDED, TrackStatus.LOST] and (self.time_since_update <= self.max_age)

    def is_terminated(self) -> bool:
        return self.status == TrackStatus.TERMINATED
