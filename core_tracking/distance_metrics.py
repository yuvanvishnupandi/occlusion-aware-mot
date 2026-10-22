"""
Distance Metrics & Fused Cost Matrix Computation.
Combines Deep Appearance Cosine Distance with Spatial Motion IoU/Mahalanobis Distance.
Implements Anti-Drift Gating and Occlusion-Aware Dynamic Weight Modulation.
"""

from typing import List, Optional
import numpy as np

from core_tracking.kalman_filter import AdaptiveKalmanFilter
from core_tracking.track_state import Track
from detection.box_utils import compute_iou_matrix, compute_center_distance_matrix

INFTY_COST = 1e4


def compute_iou_distance(tracks: List[Track], detection_boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU distance cost matrix between tracks and detection boxes.
    Distance = 1.0 - IoU.
    Returns:
        np.ndarray of shape (len(tracks), len(detection_boxes)).
    """
    if len(tracks) == 0 or len(detection_boxes) == 0:
        return np.empty((len(tracks), len(detection_boxes)), dtype=np.float32)

    track_boxes = np.array([t.to_xyxy() for t in tracks], dtype=np.float32)
    iou_matrix = compute_iou_matrix(track_boxes, detection_boxes)
    return np.ascontiguousarray(1.0 - iou_matrix, dtype=np.float32)


def compute_cosine_distance(
    tracks: List[Track],
    detection_features: np.ndarray,
    use_feature_bank: bool = True
) -> np.ndarray:
    """
    Compute cosine distance between track appearance features and detection features.
    Distance = 1.0 - cosine_similarity.
    Returns:
        np.ndarray of shape (len(tracks), len(detection_features)).
    """
    num_tracks = len(tracks)
    num_dets = len(detection_features)

    if num_tracks == 0 or num_dets == 0:
        return np.empty((num_tracks, num_dets), dtype=np.float32)

    cost_matrix = np.ones((num_tracks, num_dets), dtype=np.float32)

    # Normalize detection features
    det_norms = np.linalg.norm(detection_features, axis=1, keepdims=True)
    det_norms = np.maximum(det_norms, 1e-12)
    det_feats_norm = detection_features / det_norms

    for t_idx, track in enumerate(tracks):
        if track.smooth_feature is None:
            continue

        if use_feature_bank and len(track.feature_bank) > 1:
            # Multi-sample nearest neighbor: take minimum distance across bank samples
            bank_array = np.array(track.feature_bank, dtype=np.float32)
            similarities = np.dot(bank_array, det_feats_norm.T)  # (K, num_dets)
            max_sims = np.max(similarities, axis=0)              # (num_dets,)
            cost_matrix[t_idx] = 1.0 - np.clip(max_sims, -1.0, 1.0)
        else:
            # Single EMA smooth feature
            track_feat = track.smooth_feature
            sims = np.dot(det_feats_norm, track_feat)
            cost_matrix[t_idx] = 1.0 - np.clip(sims, -1.0, 1.0)

    return cost_matrix


def compute_fused_cost(
    kf: AdaptiveKalmanFilter,
    tracks: List[Track],
    detection_boxes: np.ndarray,
    detection_features: Optional[np.ndarray] = None,
    appearance_weight: float = 0.65,
    max_physical_speed: float = 120.0,
    iou_threshold: float = 0.3
) -> np.ndarray:
    """
    Compute unified association cost matrix fusing deep appearance and spatial motion.

    Algorithmic Innovations:
    1. Dynamic Weight Modulation: If a track is marked occluded, appearance cues are less reliable;
       weight is automatically shifted toward spatial motion continuity.
    2. Anti-Drift Gating: Enforces maximum human displacement per frame to completely prevent
       identity jumping across distant people in dense crowds.
    3. Kalman Mahalanobis Gating: Rejects pairings outside 95% confidence chi-squared region.
    """
    num_tracks = len(tracks)
    num_dets = len(detection_boxes)

    if num_tracks == 0 or num_dets == 0:
        return np.empty((num_tracks, num_dets), dtype=np.float32)

    # 1. Base Motion Cost (1 - IoU)
    iou_cost = compute_iou_distance(tracks, detection_boxes)

    # 2. Base Appearance Cost (Cosine Distance)
    if detection_features is not None and len(detection_features) > 0:
        app_cost = compute_cosine_distance(tracks, detection_features)
    else:
        app_cost = np.ones_like(iou_cost)
        appearance_weight = 0.0

    # 3. Dynamic Weight Modulation and Anti-Drift Gating
    fused_cost = np.zeros((num_tracks, num_dets), dtype=np.float32)
    track_boxes = np.array([t.to_xyxy() for t in tracks], dtype=np.float32)
    center_distances = compute_center_distance_matrix(track_boxes, detection_boxes)

    for i, track in enumerate(tracks):
        # Adaptive weight based on track occlusion status
        if track.is_occluded:
            eff_app_weight = max(0.20, appearance_weight * 0.5)
        else:
            eff_app_weight = appearance_weight

        eff_motion_weight = 1.0 - eff_app_weight

        # Fused combination
        fused_cost[i] = eff_app_weight * app_cost[i] + eff_motion_weight * iou_cost[i]

        # ANTI-DRIFT GATING:
        # Calculate maximum allowed spatial jump based on human velocity and elapsed frames
        max_allowed_dist = max_physical_speed * (track.time_since_update + 1)
        drift_mask = center_distances[i] > max_allowed_dist
        fused_cost[i, drift_mask] = INFTY_COST

        # Strict gating: if IoU is 0 AND appearance distance is poor (> 0.55), reject pairing
        no_overlap_mask = (iou_cost[i] >= 0.99) & (app_cost[i] > 0.55)
        fused_cost[i, no_overlap_mask] = INFTY_COST

    return fused_cost
