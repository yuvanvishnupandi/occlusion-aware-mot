"""
Cascaded Association Engine for Multi-Object Tracking.
Implements Hungarian matching (linear_sum_assignment) with two-stage cascaded association:
Stage 1: Confirmed Active Tracks association using Fused Cost Matrix.
Stage 2: Re-association of Occluded and Lost Tracks from memory buffer.
Stage 3: Low-confidence detection matching for partially occluded recovery.
"""

from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from core_tracking.kalman_filter import AdaptiveKalmanFilter
from core_tracking.track_state import Track, TrackStatus
from core_tracking.distance_metrics import (
    compute_fused_cost,
    compute_cosine_distance,
    compute_iou_distance,
    INFTY_COST
)


def linear_assignment(
    cost_matrix: np.ndarray,
    cost_threshold: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solve linear sum assignment problem using Hungarian algorithm.
    Returns:
        matches: list of (row_idx, col_idx) where cost <= cost_threshold
        unmatched_rows: list of row indices not matched
        unmatched_cols: list of col indices not matched
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_rows = list(range(cost_matrix.shape[0]))
    unmatched_cols = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= cost_threshold:
            matches.append((int(r), int(c)))
            if r in unmatched_rows:
                unmatched_rows.remove(r)
            if c in unmatched_cols:
                unmatched_cols.remove(c)

    return matches, unmatched_rows, unmatched_cols


class CascadedAssociationEngine:
    """
    Orchestrates the multi-stage association and re-association logic.
    """

    def __init__(
        self,
        fused_cost_threshold: float = 0.45,
        reid_cost_threshold: float = 0.42,
        iou_cost_threshold: float = 0.60
    ):
        self.fused_cost_threshold = fused_cost_threshold
        self.reid_cost_threshold = reid_cost_threshold
        self.iou_cost_threshold = iou_cost_threshold

    def associate(
        self,
        kf: AdaptiveKalmanFilter,
        confirmed_tracks: List[Track],
        unconfirmed_tracks: List[Track],
        lost_tracks: List[Track],
        detection_boxes: np.ndarray,
        detection_features: np.ndarray,
        low_conf_boxes: np.ndarray = np.empty((0, 4))
    ) -> Tuple[List[Tuple[Track, int, bool]], List[Track], List[int]]:
        """
        Execute full cascaded data association.
        Returns:
            matches: List of (Track, detection_index, is_reassociated)
            unmatched_tracks: List of Track objects
            unmatched_detections: List of detection indices
        """
        matched_results: List[Tuple[Track, int, bool]] = []

        # ==========================================================
        # STAGE 1: Match Confirmed Tracks with High-Confidence Dets
        # ==========================================================
        if len(confirmed_tracks) > 0 and len(detection_boxes) > 0:
            cost_matrix_stage1 = compute_fused_cost(
                kf=kf,
                tracks=confirmed_tracks,
                detection_boxes=detection_boxes,
                detection_features=detection_features,
                appearance_weight=0.65
            )

            matches_s1, unmatched_confirmed_idx, unmatched_dets_idx = linear_assignment(
                cost_matrix_stage1,
                cost_threshold=self.fused_cost_threshold
            )

            for t_idx, d_idx in matches_s1:
                matched_results.append((confirmed_tracks[t_idx], d_idx, False))

            remaining_confirmed_tracks = [confirmed_tracks[i] for i in unmatched_confirmed_idx]
        else:
            remaining_confirmed_tracks = confirmed_tracks.copy()
            unmatched_dets_idx = list(range(len(detection_boxes)))

        # ==========================================================
        # STAGE 2: Match Unconfirmed / Tentative Tracks
        # ==========================================================
        unmatched_unconfirmed_tracks: List[Track] = []
        if len(unconfirmed_tracks) > 0 and len(unmatched_dets_idx) > 0:
            sub_boxes = detection_boxes[unmatched_dets_idx]
            sub_feats = detection_features[unmatched_dets_idx] if len(detection_features) > 0 else np.empty((0, 512))

            cost_matrix_stage2 = compute_fused_cost(
                kf=kf,
                tracks=unconfirmed_tracks,
                detection_boxes=sub_boxes,
                detection_features=sub_feats,
                appearance_weight=0.40
            )

            matches_s2, unmatched_unconfirmed_idx, remaining_sub_idx = linear_assignment(
                cost_matrix_stage2,
                cost_threshold=self.iou_cost_threshold
            )

            for u_idx, sub_d_idx in matches_s2:
                actual_det_idx = unmatched_dets_idx[sub_d_idx]
                matched_results.append((unconfirmed_tracks[u_idx], actual_det_idx, False))

            unmatched_dets_idx = [unmatched_dets_idx[i] for i in remaining_sub_idx]
            unmatched_unconfirmed_tracks = [unconfirmed_tracks[i] for i in unmatched_unconfirmed_idx]
        else:
            unmatched_unconfirmed_tracks = unconfirmed_tracks.copy()

        # ==========================================================
        # STAGE 3: Re-associate Occluded & Lost Tracks from Buffer
        # ==========================================================
        candidate_lost_tracks = [t for t in lost_tracks if t.is_reassociable()]

        if len(candidate_lost_tracks) > 0 and len(unmatched_dets_idx) > 0:
            sub_boxes = detection_boxes[unmatched_dets_idx]
            sub_feats = (
                detection_features[unmatched_dets_idx]
                if len(detection_features) > 0
                else np.empty((len(unmatched_dets_idx), 512), dtype=np.float32)
            )

            # Re-association cost based on Deep Appearance + Spatial Gating
            reid_cost = compute_cosine_distance(
                candidate_lost_tracks,
                sub_feats,
                use_feature_bank=True
            )

            matches_s3, unmatched_lost_idx, remaining_sub_idx = linear_assignment(
                reid_cost,
                cost_threshold=self.reid_cost_threshold
            )

            for l_idx, sub_d_idx in matches_s3:
                actual_det_idx = unmatched_dets_idx[sub_d_idx]
                matched_results.append((candidate_lost_tracks[l_idx], actual_det_idx, True))

            unmatched_dets_idx = [unmatched_dets_idx[i] for i in remaining_sub_idx]
            remaining_lost_tracks = [candidate_lost_tracks[i] for i in unmatched_lost_idx]
        else:
            remaining_lost_tracks = candidate_lost_tracks

        # ==========================================================
        # STAGE 4: Partial Observation Recovery (Low-Confidence Dets)
        # ==========================================================
        if len(remaining_confirmed_tracks) > 0 and len(low_conf_boxes) > 0:
            iou_cost = compute_iou_distance(remaining_confirmed_tracks, low_conf_boxes)
            matches_s4, unmatched_active_idx, _ = linear_assignment(
                iou_cost,
                cost_threshold=self.iou_cost_threshold
            )
            for a_idx, _ in matches_s4:
                track = remaining_confirmed_tracks[a_idx]
                track.mark_occluded(True, score=0.8)

            unmatched_tracks = (
                [remaining_confirmed_tracks[i] for i in unmatched_active_idx]
                + unmatched_unconfirmed_tracks
                + remaining_lost_tracks
            )
        else:
            unmatched_tracks = remaining_confirmed_tracks + unmatched_unconfirmed_tracks + remaining_lost_tracks

        return matched_results, unmatched_tracks, unmatched_dets_idx
