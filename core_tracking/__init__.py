"""
Core Tracking package for Identity-Preserving Multi-Object Tracking through Dense Occlusion.
"""

from core_tracking.kalman_filter import AdaptiveKalmanFilter
from core_tracking.appearance_extractor import AppearanceFeatureExtractor
from core_tracking.track_state import Track, TrackStatus
from core_tracking.distance_metrics import (
    compute_iou_distance,
    compute_cosine_distance,
    compute_fused_cost,
    INFTY_COST,
)
from core_tracking.association_engine import (
    linear_assignment,
    CascadedAssociationEngine,
)
from core_tracking.occlusion_aware_tracker import OcclusionAwareTracker

__all__ = [
    "AdaptiveKalmanFilter",
    "AppearanceFeatureExtractor",
    "Track",
    "TrackStatus",
    "compute_iou_distance",
    "compute_cosine_distance",
    "compute_fused_cost",
    "INFTY_COST",
    "linear_assignment",
    "CascadedAssociationEngine",
    "OcclusionAwareTracker",
]
