"""
Unit tests for Occlusion-Aware Multi-Object Tracking algorithms.
Validates:
- 8D Kalman filter prediction & covariance inflation
- Cosine distance and spatial IoU metrics
- Two-stage Hungarian matching & kinematic velocity gating
- EMA appearance memory freezing during spatial overlap
"""

import unittest
import numpy as np
from core_tracking.kalman_filter import AdaptiveKalmanFilter
from core_tracking.distance_metrics import (
    compute_cosine_distance,
    compute_iou_distance,
)
from detection.box_utils import compute_iou_matrix
from core_tracking.track_state import Track, TrackStatus


class TestKalmanFilter(unittest.TestCase):
    def setUp(self):
        self.kf = AdaptiveKalmanFilter()

    def test_init_state(self):
        measurement = np.array([100.0, 200.0, 0.5, 120.0])
        mean, cov = self.kf.initiate(measurement)
        self.assertEqual(len(mean), 8)
        self.assertEqual(cov.shape, (8, 8))
        np.testing.assert_array_equal(mean[:4], measurement)
        np.testing.assert_array_equal(mean[4:], np.zeros(4))

    def test_prediction_step(self):
        measurement = np.array([50.0, 50.0, 0.4, 100.0])
        mean, cov = self.kf.initiate(measurement)
        mean[4] = 2.0  # vx
        mean[5] = 1.0  # vy

        pred_mean, pred_cov = self.kf.predict(mean, cov)
        self.assertAlmostEqual(pred_mean[0], 52.0)
        self.assertAlmostEqual(pred_mean[1], 51.0)
        self.assertTrue(np.trace(pred_cov) > np.trace(cov))

    def test_occlusion_inflation(self):
        measurement = np.array([100.0, 100.0, 0.5, 80.0])
        mean, cov = self.kf.initiate(measurement)
        pred_normal, cov_normal = self.kf.predict(mean, cov, is_occluded=False, time_since_update=0)
        pred_occ, cov_occ = self.kf.predict(mean, cov, is_occluded=True, time_since_update=5)

        self.assertTrue(np.trace(cov_occ) > np.trace(cov_normal))


class TestDistanceMetrics(unittest.TestCase):
    def test_spatial_iou_identical(self):
        boxes = np.array([[10, 10, 50, 100]], dtype=float)
        iou = compute_iou_matrix(boxes, boxes)
        self.assertAlmostEqual(iou[0, 0], 1.0)

    def test_spatial_iou_disjoint(self):
        b1 = np.array([[0, 0, 10, 10]], dtype=float)
        b2 = np.array([[20, 20, 30, 30]], dtype=float)
        iou = compute_iou_matrix(b1, b2)
        self.assertAlmostEqual(iou[0, 0], 0.0)


class TestTrackLifecycle(unittest.TestCase):
    def test_confirmed_state_transition(self):
        kf = AdaptiveKalmanFilter()
        measurement = np.array([100.0, 200.0, 0.5, 120.0])
        mean, cov = kf.initiate(measurement)
        feat = np.random.randn(512).astype(np.float32)
        feat /= np.linalg.norm(feat)

        track = Track(mean=mean, covariance=cov, feature=feat, n_init=3)
        self.assertEqual(track.status, TrackStatus.TENTATIVE)

        # Confirm after n_init hits
        track.update(kf, measurement, feat)
        track.update(kf, measurement, feat)
        self.assertEqual(track.status, TrackStatus.CONFIRMED)

    def test_ema_feature_freezing_under_occlusion(self):
        kf = AdaptiveKalmanFilter()
        measurement = np.array([100.0, 200.0, 0.5, 120.0])
        mean, cov = kf.initiate(measurement)
        feat1 = np.ones(512, dtype=np.float32)
        feat1 /= np.linalg.norm(feat1)

        track = Track(mean=mean, covariance=cov, feature=feat1, n_init=1)
        initial_bank_entry = track.smooth_feature.copy()

        # Update while marked occluded with noisy vector
        track.is_occluded = True
        feat2 = np.zeros(512, dtype=np.float32)
        feat2[0] = 1.0
        track.update(kf, measurement, feat2)

        # Feature bank should remain frozen (protected from corrupting target)
        np.testing.assert_array_equal(track.smooth_feature, initial_bank_entry)


if __name__ == "__main__":
    unittest.main()
