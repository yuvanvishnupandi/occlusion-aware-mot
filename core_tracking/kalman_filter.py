"""
Adaptive Kalman Filter for 8D Bounding Box Tracking.
State: [cx, cy, a, h, vx, vy, va, vh]^T
Where:
    cx, cy: Bounding box center coordinates
    a: Aspect ratio (w / h)
    h: Box height
    vx, vy, va, vh: Corresponding velocity components
Includes adaptive covariance scaling during occlusion to prevent overconfident drift.
"""

from typing import Tuple
import numpy as np
import scipy.linalg


class AdaptiveKalmanFilter:
    """
    Kalman filter for tracking bounding boxes in image space.
    Uses constant velocity motion model with dynamic covariance adaptation.
    """

    def __init__(self):
        # 8-dimensional state vector, 4-dimensional measurement vector
        ndim = 4
        dt = 1.0

        # Motion model transition matrix F (8x8)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Measurement matrix H (4x8)
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)

        # Standard deviation weights for position and velocity uncertainty
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create track state from unassociated measurement [cx, cy, a, h].
        Returns:
            mean: State vector (8,)
            covariance: State covariance matrix (8x8)
        """
        mean_pos = measurement[:4]
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        h = measurement[3]
        std = [
            2.0 * self._std_weight_position * h,
            2.0 * self._std_weight_position * h,
            1e-2,
            2.0 * self._std_weight_position * h,
            10.0 * self._std_weight_velocity * h,
            10.0 * self._std_weight_velocity * h,
            1e-5,
            10.0 * self._std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean, covariance

    def predict(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        is_occluded: bool = False,
        time_since_update: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman state prediction step.
        If is_occluded is True, adaptively inflates process noise covariance
        proportional to time_since_update to maintain a wider spatial acceptance gate.
        """
        h = mean[3]
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]

        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)

        # Adaptive inflation for occluded tracks
        if is_occluded and time_since_update > 0:
            scale_factor = 1.0 + 0.08 * min(time_since_update, 30)
            motion_cov *= scale_factor

        mean = np.dot(self._motion_mat, mean)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )

        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space [cx, cy, a, h].
        Returns:
            projected_mean: (4,)
            projected_covariance: (4x4)
        """
        h = mean[3]
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        meas_cov = np.diag(np.square(std)).astype(np.float32)

        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = (
            np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
            + meas_cov
        )
        return projected_mean, projected_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman update step with new observed measurement [cx, cy, a, h].
        """
        proj_mean, proj_cov = self.project(mean, covariance)

        # Cholesky factorization for numerical stability
        chol_factor, lower = scipy.linalg.cho_factor(
            proj_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T

        innovation = measurement[:4] - proj_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_cov = covariance - np.linalg.multi_dot(
            (kalman_gain, proj_cov, kalman_gain.T)
        )
        return new_mean, new_cov

    def mahalanobis_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False
    ) -> np.ndarray:
        """
        Compute squared Mahalanobis distance between state distribution and measurements.
        measurements: (N, 4) in [cx, cy, a, h] format.
        Returns:
            distances: (N,) squared distances.
        """
        proj_mean, proj_cov = self.project(mean, covariance)
        if only_position:
            proj_mean = proj_mean[:2]
            proj_cov = proj_cov[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(proj_cov)
        d = measurements - proj_mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        squared_distances = np.sum(z * z, axis=0)
        return squared_distances
