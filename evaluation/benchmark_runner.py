"""
Benchmark Runner for Comparative MOT Evaluation.
Evaluates tracking resilience under realistic transit gate occlusion scenarios:
1. Baseline SORT (Motion IoU only, max_age = 2)
2. Standard DeepSORT (Fixed Appearance without Occlusion Gating, max_age = 15)
3. Proposed Occlusion-Aware Tracker (Full Pipeline with Re-association Buffer & Anti-Drift)
"""

from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

from core_tracking.occlusion_aware_tracker import OcclusionAwareTracker
from evaluation.mot_metrics import MOTMetricsCalculator
from detection.box_utils import tlwh_to_xyxy, compute_iou_matrix


def load_mot_ground_truth(gt_path: str) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    """
    Parse standard MOT challenge ground truth file.
    Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
    Returns:
        frame_dict: frame_num -> list of (person_id, [x1, y1, x2, y2])
    """
    gt_dict = defaultdict(list)
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_idx = int(parts[0])
            person_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            box = tlwh_to_xyxy(np.array([x, y, w, h], dtype=np.float32))
            gt_dict[frame_idx].append((person_id, box))

    return dict(gt_dict)


class BenchmarkRunner:
    """
    Runs comparative evaluation between baseline trackers and the proposed occlusion-aware tracker.
    """

    def __init__(self, iou_thresh: float = 0.5):
        self.metrics_calc = MOTMetricsCalculator(iou_threshold=iou_thresh)

    def _generate_gt_features(self, ground_truth: Dict[int, List[Tuple[int, np.ndarray]]]):
        """Generate orthogonal 512-D identity embeddings for each ground truth person."""
        gt_ids = sorted(list(set(gid for f in ground_truth.values() for gid, _ in f)))
        rng = np.random.RandomState(42)
        base_features = {}
        for gid in gt_ids:
            feat = rng.randn(512).astype(np.float32)
            feat /= np.linalg.norm(feat)
            base_features[gid] = feat
        return base_features

    def _simulate_detector_feed(
        self,
        gts: List[Tuple[int, np.ndarray]],
        base_features: Dict[int, np.ndarray],
        pillar_x_bounds: Tuple[float, float] = (360.0, 440.0),
        rng: np.random.RandomState = None
    ) -> Tuple[np.ndarray, np.ndarray, List[int], np.ndarray]:
        """
        Simulate realistic transit gate detector behavior:
        1. Detections are dropped when a person passes behind a turnstile pillar.
        2. Detections are dropped when heavily occluded by a foreground person (mutual occlusion).
        3. Feature embeddings receive minor sensor noise (and corruption if overlapping).
        Returns:
            visible_boxes: (N, 4)
            visible_features: (N, 512)
            visible_gids: list of GT IDs visible
            low_conf_boxes: (M, 4)
        """
        if rng is None:
            rng = np.random.RandomState(42)

        visible_boxes = []
        visible_features = []
        visible_gids = []
        low_conf_boxes = []

        if len(gts) == 0:
            return np.empty((0, 4)), np.empty((0, 512)), [], np.empty((0, 4))

        # Check mutual occlusion
        boxes = np.array([box for _, box in gts], dtype=np.float32)
        iou_mat = compute_iou_matrix(boxes, boxes)
        np.fill_diagonal(iou_mat, 0.0)

        for i, (gid, box) in enumerate(gts):
            cx = (box[0] + box[2]) / 2.0
            # 1. Check pillar occlusion
            if pillar_x_bounds[0] <= cx <= pillar_x_bounds[1]:
                # Completely occluded behind turnstile pillar
                continue

            # 2. Check mutual occlusion (person behind person)
            max_overlap = float(np.max(iou_mat[i])) if len(boxes) > 1 else 0.0
            if max_overlap > 0.45:
                # 50% chance of detector dropping occluded target, or generating low confidence box
                other_idx = int(np.argmax(iou_mat[i]))
                # The person further down (higher y2) is closer to camera; the one higher up is occluded
                if box[3] < boxes[other_idx][3]:
                    # Target is behind foreground person
                    low_conf_boxes.append(box)
                    continue

            # Normal visible detection
            feat = base_features[gid].copy()
            # Minor camera noise
            feat += rng.randn(512).astype(np.float32) * 0.04
            # If overlapping, appearance feature is corrupted by occluding target
            if max_overlap > 0.20:
                other_gid = gts[int(np.argmax(iou_mat[i]))][0]
                blend_factor = min(0.65, max_overlap * 1.2)
                feat = (1.0 - blend_factor) * feat + blend_factor * base_features[other_gid]

            feat /= np.linalg.norm(feat)

            visible_boxes.append(box)
            visible_features.append(feat)
            visible_gids.append(gid)

        v_boxes = np.array(visible_boxes, dtype=np.float32) if visible_boxes else np.empty((0, 4), dtype=np.float32)
        v_feats = np.array(visible_features, dtype=np.float32) if visible_features else np.empty((0, 512), dtype=np.float32)
        l_boxes = np.array(low_conf_boxes, dtype=np.float32) if low_conf_boxes else np.empty((0, 4), dtype=np.float32)

        return v_boxes, v_feats, visible_gids, l_boxes

    def run_sort_baseline(
        self,
        ground_truth: Dict[int, List[Tuple[int, np.ndarray]]]
    ) -> Dict[int, List[Tuple[int, np.ndarray]]]:
        """
        Simulate standard SORT:
        - Motion IoU matching only (no appearance Re-ID).
        - Short retention buffer (max_age = 2).
        - When a person disappears behind the pillar (>2 frames), track terminates.
        - Upon reappearance, spawns a new ID (causing ID switches & track fragmentation).
        """
        tracker = OcclusionAwareTracker(
            max_age=2,
            n_init=2,
            fused_cost_threshold=0.60,
            ema_alpha=0.0
        )
        tracker.reset()

        base_features = self._generate_gt_features(ground_truth)
        results = defaultdict(list)
        all_frames = sorted(ground_truth.keys())
        rng = np.random.RandomState(42)

        for frame_idx in all_frames:
            tracker.predict()
            gts = ground_truth[frame_idx]
            v_boxes, _, _, _ = self._simulate_detector_feed(gts, base_features, rng=rng)

            # SORT runs with no appearance features
            active_tracks = tracker.update(
                detection_boxes_xyxy=v_boxes,
                detection_features=np.empty((0, 512), dtype=np.float32)
            )

            for track in active_tracks:
                results[frame_idx].append((track.track_id, track.to_xyxy()))

        return dict(results)

    def run_deepsort_baseline(
        self,
        ground_truth: Dict[int, List[Tuple[int, np.ndarray]]]
    ) -> Dict[int, List[Tuple[int, np.ndarray]]]:
        """
        Simulate Standard DeepSORT:
        - Fixed appearance features with cosine distance.
        - Moderate buffer (max_age = 15).
        - NO occlusion gating: Appearance features update even when persons overlap,
          corrupting embeddings and causing ID drift / swapping in tight groups.
        """
        tracker = OcclusionAwareTracker(
            max_age=15,
            n_init=2,
            fused_cost_threshold=0.50,
            reid_cost_threshold=0.45,
            mutual_occlusion_iou_thresh=1.0  # Disables occlusion detection/freezing
        )
        tracker.reset()

        base_features = self._generate_gt_features(ground_truth)
        results = defaultdict(list)
        all_frames = sorted(ground_truth.keys())
        rng = np.random.RandomState(42)

        for frame_idx in all_frames:
            tracker.predict()
            gts = ground_truth[frame_idx]
            v_boxes, v_feats, _, _ = self._simulate_detector_feed(gts, base_features, rng=rng)

            active_tracks = tracker.update(
                detection_boxes_xyxy=v_boxes,
                detection_features=v_feats
            )

            for track in active_tracks:
                results[frame_idx].append((track.track_id, track.to_xyxy()))

        return dict(results)

    def run_proposed_tracker(
        self,
        ground_truth: Dict[int, List[Tuple[int, np.ndarray]]]
    ) -> Dict[int, List[Tuple[int, np.ndarray]]]:
        """
        Run the Proposed Full Occlusion-Aware Tracker:
        - Adaptive Kalman Filter with dynamic covariance inflation during occlusion.
        - EMA appearance updates FROZEN during mutual occlusion (IoU > 0.25).
        - Cascaded Re-association Memory Buffer (max_age = 50 frames).
        - Anti-Drift Gating preventing spatial identity jumping.
        - Low-confidence observation recovery.
        """
        tracker = OcclusionAwareTracker(
            max_age=50,
            n_init=2,
            fused_cost_threshold=0.45,
            reid_cost_threshold=0.42,
            ema_alpha=0.90,
            mutual_occlusion_iou_thresh=0.25
        )
        tracker.reset()

        base_features = self._generate_gt_features(ground_truth)
        results = defaultdict(list)
        all_frames = sorted(ground_truth.keys())
        rng = np.random.RandomState(42)

        for frame_idx in all_frames:
            tracker.predict()
            gts = ground_truth[frame_idx]
            v_boxes, v_feats, _, l_boxes = self._simulate_detector_feed(gts, base_features, rng=rng)

            active_tracks = tracker.update(
                detection_boxes_xyxy=v_boxes,
                detection_features=v_feats,
                low_conf_boxes_xyxy=l_boxes
            )

            for track in active_tracks:
                results[frame_idx].append((track.track_id, track.to_xyxy()))

        return dict(results)

    def run_full_comparison(
        self,
        gt_path: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Executes comparative benchmarking on the specified ground truth file.
        """
        gt = load_mot_ground_truth(gt_path)

        res_sort = self.run_sort_baseline(gt)
        res_deepsort = self.run_deepsort_baseline(gt)
        res_proposed = self.run_proposed_tracker(gt)

        metrics_sort = self.metrics_calc.evaluate(gt, res_sort)
        metrics_deepsort = self.metrics_calc.evaluate(gt, res_deepsort)
        metrics_proposed = self.metrics_calc.evaluate(gt, res_proposed)

        return {
            "Baseline SORT (Motion Only)": metrics_sort,
            "Standard DeepSORT": metrics_deepsort,
            "Proposed Occlusion-Aware Tracker": metrics_proposed
        }
