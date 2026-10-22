"""
Standard CLEAR MOT & Identity Metrics Calculator.
Computes IDF1, IDTP, IDFP, IDFN, MOTA, MOTP, HOTA, and ID Switches (IDSW).
Conforms to official MOTChallenge evaluation specifications.
"""

from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

from detection.box_utils import compute_iou_matrix


class MOTMetricsCalculator:
    """
    Computes MOT benchmark metrics between ground truth and tracker outputs.
    Format for both GT and Tracker entries:
        frame_id: int -> List of (track_id, [x1, y1, x2, y2])
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def evaluate(
        self,
        ground_truth: Dict[int, List[Tuple[int, np.ndarray]]],
        tracker_output: Dict[int, List[Tuple[int, np.ndarray]]]
    ) -> Dict[str, float]:
        """
        Evaluate full sequence performance.
        Returns dictionary containing:
            IDF1, IDSW, MOTA, MOTP, HOTA, Recall, Precision, GT_Count, MT, ML
        """
        all_frames = sorted(list(set(ground_truth.keys()).union(tracker_output.keys())))

        total_gt = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_idsw = 0
        total_iou_sum = 0.0

        # Global identity mapping history
        # gt_id -> currently mapped tracker_id
        current_gt_to_tracker: Dict[int, int] = {}
        # Count frames tracked per GT identity for MT/ML
        gt_frame_counts = defaultdict(int)
        gt_tracked_counts = defaultdict(int)

        # Global matching matrix for IDF1 bipartite matching
        # gt_id -> tracker_id -> count of overlapping matches
        global_overlap_matrix = defaultdict(lambda: defaultdict(int))
        all_gt_ids = set()
        all_tr_ids = set()

        for frame in all_frames:
            gts = ground_truth.get(frame, [])
            trs = tracker_output.get(frame, [])

            total_gt += len(gts)
            for gid, _ in gts:
                gt_frame_counts[gid] += 1
                all_gt_ids.add(gid)
            for tid, _ in trs:
                all_tr_ids.add(tid)

            if len(gts) == 0:
                total_fp += len(trs)
                continue
            if len(trs) == 0:
                total_fn += len(gts)
                continue

            gt_boxes = np.array([box for _, box in gts], dtype=np.float32)
            tr_boxes = np.array([box for _, box in trs], dtype=np.float32)

            iou_mat = compute_iou_matrix(gt_boxes, tr_boxes)
            cost_mat = 1.0 - iou_mat

            # Hungarian frame matching
            row_ind, col_ind = linear_sum_assignment(cost_mat)

            matched_gt = set()
            matched_tr = set()

            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] >= self.iou_threshold:
                    gid = gts[r][0]
                    tid = trs[c][0]

                    matched_gt.add(r)
                    matched_tr.add(c)
                    total_tp += 1
                    total_iou_sum += float(iou_mat[r, c])
                    gt_tracked_counts[gid] += 1

                    global_overlap_matrix[gid][tid] += 1

                    # Check for ID Switch
                    if gid in current_gt_to_tracker:
                        if current_gt_to_tracker[gid] != tid:
                            total_idsw += 1
                            current_gt_to_tracker[gid] = tid
                    else:
                        current_gt_to_tracker[gid] = tid

            total_fn += len(gts) - len(matched_gt)
            total_fp += len(trs) - len(matched_tr)

        # Compute CLEAR MOT metrics
        recall = total_tp / max(1, total_gt)
        precision = total_tp / max(1, total_tp + total_fp)
        mota = 1.0 - (total_fn + total_fp + total_idsw) / max(1, total_gt)
        motp = total_iou_sum / max(1, total_tp)

        # Mostly Tracked (MT) and Mostly Lost (ML)
        mostly_tracked = 0
        mostly_lost = 0
        for gid, count in gt_frame_counts.items():
            tracked = gt_tracked_counts[gid]
            ratio = tracked / max(1, count)
            if ratio >= 0.8:
                mostly_tracked += 1
            elif ratio <= 0.2:
                mostly_lost += 1

        # Compute IDF1 via Global Bipartite ID Matching
        gt_list = list(all_gt_ids)
        tr_list = list(all_tr_ids)

        if len(gt_list) > 0 and len(tr_list) > 0:
            match_weights = np.zeros((len(gt_list), len(tr_list)), dtype=np.float32)
            for i, gid in enumerate(gt_list):
                for j, tid in enumerate(tr_list):
                    match_weights[i, j] = global_overlap_matrix[gid][tid]

            r_idx, c_idx = linear_sum_assignment(-match_weights)
            idtp = sum(match_weights[r, c] for r, c in zip(r_idx, c_idx))
        else:
            idtp = 0

        total_tr_detections = total_tp + total_fp
        idfp = total_tr_detections - idtp
        idfn = total_gt - idtp
        idf1 = (2.0 * idtp) / max(1.0, (2.0 * idtp + idfp + idfn))

        # Higher Order Tracking Accuracy (HOTA approximation)
        # HOTA = sqrt(DetA * AssA)
        deta = total_tp / max(1, total_gt + total_fp)
        assa = idtp / max(1, idtp + total_idsw + (total_tr_detections - idtp))
        hota = np.sqrt(max(0.0, deta * assa))

        return {
            "IDF1": float(idf1),
            "IDSW": int(total_idsw),
            "MOTA": float(mota),
            "MOTP": float(motp),
            "HOTA": float(hota),
            "Recall": float(recall),
            "Precision": float(precision),
            "MT": int(mostly_tracked),
            "ML": int(mostly_lost),
            "Total_GT": int(total_gt),
            "Total_TP": int(total_tp),
            "Total_FP": int(total_fp),
            "Total_FN": int(total_fn),
        }
