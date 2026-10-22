import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
Ablation Study Runner.
Isolates the individual algorithmic contribution of:
1. Full Proposed Occlusion-Aware Tracker
2. w/o EMA Feature Freezing during Occlusion
3. w/o Anti-Drift Gating Constraints
4. w/o Cascaded Re-association Memory Buffer
5. Motion-Only Baseline (SORT)
"""

import os
from collections import defaultdict
import numpy as np

from evaluation.benchmark_runner import BenchmarkRunner, load_mot_ground_truth
from core_tracking.occlusion_aware_tracker import OcclusionAwareTracker
from evaluation.mot_metrics import MOTMetricsCalculator
from visualization.plot_generator import PlotGenerator


def run_ablation_study():
    gt_path = "data/ground_truth/transit_gate_dense_gt.txt"
    if not os.path.exists(gt_path):
        from evaluate_benchmarks import run_benchmark
        run_benchmark()

    gt = load_mot_ground_truth(gt_path)
    runner = BenchmarkRunner()
    metrics_calc = MOTMetricsCalculator(iou_threshold=0.5)
    base_features = runner._generate_gt_features(gt)
    all_frames = sorted(gt.keys())

    print("=" * 80)
    print("MOT ABLATION STUDY: COMPONENT CONTRIBUTION ANALYSIS")
    print("Isolating Algorithmic Building Blocks under Transit Gate Occlusion")
    print("=" * 80)

    ablation_configs = {
        "1. Full Proposed Tracker": {
            "max_age": 50,
            "n_init": 2,
            "fused_cost_thresh": 0.45,
            "reid_cost_thresh": 0.42,
            "ema_alpha": 0.90,
            "mutual_occ_thresh": 0.25,
            "use_features": True
        },
        "2. w/o EMA Freezing (Unprotected)": {
            "max_age": 50,
            "n_init": 2,
            "fused_cost_thresh": 0.45,
            "reid_cost_thresh": 0.42,
            "ema_alpha": 0.90,
            "mutual_occ_thresh": 1.0,  # Never freezes EMA, allowing corrupted updates
            "use_features": True
        },
        "3. w/o Re-association Buffer (Short)": {
            "max_age": 5,             # Short retention, cannot bridge long occlusions
            "n_init": 2,
            "fused_cost_thresh": 0.45,
            "reid_cost_thresh": 0.42,
            "ema_alpha": 0.90,
            "mutual_occ_thresh": 0.25,
            "use_features": True
        },
        "4. Motion Only (SORT Baseline)": {
            "max_age": 2,
            "n_init": 2,
            "fused_cost_thresh": 0.60,
            "reid_cost_thresh": 0.60,
            "ema_alpha": 0.0,
            "mutual_occ_thresh": 1.0,
            "use_features": False
        }
    }

    ablation_results = {}

    for config_name, cfg in ablation_configs.items():
        tracker = OcclusionAwareTracker(
            max_age=cfg["max_age"],
            n_init=cfg["n_init"],
            fused_cost_threshold=cfg["fused_cost_thresh"],
            reid_cost_threshold=cfg["reid_cost_thresh"],
            ema_alpha=cfg["ema_alpha"],
            mutual_occlusion_iou_thresh=cfg["mutual_occ_thresh"]
        )
        tracker.reset()

        res_dict = defaultdict(list)
        rng = np.random.RandomState(42)

        for frame_idx in all_frames:
            tracker.predict()
            gts = gt[frame_idx]
            v_boxes, v_feats, _, l_boxes = runner._simulate_detector_feed(gts, base_features, rng=rng)

            if not cfg["use_features"]:
                v_feats = np.empty((0, 512), dtype=np.float32)

            active_tracks = tracker.update(
                detection_boxes_xyxy=v_boxes,
                detection_features=v_feats,
                low_conf_boxes_xyxy=l_boxes
            )

            for track in active_tracks:
                res_dict[frame_idx].append((track.track_id, track.to_xyxy()))

        metrics = metrics_calc.evaluate(gt, res_dict)
        ablation_results[config_name] = metrics

    # Format Table Output
    header = f"{'Ablation Variant':<40} | {'IDF1 (%)':<9} | {'MOTA (%)':<9} | {'IDSW':<6} | {'HOTA (%)':<9}"
    divider = "-" * len(header)
    print("\n" + header)
    print(divider)

    for name, m in ablation_results.items():
        print(f"{name:<40} | {m['IDF1']*100:<8.1f}% | {m['MOTA']*100:<8.1f}% | {m['IDSW']:<6} | {m['HOTA']*100:<8.1f}%")

    print(divider)

    full_idf1 = ablation_results["1. Full Proposed Tracker"]["IDF1"] * 100
    no_ema_idf1 = ablation_results["2. w/o EMA Freezing (Unprotected)"]["IDF1"] * 100
    no_buf_idf1 = ablation_results["3. w/o Re-association Buffer (Short)"]["IDF1"] * 100

    print("\nABLATION INSIGHTS:")
    print(f" -> Contribution of Occlusion-Aware EMA Freezing  : +{full_idf1 - no_ema_idf1:.1f}% IDF1 (prevents appearance corruption during overlap)")
    print(f" -> Contribution of Re-Association Memory Buffer  : +{full_idf1 - no_buf_idf1:.1f}% IDF1 (rescues tracks after pillar disappearance)")
    print(f" -> Overall Performance Gain vs Motion Only (SORT): +{full_idf1 - ablation_results['4. Motion Only (SORT Baseline)']['IDF1']*100:.1f}% IDF1")

    # Save Ablation Chart
    plotter = PlotGenerator(output_dir="outputs/plots")
    plotter.plot_ablation_comparison(ablation_results, filename="fig1_ablation_comparison.png")
    print("\nAblation study chart refreshed at: outputs/plots/fig1_ablation_comparison.png")
    print("=" * 80)

    return ablation_results


if __name__ == "__main__":
    run_ablation_study()
