"""
Example: Evaluating CLEAR MOT Metrics on Custom Benchmark Data.
Demonstrates:
- Loading ground truth bounding box annotations (MOT Challenge format: frame, id, x, y, w, h, conf, ...)
- Evaluating IDF1, ID Switches (IDSW), MOTA, and HOTA
- Printing formatted evaluation summary tables
"""

import argparse
from evaluation.benchmark_runner import BenchmarkRunner, load_mot_ground_truth
from evaluation.mot_metrics import MOTMetricsCalculator


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLEAR MOT metrics on benchmark ground truth")
    parser.add_argument("--gt", type=str, default="data/ground_truth/transit_gate_dense_gt.txt",
                        help="Path to ground truth text file")
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="Spatial IoU match threshold")
    args = parser.parse_args()

    print(f"Loading ground truth: {args.gt}")
    gt_dict = load_mot_ground_truth(args.gt)
    print(f"Total annotated frames: {len(gt_dict)}")

    runner = BenchmarkRunner()
    results = runner.run_sequence(
        sequence_name="Custom Benchmark",
        ground_truth=gt_dict,
        iou_threshold=args.iou_thresh
    )

    print("\n" + "=" * 50)
    print("      QUANTITATIVE BENCHMARK EVALUATION")
    print("=" * 50)
    print(f"IDF1 Score            : {results.idf1 * 100:.2f}%")
    print(f"Identity Switches (IDSW): {results.id_switches}")
    print(f"MOTA Accuracy         : {results.mota * 100:.2f}%")
    print(f"HOTA Score            : {results.hota * 100:.2f}%")
    print(f"Total Trajectories    : {results.total_ground_truth_trajectories}")
    print("=" * 50)


if __name__ == "__main__":
    main()
