"""
Benchmark Evaluation Runner.
Executes comparative evaluation across Baseline SORT, Standard DeepSORT,
and Proposed Occlusion-Aware Tracker on transit gate ground truth.
Exports comparison figures and prints metrics table.
"""

import os
from evaluation.benchmark_runner import BenchmarkRunner
from visualization.plot_generator import PlotGenerator


def run_benchmark():
    gt_path = "data/ground_truth/transit_gate_dense_gt.txt"
    if not os.path.exists(gt_path):
        from data.transit_gate_simulation import generate_transit_gate_sequence
        print("Generating benchmark ground truth sequence...")
        generate_transit_gate_sequence(
            output_video_path="data/video_sequences/transit_gate_dense.mp4",
            output_gt_path=gt_path,
            num_frames=300,
            num_people=10
        )

    print("=" * 75)
    print("ZED DIGITAL MOT BENCHMARK EVALUATION")
    print("Evaluating Tracking Algorithms on Transit Gate Occlusion Benchmark")
    print("=" * 75)

    runner = BenchmarkRunner(iou_thresh=0.5)
    results = runner.run_full_comparison(gt_path)

    # Print Table
    header = f"{'Method':<35} | {'IDF1 (%)':<9} | {'MOTA (%)':<9} | {'IDSW':<6} | {'HOTA (%)':<9} | {'MT':<4} | {'ML':<4}"
    divider = "-" * len(header)
    print("\n" + header)
    print(divider)

    for method_name, metrics in results.items():
        idf1_str = f"{metrics['IDF1'] * 100:.1f}%"
        mota_str = f"{metrics['MOTA'] * 100:.1f}%"
        idsw_str = str(metrics['IDSW'])
        hota_str = f"{metrics['HOTA'] * 100:.1f}%"
        mt_str = str(metrics['MT'])
        ml_str = str(metrics['ML'])
        print(f"{method_name:<35} | {idf1_str:<9} | {mota_str:<9} | {idsw_str:<6} | {hota_str:<9} | {mt_str:<4} | {ml_str:<4}")

    print(divider)

    # Highlight Improvements
    sort_idsw = results["Baseline SORT (Motion Only)"]["IDSW"]
    prop_idsw = results["Proposed Occlusion-Aware Tracker"]["IDSW"]
    prop_idf1 = results["Proposed Occlusion-Aware Tracker"]["IDF1"] * 100
    base_idf1 = results["Standard DeepSORT"]["IDF1"] * 100

    idsw_reduction = ((sort_idsw - prop_idsw) / max(1, sort_idsw)) * 100
    idf1_gain = prop_idf1 - base_idf1

    print("\nKEY ACHIEVEMENTS:")
    print(f" -> ID Switch Reduction vs SORT Baseline    : {idsw_reduction:.1f}% reduction (from {sort_idsw} down to {prop_idsw})")
    print(f" -> IDF1 Identity Retention Gain vs DeepSORT: +{idf1_gain:.1f}% absolute improvement")
    print(f" -> Mostly Tracked (MT) Targets              : {results['Proposed Occlusion-Aware Tracker']['MT']} / {results['Proposed Occlusion-Aware Tracker']['MT'] + results['Proposed Occlusion-Aware Tracker']['ML']}")

    # Generate Plot
    plotter = PlotGenerator(output_dir="outputs/plots")
    plot_path = plotter.plot_ablation_comparison(results, filename="fig1_ablation_comparison.png")
    print(f"\nGenerated benchmark figure saved to: {plot_path}")
    print("=" * 75)

    return results


if __name__ == "__main__":
    run_benchmark()
