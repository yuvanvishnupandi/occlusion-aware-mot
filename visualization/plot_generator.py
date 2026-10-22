"""
Publication-Quality Plot Generator for Multi-Object Tracking Benchmarks.
Produces ablation comparison charts, crowd density stress curves,
and occlusion recovery analysis figures.
"""

from typing import Dict, List
import os
import matplotlib.pyplot as plt
import numpy as np


class PlotGenerator:
    """
    Renders and exports clean benchmark comparison plots.
    """

    def __init__(self, output_dir: str = "outputs/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Clean modern aesthetic configuration
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.edgecolor"] = "#cccccc"
        plt.rcParams["axes.linewidth"] = 0.8

    def plot_ablation_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        filename: str = "fig1_ablation_comparison.png"
    ) -> str:
        """
        Bar chart comparing MOTA, IDF1, and ID Switches across tracker variants.
        """
        methods = list(results.keys())
        mota = [results[m].get("MOTA", 0.0) * 100 for m in methods]
        idf1 = [results[m].get("IDF1", 0.0) * 100 for m in methods]
        idsw = [results[m].get("IDSW", 0) for m in methods]

        x = np.arange(len(methods))
        width = 0.28

        fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
        ax2 = ax1.twinx()

        # Accuracies on Left Axis
        rects1 = ax1.bar(x - width / 2, idf1, width, label="IDF1 Score (%)", color="#2ecc71")
        rects2 = ax1.bar(x + width / 2, mota, width, label="MOTA (%)", color="#3498db")

        # ID Switches on Right Axis (Dashed line with markers)
        line = ax2.plot(x, idsw, color="#e74c3c", marker="o", linewidth=2.5, markersize=8, label="ID Switches (Lower is Better)")

        ax1.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Total ID Switches", color="#e74c3c", fontsize=11, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, fontsize=10, fontweight="bold")
        ax1.set_ylim(0, 105)
        ax2.set_ylim(0, max(idsw) * 1.35 if max(idsw) > 0 else 10)

        ax1.grid(axis="y", linestyle="--", alpha=0.5)
        plt.title("MOT Benchmark: Ablation Study Across Association Architectures", fontsize=12, fontweight="bold", pad=15)

        # Combined Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

        # Bar value annotations
        for rect in rects1:
            h = rect.get_height()
            ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
        for rect in rects2:
            h = rect.get_height()
            ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontsize=8)

        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath

    def plot_stress_analysis(
        self,
        densities: List[int],
        proposed_idf1: List[float],
        baseline_idf1: List[float],
        proposed_idsw: List[int],
        baseline_idsw: List[int],
        filename: str = "fig2_stress_analysis_density.png"
    ) -> str:
        """
        Plots degradation curves vs crowd density (N = 5 to 40 people).
        Demonstrates graceful degradation constraint compliance.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

        # Subplot 1: IDF1 vs Density
        ax1.plot(densities, [v * 100 for v in proposed_idf1], "o-", color="#2ecc71", linewidth=2.5, markersize=7, label="Proposed (Occlusion-Aware)")
        ax1.plot(densities, [v * 100 for v in baseline_idf1], "s--", color="#e74c3c", linewidth=2.0, markersize=6, label="Standard DeepSORT Baseline")
        ax1.set_xlabel("Crowd Density (Number of Simultaneous People)", fontweight="bold")
        ax1.set_ylabel("Identity F1-Score (IDF1 %)", fontweight="bold")
        ax1.set_title("IDF1 Identity Retention vs Crowd Density", fontweight="bold")
        ax1.set_ylim(20, 105)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="lower left")

        # Subplot 2: ID Switches vs Density
        ax2.plot(densities, proposed_idsw, "o-", color="#2ecc71", linewidth=2.5, markersize=7, label="Proposed Tracker")
        ax2.plot(densities, baseline_idsw, "s--", color="#e74c3c", linewidth=2.0, markersize=6, label="Baseline DeepSORT")
        ax2.set_xlabel("Crowd Density (Number of Simultaneous People)", fontweight="bold")
        ax2.set_ylabel("Total ID Switches (Lower is Better)", fontweight="bold")
        ax2.set_title("ID Switch Stability vs Crowd Density", fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper left")

        plt.suptitle("MOT Stress Analysis: Graceful Degradation in Dense Crowd Gates", fontsize=13, fontweight="bold", y=1.02)
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        return filepath

    def plot_occlusion_recovery_curve(
        self,
        occlusion_lengths: List[int],
        recovery_rates: List[float],
        filename: str = "fig3_occlusion_recovery_rate.png"
    ) -> str:
        """
        Plots track recovery success rate as a function of occlusion duration.
        """
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        ax.plot(occlusion_lengths, [r * 100 for r in recovery_rates], "o-", color="#3498db", linewidth=2.5, markersize=8)
        ax.fill_between(occlusion_lengths, [r * 100 for r in recovery_rates], color="#3498db", alpha=0.15)

        ax.set_xlabel("Occlusion Duration (Elapsed Frames / Frames Hidden)", fontweight="bold")
        ax.set_ylabel("Re-Association Success Rate (%)", fontweight="bold")
        ax.set_title("Identity Re-Association Resilience vs Occlusion Duration", fontweight="bold")
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle="--", alpha=0.5)

        for x, y in zip(occlusion_lengths, recovery_rates):
            ax.annotate(f"{y*100:.1f}%", xy=(x, y * 100), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")

        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        return filepath
