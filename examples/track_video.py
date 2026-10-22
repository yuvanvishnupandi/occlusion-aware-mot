"""
Example: Running Online Identity-Preserving MOT on a Video File.
Demonstrates:
- Initializing YOLOv8 detector & OSNet Re-ID extractor
- Frame-by-frame causal online tracking
- Exporting tracked bounding box trajectories to CSV
"""

import argparse
import csv
import cv2
from core_tracking.occlusion_aware_tracker import OcclusionAwareTracker


def main():
    parser = argparse.ArgumentParser(description="Track video with Occlusion-Aware MOT")
    parser.add_argument("--input", type=str, default="data/video_sequences/transit_gate_dense.mp4",
                        help="Input video file path")
    parser.add_argument("--output_csv", type=str, default="outputs/trajectories.csv",
                        help="Output CSV path for identity trajectory coordinates")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold")
    parser.add_argument("--max_age", type=int, default=50, help="Max lost frames before track deletion")
    args = parser.parse_args()

    tracker = OcclusionAwareTracker(
        detector_weights="models/yolov8n.onnx",
        reid_weights="models/osnet_x0_25.onnx",
        conf_threshold=args.conf,
        max_age=args.max_age
    )

    cap = cv2.VideoCapture(args.input)
    frame_idx = 0
    records = []

    print(f"Opening video: {args.input}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Process frame online (strictly causal)
        active_tracks = tracker.update(frame)

        for track in active_tracks:
            x1, y1, x2, y2 = track.to_tlbr()
            records.append([frame_idx, track.track_id, x1, y1, x2 - x1, y2 - y1])

        if frame_idx % 30 == 0:
            print(f"Processed frame {frame_idx} | Active tracks: {len(active_tracks)}")

    cap.release()

    # Save trajectory coordinates
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "x", "y", "w", "h"])
        writer.writerows(records)

    print(f"Tracking complete. Logged {len(records)} detections across {frame_idx} frames to {args.output_csv}")


if __name__ == "__main__":
    main()
