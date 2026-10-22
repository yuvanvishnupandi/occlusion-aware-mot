"""
Main Video Tracking Pipeline for Identity-Preserving Multi-Object Tracking.
Usage:
    python run_tracker.py --input data/video_sequences/transit_gate_dense.mp4 --output outputs/tracked_videos/transit_gate_tracked.mp4
    python run_tracker.py --input data/video_sequences/people-detection.mp4 --output outputs/tracked_videos/people_detection_tracked.mp4
"""

import argparse
import os
import time
import cv2
import numpy as np
from tqdm import tqdm
from typing import Optional, Callable, Dict, Any

from detection.detector import YOLOv8Detector
from core_tracking.appearance_extractor import AppearanceFeatureExtractor
from core_tracking.occlusion_aware_tracker import OcclusionAwareTracker
from core_tracking.track_state import Track
from visualization.video_annotator import VideoAnnotator


def process_video(
    input_path: str,
    output_path: str,
    conf_thresh: float = 0.25,
    max_age: int = 50,
    max_frames: int = 0,
    stride: int = 1,
    show_preview: bool = False,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Runs full detection, feature extraction, and occlusion-aware tracking on video.
    Supports a real-time callback for syncing progress, telemetry, and terminal logs.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video file not found: {input_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not cap.isOpened() or width <= 0 or height <= 0:
        raise ValueError(f"Could not open or decode video stream: {input_path}")

    # Optimal export resolution (maximum width 960 for standard video playback)
    if width > 960:
        out_w = 960
        out_h = int(height * (960.0 / width))
    else:
        out_w = width
        out_h = height

    # Ensure dimensions are strictly even numbers for H.264 / OpenH264 codec
    out_w = max(2, (out_w // 2) * 2)
    out_h = max(2, (out_h // 2) * 2)

    init_msg = f"[{time.strftime('%H:%M:%S')}] [INIT] Loading YOLOv8-Nano detector (conf={conf_thresh:.2f}, nms=0.45)..."
    print(init_msg)
    if callback:
        callback({"log": init_msg, "progress": 5})

    detector = YOLOv8Detector(
        model_path="models/yolov8n.onnx",
        confidence_threshold=conf_thresh,
        nms_threshold=0.45
    )

    reid_msg = f"[{time.strftime('%H:%M:%S')}] [INIT] Loading OSNet-x0.25 Re-ID model (512-D L2-normalized embeddings)..."
    print(reid_msg)
    if callback:
        callback({"log": reid_msg, "progress": 10})

    feature_extractor = AppearanceFeatureExtractor(
        model_path="models/osnet_x0_25.onnx",
        crop_size=(128, 256)
    )

    tracker_msg = f"[{time.strftime('%H:%M:%S')}] [TRACKER] 8D Kalman State Estimator ready (max_age={max_age}, alpha=0.90, occ_iou=0.25)."
    print(tracker_msg)
    if callback:
        callback({"log": tracker_msg, "progress": 15})

    Track.reset_counter()
    tracker = OcclusionAwareTracker(
        max_age=max_age,
        n_init=2,
        fused_cost_threshold=0.45,
        reid_cost_threshold=0.42,
        ema_alpha=0.90,
        mutual_occlusion_iou_thresh=0.25
    )
    annotator = VideoAnnotator()

    stream_msg = f"[{time.strftime('%H:%M:%S')}] [STREAM] Video input: {total_frames} frames @ {fps:.1f} FPS (Res: {out_w}x{out_h})"
    print(stream_msg)
    if callback:
        callback({"log": stream_msg, "progress": 20})

    # Output FPS adjusted for frame stride to maintain normal video playback speed
    stride = max(1, stride)
    out_fps = max(1.0, fps / stride)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h))

    raw_frame_idx = 0
    frame_idx = 0
    t_start = time.time()
    latencies = []
    prev_recoveries = 0
    prev_identities = 0

    target_processed = (total_frames + stride - 1) // stride
    pbar = tqdm(total=target_processed, desc="Tracking Video Progress", unit="frame")

    while cap.isOpened():
        if max_frames > 0 and raw_frame_idx >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        raw_frame_idx += 1

        if stride > 1 and (raw_frame_idx - 1) % stride != 0:
            continue

        frame_idx += 1
        t_frame_start = time.time()

        # 1. Motion prediction step
        tracker.predict()

        # 2. Person detection step
        detections = detector.detect(frame)

        if len(detections) > 0:
            boxes = np.array([d.box_xyxy for d in detections], dtype=np.float32)
            confs = [d.confidence for d in detections]

            # 3. Extract deep appearance features for detected persons
            features = feature_extractor.extract(frame, boxes)

            # 4. Association & Track State update
            active_tracks = tracker.update(
                detection_boxes_xyxy=boxes,
                detection_features=features
            )
        else:
            active_tracks = tracker.update(
                detection_boxes_xyxy=np.empty((0, 4), dtype=np.float32),
                detection_features=np.empty((0, 512), dtype=np.float32)
            )

        frame_latency = (time.time() - t_frame_start) * 1000.0  # ms
        latencies.append(frame_latency)
        current_fps = 1000.0 / max(1.0, frame_latency)

        # 5. Visualization rendering
        annotator.draw_tracks(frame, active_tracks, draw_trails=True)
        stats = tracker.get_stats()
        annotator.draw_glass_hud(frame, stats, fps=current_fps)

        if (out_w, out_h) != (width, height):
            frame_export = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        else:
            frame_export = frame

        out.write(frame_export)
        pbar.update(1)

        # Notify callback with real-time live syncing telemetry
        if callback:
            progress_pct = int(20 + 75 * (frame_idx / max(1, target_processed)))
            
            # Check for newly spawned identities
            current_ids = tracker.total_identities
            if current_ids > prev_identities:
                new_count = current_ids - prev_identities
                prev_identities = current_ids
                callback({
                    "log": f"[{time.strftime('%H:%M:%S')}] [DETECTION] Spawned & confirmed {new_count} new identity track(s). Total IDs: {current_ids}",
                    "progress": progress_pct
                })

            # Check for new re-association events across occlusion
            if stats["cumulative_reassociations"] > prev_recoveries:
                diff = stats["cumulative_reassociations"] - prev_recoveries
                prev_recoveries = stats["cumulative_reassociations"]
                reid_evt = f"[{time.strftime('%H:%M:%S')}] [RE-ID EVENT] Target re-associated across dense occlusion! Total Recoveries: {prev_recoveries}"
                callback({"log": reid_evt, "progress": progress_pct})

            # Check for active occlusion
            if stats["occluded_tracks"] > 0 and (frame_idx % 3 == 0):
                occ_evt = f"[{time.strftime('%H:%M:%S')}] [OCCLUSION ALERT] {stats['occluded_tracks']} tracks in mutual overlap (IoU >= 0.25) -> EMA features frozen."
                callback({"log": occ_evt, "progress": progress_pct})

            # Periodic live telemetry log (Frame 1, every 2 processed frames, or final frame)
            if frame_idx == 1 or frame_idx % 2 == 0 or raw_frame_idx >= total_frames:
                f_log = f"[{time.strftime('%H:%M:%S')}] Frame {raw_frame_idx:04d}/{total_frames}: {len(detections)} dets | {stats['active_tracks']} active | {stats['occluded_tracks']} occluded | {stats['cumulative_reassociations']} recoveries ({current_fps:.1f} FPS)"
                callback({
                    "log": f_log,
                    "progress": progress_pct,
                    "frame": raw_frame_idx,
                    "total_frames": total_frames,
                    "active_tracks": stats["active_tracks"],
                    "occluded_tracks": stats["occluded_tracks"],
                    "reassociations": stats["cumulative_reassociations"],
                    "fps": round(current_fps, 1),
                    "latency_ms": round(frame_latency, 1)
                })

    cap.release()
    out.release()
    pbar.close()

    total_time = time.time() - t_start
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    done_msg = f"[{time.strftime('%H:%M:%S')}] [COMPLETE] Full video processed: {raw_frame_idx}/{total_frames} frames ({frame_idx} sampled) in {total_time:.2f}s | {tracker.cumulative_reassociations} recoveries | Total Unique IDs: {tracker.total_identities} | Avg Speed: {avg_fps:.1f} FPS"
    print(done_msg)
    if callback:
        callback({"log": done_msg, "progress": 100})

    results = {
        "total_frames": raw_frame_idx,
        "processed_frames": frame_idx,
        "total_time": round(total_time, 2),
        "avg_fps": round(avg_fps, 1),
        "avg_latency_ms": round(avg_latency, 1),
        "unique_identities": tracker.total_identities,
        "cumulative_reassociations": tracker.cumulative_reassociations,
        "output_path": output_path
    }
    return results
    return results


def main():
    parser = argparse.ArgumentParser(description="Occlusion-Aware Identity-Preserving MOT Pipeline")
    parser.add_argument("--input", type=str, default="data/video_sequences/transit_gate_dense.mp4",
                        help="Path to input video file")
    parser.add_argument("--output", type=str, default="outputs/tracked_videos/transit_gate_tracked.mp4",
                        help="Path to output annotated video file")
    parser.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold")
    parser.add_argument("--max_age", type=int, default=50, help="Maximum frame buffer for lost tracks")
    parser.add_argument("--max_frames", type=int, default=0, help="Cap processing at N frames (0 for full video)")
    args = parser.parse_args()

    process_video(
        input_path=args.input,
        output_path=args.output,
        conf_thresh=args.conf,
        max_age=args.max_age,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
