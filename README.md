<h2 align="center"><b>occlusion-aware-mot</b></h2>

<p align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11-3776AB.svg?style=flat&logo=python&logoColor=white" alt="Python Version"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
    <img src="https://img.shields.io/badge/IDF1-79.5%25-success.svg" alt="IDF1">
    <img src="https://img.shields.io/badge/MOTA-82.2%25-blue.svg" alt="MOTA">
    <img src="https://img.shields.io/badge/latency-%3C3ms%20(350%2B%20FPS)-purple.svg" alt="Latency">
    <img src="https://img.shields.io/badge/tests-7%20passing-brightgreen.svg" alt="Tests">
</p>

<p align="center">Implementation of an online, identity-preserving Multi-Object Tracking (MOT) framework designed to maintain trajectory continuity through dense pedestrian occlusions and crowd crossings, in PyTorch / Python.</p>

## Install

```bash
$ pip install -r requirements.txt
```

Or install in editable mode:

```bash
$ pip install -e .
```

## Usage

```python
import cv2
from core_tracking.occlusion_aware_tracker import OcclusionAwareTracker

tracker = OcclusionAwareTracker(
    detector_weights = 'models/yolov8n.onnx',
    reid_weights     = 'models/osnet_x0_25.onnx',
    max_age          = 50
)

video = cv2.VideoCapture('data/video_sequences/transit_gate_dense.mp4')

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Causal online tracking update (returns active confirmed tracks)
    tracks = tracker.update(frame)

    for track in tracks:
        track_id = track.track_id
        bbox = track.to_tlbr()  # [x1, y1, x2, y2]
        print(f"Track ID {track_id}: bbox={bbox}")
```

### CLI Video Tracking

```bash
$ python run_tracker.py --input data/video_sequences/transit_gate_dense.mp4 --output outputs/tracked_videos/transit_gate_tracked.mp4
```

### Benchmark Evaluation Suite

Run quantitative evaluation across standard CLEAR MOT metrics (IDF1, IDSW, MOTA, HOTA):

```bash
$ python evaluate_benchmarks.py
```

### Interactive Web Studio

To launch the real-time tracking telemetry studio and FastAPI backend:

```bash
# Terminal 1: Backend
$ python server.py

# Terminal 2: React UI
$ cd frontend && npm run dev
```

Visit `http://localhost:3000` in your browser.

### Run Tests

```bash
$ python -m unittest discover tests
```

## Method

- **8D Adaptive Kalman Filter**: Constant-velocity state representation $[c_x, c_y, a, h, v_x, v_y, v_a, v_h]^T$ with dynamic process covariance inflation during occlusion to prevent spatial gate collapse.
- **Deep Appearance Embeddings (OSNet-x0.25)**: Extracts 512-dimensional $L_2$-normalized feature vectors capturing omni-scale structural and color features invariant to scale variations.
- **Occlusion-Aware EMA Freezing**: Automatically freezes appearance updates ($\alpha = 1.0$) when track spatial overlap $\text{IoU} \ge 0.25$, shielding memory banks from clothing embedding corruption.
- **Kinematic Anti-Drift Velocity Gating**: Enforces velocity admissibility constraints ($d_{\max} \le v_{\max} \cdot \Delta t$) before Hungarian optimization, preventing identity hopping across adjacent individuals.
- **Two-Stage Cascaded Matching**: Fuses cosine distance and spatial IoU cost formulations with a 50-frame lost-track re-association buffer.

## Benchmarks

Evaluated on realistic pedestrian flow and transit gate benchmark sequences:

| Method | IDF1 (%) &uarr; | ID Switches (IDSW) &darr; | MOTA (%) &uarr; | HOTA (%) &uarr; | Association Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline SORT (Motion Only) | 52.5% | 36 | 80.2% | 68.5% | 0.8 ms (1200 FPS) |
| Standard DeepSORT (No Gating) | 66.4% | 16 | 81.6% | 77.5% | 2.2 ms (450 FPS) |
| **Proposed Occlusion-Aware Tracker** | **79.5%** | **6** | **82.2%** | **85.0%** | **2.8 ms (350 FPS)** |

## Citations

```bibtex
@misc{occlusion-aware-mot,
    author = {Yuvan Vishnu Pandi},
    title  = {Identity-Preserving Multi-Object Tracking through Dense Occlusion},
    year   = {2026}
}
```
