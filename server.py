"""
Occlusion-Aware MOT - Backend API Server
Provides high-performance endpoints for:
- Video streaming with HTTP Range support
- Live video tracking pipeline execution (YOLOv8 + OSNet + Occlusion-Aware Tracker)
- MOT benchmark evaluation data (IDF1, IDSW, MOTA, HOTA)
- Crowd density stress analysis & Hungarian cost matrix calculation
"""

import os
import shutil
import time
import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional

from run_tracker import process_video

app = FastAPI(title="Occlusion-Aware MOT API", version="1.0.0")

# Enable CORS for local Vite dev server (http://localhost:5173) and any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
os.makedirs("outputs/tracked_videos", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("data/video_sequences", exist_ok=True)
os.makedirs("data/uploads", exist_ok=True)

# Mount media static directories
app.mount("/media/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/media/data", StaticFiles(directory="data"), name="data")

# In-memory job tracker for video processing
tracking_jobs = {}


@app.get("/api/health")
def health_check():
    return {"status": "online", "model": "8D Kalman Filter + OSNet-x0.25 Re-ID"}


@app.get("/api/benchmarks")
def get_benchmarks():
    """Returns official MOT metric evaluation results."""
    return {
        "benchmarks": [
            {
                "method": "Baseline SORT (Motion Only)",
                "idf1": 52.5,
                "mota": 80.2,
                "idsw": 36,
                "hota": 68.5,
                "mostly_tracked": "7 / 7",
                "mostly_lost": "0 / 7",
                "latency_ms": 0.8,
                "throughput_fps": 1200
            },
            {
                "method": "Standard DeepSORT (No Gating)",
                "idf1": 66.4,
                "mota": 81.6,
                "idsw": 16,
                "hota": 77.5,
                "mostly_tracked": "7 / 7",
                "mostly_lost": "0 / 7",
                "latency_ms": 2.2,
                "throughput_fps": 450
            },
            {
                "method": "Proposed Occlusion-Aware Tracker",
                "idf1": 79.5,
                "mota": 82.2,
                "idsw": 6,
                "hota": 85.0,
                "mostly_tracked": "7 / 7",
                "mostly_lost": "0 / 7",
                "latency_ms": 2.8,
                "throughput_fps": 350
            }
        ],
        "ablations": [
            {
                "variant": "Full Proposed Tracker",
                "idf1": 79.5,
                "idsw": 6,
                "hota": 85.0,
                "role": "Adaptive Kalman + EMA Freezing + 50-Frame Buffer + Anti-Drift Gating"
            },
            {
                "variant": "w/o EMA Feature Freezing",
                "idf1": 79.4,
                "idsw": 6,
                "hota": 85.0,
                "role": "Appearance feature bank corrupted during overlap; degrades in extended crossings"
            },
            {
                "variant": "w/o Re-Association Buffer (max_age = 5)",
                "idf1": 58.4,
                "idsw": 23,
                "hota": 72.5,
                "role": "Premature track deletion behind pillars, forcing false identity re-creation"
            },
            {
                "variant": "Motion Only (SORT Baseline)",
                "idf1": 52.5,
                "idsw": 36,
                "hota": 68.5,
                "role": "Linear motion extrapolation only without appearance embedding"
            }
        ],
        "highlights": {
            "idsw_reduction": "-83.3% vs Baseline SORT (36 -> 6)",
            "idf1_gain": "+13.1% absolute improvement over DeepSORT",
            "bounded_latency": "< 3 ms association time per frame"
        }
    }


@app.get("/api/stress")
def get_stress_data():
    """Returns crowd density and occlusion stress analysis figures and numbers."""
    return {
        "density_stress": [
            {"density": 4,  "proposed_idf1": 74.3, "baseline_idf1": 60.6, "proposed_idsw": 2,  "baseline_idsw": 4,  "fps": 1857},
            {"density": 8,  "proposed_idf1": 73.7, "baseline_idf1": 72.0, "proposed_idsw": 5,  "baseline_idsw": 7,  "fps": 1042},
            {"density": 12, "proposed_idf1": 74.7, "baseline_idf1": 62.3, "proposed_idsw": 8,  "baseline_idsw": 16, "fps": 723},
            {"density": 16, "proposed_idf1": 74.0, "baseline_idf1": 68.2, "proposed_idsw": 17, "baseline_idsw": 19, "fps": 568},
            {"density": 20, "proposed_idf1": 64.7, "baseline_idf1": 57.1, "proposed_idsw": 28, "baseline_idsw": 43, "fps": 525},
            {"density": 24, "proposed_idf1": 50.7, "baseline_idf1": 45.7, "proposed_idsw": 64, "baseline_idsw": 80, "fps": 424}
        ],
        "occlusion_retention": [
            {"hidden_frames": 5,  "retention_rate": 96.0},
            {"hidden_frames": 10, "retention_rate": 94.0},
            {"hidden_frames": 15, "retention_rate": 91.0},
            {"hidden_frames": 20, "retention_rate": 88.0},
            {"hidden_frames": 25, "retention_rate": 85.0},
            {"hidden_frames": 30, "retention_rate": 81.0},
            {"hidden_frames": 35, "retention_rate": 78.0},
            {"hidden_frames": 40, "retention_rate": 70.0},
            {"hidden_frames": 45, "retention_rate": 62.0},
            {"hidden_frames": 50, "retention_rate": 53.0},
            {"hidden_frames": 55, "retention_rate": 35.0},
            {"hidden_frames": 60, "retention_rate": 18.0}
        ]
    }


@app.get("/api/cost-matrix")
def get_cost_matrix(w_app: float = 0.65):
    """Computes Hungarian cost matrix with appearance and motion fusion."""
    w_motion = 1.0 - w_app
    tracks = ["Track #1 (Active)", "Track #2 (Occluded)", "Track #3 (Lost Memory)", "Track #4 (Active)"]
    detections = ["Detection A", "Detection B", "Detection C", "Detection D"]

    app_dist = [
        [0.12, 0.88, 0.76, 0.94],
        [0.82, 0.15, 0.91, 0.65],
        [0.78, 0.85, 0.22, 0.90],
        [0.91, 0.68, 0.84, 0.18]
    ]
    iou_dist = [
        [0.08, 0.95, 0.90, 0.99],
        [0.92, 0.40, 0.85, 0.70],
        [0.99, 0.99, 0.99, 0.99],
        [0.89, 0.75, 0.80, 0.11]
    ]

    fused = []
    for i in range(4):
        row = []
        for j in range(4):
            val = round(w_app * app_dist[i][j] + w_motion * iou_dist[i][j], 3)
            row.append(val)
        fused.append(row)

    return {
        "tracks": tracks,
        "detections": detections,
        "app_matrix": app_dist,
        "iou_matrix": iou_dist,
        "fused_matrix": fused,
        "w_app": w_app,
        "w_motion": round(w_motion, 2)
    }


def execute_tracking_job(job_id: str, input_path: str, output_path: str, conf_thresh: float, max_age: int, max_frames: int, stride: int = 1):
    try:
        tracking_jobs[job_id]["status"] = "processing"
        tracking_jobs[job_id]["progress"] = 5
        t0 = time.time()

        def job_callback(event: dict):
            if "log" in event:
                tracking_jobs[job_id]["logs"].append(event["log"])
                if len(tracking_jobs[job_id]["logs"]) > 100:
                    tracking_jobs[job_id]["logs"] = tracking_jobs[job_id]["logs"][-100:]
            if "progress" in event:
                tracking_jobs[job_id]["progress"] = event["progress"]
            for k in ["frame", "total_frames", "active_tracks", "occluded_tracks", "reassociations", "fps", "latency_ms"]:
                if k in event:
                    tracking_jobs[job_id][k] = event[k]

        results = process_video(
            input_path=input_path,
            output_path=output_path,
            conf_thresh=conf_thresh,
            max_age=max_age,
            max_frames=max_frames,
            stride=stride,
            callback=job_callback
        )
        elapsed = time.time() - t0
        tracking_jobs[job_id]["status"] = "completed"
        tracking_jobs[job_id]["progress"] = 100
        tracking_jobs[job_id]["elapsed"] = round(elapsed, 2)
        tracking_jobs[job_id]["results"] = results
        clean_url = output_path.replace("\\", "/")
        tracking_jobs[job_id]["video_url"] = f"/media/{clean_url}"
    except Exception as e:
        tracking_jobs[job_id]["status"] = "failed"
        tracking_jobs[job_id]["error"] = str(e)
        if "logs" in tracking_jobs[job_id]:
            tracking_jobs[job_id]["logs"].append(f"[{time.strftime('%H:%M:%S')}] [ERROR] {str(e)}")


@app.post("/api/track")
async def start_tracking(
    background_tasks: BackgroundTasks,
    video_source: str = Form(...),
    custom_file: Optional[UploadFile] = File(None),
    conf_thresh: float = Form(0.25),
    max_age: int = Form(50),
    max_frames: int = Form(0),
    stride: int = Form(1)
):
    """Initiates tracking on selected preset or user-uploaded video."""
    job_id = str(uuid.uuid4())[:8]

    if video_source == "preset_real_crowd":
        input_path = "data/video_sequences/real_people_walking.mp4"
        output_path = f"outputs/tracked_videos/job_{job_id}_tracked.mp4"
    elif video_source == "preset_transit_gate":
        input_path = "data/video_sequences/transit_gate_dense.mp4"
        output_path = f"outputs/tracked_videos/job_{job_id}_tracked.mp4"
    elif video_source == "custom" and custom_file is not None:
        file_ext = os.path.splitext(custom_file.filename)[1] or ".mp4"
        input_path = f"data/uploads/{job_id}_input{file_ext}"
        with open(input_path, "wb") as f:
            shutil.copyfileobj(custom_file.file, f)
        output_path = f"outputs/tracked_videos/job_{job_id}_tracked.mp4"
    else:
        # Default fallback
        input_path = "data/video_sequences/real_people_walking.mp4"
        output_path = f"outputs/tracked_videos/job_{job_id}_tracked.mp4"

    tracking_jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "logs": [f"[{time.strftime('%H:%M:%S')}] [SYSTEM] Job {job_id} queued. Initializing video pipeline..."],
        "input": input_path,
        "output": output_path,
        "video_url": None,
        "elapsed": None,
        "error": None,
        "results": None
    }

    background_tasks.add_task(
        execute_tracking_job,
        job_id,
        input_path,
        output_path,
        conf_thresh,
        max_age,
        max_frames,
        stride
    )

    return {"job_id": job_id, "status": "queued"}


@app.get("/api/job/{job_id}")
def get_job_status(job_id: str):
    if job_id not in tracking_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return tracking_jobs[job_id]


# Mount built React frontend
if os.path.exists("frontend/dist"):
    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = os.path.join("frontend/dist", full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse("frontend/dist/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

