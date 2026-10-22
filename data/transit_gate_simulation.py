"""
Transit Gate Crowd & Occlusion Video Generator.
Generates synthetic transit gate sequences with configurable crowd density,
controlled mutual occlusions, and precise ground truth annotations in MOTChallenge format.
"""

from typing import List, Dict, Tuple
import os
import cv2
import numpy as np


class SyntheticGatePerson:
    """Simulates an individual walking through a transit gate."""

    def __init__(
        self,
        person_id: int,
        start_pos: Tuple[float, float],
        velocity: Tuple[float, float],
        size: Tuple[int, int],
        color_rgb: Tuple[int, int, int],
        appearance_pattern: int = 0
    ):
        self.person_id = person_id
        self.x, self.y = float(start_pos[0]), float(start_pos[1])
        self.vx, self.vy = float(velocity[0]), float(velocity[1])
        self.w, self.h = size
        self.color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        self.pattern = appearance_pattern
        self.active = True

    def update(self, bounds: Tuple[int, int]):
        """Update position with boundary bounce or wraparound."""
        max_w, max_h = bounds
        self.x += self.vx
        self.y += self.vy

        # Rebound at horizontal edges with velocity reversal
        if self.x - self.w / 2 <= 50 or self.x + self.w / 2 >= max_w - 50:
            self.vx *= -1.0
            self.x = np.clip(self.x, 50 + self.w / 2, max_w - 50 - self.w / 2)

        if self.y - self.h / 2 <= 80 or self.y + self.h / 2 >= max_h - 50:
            self.vy *= -1.0
            self.y = np.clip(self.y, 80 + self.h / 2, max_h - 50 - self.h / 2)

    def get_bbox(self) -> np.ndarray:
        """Returns [x1, y1, x2, y2]."""
        x1 = self.x - self.w / 2.0
        y1 = self.y - self.h / 2.0
        x2 = self.x + self.w / 2.0
        y2 = self.y + self.h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def draw(self, canvas: np.ndarray):
        """Draw realistic pedestrian avatar with body, head, and clothing pattern."""
        x1, y1, x2, y2 = [int(v) for v in self.get_bbox()]
        w = x2 - x1
        h = y2 - y1

        # Torso / Body
        body_y1 = y1 + int(h * 0.25)
        cv2.rectangle(canvas, (x1, body_y1), (x2, y2), self.color_bgr, -1)
        cv2.rectangle(canvas, (x1, body_y1), (x2, y2), (40, 40, 40), 2)

        # Distinguishing clothing pattern / stripe
        if self.pattern == 1:
            stripe_y = body_y1 + int(h * 0.2)
            cv2.rectangle(canvas, (x1, stripe_y), (x2, stripe_y + int(h * 0.1)), (240, 240, 240), -1)
        elif self.pattern == 2:
            stripe_x = x1 + int(w * 0.4)
            cv2.rectangle(canvas, (stripe_x, body_y1), (stripe_x + int(w * 0.2), y2), (240, 240, 240), -1)

        # Head
        head_cx = int(self.x)
        head_cy = y1 + int(h * 0.12)
        head_radius = int(min(w, h * 0.25) * 0.4)
        cv2.circle(canvas, (head_cx, head_cy), head_radius, (200, 180, 160), -1)
        cv2.circle(canvas, (head_cx, head_cy), head_radius, (40, 40, 40), 1)


def generate_transit_gate_sequence(
    output_video_path: str,
    output_gt_path: str,
    num_frames: int = 300,
    num_people: int = 8,
    width: int = 800,
    height: int = 480,
    fps: int = 25,
    gate_pillar_occlusion: bool = True
):
    """
    Generate synthetic video sequence with crowded transit gates, mutual crossings,
    and a static gate pillar causing temporary full occlusion.
    Writes:
        - MP4 video file
        - MOT format Ground Truth text file: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, 1, 1, 1.0
    """
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_gt_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Pre-defined high-contrast clothing colors for consistent Re-ID testing
    colors = [
        (220, 50, 50),    # Red
        (50, 120, 220),   # Blue
        (50, 200, 80),    # Green
        (230, 180, 40),   # Yellow
        (160, 60, 200),   # Purple
        (240, 130, 40),   # Orange
        (60, 200, 200),   # Cyan
        (200, 80, 140),   # Pink
        (100, 100, 100),  # Grey
        (240, 240, 240),  # White
    ]

    # Initialize people moving along opposing crossing trajectories (creating guaranteed mutual occlusions)
    people: List[SyntheticGatePerson] = []
    for i in range(num_people):
        color = colors[i % len(colors)]
        pattern = i % 3

        # Alternate walking left-to-right and right-to-left
        if i % 2 == 0:
            start_x = 80.0 + (i * 45.0)
            vx = float(np.random.uniform(2.0, 3.5))
        else:
            start_x = float(width - 80.0 - (i * 45.0))
            vx = float(-np.random.uniform(2.0, 3.5))

        start_y = 180.0 + float(np.random.uniform(-40, 40))
        vy = float(np.random.uniform(-0.8, 0.8))

        w = int(np.random.uniform(50, 65))
        h = int(w * np.random.uniform(2.4, 2.8))

        people.append(
            SyntheticGatePerson(
                person_id=i + 1,
                start_pos=(start_x, start_y),
                velocity=(vx, vy),
                size=(w, h),
                color_rgb=color,
                appearance_pattern=pattern
            )
        )

    gt_records = []

    # Transit gate turnstile pillars (causing physical occlusion)
    pillar_x1, pillar_x2 = int(width * 0.45), int(width * 0.55)
    pillar_y1, pillar_y2 = 100, height - 80

    for frame_idx in range(1, num_frames + 1):
        # Background: Modern transit station floor
        frame = np.full((height, width, 3), 235, dtype=np.uint8)

        # Floor grid lines (perspective cues)
        for gy in range(80, height, 50):
            cv2.line(frame, (0, gy), (width, gy), (215, 215, 215), 1)
        for gx in range(0, width, 80):
            cv2.line(frame, (gx, 80), (gx, height), (220, 220, 220), 1)

        # Header bar for Transit Gate camera
        cv2.rectangle(frame, (0, 0), (width, 50), (35, 45, 60), -1)
        cv2.putText(
            frame,
            "ZED DIGITAL TRANSIT GATE - CAMERA 04 [BENCHMARK FEED]",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # Sort people by y position for natural depth rendering (back to front)
        sorted_people = sorted(people, key=lambda p: p.y)

        for person in sorted_people:
            person.update((width, height))
            bbox = person.get_bbox()

            # Record Ground Truth in MOT format
            # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, 1, 1, 1.0
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            gt_records.append(f"{frame_idx},{person.person_id},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},1,1,1.0\n")

            # Draw person
            person.draw(frame)

        # If enabled, draw transit gate turnstile pillars in foreground
        if gate_pillar_occlusion:
            # Pillar 1
            cv2.rectangle(frame, (pillar_x1, pillar_y1), (pillar_x2, pillar_y2), (70, 80, 95), -1)
            cv2.rectangle(frame, (pillar_x1, pillar_y1), (pillar_x2, pillar_y2), (30, 40, 50), 3)
            # Glass barrier badge
            cv2.putText(
                frame,
                "GATE TURNSTILE",
                (pillar_x1 + 6, int((pillar_y1 + pillar_y2) / 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (220, 220, 220),
                1
            )

        out.write(frame)

    out.release()

    # Save Ground Truth annotations
    with open(output_gt_path, "w") as f:
        f.writelines(gt_records)

    print(f"Generated {num_frames} frames to {output_video_path}")
    print(f"Saved {len(gt_records)} ground truth records to {output_gt_path}")


if __name__ == "__main__":
    generate_transit_gate_sequence(
        output_video_path="data/video_sequences/transit_gate_dense.mp4",
        output_gt_path="data/ground_truth/transit_gate_dense_gt.txt",
        num_frames=300,
        num_people=10,
        width=800,
        height=480,
        fps=25,
        gate_pillar_occlusion=True
    )
