"""
High-End Video Annotator and Heads-Up Display (HUD) for MOT.
Renders persistent ID badges, trajectory trails, occlusion alert vectors,
and glassmorphism dashboard overlay.
"""

from typing import List, Tuple, Dict
import cv2
import numpy as np

from core_tracking.track_state import Track, TrackStatus


class VideoAnnotator:
    """
    Renders professional tracking annotations onto video frames.
    """

    def __init__(self):
        # High-contrast, aesthetically pleasing color palette
        self.palette = [
            (235, 87, 87),     # Coral Red
            (46, 204, 113),    # Emerald Green
            (52, 152, 219),    # Cerulean Blue
            (241, 196, 15),    # Golden Yellow
            (155, 89, 182),    # Amethyst Purple
            (230, 126, 34),    # Carrot Orange
            (26, 188, 156),    # Turquoise
            (243, 104, 224),   # Neon Pink
            (72, 219, 251),    # Sky Blue
            (255, 159, 243),   # Blossom Pink
            (84, 160, 255),    # Pastel Blue
            (95, 39, 205),     # Royal Violet
        ]

    def _get_color(self, track_id: int) -> Tuple[int, int, int]:
        """Map track_id deterministically to a BGR color tuple."""
        idx = (track_id - 1) % len(self.palette)
        # Palette is stored as RGB, convert to BGR for OpenCV
        r, g, b = self.palette[idx]
        return (b, g, r)

    def draw_glass_hud(
        self,
        canvas: np.ndarray,
        stats: Dict[str, int],
        fps: float,
        title: str = "OCCLUSION-AWARE MOT · IDENTITY-PRESERVING TRACKER"
    ):
        """Draw semi-transparent glassmorphism Heads-Up Display in top-left."""
        hud_w, hud_h = 360, 115
        hud_x, hud_y = 15, 15

        # Create overlay with alpha blending
        overlay = canvas.copy()
        cv2.rectangle(
            overlay,
            (hud_x, hud_y),
            (hud_x + hud_w, hud_y + hud_h),
            (25, 30, 40),
            -1
        )
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
        cv2.rectangle(
            canvas,
            (hud_x, hud_y),
            (hud_x + hud_w, hud_y + hud_h),
            (60, 160, 240),
            1
        )

        # Title
        cv2.putText(
            canvas,
            title,
            (hud_x + 12, hud_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (240, 240, 240),
            1,
            cv2.LINE_AA
        )

        # Status Metrics
        frame_text = f"Frame: {stats.get('frame', 0)}   |   FPS: {fps:.1f}"
        cv2.putText(
            canvas,
            frame_text,
            (hud_x + 12, hud_y + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 220, 240),
            1,
            cv2.LINE_AA
        )

        active_text = f"Active: {stats.get('active_tracks', 0)}   |   Occluded: {stats.get('occluded_tracks', 0)}"
        cv2.putText(
            canvas,
            active_text,
            (hud_x + 12, hud_y + 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 200, 100),
            1,
            cv2.LINE_AA
        )

        reid_text = f"Recovered Through Occlusion: {stats.get('cumulative_reassociations', 0)}"
        cv2.putText(
            canvas,
            reid_text,
            (hud_x + 12, hud_y + 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (46, 204, 113),
            1,
            cv2.LINE_AA
        )

    def draw_tracks(
        self,
        canvas: np.ndarray,
        tracks: List[Track],
        draw_trails: bool = True
    ):
        """Draw bounding boxes, ID tags, status badges, and trajectory trails."""
        h_img, w_img = canvas.shape[:2]

        for track in tracks:
            color = self._get_color(track.track_id)
            box = track.to_xyxy()
            x1, y1, x2, y2 = [int(v) for v in box]

            # 1. Trajectory motion trail
            if draw_trails and len(track.trajectory) > 1:
                points = list(track.trajectory)
                for i in range(1, len(points)):
                    alpha = float(i) / len(points)
                    thickness = max(1, int(3 * alpha))
                    pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
                    pt2 = (int(points[i][0]), int(points[i][1]))
                    cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

            # 2. Bounding Box
            box_thickness = 3 if track.is_occluded else 2
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thickness)

            # 3. Corner accents
            line_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
            cv2.line(canvas, (x1, y1), (x1 + line_len, y1), (255, 255, 255), 2)
            cv2.line(canvas, (x1, y1), (x1, y1 + line_len), (255, 255, 255), 2)
            cv2.line(canvas, (x2, y2), (x2 - line_len, y2), (255, 255, 255), 2)
            cv2.line(canvas, (x2, y2), (x2, y2 - line_len), (255, 255, 255), 2)

            # 4. Status Badge
            if track.reassociated_count > 0:
                badge = f"ID: {track.track_id:02d} [RECOVERED]"
                badge_bg = (46, 204, 113)
            elif track.is_occluded:
                badge = f"ID: {track.track_id:02d} [OCCLUDED]"
                badge_bg = (230, 126, 34)
            else:
                badge = f"ID: {track.track_id:02d}"
                badge_bg = color

            (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tag_y1 = max(0, y1 - th - 8)
            tag_y2 = y1
            cv2.rectangle(canvas, (x1, tag_y1), (x1 + tw + 10, tag_y2), badge_bg, -1)
            cv2.putText(
                canvas,
                badge,
                (x1 + 5, tag_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        # 5. Draw mutual occlusion alert lines between overlapping pairs
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                t1, t2 = tracks[i], tracks[j]
                if t1.is_occluded and t2.is_occluded:
                    c1 = (int(t1.mean[0]), int(t1.mean[1]))
                    c2 = (int(t2.mean[0]), int(t2.mean[1]))
                    dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
                    if dist < 120:  # In close mutual proximity
                        cv2.line(canvas, c1, c2, (0, 165, 255), 2, cv2.LINE_AA)
                        mid_pt = ((c1[0] + c2[0]) // 2, (c1[1] + c2[1]) // 2)
                        cv2.circle(canvas, mid_pt, 4, (0, 0, 255), -1)
