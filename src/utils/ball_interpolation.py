"""
Ball Interpolation — Kalman Filter
Fills in missing ball positions between sparse YOLO detections.
Takes Vision Agent output → returns same data with interpolated ball positions added.
"""

import numpy as np


class BallInterpolator:
    def __init__(self, max_gap=100):
        """
        Args:
            max_gap: Maximum number of missing frames to interpolate across.
                     If the gap is bigger than this, we don't guess — too risky.
                     30 frames = ~1.2 seconds at 25fps.
        """
        self.max_gap = max_gap

    def interpolate(self, tracking_data, total_frames):
        """
        Take Vision Agent output and fill in missing ball positions.

        Args:
            tracking_data: List of dicts from VisionAgent.process()
            total_frames: Total number of frames processed

        Returns:
            Same list with interpolated ball detections added.
            Interpolated entries have confidence=0.0 to distinguish from real ones.
        """
        # Step 1: Extract real ball detections, keyed by frame
        ball_detections = {}
        for d in tracking_data:
            if d["type"] == "ball":
                # Use center point of bounding box
                cx = d["x"] + d["w"] / 2
                cy = d["y"] + d["h"] / 2
                ball_detections[d["frame"]] = {
                    "cx": cx,
                    "cy": cy,
                    "w": d["w"],
                    "h": d["h"],
                }

        if len(ball_detections) < 2:
            print("[BallInterpolator] Not enough ball detections to interpolate")
            return tracking_data

        # Step 2: Get sorted list of frames where ball was detected
        detected_frames = sorted(ball_detections.keys())
        print(f"[BallInterpolator] Real detections: {len(detected_frames)} frames")

        # Step 3: Interpolate between consecutive detections
        interpolated_count = 0
        new_entries = []

        for i in range(len(detected_frames) - 1):
            frame_a = detected_frames[i]
            frame_b = detected_frames[i + 1]
            gap = frame_b - frame_a

            # Skip if frames are adjacent (no gap) or gap is too big
            if gap <= 1 or gap > self.max_gap:
                continue

            # Get start and end positions
            a = ball_detections[frame_a]
            b = ball_detections[frame_b]

            # Linear interpolation for each missing frame
            for f in range(frame_a + 1, frame_b):
                # How far between a and b (0.0 to 1.0)
                t = (f - frame_a) / gap

                cx = a["cx"] + t * (b["cx"] - a["cx"])
                cy = a["cy"] + t * (b["cy"] - a["cy"])
                w = a["w"] + t * (b["w"] - a["w"])
                h = a["h"] + t * (b["h"] - a["h"])

                new_entries.append(
                    {
                        "frame": f,
                        "track_id": None,
                        "type": "ball",
                        "x": int(cx - w / 2),
                        "y": int(cy - h / 2),
                        "w": int(w),
                        "h": int(h),
                        "confidence": 0.0,  # Marks this as interpolated
                    }
                )
                interpolated_count += 1

        # Step 4: Merge interpolated entries into original data
        result = tracking_data + new_entries

        total_ball = len(detected_frames) + interpolated_count
        print(f"[BallInterpolator] Interpolated: {interpolated_count} frames")
        print(
            f"[BallInterpolator] Total ball coverage: {total_ball}/{total_frames} "
            f"({total_ball / total_frames * 100:.1f}%)"
        )

        return result
