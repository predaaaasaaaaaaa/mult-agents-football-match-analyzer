"""
Vision Agent
Runs YOLO + ByteTrack on match video.
Returns structured tracking data for downstream agents.
"""

from ultralytics import YOLO
import supervision as sv


class VisionAgent:
    def __init__(self, model_path="yolov8s.pt"):
        """Load YOLO model and initialize ByteTrack."""
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.5,
            lost_track_buffer=60,
            minimum_matching_threshold=0.8,
            frame_rate=25,
        )

    def process(self, video_path, max_frames=None):
        """
        Process a match video and return tracking data.

        Args:
            video_path: Path to the match video file.
            max_frames: Optional limit on frames to process (None = full video).

        Returns:
            List of detections, one dict per tracked person per frame:
            [
                {"frame": 1, "track_id": 3, "x": 120, "y": 85, "w": 40, "h": 90, "confidence": 0.82},
                {"frame": 1, "track_id": 7, "x": 450, "y": 200, "w": 35, "h": 80, "confidence": 0.75},
                ...
            ]
        """
        results = self.model(
            source=video_path,
            stream=True,
            conf=0.3,
            classes=[0, 32],  # person + sports ball
            device=0,
        )

        tracking_data = []
        frame_count = 0

        for result in results:
            frame_count += 1

            # Convert YOLO → supervision detections
            detections = sv.Detections.from_ultralytics(result)

            # Separate persons and balls
            person_mask = detections.class_id == 0
            person_detections = detections[person_mask]

            ball_mask = detections.class_id == 32
            ball_detections = detections[ball_mask]

            # Track persons with ByteTrack
            tracked = self.tracker.update_with_detections(person_detections)

            # Store each tracked person's data
            if tracked.tracker_id is not None:
                for i, tid in enumerate(tracked.tracker_id):
                    x1, y1, x2, y2 = tracked.xyxy[i]
                    tracking_data.append(
                        {
                            "frame": frame_count,
                            "track_id": int(tid),
                            "type": "player",
                            "x": int(x1),
                            "y": int(y1),
                            "w": int(x2 - x1),
                            "h": int(y2 - y1),
                            "confidence": round(float(tracked.confidence[i]), 3),
                        }
                    )

            # Store ball detections (no tracking, just position)
            for i in range(len(ball_detections)):
                x1, y1, x2, y2 = ball_detections.xyxy[i]
                tracking_data.append(
                    {
                        "frame": frame_count,
                        "track_id": None,
                        "type": "ball",
                        "x": int(x1),
                        "y": int(y1),
                        "w": int(x2 - x1),
                        "h": int(y2 - y1),
                        "confidence": round(float(ball_detections.confidence[i]), 3),
                    }
                )

            # Progress logging
            if frame_count % 100 == 0:
                num_tracked = (
                    len(tracked.tracker_id) if tracked.tracker_id is not None else 0
                )
                print(
                    f"[VisionAgent] Frame {frame_count}: {num_tracked} players tracked"
                )

            # Stop if max_frames is set
            if max_frames and frame_count >= max_frames:
                break

        print(
            f"[VisionAgent] Done — {frame_count} frames, {len(tracking_data)} detections"
        )
        return tracking_data
