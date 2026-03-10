"""
run_pipeline.py
Runs the full pipeline: Vision → Ball Interpolation → Save JSON → Events
Loads from cached JSON if available (skip YOLO re-run).
"""

import json
import os
from src.agents.vision.vision_agent import VisionAgent
from src.agents.events.events_agent import EventsAgent
from src.utils.ball_interpolation import BallInterpolator


def main():
    video_path = "data/match_clip.mp4"
    output_path = "data/tracking_data.json"

    # Steps 1-3: Vision + Interpolation + Save (skip if JSON exists)
    if os.path.exists(output_path):
        print("=" * 50)
        print("LOADING CACHED TRACKING DATA")
        print("=" * 50)
        with open(output_path, "r") as f:
            enriched_data = json.load(f)
        print(f"Loaded {len(enriched_data)} detections from {output_path}")
    else:
        # Step 1: Run Vision Agent
        print("=" * 50)
        print("STEP 1: Vision Agent (YOLO + ByteTrack)")
        print("=" * 50)
        vision = VisionAgent(model_path="yolov8s.pt")
        tracking_data = vision.process(video_path, max_frames=500)

        # Step 2: Interpolate missing ball positions
        print("\n" + "=" * 50)
        print("STEP 2: Ball Interpolation")
        print("=" * 50)
        interpolator = BallInterpolator(max_gap=100)
        enriched_data = interpolator.interpolate(tracking_data, total_frames=500)

        # Step 3: Save to JSON
        print("\n" + "=" * 50)
        print("STEP 3: Saving tracking data")
        print("=" * 50)
        with open(output_path, "w") as f:
            json.dump(enriched_data, f)
        print(f"Saved {len(enriched_data)} detections to {output_path}")

    # Step 4: Events Agent — possession detection + changes
    print("\n" + "=" * 50)
    print("STEP 4: Events Agent (Possession Changes)")
    print("=" * 50)
    events = EventsAgent(possession_radius=80, change_threshold=3)
    possession_log = events.detect_possession(enriched_data)
    changes = events.detect_possession_changes(possession_log)

    # Print first 10 changes to eyeball them
    print(f"\nFirst 10 possession changes:")
    for c in changes[:10]:
        print(f"  Frame {c['frame']}: track_{c['from_track_id']} → track_{c['to_track_id']}")


if __name__ == "__main__":
    main()