"""
Runs the full pipeline: Vision → Ball Interpolation → Save JSON
"""

import json
from src.agents.vision.vision_agent import VisionAgent
from src.utils.ball_interpolation import BallInterpolator


def main():
    video_path = "data/match_clip.mp4"
    output_path = "data/tracking_data.json"

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


if __name__ == "__main__":
    main()