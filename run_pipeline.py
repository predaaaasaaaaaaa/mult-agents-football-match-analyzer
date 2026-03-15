"""
run_pipeline.py
Runs the full pipeline: Vision → Interpolation → Teams → Events → Analytics → Report
"""

import json
import os
from dotenv import load_dotenv
from src.agents.vision.vision_agent import VisionAgent
from src.agents.events.events_agent import EventsAgent
from src.utils.ball_interpolation import BallInterpolator
from src.utils.team_classifier import TeamClassifier

load_dotenv()


def main():
    video_path = "data/match_clip.mp4"
    output_path = "data/tracking_data.json"
    enriched_path = "data/tracking_data_teams.json"

    # Steps 1-3: Vision + Interpolation + Save (skip if JSON exists)
    if os.path.exists(output_path):
        print("=" * 50)
        print("LOADING CACHED TRACKING DATA")
        print("=" * 50)
        with open(output_path, "r") as f:
            enriched_data = json.load(f)
        print(f"Loaded {len(enriched_data)} detections from {output_path}")
    else:
        print("=" * 50)
        print("STEP 1: Vision Agent (YOLO + ByteTrack)")
        print("=" * 50)
        vision = VisionAgent(model_path="yolov8s.pt")
        tracking_data = vision.process(video_path, max_frames=None)

        print("\n" + "=" * 50)
        print("STEP 2: Ball Interpolation")
        print("=" * 50)
        interpolator = BallInterpolator(max_gap=100)
        enriched_data = interpolator.interpolate(tracking_data, total_frames=5230)

        print("\n" + "=" * 50)
        print("STEP 3: Saving tracking data")
        print("=" * 50)
        with open(output_path, "w") as f:
            json.dump(enriched_data, f)
        print(f"Saved {len(enriched_data)} detections to {output_path}")

    # Step 4: Team Classification
    if os.path.exists(enriched_path):
        print("\n" + "=" * 50)
        print("LOADING CACHED TEAM DATA")
        print("=" * 50)
        with open(enriched_path, "r") as f:
            enriched_data = json.load(f)
        print(f"Loaded {len(enriched_data)} detections with team labels")
    else:
        print("\n" + "=" * 50)
        print("STEP 4: Team Classification (Jersey Colors)")
        print("=" * 50)
        classifier = TeamClassifier()
        classifier.fit(video_path, enriched_data, sample_frames=100)
        enriched_data = classifier.predict(video_path, enriched_data)

        with open(enriched_path, "w") as f:
            json.dump(enriched_data, f)
        print(f"Saved team-labeled data to {enriched_path}")

    # Step 5: Events Agent — possession + changes + passes + tackles
    print("\n" + "=" * 50)
    print("STEP 5: Events Agent")
    print("=" * 50)
    events = EventsAgent(possession_radius=80, change_threshold=3)
    possession_log = events.detect_possession(enriched_data)
    changes = events.detect_possession_changes(possession_log)
    passes, turnovers = events.detect_passes(changes)
    tackles, interceptions = events.detect_tackles(turnovers, enriched_data)

    print(f"\n--- Results ---")
    print(f"Possession frames: {len(possession_log)}")
    print(f"Possession changes: {len(changes)}")
    print(f"Passes: {len(passes)}")
    print(f"Turnovers: {len(turnovers)}")
    print(f"  Tackles: {len(tackles)}")
    print(f"  Interceptions: {len(interceptions)}")

    for p in passes:
        print(
            f"  PASS  Frame {p['frame']}: track_{p['from_track_id']} → track_{p['to_track_id']} (Team {p['from_team']})"
        )
    for t in tackles:
        print(
            f"  TACKLE  Frame {t['frame']}: track_{t['from_track_id']} (Team {t['from_team']}) tackled by track_{t['to_track_id']} (Team {t['to_team']}) [{t['distance']}px]"
        )
    for i in interceptions:
        print(
            f"  INTERCEPT  Frame {i['frame']}: track_{i['from_track_id']} (Team {i['from_team']}) → track_{i['to_track_id']} (Team {i['to_team']})"
        )

    # Step 6: Analytics Agent — aggregate stats + physical data
    print("\n" + "=" * 50)
    print("STEP 6: Analytics Agent")
    print("=" * 50)
    from src.agents.analytics.analytics_agent import AnalyticsAgent
    analytics = AnalyticsAgent(fps=25)
    analytics.compute_physical_stats(enriched_data)
    analytics.process(possession_log, passes, tackles, interceptions)

    # Step 7: Reporting Agent — LLM match report
    print("\n" + "=" * 50)
    print("STEP 7: Reporting Agent (Groq LLM)")
    print("=" * 50)
    from src.agents.reporting.reporting_agent import ReportingAgent

    reporter = ReportingAgent()
    report = reporter.generate_report(analytics)


if __name__ == "__main__":
    main()
