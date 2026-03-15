"""
nodes.py — LangGraph Node Functions
=====================================
Each function wraps an existing agent and makes it LangGraph-compatible.

A LangGraph node is a function that:
    - Takes the full state dict as input
    - Returns a dict with ONLY the keys it wants to update
    - LangGraph merges the returned keys into the state automatically

Caching:
    Vision, interpolation, and team classification results are cached
    per video in data/cache/. Cache filenames are derived from the
    video filename, so each video gets its own cache files.
    Example: match_clip.mp4 → data/cache/match_clip_tracking.json
"""

import os
import json
from pathlib import Path
from src.agents.vision.vision_agent import VisionAgent
from src.agents.events.events_agent import EventsAgent
from src.agents.analytics.analytics_agent import AnalyticsAgent
from src.agents.reporting.reporting_agent import ReportingAgent
from src.utils.ball_interpolation import BallInterpolator
from src.utils.team_classifier import TeamClassifier


# =============================================================================
# CACHE HELPER
# =============================================================================

def _cache_path(video_path: str, suffix: str) -> str:
    """
    Generate a cache file path based on the video filename.

    Example:
        _cache_path("data/match_clip.mp4", "tracking")
        → "data/cache/match_clip_tracking.json"
    """
    video_name = Path(video_path).stem  # "match_clip"
    cache_dir = "data/cache"
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{video_name}_{suffix}.json")


# =============================================================================
# NODE 1: Vision (YOLO + ByteTrack)
# =============================================================================

def vision_node(state: dict) -> dict:
    """
    Run YOLOv8s + ByteTrack on the video.
    Skips if cached tracking data exists for this video.

    Reads:  video_path, max_frames
    Writes: tracking_data, status
    """
    print("=" * 50)
    print("NODE: Vision Agent (YOLO + ByteTrack)")
    print("=" * 50)

    cache_file = _cache_path(state["video_path"], "tracking")

    # Check cache
    if os.path.exists(cache_file) and state.get("max_frames") is None:
        print(f"  [CACHE HIT] Loading from {cache_file}")
        with open(cache_file, "r") as f:
            tracking_data = json.load(f)
        print(f"  Loaded {len(tracking_data)} detections")
        return {
            "tracking_data": tracking_data,
            "status": "vision_done",
        }

    try:
        vision = VisionAgent(model_path="yolov8s.pt")
        tracking_data = vision.process(
            state["video_path"],
            max_frames=state.get("max_frames"),
        )

        # Save to cache (only for full runs, not limited frame tests)
        if state.get("max_frames") is None:
            with open(cache_file, "w") as f:
                json.dump(tracking_data, f)
            print(f"  [CACHED] Saved to {cache_file}")

        return {
            "tracking_data": tracking_data,
            "status": "vision_done",
        }
    except Exception as e:
        return {
            "tracking_data": [],
            "status": "error",
            "errors": [f"Vision failed: {str(e)}"],
        }


# =============================================================================
# NODE 2: Ball Interpolation
# =============================================================================

def interpolation_node(state: dict) -> dict:
    """
    Fill gaps in ball detection using linear interpolation.
    No separate cache — this is fast and always runs on tracking_data.

    Reads:  tracking_data
    Writes: enriched_data, status
    """
    print("=" * 50)
    print("NODE: Ball Interpolation")
    print("=" * 50)

    try:
        interpolator = BallInterpolator(max_gap=100)

        player_frames = [d["frame"] for d in state["tracking_data"] if d["type"] == "player"]
        total_frames = max(player_frames) + 1 if player_frames else 5230

        enriched = interpolator.interpolate(state["tracking_data"], total_frames=total_frames)
        return {
            "enriched_data": enriched,
            "status": "interpolation_done",
        }
    except Exception as e:
        return {
            "enriched_data": state["tracking_data"],
            "status": "error",
            "errors": [f"Interpolation failed: {str(e)}"],
        }


# =============================================================================
# NODE 3: Team Classification (KMeans jersey colors)
# =============================================================================

def teams_node(state: dict) -> dict:
    """
    Classify players into Team A / Team B using jersey color clustering.
    Skips if cached team-labeled data exists for this video.

    Reads:  video_path, enriched_data
    Writes: enriched_data (now with team labels), status
    """
    print("=" * 50)
    print("NODE: Team Classification")
    print("=" * 50)

    cache_file = _cache_path(state["video_path"], "teams")

    # Check cache
    if os.path.exists(cache_file) and state.get("max_frames") is None:
        print(f"  [CACHE HIT] Loading from {cache_file}")
        with open(cache_file, "r") as f:
            labeled = json.load(f)
        print(f"  Loaded {len(labeled)} detections with team labels")
        return {
            "enriched_data": labeled,
            "status": "teams_done",
        }

    try:
        classifier = TeamClassifier()
        classifier.fit(state["video_path"], state["enriched_data"], sample_frames=100)
        labeled = classifier.predict(state["video_path"], state["enriched_data"])

        # Save to cache (only for full runs)
        if state.get("max_frames") is None:
            with open(cache_file, "w") as f:
                json.dump(labeled, f)
            print(f"  [CACHED] Saved to {cache_file}")

        return {
            "enriched_data": labeled,
            "status": "teams_done",
        }
    except Exception as e:
        return {
            "status": "error",
            "errors": [f"Team classification failed: {str(e)}"],
        }


# =============================================================================
# NODE 4: Events Agent (passes, tackles, interceptions)
# =============================================================================

def events_node(state: dict) -> dict:
    """
    Detect game events from tracking + team data.

    Reads:  enriched_data
    Writes: possession_log, passes, turnovers, tackles, interceptions, status
    """
    print("=" * 50)
    print("NODE: Events Agent")
    print("=" * 50)

    try:
        events = EventsAgent(possession_radius=80, change_threshold=3)
        possession_log = events.detect_possession(state["enriched_data"])
        changes = events.detect_possession_changes(possession_log)
        passes, turnovers = events.detect_passes(changes)
        tackles, interceptions = events.detect_tackles(turnovers, state["enriched_data"])

        print(f"  Passes: {len(passes)}")
        print(f"  Tackles: {len(tackles)}")
        print(f"  Interceptions: {len(interceptions)}")

        return {
            "possession_log": possession_log,
            "passes": passes,
            "turnovers": turnovers,
            "tackles": tackles,
            "interceptions": interceptions,
            "status": "events_done",
        }
    except Exception as e:
        return {
            "possession_log": [],
            "passes": [],
            "turnovers": [],
            "tackles": [],
            "interceptions": [],
            "status": "error",
            "errors": [f"Events failed: {str(e)}"],
        }


# =============================================================================
# NODE 5: Analytics Agent (aggregate stats)
# =============================================================================

def analytics_node(state: dict) -> dict:
    """
    Aggregate events into per-player and per-team statistics.

    Reads:  enriched_data, possession_log, passes, tackles, interceptions
    Writes: player_stats, team_stats, status
    """
    print("=" * 50)
    print("NODE: Analytics Agent")
    print("=" * 50)

    try:
        analytics = AnalyticsAgent(fps=25)
        analytics.compute_physical_stats(state["enriched_data"])
        analytics.process(
            state["possession_log"],
            state["passes"],
            state["tackles"],
            state["interceptions"],
        )

        return {
            "player_stats": analytics.player_stats,
            "team_stats": analytics.team_stats,
            "status": "analytics_done",
        }
    except Exception as e:
        return {
            "player_stats": {},
            "team_stats": {},
            "status": "error",
            "errors": [f"Analytics failed: {str(e)}"],
        }


# =============================================================================
# NODE 6: Reporting Agent (Groq LLM)
# =============================================================================

def reporting_node(state: dict) -> dict:
    """
    Generate match report using Groq LLM.

    Reads:  player_stats, team_stats
    Writes: report, status

    The ReportingAgent expects an AnalyticsAgent instance, but LangGraph
    state only holds plain data (dicts, lists, strings). So a fresh
    AnalyticsAgent is created and populated with stats from state.
    """
    print("=" * 50)
    print("NODE: Reporting Agent (Groq LLM)")
    print("=" * 50)

    try:
        analytics = AnalyticsAgent(fps=25)
        analytics.player_stats = state["player_stats"]
        analytics.team_stats = state["team_stats"]

        reporter = ReportingAgent()
        report = reporter.generate_report(analytics)

        return {
            "report": report,
            "status": "completed",
        }
    except Exception as e:
        return {
            "report": f"Report generation failed: {str(e)}",
            "status": "error",
            "errors": [f"Reporting failed: {str(e)}"],
        }