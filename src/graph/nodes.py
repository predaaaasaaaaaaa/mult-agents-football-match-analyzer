"""
nodes.py — LangGraph Node Functions
=====================================
Each function wraps an existing agent and makes it LangGraph-compatible.

A LangGraph node is just a function that:
    - Takes the full state dict as input
    - Returns a dict with ONLY the keys it wants to update
    - LangGraph merges the returned keys into the state automatically
"""

import os
import json
from src.agents.vision.vision_agent import VisionAgent
from src.agents.events.events_agent import EventsAgent
from src.agents.analytics.analytics_agent import AnalyticsAgent
from src.agents.reporting.reporting_agent import ReportingAgent
from src.utils.ball_interpolation import BallInterpolator
from src.utils.team_classifier import TeamClassifier


# =============================================================================
# NODE 1: Vision (YOLO + ByteTrack)
# =============================================================================

def vision_node(state: dict) -> dict:
    """
    Run YOLOv8s + ByteTrack on the video.

    Reads:  video_path, max_frames
    Writes: tracking_data, status
    """
    print("=" * 50)
    print("NODE: Vision Agent (YOLO + ByteTrack)")
    print("=" * 50)

    try:
        vision = VisionAgent(model_path="yolov8s.pt")
        tracking_data = vision.process(
            state["video_path"],
            max_frames=state.get("max_frames"),
        )
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

    Reads:  tracking_data
    Writes: enriched_data, status
    """
    print("=" * 50)
    print("NODE: Ball Interpolation")
    print("=" * 50)

    try:
        interpolator = BallInterpolator(max_gap=100)

        # Get total frames from tracking data
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

    Reads:  video_path, enriched_data
    Writes: enriched_data (now with team labels), status
    """
    print("=" * 50)
    print("NODE: Team Classification")
    print("=" * 50)

    try:
        classifier = TeamClassifier()
        classifier.fit(state["video_path"], state["enriched_data"], sample_frames=100)
        labeled = classifier.predict(state["video_path"], state["enriched_data"])
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

    NOTE: We can't pass the AnalyticsAgent instance through LangGraph state
    (it's not serializable). So we reconstruct a minimal analytics object
    with the stats from state, which is all the ReportingAgent needs.
    """
    print("=" * 50)
    print("NODE: Reporting Agent (Groq LLM)")
    print("=" * 50)

    try:
        # Reconstruct a minimal analytics object for the reporter
        # ReportingAgent only reads .player_stats and .team_stats
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