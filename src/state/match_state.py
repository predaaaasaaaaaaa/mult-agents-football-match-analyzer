"""
match_state.py — Shared LangGraph State
=========================================
This TypedDict defines the "shared clipboard" that flows through
the entire LangGraph pipeline.

WHO WRITES WHAT:
    Vision node     → tracking_data
    Interpolation   → enriched_data (ball interpolated)
    Teams node      → enriched_data (with team labels)
    Events node     → possession_log, passes, turnovers, tackles, interceptions
    Analytics node  → player_stats, team_stats
    Reporting node  → report
"""

from typing import TypedDict, Optional


class MatchAnalysisState(TypedDict):
    # --- Input ---
    video_path: str
    max_frames: Optional[int]

    # --- Vision Agent output ---
    tracking_data: list[dict]

    # --- Interpolation + Team Classification output ---
    enriched_data: list[dict]

    # --- Events Agent output ---
    possession_log: list[dict]
    passes: list[dict]
    turnovers: list[dict]
    tackles: list[dict]
    interceptions: list[dict]

    # --- Analytics Agent output ---
    player_stats: dict
    team_stats: dict

    # --- Reporting Agent output ---
    report: str

    # --- Pipeline metadata ---
    status: str
    errors: list[str]
