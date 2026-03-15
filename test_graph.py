"""
Quick test — runs the LangGraph pipeline on cached data.
If tracking JSON exists, vision node will still run YOLO (no caching yet).
So for this test, we'll run on just 100 frames to keep it fast.
"""

from dotenv import load_dotenv
from src.graph.match_graph import build_match_graph

load_dotenv()

# Build the graph
graph = build_match_graph()

# Define initial state — every key in MatchAnalysisState needs a starting value
initial_state = {
    "video_path": "data/match_clip.mp4",
    "max_frames": 100,  # Small test — just 100 frames
    "tracking_data": [],
    "enriched_data": [],
    "possession_log": [],
    "passes": [],
    "turnovers": [],
    "tackles": [],
    "interceptions": [],
    "player_stats": {},
    "team_stats": {},
    "report": "",
    "status": "starting",
    "errors": [],
}

# Run the full pipeline
print("\n🚀 Running LangGraph pipeline...\n")
result = graph.invoke(initial_state)

# Check results
print("\n" + "=" * 50)
print("PIPELINE COMPLETE")
print("=" * 50)
print(f"Status: {result['status']}")
print(f"Errors: {result['errors']}")
print(f"Tracking detections: {len(result['tracking_data'])}")
print(f"Enriched detections: {len(result['enriched_data'])}")
print(f"Passes: {len(result['passes'])}")
print(f"Tackles: {len(result['tackles'])}")
print(f"Interceptions: {len(result['interceptions'])}")
print(f"Players tracked: {len(result['player_stats'])}")
print(f"\nReport preview: {result['report'][:200]}...")