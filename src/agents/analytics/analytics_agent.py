"""
Analytics Agent
Aggregates detected events into per-player and per-team statistics.
Takes output from EventsAgent → produces match stats summary.
Computes physical stats: distance, speed, sprints, heatmap positions.
"""

import math


class AnalyticsAgent:
    def __init__(self, fps=25):
        """
        Args:
            fps: Video frame rate. Needed to convert frame-to-frame
                 pixel movement into speed (pixels/second).
        """
        self.fps = fps
        self.player_stats = {}
        self.team_stats = {"A": self._empty_team_stats(), "B": self._empty_team_stats()}

    def _empty_player_stats(self, track_id, team):
        return {
            "track_id": track_id,
            "team": team,
            "passes_made": 0,
            "passes_received": 0,
            "turnovers_lost": 0,
            "turnovers_won": 0,
            "tackles_made": 0,
            "tackles_suffered": 0,
            "interceptions_made": 0,
            "interceptions_suffered": 0,
            "possession_frames": 0,
            # Physical stats
            "distance_px": 0.0,
            "top_speed_px_s": 0.0,
            "sprint_count": 0,
            "positions": [],  # List of (x, y) for heatmap
        }

    def _empty_team_stats(self):
        return {
            "total_passes": 0,
            "total_turnovers_lost": 0,
            "total_turnovers_won": 0,
            "total_tackles": 0,
            "total_interceptions": 0,
            "possession_frames": 0,
            # Physical stats (aggregated)
            "avg_distance_px": 0.0,
            "avg_top_speed_px_s": 0.0,
            "total_sprints": 0,
        }

    def _get_player(self, track_id, team):
        """Get or create player stats entry."""
        if track_id not in self.player_stats:
            self.player_stats[track_id] = self._empty_player_stats(track_id, team)
        return self.player_stats[track_id]

    def compute_physical_stats(self, tracking_data, sprint_threshold=150):
        """
        Compute distance, speed, sprints from raw tracking positions.

        For each player track_id:
        - Distance: sum of frame-to-frame pixel movement (feet position)
        - Top speed: max frame-to-frame distance × FPS (pixels/second)
        - Sprints: number of times speed exceeds sprint_threshold px/s

        Args:
            tracking_data: Full tracking data with team labels
            sprint_threshold: Speed in px/s above which counts as a sprint.
                              150 px/s is a starting point — we'll tune it.
        """
        # Group player detections by track_id, sorted by frame
        tracks = {}
        for d in tracking_data:
            if d["type"] == "player":
                tid = d["track_id"]
                if tid not in tracks:
                    tracks[tid] = []
                tracks[tid].append(d)

        # Sort each track by frame number
        for tid in tracks:
            tracks[tid].sort(key=lambda d: d["frame"])

        for tid, detections in tracks.items():
            team = detections[0].get("team")
            player = self._get_player(tid, team)

            total_dist = 0.0
            max_speed = 0.0
            sprint_count = 0
            in_sprint = False

            for i in range(len(detections)):
                d = detections[i]
                # Feet position = bottom-center of bbox
                x = d["x"] + d["w"] / 2
                y = d["y"] + d["h"]

                # Store position for heatmap
                player["positions"].append((x, y))

                if i == 0:
                    continue

                prev = detections[i - 1]
                prev_x = prev["x"] + prev["w"] / 2
                prev_y = prev["y"] + prev["h"]

                # Frame gap (might not be consecutive if player disappeared)
                frame_gap = d["frame"] - prev["frame"]
                if frame_gap > 5:
                    # Too big a gap — player probably left/re-entered frame
                    in_sprint = False
                    continue

                # Distance between frames
                dist = math.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                total_dist += dist

                # Speed = distance per second
                speed = (dist / frame_gap) * self.fps

                if speed > max_speed:
                    max_speed = speed

                # Sprint detection
                if speed >= sprint_threshold:
                    if not in_sprint:
                        sprint_count += 1
                        in_sprint = True
                else:
                    in_sprint = False

            player["distance_px"] = round(total_dist, 1)
            player["top_speed_px_s"] = round(max_speed, 1)
            player["sprint_count"] = sprint_count

        # Aggregate team physical stats
        for team_name in ["A", "B"]:
            team_players = [
                p for p in self.player_stats.values()
                if p["team"] == team_name and p["distance_px"] > 0
            ]
            if team_players:
                self.team_stats[team_name]["avg_distance_px"] = round(
                    sum(p["distance_px"] for p in team_players) / len(team_players), 1
                )
                self.team_stats[team_name]["avg_top_speed_px_s"] = round(
                    max(p["top_speed_px_s"] for p in team_players), 1
                )
                self.team_stats[team_name]["total_sprints"] = sum(
                    p["sprint_count"] for p in team_players
                )

        total_players = sum(
            1 for p in self.player_stats.values() if p["distance_px"] > 0
        )
        print(f"[AnalyticsAgent] Physical stats computed for {total_players} players")

    def process(self, possession_log, passes, tackles, interceptions):
        """
        Aggregate all events into stats.

        Args:
            possession_log: Frame-by-frame possession data
            passes: List of pass events
            tackles: List of tackle events
            interceptions: List of interception events
        """
        # Possession frames per player/team
        for entry in possession_log:
            tid = entry["track_id"]
            team = entry.get("team")
            if team:
                player = self._get_player(tid, team)
                player["possession_frames"] += 1
                self.team_stats[team]["possession_frames"] += 1

        # Passes
        for p in passes:
            from_team = p["from_team"]
            to_team = p["to_team"]

            passer = self._get_player(p["from_track_id"], from_team)
            passer["passes_made"] += 1

            receiver = self._get_player(p["to_track_id"], to_team)
            receiver["passes_received"] += 1

            if from_team:
                self.team_stats[from_team]["total_passes"] += 1

        # Tackles
        for t in tackles:
            loser = self._get_player(t["from_track_id"], t["from_team"])
            loser["tackles_suffered"] += 1
            loser["turnovers_lost"] += 1

            winner = self._get_player(t["to_track_id"], t["to_team"])
            winner["tackles_made"] += 1
            winner["turnovers_won"] += 1

            if t["from_team"]:
                self.team_stats[t["from_team"]]["total_turnovers_lost"] += 1
            if t["to_team"]:
                self.team_stats[t["to_team"]]["total_tackles"] += 1
                self.team_stats[t["to_team"]]["total_turnovers_won"] += 1

        # Interceptions
        for i in interceptions:
            loser = self._get_player(i["from_track_id"], i["from_team"])
            loser["interceptions_suffered"] += 1
            loser["turnovers_lost"] += 1

            winner = self._get_player(i["to_track_id"], i["to_team"])
            winner["interceptions_made"] += 1
            winner["turnovers_won"] += 1

            if i["from_team"]:
                self.team_stats[i["from_team"]]["total_turnovers_lost"] += 1
            if i["to_team"]:
                self.team_stats[i["to_team"]]["total_interceptions"] += 1
                self.team_stats[i["to_team"]]["total_turnovers_won"] += 1

        self._print_summary()

    def _print_summary(self):
        """Print match summary."""
        total_poss = sum(t["possession_frames"] for t in self.team_stats.values())

        print("\n" + "=" * 50)
        print("MATCH STATISTICS")
        print("=" * 50)

        # Team stats
        for team_name in ["A", "B"]:
            ts = self.team_stats[team_name]
            poss_pct = (ts["possession_frames"] / total_poss * 100) if total_poss > 0 else 0

            print(f"\n--- Team {team_name} ---")
            print(f"  Possession: {poss_pct:.1f}%")
            print(f"  Passes: {ts['total_passes']}")
            print(f"  Turnovers lost: {ts['total_turnovers_lost']}")
            print(f"  Turnovers won: {ts['total_turnovers_won']}")
            print(f"  Tackles won: {ts['total_tackles']}")
            print(f"  Interceptions: {ts['total_interceptions']}")
            print(f"  Avg distance (px): {ts['avg_distance_px']}")
            print(f"  Max speed (px/s): {ts['avg_top_speed_px_s']}")
            print(f"  Total sprints: {ts['total_sprints']}")

        # Top players by physical output
        print(f"\n--- Top Players (by distance covered) ---")
        sorted_physical = sorted(
            self.player_stats.values(),
            key=lambda p: p["distance_px"],
            reverse=True,
        )

        for p in sorted_physical[:10]:
            print(
                f"  Track #{p['track_id']} (Team {p['team']}): "
                f"dist={p['distance_px']}px, "
                f"top_speed={p['top_speed_px_s']}px/s, "
                f"sprints={p['sprint_count']}, "
                f"{p['passes_made']} passes, "
                f"{p['possession_frames']} frames on ball"
            )