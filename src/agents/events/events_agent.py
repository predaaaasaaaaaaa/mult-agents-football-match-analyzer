"""
Events Agent
Detects football events from tracking data.
Step 1: Ball possession (which player has the ball each frame).
Step 2: Possession changes with debouncing.
Step 3: Pass detection (same team) vs turnover (different team).
"""

import math


class EventsAgent:
    def __init__(self, possession_radius=80, change_threshold=3):
        """
        Args:
            possession_radius: Maximum pixel distance between ball and player's
                               feet (bottom-center of bbox) to count as possession.
            change_threshold: Number of consecutive frames a new player must be
                              closest to the ball before we count it as a real
                              possession change.
        """
        self.possession_radius = possession_radius
        self.change_threshold = change_threshold

    def detect_possession(self, tracking_data):
        """
        For each frame with a ball, find the closest player.

        Returns:
            List of possession events:
            [{"frame": 7, "track_id": 3, "distance": 25.4}, ...]
        """
        frames = {}
        for d in tracking_data:
            f = d["frame"]
            if f not in frames:
                frames[f] = {"players": [], "ball": None}

            if d["type"] == "player":
                frames[f]["players"].append(d)
            elif d["type"] == "ball":
                frames[f]["ball"] = d

        possession_log = []

        for frame_num in sorted(frames.keys()):
            frame_data = frames[frame_num]
            ball = frame_data["ball"]
            players = frame_data["players"]

            if ball is None:
                continue

            ball_cx = ball["x"] + ball["w"] / 2
            ball_cy = ball["y"] + ball["h"] / 2

            closest_id = None
            closest_dist = float("inf")
            closest_team = None

            for player in players:
                feet_x = player["x"] + player["w"] / 2
                feet_y = player["y"] + player["h"]

                dist = math.sqrt((ball_cx - feet_x) ** 2 + (ball_cy - feet_y) ** 2)

                if dist < closest_dist:
                    closest_dist = dist
                    closest_id = player["track_id"]
                    closest_team = player.get("team")

            if closest_dist <= self.possession_radius:
                possession_log.append(
                    {
                        "frame": frame_num,
                        "track_id": closest_id,
                        "team": closest_team,
                        "distance": round(closest_dist, 1),
                    }
                )

        print(f"[EventsAgent] Possession detected on {len(possession_log)} frames")
        return possession_log

    def detect_possession_changes(self, possession_log):
        """
        Detect when the ball switches between players, with debouncing.

        Returns:
            List of possession change events:
            [{"frame": 42, "from_track_id": 7, "to_track_id": 12,
              "from_team": "A", "to_team": "B"}, ...]
        """
        if len(possession_log) < 2:
            return []

        changes = []

        current_possessor = possession_log[0]["track_id"]
        current_team = possession_log[0].get("team")

        candidate_id = None
        candidate_team = None
        candidate_count = 0
        candidate_start_frame = None

        for entry in possession_log[1:]:
            track_id = entry["track_id"]
            team = entry.get("team")
            frame = entry["frame"]

            if track_id == current_possessor:
                candidate_id = None
                candidate_count = 0
                candidate_start_frame = None
                candidate_team = None

            elif track_id == candidate_id:
                candidate_count += 1

                if candidate_count >= self.change_threshold:
                    changes.append(
                        {
                            "frame": candidate_start_frame,
                            "from_track_id": current_possessor,
                            "to_track_id": candidate_id,
                            "from_team": current_team,
                            "to_team": candidate_team,
                        }
                    )
                    current_possessor = candidate_id
                    current_team = candidate_team
                    candidate_id = None
                    candidate_count = 0
                    candidate_start_frame = None
                    candidate_team = None

            else:
                candidate_id = track_id
                candidate_team = team
                candidate_count = 1
                candidate_start_frame = frame

        print(f"[EventsAgent] Possession changes detected: {len(changes)}")
        return changes

    def detect_passes(self, possession_changes):
        """
        Classify possession changes as passes or turnovers based on team.

        Same team → successful pass
        Different team → turnover (tackle, interception, or failed pass)

        Returns:
            (passes, turnovers) — two separate lists
        """
        passes = []
        turnovers = []

        for change in possession_changes:
            event = {
                "frame": change["frame"],
                "from_track_id": change["from_track_id"],
                "to_track_id": change["to_track_id"],
                "from_team": change["from_team"],
                "to_team": change["to_team"],
            }

            if change["from_team"] == change["to_team"] and change["from_team"] is not None:
                event["type"] = "pass"
                passes.append(event)
            else:
                event["type"] = "turnover"
                turnovers.append(event)

        print(f"[EventsAgent] Passes: {len(passes)}, Turnovers: {len(turnovers)}")
        return passes, turnovers