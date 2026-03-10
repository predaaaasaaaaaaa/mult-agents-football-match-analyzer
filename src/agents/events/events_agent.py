"""
Events Agent
Detects football events from tracking data.
Step 1: Ball possession (which player has the ball each frame).
Step 2: Possession changes with debouncing (filters out noise).
"""

import math


class EventsAgent:
    def __init__(self, possession_radius=50, change_threshold=3):
        """
        Args:
            possession_radius: Maximum pixel distance between ball and player's
                               feet (bottom-center of bbox) to count as possession.
                               50px is a starting point — we'll tune it.
            change_threshold: Number of consecutive frames a new player must be
                              closest to the ball before we count it as a real
                              possession change. Filters out 1-frame noise.
        """
        self.possession_radius = possession_radius
        self.change_threshold = change_threshold

    def detect_possession(self, tracking_data):
        """
        For each frame with a ball, find the closest player.

        Args:
            tracking_data: Enriched data from VisionAgent + BallInterpolator

        Returns:
            List of possession events:
            [
                {"frame": 7, "track_id": 3, "distance": 25.4},
                {"frame": 8, "track_id": 3, "distance": 22.1},
                {"frame": 9, "track_id": 5, "distance": 31.7},
                ...
            ]
        """
        # Group all detections by frame
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

            # Skip frames with no ball
            if ball is None:
                continue

            # Ball center
            ball_cx = ball["x"] + ball["w"] / 2
            ball_cy = ball["y"] + ball["h"] / 2

            # Find closest player (using bottom-center = feet position)
            closest_id = None
            closest_dist = float("inf")

            for player in players:
                # Bottom-center of player bbox = feet
                feet_x = player["x"] + player["w"] / 2
                feet_y = player["y"] + player["h"]  # Bottom of box

                dist = math.sqrt((ball_cx - feet_x) ** 2 + (ball_cy - feet_y) ** 2)

                if dist < closest_dist:
                    closest_dist = dist
                    closest_id = player["track_id"]

            # Only count as possession if close enough
            if closest_dist <= self.possession_radius:
                possession_log.append(
                    {
                        "frame": frame_num,
                        "track_id": closest_id,
                        "distance": round(closest_dist, 1),
                    }
                )

        print(f"[EventsAgent] Possession detected on {len(possession_log)} frames")
        return possession_log

    def detect_possession_changes(self, possession_log):
        """
        Detect when the ball switches between players, with debouncing.

        A possession change only counts if the new player holds the ball
        for at least `change_threshold` consecutive frames. This filters
        out noise like the ball flying past someone for 1 frame.

        Args:
            possession_log: Output from detect_possession()

        Returns:
            List of possession change events:
            [
                {"frame": 42, "from_track_id": 7, "to_track_id": 12},
                {"frame": 89, "from_track_id": 12, "to_track_id": 3},
                ...
            ]
        """
        if len(possession_log) < 2:
            return []

        changes = []

        # Current confirmed possessor
        current_possessor = possession_log[0]["track_id"]

        # Candidate tracking for debouncing
        candidate_id = None
        candidate_count = 0
        candidate_start_frame = None

        for entry in possession_log[1:]:
            track_id = entry["track_id"]
            frame = entry["frame"]

            if track_id == current_possessor:
                # Same player still has it — reset any candidate
                candidate_id = None
                candidate_count = 0
                candidate_start_frame = None

            elif track_id == candidate_id:
                # Same candidate as before — keep counting
                candidate_count += 1

                if candidate_count >= self.change_threshold:
                    # Confirmed possession change
                    changes.append(
                        {
                            "frame": candidate_start_frame,
                            "from_track_id": current_possessor,
                            "to_track_id": candidate_id,
                        }
                    )
                    current_possessor = candidate_id
                    candidate_id = None
                    candidate_count = 0
                    candidate_start_frame = None

            else:
                # New candidate — reset counter
                candidate_id = track_id
                candidate_count = 1
                candidate_start_frame = frame

        print(f"[EventsAgent] Possession changes detected: {len(changes)}")
        return changes