"""
Team Classifier
Assigns players to Team A or Team B using jersey color clustering.
Runs KMeans on the dominant torso color of each player crop.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    def __init__(self):
        self.kmeans = None
        self.team_colors = None

    def fit(self, video_path, tracking_data, sample_frames=100):
        """
        Learn team colors from the first N frames of the video.

        1. For each player detection in the first N frames, crop the torso
        2. Extract dominant color in HSV (hue + saturation only)
        3. Run KMeans k=2 to find two team color clusters
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("[TeamClassifier] ERROR: Cannot open video")
            return

        # Group player detections by frame
        frame_players = {}
        for d in tracking_data:
            if d["type"] == "player" and d["frame"] <= sample_frames:
                f = d["frame"]
                if f not in frame_players:
                    frame_players[f] = []
                frame_players[f].append(d)

        color_samples = []  # List of (H, S) per player crop
        sample_track_ids = []  # Corresponding track IDs

        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            if frame_num > sample_frames:
                break

            if frame_num not in frame_players:
                continue

            for player in frame_players[frame_num]:
                color = self._extract_torso_color(frame, player)
                if color is not None:
                    color_samples.append(color)
                    sample_track_ids.append(player["track_id"])

        cap.release()

        if len(color_samples) < 10:
            print(
                f"[TeamClassifier] Not enough samples ({len(color_samples)}), need 10+"
            )
            return

        # Run KMeans with k=2
        samples_array = np.array(color_samples)
        self.kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.kmeans.fit(samples_array)
        self.team_colors = self.kmeans.cluster_centers_

        print(f"[TeamClassifier] Fitted on {len(color_samples)} player crops")
        print(
            f"[TeamClassifier] Team A color (H,S): ({self.team_colors[0][0]:.0f}, {self.team_colors[0][1]:.0f})"
        )
        print(
            f"[TeamClassifier] Team B color (H,S): ({self.team_colors[1][0]:.0f}, {self.team_colors[1][1]:.0f})"
        )

    def predict(self, video_path, tracking_data):
        """
        Assign team labels to ALL player detections in tracking_data.
        Adds a "team" field ("A" or "B") to each player dict.
        """
        if self.kmeans is None:
            print("[TeamClassifier] ERROR: Must call fit() first")
            return tracking_data

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("[TeamClassifier] ERROR: Cannot open video")
            return tracking_data

        # Group player detections by frame
        frame_players = {}
        for i, d in enumerate(tracking_data):
            if d["type"] == "player":
                f = d["frame"]
                if f not in frame_players:
                    frame_players[f] = []
                frame_players[f].append(i)  # Store index into tracking_data

        assigned = 0
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            if frame_num not in frame_players:
                continue

            for idx in frame_players[frame_num]:
                player = tracking_data[idx]
                color = self._extract_torso_color(frame, player)

                if color is not None:
                    cluster = self.kmeans.predict([color])[0]
                    tracking_data[idx]["team"] = "A" if cluster == 0 else "B"
                    assigned += 1
                else:
                    tracking_data[idx]["team"] = None

        cap.release()
        print(f"[TeamClassifier] Assigned teams to {assigned} player detections")
        return tracking_data

    def _extract_torso_color(self, frame, player):
        """
        Crop the middle band of the player bounding box (torso core),
        convert to HSV, return median (H, S) — ignore V (brightness).
        Uses middle 30% vertically (skip head + shorts) and
        middle 60% horizontally (skip edges that catch grass/background).
        """
        h_frame, w_frame = frame.shape[:2]

        x = max(0, player["x"])
        y = max(0, player["y"])
        w = player["w"]
        h = player["h"]

        # Vertical: 20% to 50% of bbox = core torso (skip head and shorts)
        torso_top = y + int(h * 0.2)
        torso_bottom = y + int(h * 0.5)

        # Horizontal: 20% to 80% of bbox = skip edges (grass bleed)
        torso_left = x + int(w * 0.2)
        torso_right = x + int(w * 0.8)

        # Clamp to frame bounds
        torso_left = max(0, min(torso_left, w_frame))
        torso_right = max(0, min(torso_right, w_frame))
        torso_top = max(0, min(torso_top, h_frame))
        torso_bottom = max(0, min(torso_bottom, h_frame))

        if torso_left >= torso_right or torso_top >= torso_bottom:
            return None

        crop = frame[torso_top:torso_bottom, torso_left:torso_right]

        if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
            return None

        # Convert to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Median H and S (more robust than mean against outliers)
        median_h = np.median(hsv[:, :, 0])
        median_s = np.median(hsv[:, :, 1])

        return [median_h, median_s]

        # Convert to HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Mean H and S (ignore V — it's just brightness/shadows)
        mean_h = np.mean(hsv[:, :, 0])
        mean_s = np.mean(hsv[:, :, 1])

        return [mean_h, mean_s]
