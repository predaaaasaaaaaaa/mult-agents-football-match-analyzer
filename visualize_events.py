"""
Draws tracking + team colors + possession + events on the match video.
"""

import json
import cv2


def main():
    video_path = "data/match_clip.mp4"
    output_path = "data/events_output.mp4"
    teams_path = "data/tracking_data_teams.json"

    # Load team-labeled tracking data
    print("Loading tracking data...")
    with open(teams_path, "r") as f:
        tracking_data = json.load(f)

    # Group detections by frame
    frames = {}
    for d in tracking_data:
        f = d["frame"]
        if f not in frames:
            frames[f] = {"players": [], "ball": None}
        if d["type"] == "player":
            frames[f]["players"].append(d)
        elif d["type"] == "ball":
            frames[f]["ball"] = d

    # Run events agent to get passes and turnovers
    from src.agents.events.events_agent import EventsAgent
    events = EventsAgent(possession_radius=80, change_threshold=3)
    possession_log = events.detect_possession(tracking_data)
    changes = events.detect_possession_changes(possession_log)
    passes, turnovers = events.detect_passes(changes)

    # Build lookup: frame → event text
    event_lookup = {}
    for p in passes:
        event_lookup[p["frame"]] = f"PASS: track_{p['from_track_id']} -> track_{p['to_track_id']} (Team {p['from_team']})"
    for t in turnovers:
        event_lookup[t["frame"]] = f"TURNOVER: track_{t['from_track_id']} (Team {t['from_team']}) -> track_{t['to_track_id']} (Team {t['to_team']})"

    # Build lookup: frame → possessor track_id
    possession_lookup = {}
    for entry in possession_log:
        possession_lookup[entry["frame"]] = entry["track_id"]

    # Team colors for drawing
    TEAM_COLORS = {
        "A": (0, 255, 0),    # Green
        "B": (0, 0, 255),    # Red
        None: (200, 200, 200) # Gray
    }
    BALL_COLOR = (0, 255, 255)       # Yellow
    POSSESSOR_COLOR = (255, 0, 255)  # Magenta highlight

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    active_event = None
    event_display_frames = 0

    print(f"Writing annotated video to {output_path}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        frame_data = frames.get(frame_num, {"players": [], "ball": None})
        possessor_id = possession_lookup.get(frame_num)

        # Check for new event on this frame
        if frame_num in event_lookup:
            active_event = event_lookup[frame_num]
            event_display_frames = 50  # Show event text for 50 frames (~2 sec)

        # Draw players with team colors
        for player in frame_data["players"]:
            x, y, w, h = player["x"], player["y"], player["w"], player["h"]
            team = player.get("team")
            tid = player["track_id"]

            # Highlight possessor with thick magenta border
            if tid == possessor_id:
                cv2.rectangle(frame, (x - 2, y - 2), (x + w + 2, y + h + 2), POSSESSOR_COLOR, 3)

            color = TEAM_COLORS.get(team, (200, 200, 200))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Label: track_id + team
            label = f"#{tid} {team or '?'}"
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw ball
        ball = frame_data["ball"]
        if ball:
            bx, by, bw, bh = ball["x"], ball["y"], ball["w"], ball["h"]
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), BALL_COLOR, 2)
            cv2.putText(frame, "BALL", (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BALL_COLOR, 1)

        # Draw event banner
        if event_display_frames > 0:
            # Black background bar
            cv2.rectangle(frame, (0, height - 60), (width, height), (0, 0, 0), -1)
            # Event text
            event_color = (0, 255, 0) if "PASS" in active_event else (0, 0, 255)
            cv2.putText(frame, active_event, (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, event_color, 2)
            event_display_frames -= 1

        # Frame counter top-right
        cv2.putText(frame, f"Frame: {frame_num}", (width - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)

        if frame_num % 500 == 0:
            print(f"  Processed {frame_num} frames...")

    cap.release()
    out.release()
    print(f"Done! Output: {output_path}")


if __name__ == "__main__":
    main()