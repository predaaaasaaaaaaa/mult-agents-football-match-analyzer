"""
Step 4 — ByteTrack tracking on top of YOLO detections.
Each detected person gets a persistent ID across frames.
"""

from ultralytics import YOLO
import supervision as sv

model = YOLO("yolov8n.pt")

# Initialize ByteTrack tracker
tracker = sv.ByteTrack()

# Run YOLO in streaming mode (same as test_yolo.py)
results = model(
    source="data/match_clip.mp4",
    stream=True,
    conf=0.3,
    classes=[0, 32],  # person + sports ball
    device=0,
)

frame_count = 0
all_track_ids = set()  # Collect every unique ID we see

for result in results:
    frame_count += 1

    # Convert YOLO output → supervision Detections format
    detections = sv.Detections.from_ultralytics(result)

    # Filter to only persons (class 0) for tracking
    person_mask = detections.class_id == 0
    person_detections = detections[person_mask]

    # ByteTrack assigns persistent IDs
    tracked = tracker.update_with_detections(person_detections)

    # Collect the IDs we see this frame
    if tracked.tracker_id is not None:
        for tid in tracked.tracker_id:
            all_track_ids.add(tid)

    # Print progress every 100 frames
    if frame_count % 100 == 0:
        num_tracked = len(tracked.tracker_id) if tracked.tracker_id is not None else 0
        print(f"Frame {frame_count}: tracking {num_tracked} persons")

    # Stop after 500 frames
    if frame_count >= 500:
        break

print(f"\n--- Results ---")
print(f"Processed {frame_count} frames")
print(f"Total unique track IDs seen: {len(all_track_ids)}")
print(f"Track IDs: {sorted(all_track_ids)}")
