from ultralytics import YOLO
import supervision as sv
import cv2

model = YOLO("yolov8n.pt")

# ByteTrack tracker
tracker = sv.ByteTrack(
    track_activation_threshold=0.5,
    lost_track_buffer=60,
    minimum_matching_threshold=0.8,
    frame_rate=25,
)

# Annotators
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

# Run YOLO
results = model(
    source="data/match_clip.mp4",
    stream=True,
    conf=0.3,
    classes=[0, 32],
    device=0,
)

# Need video info (width, height, fps) to create the output file
cap = cv2.VideoCapture("data/match_clip.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
cap.release()

# Create the output video writer
output_path = "data/tracking_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0

for result in results:
    frame_count += 1

    # Get the original frame (no YOLO drawings on it)
    frame = result.orig_img

    # Convert YOLO → supervision detections
    detections = sv.Detections.from_ultralytics(result)

    # Filter persons only, then track
    person_mask = detections.class_id == 0
    person_detections = detections[person_mask]
    tracked = tracker.update_with_detections(person_detections)

    # Build labels like "ID: 3", "ID: 17"
    labels = []
    if tracked.tracker_id is not None:
        labels = [f"ID: {tid}" for tid in tracked.tracker_id]

    # Draw boxes and labels onto the frame
    frame = box_annotator.annotate(scene=frame, detections=tracked)
    frame = label_annotator.annotate(scene=frame, detections=tracked, labels=labels)

    # Write the annotated frame to the output video
    writer.write(frame)

    if frame_count % 100 == 0:
        num_tracked = len(tracked.tracker_id) if tracked.tracker_id is not None else 0
        print(f"Frame {frame_count}: tracking {num_tracked} persons")

    # Only process 500 frames for now
    if frame_count >= 500:
        break

writer.release()
print(f"\nDone! Saved {frame_count} frames to {output_path}")
print(f"Open it in VLC or any video player to see the tracking!")
