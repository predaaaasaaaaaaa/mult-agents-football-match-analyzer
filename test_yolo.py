"""
First YOLO detection — streaming mode (doesn't eat all RAM).
"""

from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

# Run with stream=True so results come one frame at a time
# instead of YOLO trying to hold all 5230 frames in memory
results = model(
    source="data/match_clip.mp4",
    stream=True,          # KEY: yields results one by one
    conf=0.3,
    classes=[0, 32],      # person + sports ball
    device=0,
)

# Process just the first 500 frames and save ONE annotated frame
frame_count = 0
persons_per_frame = []

for result in results:
    frame_count += 1

    # Count detections this frame
    boxes = result.boxes
    num_persons = sum(1 for b in boxes if int(b.cls[0]) == 0)
    num_balls = sum(1 for b in boxes if int(b.cls[0]) == 32)
    persons_per_frame.append(num_persons)

    # Save one annotated frame so you can SEE the detections
    if frame_count == 100:
        annotated = result.plot()  # Draw boxes on the frame
        cv2.imwrite("data/detection_sample.jpg", annotated)
        print(f"Saved annotated frame to data/detection_sample.jpg")

    # Stop after 500 frames (saves time for testing)
    if frame_count >= 500:
        break

# Print summary
avg_persons = sum(persons_per_frame) / len(persons_per_frame)
print(f"\nProcessed {frame_count} frames")
print(f"Average persons detected per frame: {avg_persons:.1f}")
print(f"Min: {min(persons_per_frame)}, Max: {max(persons_per_frame)}")
print("\nOpen data/detection_sample.jpg to see the detections!")