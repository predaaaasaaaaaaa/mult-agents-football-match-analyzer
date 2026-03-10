"""
First YOLO detection on real match footage.
Run this and check the output video in runs/detect/
"""

from ultralytics import YOLO

# Load YOLOv8 nano — smallest and fastest model.
# First run will auto-download the weights (~6MB).
model = YOLO("yolov8n.pt")

# Run detection on your match clip
# - source: path to your video
# - save: saves an annotated video with bounding boxes drawn
# - conf: minimum confidence threshold (0.5 = ignore weak detections)
# - classes: [0] = only detect "person" class from COCO dataset
#            [32] = "sports ball"
#            We detect both persons and sports balls.
results = model(
    source="data/match_clip.mp4",
    save=True,
    conf=0.3,
    classes=[0, 32],
    device=0,       # 0 = first GPU (your RTX 2050)
)

print("\nDone! Check the annotated video in the 'runs/detect/' folder.")