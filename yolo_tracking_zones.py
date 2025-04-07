# yolo_tracking_zones.py

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize Deep SORT
tracker = DeepSort(max_age=30)

# Define the risky zone polygon (adjust to match your camera view)
# Format: [[x1, y1], [x2, y2], ...]
risky_zone = np.array([[100, 100], [500, 100], [500, 400], [100, 400]])

def point_in_zone(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# Webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw risky zone on frame
    overlay = frame.copy()
    cv2.polylines(overlay, [risky_zone], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.putText(overlay, "Risky Zone", (risky_zone[0][0], risky_zone[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Detect with YOLO
    results = model(frame)[0]

    detections = []
    for r in results.boxes:
        cls = int(r.cls[0])
        conf = float(r.conf[0])
        if cls == 0:  # 'person'
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    # Track people
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # Check if person is inside risky zone
        in_risky_zone = point_in_zone(center, risky_zone)

        # Draw bounding box
        color = (0, 0, 255) if in_risky_zone else (0, 255, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.circle(overlay, center, 5, color, -1)
        label = f'ID: {track_id} {"[RISK]" if in_risky_zone else ""}'
        cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    cv2.imshow("Safety Zone Detection", overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
