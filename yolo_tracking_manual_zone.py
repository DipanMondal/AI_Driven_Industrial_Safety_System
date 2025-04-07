# yolo_tracking_manual_zone_resized.py

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Global list to hold clicked points
drawing = True
zone_points = []

# Resize dimensions
FRAME_WIDTH = 1024
FRAME_HEIGHT = 640
RESIZE_SHAPE = (FRAME_WIDTH, FRAME_HEIGHT)

# Mouse callback function
def draw_zone(event, x, y, flags, param):
    global zone_points, drawing
    if event == cv2.EVENT_LBUTTONDOWN and drawing:
        zone_points.append((x, y))

# Load model
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

# Open webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Define Risky Zone")
cv2.setMouseCallback("Define Risky Zone", draw_zone)

# Step 1: Risky zone drawing
print("🖱️ Click to draw the risky zone (polygon). Press ENTER when done.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, RESIZE_SHAPE)
    preview = frame.copy()

    if len(zone_points) > 0:
        for pt in zone_points:
            cv2.circle(preview, pt, 5, (0, 0, 255), -1)
        cv2.polylines(preview, [np.array(zone_points)], isClosed=True, color=(0, 0, 255), thickness=2)

    cv2.putText(preview, "Click to set risky zone. Press ENTER to confirm.",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Define Risky Zone", preview)
    key = cv2.waitKey(1)
    if key == 13:  # ENTER
        if len(zone_points) >= 3:
            print("✅ Risky zone defined. Starting detection...")
            break
        else:
            print("⚠️ You need at least 3 points to define a polygon.")

cv2.destroyWindow("Define Risky Zone")
risky_zone = np.array(zone_points)

# Step 2: Detection + tracking
def point_in_zone(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, RESIZE_SHAPE)
    overlay = frame.copy()

    # Draw risky zone
    cv2.polylines(overlay, [risky_zone], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.putText(overlay, "Risky Zone", (risky_zone[0][0], risky_zone[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Detection
    results = model(frame)[0]

    detections = []
    for r in results.boxes:
        cls = int(r.cls[0])
        conf = float(r.conf[0])
        if cls == 0:  # person
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    # Tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        in_risky_zone = point_in_zone(center, risky_zone)

        color = (0, 0, 255) if in_risky_zone else (0, 255, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.circle(overlay, center, 5, color, -1)
        label = f'ID: {track_id} {"[RISK]" if in_risky_zone else ""}'
        cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    cv2.imshow("Industry Safety System", overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
