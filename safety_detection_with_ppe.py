# safety_detection_with_ppe.py

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import csv
from datetime import datetime

# Create log directory and filename
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(
    log_dir, f"session_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
)

# Initialize CSV
with open(log_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "Person ID", "Violation", "In Risky Zone"])

# Track logged violations to avoid duplicates
logged_violations = set()


# Frame resizing
FRAME_WIDTH = 1024
FRAME_HEIGHT = 640
RESIZE_SHAPE = (FRAME_WIDTH, FRAME_HEIGHT)

# Risky zone drawing
zone_points = []
drawing = True

def draw_zone(event, x, y, flags, param):
    global zone_points, drawing
    if event == cv2.EVENT_LBUTTONDOWN and drawing:
        zone_points.append((x, y))

# Load models
person_model = YOLO("yolov8n.pt")                  # For person tracking
ppe_model = YOLO("best.pt")                 # Your downloaded Roboflow PPE model
tracker = DeepSort(max_age=30)

# Open webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Define Risky Zone")
cv2.setMouseCallback("Define Risky Zone", draw_zone)

# Step 1: Risky zone selection
print("🖱️ Define risky zone (click points). Press ENTER when done.")
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

    cv2.putText(preview, "Click to draw risky zone. Press ENTER when done.",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Define Risky Zone", preview)
    if cv2.waitKey(1) == 13:  # Enter
        if len(zone_points) >= 3:
            print("✅ Risky zone defined.")
            break
        else:
            print("⚠️ Need at least 3 points.")

cv2.destroyWindow("Define Risky Zone")
risky_zone = np.array(zone_points)

# Function to check if point is inside polygon
def point_in_zone(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# Step 2: Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, RESIZE_SHAPE)
    overlay = frame.copy()

    # Detection: Person + PPE
    person_results = person_model(frame, verbose=False)[0]
    ppe_results = ppe_model(frame, verbose=False)[0]

    # Draw risky zone
    cv2.polylines(overlay, [risky_zone], isClosed=True, color=(0, 0, 255), thickness=2)

    # Person Detections
    detections = []
    for r in person_results.boxes:
        cls = int(r.cls[0])
        conf = float(r.conf[0])
        if cls == 0:  # 'person' in yolov8n.pt
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    # Track persons
    tracks = tracker.update_tracks(detections, frame=frame)

    # Parse PPE detections
    ppe_boxes = []
    for r in ppe_results.boxes:
        cls = int(r.cls[0])
        label = ppe_model.names[cls]
        conf = float(r.conf[0])
        if conf > 0.4:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            ppe_boxes.append((label, [x1, y1, x2, y2]))

    # Analyze each tracked person
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        person_box = [x1, y1, x2, y2]

        # Check if inside risky zone
        in_risky = point_in_zone(center, risky_zone)

        # Check PPEs overlapping this person box
        missing_items = []

        required = {
            'helmet': False,
            'vest': False,
        }

        for label, box in ppe_boxes:
            px1, py1, px2, py2 = box
            iou_x1 = max(x1, px1)
            iou_y1 = max(y1, py1)
            iou_x2 = min(x2, px2)
            iou_y2 = min(y2, py2)

            intersection_area = max(0, iou_x2 - iou_x1) * max(0, iou_y2 - iou_y1)
            person_area = (x2 - x1) * (y2 - y1)
            iou = intersection_area / float(person_area)

            if iou > 0.1:
                if "Hardhat" in label:
                    required['helmet'] = "NO" not in label
                if "Vest" in label:
                    required['vest'] = "NO" not in label

        if not required['helmet']:
            missing_items.append("No Helmet")
        if not required['vest']:
            missing_items.append("No Vest")

        # Draw results
        status_color = (0, 255, 0) if not missing_items and not in_risky else (0, 0, 255)
        label = f"ID:{track_id} {' | '.join(missing_items) if missing_items else 'Safe'}"
        if in_risky:
            label += " | RISKY ZONE"
            
        # Logging 
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for v in missing_items:
            log_id = f"{track_id}_{v}_{in_risky}"
            if log_id not in logged_violations:
                with open(log_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([now_str, track_id, v, "Yes" if in_risky else "No"])
                logged_violations.add(log_id)

        if in_risky and not missing_items:
            log_id = f"{track_id}_RiskyZoneOnly"
            if log_id not in logged_violations:
                with open(log_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([now_str, track_id, "Entered Risky Zone", "Yes"])
                logged_violations.add(log_id)


        cv2.rectangle(overlay, (x1, y1), (x2, y2), status_color, 2)
        cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    cv2.imshow("Industry Safety System", overlay)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
