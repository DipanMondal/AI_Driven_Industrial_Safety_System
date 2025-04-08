# video_analyzer.py
import cv2
import json
import math
import csv
import os
from datetime import datetime
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def point_in_zone(point, zone):
    return cv2.pointPolygonTest(zone, point, False) >= 0

def analyze_video(video_path, output_path, log_path, zone_file):
    person_model = YOLO("yolov8n.pt")
    ppe_model = YOLO("best.pt")
    tracker = DeepSort(max_age=30)
    previous_positions = {}
    logged_violations = set()

    with open(zone_file, "r") as f:
        risky_zone = np.array(json.load(f), dtype=np.int32)

    cap = cv2.VideoCapture(video_path)
    width, height = 1024, 640
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    with open(log_path, "w", newline='') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["Time", "Person ID", "Violation", "In Risky Zone"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (width, height))
            overlay = frame.copy()

            person_results = person_model(frame, verbose=False)[0]
            ppe_results = ppe_model(frame, verbose=False)[0]

            detections = []
            for r in person_results.boxes:
                cls = int(r.cls[0])
                conf = float(r.conf[0])
                if cls == 0 and conf > 0.5:
                    x1, y1, x2, y2 = map(int, r.xyxy[0])
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

            tracks = tracker.update_tracks(detections, frame=frame)

            # PPE boxes
            ppe_boxes = []
            for r in ppe_results.boxes:
                cls = int(r.cls[0])
                label = ppe_model.names[cls]
                conf = float(r.conf[0])
                if conf > 0.4:
                    x1, y1, x2, y2 = map(int, r.xyxy[0])
                    ppe_boxes.append((label, [x1, y1, x2, y2]))

            cv2.polylines(overlay, [risky_zone], True, (0, 0, 255), 2)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Movement
                movement_status = "Idle"
                if track_id in previous_positions:
                    dx = center[0] - previous_positions[track_id][0]
                    dy = center[1] - previous_positions[track_id][1]
                    dist = math.sqrt(dx ** 2 + dy ** 2)
                    if dist > 8:
                        movement_status = "Running"
                    elif dist > 2:
                        movement_status = "Walking"
                previous_positions[track_id] = center

                # PPE check
                required = {"helmet": False, "vest": False}
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

                missing = []
                if not required['helmet']:
                    missing.append("No Helmet")
                if not required['vest']:
                    missing.append("No Vest")

                in_risky = point_in_zone(center, risky_zone)
                now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                for v in missing:
                    log_id = f"{track_id}_{v}_{in_risky}"
                    if log_id not in logged_violations:
                        writer.writerow([now_str, track_id, v, "Yes" if in_risky else "No"])
                        logged_violations.add(log_id)

                if in_risky and not missing:
                    log_id = f"{track_id}_RiskyZoneOnly"
                    if log_id not in logged_violations:
                        writer.writerow([now_str, track_id, "Entered Risky Zone", "Yes"])
                        logged_violations.add(log_id)

                label = f"ID:{track_id} | {movement_status}"
                if missing:
                    label += f" | {' | '.join(missing)}"
                if in_risky:
                    label += " | RISKY ZONE"

                color = (0, 255, 0) if not missing and not in_risky else (0, 0, 255)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out.write(overlay)

    cap.release()
    out.release()
