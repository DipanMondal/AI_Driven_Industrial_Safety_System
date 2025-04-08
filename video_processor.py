import cv2
import numpy as np
import json
import math
import os
import csv
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import av


class VideoProcessor:
    def __init__(self, log_file):
        self.person_model = YOLO("yolov8n.pt")
        self.ppe_model = YOLO("best.pt")
        self.tracker = DeepSort(max_age=30)

        self.previous_positions = {}
        self.logged_violations = set()
        self.violations_to_log = []

        self.log_file = log_file

        # Load risky zone
        with open("zones/risky_zone.json", "r") as f:
            self.risky_zone = np.array(json.load(f), dtype=np.int32)

    def point_in_zone(self, point):
        return cv2.pointPolygonTest(self.risky_zone, point, False) >= 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        overlay = img.copy()

        person_results = self.person_model(img, verbose=False)[0]
        ppe_results = self.ppe_model(img, verbose=False)[0]

        detections = []
        for r in person_results.boxes:
            cls = int(r.cls[0])
            conf = float(r.conf[0])
            if cls == 0 and conf > 0.5:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        tracks = self.tracker.update_tracks(detections, frame=img)

        # Parse PPE detections
        ppe_boxes = []
        for r in ppe_results.boxes:
            cls = int(r.cls[0])
            label = self.ppe_model.names[cls]
            conf = float(r.conf[0])
            if conf > 0.4:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                ppe_boxes.append((label, [x1, y1, x2, y2]))

        cv2.polylines(overlay, [self.risky_zone], isClosed=True, color=(0, 0, 255), thickness=2)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Movement estimation
            movement_status = "Idle"
            if track_id in self.previous_positions:
                dx = center[0] - self.previous_positions[track_id][0]
                dy = center[1] - self.previous_positions[track_id][1]
                dist = math.sqrt(dx**2 + dy**2)
                if dist < 2:
                    movement_status = "Idle"
                elif dist < 8:
                    movement_status = "Walking"
                else:
                    movement_status = "Running"
            self.previous_positions[track_id] = center

            # Check PPE
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

            in_risky = self.point_in_zone(center)
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            for v in missing:
                log_id = f"{track_id}_{v}_{in_risky}"
                if log_id not in self.logged_violations:
                    self.logged_violations.add(log_id)
                    self.violations_to_log.append([now_str, track_id, v, "Yes" if in_risky else "No"])

            if in_risky and not missing:
                log_id = f"{track_id}_RiskyZoneOnly"
                if log_id not in self.logged_violations:
                    self.logged_violations.add(log_id)
                    self.violations_to_log.append([now_str, track_id, "Entered Risky Zone", "Yes"])

            label = f"ID:{track_id} | {movement_status}"
            if missing:
                label += f" | {' | '.join(missing)}"
            if in_risky:
                label += " | RISKY ZONE"

            color = (0, 255, 0) if not missing and not in_risky else (0, 0, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return av.VideoFrame.from_ndarray(overlay, format="bgr24")

    def __del__(self):
        if self.log_file and self.violations_to_log:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time", "Person ID", "Violation", "In Risky Zone"])
                writer.writerows(self.violations_to_log)
