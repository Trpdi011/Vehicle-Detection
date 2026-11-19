"""
vehicle_analytics.py — Vehicle counting for video / webcam / single image.

FIXES vs original (Main.py, vehicle_detection_video.py, pv.py)
1. The video version counted detections *per frame* and once a minute
   saved whatever the counts happened to be on that single frame — not a
   traffic count at all. Now ByteTrack assigns IDs and each unique vehicle
   is counted once; the per-minute record stores new unique vehicles seen
   in that minute plus cumulative totals.
2. MongoDB port inconsistency (27018 here vs 27017 elsewhere) — all
   modules now share common.MONGO_URI.
3. Duplicate confidence filtering (model conf=0.25 plus a manual < 0.25
   check) removed.
4. Three near-identical scripts merged into one with a --image mode.
5. Aspect-ratio-preserving resize; safe DB writes; stale-track cleanup.

Run:
  python vehicle_analytics.py                 # webcam
  python vehicle_analytics.py traffic.mp4     # video file
  python vehicle_analytics.py --image im.png  # single image with ROI select
"""

import sys
import time
from datetime import datetime

import cv2
from ultralytics import YOLO

from common import (SafeDB, VideoSource, draw_header, fullscreen_window,
                    resize_letterbox)

VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}
CONF = 0.4
WINDOW = "Vehicle Analytics"


# =====================================================
# Single-image mode (ROI select, replaces Main.py / pv.py)
# =====================================================
def run_image(path):
    model = YOLO("yolov8m.pt")
    image = cv2.imread(path)
    if image is None:
        print(f"ERROR: image not found: {path}")
        return

    roi = cv2.selectROI("Select Road Area", image, False)
    cv2.destroyWindow("Select Road Area")
    x, y, w, h = roi
    if w == 0 or h == 0:          # user cancelled -> whole image
        x, y = 0, 0
        h, w = image.shape[:2]

    results = model(image[y:y + h, x:x + w], imgsz=1280, conf=0.25,
                    verbose=False)
    counts = {c: 0 for c in sorted(VEHICLE_CLASSES)}
    for box in results[0].boxes:
        name = model.names[int(box.cls[0])]
        if name not in counts:
            continue
        counts[name] += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, x2, y1, y2 = x1 + x, x2 + x, y1 + y, y2 + y
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{name} {float(box.conf[0]):.2f}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    y_pos = 35
    print("\nDetected Vehicles")
    for name, count in counts.items():
        if count:
            print(f"  {name}: {count}")
            cv2.putText(image, f"{name.capitalize()}: {count}", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_pos += 35

    fullscreen_window(WINDOW)
    cv2.imshow(WINDOW, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =====================================================
# Video / webcam mode with unique-vehicle tracking
# =====================================================
def run_stream(source):
    model = YOLO("yolov8m.pt")
    db = SafeDB("vehicle_monitoring")
    cap = VideoSource(source)
    fullscreen_window(WINDOW)

    total_unique = {c: 0 for c in sorted(VEHICLE_CLASSES)}
    minute_unique = {c: 0 for c in sorted(VEHICLE_CLASSES)}
    seen_ids = set()
    last_seen = {}
    last_save = time.time()

    print("Vehicle Analytics started. Press q or ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video finished")
            break
        frame = resize_letterbox(frame)
        now = time.time()

        results = model.track(frame, persist=True, conf=CONF,
                              tracker="bytetrack.yaml", verbose=False)
        current = {c: 0 for c in sorted(VEHICLE_CLASSES)}

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                name = model.names[int(box.cls[0])]
                if name not in VEHICLE_CLASSES:
                    continue
                current[name] += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = name
                if box.id is not None:
                    tid = int(box.id[0])
                    label = f"{name} #{tid}"
                    last_seen[tid] = now
                    if tid not in seen_ids:      # count each vehicle ONCE
                        seen_ids.add(tid)
                        total_unique[name] += 1
                        minute_unique[name] += 1
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # prune track memory (ids stay in seen_ids so they aren't recounted
        # soon, but very old entries are dropped to bound memory)
        stale = [t for t, ts in last_seen.items() if now - ts > 300]
        for t in stale:
            last_seen.pop(t, None)
            seen_ids.discard(t)

        # -------- Save unique counts every minute --------
        if now - last_save >= 60:
            db.insert("vehicle_counts", {
                "timestamp": datetime.now(),
                "new_this_minute": dict(minute_unique),
                "cumulative_total": dict(total_unique),
            })
            print("Vehicle data saved:", minute_unique)
            minute_unique = {c: 0 for c in sorted(VEHICLE_CLASSES)}
            last_save = now

        # -------- Overlay --------
        draw_header(frame, "VEHICLE ANALYTICS",
                    datetime.now().strftime("%H:%M:%S"))
        y_pos = 90
        for name in sorted(VEHICLE_CLASSES):
            cv2.putText(frame,
                        f"{name.capitalize()}: now {current[name]} | "
                        f"total {total_unique[name]}",
                        (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 255), 2)
            y_pos += 32

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--image":
        run_image(args[1] if len(args) > 1 else "im.png")
    else:
        src = args[0] if args else 0
        run_stream(int(src) if str(src).isdigit() else src)
