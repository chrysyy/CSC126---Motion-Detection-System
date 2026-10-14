"""
Motion-Triggered Security System (RECORDED VIDEO VERSION)
CSC 126 Computer Vision - Midterm Project

Topics applied:
  (a) Image Operations           - resize, grayscale, histogram equalization,
                                   Gaussian blur, Canny edges, text overlays
  (b) Image Segmentation         - background subtraction (MOG2), thresholding,
                                   morphological cleaning, contour extraction
  (e) Motion and Object Tracking - centroid tracker with IDs and trails,
                                   motion alert (banner + snapshot + CSV log)

Usage:
    1) Set VIDEO_PATH below to your video file, then just click Run, OR
    2) python motion_security_video.py --video sample_input/test.mp4
    If no valid path is found, a file picker window will open.

Keys (click the video window first):
    q = quit        e = toggle edge view
    s = save screenshot        p or SPACE = pause / resume
"""

import argparse
import csv
import os
from collections import OrderedDict
from datetime import datetime

import cv2
import numpy as np

# ============================== SETTINGS ====================================
VIDEO_PATH = "finalVid.mp4"   # change this to your video file
SAVE_OUTPUT_VIDEO = True               # save the processed dashboard as a video
OUTPUT_DIR = "output"

FRAME_WIDTH = 640             # every frame is resized to this width
WARMUP_FRAMES = 30            # frames for the background model to learn
MIN_CONTOUR_AREA = 800        # ignore blobs smaller than this (noise)
ALERT_AREA_THRESHOLD = 3000   # total motion area that triggers an alert
ALERT_COOLDOWN_FRAMES = 60    # frames between saved snapshots
MAX_DISAPPEARED = 15          # frames an object may vanish before ID is dropped
MAX_MATCH_DISTANCE = 80       # max pixels to match a centroid to an existing ID
TRAIL_LENGTH = 30             # how many past points to draw as a trail
SNAPSHOT_DIR = "snapshots"
SCREENSHOT_DIR = "screenshots"
LOG_FILE = "alerts_log.csv"

# Larger kernel used to merge broken fragments of one moving person into one blob
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))


# ============================ VIDEO SETUP ===================================
def choose_video_file():
    """Open a file picker so the user can select a video."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                       ("All files", "*.*")])
        root.destroy()
        return path
    except Exception:
        return ""


def open_video(path):
    """Open the recorded video file. Returns the capture or None."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    return cap


# ======================= (a) IMAGE OPERATIONS ===============================
def preprocess(frame):
    """Resize (keeping aspect ratio), convert to gray, equalize histogram, blur.

    Returns the resized color frame and the preprocessed gray frame.
    """
    h, w = frame.shape[:2]
    new_h = int(h * FRAME_WIDTH / w)
    frame = cv2.resize(frame, (FRAME_WIDTH, new_h))               # standard width
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                # color -> gray
    gray = cv2.equalizeHist(gray)                                 # even out lighting
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)                   # reduce noise
    return frame, blurred

def compute_edges(blurred):
    """Canny edge map of the preprocessed frame (image analysis view)."""
    return cv2.Canny(blurred, 50, 150)


def draw_timestamp(frame, frame_number, fps):
    """Overlay the video time (mm:ss) and frame number like a CCTV feed."""
    seconds = frame_number / fps if fps > 0 else 0
    text = f"VIDEO {int(seconds // 60):02d}:{int(seconds % 60):02d}  |  Frame {frame_number}"
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ======================= (b) IMAGE SEGMENTATION =============================
def create_subtractor():
    """Create the MOG2 background subtractor.

    A lower varThreshold makes it more sensitive to small movements.
    """
    return cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=25, detectShadows=True)


def segment_motion(blurred, subtractor, kernel):
    """Separate moving foreground from the static background.

    Steps: background subtraction -> threshold (remove shadows) ->
    opening (remove specks) -> closing (merge fragments) -> dilation.
    Returns a clean binary mask.
    """
    mask = subtractor.apply(blurred)                              # foreground mask
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)    # drop gray shadows
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)         # remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_KERNEL)  # merge fragments
    mask = cv2.dilate(mask, kernel, iterations=2)                 # grow slightly
    return mask


def extract_objects(mask):
    """Find contours in the mask and return bounding boxes (x, y, w, h)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA:              # skip tiny blobs
            boxes.append(cv2.boundingRect(cnt))
    return boxes


# =================== (e) MOTION AND OBJECT TRACKING =========================
class CentroidTracker:
    """Gives each moving object a persistent ID by matching centroids
    between consecutive frames (nearest-distance matching)."""

    def __init__(self):
        self.next_id = 0
        self.objects = OrderedDict()      # id -> (cx, cy)
        self.disappeared = OrderedDict()  # id -> frames missing
        self.trails = OrderedDict()       # id -> list of past centroids

    def _register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.trails[self.next_id] = [centroid]
        self.next_id += 1

    def _deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.trails[object_id]

    def update(self, centroids):
        """Update tracked objects with this frame's centroids."""
        # Case 1: nothing detected -> everything is "missing"
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > MAX_DISAPPEARED:
                    self._deregister(object_id)
            return self.objects

        # Case 2: nothing tracked yet -> register all
        if len(self.objects) == 0:
            for c in centroids:
                self._register(c)
            return self.objects

        # Case 3: match old objects to new centroids by smallest distance
        object_ids = list(self.objects.keys())
        old_points = np.array(list(self.objects.values()))
        new_points = np.array(centroids)
        dist = np.linalg.norm(old_points[:, None] - new_points[None, :], axis=2)

        rows = dist.min(axis=1).argsort()
        cols = dist.argmin(axis=1)[rows]
        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if dist[row, col] > MAX_MATCH_DISTANCE:
                continue
            object_id = object_ids[row]
            point = tuple(int(v) for v in new_points[col])
            self.objects[object_id] = point
            self.disappeared[object_id] = 0
            self.trails[object_id].append(point)
            self.trails[object_id] = self.trails[object_id][-TRAIL_LENGTH:]
            used_rows.add(row)
            used_cols.add(col)

        # Old objects with no match -> missing
        for row in set(range(len(object_ids))) - used_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > MAX_DISAPPEARED:
                self._deregister(object_id)

        # New centroids with no match -> new objects
        for col in set(range(len(new_points))) - used_cols:
            self._register(tuple(int(v) for v in new_points[col]))

        return self.objects


def draw_tracking(frame, boxes, tracker):
    """Draw bounding boxes, centroids, IDs, and movement trails."""
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for object_id, (cx, cy) in tracker.objects.items():
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"ID {object_id}", (cx - 10, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        points = tracker.trails[object_id]
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], (0, 200, 255), 2)


# =========================== ALERT HANDLING =================================
def trigger_alert(frame, object_count, total_area, frame_number):
    """Save a snapshot of the frame and append a row to the CSV log."""
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(SNAPSHOT_DIR, f"alert_frame{frame_number}_{stamp}.jpg")
    cv2.imwrite(path, frame)

    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "video_frame", "objects",
                             "motion_area", "snapshot"])
        writer.writerow([stamp, frame_number, object_count, total_area, path])


# ============================== DISPLAY =====================================
def build_dashboard(frame, mask, segmented, edges, show_edges):
    """Combine output, mask, and segmented/edge views side by side."""
    frame_h, frame_w = frame.shape[:2]
    panel_w = 427
    panel_h = int(panel_w * frame_h / frame_w)
    size = (panel_w, panel_h)

    panel1 = cv2.resize(frame, size)
    panel2 = cv2.cvtColor(cv2.resize(mask, size), cv2.COLOR_GRAY2BGR)
    if show_edges:
        panel3 = cv2.cvtColor(cv2.resize(edges, size), cv2.COLOR_GRAY2BGR)
        label3 = "Canny Edges"
    else:
        panel3 = cv2.resize(segmented, size)
        label3 = "Segmented Foreground"

    for panel, label in ((panel1, "Video Output"), (panel2, "Motion Mask"),
                         (panel3, label3)):
        cv2.putText(panel, label, (8, panel_h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return np.hstack([panel1, panel2, panel3])


# ================================ MAIN ======================================
def main():
    parser = argparse.ArgumentParser(description="Motion-triggered security (video)")
    parser.add_argument("--video", default=VIDEO_PATH, help="path to the video file")
    args = parser.parse_args()

    # Find the video: given path -> file picker if the path is invalid
    video_path = args.video
    if not os.path.isfile(video_path):
        print(f"Video not found at '{video_path}'. Please select a file...")
        video_path = choose_video_file()
    if not video_path:
        raise SystemExit("No video selected.")

    cap = open_video(video_path)
    if cap is None:
        raise SystemExit(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or video_fps > 120:
        video_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay = max(1, int(1000 / video_fps))      # play at the video's real speed

    subtractor = create_subtractor()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tracker = CentroidTracker()
    frame_count, cooldown, show_edges, paused = 0, 0, False, False
    writer = None
    dashboard = None

    print(f"Playing: {video_path}  ({total_frames} frames, {video_fps:.0f} FPS)")
    print("Keys: q = quit | e = edges | s = screenshot | p/SPACE = pause")

    while True:
        if not paused:
            ok, raw = cap.read()
            if not ok:
                print("End of video.")
                break
            frame_count += 1

            # (a) Image operations
            frame, blurred = preprocess(raw)
            edges = compute_edges(blurred)

            # (b) Segmentation
            mask = segment_motion(blurred, subtractor, kernel)
            segmented = cv2.bitwise_and(frame, frame, mask=mask)
            boxes = extract_objects(mask)

            # (e) Tracking (skipped while the background model warms up)
            learning = frame_count <= WARMUP_FRAMES
            if learning:
                boxes = []
            centroids = [(x + w // 2, y + h // 2) for (x, y, w, h) in boxes]
            tracker.update(centroids)
            draw_tracking(frame, boxes, tracker)
            draw_timestamp(frame, frame_count, video_fps)

            # Motion alert
            total_area = sum(w * h for (_, _, w, h) in boxes)
            if cooldown > 0:
                cooldown -= 1
            if total_area >= ALERT_AREA_THRESHOLD:
                cv2.putText(frame, "MOTION DETECTED!", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                if cooldown == 0:
                    trigger_alert(frame, len(boxes), total_area, frame_count)
                    cooldown = ALERT_COOLDOWN_FRAMES

            # Status text
            status = "Learning background..." if learning \
                else f"Tracked objects: {len(tracker.objects)}"
            cv2.putText(frame, status, (10, frame.shape[0] - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"{frame_count}/{total_frames}",
                        (FRAME_WIDTH - 110, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)

            dashboard = build_dashboard(frame, mask, segmented, edges, show_edges)

            # Save the processed dashboard as a video (for documentation/demo)
            if SAVE_OUTPUT_VIDEO:
                if writer is None:
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    h, w = dashboard.shape[:2]
                    writer = cv2.VideoWriter(
                        os.path.join(OUTPUT_DIR, "processed_output.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (w, h))
                writer.write(dashboard)

            cv2.imshow("Motion-Triggered Security (Video)", dashboard)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("e"):
            show_edges = not show_edges
        elif key in (ord("p"), 32):                # p or SPACE
            paused = not paused
        elif key == ord("s") and dashboard is not None:
            os.makedirs(SCREENSHOT_DIR, exist_ok=True)
            name = datetime.now().strftime("shot_%H-%M-%S.png")
            cv2.imwrite(os.path.join(SCREENSHOT_DIR, name), dashboard)
            print(f"Saved screenshot: {name}")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Processed video saved to {OUTPUT_DIR}/processed_output.mp4")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()