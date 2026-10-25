"""Turn a recorded trip into the README demo (animated WebP) and a full-trip video.

    python main.py --trips 1 --no-display --video runs/trip.mp4 --telemetry runs/trip.csv
    python tools/make_media.py runs/trip.mp4

The highlight segments below are frame ranges of that recording (seed 7, trip 1);
pick new ones from the telemetry's "decision" column for another recording.
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "media")
SEGMENTS = [  # first frame, last frame, step, caption, speed note
    (20, 100, 4, "1. Lane following from the camera", "x1.5"),
    (286, 360, 3, "2. Red light detected: braking to the stop point", ""),
    (360, 584, 12, "2. Waiting at red", "time-lapse x6"),
    (584, 640, 3, "2. Green: go", ""),
    (740, 845, 2, "3. Pedestrian stepping out: yield", ""),
    (1440, 1518, 3, "4. A* route complete: stopping at the goal", ""),
]
HUD_W, CAM_W, MAP_X = 1700, 800, 1100


def caption(img, text, note=""):
    bar = np.full((34, img.shape[1], 3), 20, np.uint8)
    cv2.putText(bar, text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if note:
        cv2.putText(bar, note, (img.shape[1] - 12 - 9 * len(note), 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (160, 200, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def camera_and_map(hud):
    """Camera view and navigation map side by side, scaled for the README."""
    cam = cv2.resize(hud[:, :CAM_W], (600, 450), interpolation=cv2.INTER_AREA)
    nav = cv2.resize(hud[:, MAP_X:], (450, 450), interpolation=cv2.INTER_AREA)
    return np.hstack([cam, nav])


def main(video):
    cap = cv2.VideoCapture(video)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    print(f"{len(frames)} frames of {frames[0].shape[1]} x {frames[0].shape[0]}")
    os.makedirs(OUT, exist_ok=True)

    demo = []
    for first, last, step, text, note in SEGMENTS:
        for i in range(first, min(last, len(frames)), step):
            demo.append(Image.fromarray(cv2.cvtColor(caption(camera_and_map(frames[i]), text, note), cv2.COLOR_BGR2RGB)))
    path = os.path.join(OUT, "demo.webp")
    demo[0].save(path, save_all=True, append_images=demo[1:], duration=100, loop=0, quality=72, method=6)
    print(f"demo.webp: {len(demo)} frames, {os.path.getsize(path) / 1e6:.1f} MB")

    path = os.path.join(OUT, "full_trip.webm")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"VP80"), 20, (HUD_W, frames[0].shape[0] + 34))
    for i in range(0, len(frames), 2):
        waiting = 380 < i < 570
        if waiting and i % 8:
            continue
        writer.write(caption(frames[i], "CARLA Town02 - camera + LiDAR autonomous driving, A* navigation",
                             "time-lapse x8" if waiting else "x2"))
    writer.release()
    print(f"full_trip.webm: {os.path.getsize(path) / 1e6:.1f} MB")
    cv2.imwrite(os.path.join(OUT, "poster.jpg"), frames[800], [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main(sys.argv[1])
