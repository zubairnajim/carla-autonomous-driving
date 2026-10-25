"""Regenerate the report figures in docs/figures.

Inputs come from runs of main.py:
    python main.py --trips 2 --dump runs/dump --dump-every 10          # raw camera / LiDAR frames
    python main.py --trips 6 --telemetry runs/nav.csv --trips-json runs/nav.json
    python tools/validate_planner.py --roads runs/roads.json          # road polylines for the route maps

Then, for example:
    python tools/make_figures.py arch
    python tools/make_figures.py lane runs/dump/001030
    python tools/make_figures.py light runs/dump/000340
    python tools/make_figures.py ped runs/dump/000810
    python tools/make_figures.py routes runs/nav runs/roads.json
    python tools/make_figures.py speed runs/nav 2

Needs matplotlib in addition to the project requirements.
"""
import csv
import json
import os
import sys

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from config import Config
from perception.lanes import LaneDetector
from perception.lidar import LidarProcessor
from perception.traffic_lights import HSV_RANGES, TrafficLightDetector

OUT = os.path.join(ROOT, "docs", "figures")
cfg = Config()
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 150, "savefig.bbox": "tight"})
INK, GREEN, RED, BLUE, PURPLE, ORANGE, GREY = "#1f2933", "#059669", "#dc2626", "#2563eb", "#7c3aed", "#d97706", "#9ca3af"
EGO_HALF_LENGTH, EGO_HALF_WIDTH = 2.4, 1.0


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def no_ticks(axes):
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


def fig_arch():
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    def box(x, y, w, h, title, sub, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#334155", lw=1))
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=9.5, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.5, sub, ha="center", va="top", fontsize=7.8, color="#374151")

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.1, color="#475569"))

    box(0.1, 3.1, 1.7, 1.05, "RGB camera", "800 x 600, 100 deg FOV\n20 Hz", "#dbeafe")
    box(0.1, 1.75, 1.7, 1.05, "LiDAR", "32 channels, 50 m\n300k points/s", "#dbeafe")
    box(0.1, 0.05, 1.7, 0.95, "HD map", "lane topology\n(route only)", "#e5e7eb")
    box(2.6, 3.5, 2.5, 0.7, "Lane detection", "IPM, top-hat, sliding windows", "#fef3c7")
    box(2.6, 2.65, 2.5, 0.7, "Traffic-light recognition", "HSV, housing test, geometric gate", "#fef3c7")
    box(2.6, 1.8, 2.5, 0.7, "Object detection", "YOLOX-S in cv2.dnn", "#fef3c7")
    box(2.6, 0.95, 2.5, 0.7, "LiDAR processing", "obstacle band, corridor, ranging", "#fef3c7")
    box(5.9, 2.1, 2.3, 2.1, "Behaviour planner",
        "lowest speed limit wins:\ncruise, junction, curve,\nred light, pedestrian,\nobstacle, destination", "#dcfce7")
    box(5.9, 0.05, 2.3, 1.3, "Route planner", "A* on lane-segment graph\nshortest legal route", "#e5e7eb")
    box(9.0, 2.1, 1.9, 2.1, "Controller", "pure-pursuit steering\nPI speed control", "#fce7f3")
    box(9.0, 0.05, 1.9, 1.3, "Vehicle", "throttle, brake, steer", "#e5e7eb")
    for y in (3.85, 3.0, 2.15):
        arrow((1.8, 3.6), (2.6, y))
    for y in (3.0, 2.15, 1.3):
        arrow((1.8, 2.25), (2.6, y))
    arrow((1.8, 0.5), (5.9, 0.5))
    for y, ty in ((3.85, 3.6), (3.0, 3.2), (2.15, 2.8), (1.3, 2.4)):
        arrow((5.1, y), (5.9, ty))
    arrow((7.05, 1.35), (7.05, 2.1))
    arrow((8.2, 3.15), (9.0, 3.15))
    arrow((9.95, 2.1), (9.95, 1.35))
    save(fig, "architecture.png")


def fig_lane(stem):
    img = cv2.imread(stem + ".png")
    det = LaneDetector(cfg)
    res = det.detect(img)
    bev = cv2.warpPerspective(img, det.M, (det.bev_w, det.bev_h))
    corners = np.float32([[[0, 0], [det.bev_w, 0], [det.bev_w, det.bev_h], [0, det.bev_h]]])
    src = cv2.perspectiveTransform(corners, det.Minv)[0]
    over = img.copy()
    det.draw(over, res)
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.6), gridspec_kw={"width_ratios": [2.2, 0.9, 0.9, 2.2]})
    ax[0].imshow(rgb(img))
    ax[0].plot(*np.vstack([src, src[:1]]).T, color="#facc15", lw=1.8)
    ax[0].set_title("(a) Camera frame, road region outlined")
    ax[1].imshow(rgb(bev))
    ax[1].set_title("(b) Bird's-eye view")
    ax[2].imshow(res.mask, cmap="gray")
    fwd = np.linspace(cfg.bev_x_min, cfg.bev_x_max, 50)
    for fit, color in ((res.left, "#22d3ee"), (res.right, "#fb923c")):
        if fit is not None:
            ax[2].plot(det._col(np.polyval(fit, fwd)), det._row(fwd), color=color, lw=2)
    ax[2].set_xlim(0, det.bev_w)
    ax[2].set_ylim(det.bev_h, 0)
    ax[2].set_title("(c) Top-hat mask + fits")
    ax[3].imshow(rgb(over))
    ax[3].set_title(f"(d) Lane projected back (confidence {res.confidence:.1f})")
    no_ticks(ax)
    save(fig, "lane_pipeline.png")


def fig_light(stem):
    img = cv2.imread(stem + ".png")
    meta = json.load(open(stem + ".json"))
    lidar = LidarProcessor(cfg, EGO_HALF_LENGTH, EGO_HALF_WIDTH)
    lidar.update(np.load(stem + "_lidar.npy"))
    det = TrafficLightDetector(cfg)
    if meta["light_box"]:
        x, y, w, h = meta["light_box"]
        state = meta["light"]
    else:
        cands = [c for c in det._candidates(img) if lidar.light_is_relevant(c[2], c[1], meta["dist_junction"])]
        _, state, (x, y, w, h) = max(cands, key=lambda c: c[0])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(img.shape[:2], np.uint8)
    for ranges in HSV_RANGES.values():
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    mask[det.horizon:] = 0
    cx, cy, r = x + w // 2, y + h // 2, 70
    crop = (slice(max(cy - r, 0), cy + r), slice(max(cx - r, 0), cx + r))
    u, v, ok = lidar.project(lidar.obstacles)
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.6), gridspec_kw={"width_ratios": [2.2, 1, 1, 2.2]})
    ax[0].imshow(rgb(img))
    ax[0].axhline(det.horizon, color="#facc15", lw=1, ls="--")
    ax[0].add_patch(plt.Rectangle((cx - r, cy - r), 2 * r, 2 * r, fill=False, ec="white", lw=1.2))
    ax[0].set_title("(a) Camera frame; search above the horizon")
    ax[1].imshow(rgb(img[crop]))
    ax[1].set_title("(b) Close-up")
    ax[2].imshow(mask[crop], cmap="gray")
    ax[2].set_title("(c) Colour mask")
    ax[3].imshow(rgb(img))
    sel = ok & (np.abs(u - cx) < 60) & (v > y - 20) & (v < 520)
    ax[3].scatter(u[sel], v[sel], s=5, c=lidar.obstacles[sel, 0], cmap="viridis", vmin=3, vmax=25)
    ax[3].add_patch(plt.Rectangle((x - 5, y - 5), w + 10, h + 10, fill=False, ec="red", lw=1.8))
    dist = meta["light_dist"] or lidar.light_distance((x, y, w, h), state)
    ax[3].set_title(f"(d) {state}, LiDAR range to pole {dist:.1f} m")
    ax[3].set_xlim(cx - 220, cx + 220)
    ax[3].set_ylim(min(cy + 300, cfg.cam_h - 1), max(cy - 110, 0))
    no_ticks(ax)
    save(fig, "traffic_light.png")


def fig_ped(stem):
    img = cv2.imread(stem + ".png")
    meta = json.load(open(stem + ".json"))
    pts = np.load(stem + "_lidar.npy")
    lidar = LidarProcessor(cfg, EGO_HALF_LENGTH, EGO_HALF_WIDTH)
    lidar.update(pts)
    path = np.array(meta["path"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax[0].imshow(rgb(img))
    for d in meta["dets"]:
        x0, y0, x1, y1 = d["box"]
        person = d["cls"] == 0
        color = "#e11d9b" if person else BLUE
        ax[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=1.8))
        if d["pos"]:
            label = f"{'person' if person else 'vehicle'} {np.hypot(*d['pos']):.1f} m, {d['hits']} LiDAR pts"
            ax[0].text(x0 if person else x0 - 40, (y0 - 8) if person else (y1 + 22), label,
                       color="white", fontsize=8, bbox=dict(fc=color, ec="none", pad=1.5))
    ax[0].set_title("(a) YOLOX detections with fused range")
    no_ticks(ax[:1])
    ax[1].scatter(pts[:, 1], pts[:, 0], s=0.4, c="#cbd5e1")
    ax[1].scatter(lidar.obstacles[:, 1], lidar.obstacles[:, 0], s=1.5, c="#f59e0b", label="obstacle band (0.4-1.8 m)")
    ax[1].plot(path[:, 1], path[:, 0], color=GREEN, lw=2, label="planned path")
    for d in meta["dets"]:
        if d["pos"] and d["cls"] == 0:
            ax[1].scatter([d["pos"][1]], [d["pos"][0]], s=140, facecolors="none", edgecolors="#e11d9b", lw=2)
    ax[1].scatter([], [], s=60, facecolors="none", edgecolors="#e11d9b", lw=2, label="person")
    ax[1].add_patch(plt.Rectangle((-EGO_HALF_WIDTH, -EGO_HALF_LENGTH), 2 * EGO_HALF_WIDTH, 2 * EGO_HALF_LENGTH, fc=INK))
    ax[1].set_xlim(-16, 12)
    ax[1].set_ylim(-4, 30)
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("right of car (m)")
    ax[1].set_ylabel("ahead of car (m)")
    ax[1].legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9, bbox_to_anchor=(-0.02, 1.0))
    ax[1].set_title("(b) LiDAR top view")
    save(fig, "pedestrian.png")


def load_run(stem):
    return list(csv.DictReader(open(stem + ".csv"))), json.load(open(stem + ".json"))


def fig_routes(stem, roads_json):
    rows, trips = load_run(stem)
    roads = json.load(open(roads_json))
    n_rows = int(np.ceil(len(trips) / 3))
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4.2 * n_rows))
    axes = np.ravel(axes)
    for ax, t in zip(axes, trips):
        for seg in roads:
            a = np.array(seg)
            ax.plot(a[:, 0], a[:, 1], color="#d1d5db", lw=2.5, solid_capstyle="round", zorder=1)
        r = np.array(t["route_xy"])
        ax.plot(r[:, 0], r[:, 1], color=BLUE, lw=4, alpha=0.45, zorder=2, label="planned (A*)")
        d = np.array([[float(q["x"]), float(q["y"])] for q in rows if int(q["trip"]) == t["trip"]])
        ax.plot(d[:, 0], d[:, 1], color=INK, lw=1.2, zorder=3, label="driven")
        ax.scatter(*t["start"], s=70, c=GREEN, zorder=4, label="start")
        ax.scatter(*t["goal"], s=90, c=RED, marker="*", zorder=4, label="goal")
        ax.set_title(f"Trip {t['trip']}: {t['planned_m']:.0f} m route, {t['time_s']:.0f} s")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in axes[len(trips):]:
        ax.axis("off")
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    save(fig, "routes.png")


def fig_speed(stem, trip_no):
    rows, _ = load_run(stem)
    q = [r for r in rows if int(r["trip"]) == trip_no]
    t = (np.array([int(r["frame"]) for r in q]) - int(q[0]["frame"])) * cfg.dt
    v = np.array([float(r["speed_kmh"]) for r in q])
    colors = {"CRUISE": GREY, "JUNCTION": BLUE, "CURVE": PURPLE, "RED LIGHT": RED, "PEDESTRIAN": "#e11d9b",
              "OBSTACLE": ORANGE, "DESTINATION": GREEN}
    decisions = [r["decision"] for r in q]
    fig, ax = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(t, v, color=INK, lw=1.4)
    start = 0
    for i in range(1, len(decisions) + 1):
        if i == len(decisions) or decisions[i] != decisions[start]:
            ax[0].axvspan(t[start], t[i - 1] + cfg.dt, color=colors[decisions[start]], alpha=0.28, lw=0)
            start = i
    for k, c in colors.items():
        if k in decisions:
            ax[0].fill_between([], [], color=c, alpha=0.4, label=k)
    ax[0].legend(ncol=7, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    ax[0].set_ylabel("speed (km/h)")
    ax[0].set_ylim(0, 34)
    camera = np.array([r["steer_mode"] == "CAMERA" for r in q], float)
    ax[1].fill_between(t, 0, camera, step="post", color=ORANGE, alpha=0.7, lw=0)
    ax[1].set_yticks([0, 1])
    ax[1].set_yticklabels(["map", "camera"])
    ax[1].set_ylabel("steering\nsource")
    ax[1].set_xlabel("time since trip start (s)")
    for a in ax:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    save(fig, "speed_profile.png")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"arch": lambda: fig_arch(),
     "lane": lambda: fig_lane(args[0]),
     "light": lambda: fig_light(args[0]),
     "ped": lambda: fig_ped(args[0]),
     "routes": lambda: fig_routes(args[0], args[1]),
     "speed": lambda: fig_speed(args[0], int(args[1]))}[cmd]()
