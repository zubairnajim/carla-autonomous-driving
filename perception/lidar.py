"""LiDAR processing: ground removal, obstacle check along the planned path,
and range for camera detections by projecting points into the image."""
import cv2
import numpy as np


class LidarProcessor:
    def __init__(self, cfg, half_length, half_width):
        self.cfg = cfg
        self.half_length = half_length
        self.half_width = half_width
        self.f = cfg.cam_w / (2 * np.tan(np.radians(cfg.cam_fov) / 2))
        self.obstacles = np.zeros((0, 3), np.float32)
        self.in_path = np.zeros((0, 3), np.float32)
        self.points = np.zeros((0, 3), np.float32)

    def update(self, points):
        """points: Nx3 in the LiDAR frame (x forward, y right, z up)."""
        cfg = self.cfg
        self.points = points
        height = points[:, 2] + cfg.lidar_z
        on_ego = (np.abs(points[:, 0]) < self.half_length + 0.3) & (np.abs(points[:, 1]) < self.half_width + 0.3)
        keep = (height > cfg.obst_min_h) & (height < cfg.obst_max_h) & ~on_ego
        self.obstacles = points[keep]

    def obstacle_distance(self, path):
        """Free distance (m, from the front bumper) along `path` (Mx2, vehicle frame)."""
        self.in_path = np.zeros((0, 3), np.float32)
        obs = self.obstacles[self.obstacles[:, 0] > -1.0]
        if len(obs) == 0 or len(path) < 2:
            return float("inf")
        arclen = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        d = np.linalg.norm(obs[:, None, :2] - path[None, :, :], axis=2)
        nearest = d.argmin(axis=1)
        inside = d[np.arange(len(obs)), nearest] < self.cfg.corridor_half_width
        along = arclen[nearest] + path[0, 0]          # path[0] may be slightly ahead/behind
        inside &= along > self.half_length
        if inside.sum() < self.cfg.obst_min_points:
            return float("inf")
        self.in_path = obs[inside]
        # robust nearest: ignore the closest couple of stray points
        return float(np.partition(along[inside], 2)[2] - self.half_length)

    def project(self, pts):
        """LiDAR points -> pixel coordinates of the RGB camera, plus validity mask."""
        cfg = self.cfg
        x = pts[:, 0] - cfg.cam_x
        z = pts[:, 2] + cfg.lidar_z - cfg.cam_z
        ok = x > 0.5
        xs = np.where(ok, x, 1.0)
        u = cfg.cam_w / 2 + self.f * pts[:, 1] / xs
        v = cfg.cam_h / 2 - self.f * z / xs
        ok &= (u >= 0) & (u < cfg.cam_w) & (v >= 0) & (v < cfg.cam_h)
        return u, v, ok

    def light_distance(self, box, state):
        """Forward distance (m, from vehicle centre) to a traffic light whose lit
        lamp is at `box` (x, y, w, h). A pinhole estimate from the lamp's assumed
        height picks which LiDAR returns on the pole below it to trust."""
        cfg = self.cfg
        x, y, w, h = box
        rise = max(cfg.cam_h / 2 - (y + h / 2), 1.0)
        pinhole = cfg.cam_x + self.f * (cfg.tl_lamp_heights[state] - cfg.cam_z) / rise
        u, v, ok = self.project(self.obstacles)
        sel = ok & (np.abs(u - (x + w / 2)) < max(w, 6)) & (v > y + h)
        fwd = self.obstacles[sel, 0]
        fwd = fwd[np.abs(fwd - pinhole) < 0.35 * pinhole]
        return float(np.median(fwd)) if len(fwd) >= 2 else float(pinhole)

    def light_is_relevant(self, box, state, dist_junction):
        """Geometric gate for lamp candidates: the signal for our lane stands on
        our side of the road, at or before the entry of the next junction."""
        cfg = self.cfg
        fwd = self.light_distance(box, state)
        right = (box[0] + box[2] / 2 - cfg.cam_w / 2) * (fwd - cfg.cam_x) / self.f
        return (cfg.tl_lateral_min < right < cfg.tl_lateral_max
                and fwd < min(dist_junction + cfg.tl_past_junction, cfg.tl_junction_range))

    def localize(self, detections):
        """Give each camera detection a position in the vehicle frame.

        Range comes first from camera geometry (where the box touches the
        ground, or the box height if it is cut off), then LiDAR returns inside
        the box refine it when they agree. LiDAR alone is not trusted: boxes
        also contain background, and CARLA's AI walkers are not always hit by
        ray-cast LiDAR."""
        if not detections:
            return
        cfg = self.cfg
        u, v, ok = self.project(self.obstacles)
        for det in detections:
            x0, y0, x1, y1 = det.box
            foot = y1 - cfg.cam_h / 2
            if y1 < cfg.cam_h - 4 and foot > 8:
                fwd_cam = self.f * cfg.cam_z / foot
            else:
                fwd_cam = self.f * (1.7 if det.cls == 0 else 1.5) / max(y1 - y0, 1)
            right = ((x0 + x1) / 2 - cfg.cam_w / 2) * fwd_cam / self.f
            fwd = fwd_cam + cfg.cam_x

            mx = 0.15 * (x1 - x0)
            sel = ok & (u > x0 + mx) & (u < x1 - mx) & (v > y0) & (v < y1)
            pts = self.obstacles[sel]
            pts = pts[np.abs(pts[:, 0] - fwd) < max(0.3 * fwd, 1.5)]
            det.lidar_hits = len(pts)
            if len(pts) >= 3:
                fwd, right = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
            det.position = (float(fwd), float(right))
            det.distance = max(float(np.hypot(fwd, right)) - self.half_length, 0.0)

    def draw_bev(self, size=300, span=30.0, path=None, detections=()):
        """Top-down LiDAR view: ego at bottom centre, `span` metres ahead."""
        img = np.zeros((size, size, 3), np.uint8)
        s = size / span

        def to_px(fwd, right):
            return (np.asarray(right) * s + size / 2).astype(int), (size - 1 - (np.asarray(fwd) + 3.0) * s).astype(int)

        def scatter(pts, color):
            if len(pts) == 0:
                return
            c, r = to_px(pts[:, 0], pts[:, 1])
            ok = (c >= 0) & (c < size) & (r >= 0) & (r < size)
            img[r[ok], c[ok]] = color

        scatter(self.points, (70, 70, 70))
        scatter(self.obstacles, (230, 200, 60))
        scatter(self.in_path, (0, 0, 255))
        if path is not None and len(path) > 1:
            c, r = to_px(path[:, 0], path[:, 1])
            cv2.polylines(img, [np.stack([c, r], axis=1).astype(np.int32)], False, (0, 200, 0), 1)
        for d in detections:
            if d.position is not None and d.cls == 0:
                c, r = to_px(d.position[0], d.position[1])
                cv2.circle(img, (int(c), int(r)), 5, (255, 0, 255), 2)
        c0, r0 = to_px(self.half_length, -self.half_width)
        c1, r1 = to_px(-self.half_length, self.half_width)
        cv2.rectangle(img, (int(c0), int(r0)), (int(c1), int(r1)), (255, 255, 255), 1)
        cv2.putText(img, "LiDAR", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return img
