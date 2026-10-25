"""Pedestrian / vehicle detection with YOLOX running in OpenCV's DNN module
(model from the official OpenCV Zoo, COCO classes)."""
import cv2
import numpy as np

PERSON = 0
CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
INPUT = 640


class Detection:
    def __init__(self, box, score, cls):
        self.box = box            # x0, y0, x1, y1 in image pixels
        self.score = score
        self.cls = cls
        self.position = None      # (forward, right) metres in vehicle frame, from LiDAR
        self.distance = None
        self.lidar_hits = 0

    @property
    def name(self):
        return CLASS_NAMES[self.cls]


class ObjectDetector:
    def __init__(self, cfg, model_path):
        self.cfg = cfg
        self.net = cv2.dnn.readNet(model_path)
        grids, strides = [], []
        for s in (8, 16, 32):
            n = INPUT // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack([xv, yv], axis=2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grids = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        ratio = min(INPUT / h, INPUT / w)
        resized = cv2.resize(bgr, (int(w * ratio), int(h * ratio)))
        padded = np.full((INPUT, INPUT, 3), 114, np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        blob = cv2.dnn.blobFromImage(padded, 1.0, (INPUT, INPUT), swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward()[0]

        xy = (out[:, :2] + self.grids) * self.strides
        wh = np.exp(out[:, 2:4]) * self.strides
        scores_all = out[:, 4:5] * out[:, 5:]
        cls = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(len(cls)), cls]
        keep = (scores > self.cfg.yolo_conf) & np.isin(cls, list(CLASS_NAMES))
        if not keep.any():
            return []
        xy, wh, cls, scores = xy[keep], wh[keep], cls[keep], scores[keep]
        boxes = np.concatenate([xy - wh / 2, wh], axis=1) / ratio      # x, y, w, h
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.cfg.yolo_conf, self.cfg.yolo_nms)
        dets = []
        for i in np.array(idx).flatten():
            x, y, bw, bh = boxes[i]
            box = (max(int(x), 0), max(int(y), 0), min(int(x + bw), w - 1), min(int(y + bh), h - 1))
            dets.append(Detection(box, float(scores[i]), int(cls[i])))
        return dets

    @staticmethod
    def draw(frame, dets):
        for d in dets:
            x0, y0, x1, y1 = d.box
            color = (255, 0, 255) if d.cls == PERSON else (255, 160, 0)
            label = d.name if d.distance is None else f"{d.name} {d.distance:.1f}m"
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            cv2.putText(frame, label, (x0, max(y0 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
