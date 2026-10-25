"""Point-to-point navigation: a graph of the map's lane segments, A* search for
the shortest drivable route, and a top-down map view for picking a destination."""
import heapq
import math

import cv2
import numpy as np


def _key(wp):
    loc = wp.transform.location
    return round(loc.x), round(loc.y)


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


class RoadGraph:
    """Directed graph whose nodes are lane segments (one lane of one road, or
    one path through a junction). An edge joins a segment to each segment that
    starts where it ends, so every path in the graph is legally drivable."""

    def __init__(self, carla_map, step=1.0):
        self.map = carla_map
        self.step = step
        self.segments = []          # list of waypoint lists, sampled every `step` metres
        self.lengths = []
        self.by_lane = {}
        by_entry, exits = {}, []
        for entry, exit_wp in carla_map.get_topology():
            wps = [entry] + entry.next_until_lane_end(step)
            idx = len(self.segments)
            self.segments.append(wps)
            self.lengths.append((len(wps) - 1) * step)
            self.by_lane[(entry.road_id, entry.section_id, entry.lane_id)] = idx
            by_entry.setdefault(_key(entry), []).append(idx)
            exits.append(_key(exit_wp))
        self.successors = [by_entry.get(k, []) for k in exits]

    def locate(self, location):
        """(segment index, index of the nearest waypoint in it) for a world location."""
        wp = self.map.get_waypoint(location)
        seg = self.by_lane.get((wp.road_id, wp.section_id, wp.lane_id))
        if seg is None:             # not a topology lane: fall back to the nearest segment
            seg = min(range(len(self.segments)),
                      key=lambda i: min(_dist(w.transform.location, location) for w in self.segments[i]))
        d = [_dist(w.transform.location, location) for w in self.segments[seg]]
        return seg, int(np.argmin(d))

    def plan(self, start_location, goal_location):
        """Shortest route as (waypoints, length in metres, segments expanded) or None.

        A*: g = metres driven to the end of a segment, h = straight-line distance
        from there to the goal (never an overestimate, so the result is optimal)."""
        s_seg, s_idx = self.locate(start_location)
        g_seg, g_idx = self.locate(goal_location)
        goal_loc = self.segments[g_seg][g_idx].transform.location
        if s_seg == g_seg and g_idx >= s_idx:
            wps = self.segments[s_seg][s_idx:g_idx + 1]
            return wps, (len(wps) - 1) * self.step, 0

        def h(seg):
            return _dist(self.segments[seg][-1].transform.location, goal_loc)

        g_start = (len(self.segments[s_seg]) - 1 - s_idx) * self.step
        best = {s_seg: g_start}
        parent = {}
        heap = [(g_start + h(s_seg), g_start, s_seg)]
        goal_cost, goal_parent, expanded = float("inf"), None, 0
        while heap:
            f, g, seg = heapq.heappop(heap)
            if f >= goal_cost:
                break
            if g > best.get(seg, float("inf")):
                continue
            expanded += 1
            for nxt in self.successors[seg]:
                if nxt == g_seg and g + g_idx * self.step < goal_cost:
                    goal_cost, goal_parent = g + g_idx * self.step, seg
                g_next = g + self.lengths[nxt]
                if g_next < best.get(nxt, float("inf")):
                    best[nxt], parent[nxt] = g_next, seg
                    heapq.heappush(heap, (g_next + h(nxt), g_next, nxt))
        if goal_parent is None:
            return None

        chain, seg = [], goal_parent
        while seg != s_seg:
            chain.append(seg)
            seg = parent[seg]
        wps = list(self.segments[s_seg][s_idx:])
        for seg in reversed(chain):
            wps += self.segments[seg][1:] if _dist(wps[-1].transform.location,
                                                   self.segments[seg][0].transform.location) < 0.5 else self.segments[seg]
        wps += self.segments[g_seg][:g_idx + 1]
        return wps, goal_cost, expanded


class MapView:
    """Top-down drawing of the road network with the route, car and destination."""

    def __init__(self, graph, size=600, margin=30):
        self.size = size
        pts = np.array([[w.transform.location.x, w.transform.location.y] for s in graph.segments for w in s])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        self.scale = (size - 2 * margin) / float((hi - lo).max())
        self.origin = (lo + hi) / 2
        self.base = np.full((size, size, 3), 28, np.uint8)
        lane_px = max(int(3.5 * self.scale), 2)
        for seg in graph.segments:
            cv2.polylines(self.base, [self.to_px([w.transform.location for w in seg])], False, (95, 95, 95), lane_px, cv2.LINE_AA)
        cv2.putText(self.base, "NAVIGATION", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    def to_px(self, locations):
        xy = np.array([[l.x, l.y] for l in locations], dtype=np.float64)
        return ((xy - self.origin) * self.scale + self.size / 2).astype(np.int32)

    def to_world(self, px, py):
        return (np.array([px, py], dtype=np.float64) - self.size / 2) / self.scale + self.origin

    def draw(self, ego_transform, route_wps=None, goal=None, trail=None, lines=()):
        img = self.base.copy()
        if route_wps is not None and len(route_wps) > 1:
            cv2.polylines(img, [self.to_px([w.transform.location for w in route_wps])], False, (255, 170, 40), 3, cv2.LINE_AA)
        if trail is not None and len(trail) > 1:
            cv2.polylines(img, [self.to_px(trail)], False, (90, 220, 90), 2, cv2.LINE_AA)
        if goal is not None:
            gx, gy = self.to_px([goal])[0]
            cv2.circle(img, (int(gx), int(gy)), 8, (60, 60, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (int(gx), int(gy)), 8, (255, 255, 255), 1, cv2.LINE_AA)
        ex, ey = self.to_px([ego_transform.location])[0]
        yaw = math.radians(ego_transform.rotation.yaw)
        tri = np.array([[10, 0], [-6, 5], [-6, -5]], dtype=np.float64)
        rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        cv2.fillPoly(img, [(tri @ rot.T + [ex, ey]).astype(np.int32)], (255, 255, 255), cv2.LINE_AA)
        for i, text in enumerate(lines):
            cv2.putText(img, text, (10, self.size - 14 - 20 * (len(lines) - 1 - i)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (230, 230, 230), 1, cv2.LINE_AA)
        return img
