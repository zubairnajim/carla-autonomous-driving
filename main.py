"""Autonomous driving demo for CARLA: OpenCV lane following, traffic-light
obedience, pedestrian detection (YOLOX via cv2.dnn) and LiDAR obstacle checks.

    python main.py                 # wander until Q / Esc is pressed
    python main.py --navigate      # click a destination on the map; the car drives there
    python main.py --start 5 --dest 40      # one trip between two spawn points
    python main.py --trips 5       # five trips to random destinations, then a table
"""
import sys

sys.dont_write_bytecode = True      # no __pycache__ folders

import argparse
import csv
import json
import math
import os
import queue
import random
import time

import carla
import cv2
import numpy as np

from carla_env import CarlaEnv
from config import Config
from control import Planner
from perception.lanes import LaneDetector
from perception.lidar import LidarProcessor
from perception.objects import PERSON, ObjectDetector
from perception.traffic_lights import DRAW_COLORS, NONE, RED, TrafficLightDetector
from navigation import MapView, RoadGraph
from route import Route

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_W = 300
MAP_W = 600
WINDOW = "CARLA autonomous driving"


class GroundTruthLights:
    """Simulator's own light states, used only to score the camera detector."""

    def __init__(self, world):
        self.lights = []
        for tl in world.get_actors().filter("traffic.traffic_light"):
            vol = tl.trigger_volume
            inv = np.array(tl.get_transform().get_inverse_matrix())
            centre = np.array([vol.location.x, vol.location.y])
            half = np.array([vol.extent.x, vol.extent.y]) + 0.5
            self.lights.append((tl, inv, centre, half))
        self.was_at_light = False
        self.last_state = NONE

    def state(self, route):
        """State of the light whose trigger box the route crosses before the next junction."""
        for wp in route.wps:
            if wp.is_junction:
                break
            loc = wp.transform.location
            p = np.array([loc.x, loc.y, loc.z, 1.0])
            for tl, inv, centre, half in self.lights:
                if np.all(np.abs((inv @ p)[:2] - centre) < half):
                    return str(tl.get_state()).split(".")[-1].upper()
        return NONE

    def ran_red(self, ego, speed):
        """True on the frame the car drives out of a light's trigger box while it is red."""
        at_light = ego.is_at_traffic_light()
        ran = self.was_at_light and not at_light and self.last_state == RED and speed > 1.0
        if at_light:
            self.last_state = str(ego.get_traffic_light_state()).split(".")[-1].upper()
        self.was_at_light = at_light
        return ran


def build_hud(frame, lane_det, lanes, lidar, path, dets, tl_state, gt_state, v, control, info, fps):
    h = frame.shape[0]
    panel = np.zeros((h, PANEL_W, 3), np.uint8)
    panel[:300] = lidar.draw_bev(300, 30.0, path, dets)

    mask = cv2.cvtColor(lanes.mask * 255, cv2.COLOR_GRAY2BGR)
    mask = cv2.resize(mask, (100, 200), interpolation=cv2.INTER_NEAREST)
    panel[310:510, 190:290] = mask
    cv2.putText(panel, "lane BEV", (190, 525), FONT, 0.4, (200, 200, 200), 1)

    people = sum(d.cls == PERSON for d in dets)
    lines = [
        (f"{v * 3.6:4.1f} km/h", (255, 255, 255)),
        (f"-> {info['v_ref'] * 3.6:4.1f}", (180, 180, 180)),
        (info["reason"], (0, 200, 255) if info["reason"] != "CRUISE" else (0, 255, 0)),
        (f"steer: {info['steer_mode']}", (0, 255, 255) if info["steer_mode"] == "CAMERA" else (255, 160, 0)),
        (f"light: {tl_state}", DRAW_COLORS[tl_state]),
        (f"people: {people}", (255, 0, 255)),
        (f"sim {fps:4.1f} fps", (140, 140, 140)),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(panel, text, (8, 330 + i * 26), FONT, 0.55, color, 1, cv2.LINE_AA)
    cv2.putText(panel, f"T {control.throttle:.2f}  B {control.brake:.2f}  S {control.steer:+.2f}",
                (8, h - 40), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(panel, f"sim says light: {gt_state}", (8, h - 16), FONT, 0.42, (120, 120, 120), 1, cv2.LINE_AA)
    return np.hstack([frame, panel])


def run(args):
    cfg = Config()
    if args.town:
        cfg.town = args.town
    if args.seed is not None:
        cfg.seed = args.seed
    if args.start is not None:
        cfg.start_index = args.start
    nav_mode = args.navigate or args.dest is not None or args.trips > 0
    env = CarlaEnv(cfg)
    writer = telemetry = telemetry_file = None
    trips = []
    stats = dict(frames=0, camera_steer=0, tl_seen=0, tl_agree=0, red_runs=0, distance=0.0,
                 people_frames=0, stops_for_people=0, stops_for_light=0)
    try:
        ext = env.ego.bounding_box.extent
        lane_det = LaneDetector(cfg)
        tl_det = TrafficLightDetector(cfg)
        obj_det = ObjectDetector(cfg, os.path.join(HERE, cfg.yolo_model))
        lidar = LidarProcessor(cfg, ext.x, ext.y)
        planner = Planner(cfg, env.max_steer_rad, ext.x)
        gt_lights = GroundTruthLights(env.world)
        rng = random.Random(cfg.seed)
        graph = RoadGraph(env.map) if nav_mode else None
        map_view = MapView(graph, MAP_W) if nav_mode else None
        trip, trail, clicked = None, [], []
        goals_left = args.trips

        def idle_route():
            return Route(env.map, None, waypoints=[env.map.get_waypoint(env.ego.get_location())])

        def start_trip(goal_location):
            """Plan from the car's position to `goal_location` and return (route, trip record)."""
            t0 = time.perf_counter()
            plan = graph.plan(env.ego.get_location(), goal_location)
            plan_ms = (time.perf_counter() - t0) * 1000
            if plan is None or len(plan[0]) < 2:
                print("No drivable route to that point.")
                return idle_route(), None
            wps, length, expanded = plan
            start = env.ego.get_location()
            goal = wps[-1].transform.location
            print(f"Trip {len(trips) + 1}: route {length:.0f} m (straight line {start.distance(goal):.0f} m), "
                  f"A* expanded {expanded} of {len(graph.segments)} segments in {plan_ms:.1f} ms", flush=True)
            record = dict(trip=len(trips) + 1, planned_m=round(length, 1), straight_m=round(start.distance(goal), 1),
                          expanded=expanded, plan_ms=round(plan_ms, 2), start=(start.x, start.y), goal=(goal.x, goal.y),
                          frame0=stats["frames"], driven0=stats["distance"], collisions0=len(env.collisions),
                          red0=stats["red_runs"], light_stops0=stats["stops_for_light"], ped_stops0=stats["stops_for_people"],
                          route_xy=[(w.transform.location.x, w.transform.location.y) for w in wps[::3]])
            trail.clear()
            return Route(env.map, None, waypoints=wps), record

        def random_goal():
            here = env.ego.get_location()
            far = [p.location for p in env.spawn_points if p.location.distance(here) > 120.0]
            return rng.choice(far)

        route = idle_route() if nav_mode else Route(env.map, env.ego.get_location(), seed=cfg.seed)

        for _ in range(20):          # let the car settle on its wheels
            env.tick()
        if args.video:
            writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / cfg.dt,
                                     (cfg.cam_w + PANEL_W + (MAP_W if nav_mode else 0), cfg.cam_h))
        if args.snapshots:
            os.makedirs(args.snapshots, exist_ok=True)
        if args.telemetry:
            telemetry_file = open(args.telemetry, "w", newline="")
            telemetry = csv.writer(telemetry_file)
            telemetry.writerow(["frame", "trip", "x", "y", "speed_kmh", "target_kmh", "decision", "steer_mode",
                                "light_camera", "light_sim", "people", "free_m", "remaining_m", "lane_conf"])
        if nav_mode and not args.no_display:
            cv2.namedWindow(WINDOW)
            cv2.setMouseCallback(WINDOW, lambda ev, x, y, *_: clicked.append((x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)
        if args.dest is not None:
            route, trip = start_trip(env.spawn_points[args.dest % len(env.spawn_points)].location)
        elif goals_left:
            goals_left -= 1
            route, trip = start_trip(random_goal())
        elif args.navigate:
            print("Click a point on the map to set the destination.")

        dets, prev_stopped, n_collisions = [], True, 0
        t_last, fps = time.time(), 0.0
        print(f"Driving in {cfg.town}: {len(env.actors)} actors, {env.n_walkers_spawned} walkers. Q / Esc to quit.", flush=True)
        while args.frames == 0 or stats["frames"] < args.frames:
            data = env.tick()
            frame = env.image_to_bgr(data["camera"])
            tf = env.ego.get_transform()
            vel = env.ego.get_velocity()
            v = math.hypot(vel.x, vel.y)

            route.update(tf.location)
            path = route.path_vehicle_frame(tf)[:40]

            lidar.update(env.lidar_to_points(data["lidar"]))
            dist_junction = route.distance_to_junction()
            lanes = lane_det.detect(frame)
            tl_state = tl_det.detect(frame, gate=lambda box, state: lidar.light_is_relevant(box, state, dist_junction))
            if stats["frames"] % cfg.yolo_every == 0:
                dets = obj_det.detect(frame)
            lidar.localize(dets)
            free = lidar.obstacle_distance(path)
            light_dist = lidar.light_distance(tl_det.box, tl_det.raw) if tl_det.box is not None and tl_state != NONE else None

            if args.dump and stats["frames"] % args.dump_every == 0:
                # raw sensor data for offline figures / debugging
                os.makedirs(args.dump, exist_ok=True)
                stem = os.path.join(args.dump, f"{stats['frames']:06d}")
                cv2.imwrite(stem + ".png", frame)
                np.save(stem + "_lidar.npy", lidar.points)
                with open(stem + ".json", "w") as f:
                    json.dump(dict(light=tl_state, light_box=[int(b) for b in tl_det.box] if tl_det.box is not None else None,
                                   light_dist=light_dist, dist_junction=min(dist_junction, 999.0), speed=v,
                                   path=path.tolist(), dets=[dict(box=d.box, cls=d.cls, pos=d.position, hits=d.lidar_hits)
                                                             for d in dets]), f)

            control, info = planner.step(v, lanes, path, route, tl_state, light_dist, dets, free)
            env.ego.apply_control(control)
            env.follow_with_spectator()

            # ---- bookkeeping against simulator ground truth
            gt = gt_lights.state(route)
            stats["frames"] += 1
            stats["distance"] += v * cfg.dt
            stats["camera_steer"] += info["steer_mode"] == "CAMERA"
            stats["people_frames"] += any(d.cls == PERSON for d in dets)
            if not info["in_junction"] and info["dist_junction"] < cfg.tl_junction_range and gt != NONE:
                stats["tl_seen"] += 1
                stats["tl_agree"] += tl_state == gt
            if gt_lights.ran_red(env.ego, v):
                stats["red_runs"] += 1
                print(f"[frame {stats['frames']}] crossed a stop line on RED (camera said {tl_state})")
            stopped = v < 0.3
            if stopped and not prev_stopped:
                stats["stops_for_people"] += info["reason"] == "PEDESTRIAN"
                stats["stops_for_light"] += info["reason"] == "RED LIGHT"
            if len(env.collisions) > n_collisions:
                n_collisions = len(env.collisions)
                print(f"[frame {stats['frames']}] COLLISION with {env.collisions[-1][1]} at {v * 3.6:.1f} km/h "
                      f"({info['reason']}, free={free:.1f}m)", flush=True)
                if args.snapshots:
                    cv2.imwrite(os.path.join(args.snapshots, f"collision_{stats['frames']:06d}.jpg"), frame)
            prev_stopped = stopped
            if telemetry:
                telemetry.writerow([stats["frames"], trip["trip"] if trip else 0, round(tf.location.x, 2), round(tf.location.y, 2),
                                    round(v * 3.6, 2), round(info["v_ref"] * 3.6, 2), info["reason"], info["steer_mode"],
                                    tl_state, gt, sum(d.cls == PERSON for d in dets),
                                    round(min(free, 99.0), 1), round(min(route.remaining, 9999.0), 1), lanes.confidence])

            # ---- navigation: arrival, next destination, map clicks
            if nav_mode:
                if stats["frames"] % 10 == 0:
                    trail.append(tf.location)
                if trip and route.remaining < 1.0 and stopped:
                    goal = trip["goal"]
                    trip.update(time_s=round((stats["frames"] - trip.pop("frame0")) * cfg.dt, 1),
                                driven_m=round(stats["distance"] - trip.pop("driven0"), 1),
                                arrival_error_m=round(math.hypot(tf.location.x - goal[0], tf.location.y - goal[1]), 2),
                                collisions=len(env.collisions) - trip.pop("collisions0"),
                                red_lights_run=stats["red_runs"] - trip.pop("red0"),
                                red_light_stops=stats["stops_for_light"] - trip.pop("light_stops0"),
                                pedestrian_stops=stats["stops_for_people"] - trip.pop("ped_stops0"))
                    print(f"Arrived: {trip['driven_m']:.0f} m in {trip['time_s']:.0f} s, {trip['arrival_error_m']:.2f} m from "
                          f"the goal, collisions {trip['collisions']}, red lights run {trip['red_lights_run']}", flush=True)
                    trips.append(trip)
                    trip, route = None, idle_route()
                    if goals_left:
                        goals_left -= 1
                        route, trip = start_trip(random_goal())
                    elif not args.navigate:
                        break
                for x, y in clicked:
                    if x >= cfg.cam_w + PANEL_W:
                        wx, wy = map_view.to_world(x - cfg.cam_w - PANEL_W, y)
                        route, trip = start_trip(carla.Location(x=float(wx), y=float(wy)))
                clicked.clear()

            # ---- visualisation
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-3)
            t_last = now
            if not args.no_display or writer or args.snapshots:
                lane_det.draw(frame, lanes)
                tl_det.draw(frame)
                obj_det.draw(frame, dets)
                hud = build_hud(frame, lane_det, lanes, lidar, path, dets, tl_state, gt, v, control, info, fps)
                if nav_mode:
                    if trip:
                        eta = route.remaining / max(cfg.cruise_kmh / 3.6 * 0.6, 0.1)
                        lines = (f"trip {trip['trip']}: {trip['planned_m']:.0f} m planned",
                                 f"to go: {route.remaining:.0f} m  (~{eta:.0f} s)")
                    else:
                        lines = ("click the map to set a destination",) if args.navigate else ()
                    goal = carla.Location(x=trip["goal"][0], y=trip["goal"][1]) if trip else None
                    hud = np.hstack([hud, map_view.draw(tf, route.wps if trip else None, goal, trail, lines)])
                if writer:
                    writer.write(hud)
                if args.snapshots and stats["frames"] % args.snapshot_every == 0:
                    cv2.imwrite(os.path.join(args.snapshots, f"{stats['frames']:06d}.jpg"), hud)
                if not args.no_display:
                    cv2.imshow(WINDOW, hud)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
            if args.log_every and stats["frames"] % args.log_every == 0:
                print(f"[{stats['frames']:5d}] {v * 3.6:4.1f} km/h  {info['reason']:<10} steer={info['steer_mode']:<6} "
                      f"light cam={tl_state:<6} sim={gt:<6} ld={light_dist or 0:4.1f} junc={info['dist_junction']:5.1f}m free={free:5.1f}m "
                      f"lane_conf={lanes.confidence:.1f} off={path[0][1]:+.2f} tgt=({info['target'][0]:.1f},{info['target'][1]:+.2f}) "
                      f"S={control.steer:+.2f} fps={fps:.1f}", flush=True)
    except KeyboardInterrupt:
        pass
    except queue.Empty:
        print("The simulator stopped sending sensor data (server crashed or closed).")
    finally:
        if writer:
            writer.release()
        if telemetry_file:
            telemetry_file.close()
        cv2.destroyAllWindows()
        collisions = list(env.collisions)
        env.close()

    n = max(stats["frames"], 1)
    print("\n===== run summary =====")
    print(f"frames: {stats['frames']}  ({stats['frames'] * cfg.dt:.0f} s simulated)   distance: {stats['distance']:.0f} m")
    print(f"steering from camera lanes: {100 * stats['camera_steer'] / n:.0f}% of frames (rest: map route)")
    if stats["tl_seen"]:
        print(f"traffic light camera vs simulator: {100 * stats['tl_agree'] / stats['tl_seen']:.0f}% "
              f"agreement over {stats['tl_seen']} frames")
    print(f"stops for red light: {stats['stops_for_light']}   red lights run: {stats['red_runs']}")
    print(f"frames with a person detected: {stats['people_frames']}   stops for pedestrians: {stats['stops_for_people']}")
    print(f"collisions: {len(collisions)} {sorted(set(c[1] for c in collisions))}")
    if trips:
        print("\n===== trips =====")
        print("trip  planned  driven  straight   time  arrival err  A* expanded  plan ms  red stops  red run  collisions")
        for t in trips:
            print(f"{t['trip']:>4}  {t['planned_m']:>6.0f}m {t['driven_m']:>6.0f}m  {t['straight_m']:>6.0f}m  {t['time_s']:>4.0f}s "
                  f"{t['arrival_error_m']:>10.2f}m  {t['expanded']:>11}  {t['plan_ms']:>7.1f}  {t['red_light_stops']:>9}  "
                  f"{t['red_lights_run']:>7}  {t['collisions']:>10}")
        if args.trips_json:
            with open(args.trips_json, "w") as f:
                json.dump(trips, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--town", help="map to load (default from config.py)")
    ap.add_argument("--seed", type=int, help="random seed: start position, route, traffic")
    ap.add_argument("--navigate", action="store_true", help="click destinations on the map")
    ap.add_argument("--start", type=int, help="spawn-point index to start from")
    ap.add_argument("--dest", type=int, help="spawn-point index to drive to, then stop")
    ap.add_argument("--trips", type=int, default=0, help="drive N trips to random destinations")
    ap.add_argument("--telemetry", help="write per-frame data to this .csv file")
    ap.add_argument("--trips-json", help="write the trip table to this .json file")
    ap.add_argument("--dump", help="folder for raw camera / LiDAR dumps (for figures and debugging)")
    ap.add_argument("--dump-every", type=int, default=50)
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until Q)")
    ap.add_argument("--no-display", action="store_true", help="do not open the OpenCV window")
    ap.add_argument("--video", help="write the HUD to this .mp4 file")
    ap.add_argument("--snapshots", help="folder to save periodic HUD frames into")
    ap.add_argument("--snapshot-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=0, help="print a status line every N frames")
    run(ap.parse_args())
