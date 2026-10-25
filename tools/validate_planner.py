"""Check the A* route planner against an independent Dijkstra search.

Needs a running CARLA server. Plans routes between random pairs of the map's
spawn points and reports failures, length differences, nodes expanded and
planning time. Optionally exports the road polylines used by make_figures.py.

    python tools/validate_planner.py --pairs 1000
    python tools/validate_planner.py --roads runs/roads.json
"""
import argparse
import heapq
import json
import os
import random
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carla
import numpy as np

from navigation import RoadGraph


def dijkstra_length(graph, start, goal):
    """Uninformed reference search over the same lane-segment graph."""
    s_seg, s_idx = graph.locate(start)
    g_seg, g_idx = graph.locate(goal)
    if s_seg == g_seg and g_idx >= s_idx:
        return (g_idx - s_idx) * graph.step, 0
    dist = {s_seg: (len(graph.segments[s_seg]) - 1 - s_idx) * graph.step}
    heap, best, expanded = [(dist[s_seg], s_seg)], float("inf"), 0
    while heap:
        d, u = heapq.heappop(heap)
        if d >= best:
            break
        if d > dist[u]:
            continue
        expanded += 1
        for v in graph.successors[u]:
            if v == g_seg:
                best = min(best, d + g_idx * graph.step)
            nd = d + graph.lengths[v]
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return best, expanded


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--roads", help="also write road polylines to this .json file")
    args = ap.parse_args()

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(60.0)
    carla_map = client.get_world().get_map()
    t0 = time.perf_counter()
    graph = RoadGraph(carla_map)
    print(f"{carla_map.name}: {len(graph.segments)} segments, {sum(map(len, graph.successors))} edges, "
          f"{sum(graph.lengths) / 1000:.1f} km of lane, built in {(time.perf_counter() - t0) * 1000:.0f} ms")
    if args.roads:
        json.dump([[(w.transform.location.x, w.transform.location.y) for w in s[::2] + s[-1:]] for s in graph.segments],
                  open(args.roads, "w"))

    points = carla_map.get_spawn_points()
    pairs = [(a, b) for a in points for b in points if a is not b]
    random.Random(args.seed).shuffle(pairs)
    failed, diffs, astar_n, dijkstra_n, ms = 0, [], [], [], []
    for a, b in pairs[:args.pairs]:
        t0 = time.perf_counter()
        result = graph.plan(a.location, b.location)
        ms.append((time.perf_counter() - t0) * 1000)
        if result is None:
            failed += 1
            continue
        ref, n = dijkstra_length(graph, a.location, b.location)
        diffs.append(abs(result[1] - ref))
        astar_n.append(result[2])
        dijkstra_n.append(n)
    print(f"pairs: {args.pairs}   no route found: {failed}")
    print(f"largest |A* - Dijkstra| route length: {max(diffs):.3f} m")
    print(f"segments expanded: A* {np.mean(astar_n):.1f}, Dijkstra {np.mean(dijkstra_n):.1f} (mean)")
    print(f"planning time: {np.mean(ms):.2f} ms mean, {max(ms):.2f} ms max")


if __name__ == "__main__":
    main()
