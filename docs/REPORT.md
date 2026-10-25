# A Camera-and-LiDAR Autonomous Driving Stack in CARLA with Classical Computer Vision

Technical report, October 2026.

## Abstract

This report describes a self-driving stack for the CARLA simulator that follows lanes, obeys traffic lights, yields to pedestrians and navigates to a chosen destination, using one RGB camera and one LiDAR. Lane and traffic-light perception use classical OpenCV image processing rather than learned models; pedestrians are detected with a YOLOX network run inside OpenCV; routes are planned with A* search over the map's lane graph.

In the final evaluation the car completed six of six navigated trips in live traffic, covering 2,592 m with no collisions and no red lights run, stopping on average 0.16 m from each goal. It made 16 red-light stops, and its camera reading of the light matched the simulator's ground truth in 93% of frames on signal approaches. The A* planner matched an independent Dijkstra search on 1,000 of 1,000 random routes while expanding 39% fewer nodes.

The system runs on a single laptop with a 4 GB GPU. Reaching that result required diagnosing eleven distinct failures, including a simulator behaviour that makes pedestrians invisible to LiDAR and caused a collision during development. Those failures and their fixes are documented in section 9, and the limits of the evidence (one map, in daylight, at up to 30 km/h) in section 10.

## Contents

1. [System overview](#1-system-overview)
2. [Experimental setup](#2-experimental-setup)
3. [Lane detection](#3-lane-detection)
4. [Traffic-light recognition](#4-traffic-light-recognition)
5. [Pedestrians and obstacles](#5-pedestrians-and-obstacles)
6. [Planning and control](#6-planning-and-control)
7. [Point-to-point navigation](#7-point-to-point-navigation)
8. [Results](#8-results)
9. [Failures found and fixed](#9-failures-found-and-fixed)
10. [Limitations and next steps](#10-limitations-and-next-steps)
11. [Reproducibility](#11-reproducibility)

## 1. System overview

The stack is a conventional sense-plan-act pipeline: two sensors feed four perception modules, a rule-based behaviour planner picks a target speed, and a geometric controller drives the vehicle.

![Architecture of the driving stack](figures/architecture.png)

*Figure 1. Data flow. The map supplies the route, junction locations and a steering fallback; light state, people, obstacles and the lane on marked roads all come from the sensors.*

![Live display during a navigated trip](screenshots/hud_navigation.jpg)

*Figure 2. The live display during a navigated trip. Left: camera with the detected lane, a pedestrian ranged at 5.1 m and the signal state. Centre: LiDAR top view and the current decision. Right: navigation map with the remaining route (blue), distance driven (green) and destination (red).*

Three design decisions shape everything that follows:

- **Classical vision where it suffices.** Lanes and traffic lights are found with OpenCV image processing, which is transparent and needs no training data. A neural detector is used only for people and vehicles, where classical methods are weak.
- **Sensors cross-check each other.** The camera says what something is; LiDAR says how far away it is. Neither is trusted alone for a safety decision.
- **Ground truth is for scoring only.** The simulator knows every light state and actor position. The driving code never reads them; a separate scorer does, so each run reports its own error rates.

The whole stack runs in one Python process at roughly 15 to 20 simulation frames per second on the test laptop, with the simulator in synchronous mode so that slow frames never cause missed sensor data.

## 2. Experimental setup

All experiments ran on one laptop, with the simulator and the driving stack sharing the same machine.

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop, 4 GB VRAM, driver 616.92 |
| Memory / OS | 16 GB RAM, Windows 11 Pro |
| Simulator | CARLA 0.9.16 (Unreal Engine 4.26), map Town02 |
| Simulation | Synchronous mode, fixed step 0.05 s (20 Hz) |
| Ego vehicle | Tesla Model 3 blueprint |
| Camera | RGB, 800 x 600, 100° horizontal field of view, 1.6 m above ground |
| LiDAR | 32 channels, 50 m range, 300,000 points/s, 20 Hz rotation, roof mounted at 1.9 m |
| Traffic | 15 autopilot vehicles and 40 requested AI pedestrians (23 spawned) |
| Software | Python 3.12, OpenCV 5.0, NumPy 2.5, CARLA client 0.9.16 |

### 2.1 Getting a stable simulator

The first obstacle was not perception but the simulator itself, which crashed within seconds to minutes. Each configuration was tested by driving the same route with the camera attached and recording how long the server survived.

| Rendering backend | Quality | Outcome |
| --- | --- | --- |
| DirectX 11 | Epic | Server died on the first camera frame |
| Vulkan | Low | Crashed after about 150 frames, at the same spot on the map every run |
| Vulkan | Epic | Ran 4,760 frames (about 4 minutes), then crashed |
| DirectX 12 | Epic | Stable: 6,000-frame runs completed repeatedly |

Two further findings came out of this study:

- The default map (Town10HD) with CARLA's 4,000 MB texture pool filled 3.8 of the 4 GB of VRAM. Switching the default map to Town02 and the pool to 1,200 MB brought usage down to 2.6 GB.
- Crash reports always pointed at `FCarlaServer::TickCueReceived`. Decoding the addresses with the shipped debug symbols showed this was only the game thread's position when the renderer failed, so the stack traces were misleading and controlled experiments were the only reliable diagnostic.

## 3. Lane detection

Lane following uses no learned model: the lane is recovered from each camera frame with four classical OpenCV steps and expressed in metres, so the controller can steer on it directly.

1. **Inverse perspective mapping.** The camera's height and focal length are known, so a homography maps the road region 5 to 25 m ahead into a bird's-eye view at 20 pixels per metre. Lane lines become near-vertical and parallel.
2. **Top-hat filtering.** A morphological top-hat with a 0.8 m wide horizontal kernel keeps only structures that are narrow and brighter than their surroundings. This isolates painted markings and kerb edges regardless of colour, and is insensitive to shadows because it is a local contrast test.
3. **Sliding-window search.** A column histogram of the lower half of the mask seeds one search per side; ten stacked windows then follow each line upward. When a line was found in the previous frame, the search is restricted to within 0.5 m of that fit.
4. **Polynomial fit and validation.** Each line is fitted as lateral offset = f(forward distance), second order when it spans more than 10 m. A left and right pair is accepted only if the implied lane width stays between 2.4 and 5.0 m and varies by less than 1 m along its length.

The focal length follows from the field of view, and a ground point x metres ahead and y metres to the right of a camera at height h projects to pixel (u, v):

$$f = \frac{W}{2\tan(\mathrm{FOV}/2)} = \frac{800}{2\tan 50^\circ} \approx 335.6\ \text{px}, \qquad u = c_x + f\,\frac{y}{x}, \qquad v = c_y + f\,\frac{h}{x}$$

The detector reports a confidence of 1.0 when both lines are found, 0.6 when one is found and the other is inferred from the running lane-width estimate, and 0 otherwise. Town02 has a painted centre line but no painted edge line, so the right boundary is usually the kerb and single-line detections are common.

![Lane detection pipeline](figures/lane_pipeline.png)

*Figure 3. Lane detection on a recorded frame. (a) The road region 5 to 25 m ahead. (b) The same region warped to a bird's-eye view. (c) Top-hat mask with the fitted left (cyan) and right (orange) lines. (d) The lane projected back onto the image. Tree shadows crossing the road do not appear in the mask.*

## 4. Traffic-light recognition

The light state is read from the camera alone; the simulator's own light state is used only afterwards, to score the detector. Colour thresholding by itself proved far too permissive, and the final detector is four tests applied in sequence.

1. **Colour and brightness.** Pixels above the horizon are thresholded in HSV for red, amber and green. A lit lamp is emissive, so the value channel must exceed 215 of 255; sunlit brick and foliage reach only about 190.
2. **Shape.** Connected components must be compact: aspect ratio between 0.5 and 2, and at least 45% of their bounding box filled.
3. **Housing.** Directly above or below the lamp there must be unlit housing: median saturation under 90 and almost no pixels of the lamp's own colour. This is what separates a lamp from a red wall or green leaves.
4. **Geometric gate.** The lamp's position is estimated and must lie on the car's side of the road (1.5 m left to 8 m right of the camera axis) and no further than 3 m beyond the next junction entry.

The state is then smoothed by a majority vote over the last five frames.

### 4.1 Where to stop

Knowing the colour is not enough: the car must stop where it can still see the lamp, or it will never see it turn green. Each lamp colour sits at a known height on the pole (red 2.95 m, amber 2.63 m, green 2.30 m), so its image row gives a first range estimate:

$$d \approx x_{cam} + f\,\frac{H_{lamp} - h_{cam}}{c_y - v_{lamp}}$$

That estimate selects which LiDAR returns on the pole beneath the lamp to trust, and their median replaces it. The car then aims to stop with its bumper 5.5 m short of the pole, which keeps the lamp inside the camera's field of view while waiting. Between sightings the stop distance is dead-reckoned from speed, and a median over five range samples rejects single-frame outliers.

![Traffic light recognition](figures/traffic_light.png)

*Figure 4. Traffic-light recognition on a recorded frame. (a) Only the region above the horizon line is searched. (b, c) The lamp and its colour mask. (d) The accepted detection with nearby LiDAR returns coloured by range; the returns on the pole give 9.6 m.*

A red light latches the stop decision until green is seen. Amber triggers a stop only if the car can do so at under 4 m/s².

## 5. Pedestrians and obstacles

People are detected in the camera image and located by camera geometry first, with LiDAR as a refinement; everything else in the car's way is handled by LiDAR alone.

### 5.1 Detection

Pedestrians and vehicles are detected with YOLOX-S from the official OpenCV Zoo, run through `cv2.dnn` on the CPU. No deep-learning framework is installed. Inference takes about 165 ms per 640 x 640 frame on this laptop, so the detector runs every third frame and its boxes are reused in between.

### 5.2 Ranging: why the camera comes first

The original design took range from the LiDAR points falling inside each detection box. It failed in testing: the car ran into a pedestrian it had detected, because it believed the person was 29.8 m away when the true distance was about 6 m.

A controlled test found the cause. A pedestrian standing 8 m ahead returned 62 LiDAR points; the same pedestrian returned zero once CARLA's AI walker controller was started. Since version 0.9.15 CARLA disables collision geometry on AI walkers for speed, and ray-cast LiDAR only hits collision geometry. The box therefore contained only background returns.

The fix has two parts:

- Range now comes from the camera: the row where the box touches the ground gives distance through the flat-ground projection in section 3, or the box height does if the feet are cut off. LiDAR points replace this estimate only when at least three of them agree with it to within 30%.
- Collisions are re-enabled on every spawned walker, which makes them visible to LiDAR again.

![Pedestrian detection and LiDAR view](figures/pedestrian.png)

*Figure 5. After the fix. (a) A pedestrian stepping off the kerb is ranged at 6.3 m, with 46 LiDAR returns confirming the camera estimate. (b) The same instant from above: obstacle-band returns in orange, the person circled beside the planned path.*

### 5.3 Reaction

Each person is compared with the planned path. Within 1.8 m of it the car plans a stop 3 m short of them; within 2.6 m it slows to 10 km/h. Because detections flicker and a close pedestrian drops below the camera's view, the last stop decision is held for 1.5 s and its distance dead-reckoned.

### 5.4 LiDAR obstacle corridor

LiDAR points between 0.4 and 1.8 m above the road are kept as obstacles. Those within 1.15 m of the planned path, at least eight of them, define the free distance ahead, and the target speed is chosen to stop 4 m short. The same rule produces car-following behaviour and queueing behind stopped traffic. The upper limit was lowered from 2.2 m after overhanging tree branches stopped the car on an empty road.

## 6. Planning and control

Every 50 ms the planner turns perception into one target speed and one steering angle.

### 6.1 Speed: the most cautious rule wins

Each rule proposes a speed limit and the lowest one is applied. The winning rule is shown on the display as the car's current decision.

| Decision | Speed limit |
| --- | --- |
| CRUISE | 30 km/h |
| JUNCTION | 18 km/h within 25 m of a junction or inside one |
| CURVE | Speed giving at most 2 m/s² lateral acceleration on the curvature ahead |
| RED LIGHT | Braking profile to the stop point before the pole |
| PEDESTRIAN | Braking profile to 3 m before the person, or 10 km/h when they are near the path |
| OBSTACLE | Braking profile to 4 m behind the LiDAR obstacle |
| DESTINATION | Braking profile to the goal |

Every braking profile uses the same comfortable deceleration a = 2.5 m/s² for a remaining distance d, and a proportional-integral controller tracks the chosen speed with throttle or brake:

$$v_{max}(d) = \sqrt{2\,a\,d}$$

### 6.2 Steering: pure pursuit on the camera lane, with a map fallback

Steering aims at a point a speed-dependent distance ahead (6.5 to 14 m). The point comes from the camera-detected lane centre when detection confidence is at least 0.6, the car is more than 8 m from a junction, and the camera and map agree to within 1 m. Otherwise it comes from the planned route's waypoints, because junctions and tight corners have no usable markings. For a target at (x, y) measured from the rear axle and wheelbase L:

$$\delta = \arctan\left(\frac{2\,L\,y}{x^2 + y^2}\right)$$

### 6.3 A calibration finding

In early runs the car ran wide on every tight corner and struck a pole. A trace through the curve showed a commanded turning radius of 10.8 m against a measured 13.1 m. The cause is that CARLA reports the maximum steering angle of the inner front wheel (70°); under Ackermann geometry the outer wheel turns less, so the effective angle is about 20% smaller than the bicycle model assumes. A steering gain of 1.3 plus a small cross-track correction (0.06 of full lock per metre of lateral error) removed the problem.

## 7. Point-to-point navigation

Given a destination anywhere on the map, the car plans the shortest legal route from its current position, drives it, and stops at the goal. The destination is set by clicking on a map panel, or from the command line.

### 7.1 Road graph

The graph is built from CARLA's map topology. Each node is one lane segment: a single lane of a road, or a single path through a junction, sampled every metre. A directed edge joins a segment to every segment that begins where it ends. Because edges exist only where the map allows driving, any path through the graph respects lane direction and permitted turns. Town02 yields 112 segments, 202 edges and 4.2 km of lane, built in under 10 ms.

### 7.2 A* search

The planner runs A* over segments. The cost g is the distance driven to the end of a segment, and the heuristic h is the straight-line distance from there to the goal:

$$f(n) = g(n) + h(n), \qquad h(n) = \lVert p_{end}(n) - p_{goal} \rVert_2$$

A straight line can never be longer than a drivable path, so h never overestimates and the route found is the shortest one. The start and goal may lie partway along a segment; only the remaining part of the first segment and the leading part of the last are counted.

### 7.3 Validation

The planner was checked against an independent Dijkstra search on 1,000 random pairs of the map's 101 spawn points (`tools/validate_planner.py`).

| Measure | Result |
| --- | --- |
| Pairs with a route found | 1,000 of 1,000 |
| Largest length difference from Dijkstra | 0.0 m |
| Segments expanded, A* (mean) | 31.0 |
| Segments expanded, Dijkstra (mean) | 51.1 |
| Planning time (mean / max) | 0.7 ms / 7.2 ms |

A* therefore returns provably shortest routes while examining 39% fewer segments than uninformed search. "Best" here means shortest distance; the planner does not yet weigh traffic or signal timing.

### 7.4 Driving the route

The planned waypoints replace the wandering route used in earlier tests, so every downstream module is unchanged: the same lane following, light handling and obstacle checks apply. A DESTINATION rule in the speed arbitration brakes the car to a stop at the final waypoint, and the trip ends when the car is stationary within 1 m of it.

## 8. Results

In the final evaluation the car completed six of six navigated trips, 2,592 m in total, with no collisions and no red lights run.

### 8.1 Navigated trips

Six destinations were drawn at random, each at least 120 m in a straight line from the car's position, and driven back to back in live traffic with pedestrians.

| Trip | Planned route (m) | Driven (m) | Time (s) | Stopped from goal (m) | Red-light stops | Red lights run | Collisions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 387 | 386 | 76 | 0.08 | 1 | 0 | 0 |
| 2 | 455 | 443 | 181 | 0.36 | 5 | 0 | 0 |
| 3 | 358 | 340 | 75 | 0.10 | 1 | 0 | 0 |
| 4 | 233 | 228 | 102 | 0.08 | 3 | 0 | 0 |
| 5 | 460 | 441 | 183 | 0.15 | 4 | 0 | 0 |
| 6 | 773 | 754 | 192 | 0.19 | 2 | 0 | 0 |
| **Total** | **2,666** | **2,592** | **809** | **mean 0.16** | **16** | **0** | **0** |

Driven distance is 1 to 5% below planned distance because the car rounds corners slightly inside the lane-centre waypoints. Trip time is dominated by signals rather than distance: trip 3 covered 340 m in 75 s with one red light, while trip 4 needed 102 s for 228 m with three.

![Planned and driven routes for six trips](figures/routes.png)

*Figure 6. Planned route (blue) and driven trajectory (black) for the six trips. Routes obey lane direction, which is why trip 6 loops around a block to reach a goal only 142 m away in a straight line.*

### 8.2 Behaviour over a trip

![Speed profile of trip 2](figures/speed_profile.png)

*Figure 7. Trip 2: speed with the active decision shaded (top) and steering source (bottom). The five flat sections at zero are red-light waits.*

Across all six trips the active decision was distributed as follows.

| Decision | Share of time |
| --- | --- |
| RED LIGHT (braking for or waiting at a signal) | 40.2% |
| CRUISE | 18.6% |
| CURVE | 17.7% |
| JUNCTION | 10.5% |
| OBSTACLE (following or queueing behind traffic) | 10.3% |
| DESTINATION | 1.7% |
| PEDESTRIAN | 0.9% |

Steering came from the camera-detected lane in 53% of frames and from map waypoints in the remaining 47%.

### 8.3 Perception accuracy

On the approach to signals, the camera's reading of the light matched the simulator's true state in 93% of 6,155 frames. Expected sources of disagreement are the first frames of an approach, before the lamp is large enough to detect, and the five-frame voting delay at each change of state; the errors were not broken down by cause. A person was detected in 3,702 of 16,176 frames, and the car came to a full stop for a pedestrian once.

### 8.4 Free-driving soak tests

Before navigation was added, the car wandered with a random turn at each junction. These runs used the stack as it stood before the final two stopping adjustments in section 9.

| Run | Simulated time | Distance (m) | Collisions | Red-light stops | Red lights run | Light agreement |
| --- | --- | --- | --- | --- | --- | --- |
| Seed 3 | 300 s | 724 | 0 | 4 | 0 | 92% |
| Seed 7 | 180 s | 737 | 0 | 4 | 0 | 91% |

## 9. Failures found and fixed

The system reached its final behaviour through eleven observed failures, each diagnosed from logged data or saved frames rather than by guesswork. The first nine are tabulated in the order they appeared; the last two surfaced during navigation testing and are described below the table.

| # | Observed failure | Root cause | Remedy |
| --- | --- | --- | --- |
| 1 | Simulator crashed within seconds or minutes | DirectX 11 and Vulkan backends unstable on this driver; Low quality mode crashes with a camera attached; default map exhausted VRAM | DirectX 12, Epic quality, Town02 as default map, smaller texture pool |
| 2 | Car ran wide on tight corners and hit a pole | Reported maximum steer is the inner wheel's; effective angle about 20% lower | Steering gain 1.3 and cross-track correction |
| 3 | Clearly visible red lamp not detected | Housing test examined a wide region that included a bright wall | Test only the strip directly above and below the lamp |
| 4 | Car stopped at red, then never saw green | Stop point was set relative to the junction; signal poles stand about 8 m before it, so the lamp left the field of view | Stop relative to the pole, ranged by LiDAR |
| 5 | Car stopped on an empty road | Overhanging branches fell inside the obstacle height band | Band upper limit lowered from 2.2 m to 1.8 m; at least 8 points required |
| 6 | Collision with a detected pedestrian | AI walkers are invisible to LiDAR, so range came from background points | Camera-first ranging; walker collisions re-enabled; 1.5 s decision memory |
| 7 | Green lamp ignored when its housing was sunlit | Housing was assumed dark; in sunlight it is grey | Housing defined as unsaturated and not lamp-coloured, instead of dark |
| 8 | Car halted for red brick and red wall trim | Colour and shape alone match building materials | Brightness threshold raised to 215; geometric gate on lamp position |
| 9 | Green lamps rejected by the new gate | Range assumed every lamp at the red lamp's height; green sits 0.65 m lower | Per-colour lamp heights |

Navigation testing added two more. In the first six-trip run the program crashed on trip 2: the running lane-width estimate had drifted wide enough to push a search window outside the image, and the estimate is now clamped to 3.0 to 4.2 m. In the second run five trips arrived, and on the sixth the car braked late for a red light, overshot its stop point by about 1.5 m and left the simulator's stop zone while the light was red. It still halted about 2 m before the pole, but the scorer counted a red light run. The stop point was moved from 3.5 m to 5.5 m before the pole and the brake gain raised from 0.35 to 0.6; the third run is the one reported in section 8.

Two lessons generalise. First, a colour detector needs a structural test and a geometric test before it is safe to act on. Second, the evaluation code needs the same scrutiny as the driving code: an early scorer reported three red lights run when the car had in fact crossed each stop line on green, because it matched the wrong signal after the car had passed its own.

## 10. Limitations and next steps

The results hold for one map, one weather condition and low speeds, and several design choices are specific to that setting.

- **One town.** Lamp heights, the right-hand pole position and the stop distance are calibrated to Town02's European-style signals. Maps with overhead or far-side signals need a different gate.
- **Daylight only.** Night, rain and low sun were not tested; the brightness threshold in particular will need revisiting.
- **Map-assisted steering.** The camera steers on marked straights and gentle curves; junctions and tight corners use map waypoints. The route itself always comes from the map.
- **Shortest, not fastest.** The planner minimises distance and ignores traffic and signal timing.
- **No lane changes or overtaking.** Town02 has one lane per direction, and the car waits behind any obstacle indefinitely.
- **Low speed.** Cruise speed is 30 km/h. The detector runs at about 6 Hz, which would be too slow for higher speeds.
- **Simulation only.** Sensor noise, calibration error and actuation delay are far milder than on a real vehicle.

Natural next steps are a learned lane and signal detector trained on CARLA data for comparison against the classical one, time-based route costs, testing across towns and weather presets, and an object tracker to replace the fixed decision memory.

## 11. Reproducibility

The project is about 1,500 lines of Python, with every tunable value in a single configuration file (`config.py`).

| Component | Version or source |
| --- | --- |
| Simulator | CARLA 0.9.16 Windows package, launched with `-dx12 -quality-level=Epic` |
| Python | 3.12, packages `carla==0.9.16`, `opencv-python`, `numpy` |
| Detector weights | `object_detection_yolox_2022nov.onnx` from the OpenCV Zoo |
| Random seeds | 7 (default) and 3, covering start position, route and traffic |

```
python main.py --navigate                               # click a destination on the map
python main.py --trips 6 --telemetry runs/nav.csv --trips-json runs/nav.json
python main.py --seed 3 --frames 6000 --no-display      # wandering soak test
python tools/validate_planner.py --pairs 1000           # A* vs Dijkstra
```

Each run prints its own scorecard against simulator ground truth: distance, collisions, red lights obeyed and run, and camera-versus-simulator agreement on light state. Runs are not bit-for-bit repeatable, because pedestrian behaviour varies between runs even with a fixed seed.

## Sources

- [CARLA 0.9.16 release notes and downloads](https://github.com/carla-simulator/carla/releases/tag/0.9.16), including the 0.9.15 change that disables collisions on AI pedestrians
- [CARLA 0.9.16 release announcement](https://carla.org/2025/09/16/release-0.9.16)
- [OpenCV Zoo YOLOX object detection model](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox)
