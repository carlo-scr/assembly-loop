# Audit, 23 November 2026

What was looked at, what was found, what was changed, and what comes next. **Nothing written on this day has run
on the real arm or seen a real picture.** Every claim below is from the stand-in arm, drawn pictures, or reading the
code and the pinned SDK's source (xarm-python-sdk 1.18.4).

## 1. Where the program stood

A careful single-arm system: one guard, hold-to-run, 62 tests on a stand-in arm. Against the goals (data for
policies, closed loop, two arms, taller towers) four things were missing:

| Goal | What was there | What was missing |
|---|---|---|
| Data | `guard.jsonl`, one line per block in `moves.jsonl` | no pictures, no time series, no task in words, no success label |
| Closed loop | one look, then every block from that plan | no look after a move, no check of a grasp or a put-down |
| Perception | fixed colour thresholds, one block per patch | dim light, highlights, touching blocks; no real picture ever seen |
| Two arms | the left arm's address and mount | nothing tying the two base frames together, nothing keeping them apart |

## 2. Defects found and fixed

Confirmed by running code against the stand-in arm unless marked (read).

| # | What | Where | Fix |
|---|---|---|---|
| 1 | A refused `pause()` or `resume()` left the guard without a stop or a latch; the rest of the segment ran unwatched, or stayed queued for whoever enabled the arm next | `guard.py`, three polling loops | both now stop and latch |
| 2 | The hold-to-run switch took frames from any sender and merged them: a second sender reporting "held" kept the arm moving after the operator let go. Frames without a running number skipped the freshness check | `deadman.py` | the first headset heard is the switch until the session ends; a number is required |
| 3 | A stand-in session wrote over the real arm's `spots.json`; `use_saved` never looked at which arm the spots belonged to | `session.py` | one file per arm; the plain file is never written over by another arm; the side is checked |
| 4 | An unexpected exception in a nudge or gripper action was reported as "done", nothing stopped. The SDK returns a bare int on a gripper fault, which is exactly such an exception (read) | `session.py`, `arms.py` | caught, arm stopped, said; the bare int is an `ArmFault` |
| 5 | The first turn of a session, if close to half a turn with a joint that spins the other way, was misread at the 180 / -180 wrap: 272 degrees the wrong way, then a latch | `guard.py` `turn_to` | judged by signed progress |
| 6 | STOP sent `arm.stop()` only after waiting up to 3 s for the worker; a worker that outlived its session wrote into the next one | `session.py` | stop first; a worker writes only if it still is the session's worker |
| 7 | A malformed `spots.json` took the page down (`AttributeError` on every `/api/state`) | `session.py` | caught |
| 8 | `count: Infinity` and `Content-Length: -1` were unhandled in the web handler | `dashboard.py` | handled |
| 9 | An empty or wrong grasp retreated in a straight line to above where the arm had STARTED, across the table at table height | `cycle.py`, `spots.py` | straight back up above the block |
| 10 | `--air` could be negative: the whole cycle and its floor moved down | `cycle.py` | refused |
| 11 | The stand-in arm itself had a race (position and plan read half way through a step), which made one tower test fail now and then under load | `arms.py` | a step is atomic |
| 12 | The switch was checked before two blocking round trips, not last before the send (read) | `guard.py` | fault first, switch last |

## 3. Found and NOT fixed: decide at the rig

- **Unwatched moves notice a collision late.** The comparison of measured against planned runs after each segment,
  and a session raises the segment from 30 to 80 mm. With something in the way the plan is up to about 78 mm past
  it before the guard fails. Collision detection is off on this rig, so what limits the push is the controller's own
  servo error. Options: back to 30 mm, or a cap by speed.
- **The gripper does not watch the switch.** `XArm.grip` polls for up to 4 s; letting go does not stop the jaws.
- **Report rate (unverified).** `XArmAPI(ip)` uses the SDK's default report type. If that stream is 5 Hz rather than
  100 Hz, everything read "now" (arrival, faults, the every-joint watch while tilting, and the new watch on the other
  arm) is up to 200 ms old. Measure it; `report_type='real'` if so.
- **A piece of 1 mm or less "arrives" at once** on the real arm (it is within the arrival distance before it starts,
  and the published state has not yet turned to moving). The stand-in hides this.
- **`connect()` refuses an arm parked at all-zero joints** ("publishes no state").
- **`get_gripper_position` may write the tool's baud rate**, against "never changes a controller setting".
- `eyes.SCENE_CAMERA` is commented as the left scene camera but is port 4.3, which `feeds.RIG_CAMERAS` calls
  "Scene, right". One of the two labels is wrong.

## 4. What was added

**A tower with a look after every block** (`closed.py`, the Loop tab). After every block the arm steps aside and
the picture answers: is the last block on the tower (its outline drawn at its height and fitted to its colour);
which block next (planned anew every time); can the jaws get at it (finger rectangles against every footprint,
including things that were not made out). The next block is aimed half way back to the tower's axis. The jaws open
only as wide as the block needs. What the jaws close on is checked against the width the camera named. Blocks in
each other's way are pushed clear, heaps are pushed apart, both as watched moves. `scatter` and "build and spread,
for the rounds" produce a new arrangement for every round.

**Recording** (`record.py`, `export.py`). Every job is an episode: 10 rows a second of joints, pose, gripper, what
was asked, the switch, step and sub-task in words, and every camera's newest picture; events for steps, looks,
pushes, falls, missed grasps. Export to `.npz` (tested) and LeRobot (run against lerobot 0.7.0, format v3.0).
**openpi pins LeRobot format v2.1**: export inside openpi's environment; that path is written from source and has
not been run.

**Two arms** (`pair.py`, `rig.json`). The arms are fitted to each other from marks both have touched (the odd mark
is named; a fit against the mounts is refused). One arm takes over the other's spots and camera fit. The other arm
is read and the guard keeps away from it, using capsules fitted to the arm's meshes.

**The block finder** (`eyes.py`). Thresholds relative to the mat in the picture, hue groups instead of fixed
borders, highlights and holes filled, a patch explained as two or three touching blocks, a doubt signal (`margin`,
`rival`) and a crowding signal. Every picture a session looks at is kept (`looks/`), and
`python -m assemblyloop look --again assembly_logs` reads them all again.

## 5. Perception, measured on a harsher drawn world

Shaded faces, shadows, noise, JPEG artefacts, white balance, exposure, highlights, an uneven mat, an arm in view,
touching blocks; six blocks a scene, 16 kinds, 4K. "ok" = right kind and rest, within 3 mm and 6 degrees. **A proxy,
not the lab.**

| Condition (4K) | ok before | ok after |
|---|---|---|
| shaded faces | 100 % | 100 % |
| warm white balance | 44 % | 94 % |
| exposure halved | 0 % | 100 % |
| highlight on the tops | 33 % | 94 % |
| two colours touching | 75 % | 90 % |
| one colour touching | 0 % | 85 % |
| one colour, 4 mm apart | 8 % | 92 % |
| everything mild at once | 94 % | 98 % |
| ... with touching same-colour pairs | 2 % | 79 % |

What broke the old finder, largest first: same-colour blocks within 10 mm; the fixed brightness floor in a dim
room; highlights; yellow read as orange under warm light; a block half hidden by its neighbour. Calibration was not
the problem (the centroid bias for the bar is under 0.6 mm). One regression: under strong cool light at 1080p the
yaw of square blocks got worse. Stay at 4K.

Still wrong after the change: two cubes flush in a row read as a bar; kind or rest wrong for 10 to 17 % of blocks
inside a touching pair (their `margin` is low: pushed apart and looked at again, most resolve); **nothing above the
table is perceived at all** except the one block the builder expects on top of the tower.

## 6. What must be confirmed on the rig before the new parts are trusted

1. Both controllers report the flange, **no tool offset set**; both grippers are the same, 172 mm to the fingertips.
2. The two bases' forward axes are parallel and the mounts really are 45 degrees each way.
3. Everything bolted to an arm (the gripper cameras, cables) is within 95 mm of the tool axis: the capsules are from
   drawing meshes, not from the real covers.
4. The gripper's counts are tenths of a millimetre (rung 7), or "check widths" stays off.
5. The Quest app sends a running `seq` in every frame. If not, the grip never registers now.
6. How far the elbow moves for a short straight move of the flange. The watch assumes at most twice as far; the
   stand-in's solver showed 2.9 times in stretched-out postures. The recorded joints of any session answer this.
7. A fingertip is taken as 24 by 12 mm, the closed fingertips as a 14 mm disc. Not measured.
8. That wooden blocks slide on this mat under a 30 mm/s push ending 3 mm behind plan, rather than tip.

## 7. First time in the lab with this version

1. `pip install -e ".[real,eyes,dev]"`, `pytest -q`.
2. `status` on both arms; note `tcp_offset`.
3. README rungs 1 to 9 on the right arm. Watch the grip signal (item 5 above), pause and resume, the gripper's counts.
4. One block, a taught tower. Open the episode folder afterwards: pictures from every camera, few dropped ticks.
5. Blocks measured, table found, camera shown the table. **Look at 30 to 50 layouts, touching blocks included, and
   keep `looks/`.** This is the first real evidence about perception and the most valuable thing to bring back.
6. Loop tab: three blocks well apart, push off, step mode. Then two side by side, push on.
7. Left arm: status, a short session, five marks with both arms, fit, compare the distance on the page with a tape
   measure, then "Use the other arm's spots", in the air only.

## 8. Towards two arms on one tower, and the paper

One metric as the spine: **the height of tower reached**, with the scripted builder as baseline and as the source
of data. It labels its own failures (falls, missed grasps, pushes), which teleoperated data does not.

1. **A streaming door in the guard.** Motion is stop-and-go segments in position mode, on purpose. A policy emits
   action chunks at 10 to 50 Hz, and stop-and-go demonstrations teach it to pause. The guard needs a second mode
   that streams targets with the same box, speed, clearance and hold-to-run checks at every tick. This reverses the
   "never streams" rule in `arms.py` and is the owner's decision.
2. **Depth.** The second scene camera for triangulation, the gripper cameras to centre a grasp before closing and
   to measure the top of the tower. One oblique camera will not carry tall towers.
3. **A first policy.** A few hundred "build and spread" episodes, pi0.5 fine-tuned in openpi's environment, rolled
   out through the guard. Ablation: with and without the recovery episodes.
4. **Motion planning.** Tilted grasps, regrasps and two arms in one space need joint-space plans checked against
   the capsules. The guard stays the executor. The 300 mm tower limit goes when the travel is planned, not before.
5. **Two arms.** Rungs: (a) today: one moves, the other is watched; (b) one session, two guards, one mover at a
   time, a plane between them; (c) hand-over and regrasp; (d) one arm braces the tower while the other places;
   (e) both at once. Quest teleoperation for what cannot be scripted (tilting), mixed into the data.
6. **A world model.** First a predictor of "will this put-down stand", trained on the recorded outcomes, used to
   veto or rank a policy's chunks. An action-conditioned model on the same data later.
