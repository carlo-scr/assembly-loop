# assembly-loop

![The dashboard during a tower build on the stand-in arm: the sequence and the controls on the left, a model of the arm and the camera pictures on the right](docs/dashboard.png)

Hold-to-run assembly on a UFACTORY xArm 7. One local web page starts a session at the robot, teaches it places, arms
it, runs a task and stops it. Every motion goes through one guard, and the arm moves only while a person holds the
grip of a Quest controller.

Today it moves wooden blocks between places taught by hand and stacks them into a tower. Press-fit bricks and cables
are meant to follow on the same base.

> **Status, 21 November 2026.** Everything here is tested on a stand-in arm (77 tests). On the real right arm, a session,
> the Quest grip and nudging have run; moving blocks, towers and the camera have not. The first session on the arm
> follows the rungs below, one at a time. The page that fits one screen (the model of the arm, the live pictures) has
> run on the stand-in arm only: its cameras have not been opened on the rig yet. The older page is kept at `/classic`.

## Run it

It needs Python 3.9 or newer, numpy, and for a real arm UFACTORY's xArm Python SDK. Set these up once, in an
environment inside this folder, so that nothing else on the machine is touched:

```
git clone https://github.com/carlo-scr/assembly-loop.git
cd assembly-loop
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[real,eyes]"
```

`real` is the arm's SDK; `eyes` is OpenCV, for the live pictures and the camera that finds the blocks. After a
`git pull` that changes what is needed, run the `pip install` line again.

From then on, in every new terminal:

```
cd assembly-loop
source .venv/bin/activate
python -m assemblyloop dashboard
```

Then open `http://127.0.0.1:8777/` in a browser on the same machine.

## The page

It fits one screen and does not scroll:

- **Top left, the sequence:** session, checks, arm, places, run. Only the step that is next is spelled out.
- **Below it, the controls,** once armed: the nudge keys (W A S D, Q E, Z C, F R) while places are taught, then moving
  blocks, towers and the block finder.
- **Top right, a model of the arm** you are working with, in the pose the controller reports. Around it: the box the
  fingertips may be in, the taught spots, and a dashed line through every place the current move still goes to. Drag
  to turn it. It is a drawing: nothing is steered or judged by it.
- **Bottom right, the cameras** that belong to that arm (the two scene cameras and the two on its gripper), live. Every
  picture carries its age: one that stopped arriving turns grey and says for how long. Click a picture to enlarge it.
- **Top bar:** the state, what to do next, the hold-to-run signal, STOP SESSION.

Good to know:

- **The older page** (one long column, no model, no pictures) is at `http://127.0.0.1:8777/classic`. Same program,
  same buttons: use it if the new page misbehaves.
- **Cameras are opened only while the page is open,** and let go a few seconds after it is closed or hidden. With
  `--no_feeds` the page opens no camera at all.
- **The page needs no internet:** three.js and the arm's meshes are in `assemblyloop/web/`.

- **`python: command not found`** means the environment is not active in that terminal: run the `source` line.
- **`python3 -m venv` fails** (some Ubuntu installs lack it): with [uv](https://docs.astral.sh/uv/) installed, use
  `uv venv && uv pip install -e ".[real]"` instead of the two lines after `cd`.
- **The arm does not answer:** the machine needs an address on the arm's network (192.168.1.x) on its wired port.
  `ip -4 -brief addr` shows whether it has one.

- **Teleoperation must be closed first.** Both listen for the Quest on port 6666.
- **Without a robot,** choose "Stand-in arm" on the page. The space bar stands in for the grip.
- **Without the Quest:** `python -m assemblyloop dashboard --key_hold` lets the space bar of the page be the
  hold-to-run switch of a real arm. It counts only while the page is in front and has the keyboard.
- **The arms are mounted at 45 degrees,** rolled about their forward axis like shoulders. "Straight down" on the page
  takes that into account. On another rig: `--mount_deg 0` for an upright arm.
- **Other arm addresses:** `python -m assemblyloop dashboard --right 192.168.1.199 --left 192.168.1.219`
- **Read only:** `python -m assemblyloop status --ip 192.168.1.199` prints what the controller reports and changes
  nothing.

- **The Quest app finds this computer by itself** while a session is running: it is listed as
  "<computer name> (assembly-loop)". The headset must be on the arm's network (the "swoosh" Wi-Fi on this rig).

A session writes `assembly_logs/guard.jsonl` (everything the arm was asked to do, and why each move ended),
`assembly_logs/moves.jsonl` (one line per block moved) and `spots.json` (the taught places). None of these are
tracked by git.

## Designing what to build

`http://127.0.0.1:8777/build` is the designer. It needs no arm and no session.

- **Right:** every kind of block in `blocks.json` (or the example sizes, if there is none), in its colours. Drag one
  onto the mat. It drops straight down until it meets something, as when the arm puts it down.
- **Left:** the block you clicked: its colour, how it lies, turning (R, Q, E), moving (arrow keys), copy, remove.
  Below it the steps, in the order the arm puts the blocks down, and whether each step stands.
- **Middle:** the mat around the spot it will be built on. The robot is on the near side. Drag the slider at the
  bottom, or press ▶, to see it being built step by step.

After every change the program checks every step of the build:

- **Does it stand?** Wood on wood, nothing glued: forces that only push, through faces that touch, must carry
  every block's weight. The bars show how far the table could tilt before something tips over. Below 4° the design
  is called shaky. A design that stands only once finished is refused: the arm builds it step by step.
- **Can the arm do it?** Each block comes straight down onto blocks already there, with nothing over it. The jaws
  close across its short side, and the fingers need room on both sides when they let go: the selected block shows
  where they go, in red if they would hit something. Square and round blocks are taken the other way round when that
  leaves room. "Order for the robot" finds an order in which every step stands, if there is one.
- **Limits:** 150 mm from the spot each way, and about 220 mm high.

Saved designs go to `designs/` (not tracked by git). Four examples come with the program. To build one: on the
dashboard, Camera tab, press Look, then under "A design" choose it and the spot, and press "Build the design". Every
block is taken from the last look (the same kind, lying the same way), and let go opening only 10 mm past each side.
Nothing is built unless the check passes. Lay the blocks out on the mat at least 30 mm from where the design goes.

The gripper's sizes come from UFACTORY's model of it, not from measuring the rig.

## What keeps the arm safe

- **Hold to run.** Motion needs the session armed *and* a fresh "grip held" signal from the headset. No signal counts
  as released. Letting go pauses within a moment; squeezing again continues.
- **One door to the arm.** Only the guard sends commands. It refuses anything outside a box around the taught places
  and anything faster than 120 mm/s (travel is 100, coming down 30, pressing 15).
- **Straight moves.** Translations at a fixed tilt, in segments of at most 80 mm (3 mm when contact is watched). The gripper turns by its last
  joint alone, 10 degrees at a time.
- **Contact is watched.** Lowering and lifting with a block go 3 mm at a time and compare where the joints are with
  where they were told to be. A move that meets something ends by itself.
- **Faults end the session.** A controller error, or a free move that ends up off its plan, stops the arm and the
  session stays stopped. The program never clears a controller error and never changes controller settings.
- **The page is local.** It answers on 127.0.0.1 only, and every button needs a token that only the served page has.

The red emergency stop stays within reach the whole time. Hold-to-run is not a substitute for it.

## First time on the real arm

One rung at a time. Stop at the first that misbehaves and keep the two log files.

1. `status` (above) prints a pose and `error 0`.
2. On the page, start a session on the arm. The label reads DISARMED, "Hold-to-run signal arriving" is ticked, and
   squeezing the Quest grip shows GRIP HELD.
3. Bring the gripper about 20 cm above the table with nothing near. Let go of the grip and press Arm. If the gripper
   does not point straight down (the lab's teleoperation leaves it tilted), hold the grip and
   press "Point the gripper down": it tilts 5 degrees at a time with the flange in place, the fingertips swing
   sideways, and it stops by itself if any joint moves much further than the tilt asks for. The label reads AWAITING CALIBRATION. Nothing may move, and "Measured against planned" reads under 1.5 mm.
4. Hold the grip, set the step to 1 mm and press Up once. Then 5 mm in each direction: each must go where its button
   says.
5. Set the step to 20 mm, press Up and let go of the grip half way. It must stop within a moment and go on when you
   squeeze again.
6. Turn by 5 degrees. Only the gripper spins; the rest of the arm stays still.
7. Open and close the gripper, first on nothing, then on a block held between the jaws. It says how wide the block is.
8. **Contact.** Put an empty cardboard box under the gripper, set the step to 1 mm and press Down repeatedly onto it.
   The nudge that meets the box must end by itself ("blocked: something is in the way") before the box is crushed.
   If it keeps pushing, stop: put nothing down on the table until this is understood.
9. Press STOP SESSION once on purpose. The label reads STOPPED and stays that way.

## Moving and stacking blocks

1. **Teach the places.** With the label at AWAITING CALIBRATION, stand a block where it should be picked up, nudge
   the open gripper around it (fingertips a few millimetres above the table) and press "Save spot here". Do the same
   for every other place, then press Done. A spot remembers how the gripper was turned there, so a block can be
   picked up one way round and put down another. The tilt of the gripper must be the same at every spot.
2. **One block.** Say where the block stands and where it goes, and press Go. Start with both boxes ticked: "rehearse
   in the air" does the whole travel without coming down, and "one step at a time" waits for a button before every
   step.
3. **A tower.** Stand one block on every spot except the one the tower goes on, choose that spot, enter the block
   height and press Build. The first block goes down at the taught height and each next one a block height higher,
   let go 2 mm above the block below. Take down returns every block to its own spot, top first.

## A tower from what the camera sees

One of the rig's scene cameras looks at the table. It finds the painted blocks on the grey mat, works out which block
each one is and how it lies, and the arm stacks them, widest at the bottom. Nothing is trained and nothing is
teleoperated. **This part has run on the stand-in arm and a made-up camera only (11 tests); it has not yet run on the
real arm or with the real camera.**

It needs OpenCV: `pip install -e ".[real,eyes]"` (with uv: `uv pip install -e ".[real,eyes]"`).

1. **Measure the blocks.** Copy `blocks.example.json` to `blocks.json` and enter every kind of block with its
   colour and its sizes in millimetres (a box: three edges; a cylinder: diameter, length). The sizes in the example
   are guesses. `finger_mm` is how deep a block can sit between the fingers, `jaw_mm` how far the jaws open.
2. **Find the table.** With the gripper pointing straight down over the middle of the area, close it on a bar held
   level between the jaws (jaws across its short side), press "Find the table" and hold the grip. The gripper comes
   straight down at 15 mm/s until the bar meets the mat, lets go, and makes the spots itself: T where the bar lies
   (the tower will stand there) and P and Q at opposite corners of the area. Or teach the spots by hand.
3. **Show the camera the table** (Camera card, once, and again whenever the camera or the table was moved). Lay one
   bar flat on a taught spot, the jaws across its short side, say which spot under "The block stands on", keep
   everything else off the mat, and hold the grip. The arm carries the bar to a dozen places and steps aside for a
   picture after each. From where it put the bar and where the camera saw it, the camera is fitted and kept in
   `camera.json`. The log says within how many millimetres the camera then finds the bar; more than 6 mm is refused.
4. **Look.** Lay blocks on the mat, apart from each other and away from the tower spot, move the arm out of the
   camera's view, and press Look. The picture shows what was made out, the order of the tower, and what is left out
   and why.
5. **Build a tower from what it sees.** Rehearse in the air first. Every block is taken across its short side and put
   down the way it lay; tall blocks are held near their top. The gripper is turned to each block as the camera sees it
   lie, by its last joint, before it comes down.

**Checked by the camera after every block** (the tick box "check every block", on unless you untick it; towers and
designs alike). After each put-down the arm goes up and moves above the taught spot that shows farthest from the
build in the picture, and the camera takes one picture:

- Every block put down so far is looked for where it should be. One that is not there, or more than 8 mm or 12°
  off, stops the job; the arm stays armed and the log says which block and by how much. If the first picture
  disagrees, a second one is taken from another place first, in case the arm was in the way.
- The same picture is looked at as a whole, and the next block is taken where it lies now and turned as it lies now
  (it may have been pushed), if it is within 30 mm of where it was seen before.
- When the jaws close, they must close to the width of the block meant, within 10 mm. Another block, or the right one
  taken the wrong way round, stops the job. (Unticking "check every block" drops this check too.)

Every such picture is kept in the session's folder under `assembly_logs/<session>/pictures/`, with a line in
`moves.jsonl` that says what was checked and found. No picture is taken while the arm moves, and pictures never steer
the arm while it moves: each one changes only where the next move goes. This costs a few seconds per block.

What it does not do yet: blocks of one colour that touch are one patch and are not made out on the mat (on the build
they are checked one by one, since where each should be is known); a lying cylinder is left out (it rolls); a block is
never turned over; the tower and the travel above it stay under 300 mm; the wrist cameras are not used. `python -m assemblyloop look`
takes one picture and writes `look.jpg` with what the camera makes out, without any arm.

## Layout

```
assemblyloop/
  arms.py       the few calls made to the arm, and a stand-in arm with the same calls
  deadman.py    hold-to-run: the Quest grip, or the space bar with the stand-in arm
  guard.py      the only door to the arm: limits, segments, pause, stop, log
  cycle.py      a task as a list of named steps, and their execution through the guard
  spots.py      places taught by hand; moving a block between them; the levels of a tower
  blocks.py     the kinds of block (blocks.json), how each can lie, and which goes onto a tower next
  eyes.py       the camera: coloured patches, the camera fitted to the table, which block each patch is
  sight.py      the camera in a session: show it the table, look, build a tower or a design from what it sees
  design.py     designs: what to build, block by block; whether each step stands and the gripper fits
  plate.py      a stud plate taught from three poses (press-fit bricks; command line only for now)
  session.py    one session: OFF, DISARMED, AWAITING CALIBRATION, ARMED, RUNNING
  feeds.py      the live pictures: one owner per camera, lent to the block finder for its own picture
  dashboard.py  the local web server of a session, and the older page
  designs/      the example designs
  web/          the page that fits one screen (index.html, app.css, app.js), the model of the arm (scene.js),
                the designer (build.html, build.css, build.js),
                three.js (vendor/, MIT) and UFACTORY's xArm 7 meshes (xarm7/, BSD-3-Clause, from MuJoCo Menagerie)
tests/          the safety rules and the session, on the stand-in arm
```

## Tests

```
pip install -e ".[dev]"
pytest -q
ruff check .
```
