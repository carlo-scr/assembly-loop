# assembly-loop

![The dashboard during a tower build on the stand-in arm](docs/dashboard.png)

Hold-to-run assembly on a UFACTORY xArm 7. One local web page starts a session at the robot, teaches it places, arms
it, runs a task and stops it. Every motion goes through one guard, and the arm moves only while a person holds the
grip of a Quest controller.

Today it moves wooden blocks between places taught by hand and stacks them into a tower. Press-fit bricks and cables
are meant to follow on the same base.

> **Status, 19 November 2026.** Everything here is tested on a stand-in arm (34 tests). It has not yet been run on the
> real arm. The first session on the arm follows the rungs below, one at a time.

## Run it

It needs Python 3.9 or newer, numpy, and for a real arm UFACTORY's xArm Python SDK. Set these up once, in an
environment inside this folder, so that nothing else on the machine is touched:

```
git clone https://github.com/carlo-scr/assembly-loop.git
cd assembly-loop
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[real]"
```

From then on, in every new terminal:

```
cd assembly-loop
source .venv/bin/activate
python -m assemblyloop dashboard
```

Then open `http://127.0.0.1:8777/` in a browser on the same machine.

- **`python: command not found`** means the environment is not active in that terminal: run the `source` line.
- **`python3 -m venv` fails** (some Ubuntu installs lack it): with [uv](https://docs.astral.sh/uv/) installed, use
  `uv venv && uv pip install -e ".[real]"` instead of the two lines after `cd`.
- **The arm does not answer:** the machine needs an address on the arm's network (192.168.1.x) on its wired port.
  `ip -4 -brief addr` shows whether it has one.

- **Teleoperation must be closed first.** Both listen for the Quest on port 6666.
- **Without a robot,** choose "Stand-in arm" on the page. The space bar stands in for the grip.
- **Other arm addresses:** `python -m assemblyloop dashboard --right 192.168.1.199 --left 192.168.1.219`
- **Read only:** `python -m assemblyloop status --ip 192.168.1.199` prints what the controller reports and changes
  nothing.

A session writes `assembly_logs/guard.jsonl` (everything the arm was asked to do, and why each move ended),
`assembly_logs/moves.jsonl` (one line per block moved) and `spots.json` (the taught places). None of these are
tracked by git.

## What keeps the arm safe

- **Hold to run.** Motion needs the session armed *and* a fresh "grip held" signal from the headset. No signal counts
  as released. Letting go pauses within a moment; squeezing again continues.
- **One door to the arm.** Only the guard sends commands. It refuses anything outside a box around the taught places
  and anything faster than 60 mm/s.
- **Small, straight moves.** Translations at a fixed tilt, in segments of at most 30 mm. The gripper turns by its last
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
3. Bring the gripper about 20 cm above the table, pointing at it, with nothing near. Let go of the grip and press
   Arm. The label reads AWAITING CALIBRATION. Nothing may move, and "Measured against planned" reads under 1.5 mm.
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

## Layout

```
assemblyloop/
  arms.py       the few calls made to the arm, and a stand-in arm with the same calls
  deadman.py    hold-to-run: the Quest grip, or the space bar with the stand-in arm
  guard.py      the only door to the arm: limits, segments, pause, stop, log
  cycle.py      a task as a list of named steps, and their execution through the guard
  spots.py      places taught by hand; moving a block between them; the levels of a tower
  plate.py      a stud plate taught from three poses (press-fit bricks; command line only for now)
  session.py    one session: OFF, DISARMED, AWAITING CALIBRATION, ARMED, RUNNING
  dashboard.py  the web page of a session
tests/          the safety rules and the session, on the stand-in arm
```

## Tests

```
pip install -e ".[dev]"
pytest -q
ruff check .
```
