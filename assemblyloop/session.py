"""A session at the robot, as the dashboard sees it. All the dashboard's buttons end up in the methods here.

ONE program has the arm. Teleoperation is not needed while this runs (and must be closed: it owns the Quest's port).
The ladder (the big label on the dashboard), every rung reached by a button on the page:
  OFF                    nothing is connected. The page shows which arms answer on the network (powered or not).
  DISARMED               "Start session" connected to an arm. Read only: nothing can move.
  AWAITING CALIBRATION   "Arm" switched the arm on. It does not know any place yet: nudge and turn the gripper to each
                         place with the page's own buttons (small steps, only while the grip is held), save it as a spot.
  ARMED                  the spots are known; a move may be started.
  RUNNING / PAUSED       a move is under way; PAUSED = the grip is released, or a step waits for its button.
  STOPPED                the session was ended by STOP, a fault, or a grip released too long. "Start session" begins anew.
Anything that moves needs BOTH the dashboard armed AND the grip held (`Gate`). Arming never moves anything by itself.

The job for now is deliberately simple: take the thing that stands on one taught spot and put it on another, once or
again and again. The stud grid of `plate.py` comes back when the plates are fixed down.
"""
from __future__ import annotations

import collections
import copy
import itertools
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np

from .arms import ArmFault, FakeArm, XArm, run_fake_clock
from .cycle import NotHeld, NothingHeld, Params, execute
from .deadman import Deadman, QuestDeadman
from .feeds import TABLE_CAMERA, MadeUp, same_device
from .guard import Blocked, Guard, Limits, Refused, Released
from .sight import NotAsPlanned, Sight
from .spots import Spots, demo_spots, plan_move, tool_axis

JOG = dict(up=(0, 0, 1), down=(0, 0, -1), fwd=(1, 0, 0), back=(-1, 0, 0), left=(0, 1, 0), right=(0, -1, 0))  # in (forward, left, up)
TOOL_MM = 172.0  # from the flange to the fingertips of UFACTORY's gripper. FOR DRAWING ONLY: every spot and limit is a flange position


class Gate:
    """Hold-to-run as the guard sees it: armed on the dashboard AND the switch held."""

    def __init__(self, deadman):
        self.deadman, self.armed = deadman, False

    def held(self):
        return bool(self.armed and self.deadman.held())

    def why(self):
        return "not armed on the dashboard" if not self.armed else self.deadman.why()


def answers(ip, port=502, timeout=0.3):
    """Does an arm's controller answer on the network? (A bare connection that is closed again; nothing is sent.)"""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def mount_down(side, deg=45.0):
    """Straight down (towards the floor) in the base frame of an arm that is mounted like a shoulder: rolled `deg`
    about its own forward (x) axis, the right arm one way and the left arm the other. On this rig both are at 45
    degrees ("1:30" and "10:30"), which is what the lab's teleoperation uses: for the right arm, up is (0, 0.707, 0.707)."""
    a = np.radians(float(deg))
    return np.array([0.0, (-1.0 if side == "right" else 1.0) * np.sin(a), -np.cos(a)])


def default_arms(right="192.168.1.199", left="192.168.1.219", mount_deg=45.0):
    return {"right": dict(label="Right arm", ip=right, make=lambda: XArm(right), down=mount_down("right", mount_deg)),
            "left": dict(label="Left arm", ip=left, make=lambda: XArm(left), down=mount_down("left", mount_deg)),
            "stand-in": dict(label="Stand-in arm (no robot)", ip=None, make=FakeArm)}


def jog_axes(rpy_deg):
    """Directions for nudging, as columns (forward, left, up) in the arm's base frame: up is away from the table along
    the gripper's own axis; forward is away from the robot's base (the base's x axis, levelled against up)."""
    up = -tool_axis(rpy_deg)
    x = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    fwd = x - (x @ up) * up
    fwd = fwd / np.linalg.norm(fwd)
    return np.stack([fwd, np.cross(up, fwd), up], 1)


class Session(Sight):
    def __init__(self, arms=None, quest_port=6666, hand="right", spots_path="spots.json", log_dir="assembly_logs", probe=answers, say=None, fake_speed=1.0,
                 guard_kw=None, limit_kw=None, camera=None, camera_path="camera.json", blocks_path="blocks.json", key_hold=False, feeds=None, designs_dir="designs"):
        self.arms, self.quest_port, self.hand, self.probe, self.fake_speed = arms or default_arms(), int(quest_port), hand, probe, float(fake_speed)
        # a move that should meet nothing goes in pieces of up to 80 mm (the guard's own default is 30) at up to 120 mm/s
        # a move that was asked for, or was paused, waits two minutes for the switch (ten with the page's own switch) before the session is ended
        self.guard_kw, self.limit_kw = {"resume_s": 600.0 if key_hold else 120.0, "start_s": 600.0 if key_hold else 120.0, **(guard_kw or {})}, {"max_step_mm": 80.0, "max_speed_mm_s": 120.0, **(limit_kw or {})}
        self.spots_path, self.log_root, self._say = Path(spots_path), Path(log_dir), say
        self.P, self.lock, self._next = Params(), threading.RLock(), threading.Event()
        self.lines = collections.deque(maxlen=120)
        self.reach, self._gen, self._quit = {}, 0, False
        self.ended = None
        self.key_hold = bool(key_hold)  # the space bar of the page may be the hold-to-run switch of a REAL arm too (asked for when the program is started)
        self.feeds = feeds  # live pictures for the page (`feeds.Feeds`), or None: no pictures
        if feeds is not None and "stand-in" in self.arms:
            feeds.add(dict(id="made_up", label="Made-up camera", arms=("stand-in",), make=lambda: MadeUp(self._made_up_world)))
        self._sight_init(camera, camera_path, blocks_path, designs_dir)
        self._reset()
        threading.Thread(target=self._probe_loop, daemon=True).start()

    def _reset(self):
        self.arm, self.dm, self.gate, self.guard = None, None, None, None
        self.connected, self.fake, self.hold, self.chosen = False, False, None, None
        self.spots, self.calibrated, self.teaching, self.draft = None, False, False, None
        self.worker, self.job, self.thing_at, self.snap, self.quest_why, self.note = None, None, None, {}, None, ""
        self.stack = None  # the tower, once one was asked for: which spot, block height, how many blocks stand on it
        self._hz, self.log_dir = (time.monotonic(), 0, 0.0), None
        self._page = None  # the page's last report of its key: (when it came in, when the page says it sent it)

    # ------------------------------------------------------------------ plumbing
    def say(self, msg):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        self.lines.append(line)
        if self._say:
            self._say(line)

    def _probe_loop(self):
        while not self._quit:
            if not self.connected:
                for key, a in self.arms.items():
                    if a.get("ip"):
                        self.reach[key] = bool(self.probe(a["ip"]))
            time.sleep(2.0)

    def _quest(self):
        """Take the Quest's port as soon as it is free (teleoperation holds it while its dashboard is open)."""
        dm = self.dm
        if self.hold != "quest" or dm is None or dm.listening:
            return
        try:
            dm.start()
            self.quest_why = None
            self.say(f"listening for the Quest grip on port {self.quest_port}")
        except OSError:
            self.quest_why = f"port {self.quest_port} is in use: close the teleoperation dashboard"

    def _monitor(self, gen):
        """Keeps a snapshot of the arm for the dashboard. While something moves only the published state is read, and
        before arming nothing is asked of the controller at all."""
        k = 0
        while self._gen == gen and self.connected and not self._quit:
            time.sleep(0.25)
            k += 1
            try:
                if k % 4 == 1:
                    with self.lock:
                        if self._gen == gen:
                            self._quest()
                snap = dict(xyz=self.arm.xyz_now().tolist(), error=int(self.arm.fault_now()), moving=bool(self.arm.moving_now()))
                if self.worker is None and self.gate.armed and k % 3 == 0:  # ours and idle: round trips are allowed
                    snap["rpy"] = self.arm.pose()[3:].tolist()
                    snap["lag_mm"] = float(np.linalg.norm(self.arm.lag_mm()))
                with self.lock:
                    if self._gen == gen:
                        self.snap = {**self.snap, **snap}
            except Exception as e:  # noqa: BLE001  the dashboard shows it and stays up
                with self.lock:
                    self.snap = {**self.snap, "read_error": f"{type(e).__name__}: {e}"}
            t0, n0, _ = self._hz
            if self.dm is not None and time.monotonic() - t0 >= 1.0:
                self._hz = (time.monotonic(), self.dm.n, (self.dm.n - n0) / (time.monotonic() - t0))

    def close(self):
        self._quit = True
        self.end()
        if self.feeds is not None:
            self.feeds.close()

    # ------------------------------------------------------------------ OFF <-> DISARMED
    def start(self, choice, hold=None):
        """Connect to an arm. Read only: nothing is enabled, cleared or moved."""
        with self.lock:
            if self.connected:
                raise Refused("a session is already running: end it first")
            if choice not in self.arms:
                raise Refused(f"unknown arm {choice!r}")
            fake = self.arms[choice].get("ip") is None
            hold = hold or ("page" if fake else "quest")
            if hold not in ("quest", "page") or (hold == "page" and not fake and not self.key_hold):
                raise Refused("with a real arm the only hold-to-run switch is the Quest grip (unless the dashboard was started with --key_hold)")
            self._reset()
            self.ended = None
            arm = self.arms[choice]["make"]()
            try:
                arm.connect()
            except Exception as e:  # noqa: BLE001  whatever the SDK raises
                raise Refused(f"no connection to the {self.arms[choice]['label'].lower()}: {type(e).__name__}: {e}. Is it powered?") from e
            # the page reports its key twenty times a second; a browser stalls now and then, so only 0.6 s of silence counts as lost
            # ... and a silence of up to 3 s only pauses: the key need not be let go and pressed again after it
            # ... with a real arm the page may be behind another window, where a browser lets it report only once a second:
            # 2.5 s of silence is what counts as lost there (a click on the switch, the space bar and Esc are reported at once)
            dm = Deadman(timeout_s=2.5 if not fake else 0.8, forgive_s=5.0 if not fake else 0.0) if hold == "page" else QuestDeadman(port=self.quest_port, hand=self.hand)
            self.arm, self.dm, self.gate, self.fake, self.hold, self.chosen, self.connected = arm, dm, Gate(dm), fake, hold, choice, True
            self.log_dir = self.log_root / time.strftime("%Y%m%d_%H%M%S")
            self._gen += 1
            self._quest()
            if fake:
                run_fake_clock(arm, self.fake_speed, alive=lambda g=self._gen: self._gen == g and self.connected)
            threading.Thread(target=self._monitor, args=(self._gen,), daemon=True).start()
            self.say(f"session started on the {self.arms[choice]['label'].lower()}: connected, read only (DISARMED)")

    def end(self, why=None):
        """Stop the arm, let go of the connection and of the Quest's port. With `why` the label reads STOPPED."""
        with self.lock:
            worker, guard = self.worker, self.guard
            if self.gate is not None:
                self.gate.armed = False
            if worker is not None and guard is not None:
                guard.abort(why or "the session was ended", latch=True)
        if worker is not None:
            worker.join(timeout=3.0)
        with self.lock:
            if self.connected:
                try:
                    self.arm.stop()
                    self.arm.disconnect()
                except Exception as e:  # noqa: BLE001
                    self.say(f"while disconnecting: {type(e).__name__}: {e}")
                if hasattr(self.dm, "stop"):
                    self.dm.stop()
                self.say(f"SESSION STOPPED ({why})" if why else "session ended (OFF)")
            self._gen += 1
            job = self.job
            self._reset()
            self.job, self.ended = job, why

    def stop(self, why="stopped on the dashboard"):
        self.end(why)

    # ------------------------------------------------------------------ DISARMED <-> armed
    def checks(self):
        """Everything that must hold before Arm is accepted, as (text, ok, detail)."""
        s, sig = self.snap, self.dm.why()
        return [("No controller error", s.get("error", 1) == 0, f"error {s.get('error')}" if s.get("error") else "not read yet" if "error" not in s else ""),
                ("Hold-to-run signal arriving", not sig.startswith("no signal"), (self.quest_why or sig) if sig.startswith("no signal") else ""),
                ("Grip released right now", not self.dm.held(), "let go of the grip to arm")]

    def _jog_limits(self):
        """While the places are being taught: a box around where the arm is now (250 mm to every side and down, 150 up)."""
        pose = self.arm.pose()
        return Limits(pose[:3].copy(), jog_axes(pose[3:]), np.array([-250.0, -250.0, -250.0]), np.array([250.0, 250.0, 150.0]), **self.limit_kw)

    def arm_on(self):
        """Switch the arm on (position mode). Moves nothing. Without known spots the next rung is AWAITING CALIBRATION."""
        with self.lock:
            if not self.connected:
                raise Refused("start a session first")
            if self.gate.armed:
                return
            bad = [f"{t} ({d})" if d else t for t, ok, d in self.checks() if not ok]
            if bad:
                raise Refused("cannot arm: " + "; ".join(bad))
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.guard = Guard(self.arm, self.gate, self._jog_limits(), log_path=self.log_dir / "guard.jsonl", say=self.say, **self.guard_kw)
            try:
                self.guard.start()
            except ArmFault as e:
                why = self.guard.latched or str(e)
                threading.Thread(target=self.end, args=(why,), daemon=True).start()
                raise
            self.gate.armed, self.calibrated, self.spots, self.thing_at, self.stack = True, False, None, None, None
            self.teaching, self.draft, self.note = True, self.draft or Spots(side=self.chosen), ""
            self.say("armed, AWAITING CALIBRATION: nudge the gripper to each place and save it as a spot (it moves only while the grip is held)")

    def disarm(self, why="disarmed on the dashboard"):
        with self.lock:
            if not self.connected:
                return
            self.gate.armed, self.teaching = False, False
            if self.worker is not None and self.guard is not None:
                self.guard.abort(why, latch=False)
            else:
                self.arm.stop()
            self.say(f"DISARMED ({why})")

    # ------------------------------------------------------------------ calibration: nudge, grip, save spots
    def _teach(self, what):
        if not (self.connected and self.gate.armed and self.teaching):
            raise Refused(f"{what}: arm first (the label must read AWAITING CALIBRATION)")
        if self.worker is not None:
            raise Refused(f"{what}: wait until the arm has finished")

    def _small(self, name, fn):
        """Run one small action (a nudge, the gripper) in the background so that the grip, the stop button and the
        page keep working while it runs."""
        self.job = dict(steps=[dict(name=name)], i=0, waiting=False, outcome=None, done=0, total=1, src=None, dst=None, rehearse=False, step=False, kind="nudge")

        def run():
            g, note = self.guard, ""
            try:
                note = fn(g) or ""
            except Blocked as e:
                g.recover()
                note = f"blocked: something is in the way ({e})"
            except Released as e:
                note = f"stopped: {e}"
            except (ArmFault, Refused) as e:
                note = f"{type(e).__name__}: {e}"
            finally:
                latched = g.latched
                with self.lock:
                    self.job["outcome"], self.job["i"], self.worker, self.note = note or "done", None, None, note
                    if note.startswith("stopped") and self.gate is not None:
                        self.gate.armed = self.teaching = False
                if note:
                    self.say(f"{name}: {note}")
                if latched:
                    threading.Thread(target=self.end, args=(latched,), daemon=True).start()

        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def jog(self, direction, mm):
        """Nudge the gripper: up / down along its own axis, fwd / back away from and towards the robot's base, left /
        right. At most 25 mm at a time, slowly, watched (it ends at once if something is in the way)."""
        with self.lock:
            self._teach("nudging")
            mm = float(mm)
            if direction not in JOG or not 0.5 <= mm <= 60.0:
                raise Refused("a nudge is up, down, fwd, back, left or right, by 0.5 to 60 mm")
            d = self.guard.lim.axes @ (np.array(JOG[direction], float) * mm)
            pose = self.arm.pose()
            self._reach([pose[:3] + d], pose[3:], f"nudge {direction}")
            if direction == "up":  # away from everything: in one piece, not watched
                self._small(f"nudge up {mm:g} mm", lambda g: g.move_to(g.where() + d, self.P.lift, "nudge up"))
            else:  # coming down is checked every 3 mm, going sideways every 10 mm
                inch = None if direction == "down" else 10.0
                self._small(f"nudge {direction} {mm:g} mm", lambda g: g.move_to(g.where() + d, self.P.slow, f"nudge {direction}", stall_mm=self.P.stall_mm, inch_mm=inch))

    def turn(self, direction, deg):
        """Spin the gripper about its own axis (only the last joint moves): "ccw" or "cw", 1 to 90 degrees."""
        with self.lock:
            self._teach("turning")
            deg = float(deg)
            if direction not in ("ccw", "cw") or not 1.0 <= deg <= 90.0:
                raise Refused("a turn is ccw or cw, by 1 to 90 degrees")
            self._small(f"turn {direction} {deg:g} deg", lambda g: g.turn_by(deg if direction == "ccw" else -deg, f"turn {direction}"))

    def level(self):
        """Tilt the gripper until it points straight down at the table, the flange staying where it is. Before any
        spot is saved (all spots share one tilt). The fingertips swing sideways while it tilts: keep the space free."""
        with self.lock:
            self._teach("pointing the gripper down")
            if self.draft is not None and self.draft.at:
                raise Refused("remove the saved spots first: every spot is taught with the gripper tilted the same way")

            def act(g):
                deg = g.level(down=self.down())
                g.lim = self._jog_limits()  # forward, left and up are counted from the gripper's new axis
                return f"the gripper points straight down now (it was tilted {deg:.0f} deg)" if deg else "the gripper already points straight down"

            self._small("point the gripper down", act)

    def _reach(self, points, rpy, what):
        """Refuse `what` unless the arm can be at every one of these points (asked of the controller; nothing moves)."""
        seen = set()
        for p in points:
            key = tuple(np.round(np.asarray(p, float), 1))
            if key in seen:
                continue
            seen.add(key)
            why = self.arm.reach(p, rpy)
            if why:
                raise Refused(f"{what}: {why} (at {[round(float(v)) for v in p]} mm). Nothing was moved: bring the area closer to the arm")

    def _empty_handed(self, what):
        """Refuse `what` if the jaws are not open: they may still hold a block, which would be dropped."""
        counts = self.arm.status().get("gripper")
        if counts is not None and counts < self.P.open_counts - 150:
            raise Refused(f"{what}: the gripper is not open (it may still hold a block). Use 'Put it down' first, or open it")

    def down(self):
        """Straight down in the chosen arm's base frame (the arms of this rig are mounted at an angle)."""
        return np.asarray(self.arms[self.chosen].get("down", (0.0, 0.0, -1.0)), float)

    def find_table(self, wide_mm=200.0, deep_mm=140.0, here=False, as_before=False):
        """Instead of teaching spots by hand. With a bar held in the jaws (lying as it would lie on the table, the
        jaws across its short side) and the gripper pointing straight down over the middle of the area the blocks will
        lie in: come down slowly until the bar meets the table, let go of it there, and make the spots from that:
          T   where the bar was put down: the height at which blocks are taken, and where the tower will stand;
          P, Q  two opposite corners of the area, `wide_mm` across (left to right) and `deep_mm` deep, around T.
        The label then reads ARMED with the bar on T, ready for the camera to be shown the table.
        `here`: the gripper was brought down by hand and its fingertips touch the table now. Nothing comes down: it
        is raised by a millimetre and a half, lets go of the bar, and the spots are made at that height.
        `as_before`: the table's height is known from the spots saved earlier, and only the place is new: the gripper
        comes straight down from wherever it is to that height (watched for contact) and lets go of the bar there."""
        with self.lock:
            self._teach("finding the table")
            wide, deep = float(wide_mm), float(deep_mm)
            if not (100.0 <= wide <= 400.0 and 100.0 <= deep <= 400.0):
                raise Refused("the area is 100 to 400 mm each way")
            if self.draft is not None and self.draft.at:
                raise Refused("remove the saved spots first")
            pose = self.arm.pose()
            tilt = float(np.degrees(np.arccos(np.clip(tool_axis(pose[3:]) @ self.down(), -1.0, 1.0))))
            if tilt > 1.5:
                raise Refused(f"the gripper is tilted {tilt:.0f} deg from straight down: press 'Point the gripper down' first")
            axes, by_hand, height = jog_axes(pose[3:]), bool(here), None
            counts = self.arm.status().get("gripper")
            if counts is None or counts > self.P.open_counts - 150:  # the jaws are not closed again here: that would squeeze the bar once more
                raise Refused("the gripper is open: hold a bar between the jaws and press Close gripper first")
            if as_before:
                try:
                    old = Spots.load(self.spots_path)
                    height = float(old.grasp("T") @ old.up)
                except (OSError, ValueError, KeyError, TypeError, Refused):
                    raise Refused("there is no table height from before: use 'Find the table' or 'The gripper is at the table now'") from None
                if old.tilt_from(pose[3:]) > 2.0:
                    raise Refused("the gripper is tilted differently than when the table was found: find the table again")
                drop = float(pose[:3] @ axes[:, 2]) - height
                if not -5.0 <= drop <= 440.0:
                    raise Refused(f"the table's height from before is {drop:.0f} mm below the gripper: that cannot be right here, find the table again")

            def act(g):
                start = g.where()
                g.lim = Limits(start.copy(), axes, np.array([-15.0, -15.0, -450.0]), np.array([15.0, 15.0, 30.0]), **self.limit_kw)
                if height is not None:
                    try:
                        g.move_to(start + (height - float(start @ axes[:, 2])) * axes[:, 2], self.P.slow, "come down to the table's height", stall_mm=self.P.stall_mm)
                    except Blocked:  # the table is a little higher here than it was there
                        g.recover()
                        g.move_to(g.where() + 4.0 * axes[:, 2], self.P.press, "ease off the table")
                elif by_hand:
                    g.move_to(start + 1.5 * axes[:, 2], self.P.press, "ease off the table")  # the fingertips touch it: a little air under them
                else:
                    try:
                        g.move_to(start - 440.0 * axes[:, 2], self.P.press, "come down until the bar meets the table", stall_mm=2.0)
                    except Blocked:
                        g.recover()
                    else:
                        g.move_to(start, self.P.lift, "back up: no table was met")
                        g.lim = self._jog_limits()
                        raise Refused("no table within 440 mm below the gripper")
                    g.move_to(g.where() + 4.0 * axes[:, 2], self.P.press, "ease off the table")  # the plan was a few mm into the table: now the bar just rests on it
                g.grip(self.P.open_counts, "let go of the bar")
                here = self.arm.pose()
                with self.lock:
                    for part in (1.0, 0.8, 0.6, 0.45, 0.3):  # the corners are pulled in towards the bar until the arm can reach both
                        d = Spots(side=self.chosen)
                        d.add("T", here)
                        for name, sign in (("P", -1.0), ("Q", 1.0)):
                            d.add(name, np.r_[here[:3] + sign * part * (deep / 2.0 * axes[:, 0] + wide / 2.0 * axes[:, 1]), here[3:]])
                        if not any(self.arm.reach(q + h * axes[:, 2], here[3:]) for q in d.at.values() for h in (0.0, d.lift)):
                            break
                    else:
                        raise Refused("the bar was put down, but this place is at the very end of the arm's reach: pick it up again and go back towards the arm")
                    across, depth = part * wide, part * deep
                    d.check().save(self.spots_path)
                    self._adopt(d, thing_at="T")
                came = float((start - here[:3]) @ axes[:, 2])
                self.say(f"ARMED: the table is {came:.0f} mm below where the gripper was. The bar lies on T; the corners P and Q are {across:.0f} mm across and {depth:.0f} mm deep around it")
                return "the area is here now, at the table's height from before; the bar lies on spot T" if height is not None else "the table is where the gripper was; the bar lies on spot T" if by_hand else f"the table was met {came:.0f} mm down; the bar lies on spot T"

            self._small("find the table", act)

    def grip(self, close, width_mm=None):
        """Open the gripper, or close it. With `width_mm` (how wide the thing between the jaws is) it closes to just
        under that width and no further; without, it closes as far as it goes and says how wide the thing is."""
        with self.lock:
            self._teach("the gripper")
            gentle = None
            if close and width_mm not in (None, "", 0):
                width_mm = float(width_mm)
                if not 5.0 <= width_mm <= 84.0:
                    raise Refused("the width of the block is between 5 and 84 mm")
                gentle = max(0, int(round((width_mm - self.P.squeeze_mm - self.P.jaw_offset_mm) * 10.0)))

            def act(g):
                if gentle is not None:
                    g.grip(gentle, "close the gripper, gently")
                    return f"closed to {gentle / 10.0 + self.P.jaw_offset_mm:.0f} mm, for a block {width_mm:g} mm wide"
                got = g.grip(self.P.shut_counts if close else self.P.open_counts, "close the gripper" if close else "open the gripper")
                return "" if not close else (f"holding something {got / 10.0:.0f} mm wide" if got >= self.P.held_min_counts else "closed on nothing")

            self._small("close the gripper" if close else "open the gripper", act)

    def save_spot(self, name):
        with self.lock:
            self._teach("saving a spot")
            pose = self.arm.pose()
            self.draft.add(name, pose)
            self.say(f"spot {str(name).strip()} saved at {pose[:3].round(1).tolist()}")

    def remove_spot(self, name):
        with self.lock:
            if self.teaching and self.draft is not None:
                self.draft.remove(name)

    def set_lift(self, mm):
        with self.lock:
            self._teach("the lift")
            mm = float(mm)
            if not 30.0 <= mm <= 200.0:
                raise Refused("the lift is between 30 and 200 mm")
            self.draft.lift = mm

    def _adopt(self, spots, thing_at=None):
        self.spots, self.calibrated, self.teaching, self.draft, self.thing_at, self.note, self.stack = spots, True, False, None, thing_at, "", None
        self.guard.lim = spots.limits(**self.limit_kw)

    def calibrate_done(self):
        with self.lock:
            self._teach("finishing")
            self.draft.check()
            self.draft.save(self.spots_path)
            self._adopt(self.draft)
            self.say(f"ARMED: spots {', '.join(sorted(self.spots.at))} are known; travel height {self.spots.lift:.0f} mm above the highest")

    def calibrate_begin(self):
        """From ARMED: teach again (the label goes back to AWAITING CALIBRATION)."""
        with self.lock:
            if not (self.connected and self.gate.armed and self.calibrated and self.worker is None):
                raise Refused("teaching again: only when ARMED and idle")
            self.draft, self.teaching, self.calibrated = copy.deepcopy(self.spots), True, False
            self.guard.lim = self._jog_limits()
            self.say("AWAITING CALIBRATION: teach the spots again")

    def saved(self):
        if not self.spots_path.exists():
            return None
        try:
            s = Spots.load(self.spots_path).check()
            return dict(names=sorted(s.at), when=time.strftime("%d %b %H:%M", time.localtime(self.spots_path.stat().st_mtime)), side=s.side)
        except (Refused, ValueError, KeyError, TypeError, OSError):
            return None

    def use_saved(self):
        """Take the spots of an earlier session, if the arm is where they are and turned as it was then."""
        with self.lock:
            self._teach("using saved spots")
            s = Spots.load(self.spots_path).check()
            pose = self.arm.pose()
            tilt = s.tilt_from(pose[3:])
            if tilt > 2.0:
                raise Refused(f"the gripper is tilted {tilt:.1f} deg away from how the saved spots were taught: teach them again")
            why = s.limits(start=pose[:3]).why_outside(pose[:3])  # near enough is enough: up to 250 mm beside the spots
            if why:
                raise Refused("the arm is not over the saved spots: nudge it there, or teach them again")
            self._adopt(s)
            self.say(f"ARMED with the saved spots {', '.join(sorted(s.at))}: check that nothing has been moved since")

    def use_demo(self):
        with self.lock:
            self._teach("made-up spots")
            if not self.fake:
                raise Refused("made-up spots are for the stand-in arm only")
            s, a = demo_spots(), self.arm
            a.p = s.above(s.grasp("B"))
            a.plan, a.rpy, a.up, a.bricks = a.p.copy(), s.rpy.copy(), s.up.copy(), {"thing": s.grasp("A").copy()}
            self._adopt(s, thing_at="A")
            self.say("ARMED: stand-in arm placed over made-up spots A, B, C and T, with a thing standing on A")

    # ------------------------------------------------------------------ the job: move the thing around
    def set_thing(self, spot):
        with self.lock:
            if not self.calibrated or spot not in self.spots.at:
                raise Refused(f"there is no spot called {spot!r}")
            self.thing_at = spot

    def move(self, dst=None, count=1, rehearse=False, step=False):
        """Put the thing on `dst`, or (dst None) on a spot picked at random, `count` times over."""
        with self.lock:
            if self.phase() != "ARMED":
                raise Refused(f"a move can only be started when ARMED (now: {self.phase()})")
            if self.thing_at is None:
                raise Refused("say first on which spot the thing stands")
            count = int(count)
            if not 1 <= count <= 500:
                raise Refused("between 1 and 500 moves")
            if dst is not None:
                self.spots.grasp(dst)
                if dst == self.thing_at:
                    raise Refused("the thing is already there")
            if not rehearse:
                self._empty_handed("a move")
            self.job = dict(steps=[], i=None, waiting=False, outcome=None, done=0, total=count, src=self.thing_at, dst=dst, rehearse=bool(rehearse), step=bool(step), fixed=dst, kind="move")
            self._next.clear()
            self.worker = threading.Thread(target=self._work, daemon=True)
            self.worker.start()

    def tower(self, on, block_mm, what="build", rounds=1, n_now=0, rehearse=False, step=False):
        """A tower on spot `on`, from the blocks that stand on the other spots (one each, taken in name order).
        The first block goes onto the taught spot itself, each next one `block_mm` higher, let go a little above the
        one below. `n_now` says how many already stand there. `what`: "build" the rest, take it "down" again (top
        first, every block back to its own spot), or "loop": build and take down `rounds` times over."""
        with self.lock:
            if self.phase() != "ARMED":
                raise Refused(f"a tower can only be started when ARMED (now: {self.phase()})")
            self.spots.grasp(on)
            feeds = [n for n in sorted(self.spots.at) if n != on][:6]
            n, h = len(feeds), float(block_mm)
            if not 10.0 <= h <= 80.0:
                raise Refused("the block height is between 10 and 80 mm")
            if n_now is None or int(n_now) != float(n_now) or not 0 <= int(n_now) <= n:
                raise Refused(f"say how many blocks stand on the tower now: 0 to {n}")
            n_now, rounds = int(n_now), int(rounds)
            if what not in ("build", "down", "loop") or not 1 <= rounds <= 50:
                raise Refused("a tower job is build, down or loop, with 1 to 50 rounds")
            if (n - 1) * h + self.spots.lift > 300.0:
                raise Refused(f"{n} blocks of {h:g} mm plus the lift is more than 300 mm: use fewer spots or a smaller lift")
            put = lambda k: dict(src=feeds[k], dst=on, su=0.0, du=k * h + (self.P.drop_mm if k else 0.0), text=f"block from {feeds[k]} onto {on}, level {k + 1}")  # noqa: E731
            take = lambda k: dict(src=on, dst=feeds[k], su=k * h, du=0.0, text=f"top block of {on} (level {k + 1}) back to {feeds[k]}")  # noqa: E731
            build, down = [put(k) for k in range(n_now, n)], [take(k) for k in reversed(range(n_now))]
            if what == "build":
                legs = build
            elif what == "down":
                legs = down
            else:
                full = [put(k) for k in range(n)] + [take(k) for k in reversed(range(n))]
                legs = build + [take(k) for k in reversed(range(n))] + full * (rounds - 1)
            if not legs:
                raise Refused("nothing to do: the tower is complete" if what == "build" else "nothing to do: no block stands on the tower")
            sp = copy.deepcopy(self.spots)
            sp.extra = (n - 1) * h  # travel higher by the tower's height, so that a carried block clears it
            self.stack, self.thing_at = dict(on=on, block_mm=h, n=n_now, of=n, feeds=feeds), None
            if self.fake and not rehearse:  # the stand-in world is made to match what was said
                self.arm.holding = None
                self.arm.bricks = {**{f: sp.grasp(f).copy() for f in feeds[n_now:]}, **{f: sp.grasp(on) + k * h * sp.up for k, f in enumerate(feeds[:n_now])}}
            self.job = dict(steps=[], i=None, waiting=False, outcome=None, done=0, total=len(legs), src=legs[0]["src"], dst=legs[0]["dst"], rehearse=bool(rehearse), step=bool(step),
                            fixed=None, kind="move", legs=legs, spots=sp, text=legs[0]["text"], tower=True)
            self._next.clear()
            self.worker = threading.Thread(target=self._work, daemon=True)
            self.worker.start()

    def page_hold(self, pressed, sent_ms=None):
        """The page's report of its key. A report that comes late is looked at: did the PAGE send late (the browser
        stalled) or was it sent on time and held up on the way in (this program stalled)? Said once per stall."""
        now, last = time.monotonic(), self._page
        self._page = (now, sent_ms)
        self.dm.feed(bool(pressed))
        if last is not None and now - last[0] > 0.4 and sent_ms is not None and last[1] is not None:
            late = (float(sent_ms) - float(last[1])) / 1000.0
            self.say(f"  the hold-to-run signal was silent for {now - last[0]:.1f} s: " + ("the page itself sent nothing for that long (the browser stalled)" if late > 0.6 * (now - last[0])
                     else f"the page had sent on time ({late:.2f} s apart), the reports were held up on the way into this program"))

    def put_held(self, dst):
        """When ARMED and the gripper still holds a block (a run was stopped with it in the air): put it down on the
        spot `dst`, watched for contact, and let go. Afterwards the block stands on `dst`."""
        with self.lock:
            if self.phase() != "ARMED":
                raise Refused(f"only when ARMED (now: {self.phase()})")
            self.spots.grasp(dst)
            sp, here, P = self.spots, self.guard.where(), self.P
            gd = sp.grasp(dst)
            steps = [dict(name="go to the travel height", kind="move", to=sp.above(here), speed=P.lift),
                     dict(name=f"go above {dst}", kind="move", to=sp.above(gd), speed=P.travel),
                     dict(name=f"turn the gripper as at {dst}", kind="turn", rpy=sp.turn(dst)),
                     dict(name=f"lower it onto {dst}", kind="move", to=gd, speed=P.slow, stall_mm=P.stall_mm, rest_ok_mm=P.rest_ok_mm),
                     dict(name="let go", kind="grip", counts=P.open_counts),
                     dict(name="lift clear", kind="move", to=sp.above(gd), speed=P.lift)]
            self._reach([s["to"] for s in steps if s["kind"] == "move"], sp.rpy, f"putting it down on {dst}")
            self.job = dict(steps=steps, i=None, waiting=False, outcome=None, done=0, total=1, src=None, dst=dst, rehearse=False, step=False, kind="move", text=f"the held block onto {dst}")
            self._next.clear()

            def run():
                g, job, outcome = self.guard, self.job, "done"
                top = float(here @ sp.up) + 1.0
                g.lim = sp.limits(ceil_w=top, start=here, **self.limit_kw)
                try:
                    rec = execute(g, steps, P, before=lambda s: job.update(i=next(i for i, x in enumerate(steps) if x is s)))
                    if rec["blocked"]:
                        outcome = f"blocked at '{rec['blocked']['step']}': lifted clear, still holding it"
                    else:
                        self.thing_at, job["done"] = dst, 1
                except Released as e:
                    outcome = f"stopped: {e}"
                except (ArmFault, Refused) as e:
                    outcome = f"{type(e).__name__}: {e}"
                finally:
                    latched = g.latched
                    with self.lock:
                        job["outcome"], job["i"], self.worker = outcome, None, None
                        if outcome.startswith("stopped") and self.gate is not None:
                            self.gate.armed = self.teaching = False
                    self.say(f"ended: {outcome}")
                    if latched:
                        threading.Thread(target=self.end, args=(latched,), daemon=True).start()

            self.say(f"putting the held block down on {dst}")
            self.worker = threading.Thread(target=run, daemon=True)
            self.worker.start()

    def next_step(self):
        self._next.set()

    def _work(self):
        job, g = self.job, self.guard
        legs, sp = job.get("legs"), job.get("spots") or self.spots  # a tower job brings its own legs and a higher travel height
        rng, at, tower = np.random.default_rng(), self.thing_at, bool(job.get("tower"))
        if self.sight is not None:
            self.sight["stale"] = True  # what the camera saw is no longer what is there

        def lost():  # after a move that did not end as planned, nobody knows where the block is
            if tower:
                self.stack = {**self.stack, "n": None}
            else:
                self.thing_at = None

        def before(s):
            job["i"] = next((i for i, x in enumerate(job["steps"]) if x is s), job["i"])
            if job["step"]:
                job["waiting"] = True
                while not self._next.wait(0.05):
                    g._check_abort()
                self._next.clear()
                job["waiting"] = False

        outcome = "done"
        try:
            for k in itertools.count():
                if legs is not None and k >= len(legs) and job.get("more"):  # a job that decides its next move from what it has seen so far
                    nxt = job["more"]()
                    if nxt is not None:
                        legs.append(nxt)
                    job["total"] = max(job["total"], len(legs)) if nxt is not None else len(legs)
                if k >= (len(legs) if legs is not None else job["total"]):
                    break
                leg = legs[k] if legs else dict(src=at, su=0.0, du=0.0, dst=job["fixed"] if job["fixed"] is not None and k == 0 else str(rng.choice([n for n in sorted(sp.at) if n != at])))
                at, dst = leg["src"], leg["dst"]
                text = leg.get("text") or f"{at} to {dst}"
                if leg.get("prepare"):  # a camera job: the block is taken where the newest picture shows it
                    leg["prepare"]()
                tries = 0
                while True:  # the same block again, the jaws closed a little further, if the gripper's camera saw it stay behind
                    here = g.where()
                    steps = plan_move(sp, at, dst, here, self.P, rehearse=job["rehearse"], src_up=leg["su"], dst_up=leg["du"], release_counts=leg.get("release"), grip_mm=leg.get("width"),
                                      tighter_mm=1.5 * tries)
                    self._reach([s["to"] for s in steps if s["kind"] == "move"], sp.rpy, text)
                    top = float(here @ sp.up) + 1.0  # the arm may start above the travel height (after a tower): it comes down first
                    g.lim = sp.limits(floor_w=min(float(here @ sp.up), sp.travel_w()) - 1.0, ceil_w=top, start=here, **self.limit_kw) if job["rehearse"] else sp.limits(ceil_w=top, start=here, **self.limit_kw)
                    with self.lock:
                        job.update(steps=steps, i=None, src=at, dst=dst, text=text)
                    if not tries:
                        self.say(f"move {k + 1} of {job['total']}: {text}" + (" (rehearsal in the air)" if job["rehearse"] else ""))
                        if leg.get("first"):
                            leg["first"]()
                    try:
                        rec = execute(g, steps, self.P, air=job["rehearse"], before=before, after=self._grip_watch(job, leg))
                        break
                    except NotHeld as e:
                        g.grip(self.P.open_counts, "open again: the block stayed behind")
                        tries += 1
                        if tries > 2:
                            raise NothingHeld(f"{e}, three times over, the jaws closed up to 3 mm further") from None
                        self.say(f"  {e}: once more, the jaws closed {1.5 * tries:g} mm further")
                with open(self.log_dir / "moves.jsonl", "a") as fh:
                    fh.write(json.dumps(dict(t=time.time(), src=at, dst=dst, src_up=leg["su"], dst_up=leg["du"], rehearse=job["rehearse"], **rec), default=float) + "\n")
                if rec["blocked"]:
                    outcome = f"blocked at '{rec['blocked']['step']}': lifted clear. Look, then say where " + ("the blocks are" if legs else "the thing is")
                    lost()
                    break
                if rec["rested"]:
                    self.say(f"it came to rest {rec['rested']['short_mm']:.1f} mm above the planned height and was let go there")
                at = dst
                job["done"] = k + 1
                if not job["rehearse"]:
                    if tower:
                        self.stack = {**self.stack, "n": self.stack["n"] + (1 if leg.get("adds", dst == self.stack["on"]) else -1)}
                    else:
                        self.thing_at = dst
                if leg.get("then"):
                    leg["then"]()
            if outcome == "done" and job.get("finish"):
                job["finish"]()
        except NothingHeld as e:
            outcome = f"{e}. Say where " + ("the blocks are" if legs else "the thing is")
            lost()
        except NotAsPlanned as e:  # the arm is fine; what was built is not as planned
            outcome = f"the camera: {e}. Look at the build before going on"
            lost()
        except Released as e:
            outcome = f"stopped: {e}"
            if not job["rehearse"]:
                lost()
        except (ArmFault, Refused) as e:
            outcome = f"{type(e).__name__}: {e}"
            if not job["rehearse"]:
                lost()
        except Exception as e:  # noqa: BLE001  a mistake in the program itself: stop the arm and say so, never report "done"
            outcome = f"stopped: the program failed ({type(e).__name__}: {e})"
            lost()
            try:
                self.arm.stop()
            except Exception:  # noqa: BLE001
                pass
            raise
        finally:
            latched = g.latched
            with self.lock:
                job["outcome"], job["i"], job["waiting"], self.worker = outcome, None, False, None
                if outcome.startswith("stopped") and self.gate is not None:
                    self.gate.armed = self.teaching = False
            self.say(f"ended: {outcome}")
            if latched:
                threading.Thread(target=self.end, args=(latched,), daemon=True).start()

    # ------------------------------------------------------------------ what the dashboard draws (nothing here moves or judges anything)
    def view(self):
        """Ten times a second, for the page's model of the arm: the joint angles the controller publishes anyway (no
        round trip is made for a drawing), and where the gripper last came to rest."""
        arm, g = self.arm, self.guard
        if not self.connected or arm is None:
            return dict(connected=False)
        try:
            grip = g.counts if g is not None and g.counts is not None else getattr(arm, "counts", None)
            return dict(connected=True, joints=[round(float(v), 2) for v in arm.joints_view()], grip=grip, moving=bool(arm.moving_now()))
        except Exception as e:  # noqa: BLE001  a drawing must never take the program down
            return dict(connected=True, joints=None, error=f"{type(e).__name__}: {e}")

    def scene_state(self):
        """What is around the arm, for the page's model, in the arm's base frame [mm]: which way the gripper's 'up'
        points, the box the guard keeps the flange in, where the steps of the current move lead, the stand-in's
        blocks, and the blocks the camera made out. Spots, box and path are FLANGE positions; the page draws them
        `tool_mm` further along the gripper, where the fingertips are."""
        out = dict(tool_mm=TOOL_MM, up=None, box=None, path=None, at=None, things=[], seen=[], plan=[])
        if not self.connected:
            return out
        sp = self.draft if self.teaching else self.spots
        if sp is not None and sp.rpy is not None:
            up = sp.up
        elif "rpy" in self.snap:
            up = -tool_axis(self.snap["rpy"])
        else:
            up = -self.down()
        out["up"] = [float(v) for v in up]
        g, j = self.guard, self.job
        if g is not None and self.gate.armed:
            lim = g.lim
            out["box"] = dict(origin=lim.origin.tolist(), axes=lim.axes.tolist(), lo=lim.lo.tolist(), hi=lim.hi.tolist())
        if j is not None and j.get("kind") == "move" and self.worker is not None:
            out["path"], out["at"] = [None if x.get("kind") != "move" else [round(float(v), 1) for v in x["to"]] for x in j["steps"]], j["i"]
        if j is not None and j.get("plan") and (self.worker is not None or j["done"] < j["total"]):
            out["plan"] = [dict(e, done=k < j["done"], next=k == j["done"]) for k, e in enumerate(j["plan"])]  # a design being built: where every block goes
        if self.fake:
            out["things"] = [dict(name=k, xyz=[round(float(x), 1) for x in v]) for k, v in sorted(self.arm.bricks.items())]
        s, setup = self.sight, self.setup
        if s is not None and setup is not None:
            for b in s["seen"]:
                a = np.radians(b.angle)
                out["seen"].append(dict(name=b.name, xyz=[round(float(v), 1) for v in setup.table.point(b.uv[0], b.uv[1], setup.w_grasp)],
                                        along=[float(v) for v in setup.table.axes @ np.array([np.cos(a), np.sin(a), 0.0])],
                                        size=[b.rest.long, b.rest.short, b.rest.height], round=bool(b.rest.round), colour=b.kind.colour, stale=bool(s["stale"])))
        return out

    def frames(self, after=0, side=None):
        """The live pictures of the cameras that belong to the arm of this session (before a session: to the arm
        picked on the page). Returns what the page is told about each camera, and the new pictures themselves."""
        if self.feeds is None:
            return dict(on=False, why="the live pictures are switched off", seq=int(after), cams=[]), []
        why = self.feeds.why_not()
        if why:
            return dict(on=False, why=why, seq=int(after), cams=[]), []
        seq, cams = self.feeds.frames(self.chosen if self.connected else (side if side in self.arms else None), int(after))
        look = getattr(self.camera, "device", None) or TABLE_CAMERA
        pictures = [c.pop("jpg") for c in cams]
        for c, jpg in zip(cams, pictures):
            c["len"], c["finds"] = len(jpg), same_device(c.pop("device"), look)
        return dict(on=True, why="", seq=seq, cams=cams), pictures

    def _made_up_world(self):
        """What the stand-in's made-up camera draws (None when there is no stand-in session)."""
        arm = self.arm
        if arm is None or not self.fake:
            return None
        sp = self.draft if self.teaching else self.spots
        spots = {} if sp is None else {k: v.copy() for k, v in sp.at.items()}
        return dict(axes=jog_axes(arm.rpy), centre=np.mean(list(spots.values()), 0) if spots else None, gripper=arm.p.copy(), open=arm.counts >= 800, spots=spots,
                    things={k: v.copy() for k, v in arm.bricks.items()})

    # ------------------------------------------------------------------ what the dashboard shows
    def phase(self):
        if not self.connected:
            return "STOPPED" if self.ended else "OFF"
        if not self.gate.armed:
            return "DISARMED"
        if self.teaching:
            return "AWAITING CALIBRATION"
        if self.worker is None:
            return "ARMED"
        return "RUNNING" if (self.gate.held() and not (self.job or {}).get("waiting")) else "PAUSED"

    def hint(self):
        text = self._hint()
        if self.hold == "page":  # the page's own key is switched on and off, not held
            for old, new in (("Moving while the grip is held. Let go to pause.", "Moving. The MOVING button, the space bar or Esc stop it."), ("LET GO of the switch once, then hold it again.", "Click the MOVING button again."),
                             ("press Go, and hold the grip", "press Go, then switch MOVING on"), ("hold the grip, nudge", "switch MOVING on (the button at the top), nudge"), ("Paused: hold the grip", "Paused: switch MOVING on to go on")):
                text = text.replace(old, new)
        return text

    def _hint(self):
        ph = self.phase()
        if ph in ("OFF", "STOPPED"):
            return "Choose an arm and start a session."
        if ph == "DISARMED":
            return "When every check is ticked, press Arm. Arming moves nothing."
        if ph == "AWAITING CALIBRATION":
            return "Show it the places: hold the grip, nudge the open gripper around the block, save that spot. Do the same where the block should go. Then press Done."
        if ph == "ARMED":
            return "One block: say where it stands and where it goes, press Go, and hold the grip. Or build a tower from the blocks on the other spots."
        if ph == "RUNNING":
            return "Moving while the grip is held. Let go to pause."
        if self.dm.why().startswith("release"):
            return "Paused: the signal was lost for a moment. LET GO of the switch once, then hold it again."
        return "Paused: hold the grip" + (" and press Next step." if (self.job or {}).get("waiting") else ".")

    def state(self):
        with self.lock:
            ph, s, j = self.phase(), dict(self.snap), self.job
            sp = self.draft if self.teaching else self.spots
            spots = [] if sp is None else [dict(name=k, mm=[round(float(x), 1) for x in v], yaw=round(float(sp.turn(k)[2]), 0)) for k, v in sorted(sp.at.items())]
            out = dict(phase=ph, hint=self.hint(), ended=self.ended, fake=self.fake, hold=self.hold, chosen=self.chosen, connected=self.connected, key_hold=self.key_hold,
                       arms=[dict(key=k, label=a["label"], ip=a.get("ip"), answers=None if a.get("ip") is None else self.reach.get(k),
                                  up=[-float(v) for v in a.get("down", (0.0, 0.0, -1.0))]) for k, a in self.arms.items()],
                       scene=self.scene_state(), feeds=self.feeds is not None,
                       calibration=dict(teaching=self.teaching, done=self.calibrated, spots=spots, lift_mm=None if sp is None else sp.lift, saved=self.saved(), note=self.note,
                                        busy=bool(self.teaching and self.worker is not None)),
                       thing_at=self.thing_at, log=list(self.lines)[-40:], stack=self.stack, drop_mm=self.P.drop_mm, eye=self.sight_state(),
                       job=None if j is None or j.get("kind") != "move" else dict(steps=[x["name"] for x in j["steps"]], i=j["i"], waiting=j["waiting"], outcome=j["outcome"], done=j["done"],
                                                                                  total=j["total"], src=j["src"], dst=j["dst"], rehearse=j["rehearse"], text=j.get("text")))
            if self.connected:
                if self.calibrated and "xyz" in s:
                    s["height_mm"] = round(float(np.asarray(s["xyz"]) @ self.spots.up - min(float(p @ self.spots.up) for p in self.spots.at.values())), 1)
                if "rpy" in s:
                    s["tilt_deg"] = round(float(np.degrees(np.arccos(np.clip(tool_axis(s["rpy"]) @ self.down(), -1.0, 1.0)))), 1)
                out.update(arm=s, checks=[dict(text=t, ok=bool(ok), detail=d) for t, ok, d in self.checks()],
                           quest=dict(held=self.dm.held(), why=self.quest_why or self.dm.why(), frames=self.dm.n, hz=round(self._hz[2], 1), grip=getattr(self.dm, "last_grip", None), gap_ms=round(1000.0 * self.dm.worst_gap())))
            return out
