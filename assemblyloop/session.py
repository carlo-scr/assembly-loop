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
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np

from .arms import ArmFault, FakeArm, XArm, run_fake_clock
from .closed import Closed
from .cycle import NothingHeld, Params, execute
from .deadman import Deadman, QuestDeadman
from .feeds import TABLE_CAMERA, MadeUp, same_device
from .guard import Blocked, Guard, Limits, Refused, Released
from .sight import Sight
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


class Session(Sight, Closed):
    def __init__(self, arms=None, quest_port=6666, hand="right", spots_path="spots.json", log_dir="assembly_logs", probe=answers, say=None, fake_speed=1.0,
                 guard_kw=None, limit_kw=None, camera=None, camera_path="camera.json", blocks_path="blocks.json", key_hold=False, feeds=None, rig_path="rig.json",
                 record=True, record_hz=10.0):
        self.arms, self.quest_port, self.hand, self.probe, self.fake_speed = arms or default_arms(), int(quest_port), hand, probe, float(fake_speed)
        # a move that should meet nothing goes in pieces of up to 80 mm (the guard's own default is 30) at up to 120 mm/s
        # a paused move waits 45 s for the switch (the guard's own default is 15) before the session is ended
        self.guard_kw, self.limit_kw = {"resume_s": 45.0, **(guard_kw or {})}, {"max_step_mm": 80.0, "max_speed_mm_s": 120.0, **(limit_kw or {})}
        self.spots_path, self.log_root, self._say = Path(spots_path), Path(log_dir), say
        self.P, self.lock, self._next = Params(), threading.RLock(), threading.Event()
        self.lines = collections.deque(maxlen=120)
        self.reach, self._gen, self._quit = {}, 0, False
        self.ended = None
        self.key_hold = bool(key_hold)  # the space bar of the page may be the hold-to-run switch of a REAL arm too (asked for when the program is started)
        self.feeds = feeds  # live pictures for the page (`feeds.Feeds`), or None: no pictures
        self.rec, self.recording, self.record_hz = None, bool(record), float(record_hz)  # the recorder of the session that is running (`record.Recorder`), and whether jobs are recorded
        self.rig_path, self.other, self.other_side, self.watch = Path(rig_path), None, None, None  # the two arms' places (`pair`), the OTHER arm (read only) and the watch on it
        self.check_width, self.keep_looks, self.tower_levels, self.left_out = True, True, None, 0  # see `closed`
        if feeds is not None and "stand-in" in self.arms:
            feeds.add(dict(id="made_up", label="Made-up camera", arms=("stand-in",), make=lambda: MadeUp(self._made_up_world)))
        self._sight_init(camera, camera_path, blocks_path)
        self._reset()
        threading.Thread(target=self._probe_loop, daemon=True).start()

    def _reset(self):
        self.arm, self.dm, self.gate, self.guard = None, None, None, None
        self.connected, self.fake, self.hold, self.chosen = False, False, None, None
        self.spots, self.calibrated, self.teaching, self.draft = None, False, False, None
        self.worker, self.job, self.thing_at, self.snap, self.quest_why, self.note = None, None, None, {}, None, ""
        self.stack = None  # the tower, once one was asked for: which spot, block height, how many blocks stand on it
        self._hz, self.log_dir = (time.monotonic(), 0, 0.0), None

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
            dm = Deadman(timeout_s=0.6 if not fake else 0.3) if hold == "page" else QuestDeadman(port=self.quest_port, hand=self.hand)
            self.arm, self.dm, self.gate, self.fake, self.hold, self.chosen, self.connected = arm, dm, Gate(dm), fake, hold, choice, True
            self.log_dir = self.log_root / time.strftime("%Y%m%d_%H%M%S")
            self.setup, self._setup_read, self.sight = None, False, None  # the camera's fit is this arm's own
            self._watch_other()
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
            if self.connected:
                try:
                    self.arm.stop()  # now, not after the worker has been waited for
                except Exception as e:  # noqa: BLE001
                    self.say(f"while stopping: {type(e).__name__}: {e}")
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
                if self.rec is not None:
                    try:
                        self.rec.end(why or "the session was ended", success=False)
                        self.rec.close()
                    except Exception as e:  # noqa: BLE001
                        self.say(f"while closing the recording: {type(e).__name__}: {e}")
                    self.rec = None
                    self._rest_cameras()
                if self.other is not None:
                    try:
                        self.other.disconnect()
                    except Exception:  # noqa: BLE001
                        pass
                    self.other, self.other_side, self.watch = None, None, None
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
            self.guard = Guard(self.arm, self.gate, self._jog_limits(), log_path=self.log_dir / "guard.jsonl", say=self.say, watch=self.watch, **self.guard_kw)
            if self.rec is None and self.recording:
                try:
                    from .record import Recorder

                    self.rec = Recorder(self.log_root, self._sample, frames=self._rec_frames, hz=self.record_hz, say=self.say)
                except Exception as e:  # noqa: BLE001  no recording is no reason not to work
                    self.say(f"not recording: {type(e).__name__}: {e}")
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

        job = self.job

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
            except Exception as e:  # noqa: BLE001  a mistake in the program itself: stop the arm and say so, never report "done"
                note = f"stopped: the program failed ({type(e).__name__}: {e})"
                try:
                    self.arm.stop()
                except Exception:  # noqa: BLE001
                    pass
                raise
            finally:
                latched = g.latched
                with self.lock:
                    job["outcome"], job["i"] = note or "done", None
                    if self.worker is threading.current_thread():  # not a worker that outlived its session: that one touches nothing
                        self.worker, self.note = None, note
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

    def down(self):
        """Straight down in the chosen arm's base frame (the arms of this rig are mounted at an angle)."""
        return np.asarray(self.arms[self.chosen].get("down", (0.0, 0.0, -1.0)), float)

    def find_table(self, wide_mm=200.0, deep_mm=140.0, here=False):
        """Instead of teaching spots by hand. With a bar held in the jaws (lying as it would lie on the table, the
        jaws across its short side) and the gripper pointing straight down over the middle of the area the blocks will
        lie in: come down slowly until the bar meets the table, let go of it there, and make the spots from that:
          T   where the bar was put down: the height at which blocks are taken, and where the tower will stand;
          P, Q  two opposite corners of the area, `wide_mm` across (left to right) and `deep_mm` deep, around T.
        The label then reads ARMED with the bar on T, ready for the camera to be shown the table.
        `here`: the gripper was brought down by hand and its fingertips touch the table now. Nothing comes down: it
        is raised by a millimetre and a half, lets go of the bar, and the spots are made at that height."""
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
            axes, by_hand = jog_axes(pose[3:]), bool(here)

            def act(g):
                got = g.grip(self.P.shut_counts, "close on the bar")
                if got < self.P.held_min_counts:
                    g.grip(self.P.open_counts, "open again: nothing was held")
                    raise Refused("nothing is between the jaws: hold a bar between them and press Close gripper first")
                start = g.where()
                g.lim = Limits(start.copy(), axes, np.array([-15.0, -15.0, -450.0]), np.array([15.0, 15.0, 30.0]), **self.limit_kw)
                if by_hand:
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
                    d = Spots(side=self.chosen)
                    d.add("T", here)
                    for name, sign in (("P", -1.0), ("Q", 1.0)):
                        d.add(name, np.r_[here[:3] + sign * (deep / 2.0 * axes[:, 0] + wide / 2.0 * axes[:, 1]), here[3:]])
                    self._keep_spots(d.check())
                    self._adopt(d, thing_at="T")
                came = float((start - here[:3]) @ axes[:, 2])
                self.say(f"ARMED: the table is {came:.0f} mm below where the gripper was. The bar lies on T; the area is {wide:.0f} mm across and {deep:.0f} mm deep around it")
                return "the table is where the gripper was; the bar lies on spot T" if by_hand else f"the table was met {came:.0f} mm down; the bar lies on spot T"

            self._small("find the table", act)

    def grip(self, close):
        with self.lock:
            self._teach("the gripper")

            def act(g):
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
            self._keep_spots(self.draft)
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

    def _file(self, path, side=None):
        """The file of this kind that belongs to one arm: spots.json -> spots.left.json. What one arm was taught is
        never written over by a session on another arm (or on the stand-in)."""
        path, side = Path(path), side or self.chosen
        return path if side is None else path.with_name(f"{path.stem}.{side}{path.suffix}")

    def _keep_spots(self, spots):
        """Save taught spots under their arm's own name, and under the plain name unless that holds another arm's."""
        spots.save(self._file(self.spots_path, spots.side))
        try:
            other = Spots.load(self.spots_path).side if self.spots_path.exists() else spots.side
        except Exception:  # noqa: BLE001  a file nobody can read is in nobody's way
            other = spots.side
        if other == spots.side:
            spots.save(self.spots_path)

    def _saved_spots(self, side=None):
        """The spots taught earlier with this arm, or None."""
        side = side or self.chosen
        for path in (self._file(self.spots_path, side), self.spots_path):
            if path.exists():
                try:
                    s = Spots.load(path).check()
                except (Refused, ValueError, KeyError, TypeError, AttributeError, OSError):
                    continue
                if side is None or s.side == side:
                    return s, path
        return None

    def saved(self):
        got = self._saved_spots()
        if got is None:
            return None
        s, path = got
        return dict(names=sorted(s.at), when=time.strftime("%d %b %H:%M", time.localtime(path.stat().st_mtime)), side=s.side)

    def use_saved(self):
        """Take the spots of an earlier session with this arm, if the arm is where they are and turned as it was then."""
        with self.lock:
            self._teach("using saved spots")
            got = self._saved_spots()
            if got is None:
                raise Refused(f"no spots were saved with the {self.arms[self.chosen]['label'].lower()}")
            s = got[0]
            pose = self.arm.pose()
            tilt = s.tilt_from(pose[3:])
            if tilt > 2.0:
                raise Refused(f"the gripper is tilted {tilt:.1f} deg away from how the saved spots were taught: teach them again")
            why = s.limits().why_outside(pose[:3])
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
            self.job = dict(steps=[], i=None, waiting=False, outcome=None, done=0, total=count, src=self.thing_at, dst=dst, rehearse=bool(rehearse), step=bool(step), fixed=dst, kind="move",
                            task=f"move the block from spot {self.thing_at} to spot {dst}" if dst is not None and count == 1 else "move the block from spot to spot")
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
                            fixed=None, kind="move", legs=legs, spots=sp, text=legs[0]["text"], tower=True,
                            task=dict(build="stack the blocks into a tower", down="take the tower apart and put every block back on its spot", loop="build a tower and take it apart again")[what])
            self._next.clear()
            self.worker = threading.Thread(target=self._work, daemon=True)
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

        before = self._before(job, g)
        self._ep_begin(job.get("task") or job.get("text") or "move the block to another place", kind="spots", rehearse=job["rehearse"])

        outcome = "done"
        try:
            for k in range(job["total"]):
                leg = legs[k] if legs else dict(src=at, su=0.0, du=0.0, dst=job["fixed"] if job["fixed"] is not None and k == 0 else str(rng.choice([n for n in sorted(sp.at) if n != at])))
                at, dst = leg["src"], leg["dst"]
                text = leg.get("text") or f"{at} to {dst}"
                here = g.where()
                steps = plan_move(sp, at, dst, here, self.P, rehearse=job["rehearse"], src_up=leg["su"], dst_up=leg["du"])
                top = float(here @ sp.up) + 1.0  # the arm may start above the travel height (after a tower): it comes down first
                g.lim = sp.limits(floor_w=min(float(here @ sp.up), sp.travel_w()) - 1.0, ceil_w=top, **self.limit_kw) if job["rehearse"] else sp.limits(ceil_w=top, **self.limit_kw)
                with self.lock:
                    job.update(steps=steps, i=None, src=at, dst=dst, text=text)
                self.say(f"move {k + 1} of {job['total']}: {text}" + (" (rehearsal in the air)" if job["rehearse"] else ""))
                self._mark("leg", text=text, src=at, dst=dst)
                if leg.get("first"):
                    leg["first"]()
                rec = execute(g, steps, self.P, air=job["rehearse"], before=before)
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
                        self.stack = {**self.stack, "n": self.stack["n"] + (1 if dst == self.stack["on"] else -1)}
                    else:
                        self.thing_at = dst
                if leg.get("then"):
                    leg["then"]()
            if outcome == "done" and job.get("finish"):
                job["finish"]()
        except NothingHeld as e:
            outcome = f"{e}. Say where " + ("the blocks are" if legs else "the thing is")
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
            self._wrap_up(job, g, outcome)

    def _before(self, job, g):
        """What runs before every step of a job: the page is told which step it is, the recorder too, and in step
        mode the step waits for its button."""
        def before(s):
            job["i"] = next((i for i, x in enumerate(job["steps"]) if x is s), job["i"])
            self._mark("step", name=s["name"], what=s["kind"])
            if job["step"]:
                job["waiting"] = True
                while not self._next.wait(0.05):
                    g._check_abort()
                self._next.clear()
                job["waiting"] = False

        return before

    def _wrap_up(self, job, g, outcome):
        """The end of every job, however it ended."""
        latched = g.latched
        self._ep_end(outcome, success=outcome == "done" and not job.get("rehearse"))  # a job that keeps its own episodes has ended them already
        with self.lock:
            job["outcome"], job["i"], job["waiting"] = outcome, None, False
            if self.worker is threading.current_thread():  # not a worker that outlived its session: that one touches nothing
                self.worker = None
                if outcome.startswith("stopped") and self.gate is not None:
                    self.gate.armed = self.teaching = False
        self.say(f"ended: {outcome}")
        if latched:
            threading.Thread(target=self.end, args=(latched,), daemon=True).start()

    def _begin_job(self, text, body, step=False, tower=False):
        """Start a job whose legs are not known beforehand: `body(job, guard)` runs in the worker's thread, plans as
        it goes and returns how it ended."""
        self.job = dict(steps=[], i=None, waiting=False, outcome=None, done=0, total=0, src=None, dst=None, rehearse=False, step=bool(step), fixed=None, kind="move",
                        text=text, tower=bool(tower))
        self._next.clear()
        self.say(text)
        self.worker = threading.Thread(target=self._task, args=(self.job, body), daemon=True)
        self.worker.start()

    def _task(self, job, body):
        g, outcome = self.guard, "done"
        try:
            outcome = body(job, g) or "done"
        except NothingHeld as e:
            outcome = f"{e}. Look at the table"
        except Released as e:
            outcome = f"stopped: {e}"
        except (ArmFault, Refused) as e:
            outcome = f"{type(e).__name__}: {e}"
        except Exception as e:  # noqa: BLE001  a mistake in the program itself: stop the arm and say so, never report "done"
            outcome = f"stopped: the program failed ({type(e).__name__}: {e})"
            try:
                self.arm.stop()
            except Exception:  # noqa: BLE001
                pass
            raise
        finally:
            if self.stack is not None and not outcome.startswith("done"):
                self.stack = {**self.stack, "of": None}
            self._wrap_up(job, g, outcome)

    # ------------------------------------------------------------------ the recorder (a no-op without one)
    def _arm_row(self, arm, guard=None):
        """One arm as the recorder writes it down: only what its controller publishes anyway."""
        doing = guard.doing if guard is not None else None
        grip = guard.counts if guard is not None and guard.counts is not None else getattr(arm, "counts", None)
        return dict(joints=[round(float(v), 3) for v in arm.joints_view()], xyz=[round(float(v), 2) for v in arm.xyz_now()], rpy=[round(float(v), 3) for v in arm.rpy_now()],
                    grip=None if grip is None else int(grip), moving=bool(arm.moving_now()), goal_xyz=None if doing is None else doing["goal"],
                    goal_grip=None if guard is None or guard.asked is None else int(guard.asked), what="" if doing is None else doing["what"])

    def _side(self, key):
        """Under which name an arm is written down: the stand-in arm stands in for the right one (the episode says so)."""
        return key if key in ("right", "left") else "right"

    def _sample(self):
        arm, job = self.arm, self.job or {}
        if arm is None:
            return {}
        arms = {self._side(self.chosen): self._arm_row(arm, self.guard)}
        if self.other is not None:
            try:
                arms[self._side(self.other_side)] = self._arm_row(self.other)
            except Exception:  # noqa: BLE001  the other arm is only watched
                pass
        i, steps = job.get("i"), job.get("steps") or []
        return dict(arms=arms, held=bool(self.gate.held()) if self.gate is not None else False, phase=self.phase(), step=steps[i]["name"] if i is not None and i < len(steps) else "",
                    leg=job.get("text") or "", waiting=bool(job.get("waiting")))

    def _rec_frames(self):
        if self.feeds is None or self.feeds.why_not():
            return []
        return self.feeds.latest([k for k in (self.chosen, self.other_side) if k])

    def set_recording(self, on):
        """Whether the jobs from now on are recorded (pictures, joints and what was asked, ten times a second)."""
        with self.lock:
            self.recording = bool(on)
            self.say("jobs are recorded from now on" if self.recording else "jobs are NOT recorded from now on")

    def judge(self, good):
        """A person's word on the episode that was recorded last: did it do what its task says?"""
        from .record import episodes

        eps = episodes(self.log_root)
        if not eps or (self.rec is not None and self.rec.active):
            raise Refused("there is no finished episode to judge")
        path = eps[-1] / "episode.json"
        meta = json.loads(path.read_text())
        meta["success"], meta["judged_by"] = bool(good), "a person"
        path.write_text(json.dumps(meta, indent=1))
        self.say(f"episode {eps[-1].name} is marked as " + ("a success" if good else "a failure"))

    # ------------------------------------------------------------------ the other arm (see `pair`)
    def _other_key(self, key=None):
        key = key or self.chosen
        other = {"right": "left", "left": "right"}.get(key)
        return other if other in self.arms else None

    def rig(self):
        """How the two arms stand to each other (`rig.json`: the left arm's base in the right arm's), or None."""
        if not self.rig_path.exists():
            return None
        from .pair import Frame

        try:
            return Frame.from_dict(json.loads(self.rig_path.read_text())["left_in_right"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError, Refused):
            return None

    def _other_in_mine(self, key=None):
        frame, key = self.rig(), key or self.chosen
        if frame is None:
            return None
        return frame if key == "right" else frame.inverse()

    def _watch_other(self):
        """At the start of a session: if the two arms' places are known and the other arm answers, connect to it READ
        ONLY (it is never enabled or moved from here) and have the guard keep this arm away from it."""
        self.other, self.other_side, self.watch = None, None, None
        key, frame = self._other_key(), self._other_in_mine()
        if key is None:
            return
        if frame is None:
            self.say(f"the {self.arms[key]['label'].lower()} is NOT watched (the arms have not been fitted to each other): keep it out of the way")
            return
        spec = self.arms[key]
        if spec.get("ip") is not None and not self.probe(spec["ip"]):
            self.say(f"the {spec['label'].lower()} does not answer and is NOT watched: keep it out of the way")
            return
        try:
            other = spec["make"]()
            other.connect()
        except Exception as e:  # noqa: BLE001
            self.say(f"the {spec['label'].lower()} is NOT watched ({type(e).__name__}: {e}): keep it out of the way")
            return
        from .pair import Watch

        def joints(arm):
            def read():
                if not getattr(arm, "connected", True):
                    raise ArmFault("the arm is not connected")
                return arm.joints_view()

            return read

        self.other, self.other_side = other, key
        self.watch = Watch(joints(self.arm), joints(other), frame, other_step_mm=0.0)  # nothing here moves the other arm; that it stays put is checked at every poll
        self.say(f"the {spec['label'].lower()} is watched (read only): this arm keeps {self.watch.stop_mm:.0f} mm away from it")

    def pair_fit(self):
        """Fit the two arms to each other from spots that were taught with BOTH arms at the same marks on the table,
        under the same names (the fingertips touching the mark, the gripper pointing straight down). Kept in
        `rig.json`. Moves nothing."""
        from . import pair

        with self.lock:
            got = {side: self._saved_spots(side) for side in ("right", "left")}
            for side, g in got.items():
                if g is None:
                    raise Refused(f"no spots were saved with the {side} arm: teach the marks with it first")
            R, L = got["right"][0], got["left"][0]
            names = sorted(set(R.at) & set(L.at))
            if len(names) < 4:
                raise Refused(f"the two arms share only {len(names)} spot names: teach at least 4 marks (better 5 or more) with both arms, under the same names")
            tip = lambda s, n: pair.fingertip(np.r_[s.at[n], s.turn(n)])  # noqa: E731
            frame = pair.fit_frames([tip(L, n) for n in names], [tip(R, n) for n in names], names=names)
            off = pair.check_against_mounts(frame, self.arms["left"].get("down", (0.0, 0.0, -1.0)), self.arms["right"].get("down", (0.0, 0.0, -1.0)))
            frame.when = time.time()
            self.rig_path.write_text(json.dumps(dict(left_in_right=frame.to_dict()), indent=1))
            self.say(f"the arms are fitted to each other from {len(names)} marks, within {frame.rms_mm:.1f} mm ({off:.1f} deg from what the mounts say). Kept in {self.rig_path}")
            return frame

    def use_other(self):
        """Take the OTHER arm's spots, and the camera as it was shown the table with the other arm, carried over into
        this arm's own frame (no teaching and no showing again). The arm must be over those spots, turned as they
        need. Rehearse in the air first: nothing checks that this arm can reach every place."""
        from . import pair

        with self.lock:
            self._teach("using the other arm's spots")
            key, frame = self._other_key(), self._other_in_mine()
            if key is None or frame is None:
                raise Refused("the arms have not been fitted to each other yet")
            got = self._saved_spots(key)
            if got is None:
                raise Refused(f"no spots were saved with the {key} arm")
            s = pair.spots_in(frame, got[0], self.chosen).check()
            pose = self.arm.pose()
            tilt = s.tilt_from(pose[3:])
            if tilt > 2.0:
                raise Refused(f"the gripper is tilted {tilt:.1f} deg away from how those spots need it: point it straight down first")
            if s.limits().why_outside(pose[:3]):
                raise Refused("the arm is not over those spots: nudge it there")
            self._adopt(s)
            self._keep_spots(s)
            line = ""
            cam = self._file(self.camera_path, key)
            if cam.exists():
                try:
                    eyes = self._eyes()
                    self.setup = pair.setup_in(frame, eyes.Setup.load(cam), self.chosen)
                    self._keep_setup(self.setup)
                    line = f", and the camera as the {key} arm showed it the table"
                except Exception as e:  # noqa: BLE001
                    line = f" (the camera was not carried over: {type(e).__name__}: {e})"
            self.say(f"ARMED with the {key} arm's spots {', '.join(sorted(s.at))}{line}. REHEARSE IN THE AIR FIRST: they were never reached with this arm")

    def pair_state(self):
        """For the page: are the arms fitted to each other, is the other one watched, and how near is it?"""
        out = dict(fitted=self.rig_path.exists(), watched=self.watch is not None, other=self.other_side, mm=None, part=None, why="")
        if self.watch is not None:
            try:
                mm, mine, theirs = self.watch.mm()
                out.update(mm=round(float(mm)), part=f"{mine} / {theirs}")
            except Exception as e:  # noqa: BLE001
                out["why"] = str(e)
        return out

    def _ep_begin(self, task, **meta):
        rec = self.rec
        if rec is not None and self.recording and not rec.active:
            try:
                rec.begin(task, arm=self.chosen, fake=self.fake, **meta)
            except Exception as e:  # noqa: BLE001  recording must never stop a job
                self.say(f"not recording: {type(e).__name__}: {e}")

    def _ep_end(self, outcome, success=None, **kw):
        rec = self.rec
        if rec is not None and rec.active:
            try:
                rec.end(outcome, success=success, **kw)
            except Exception as e:  # noqa: BLE001
                self.say(f"the recording was not closed properly: {type(e).__name__}: {e}")
            self._rest_cameras()

    def _rest_cameras(self):
        """The recorder no longer needs the cameras: they are let go unless the page is showing them."""
        kept = getattr(self.feeds, "kept", None)
        if kept is not None:
            kept.clear()

    def _mark(self, kind, **kw):
        rec = self.rec
        if rec is not None and rec.active:
            try:
                rec.mark(kind, **kw)
            except Exception:  # noqa: BLE001
                pass

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
        out = dict(tool_mm=TOOL_MM, up=None, box=None, path=None, at=None, things=[], seen=[])
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
                       pair=self.pair_state(), rec=dict(on=self.recording, **(self.rec.stats() if self.rec is not None else {})), check_width=self.check_width,
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
