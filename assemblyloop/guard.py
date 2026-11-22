"""The only door to the arm. The cycle never talks to an arm directly; it asks the guard, and the guard

  1. refuses, BEFORE anything is sent, a move that ends outside the allowed volume, is faster than the cap, or needs a
     single segment longer than `max_step_mm`;
  2. sends nothing unless the hold-to-run switch is held at that moment;
  3. cuts every move into short segments and never has more than one in the controller, so that if this program or
     the laptop dies, the arm finishes at most one short, already checked segment and stops there;
  4. pauses the arm within one poll (20 ms) when the switch is released, resumes when it is pressed again, and stops
     and ends the run if it stays released;
  5. stops the arm and LATCHES on any controller error, refused call or move that does not arrive: after that every
     further request is refused until a person restarts the program;
  6. INCHES through a WATCHED move (lowering onto a brick, pressing, pulling): `inch_mm` at a time, and the next inch
     is sent only if the arm's MEASURED joints show that it is not more than `stall_mm` behind where the controller
     plans it to be. Collision detection is off on this rig, so without this a press onto something that does not
     give would keep pushing; with it the plan is never more than `inch_mm + stall_mm` ahead of the arm, whatever the
     timing. This raises `Blocked`, which is not a fault: the cycle lifts clear and records it. After every segment
     of any OTHER move the same comparison is made with a wider limit (`free_lag_mm`): there, being held back means
     the arm has run into something, which stops the arm and ends the run;
  7. TURNS the gripper only by spinning its last joint, in pieces, each checked the same way (`turn_by`, `turn_to`);
  8. writes every request and every refusal to a log.

While a move is under way only the state the controller publishes by itself is read (no call that can block), so the
switch and the stall watch are checked every `poll_s`.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from .arms import ArmFault, matrix_rpy, rpy_matrix, twist_deg


class Refused(RuntimeError):
    """The guard would not send this. Nothing moved."""


class Released(RuntimeError):
    """The hold-to-run switch stayed released. The arm was stopped and the run ends."""


class Blocked(RuntimeError):
    """A watched move met something that did not give. The arm was stopped; call `recover()` and lift clear."""


@dataclass
class Limits:
    """The allowed volume as bounds in a frame: p = origin + axes @ (u, v, w). For a plate: u along the columns, v
    along the rows, w up along the plate's normal, measured from the hover height."""

    origin: np.ndarray
    axes: np.ndarray  # columns: unit vectors of u, v, w in the arm's base frame
    lo: np.ndarray
    hi: np.ndarray
    max_step_mm: float = 30.0
    max_speed_mm_s: float = 120.0
    acc_mm_s2: float = 500.0
    arrive_mm: float = 1.0
    inch_mm: float = 3.0  # segment length of a watched move
    settle_s: float = 0.06  # wait after a segment before the measured joints are read
    free_lag_mm: float = 5.0  # a move that should meet nothing: further than this from its plan is a collision
    turn_speed_deg_s: float = 20.0  # the lab's own pace for joint moves
    turn_acc_deg_s2: float = 150.0
    turn_step_deg: float = 10.0  # a turn is sent in pieces of at most this, and checked after each
    turn_stall_deg: float = 1.5  # a joint this far behind its plan after a piece: something holds the gripper
    tilt_step_deg: float = 5.0  # pointing the gripper down: the tilt is sent in pieces of at most this
    tilt_speed: float = 15.0  # ... at this speed (read by the controller as degrees per second when nothing translates; not yet confirmed on the arm)
    tilt_joint_deg: float = 12.0  # ... and no joint may move further than this during one piece: more means the arm is re-posing itself
    tilt_drift_mm: float = 4.0  # ... nor may the flange leave its place by more than this
    j7_limit_deg: float = 340.0  # the last joint may go to 360; the lab stops at 350
    extra: dict = field(default_factory=dict)

    def coords(self, p):
        return self.axes.T @ (np.asarray(p, float) - self.origin)

    def why_outside(self, p):
        c = self.coords(p)
        for name, x, a, b in zip("uvw", c, self.lo, self.hi):
            if not (a - 1e-6 <= x <= b + 1e-6):
                return f"{name} = {x:.1f} mm is outside [{a:.1f}, {b:.1f}]"
        return ""


class Guard:
    def __init__(self, arm, deadman, limits, log_path=None, clock=time.monotonic, sleep=time.sleep, poll_s=0.02, start_s=120.0, resume_s=15.0, say=print):
        self.arm, self.deadman, self.lim = arm, deadman, limits
        self.clock, self.sleep, self.poll_s, self.start_s, self.resume_s, self.say = clock, sleep, float(poll_s), float(start_s), float(resume_s), say
        self.latched, self.events, self._abort = None, [], None
        self.counts = None  # where the gripper came to rest the last time it was sent somewhere (for the page's model)
        self.turn_sign, self._sign_known = 1.0, False  # which way the gripper spins when joint 7 goes up: found out on the first turn
        self._log = open(log_path, "a") if log_path else None

    # ------------------------------------------------------------------ bookkeeping
    def note(self, kind, **kw):
        row = dict(t=round(self.clock(), 4), kind=kind, **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in kw.items()})
        self.events.append(row)
        if self._log:
            self._log.write(json.dumps(row) + "\n")
            self._log.flush()
        return row

    def _alive(self):
        if self.latched:
            raise ArmFault(f"the run is over ({self.latched}); restart the program after looking at the arm")

    def abort(self, why, latch=True):
        """Asked from ANOTHER thread (a stop button): the thread that is moving the arm stops it at its next poll,
        at most `poll_s` later, and its current request ends with `Released`. With `latch` the whole run is over."""
        self._abort = (str(why), bool(latch))

    def _check_abort(self):
        if self._abort is not None:
            why, latch = self._abort
            self._abort = None
            self.arm.stop()
            if latch:
                self.latched = why
            self.note("abort", why=why, latch=latch)
            raise Released(why)

    def _fail(self, why):
        try:
            self.arm.stop()
        finally:
            self.latched = why
            self.note("fault", why=why)
        raise ArmFault(why)

    def _check_fault(self, now=False):
        try:
            code = self.arm.fault_now() if now else self.arm.fault()
        except ArmFault as e:
            self._fail(str(e))
        if code:
            self._fail(f"controller error {code}")

    def _wait_hold(self, what, limit_s, moving=False):
        """Block until the switch is held. If it is not within `limit_s`: stop the arm, end the run."""
        if self.deadman.held():
            return
        t0, told = self.clock(), False
        while not self.deadman.held():
            self._check_abort()
            if not told:
                self.say(f"  waiting for the hold-to-run switch ({self.deadman.why()}) before: {what}")
                told = True
            if self.clock() - t0 > limit_s:
                self.arm.stop()
                self.latched = "the hold-to-run switch stayed released"
                self.note("released", what=what, moving=moving)
                raise Released(self.latched)
            self.sleep(self.poll_s)

    # ------------------------------------------------------------------ what the cycle may ask for
    def start(self):
        self._alive()
        problem = self.arm.ready()
        if problem:
            self._fail(problem)
        self.arm.enable()
        off = float(np.linalg.norm(self._lag()))
        self.note("enable", lag_mm=off)
        if off > 1.5:  # the stall watch relies on this comparison: it must read about zero on a free arm at rest
            self._fail(f"measured and planned joints disagree by {off:.1f} mm at rest: the stall watch cannot be trusted")

    def where(self):
        return self.arm.pose()[:3]

    def recover(self):
        """After `Blocked`: enable the stopped arm again so that it can be lifted clear. Not after a fault."""
        self._alive()
        try:
            self.arm.enable()
        except ArmFault as e:
            self._fail(str(e))
        self.note("recover")

    def move_to(self, target, speed, what="", stall_mm=None, inch_mm=None):
        """Straight to `target` (base frame, mm) at the taught orientation, in segments of at most `max_step_mm`.
        With `stall_mm` the move is watched: it stops (Blocked) once the measured arm is that far behind the plan.
        A watched move goes in pieces of `inch_mm` (the limits' own, 3 mm, unless given) and is checked after each."""
        self._alive()
        target = np.asarray(target, float)
        if speed > self.lim.max_speed_mm_s + 1e-9 or speed <= 0:
            self.note("refused", what=what, why="speed", speed=speed)
            raise Refused(f"{what}: {speed:.0f} mm/s is not within (0, {self.lim.max_speed_mm_s:.0f}]")
        start = self.where()
        for p in (start, target):
            why = self.lim.why_outside(p)
            if why:
                self.note("refused", what=what, why=why, target=target)
                raise Refused(f"{what}: {'the arm is' if p is start else 'the target is'} outside the allowed volume ({why})")
        length = float(np.linalg.norm(target - start))
        if stall_mm is None:
            n = max(int(np.ceil(length / self.lim.max_step_mm - 1e-9)), 1)
            for i in range(n):
                self._segment(start + (target - start) * (i + 1) / n, speed, what)
                self.sleep(self.lim.settle_s)
                off = float(np.linalg.norm(self._lag()))
                if off > self.lim.free_lag_mm:
                    self._fail(f"{what}: the arm is {off:.1f} mm away from its plan on a move that should meet nothing")
            return
        if length < 1e-3:
            return
        inch = float(np.clip(self.lim.inch_mm if inch_mm is None else inch_mm, 1.0, self.lim.max_step_mm))
        n, u, worst = max(int(np.ceil(length / inch - 1e-9)), 1), (target - start) / length, 0.0
        for i in range(n):
            self._segment(start + (target - start) * (i + 1) / n, speed, what)
            self.sleep(self.lim.settle_s)
            behind = -float(self._lag() @ u)  # how far the measured arm is behind the plan, along the move
            worst = max(worst, behind)
            if behind > stall_mm:
                self.arm.stop()
                self.note("blocked", what=what, behind_mm=behind, done_mm=length * (i + 1) / n)
                raise Blocked(f"{what}: the arm is {behind:.1f} mm behind its plan, {length * (i + 1) / n:.0f} mm into the move")
        self.note("watched", what=what, worst_behind_mm=worst)

    def _lag(self):
        try:
            return np.asarray(self.arm.lag_mm(), float)
        except ArmFault as e:
            self._fail(str(e))

    def _segment(self, goal, speed, what):
        self._alive()
        d = goal - self.where()
        length = float(np.linalg.norm(d))
        if length < 1e-3:
            return
        if length > self.lim.max_step_mm * 1.5:  # the arm is not where the plan says it is
            self._fail(f"{what}: the next segment would be {length:.1f} mm, the arm is not where it should be")
        self._check_abort()
        self._wait_hold(what, self.start_s)
        self._check_fault()
        self.note("move", what=what, d=d, speed=speed, goal=goal)
        try:
            self.arm.move_rel(d, speed, self.lim.acc_mm_s2)
        except ArmFault as e:
            self._fail(str(e))
        t0, budget, held_back = self.clock(), length / speed * 3.0 + 3.0, 0.0
        while True:
            self._check_abort()
            self._check_fault(now=True)
            if not self.deadman.held():
                t_p = self.clock()
                self.arm.pause()
                self.note("pause", what=what)
                self._wait_hold(what, self.resume_s, moving=True)
                self.arm.resume()
                self.note("resume", what=what)
                held_back += self.clock() - t_p
            if float(np.linalg.norm(self.arm.xyz_now() - goal)) <= self.lim.arrive_mm and not self.arm.moving_now():
                return
            if self.clock() - t0 - held_back > budget:
                self._fail(f"{what}: the move did not arrive ({np.linalg.norm(self.arm.xyz_now() - goal):.1f} mm short)")
            self.sleep(self.poll_s)

    def turn_by(self, deg, what=""):
        """Spin the gripper about its own axis by turning the last joint `deg` degrees (nothing else moves), in pieces
        of at most `turn_step_deg`. After each piece the measured joints are compared with the planned ones: if they
        are apart, something is holding the gripper and the turn ends there (Blocked)."""
        self._alive()
        deg = float(deg)
        if abs(deg) < 0.05:
            return
        if abs(deg) > 185.0:
            self.note("refused", what=what, why="turn", deg=deg)
            raise Refused(f"{what}: a turn is at most half a turn at a time, not {deg:.0f} deg")
        j0 = float(self.arm.joints(False)[6])
        if abs(j0 + deg) > self.lim.j7_limit_deg:
            self.note("refused", what=what, why="joint 7 limit", deg=deg, j7=j0)
            raise Refused(f"{what}: the last joint is at {j0:.0f} deg and would pass its limit of {self.lim.j7_limit_deg:.0f}: turn the other way")
        n = max(int(np.ceil(abs(deg) / self.lim.turn_step_deg - 1e-9)), 1)
        for i in range(n):
            self._turn_piece(j0 + deg * (i + 1) / n, what)
            self.sleep(self.lim.settle_s)
            try:
                behind = float(self.arm.joint_lag_deg())
            except ArmFault as e:
                self._fail(str(e))
            if behind > self.lim.turn_stall_deg:
                self.arm.stop()
                self.note("blocked", what=what, behind_deg=behind)
                raise Blocked(f"{what}: a joint is {behind:.1f} deg behind its plan: something is holding the gripper")

    def _turn_piece(self, goal, what):
        self._alive()
        d = goal - float(self.arm.joints(False)[6])
        if abs(d) < 1e-3:
            return
        self._check_abort()
        self._wait_hold(what, self.start_s)
        self._check_fault()
        self.note("turn", what=what, deg=d, goal=goal)
        try:
            self.arm.turn_rel(d, self.lim.turn_speed_deg_s, self.lim.turn_acc_deg_s2)
        except ArmFault as e:
            self._fail(str(e))
        t0, budget, held_back = self.clock(), abs(d) / self.lim.turn_speed_deg_s * 3.0 + 3.0, 0.0
        while True:
            self._check_abort()
            self._check_fault(now=True)
            if not self.deadman.held():
                t_p = self.clock()
                self.arm.pause()
                self.note("pause", what=what)
                self._wait_hold(what, self.resume_s, moving=True)
                self.arm.resume()
                self.note("resume", what=what)
                held_back += self.clock() - t_p
            if abs(self.arm.j7_now() - goal) <= 0.5 and not self.arm.moving_now():
                return
            if self.clock() - t0 - held_back > budget:
                self._fail(f"{what}: the turn did not arrive ({abs(self.arm.j7_now() - goal):.1f} deg short)")
            self.sleep(self.poll_s)

    def turn_to(self, rpy_deg, what=""):
        """Spin the gripper until it is turned as in `rpy_deg`. Only a spin about its own axis is ever made: if the
        wanted orientation also tilts the gripper, this refuses. Which way the gripper spins when the last joint goes
        up is not assumed: the first turn starts with a few degrees, looks at what happened, and goes on from there."""
        self._alive()
        want = rpy_matrix(rpy_deg)
        for _ in range(6):
            left, tilt = twist_deg(rpy_matrix(self.arm.pose()[3:]), want)
            if tilt > 3.0:
                self.note("refused", what=what, why="tilt", tilt_deg=tilt)
                raise Refused(f"{what}: the gripper's axis is {tilt:.1f} deg away from where it was taught; only spins about its own axis are made")
            if abs(left) <= 1.0:
                return
            go = left if self._sign_known else float(np.clip(left, -5.0, 5.0))
            j = float(self.arm.joints(False)[6])
            cmd = self.turn_sign * go
            if abs(j + cmd) > self.lim.j7_limit_deg:  # the same orientation, the other way round
                cmd -= float(np.sign(cmd)) * 360.0
                if abs(cmd) > 185.0 or abs(j + cmd) > self.lim.j7_limit_deg:
                    raise Refused(f"{what}: the last joint is at {j:.0f} deg and cannot reach that turn from here")
            self.turn_by(cmd, what)
            after, _ = twist_deg(rpy_matrix(self.arm.pose()[3:]), want)
            if not self._sign_known:
                if abs(after) > abs(left) + 1.0:
                    self.turn_sign = -self.turn_sign
                    self.note("turn_sign", sign=self.turn_sign)
                self._sign_known = True
        left, _ = twist_deg(rpy_matrix(self.arm.pose()[3:]), want)
        if abs(left) > 2.0:
            self._fail(f"{what}: the gripper ended {left:.1f} deg from the wanted turn")

    def level(self, what="point the gripper down", down=(0.0, 0.0, -1.0)):
        """Tilt the gripper until it points straight down, the flange staying where it is, in pieces of a few
        degrees. This is the only move in which the controller is free to move every joint, so every joint is watched
        while a piece runs: one that moves much further than the piece asks for means the arm is re-posing itself, and
        the arm is stopped and the run ends. Returns how far it was tilted [deg]."""
        self._alive()
        pose = self.arm.pose()
        xyz, R0, down = pose[:3].copy(), rpy_matrix(pose[3:]), np.asarray(down, float)
        angle = float(np.degrees(np.arccos(np.clip(R0[:, 2] @ down, -1.0, 1.0))))
        if angle < 0.3:
            return 0.0
        if angle > 95.0:
            self.note("refused", what=what, why="tilt", tilt_deg=angle)
            raise Refused(f"{what}: the gripper points {angle:.0f} deg away from straight down, which is too far to do in small pieces from here")
        k = np.cross(R0[:, 2], down)
        k = k / np.linalg.norm(k)
        K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        n = max(int(np.ceil(angle / self.lim.tilt_step_deg - 1e-9)), 1)
        for i in range(n):
            a = np.radians(angle * (i + 1) / n)
            goal = matrix_rpy((np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * K @ K) @ R0)
            self._tilt_piece(xyz, goal, what)
            self.sleep(self.lim.settle_s)
            try:
                behind, drift = float(self.arm.joint_lag_deg()), float(np.linalg.norm(self.arm.pose()[:3] - xyz))
            except ArmFault as e:
                self._fail(str(e))
            if behind > self.lim.turn_stall_deg:
                self.arm.stop()
                self.note("blocked", what=what, behind_deg=behind)
                raise Blocked(f"{what}: a joint is {behind:.1f} deg behind its plan: something is holding the arm")
            if drift > self.lim.tilt_drift_mm:
                self._fail(f"{what}: the flange moved {drift:.1f} mm although it should stay in place")
        return angle

    def _tilt_piece(self, xyz, goal, what):
        self._alive()
        self._check_abort()
        self._wait_hold(what, self.start_s)
        self._check_fault()
        want, j0 = rpy_matrix(goal), np.array(self.arm.joints(False), float)
        piece = float(np.degrees(np.arccos(np.clip((np.trace(rpy_matrix(self.arm.pose()[3:]).T @ want) - 1.0) / 2.0, -1.0, 1.0))))
        self.note("tilt", what=what, deg=piece, goal=np.asarray(goal, float))
        try:
            self.arm.pose_abs(xyz, goal, self.lim.tilt_speed, 100.0)
        except ArmFault as e:
            self._fail(str(e))
        t0, budget, held_back = self.clock(), piece / self.lim.tilt_speed * 3.0 + 4.0, 0.0
        while True:
            self._check_abort()
            self._check_fault(now=True)
            swing = float(np.max(np.abs(np.array(self.arm.joints_now(), float) - j0)))
            if swing > self.lim.tilt_joint_deg:
                self._fail(f"{what}: a joint moved {swing:.0f} deg for a tilt of {piece:.1f} deg: the arm is re-posing itself. Point the gripper down by hand or in UFACTORY Studio")
            if not self.deadman.held():
                t_p = self.clock()
                self.arm.pause()
                self.note("pause", what=what)
                self._wait_hold(what, self.resume_s, moving=True)
                self.arm.resume()
                self.note("resume", what=what)
                held_back += self.clock() - t_p
            off = float(np.degrees(np.arccos(np.clip((np.trace(rpy_matrix(self.arm.rpy_now()).T @ want) - 1.0) / 2.0, -1.0, 1.0))))
            if off <= 0.5 and not self.arm.moving_now():
                return
            if self.clock() - t0 - held_back > budget:
                self._fail(f"{what}: the tilt did not arrive ({off:.1f} deg short)")
            self.sleep(self.poll_s)

    def grip(self, counts, what=""):
        self._alive()
        self._check_abort()
        self._wait_hold(what, self.start_s)
        self._check_fault()
        try:
            got = self.arm.grip(counts)
        except ArmFault as e:
            self._fail(str(e))
        self.note("grip", what=what, asked=int(counts), got=int(got))
        self.counts = int(got)
        return got

    def read(self, what=""):
        """What the arm measured, for the label. Reading needs no hold."""
        self._alive()
        row = dict(pose=self.arm.pose().tolist(), joints_planned=self.arm.joints(False), joints_measured=self.arm.joints(True))
        self.note("read", what=what, **row)
        return row

    def finish(self):
        """Leave the arm stopped, whatever happened."""
        try:
            self.arm.stop()
        finally:
            self.note("finish", latched=self.latched)
            if self._log:
                self._log.close()
                self._log = None
