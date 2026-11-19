"""The few calls the loop makes to an arm, twice: `XArm` for the real controller and `FakeArm`, a stand-in with the same
calls, for dry runs and for every test. Units as the controller's: millimetres, degrees, gripper counts (850 = open).

What the real arm is asked to do, and nothing else:
  - read its state (errors, pose, planned and measured joints, gripper)
  - planned straight-line moves RELATIVE to where it is, with zero rotation, at a given speed (position mode 0). This
    is how the lab's own "nudge" moves: a relative translation leaves the controller no licence to re-pose the arm.
  - turn the gripper about its own axis by moving the LAST JOINT ONLY, every other joint held where it is. That is how
    the lab turns it: asking the controller for a new orientation instead makes it re-solve all seven joints, and the
    whole arm swings.
  - pause, resume and stop such a move
  - open the gripper, or close it fully and let it stall on the brick (the lab's way of gripping)
It never streams targets (servo mode), never clears a controller error, never changes a controller setting (payload,
collision sensitivity, boundaries).

Two kinds of read, because the lab found the difference the hard way: `get_position()` / `get_servo_angle()` are
round trips that can block for up to 2 s on a faulted arm, while the properties `position`, `state`, `error_code` are
published continuously by the controller and cost nothing. The loop that watches a move (`*_now` calls) uses only the
published ones, so the hold-to-run switch is never waiting behind a blocked read. The measured joints exist only as a
round trip; they are read between segments, when the arm is at rest (`lag_mm`).

`XArm` has NOT been run on the arm yet: its first use is `status`, read-only.
"""
from __future__ import annotations

import time

import numpy as np

# xArm 7 chain (mujoco_menagerie `ufactory_xarm7`): per link an offset [m] and a twist about x [rad], then the joint's
# rotation about z. Used only for DIFFERENCES of the measured flange position, where its absolute error cancels.
_CHAIN = np.array([[0.0, 0.0, 0.267, 0.0], [0.0, 0.0, 0.0, -np.pi / 2], [0.0, -0.293, 0.0, np.pi / 2], [0.0525, 0.0, 0.0, np.pi / 2],
                   [0.0775, -0.3425, 0.0, np.pi / 2], [0.0, 0.0, 0.0, np.pi / 2], [0.076, 0.097, 0.0, -np.pi / 2]])


def flange_mm(joints_deg):
    """Flange position [mm] in the arm's base frame from seven joint angles [deg]."""
    T = np.eye(4)
    for (x, y, z, a), th in zip(_CHAIN, np.radians(np.asarray(joints_deg, float)[:7])):
        ca, sa, ct, st = np.cos(a), np.sin(a), np.cos(th), np.sin(th)
        T = T @ np.array([[ct, -st, 0.0, x], [ca * st, ca * ct, -sa, y], [sa * st, sa * ct, ca, z], [0.0, 0.0, 0.0, 1.0]])
    return T[:3, 3] * 1e3


def rpy_matrix(rpy_deg):
    """The controller's roll / pitch / yaw [deg] as a rotation matrix: R = Rz(yaw) Ry(pitch) Rx(roll). Its third column
    is the gripper's own axis in the base frame."""
    r, p, y = np.radians(np.asarray(rpy_deg, float))
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def matrix_rpy(R):
    R = np.asarray(R, float)
    return np.degrees([np.arctan2(R[2, 1], R[2, 2]), np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0])), np.arctan2(R[1, 0], R[0, 0])])


def twist_deg(R_from, R_to):
    """How far the gripper must spin about ITS OWN axis to get from one orientation to the other [deg], and how far the
    two axes are apart [deg] (a pure spin has 0 there)."""
    Q = np.asarray(R_from, float).T @ np.asarray(R_to, float)
    return float(np.degrees(np.arctan2(Q[1, 0], Q[0, 0]))), float(np.degrees(np.arccos(np.clip(Q[2, 2], -1.0, 1.0))))


def _rz(deg):
    c, s_ = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


def run_fake_clock(arm, scale=1.0, alive=lambda: True):
    """Let a stand-in arm move in real time (a background thread, until `alive()` says no)."""
    import threading

    def loop():
        last = time.monotonic()
        while alive():
            time.sleep(0.005)
            now = time.monotonic()
            arm.advance((now - last) * scale)
            last = now

    threading.Thread(target=loop, daemon=True).start()


class ArmFault(RuntimeError):
    """The controller reported an error or refused a call. The run ends; a person looks at the arm."""


class FakeArm:
    """Moves in straight lines at the commanded speed in simulated time (`advance`), grips a brick if one is under the
    fingers, can be blocked by an obstacle (the PLAN goes on, the arm does not: what a position-controlled arm does
    against something that does not give) and can be told to fail. Everything it is asked to do is kept in `log`."""

    BRICK_COUNTS = 315  # the jaws stall on a 31.5 mm brick

    def __init__(self, pose=(450.0, 0.0, 300.0, 180.0, 0.0, 0.0), bricks=None, up=(0.0, 0.0, 1.0)):
        self.p, self.rpy, self.up = np.array(pose[:3], float), np.array(pose[3:], float), np.array(up, float)
        self.plan = self.p.copy()  # where the controller thinks it is (what get_position reports)
        self.target, self.speed, self.paused, self.stopped = None, 0.0, False, True
        self.err, self.enabled, self.connected = 0, False, False
        self.counts, self.holding = 850, None
        self.bricks = {k: np.array(v, float) for k, v in (bricks or {}).items()}  # name -> where the TCP is when it holds that brick centred
        self.log, self.t, self.fault_when, self.blocked = [], 0.0, None, None  # blocked(p) -> True where the arm cannot go
        self.j7, self.j7_real, self.turn_target, self.turn_speed = 0.0, 0.0, None, 0.0  # the last joint: planned, measured, where it is going
        self.turn_sign, self.turn_stuck = 1.0, False  # which way the gripper spins for +joint 7; a gripper that is held and cannot turn

    # --- simulated time
    def advance(self, dt):
        self.t += dt
        if self.fault_when is not None and not self.err:
            self.err = int(self.fault_when(self) or 0)
        if self.err:
            self.target = self.turn_target = None
        if self.turn_target is not None and not self.paused and not self.stopped:
            step = float(np.clip(self.turn_target - self.j7, -self.turn_speed * dt, self.turn_speed * dt))
            self.j7 += step
            if abs(self.turn_target - self.j7) < 1e-9:
                self.turn_target = None
            if not self.turn_stuck:
                self.j7_real = self.j7
                self.rpy = matrix_rpy(rpy_matrix(self.rpy) @ _rz(self.turn_sign * step))
        if self.target is None or self.paused or self.stopped:
            return
        d = self.target - self.plan
        dist = float(np.linalg.norm(d))
        step = self.speed * dt
        self.plan = self.target.copy() if step >= dist else self.plan + d / dist * step
        if step >= dist:
            self.target = None
        if self.blocked is None or not self.blocked(self.plan):
            self.p = self.plan.copy()
        if self.holding is not None:
            self.bricks[self.holding[0]] = self.p + self.holding[1]

    # --- the interface
    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def fault(self):
        return int(self.err)

    fault_now = fault

    def ready(self):
        return f"controller error {self.err}" if self.err else ""

    def enable(self):
        if self.err:
            raise ArmFault(self.ready())
        self.enabled, self.stopped, self.paused = True, False, False  # the plan stays where it was: a deflection persists
        self.log.append(("enable", self.t))

    def pose(self):
        return np.concatenate([self.plan, self.rpy])

    def xyz_now(self):
        return self.plan.copy()

    def lag_mm(self):
        return self.p - self.plan

    def joints(self, measured=True):
        return [0.0] * 6 + [self.j7_real if measured else self.j7]

    def joint_lag_deg(self):
        return abs(self.j7_real - self.j7)

    def j7_now(self):
        return self.j7

    def move_rel(self, d, speed, acc):
        if not self.enabled or self.stopped:
            raise ArmFault("a move was sent to an arm that is not enabled")
        self.target, self.speed = self.plan + np.asarray(d, float), float(speed)
        self.log.append(("move", self.t, tuple(float(v) for v in d), float(speed), self.target.copy()))

    def turn_rel(self, deg, speed, acc):
        if not self.enabled or self.stopped:
            raise ArmFault("a turn was sent to an arm that is not enabled")
        self.turn_target, self.turn_speed = self.j7 + float(deg), float(speed)
        self.log.append(("turn", self.t, float(deg), float(speed)))

    def moving_now(self):
        return (self.target is not None or self.turn_target is not None) and not self.paused and not self.stopped

    def pause(self):
        self.paused = True
        self.log.append(("pause", self.t))

    def resume(self):
        self.paused = False
        self.log.append(("resume", self.t))

    def stop(self):
        self.stopped, self.target, self.turn_target = True, None, None
        self.log.append(("stop", self.t))

    def grip(self, counts):
        self.log.append(("grip", self.t, int(counts)))
        if counts >= 800:  # open: a held brick stays where it is
            self.counts, self.holding = int(counts), None
            return self.counts
        for name, at in self.bricks.items():
            off = at - self.p
            high = float(off @ self.up)  # the jaws must be at the brick's height, and within reach along the plate
            if abs(high) <= 3.0 and float(np.linalg.norm(off - high * self.up)) <= 20.0:
                self.holding, self.counts = (name, off), self.BRICK_COUNTS
                return self.counts
        self.counts = int(counts)
        return self.counts

    def status(self):
        return dict(error=self.err, warn=0, state=1 if self.moving_now() else 2, mode=0, pose=self.pose().tolist(), gripper=self.counts, fake=True)


class XArm:
    """The real arm through UFACTORY's Python SDK (pinned by the lab at v1.18.4). The SDK is imported in `connect`
    and nowhere else, so a dry run never loads it."""

    def __init__(self, ip, gripper_speed=2000):
        self.ip, self.api, self.gripper_speed, self._gripper_on = str(ip), None, int(gripper_speed), False

    def connect(self):
        from xarm.wrapper import XArmAPI

        self.api = XArmAPI(self.ip, is_radian=False)
        for _ in range(40):  # the published state reads all zeros until the first reports arrive
            time.sleep(0.05)
            if any(abs(float(v)) > 1e-6 for v in (self.api.angles or [0.0])):
                return
        raise ArmFault("the arm is connected but publishes no state")

    def disconnect(self):
        if self.api is not None:
            self.api.disconnect()
            self.api = None

    def _ok(self, code, what):
        if code != 0:
            raise ArmFault(f"{what} was refused by the controller (code {code})")

    def _err_warn(self):
        code, ew = self.api.get_err_warn_code()
        if code != 0:
            raise ArmFault(f"could not read the error code (code {code})")
        return int(ew[0]), int(ew[1])

    # --- round trips: at the start, between moves, for the label
    def fault(self):
        return self._err_warn()[0]

    def ready(self):
        """'' if the arm may be enabled, otherwise why not. An error is never cleared from here."""
        err, _ = self._err_warn()
        if err:
            return f"controller error {err}: look at the arm and clear it in the dashboard, then start again"
        code, _ = self.api.get_state()
        return "" if code == 0 else f"could not read the state (code {code})"

    def enable(self):
        problem = self.ready()
        if problem:
            raise ArmFault(problem)
        self._ok(self.api.motion_enable(enable=True), "motion_enable")
        self._ok(self.api.set_mode(0), "set_mode(0)")  # position mode: planned moves
        self._ok(self.api.set_state(0), "set_state(0)")
        time.sleep(0.5)
        code, state = self.api.get_state()
        if code != 0 or int(state) == 4 or self.fault():
            raise ArmFault(f"the arm did not enable (state {state}, error {self.fault()})")

    def pose(self):
        code, p = self.api.get_position(is_radian=False)
        self._ok(code, "get_position")
        return np.array(p[:6], float)

    def joints(self, measured=True):
        code, a = self.api.get_servo_angle(is_radian=False, is_real=bool(measured))
        self._ok(code, "get_servo_angle")
        return [float(v) for v in a[:7]]

    def lag_mm(self):
        """Where the arm IS minus where the controller PLANS it to be, as a vector [mm]: the flange position from the
        measured joints minus the same from the planned joints (two round trips; the kinematic model's own error
        cancels in the difference). Near zero for a free arm at rest; grows when something holds the arm back."""
        return flange_mm(self.joints(True)) - flange_mm(self.joints(False))

    # --- published state: free to read, used while a move is watched
    def xyz_now(self):
        return np.array(self.api.position[:3], float)

    def fault_now(self):
        return int(self.api.error_code or 0)

    def moving_now(self):
        return int(self.api.state) == 1

    def joint_lag_deg(self):
        """The largest difference between a measured and a planned joint angle [deg] (two round trips)."""
        return float(max(abs(a - b) for a, b in zip(self.joints(True), self.joints(False))))

    def j7_now(self):
        return float(self.api.angles[6])

    # --- commands
    def turn_rel(self, deg, speed, acc):
        """Spin the gripper by moving joint 7 only: all seven planned angles are read and sent back with the last one
        changed, exactly as the lab's own rotate does."""
        here = self.joints(False)
        here[6] += float(deg)
        self._ok(self.api.set_servo_angle(angle=here, speed=float(speed), mvacc=float(acc), wait=False, is_radian=False), "set_servo_angle")

    def move_rel(self, d, speed, acc):
        self._ok(self.api.set_position(x=float(d[0]), y=float(d[1]), z=float(d[2]), roll=0.0, pitch=0.0, yaw=0.0, relative=True,
                                       speed=float(speed), mvacc=float(acc), wait=False, is_radian=False), "set_position")

    def pause(self):
        self._ok(self.api.set_state(3), "pause")

    def resume(self):
        self._ok(self.api.set_state(0), "resume")

    def stop(self):
        self.api.set_state(4)  # no check: this is what runs when something is already wrong

    def grip(self, counts):
        """Send the gripper to `counts` and return where it came to rest (it stalls on a brick)."""
        if not self._gripper_on:
            self._ok(self.api.set_gripper_mode(0), "set_gripper_mode")
            self._ok(self.api.set_gripper_enable(True), "set_gripper_enable")
            self._gripper_on = True
        self._ok(self.api.set_gripper_position(int(counts), wait=False, speed=self.gripper_speed), "set_gripper_position")
        last, still, t0 = None, 0, time.monotonic()
        while time.monotonic() - t0 < 4.0:
            time.sleep(0.1)
            code, pos = self.api.get_gripper_position()
            if code != 0 or pos is None:
                continue
            still = still + 1 if (last is not None and abs(pos - last) <= 2) else 0
            last = pos
            if still >= 3:
                break
        if last is None:
            raise ArmFault("the gripper did not report its position")
        return int(last)

    def status(self):
        err, warn = self._err_warn()
        _, state = self.api.get_state()
        code, grip = self.api.get_gripper_position()
        return dict(error=err, warn=warn, state=int(state), mode=int(getattr(self.api, "mode", -1)), pose=self.pose().round(2).tolist(),
                    joints_planned=[round(v, 3) for v in self.joints(False)], joints_measured=[round(v, 3) for v in self.joints(True)],
                    gripper=None if code != 0 else grip, tcp_load=getattr(self.api, "tcp_load", None),
                    collision_sensitivity=getattr(self.api, "collision_sensitivity", None), tcp_offset=getattr(self.api, "tcp_offset", None))
