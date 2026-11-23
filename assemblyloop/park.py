"""Moving the OTHER arm out of the way, so that the arm of a session has the table to itself.

The rig has two arms side by side, mounted like shoulders, with collision detection off on both. Teaching or running
one arm while the other stands over the table invites a collision. So the dashboard offers one thing for the other
arm: fold it away to the resting pose the lab's own tools use.

What it does is the lab's own "Rehome", in the lab's order (PO-Assembly `pi05/deploy/rehome.py`, which copies
manipulation-mono's sanity-check flow; xarm7-data-collection's `swoosh-home --park-left` uses the same pose):
  1. open the gripper before the arm moves, so that a block still held is let go where it is, not carried off;
  2. unless it is folded nearly already: go to the lab's waypoint in front of the arm (400, 0, 300 mm in that arm's own
     base frame, gripper down). A direct joint move from over the table to the folded pose once swept the lab's left
     gripper through the head-camera mount and broke the camera (2 Sep 2026); from this waypoint the lab has folded
     both arms many times. On the way there it first comes straight up towards the waypoint's height, then goes across;
  3. fold to the resting pose in joint space, at the lab's 20 deg/s.
What is OURS (the lab's tools do none of it): every move goes through the guard, so it moves only while the hold-to-run
switch is held, in short pieces, each one checked; a controller error is never cleared from here (the lab's tools
clear it); a gripper that is tilted is first pointed straight down in place, in small watched pieces (`Guard.level`),
so that it reaches the waypoint as the lab's arms do; and nothing is sent unless the controller says that every point
on the way is within reach. Afterwards the arm is left stopped and the connection closed, as the lab's tools leave it.
"""
from __future__ import annotations

import numpy as np

from .arms import matrix_rpy, rpy_matrix
from .guard import Guard, Limits, Refused

# The lab's resting ("pre-retracted") pose [deg], joints 1 to 7, the two arms mirror images of each other. From
# PO-Assembly rehome.py HOME_DEG, which cites four copies in manipulation-mono that agree; xarm7-data-collection's
# `arm.left_park` is the same pose. There the flange is about 440 mm above the arm's shoulder and 440 mm out to its own
# side, the gripper pointing forward and a little down: out of the space between the arms.
PARK_DEG = {"right": (4.0, -37.0, -3.0, 49.0, -7.0, 16.0, -40.0), "left": (-4.0, -37.0, 3.0, 49.0, 7.0, 16.0, 40.0)}
VIA_MM = (400.0, 0.0, 300.0)  # the lab's waypoint (platecal START_XYZ, the same for both arms in each one's own base frame)
VIA_SPEED_MM_S = 60.0  # the lab's speed towards it (platecal MOVE_SPEED_MM_S)
FOLD_SPEED_DEG_S, FOLD_ACC_DEG_S2 = 20.0, 150.0  # the lab's Rehome
AT_REST_DEG = 1.0  # every joint this close: it is folded already and nothing is sent (the lab's HOME_TOLERANCE_DEG)
NEAR_DEG = 15.0  # every joint this close: folded from where it is, without the waypoint (a small joint move sweeps little)
FAR_DEG = 200.0  # a joint further than this from the resting pose: refused (the lab's MAX_HOME_DELTA_DEG: a misread, or not the arm we think)
STEP_MM = 40.0  # the way to the waypoint is asked of the controller every this many millimetres
DONE = "out of the way: folded to the lab's resting pose and left stopped"


def levelled(rpy_deg, down):
    """Where `Guard.level` ends from `rpy_deg`: the gripper's axis turned onto `down` the shortest way."""
    R0, down = rpy_matrix(rpy_deg), np.asarray(down, float)
    k = np.cross(R0[:, 2], down)
    s, c = float(np.linalg.norm(k)), float(np.clip(R0[:, 2] @ down, -1.0, 1.0))
    if s < 1e-9:
        return np.asarray(rpy_deg, float)
    k = k / s
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    a = float(np.arctan2(s, c))
    return matrix_rpy((np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * K @ K) @ R0)


def world_axes(up):
    """Forward (the base's x, levelled), left and up as columns, in the arm's base frame."""
    up = np.asarray(up, float) / np.linalg.norm(up)
    x = np.array([1.0, 0.0, 0.0])
    fwd = x - (x @ up) * up
    fwd = fwd / np.linalg.norm(fwd)
    return np.stack([fwd, np.cross(up, fwd), up], 1)


def _line(a, b):
    n = max(int(np.ceil(float(np.linalg.norm(b - a)) / STEP_MM)), 1)
    return [a + (b - a) * (i + 1) / n for i in range(n)]


def way_up(arm, xyz, rpy, up, via):
    """The point it comes straight up to before it goes across to the waypoint: as high as the waypoint, or as much of
    that as the arm reaches (the rest is climbed on the way across). Every point on the way is asked of the
    controller first; nothing moves. Refused if there is no such way."""
    rise = max(0.0, float((via - xyz) @ up))
    for part in ((1.0, 0.75, 0.5, 0.25, 0.0) if rise > 1.0 else (0.0,)):
        top = xyz + part * rise * up
        why = next((w for w in (arm.reach(p, rpy) for p in _line(xyz, top) + _line(top, via)) if w), "")
        if not why:
            return top
    raise Refused(f"the way to the lab's waypoint is not within its reach from here ({why}): bring it closer to the waypoint in front of it by hand, or in UFACTORY Studio")


def plan(arm, side, down, open_counts=850):
    """What folding `arm` away would do from where it stands, as [(kind, words, point or None)], or a reason (str)
    why nothing needs to be sent. Asks the controller only; nothing moves. Refused if it must not be done."""
    home = np.array(PARK_DEG[side], float)
    problem = arm.ready()
    if problem:
        raise Refused(problem)
    if arm.moving_now():
        raise Refused("it is moving: another program may be driving it (close the lab's teleoperation first)")
    q = np.asarray(arm.joints(False), float)[:7]
    worst = float(np.max(np.abs(q - home)))
    if worst <= AT_REST_DEG:
        return "already out of the way (in the lab's resting pose): nothing was sent"
    if worst > FAR_DEG:
        j = int(np.argmax(np.abs(q - home)))
        raise Refused(f"joint {j + 1} is at {q[j]:.0f} deg, {worst:.0f} deg from the resting pose: check the arm, then move it by hand or in UFACTORY Studio")
    counts = arm.gripper_now()
    if counts is None:
        raise Refused("its gripper does not say where it is (a gripper fault?): look at it in UFACTORY Studio")
    steps = [("open", "open the gripper", None)] if counts < open_counts - 20 else []
    if worst > NEAR_DEG:
        pose, down = np.asarray(arm.pose(), float), np.asarray(down, float) / np.linalg.norm(down)
        tilt = float(np.degrees(np.arccos(np.clip(rpy_matrix(pose[3:])[:, 2] @ down, -1.0, 1.0))))
        if tilt > 95.0:
            raise Refused(f"its gripper points {tilt:.0f} deg away from straight down: too far to turn it down in small pieces. Move it by hand or in UFACTORY Studio")
        rpy = pose[3:]
        if tilt > 2.0:
            steps.append(("level", f"point its gripper down (tilted {tilt:.0f} deg)", None))
            rpy = levelled(rpy, down)
        via = np.array(VIA_MM, float)
        top = way_up(arm, pose[:3], rpy, -down, via)
        rise = float((top - pose[:3]) @ -down)
        if rise > 1.0:
            steps.append(("up", f"come straight up {rise:.0f} mm", top))
        steps.append(("via", "go to the lab's waypoint in front of it", via))
    steps.append(("fold", "fold it to the lab's resting pose", None))
    return steps


def fold_away(arm, gate, side, down, job, log_path=None, say=print, limit_kw=None, guard_kw=None, open_counts=850):
    """Fold `arm` (connected; NOT the arm of the session) away to the lab's resting pose, through a guard of its own
    that moves only while `gate` is held. `job` is told the steps (`steps`, `i`) and gets the guard (`guard`), so that
    a stop button can abort it; a `cancel` already in `job` aborts it at once. Returns what happened, in words.
    Refused: nothing was sent. Released, Blocked, ArmFault: the arm was stopped where it was."""
    steps = plan(arm, side, down, open_counts)
    if isinstance(steps, str):
        return steps
    job["steps"] = [w for _, w, _ in steps]
    here = np.asarray(arm.pose(), float)[:3]
    axes = world_axes(-np.asarray(down, float))
    c = [np.zeros(3)] + [axes.T @ (p - here) for _, _, p in steps if p is not None]  # a box around the way: the guard keeps the flange in it
    lim = Limits(here.copy(), axes, np.min(c, 0) - 30.0, np.max(c, 0) + 30.0, **(limit_kw or {}))
    guard = Guard(arm, gate, lim, log_path=log_path, say=say, **(guard_kw or {}))
    job["guard"] = guard
    if job.get("cancel"):
        guard.abort(job["cancel"], latch=True)
    try:
        guard._check_abort()
        guard.start()  # switches it on, as Arm does for the session's own arm: nothing moves yet
        for i, (kind, words, p) in enumerate(steps):
            job["i"] = i
            if kind == "open":
                guard.grip(open_counts, "open the gripper before it moves")
            elif kind == "level":
                guard.level("point its gripper down", down=down)
            elif kind in ("up", "via"):
                guard.move_to(p, VIA_SPEED_MM_S, words)
            else:
                guard.joints_to(PARK_DEG[side], words, FOLD_SPEED_DEG_S, FOLD_ACC_DEG_S2)
        off = float(np.max(np.abs(np.asarray(arm.joints(False), float)[:7] - np.array(PARK_DEG[side]))))
        if off > AT_REST_DEG:
            guard._fail(f"it ended {off:.1f} deg from the resting pose")
        return DONE
    finally:
        guard.finish()
