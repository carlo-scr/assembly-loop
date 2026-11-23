"""Named spots, taught by hand: the simplest thing a policy needs in order to move stuff around. No grid, no camera.

Teaching: with the gripper open and pointing at the table, bring it (by teleoperation) to where it should close on the
thing, and save that as a spot. Do that for every place the thing may stand. Everything else follows:
  - every spot remembers how the gripper was TURNED there (spun about its own axis), so a block can be picked up one
    way round and put down another; the gripper's axis itself (its tilt) must be the same at every spot;
  - between spots the arm only translates; turning is done at the travel height, by the last joint alone;
  - it approaches and leaves along its OWN axis (`up` = away from the table along the gripper), by `lift_mm`;
  - it travels between spots at one height, `lift_mm` above the highest spot;
  - the volume it may be in is the box around the taught spots plus a margin, from 2 mm under the lowest spot to a
    little above the travel height;
  - a TOWER is one spot used at several levels: level k is k block heights above the taught spot, and the travel
    height is raised by the tower's height so that a carried block clears it.
File (JSON, millimetres and degrees in the arm's base frame):
{"side", "rpy_deg", "lift_mm", "spots": {"A": {"xyz": [x, y, z], "rpy_deg": [r, p, y]}, ...}}.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .arms import rpy_matrix, twist_deg
from .guard import Limits, Refused


def per_arm(path, side):
    """The file of one arm: the right arm keeps the name as given (its places and its camera fit were made before
    there were two arms), every other arm has its own beside it: spots.json -> spots_left.json, spots_stand-in.json."""
    path = Path(path)
    return path if side in (None, "right") else path.with_name(f"{path.stem}_{side}{path.suffix}")


def tool_axis(rpy_deg):
    """The gripper's own axis (the direction it points) in the base frame, from the controller's roll / pitch / yaw."""
    return rpy_matrix(rpy_deg)[:, 2]


@dataclass
class Spots:
    side: str = "right"
    rpy: np.ndarray | None = None
    lift: float = 80.0
    at: dict = field(default_factory=dict)  # name -> where the gripper closes there
    rot: dict = field(default_factory=dict)  # name -> roll / pitch / yaw of the gripper there (how it is turned)
    extra: float = 0.0  # added to the travel height while a tower is being built (never saved)

    # ------------------------------------------------------------------ teaching
    def add(self, name, pose):
        name = str(name).strip()
        if not re.fullmatch(r"[A-Za-z0-9 _-]{1,12}", name):
            raise Refused("a spot's name is 1 to 12 letters, digits, spaces, - or _")
        pose = np.asarray(pose, float)
        if self.rpy is None or not self.at:
            self.rpy = pose[3:6].copy()
        elif twist_deg(rpy_matrix(self.rpy), rpy_matrix(pose[3:6]))[1] > 2.0:
            raise Refused("the gripper is TILTED differently than at the first spot: it may be turned about its own axis, but its axis must stay the same")
        for other, p in self.at.items():
            if other != name and float(np.linalg.norm(p - pose[:3])) < 25.0:
                raise Refused(f"that is only {np.linalg.norm(p - pose[:3]):.0f} mm from spot {other}: spots must be at least 25 mm apart")
        self.at[name], self.rot[name] = pose[:3].copy(), pose[3:6].copy()

    def remove(self, name):
        self.at.pop(name, None)
        self.rot.pop(name, None)

    def turn(self, name):
        """Roll / pitch / yaw of the gripper as it was taught at this spot."""
        self.grasp(name)
        return self.rot.get(name, self.rpy)

    def tilt_from(self, rpy_deg):
        """How far [deg] the gripper's axis is now from the taught one (its spin about that axis does not count)."""
        return twist_deg(rpy_matrix(self.rpy), rpy_matrix(rpy_deg))[1]

    def check(self):
        if len(self.at) < 2:
            raise Refused("teach at least two spots")
        if not 30.0 <= self.lift <= 200.0:
            raise Refused("the lift is between 30 and 200 mm")
        return self

    # ------------------------------------------------------------------ geometry
    @property
    def up(self):
        return -tool_axis(self.rpy)

    def grasp(self, name):
        if name not in self.at:
            raise Refused(f"there is no spot called {name!r}")
        return self.at[name]

    def travel_w(self):
        return max(float(p @ self.up) for p in self.at.values()) + self.lift + self.extra

    def above(self, p):
        """The point straight 'above' p (along the gripper's axis) at the travel height."""
        p = np.asarray(p, float)
        return p + (self.travel_w() - float(p @ self.up)) * self.up

    def limits(self, margin_mm=60.0, above_mm=40.0, below_mm=2.0, floor_w=None, ceil_w=None, start=None, **kw):
        up = self.up
        a = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(up, a)
        u = u / np.linalg.norm(u)
        v = np.cross(up, u)
        A = np.stack([u, v, up], 1)
        C = np.array([A.T @ p for p in self.at.values()])
        lo = np.array([C[:, 0].min() - margin_mm, C[:, 1].min() - margin_mm, C[:, 2].min() - below_mm if floor_w is None else float(floor_w)])
        top = C[:, 2].max() + self.lift + self.extra + above_mm  # `ceil_w`: where the arm is now, if that is higher (it comes down first)
        hi = np.array([C[:, 0].max() + margin_mm, C[:, 1].max() + margin_mm, top if ceil_w is None else max(top, float(ceil_w))])
        if start is not None:  # where the arm stands now, if that is a little outside (it was stopped while stepping aside): it may come back in from there
            c = A.T @ np.asarray(start, float)
            if np.all(c[:2] >= lo[:2] - 250.0) and np.all(c[:2] <= hi[:2] + 250.0):
                lo[:2], hi[:2] = np.minimum(lo[:2], c[:2] - 2.0), np.maximum(hi[:2], c[:2] + 2.0)
        return Limits(np.zeros(3), A, lo, hi, **kw)

    # ------------------------------------------------------------------ file
    def to_dict(self):
        return dict(side=self.side, rpy_deg=None if self.rpy is None else [round(float(v), 3) for v in self.rpy], lift_mm=self.lift,
                    spots={k: dict(xyz=[round(float(x), 3) for x in v], rpy_deg=[round(float(x), 3) for x in self.turn(k)]) for k, v in self.at.items()})

    @classmethod
    def from_dict(cls, d):
        s = cls(side=str(d.get("side", "right")), rpy=None if d.get("rpy_deg") is None else np.array(d["rpy_deg"], float), lift=float(d.get("lift_mm", 80.0)))
        for k, v in (d.get("spots") or {}).items():
            if isinstance(v, dict):
                s.at[str(k)], s.rot[str(k)] = np.array(v["xyz"], float), np.array(v.get("rpy_deg", d.get("rpy_deg")), float)
            else:  # an older file: positions only, one orientation for all
                s.at[str(k)], s.rot[str(k)] = np.array(v, float), np.array(d["rpy_deg"], float)
        return s

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def demo_spots():
    """Made-up spots on a table in front of the base, for the stand-in arm: A, B, C, and T for a tower."""
    s = Spots(rpy=np.array([180.0, 0.0, 0.0]), lift=80.0)
    s.at = {"A": np.array([430.0, -90.0, 120.0]), "B": np.array([470.0, 20.0, 120.0]), "C": np.array([400.0, 90.0, 120.0]), "T": np.array([520.0, -40.0, 120.0])}
    s.rot = {"A": np.array([180.0, 0.0, 0.0]), "B": np.array([180.0, 0.0, 90.0]), "C": np.array([180.0, 0.0, 30.0]), "T": np.array([180.0, 0.0, 0.0])}  # the gripper turned differently at each
    return s


def plan_move(spots, src, dst, here, P, rehearse=False, src_up=0.0, dst_up=0.0, release_counts=None, grip_mm=None, tighter_mm=0.0):
    """Steps to take the thing at `src` and put it down at `dst`, starting from wherever the arm is (`here`).
    `src_up` / `dst_up` [mm] raise the pick-up / the put-down above the taught spot (the levels of a tower).
    `release_counts`: let go by opening only this far (a design's block, whose neighbours stand close), not fully.
    `grip_mm`: how wide the block is across the jaws, if known. It is then closed on GENTLY: to `P.squeeze_mm` under
    that width and no further, so that the fingers do not bend. The jaws then stop where they were sent whether a block
    is between them or not: the grip cannot tell that the block is there (a job that takes pictures looks afterwards
    whether it arrived, sight.py). It can still tell that something WIDER is there, which stops the jaws short: that
    stops the move (WrongGrip). With `P.squeeze_mm` at 0 the jaws close as far as they go, and closing to any other
    width than the block's stops the move.
    `tighter_mm`: close this much further than that (a second try, after the block was seen to stay behind).
    A rehearsal does the same travel at the travel height and never comes down."""
    if src == dst:
        raise Refused("from and to are the same spot")
    up = spots.up
    gs, gd = spots.grasp(src) + float(src_up) * up, spots.grasp(dst) + float(dst_up) * up
    hs, hd = spots.above(gs), spots.above(gd)
    ns, nd = (src if not src_up else f"{src}, {src_up:.0f} mm up"), (dst if not dst_up else f"{dst}, {dst_up:.0f} mm up")
    S = [dict(name="go to the travel height", kind="move", to=spots.above(here), speed=P.lift),
         dict(name=f"go above {src}", kind="move", to=hs, speed=P.travel),
         dict(name=f"turn the gripper as at {src}", kind="turn", rpy=spots.turn(src)),
         dict(name="open the gripper", kind="grip", counts=P.open_counts)]
    if not rehearse:
        first = gs + min(20.0, spots.lift) * up
        S += [dict(name=f"lower around the thing at {ns}", kind="move", to=gs, speed=P.slow, stall_mm=P.stall_mm),
              dict(name="close on it", kind="grip", counts=P.shut_counts, must_hold=True, expect_mm=grip_mm) if not grip_mm or P.squeeze_mm <= 0 else
              dict(name="close on it, gently", kind="grip", counts=max(0, int(round((float(grip_mm) - P.squeeze_mm - P.jaw_offset_mm - float(tighter_mm)) * 10.0))), expect_mm=grip_mm, gentle=True),
              dict(name="read the grasp", kind="read"),
              dict(name="lift it, slowly at first", kind="move", to=first, speed=P.press, stall_mm=P.pull_stall_mm, let_go_if_blocked=True),
              dict(name="lift to the travel height", kind="move", to=hs, speed=P.lift)]
    else:
        S += [dict(name="close the gripper (in the air)", kind="grip", counts=P.shut_counts)]
    S += [dict(name=f"go above {dst}", kind="move", to=hd, speed=P.travel),
          dict(name=f"turn the gripper as at {dst}", kind="turn", rpy=spots.turn(dst))]
    if not rehearse:
        S += [dict(name=f"lower it onto {nd}", kind="move", to=gd, speed=P.slow, stall_mm=P.stall_mm, rest_ok_mm=P.rest_ok_mm),
              dict(name="read the put-down", kind="read"),
              dict(name="let go", kind="grip", counts=P.open_counts if release_counts is None else int(min(P.open_counts, max(P.shut_counts, release_counts)))),
              dict(name="lift clear", kind="move", to=hd, speed=P.lift)]
    else:
        S += [dict(name="open the gripper (in the air)", kind="grip", counts=P.open_counts)]
    return S
