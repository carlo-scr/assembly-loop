"""One cycle: take the brick that is seated on stud `src`, gripping it `offset_mm` off-centre along the fingers, and
seat it on stud `dst`, aiming with `correction_mm`. A plan is a list of named steps so that a dry run can print it and
a live run can execute it one step at a time.

  correction = 0        the placer assumes the brick is centred: it lands `offset` off the stud
  correction = offset   the placer knows how it holds the brick: it lands on the stud (the sanity case, run first)
  correction = estimate later: read from the wrist camera
Speeds follow the lab's own brick code on this rig: 60 mm/s to travel, 30 to approach, 15 to press and to pull.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .guard import Blocked, Refused


class NothingHeld(RuntimeError):
    """The jaws closed on nothing. The arm was lifted clear with the gripper open."""


class WrongWidth(NothingHeld):
    """The jaws closed on something that is not as wide as the block they were sent for. It was let go where it
    stood and the arm was lifted clear with the gripper open."""


@dataclass
class Params:
    travel: float = 100.0  # across, at the travel height, with nothing near
    lift: float = 60.0  # straight up to the travel height, and nudging upwards
    slow: float = 30.0  # coming down, and every nudge that is watched for contact
    press: float = 15.0
    engage_mm: float = 18.0  # the last stretch above seated is done at press speed
    overtravel_mm: float = 1.0  # how far past the taught grasp height the press may be planned (the lab uses 2, at most 3)
    pull_step_mm: float = 5.0
    shake_mm: float = 1.5  # sideways, across the fingers, while the brick comes off its studs
    shake_steps: int = 4
    open_counts: int = 850
    shut_counts: int = 0
    held_min_counts: int = 100  # the jaws stalled on something if they stop above this
    max_offset_mm: float = 15.0
    stall_mm: float = 2.5  # lowering and pressing end once the measured arm is this far behind the plan
    pull_stall_mm: float = 4.0  # pulling a brick off its studs is allowed a little more, then it counts as stuck
    drop_mm: float = 2.0  # on a tower a block is let go this far above where it should rest, so that nothing is pressed
    rest_ok_mm: float = 6.0  # a put-down that meets something this close to its target counts as put down
    width_tol_mm: float = 6.0  # a block the camera named is taken for another one if the jaws stop this far from its width
    counts_per_mm: float = 10.0  # the gripper's counts for a millimetre between the jaws (UFACTORY's gripper: 850 counts = 85 mm)
    push_stall_mm: float = 3.0  # a push along the table ends once the arm is this far behind its plan: what is pushed does not slide

    def check(self):
        if not (0 < self.press <= self.slow <= self.lift <= self.travel <= 120.0):
            raise Refused("speeds must satisfy 0 < press <= slow <= lift <= travel <= 120 mm/s")
        if not (0.0 <= self.overtravel_mm <= 2.0):
            raise Refused("overtravel must be between 0 and 2 mm")
        if self.shake_mm > 3.0:
            raise Refused("shake is at most 3 mm")
        if not (0.5 <= self.stall_mm <= 4.0 and self.stall_mm <= self.pull_stall_mm <= 6.0):
            raise Refused("stall limits: 0.5 to 4 mm to press, up to 6 mm to pull")
        if not (0.0 <= self.drop_mm <= 5.0 and 0.0 <= self.rest_ok_mm <= 10.0):
            raise Refused("a block is let go at most 5 mm above its place, and may rest at most 10 mm short of it")
        if not (2.0 <= self.width_tol_mm <= 15.0 and 1.0 <= self.push_stall_mm <= 4.0):
            raise Refused("the width of a held block may be 2 to 15 mm off; a push ends 1 to 4 mm behind its plan")


def plan(plate, src, dst, offset_mm=0.0, correction_mm=0.0, P=None, air_mm=0.0):
    P = P or Params()
    P.check()
    if not 0.0 <= float(air_mm) <= 300.0:
        raise Refused("the whole cycle can be lifted 0 to 300 mm into the air, never lowered")
    for name, v in (("offset", offset_mm), ("correction", correction_mm)):
        if abs(v) > P.max_offset_mm:
            raise Refused(f"{name} of {v:.1f} mm is beyond {P.max_offset_mm:.0f} mm")
    f, c, n = plate.f(), plate.c(), plate.n
    up = air_mm * n
    hs, gs = plate.hover(src) + up + offset_mm * f, plate.grasp(src) + up + offset_mm * f
    hd, gd = plate.hover(dst) + up + correction_mm * f, plate.grasp(dst) + up + correction_mm * f
    S = [dict(name="go above the brick", kind="move", to=hs, speed=P.travel),
         dict(name="open the gripper", kind="grip", counts=P.open_counts),
         dict(name="lower the jaws around the brick", kind="move", to=gs, speed=P.slow, stall_mm=P.stall_mm),
         dict(name="close on the brick", kind="grip", counts=P.shut_counts, must_hold=True),
         dict(name="read the grasp", kind="read")]
    for k in range(1, P.shake_steps + 1):
        h = k * P.pull_step_mm
        if h >= plate.grasp_drop:
            break
        S.append(dict(name=f"pull the brick off, step {k}", kind="move", to=gs + h * n + (P.shake_mm if k % 2 else -P.shake_mm) * c, speed=P.press, stall_mm=P.pull_stall_mm,
                      let_go_if_blocked=True))
    S += [dict(name="lift", kind="move", to=hs, speed=P.slow),
          dict(name="read the carry", kind="read"),
          dict(name="go above the target stud", kind="move", to=hd, speed=P.travel),
          dict(name="lower onto the studs", kind="move", to=gd + P.engage_mm * n, speed=P.slow, stall_mm=P.stall_mm),
          dict(name="press home", kind="move", to=gd - P.overtravel_mm * n, speed=P.press, stall_mm=P.stall_mm),
          dict(name="read the seat", kind="read"),
          dict(name="let go", kind="grip", counts=P.open_counts),
          dict(name="lift clear", kind="move", to=hd, speed=P.slow)]
    return S


def execute(guard, steps, P=None, air=False, before=None, after=None):
    """Run a plan through the guard. `before(step)` / `after(step)` are called around every step (the step mode asks
    for a key there). Returns what was read and how the cycle ended:
      - the jaws close on nothing (not in the air): gripper opened, arm lifted clear, NothingHeld raised;
      - a watched move is blocked: the arm is enabled again and lifted straight back to the hover height it last came
        from, WITHOUT opening the gripper, and the result says where (`blocked`). The run can go on;
      - a put-down (a step with `rest_ok_mm`) that is blocked that close to its target stands on what is below it:
        that IS the put-down, and the plan goes on (`rested` says how far short);
      - a grip step with `width_mm` closes on something of another width (not in the air): it is let go where it
        stands, the arm is lifted clear, WrongWidth raised;
      - a step with `stop_ok` (a push) that is blocked has simply pushed as far as it could: the plan goes on;
      - a step with `straight_up` = (the direction 'up', the height to go back to): when blocked, the arm is lifted
        along that direction instead of back along the line it came."""
    P = P or Params()
    out = dict(reads={}, grips={}, blocked=None, rested=None)
    hover = next(s["to"] for s in steps if s["kind"] == "move")
    last_hover = hover
    for s in steps:
        if before:
            before(s)
        try:
            if s["kind"] == "move":
                try:
                    guard.move_to(s["to"], s["speed"], s["name"], stall_mm=s.get("stall_mm"))
                except Blocked as e:
                    short = float(np.linalg.norm(np.asarray(s["to"], float) - guard.where()))
                    if s.get("rest_ok_mm") is not None and short <= s["rest_ok_mm"]:
                        guard.recover()
                        out["rested"] = dict(step=s["name"], short_mm=round(short, 1))
                        continue
                    if s.get("stop_ok"):
                        guard.recover()
                        out["pushed_short"] = dict(step=s["name"], short_mm=round(short, 1))
                        continue
                    out["blocked"] = dict(step=s["name"], why=str(e))
                    guard.recover()
                    if s.get("let_go_if_blocked"):  # the thing will not come off: let go of it before lifting
                        guard.grip(P.open_counts, "let go: the brick is stuck")
                    here = guard.where()
                    up = last_hover - s["to"]  # straight back up along the line the arm came down
                    lift = here + up * (float((last_hover - here) @ up) / float(up @ up)) if float(up @ up) > 1e-9 else last_hover
                    if s.get("straight_up") is not None:
                        u, w = np.asarray(s["straight_up"][0], float), float(s["straight_up"][1])
                        lift = here + max(0.0, w - float(here @ u)) * u
                    guard.move_to(lift, P.press, "lift clear: the move was blocked")
                    return out
                if s.get("stall_mm") is None:
                    last_hover = s["to"]
            elif s["kind"] == "rise":  # straight up from wherever the arm is now, to a height
                here, u = guard.where(), np.asarray(s["up"], float)
                if float(s["w"]) - float(here @ u) > 1e-3:
                    guard.move_to(here + (float(s["w"]) - float(here @ u)) * u, s["speed"], s["name"])
                last_hover = guard.where()
            elif s["kind"] == "turn":
                try:
                    guard.turn_to(s["rpy"], s["name"])
                except Blocked as e:  # turns happen up in the air: stop there and say so
                    out["blocked"] = dict(step=s["name"], why=str(e))
                    guard.recover()
                    return out
            elif s["kind"] == "grip":
                got = guard.grip(s["counts"], s["name"])
                out["grips"][s["name"]] = got
                if s.get("must_hold") and not air and got < P.held_min_counts:
                    guard.grip(P.open_counts, "open again: nothing was held")
                    guard.move_to(hover if s.get("back_to") is None else s["back_to"], P.slow, "lift clear: nothing was held")
                    raise NothingHeld(f"the jaws closed to {got} counts: no brick")
                if s.get("width_mm") is not None and not air and abs(got / P.counts_per_mm - s["width_mm"]) > P.width_tol_mm:
                    guard.grip(P.open_counts, "open again: this is not the block that was meant")
                    guard.move_to(hover if s.get("back_to") is None else s["back_to"], P.slow, "lift clear: another block was held")
                    raise WrongWidth(f"the jaws stopped {got / P.counts_per_mm:.0f} mm apart, but the block that was meant is {s['width_mm']:.0f} mm across")
            elif s["kind"] == "read":
                out["reads"][s["name"]] = guard.read(s["name"])
        finally:
            if after:
                after(s)
    return out
