"""The blocks on the table, and which of them goes onto a tower next. No camera and no arm in here.

A KIND of block is measured once with a ruler and written into `blocks.json`:
{"finger_mm": 40, "jaw_mm": 85,
 "kinds": [{"name": "green bar", "shape": "box", "size_mm": [75, 25, 25], "colour": "green"},
           {"name": "yellow cylinder", "shape": "cylinder", "size_mm": [25, 75], "colour": "yellow"}]}
A box has three edge lengths; a cylinder has a diameter and a length. `finger_mm` is how deep a block may sit between
the fingers before it meets the palm, `jaw_mm` how far the jaws open.

A block can lie on the table in a few ways (its RESTS): a box on any of its three faces, a cylinder upright or lying.
What the camera reports is a block SEEN: its kind, its rest, where the middle of it is on the table and which way its
long side points. The arm takes a block from above across its short side and puts it down the way it found it.

The tower rules (`plan_tower`), all of them about what can be put down from above and stays there:
  - a lying cylinder rolls: it is left out;
  - a block wider across its short side than the jaws open is left out;
  - the widest blocks go to the bottom, and among equally wide ones the lower ones first;
  - the tower ends below `max_mm`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .guard import Refused

COLOUR_NAMES = ("red", "orange", "yellow", "green", "blue", "purple")


@dataclass(frozen=True)
class Rest:
    name: str  # "flat", "on its side", "on end", "upright", "lying"
    height: float  # how high the block stands this way [mm]
    long: float  # its footprint: the long and the short side of the rectangle (for a circle, both are its diameter)
    short: float
    round: bool = False  # the footprint is a circle (an upright cylinder): it has no direction
    rolls: bool = False  # a lying cylinder: it has no flat top and does not stay where it is put

    @property
    def turns(self):
        """After how many degrees the footprint looks the same again (0: it has no direction at all)."""
        return 0.0 if self.round else 90.0 if abs(self.long - self.short) < 1.0 else 180.0


@dataclass(frozen=True)
class Kind:
    name: str
    shape: str  # "box" or "cylinder"
    size: tuple  # box: the three edges; cylinder: (diameter, length) [mm]
    colour: str | None = None

    def check(self):
        if self.shape not in ("box", "cylinder") or len(self.size) != (3 if self.shape == "box" else 2):
            raise Refused(f"block {self.name!r}: a box has three sizes, a cylinder a diameter and a length")
        if not all(5.0 <= float(v) <= 300.0 for v in self.size):
            raise Refused(f"block {self.name!r}: sizes are between 5 and 300 mm")
        if self.colour is not None and self.colour not in COLOUR_NAMES:
            raise Refused(f"block {self.name!r}: the colour is one of {', '.join(COLOUR_NAMES)}, or left out")
        return self

    def rests(self):
        if self.shape == "cylinder":
            d, length = (float(v) for v in self.size)
            return [Rest("upright", length, d, d, round=True), Rest("lying", d, max(length, d), min(length, d), rolls=True)]
        a, b, c = sorted((float(v) for v in self.size), reverse=True)
        out = []
        for name, h, long, short in (("flat", c, a, b), ("on its side", b, a, c), ("on end", a, b, c)):
            if not any(abs(r.height - h) < 0.5 for r in out):  # a cube has one rest, a square bar two
                out.append(Rest(name, h, long, short))
        return out


@dataclass
class Catalogue:
    kinds: list = field(default_factory=list)
    finger_mm: float = 40.0
    jaw_mm: float = 85.0
    jaw_offset_mm: float = 0.0  # how much wider the gap between the fingers really is than the gripper says (fingers that are not UFACTORY's own)
    seen_wider_mm: float = 0.0  # how far each block's colour spills over its edges in the camera's pictures (the fit draws its outline that much wider)

    def check(self):
        if not self.kinds:
            raise Refused("the catalogue of blocks is empty: measure the blocks and write them into blocks.json")
        if len({k.name for k in self.kinds}) != len(self.kinds):
            raise Refused("two kinds of block have the same name")
        if not (10.0 <= self.finger_mm <= 120.0 and 20.0 <= self.jaw_mm <= 120.0):
            raise Refused("finger_mm is between 10 and 120, jaw_mm between 20 and 120")
        if not -15.0 <= self.jaw_offset_mm <= 15.0:
            raise Refused("jaw_offset_mm is between -15 and 15")
        if not 0.0 <= self.seen_wider_mm <= 10.0:
            raise Refused("seen_wider_mm is between 0 and 10")
        for k in self.kinds:
            k.check()
        return self

    def kind(self, name):
        for k in self.kinds:
            if k.name == name:
                return k
        raise Refused(f"there is no kind of block called {name!r}")

    def to_dict(self):
        return dict(finger_mm=self.finger_mm, jaw_mm=self.jaw_mm, jaw_offset_mm=self.jaw_offset_mm, seen_wider_mm=self.seen_wider_mm,
                    kinds=[dict(name=k.name, shape=k.shape, size_mm=list(k.size), **({"colour": k.colour} if k.colour else {})) for k in self.kinds])

    @classmethod
    def from_dict(cls, d):
        try:
            kinds = [Kind(str(k["name"]), str(k["shape"]), tuple(float(v) for v in k["size_mm"]), k.get("colour")) for k in d.get("kinds", [])]
            return cls(kinds, float(d.get("finger_mm", 40.0)), float(d.get("jaw_mm", 85.0)), float(d.get("jaw_offset_mm", 0.0)), float(d.get("seen_wider_mm", 0.0))).check()
        except (KeyError, TypeError, ValueError) as e:
            raise Refused(f"blocks.json is not readable: {e!r}") from None

    @classmethod
    def load(cls, path):
        try:
            return cls.from_dict(json.loads(Path(path).read_text()))
        except (OSError, ValueError) as e:
            raise Refused(f"blocks.json is not readable: {e}") from None

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))


@dataclass
class Seen:
    """One block as the camera made it out. `uv` [mm] is the middle of its footprint in the table's own two
    directions, `angle` [deg] the direction of its long side from the first towards the second of them."""
    kind: Kind
    rest: Rest
    uv: tuple
    angle: float = 0.0
    score: float = 1.0  # how well its outline in the picture fits (1 = exactly)
    colour: str | None = None
    name: str = ""  # a short name for the page and the log, given by `name_all`

    @property
    def grip_mm(self):
        return self.rest.short

    def hold_mm(self, finger_mm):
        """How far above the table the fingertips close on it: a tall block is taken near its top."""
        return max(0.0, self.rest.height - float(finger_mm))


def name_all(seen):
    """Names that tell the blocks apart in the log: "green bar", or "green bar 1" / "green bar 2" left to right."""
    by = {}
    for s in sorted(seen, key=lambda s: (s.kind.name, s.uv[0], s.uv[1])):
        by.setdefault(s.kind.name, []).append(s)
    for kind, group in by.items():
        for i, s in enumerate(group):
            s.name = kind if len(group) == 1 else f"{kind} {i + 1}"
    return seen


def plan_tower(seen, cat, max_mm=250.0, n_max=12, clear_mm=10.0):
    """Which of the seen blocks go onto the tower, bottom first, and which are left out and why.
    Returns (levels, left_out). A level is dict(block, base_mm, hold_mm, top_mm): the block stands `base_mm` above
    the table, is held `hold_mm` above its own bottom, and the tower is `top_mm` high once it is on."""
    usable, left_out = [], []
    for s in seen:
        if s.rest.rolls:
            left_out.append((s, "it is lying and would roll: stand it on its end"))
        elif s.grip_mm > cat.jaw_mm - clear_mm:
            left_out.append((s, f"it is {s.grip_mm:.0f} mm across: too wide for the jaws"))
        else:
            usable.append(s)
    usable.sort(key=lambda s: (-s.rest.short, -s.rest.long, s.rest.height, s.name))
    levels, top = [], 0.0
    for s in usable:
        if len(levels) >= n_max or top + s.rest.height > max_mm:
            left_out.append((s, f"the tower ends at {max_mm:.0f} mm" if len(levels) < n_max else f"the tower ends at {n_max} blocks"))
            continue
        levels.append(dict(block=s, base_mm=top, hold_mm=s.hold_mm(cat.finger_mm), top_mm=top + s.rest.height))
        top += s.rest.height
    return levels, left_out
