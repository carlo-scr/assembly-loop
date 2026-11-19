"""Where the studs are, from poses taught by hand. Nothing here is computed from the robot's mounting or from a
camera: a person brings the open gripper over a seated brick at three studs and lowers it once to the height where
the jaws should close, and every other position is a sum of those taught vectors. All moves keep the taught
orientation, so the controller is only ever asked to translate.

Taught file (JSON, millimetres in the arm's base frame, as `get_position` reports them):
  {"side": "right", "rpy_deg": [r, p, y],
   "taught": {"origin": {"cell": "C3", "hover_mm": [x, y, z]},     gripper centred over the brick on this stud
              "col":    {"cell": "F3", "hover_mm": [x, y, z]},     same row, another column
              "row":    {"cell": "C6", "hover_mm": [x, y, z]}},    same column, another row
   "grasp_mm": [x, y, z],        at the origin stud, lowered to where the jaws close on the seated brick
   "fingers": "col" | "row",     the plate direction along which a brick could slide between the jaws
   "cells": ["B2", "G7"]}        optional: the corner cells of the area the loop may use (default: the taught ones)
Cells are named like a chess board: letter = column, number = row.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .guard import Limits, Refused


def cell(name):
    """'C3' -> (2, 2): zero-based (column, row)."""
    name = str(name).strip().upper()
    if len(name) < 2 or not name[0].isalpha() or not name[1:].isdigit() or int(name[1:]) < 1:
        raise ValueError(f"{name!r} is not a cell like C3")
    return ord(name[0]) - ord("A"), int(name[1:]) - 1


@dataclass
class Plate:
    side: str
    rpy: np.ndarray
    c0: tuple
    origin: np.ndarray  # hover point over the origin cell
    ec: np.ndarray  # one stud along the columns
    er: np.ndarray  # one stud along the rows
    n: np.ndarray  # unit normal, pointing from the plate towards the hover points
    grasp_drop: float
    fingers: str
    area: tuple  # ((cmin, rmin), (cmax, rmax))

    @classmethod
    def load(cls, path):
        return cls.from_taught(json.loads(Path(path).read_text()))

    @classmethod
    def from_taught(cls, d):
        t = d["taught"]
        (c0, r0), (c1, r1), (c2, r2) = cell(t["origin"]["cell"]), cell(t["col"]["cell"]), cell(t["row"]["cell"])
        if r1 != r0 or c1 == c0 or c2 != c0 or r2 == r0:
            raise Refused("taught cells: 'col' must share the origin's row and 'row' the origin's column")
        o, pc, pr = (np.array(t[k]["hover_mm"], float) for k in ("origin", "col", "row"))
        ec, er = (pc - o) / (c1 - c0), (pr - o) / (r2 - r0)
        for name, v in (("columns", ec), ("rows", er)):
            if not 30.5 <= np.linalg.norm(v) <= 33.0:
                raise Refused(f"taught stud pitch along the {name} is {np.linalg.norm(v):.2f} mm; a stud is 31.5 to 32 mm: re-teach")
        ang = float(np.degrees(np.arccos(np.clip(ec @ er / np.linalg.norm(ec) / np.linalg.norm(er), -1, 1))))
        if abs(ang - 90.0) > 2.0:
            raise Refused(f"taught columns and rows meet at {ang:.1f} deg, not 90: re-teach")
        n = np.cross(ec, er)
        n = n / np.linalg.norm(n)
        g = np.array(d["grasp_mm"], float)
        drop = float((o - g) @ n)
        if drop < 0:
            n, drop = -n, -drop
        side_slip = float(np.linalg.norm((o - g) - drop * n))
        if not 20.0 <= drop <= 150.0 or side_slip > 3.0:
            raise Refused(f"taught grasp point is {drop:.1f} mm below the hover point and {side_slip:.1f} mm to the side of it; expected 20 to 150 mm straight down")
        if d.get("fingers") not in ("col", "row"):
            raise Refused("'fingers' must be 'col' or 'row'")
        if "cells" in d:
            a, b = cell(d["cells"][0]), cell(d["cells"][1])
        else:
            a, b = (min(c0, c1), min(r0, r2)), (max(c0, c1), max(r0, r2))
        area = ((min(a[0], b[0]), min(a[1], b[1])), (max(a[0], b[0]), max(a[1], b[1])))
        return cls(str(d.get("side", "right")), np.array(d["rpy_deg"], float), (c0, r0), o, ec, er, n, drop, d["fingers"], area)

    def _inside(self, name):
        c, r = cell(name)
        (ca, ra), (cb, rb) = self.area
        if not (ca <= c <= cb and ra <= r <= rb):
            raise Refused(f"cell {name} is outside the taught area")
        return c, r

    def hover(self, name):
        c, r = self._inside(name)
        return self.origin + (c - self.c0[0]) * self.ec + (r - self.c0[1]) * self.er

    def grasp(self, name):
        return self.hover(name) - self.grasp_drop * self.n

    def f(self):
        """Unit vector along the fingers: the direction in which the jaws leave the brick free."""
        v = self.ec if self.fingers == "col" else self.er
        return v / np.linalg.norm(v)

    def c(self):
        """Unit vector along the closing direction."""
        v = self.er if self.fingers == "col" else self.ec
        return v / np.linalg.norm(v)

    def limits(self, margin_mm=45.0, above_mm=40.0, below_mm=2.0, air_mm=0.0, **kw):
        """The volume the arm may be in: the taught area plus a margin, from `below_mm` under the grasp height to
        `above_mm` over the hover height. For a run in the air everything is lifted by `air_mm`, except that the floor
        never rises above the hover height itself, which is where the arm starts from and is clear of the bricks."""
        u = self.ec / np.linalg.norm(self.ec)
        w = self.n
        v = np.cross(w, u)
        if v @ self.er < 0:
            v = -v
        (ca, ra), (cb, rb) = self.area
        pu, pv = float(np.linalg.norm(self.ec)), float(abs(self.er @ v))
        lo = np.array([(ca - self.c0[0]) * pu - margin_mm, (ra - self.c0[1]) * pv - margin_mm, min(0.0, -(self.grasp_drop + below_mm) + air_mm)])
        hi = np.array([(cb - self.c0[0]) * pu + margin_mm, (rb - self.c0[1]) * pv + margin_mm, above_mm + air_mm])
        return Limits(self.origin.copy(), np.stack([u, v, w], 1), lo, hi, **kw)


def demo(side="right"):
    """A made-up taught file (a plate lying flat in front of the base) for dry runs away from the robot."""
    o = np.array([450.0, -40.0, 180.0])
    return dict(side=side, rpy_deg=[180.0, 0.0, 0.0], fingers="col", cells=["B2", "G7"], grasp_mm=(o - [0, 0, 62.0]).tolist(),
                taught=dict(origin=dict(cell="C3", hover_mm=o.tolist()), col=dict(cell="F3", hover_mm=(o + [0, 3 * 31.75, 0]).tolist()),
                            row=dict(cell="C6", hover_mm=(o + [3 * 31.75, 0, 0]).tolist())),
                note="made up, NOT taught on the robot: for `plan` only")
