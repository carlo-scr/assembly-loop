"""Designs: what to build, block by block, and whether it can be built and stays standing. No arm and no camera here.

A design is a list of blocks in the order the arm puts them down. Each block is
  kind    a kind of the catalogue (blocks.json), e.g. "red plank";
  rest    how it lies: "flat", "on its side", "on end" (boxes) or "upright" (cylinders);
  x, y    where the middle of its footprint is [mm], in the design's own frame: x forward (away from the robot's
          base), y to the robot's left, the origin on the spot the design is built on;
  angle   which way its long side points [deg], from forward towards left;
  z       how high its bottom is [mm] (0 on the table). The designer drops every block straight down until it meets
          something, which is also how the arm puts it down.

`check(design, cat)` looks at every step of the build, i.e. at what stands after each block is put down:
  - every block rests on the table or on blocks put down BEFORE it, overlaps nothing, and nothing stands over it
    (the arm brings it straight down from above);
  - the gripper fits: the jaws close across the block's short side; the fingers need room on both sides when they
    let go (they open only `release_gap_mm` wider than the block), and the body of the gripper needs room above them;
  - what stands so far stays standing: forces that only push (no glue), through faces that touch, from the table up,
    must carry every block's weight (`stands`). How much it can take is the largest tilt of the table it survives
    without anything tipping over (`tilt_deg`, a stand-in for a bump: more is safer).
The statics are a small linear programme per step (`_simplex`, numpy only). Blocks are wood of one density, so a
block's weight is its volume. Blocks that touch side by side do not lean on each other here (on the safe side).

The gripper's sizes come from UFACTORY's meshes of the xArm gripper (`web/xarm7`), not from measuring the rig.
"""
from __future__ import annotations

import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .blocks import Catalogue
from .guard import Refused

GRIPPER = dict(
    finger_width_mm=32.0,  # a finger, along the block's long side (the meshes: 32 mm)
    finger_thick_mm=14.0,  # a finger, across (the meshes: 12 to 15 mm)
    finger_len_mm=58.0,  # from the fingertips up to where the gripper's body begins (the meshes: 59 mm)
    release_gap_mm=10.0,  # when a design's block is let go, each finger opens this far off the block, no further
    body_along_mm=80.0,  # the body above the fingers (the meshes: 75 x 102 mm, the knuckles wider when fully open)
    body_across_mm=120.0,
)
TOUCH_MM = 0.5  # faces this close count as touching
CLEAR_MM = 30.0  # blocks lying on the mat must be this far from where a design goes: the fingers reach 24 mm beside a block
MIN_CONTACT_MM2 = 20.0  # less touching area than this carries nothing
MAX_TILT_DEG = 20.0  # the tilt is looked for up to here: more counts as solid
SHAKY_DEG = 4.0  # below this a design is "shaky": a small bump may topple it
REACH_MM = 150.0  # how far from its spot a design may reach
HEIGHT_MM = 220.0  # how high a design may be (the guard allows the travel height up to 300 mm above the spot; lift 80)
DIRECTIONS = 8  # directions in which the tilt is tried
EXAMPLES = Path(__file__).parent / "designs"
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,39}")


# ---------------------------------------------------------------------------------------------------- the design
@dataclass
class Block:
    kind: str
    rest: str
    x: float
    y: float
    angle: float = 0.0
    z: float = 0.0
    id: str = ""

    def to_dict(self):
        return dict(id=self.id, kind=self.kind, rest=self.rest, x=round(self.x, 2), y=round(self.y, 2), angle=round(self.angle, 2), z=round(self.z, 2))


@dataclass
class Design:
    name: str
    blocks: list
    note: str = ""

    def to_dict(self):
        return dict(name=self.name, note=self.note, blocks=[b.to_dict() for b in self.blocks])

    @classmethod
    def from_dict(cls, d):
        try:
            blocks = []
            for i, b in enumerate(d.get("blocks") or []):
                blocks.append(Block(str(b["kind"]), str(b["rest"]), float(b["x"]), float(b["y"]), float(b.get("angle", 0.0)), float(b.get("z", 0.0)), str(b.get("id") or f"b{i + 1}")))
            if len(blocks) > 40:
                raise Refused("a design has at most 40 blocks")
            if len({b.id for b in blocks}) != len(blocks):
                raise Refused("two blocks of the design have the same id")
            if not all(np.isfinite([b.x, b.y, b.angle, b.z]).all() for b in blocks):
                raise Refused("a block of the design has no proper position")
            return cls(str(d.get("name") or "untitled"), blocks, str(d.get("note") or ""))
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise Refused(f"the design is not readable: {e!r}") from None


# ---------------------------------------------------------------------------------------------------- footprints
def rect(cx, cy, long, short, angle_deg):
    """A rectangle's corners, counter-clockwise."""
    a = np.radians(angle_deg)
    u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    c = np.array([cx, cy], float)
    return np.array([c + sx * long / 2.0 * u + sy * short / 2.0 * v for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def circle(cx, cy, d, n=16):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([cx + d / 2.0 * np.cos(t), cy + d / 2.0 * np.sin(t)], 1)


def area(p):
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clip(subject, clipper):
    """The part of convex polygon `subject` inside convex polygon `clipper` (both counter-clockwise)."""
    out = [np.asarray(p, float) for p in subject]
    n = len(clipper)
    for k in range(n):
        a, b = clipper[k], clipper[(k + 1) % n]
        e = b - a
        inside = lambda p, a=a, e=e: e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0]) >= -1e-9  # noqa: E731
        pts, out = out, []
        for i in range(len(pts)):
            p, q = pts[i], pts[(i + 1) % len(pts)]
            if inside(q):
                if not inside(p):
                    out.append(_cross_at(p, q, a, e))
                out.append(q)
            elif inside(p):
                out.append(_cross_at(p, q, a, e))
        if not out:
            return np.zeros((0, 2))
    return np.array(out)


def _cross_at(p, q, a, e):
    d = q - p
    den = e[0] * d[1] - e[1] * d[0]
    t = 0.0 if abs(den) < 1e-12 else (e[0] * (a[1] - p[1]) - e[1] * (a[0] - p[0])) / den  # where p + t d meets the line through a along e
    return p + np.clip(t, 0.0, 1.0) * d


def overlap(p, q):
    return area(clip(p, q)) if len(p) and len(q) else 0.0


def gap(p, q):
    """How far apart two convex polygons are [mm] (0 if they touch or overlap)."""
    if overlap(p, q) > 0.0:
        return 0.0

    def seg(c, a, b):
        e = b - a
        t = np.clip(np.dot(c - a, e) / max(np.dot(e, e), 1e-12), 0.0, 1.0)
        return float(np.linalg.norm(c - (a + t * e)))

    return min(seg(c, P[k], P[(k + 1) % len(P)]) for P, Q in ((p, q), (q, p)) for c in Q for k in range(len(P)))


# ---------------------------------------------------------------------------------------------------- one block, resolved
@dataclass
class Part:
    i: int
    block: Block
    kind: object
    rest: object
    poly: np.ndarray
    z: float
    top: float
    mass: float
    com: np.ndarray
    hold: float  # how far above its bottom the fingertips close on it

    @property
    def id(self):
        return self.block.id


def resolve(design, cat):
    """Every block with its kind, rest, footprint and weight. Unknown kinds or rests are Refused."""
    parts = []
    for i, b in enumerate(design.blocks):
        kind = cat.kind(b.kind)
        rests = {r.name: r for r in kind.rests()}
        if b.rest not in rests:
            raise Refused(f"a {kind.name} cannot lie '{b.rest}': it can lie {', '.join(rests)}")
        r = rests[b.rest]
        poly = circle(b.x, b.y, r.long) if r.round else rect(b.x, b.y, r.long, r.short, b.angle)
        vol = (np.pi * (r.long / 2.0) ** 2 * r.height if r.round else r.long * r.short * r.height) if not r.rolls else np.pi * (r.short / 2.0) ** 2 * r.long
        parts.append(Part(i, b, kind, r, poly, float(b.z), float(b.z) + r.height, vol / 1e4, np.array([b.x, b.y, b.z + r.height / 2.0]), max(0.0, r.height - float(cat.finger_mm))))
    return parts


def settle(design, cat):
    """Drop every block straight down until it meets the table or a block under it (lowest first): what the
    designer does after every change, so that nothing floats and nothing overlaps vertically."""
    parts = resolve(design, cat)
    done = []
    for p in sorted(parts, key=lambda p: (p.z, p.i)):
        z = round(max([0.0] + [q.top for q in done if overlap(p.poly, q.poly) > 1.0]), 2)  # as the designer rounds it
        p.block.z, p.top, p.z = z, z + p.rest.height, z
        done.append(p)
    return design


# ---------------------------------------------------------------------------------------------------- the gripper
def grip_options(p):
    """The ways the gripper can be turned to hold block p [deg, the direction of the fingers' width]: along its long
    side; a square footprint also turned by 90 degrees, a round one in steps of 45."""
    a = p.block.angle
    if p.rest.round:
        return [a, a + 90.0, a + 45.0, a + 135.0]
    return [a, a + 90.0] if abs(p.rest.long - p.rest.short) < 1.0 else [a]


def zones(p, release_gap=None, grip_deg=None):
    """Where the gripper is when it lets go of block p: two fingers (across the short side, opened `release_gap`
    beyond the block) and the body above them. Each zone: (footprint, from height, to height)."""
    g = GRIPPER
    gap = g["release_gap_mm"] if release_gap is None else release_gap
    grip = p.block.angle if grip_deg is None else grip_deg
    a = np.radians(grip)
    u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    c, s = np.array([p.block.x, p.block.y]), p.rest.short / 2.0
    tip = p.z + p.hold
    w, t = g["finger_width_mm"] / 2.0, gap + g["finger_thick_mm"]
    out = []
    for side in (-1.0, 1.0):
        lo, hi = s, s + t
        pts = [c + au * w * u + side * av * v for au, av in ((-1, lo), (1, lo), (1, hi), (-1, hi))]
        if side < 0:
            pts = pts[::-1]
        out.append(dict(what="finger", poly=np.array(pts), z0=tip, z1=tip + g["finger_len_mm"]))
    out.append(dict(what="body", poly=rect(c[0], c[1], g["body_along_mm"], g["body_across_mm"], grip), z0=tip + g["finger_len_mm"], z1=tip + g["finger_len_mm"] + 250.0))
    return out


def grip_for(p, before):
    """(how the gripper is turned, its zones, what each zone runs into): the first way of holding p that runs into
    nothing put down before it, or else the first way."""
    tried = []
    for deg in grip_options(p):
        zs = zones(p, grip_deg=deg)
        hits = [_hits(z, before) for z in zs]
        if not any(hits):
            return deg, zs, hits
        tried.append((deg, zs, hits))
    return tried[0]


# ---------------------------------------------------------------------------------------------------- statics
def _simplex(A, b, c, tol=1e-9, max_iter=20000):
    """min c.x subject to A x = b, x >= 0 (two phases, dense tableau; Dantzig's rule, Bland's while degenerate).
    Returns x, or None if there is no such x."""
    A, b, c = np.array(A, float), np.array(b, float), np.asarray(c, float)
    m, n = A.shape
    flip = np.where(b < 0, -1.0, 1.0)
    A *= flip[:, None]
    b *= flip
    T = np.zeros((m + 1, n + m + 1))
    T[:m, :n], T[:m, n:n + m], T[:m, -1] = A, np.eye(m), b
    T[m, :n], T[m, -1] = -A.sum(0), -b.sum()
    basis = np.arange(n, n + m)

    def pivot(i, j):
        T[i] /= T[i, j]
        f = T[:, j].copy()
        f[i] = 0.0
        T[:] -= np.outer(f, T[i])
        basis[i] = j

    def run(ncols):
        stall = 0
        for _ in range(max_iter):
            r = T[m, :ncols]
            if stall > 25:
                cand = np.nonzero(r < -tol)[0]
                if not len(cand):
                    return True
                j = int(cand[0])
            else:
                j = int(np.argmin(r))
                if r[j] >= -tol:
                    return True
            col, rhs = T[:m, j], T[:m, -1]
            pos = col > tol
            if not pos.any():
                return False
            ratio = np.full(m, np.inf)
            ratio[pos] = rhs[pos] / col[pos]
            best = ratio.min()
            ties = np.nonzero(ratio <= best + tol)[0]
            i = int(ties[np.argmin(basis[ties])])
            stall = stall + 1 if best <= tol else 0
            pivot(i, j)
        raise RuntimeError("the statics did not settle (simplex iteration limit)")

    run(n + m)
    if -T[m, -1] > 1e-7 * max(1.0, float(b.sum())):
        return None
    for i in range(m):
        if basis[i] >= n:
            js = np.nonzero(np.abs(T[i, :n]) > 1e-9)[0]
            if len(js):
                pivot(i, int(js[0]))
    T[m, :] = 0.0
    T[m, :n] = c
    for i in range(m):
        if basis[i] < n and c[basis[i]] != 0.0:
            T[m] -= c[basis[i]] * T[i]
    if not run(n):
        raise RuntimeError("the statics are unbounded")
    x = np.zeros(n)
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = T[i, -1]
    return x


def contacts(parts):
    """Where blocks rest on each other or on the table: (index of the block below or -1 for the table, index of the
    block above, the touching area as a polygon, its height)."""
    out = []
    for k, up in enumerate(parts):
        if up.z <= TOUCH_MM:
            out.append((-1, k, up.poly, 0.0))
        for j, lo in enumerate(parts):
            if j != k and abs(lo.top - up.z) <= TOUCH_MM and up.z > TOUCH_MM:
                P = clip(up.poly, lo.poly)
                if area(P) >= MIN_CONTACT_MM2:
                    out.append((j, k, P, up.z))
    return out


def _forces(parts):
    """The balance of every block as rows of A x = b: forces along x, y, z and turning about x and y (about each
    block's centre of mass). Per contact: one pushing force at every corner of the touching area, and a sideways
    (friction) force at its middle, either way (friction is not limited: tipping is what is looked for)."""
    n, cons = len(parts), contacts(parts)
    cols, Tm = [], 50.0  # turning rows are divided by Tm so that all rows are of one size
    for lo, up, P, zc in cons:
        for v in P:
            col = np.zeros(5 * n)
            for k, sgn in ((up, 1.0), (lo, -1.0)):
                if k < 0:
                    continue
                g = parts[k].com
                col[5 * k + 2] += sgn
                col[5 * k + 3] += sgn * (v[1] - g[1]) / Tm
                col[5 * k + 4] += -sgn * (v[0] - g[0]) / Tm
            cols.append(col)
        for axis in (0, 1):
            for sense in (1.0, -1.0):
                col = np.zeros(5 * n)
                for k, sgn in ((up, 1.0), (lo, -1.0)):
                    if k < 0:
                        continue
                    g, f = parts[k].com, sgn * sense
                    col[5 * k + axis] += f
                    if axis == 0:
                        col[5 * k + 4] += (zc - g[2]) * f / Tm
                    else:
                        col[5 * k + 3] += -(zc - g[2]) * f / Tm
                cols.append(col)
    A = np.array(cols).T if cols else np.zeros((5 * n, 0))
    b = np.zeros(5 * n)
    for k, p in enumerate(parts):
        b[5 * k + 2] = p.mass
    return A, b


_CACHE: OrderedDict = OrderedDict()


def statics(parts):
    """(stands, tilt_deg, toward_deg): does it stand, and how far can the table tilt (in the worst of DIRECTIONS
    directions, at most MAX_TILT_DEG) before anything tips; toward which direction it tips first."""
    key = tuple((p.kind.name, p.rest.name, round(p.block.x, 2), round(p.block.y, 2), round(p.block.angle, 2), round(p.z, 2)) for p in parts)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    A, b = _forces(parts)
    if _simplex(A, b, np.zeros(A.shape[1])) is None:
        out = (False, 0.0, None)
    else:
        amax, worst, toward = np.tan(np.radians(MAX_TILT_DEG)), np.inf, None
        n = len(parts)
        for k in range(DIRECTIONS):
            t = 2.0 * np.pi * k / DIRECTIONS
            g = np.zeros(5 * n)
            for i, p in enumerate(parts):
                g[5 * i], g[5 * i + 1] = p.mass * np.cos(t), p.mass * np.sin(t)
            Ad = np.zeros((5 * n + 1, A.shape[1] + 2))
            Ad[:5 * n, :A.shape[1]], Ad[:5 * n, -2], Ad[5 * n, -2:] = A, g, 1.0
            c = np.zeros(A.shape[1] + 2)
            c[-2] = -1.0
            x = _simplex(Ad, np.r_[b, amax], c)
            a = 0.0 if x is None else float(x[-2])
            if a < worst - 1e-9:
                worst, toward = a, float(np.degrees(t))
        out = (True, float(np.degrees(np.arctan(worst))), toward)
    _CACHE[key] = out
    if len(_CACHE) > 2000:
        _CACHE.popitem(last=False)
    return out


def held_by(p, parts):
    """How far [mm] the middle of block p is inside the area it rests on (negative: it sticks out beyond it, and
    only something on its other end can keep it up). None if it rests on nothing."""
    pts = [v for lo, up, P, _ in contacts(parts) if parts[up] is p for v in P]
    if not pts:
        return None
    hull = _hull(np.array(pts))
    c = p.com[:2]
    if len(hull) < 3:
        return -float(np.min(np.linalg.norm(hull - c, axis=1)))
    d = []
    for k in range(len(hull)):
        a, e = hull[k], hull[(k + 1) % len(hull)] - hull[k]
        d.append((e[0] * (c[1] - a[1]) - e[1] * (c[0] - a[0])) / max(np.linalg.norm(e), 1e-9))
    return float(min(d))


def _hull(pts):
    pts = sorted(set(map(tuple, np.round(pts, 6))))
    if len(pts) < 3:
        return np.array(pts)

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1]) - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0]) <= 0:
                out.pop()
            out.append(p)
        return out

    lo, hi = half(pts), half(pts[::-1])
    return np.array(lo[:-1] + hi[:-1])


# ---------------------------------------------------------------------------------------------------- the check
def _hits(z, parts_before):
    """Earlier blocks that a gripper zone runs into."""
    return [q for q in parts_before if overlap(z["poly"], q.poly) > 1.0 and q.top > z["z0"] + 1.0 and q.z < z["z1"] - 1.0]


def check(design, cat, max_mm=HEIGHT_MM, reach_mm=REACH_MM, statics_on=True):
    """Everything that decides whether the arm can build `design`, step by step. See the module's docstring."""
    t0 = time.monotonic()
    parts = resolve(design, cat)
    label = {p.id: f"step {p.i + 1} ({p.kind.name})" for p in parts}
    blocks, steps, problems, structural = [], [], [], False
    for k, p in enumerate(parts):
        bad, warn = [], []
        if p.rest.rolls:
            bad.append("lying like this it rolls: stand it upright")
        if p.rest.short > cat.jaw_mm - 10.0:
            bad.append(f"{p.rest.short:.0f} mm across: too wide for the jaws")
        for q in parts:
            if q is not p and overlap(p.poly, q.poly) > 1.0 and min(p.top, q.top) - max(p.z, q.z) > TOUCH_MM:
                bad.append(f"it overlaps {label[q.id]}")
                structural = True
        under = [lo for lo, up, P, _ in contacts(parts) if up == k]
        if p.z > TOUCH_MM and not under:
            edge = [q for q in parts if q is not p and abs(q.top - p.z) <= TOUCH_MM and overlap(p.poly, q.poly) > 1.0]
            bad.append(f"it touches {', '.join(label[q.id] for q in edge)} only at the edge: it would tip off" if edge else "it floats: nothing under it")
            structural = True
        later = [parts[j] for j in under if j > k]
        if later:
            bad.append(f"it rests on {', '.join(label[q.id] for q in later)}, which comes later: change the order")
        before = parts[:k]
        over = [q for q in before if overlap(p.poly, q.poly) > 1.0 and q.z >= p.top - TOUCH_MM]
        if over:
            bad.append(f"{', '.join(label[q.id] for q in over)} already stands over it: the arm cannot bring it down from above")
        zs = []
        grip, gz, ghits = grip_for(p, before)
        for z, hit in zip(gz, ghits):
            zs.append(dict(what=z["what"], poly=np.round(z["poly"], 2).tolist(), z0=round(z["z0"], 2), z1=round(z["z1"], 2), hit=[q.id for q in hit]))
            if hit:
                bad.append(f"the {'fingers' if z['what'] == 'finger' else 'gripper'} would hit {', '.join(label[q.id] for q in hit)} when it lets go")
        if max(abs(p.block.x), abs(p.block.y)) > reach_mm:
            bad.append(f"more than {reach_mm:.0f} mm from the spot")
        if p.top > max_mm:
            bad.append(f"its top is {p.top:.0f} mm high: the arm builds up to {max_mm:.0f} mm")
        margin = held_by(p, parts[:k + 1])
        if margin is not None and margin < 0:
            warn.append(f"its middle is {-margin:.0f} mm beyond what it rests on")
        blocks.append(dict(id=p.id, z=round(p.z, 2), top=round(p.top, 2), problems=sorted(set(bad)), warnings=warn, held_by_mm=None if margin is None else round(margin, 1),
                           zones=zs, grip_deg=round(grip % 360.0, 1), rests_on=[parts[j].id for j in under if j >= 0], on_table=-1 in under,
                           release_mm=round(p.rest.short + 2 * GRIPPER["release_gap_mm"], 1)))
    for k, p in enumerate(parts):
        if statics_on and not structural:
            stands, tilt, toward = statics(parts[:k + 1])
        else:
            stands, tilt, toward = None, None, None
        steps.append(dict(id=p.id, stands=stands, tilt_deg=None if tilt is None else round(tilt, 1), toward_deg=toward))
        if stands is False:
            blocks[k]["problems"].append("once it is put down, what stands falls over")
    if structural:
        problems.append("Whether it stands is worked out once nothing overlaps and nothing hangs in the air.")
    tilts = [s["tilt_deg"] for s in steps if s["stands"]]
    weakest = min(range(len(steps)), key=lambda k: steps[k]["tilt_deg"] if steps[k]["stands"] else 1e9) if tilts else None
    stands = None if any(s["stands"] is None for s in steps) else bool(steps) and all(s["stands"] for s in steps)
    if stands and min(tilts) < SHAKY_DEG:
        problems.append(f"shaky: at step {weakest + 1} a tilt of {steps[weakest]['tilt_deg']:.1f} degrees tips it; a small bump may topple it")
    need = {}
    for p in parts:
        need[(p.kind.name, p.rest.name)] = need.get((p.kind.name, p.rest.name), 0) + 1
    ok = bool(parts) and not any(b["problems"] for b in blocks) and stands
    return dict(ok=ok, stands=stands if parts else None, tilt_deg=min(tilts) if tilts and stands else None, weakest=weakest, steps=steps, blocks=blocks, problems=problems,
                height_mm=round(max([p.top for p in parts] + [0.0]), 1), needs=[dict(kind=k, rest=r, n=n) for (k, r), n in sorted(need.items())], ms=round(1000 * (time.monotonic() - t0)))


def find_order(design, cat, budget=400):
    """An order in which the arm can build the same blocks: every block on what is already there, nothing over it,
    the gripper clear, and every step standing. Keeps the given order where it can. Returns the reordered design,
    or Refused with the reason if none is found within `budget` checks of the statics."""
    parts = resolve(design, cat)
    n, cons = len(parts), contacts(parts)
    under = {k: {lo for lo, up, _, _ in cons if up == k and lo >= 0} for k in range(n)}
    seen, calls = set(), [0]

    def ok(k, placed):
        if not under[k] <= placed:
            return False
        p, before = parts[k], [parts[j] for j in placed]
        if any(overlap(p.poly, q.poly) > 1.0 and q.z >= p.top - TOUCH_MM for q in before):
            return False
        if any(grip_for(p, before)[2]):
            return False
        calls[0] += 1
        if calls[0] > budget:
            raise Refused("no order found in time: build it in the order shown, or make it simpler")
        return statics([parts[j] for j in path + [k]])[0]

    path = []

    def go(placed):
        if len(placed) == n:
            return True
        if frozenset(placed) in seen:
            return False
        seen.add(frozenset(placed))
        for k in range(n):
            if k not in placed and ok(k, placed):
                path.append(k)
                if go(placed | {k}):
                    return True
                path.pop()
        return False

    if not go(set()):
        raise Refused("there is no order in which every step stands and the gripper is clear")
    return Design(design.name, [parts[k].block for k in path], design.note)


# ---------------------------------------------------------------------------------------------------- files
def check_name(name):
    name = str(name or "").strip()
    if not NAME.fullmatch(name):
        raise Refused("a design's name is 1 to 40 letters, digits, spaces, - or _, starting with a letter or digit")
    return name


def file_of(folder, name):
    return Path(folder) / (re.sub(r"[^A-Za-z0-9_-]+", "_", check_name(name)).strip("_").lower() + ".json")


def listing(folder):
    """Saved designs and the examples that come with the program (a saved one of the same name hides the example)."""
    out = {}
    for where, example in ((EXAMPLES, True), (Path(folder), False)):
        for f in sorted(where.glob("*.json")) if where.is_dir() else []:
            try:
                d = Design.from_dict(json.loads(f.read_text()))
            except (OSError, ValueError, Refused):
                continue
            out[d.name.lower()] = dict(name=d.name, n=len(d.blocks), example=example, saved=time.strftime("%d %b %H:%M", time.localtime(f.stat().st_mtime)))
    return sorted(out.values(), key=lambda d: (d["example"], d["name"].lower()))


def load(folder, name):
    name = check_name(name)
    for f in (file_of(folder, name), file_of(EXAMPLES, name)):
        if f.exists():
            return Design.from_dict(json.loads(f.read_text()))
    raise Refused(f"there is no design called {name!r}")


def save(folder, design):
    f = file_of(folder, design.name)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(design.to_dict(), indent=1))
    return f


def catalogue_for_design(blocks_path):
    """The kinds of block to design with: the rig's blocks.json, or the example that comes with the program."""
    for p in (Path(blocks_path), Path(__file__).parent.parent / "blocks.example.json"):
        if p.exists():
            return Catalogue.load(p), p
    raise Refused("there is no blocks.json: measure the blocks and write them down (see blocks.example.json)")
