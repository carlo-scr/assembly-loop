"""Looking again after every move: a tower that is built from what the camera sees NOW, not from one picture taken
before the first block was touched. Mixed into `Session`, next to `Sight`.

`sight.build_seen` plans every block from one look and then never looks again: a block that slips, a tower that
falls or a hand that tidies the mat goes unnoticed until the jaws close on nothing. Here the arm steps aside after
every block, the camera looks, and three questions are answered from that picture before anything else moves:

  1. Is the block that was just put on the tower where it should be? Its outline is drawn at the height it should
     stand at and fitted to its colour in the picture (`on_tower`). Found: the next block is aimed half way between
     where it really stands and the tower's own axis, so that an error is not carried up the tower. Not found: it
     fell; it is taken off the list, and the level below is checked the same way in the next picture.
  2. Which block goes next? `blocks.plan_tower` over the blocks that lie on the table in THIS picture, so that a
     block that fell or was moved is simply taken again from where it lies now.
  3. Can the jaws get at it? The two fingers come down beside a block, across its short side (`jaws_room`). A block
     that has another one where a finger would be is first pushed clear with the closed gripper (`plan_push`), and
     a patch of colour that is not made out as a block (two that touch) is pushed apart the same way. Then: look
     again.

What the jaws close on is checked against what the camera said: the gripper reports how far apart its jaws stopped,
and a block of another width is let go again (`cycle.WrongWidth`).

`scatter` takes a tower down again and lays its blocks at places and angles picked at random, and `collect` does
both in turn, again and again: every round starts from another arrangement, which is what a policy needs to learn
from. Every build and every scatter is one recorded episode with its task in words.

Everything still goes through the guard, only while the hold-to-run switch is held, and nothing here widens where
the arm may go. NONE OF THIS HAS RUN ON THE REAL ARM OR WITH THE REAL CAMERA: it is tested on the stand-in arm
and drawn pictures only. The pushes in particular are a guess at what separates wooden blocks on this mat.
"""
from __future__ import annotations

import copy
import json
import time

import numpy as np

from .arms import rpy_matrix, twist_deg
from .blocks import plan_tower
from .cycle import NothingHeld, WrongWidth, execute
from .guard import Refused
from .sight import spin
from .spots import plan_move

FINGER_MM = (24.0, 12.0)  # a fingertip of UFACTORY's gripper seen from above: along the jaws' faces, and across them (not measured on this rig)
PUSHER_MM = 14.0  # the closed fingertips, as a disc
SPARE_MM = 10.0  # the jaws open this much wider than the block, on either side, before they come down around it


def opening(b, jaw_mm):
    """How far the jaws open for this block [mm]: no wider than needed, so that the fingers need little room."""
    return float(min(jaw_mm, b.grip_mm + 2.0 * SPARE_MM))


# ---------------------------------------------------------------------------------------------------- flat geometry on the table
def rect(uv, angle_deg, long, short):
    """The four corners of a rectangle on the table: its middle, the direction of its long side, its two sides."""
    a = np.radians(angle_deg)
    along, across = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    c = np.asarray(uv, float)
    return np.array([c + i * long / 2.0 * along + j * short / 2.0 * across for i, j in ((1, 1), (-1, 1), (-1, -1), (1, -1))])


def rects_touch(A, B):
    """Do two convex outlines overlap? (No line along a side of either keeps them apart.)"""
    for P in (A, B):
        for i in range(len(P)):
            n = np.array([-(P[i - 1][1] - P[i][1]), P[i - 1][0] - P[i][0]])
            a, b = A @ n, B @ n
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True


class Heap:
    """Something on the table that the camera did not make out (two blocks that touch, a block half hidden behind
    another one): not a block that can be taken, but in the way all the same. `uvs`: points of it on the table."""

    def __init__(self, uvs, colour=""):
        self.uvs, self.colour, self.name = np.asarray(uvs, float), colour, "something that was not made out"
        c = self.uvs.mean(0)
        _, _, vt = np.linalg.svd(self.uvs - c, full_matrices=False)
        along, across = vt[0], np.array([-vt[0][1], vt[0][0]])
        a, b = (self.uvs - c) @ along, (self.uvs - c) @ across
        self.uv = tuple(c + along * (a.max() + a.min()) / 2.0 + across * (b.max() + b.min()) / 2.0)
        self.angle, self.long, self.short = float(np.degrees(np.arctan2(along[1], along[0]))), float(np.ptp(a)), float(np.ptp(b))


def footprint(b, grow=0.0):
    """Where a seen block covers the table (a standing cylinder: the square around its circle), or a heap."""
    if isinstance(b, Heap):
        return rect(b.uv, b.angle, b.long + 2.0 * grow, b.short + 2.0 * grow)
    return rect(b.uv, b.angle, b.rest.long + 2.0 * grow, b.rest.short + 2.0 * grow)


def grasp_angles(b):
    """The directions the gripper may take a block from, as the direction its jaws' faces run along: a bar only
    across its short side, a square either way, something round from four sides."""
    if b.rest.round:
        return [0.0, 45.0, 90.0, 135.0]
    return [b.angle] if b.rest.turns == 180.0 else [b.angle, b.angle + 90.0]


def fingers(b, angle_deg, jaw_mm):
    """Where the two open fingers come down beside a block taken from `angle_deg`: two small rectangles."""
    a = np.radians(angle_deg + 90.0)
    across = np.array([np.cos(a), np.sin(a)])
    out = float(jaw_mm) / 2.0 + FINGER_MM[1] / 2.0
    return [rect(np.asarray(b.uv, float) + s * out * across, angle_deg, FINGER_MM[0], FINGER_MM[1]) for s in (1.0, -1.0)]


def in_the_way(b, angle_deg, others, jaw_mm, clear_mm=3.0):
    """The blocks a finger would come down on if `b` were taken from `angle_deg`."""
    fs = fingers(b, angle_deg, opening(b, jaw_mm))
    return [o for o in others if o is not b and any(rects_touch(f, footprint(o, clear_mm)) for f in fs)]


def jaws_room(b, others, jaw_mm):
    """From which direction the jaws have room beside this block, or None if from none."""
    return next((a for a in grasp_angles(b) if not in_the_way(b, a, others, jaw_mm)), None)


def push_clear(b, others, jaw_mm, inside, slide_mm=60.0):
    """A push that slides a crowded block out sideways: (start, end) on the table for the closed fingertips, or
    None. It goes across the line to the block that is in the way, from the side on which the fingertips have room
    to come down, and slides the block `slide_mm` to a place where it will be free."""
    blockers = in_the_way(b, grasp_angles(b)[0], others, jaw_mm) or [o for o in others if o is not b]
    if not blockers:
        return None
    c = np.asarray(b.uv, float)
    to = np.asarray(min(blockers, key=lambda o: np.hypot(*(np.asarray(o.uv) - c))).uv, float) - c
    if np.linalg.norm(to) < 1e-6:
        return None
    to = to / np.linalg.norm(to)
    for d in (np.array([-to[1], to[0]]), np.array([to[1], -to[0]])):
        half = float(np.max(np.abs((footprint(b) - c) @ d)))
        touch = c - d * (half + PUSHER_MM)  # where the fingertips meet the block
        start, end = touch - d * 8.0, touch + d * slide_mm
        disc = rect(start, 0.0, 2 * PUSHER_MM, 2 * PUSHER_MM)
        there = rect(c + d * slide_mm, b.angle, b.rest.long, b.rest.short)  # where the block ends up
        if inside(start) and inside(there.mean(0)) and not any(rects_touch(disc, footprint(o, 4.0)) for o in others) and \
                not any(rects_touch(there, footprint(o, 4.0)) for o in others if o is not b):
            return start, end
    return None


def push_apart(uvs, attempt, inside, travel_mm=20.0):
    """A push through a heap that was not made out (`uvs`: points of it on the table). Nobody knows how the blocks
    lie in it, so successive attempts come from different sides and off its middle: across its long direction a
    quarter of the way from one end, then from the other, then along it beside its middle, on either side. Each
    shears a part of the heap off. Returns (start, end) or None."""
    uvs = np.asarray(uvs, float)
    c = uvs.mean(0)
    _, _, vt = np.linalg.svd(uvs - c, full_matrices=False)
    along, across = vt[0], np.array([-vt[0][1], vt[0][0]])
    L, W = float(np.ptp((uvs - c) @ along)), float(np.ptp((uvs - c) @ across))
    for k in range(4):
        way = (attempt + k) % 4
        d, off, half = ((across, along * L / 4.0, W / 2.0), (across, -along * L / 4.0, W / 2.0), (along, across * W / 4.0, L / 2.0), (along, -across * W / 4.0, L / 2.0))[way]
        for sign in (1.0, -1.0):
            start, end = c + off - sign * d * (half + PUSHER_MM + 20.0), c + off + sign * d * travel_mm
            if inside(start) and inside(end):
                return start, end
    return None


def plan_push(spots, a, b, here, P, rpy):
    """Steps to push along the table with the closed gripper, from `a` to `b` (where the flange is at each end, in
    the arm's base frame), the gripper turned as in `rpy`. The way down is watched like every lowering; the push is
    watched too, and one that meets something that does not slide ends there and counts as done."""
    up, tw = spots.up, spots.travel_w()
    return [dict(name="go to the travel height", kind="move", to=spots.above(here), speed=P.lift),
            dict(name="go above where the push starts", kind="move", to=spots.above(a), speed=P.travel),
            dict(name="turn the gripper across the push", kind="turn", rpy=rpy),
            dict(name="close the gripper", kind="grip", counts=P.shut_counts),
            dict(name="lower the closed gripper to the table", kind="move", to=a, speed=P.slow, stall_mm=P.stall_mm, straight_up=(up, tw)),
            dict(name="push", kind="move", to=b, speed=P.slow, stall_mm=P.push_stall_mm, stop_ok=True),
            dict(name="lift clear", kind="rise", up=up, w=tw, speed=P.lift),
            dict(name="open the gripper", kind="grip", counts=P.open_counts)]


def scatter_places(n, rng, inside, taken, apart_mm=90.0, edge_mm=30.0, box=None, tries=4000):
    """`n` places on the table picked at random: inside the area and `edge_mm` from its edge, and `apart_mm` from
    each other and from everything in `taken`. Returns [(u, v, angle)]; fewer than `n` if the table is too full."""
    lo, hi = box
    fixed, best = [np.asarray(t, float) for t in taken], []
    for _ in range(40):  # a first place picked badly can leave no room for the rest: start over a few times
        out, held = [], list(fixed)
        for _ in range(int(tries) // 40):
            if len(out) >= n:
                break
            p = rng.uniform(lo, hi)
            if all(inside(p + edge_mm * np.array(d)) for d in ((1, 0), (-1, 0), (0, 1), (0, -1))) and all(np.hypot(*(p - t)) >= apart_mm for t in held):
                out.append((float(p[0]), float(p[1]), float(rng.uniform(0.0, 180.0))))
                held.append(p)
        best = out if len(out) > len(best) else best
        if len(best) >= n:
            break
    return best


# ---------------------------------------------------------------------------------------------------- the picture
def tower_mask(eyes, setup, size, tower, grow_mm=8.0):
    """Where the tower shows in a picture of this size, and a little around it: the block finder must not take the
    tower's blocks for blocks that lie on the table behind it."""
    cam, table = setup.camera.scaled(size), setup.table
    m = np.zeros((size[1], size[0]), np.uint8)
    for lv in tower:
        hull = eyes.outline(cam, eyes.solid(lv["rest"], eyes.Table(table.axes, table.w0 + lv["base_mm"]), lv["uv"][0], lv["uv"][1], lv["angle"], lv["kind"].shape))
        eyes.cv2.fillConvexPoly(m, np.round(hull).astype(np.int32), 1)
    if tower:
        u, v = tower[0]["uv"]
        px = cam.project([table.point(u, v, table.w0), table.point(u + grow_mm, v, table.w0), table.point(u, v + grow_mm, table.w0)])
        k = int(max(3, round(max(np.linalg.norm(px[1] - px[0]), np.linalg.norm(px[2] - px[0]))))) * 2 + 1
        m = eyes.cv2.dilate(m, eyes.cv2.getStructuringElement(eyes.cv2.MORPH_ELLIPSE, (k, k)))
    return m.astype(bool)


def on_tower(eyes, im, setup, tower, good=0.5, far_mm=20.0):
    """Is the top block of the tower where it was put? Its outline, drawn at the height it should stand at, is
    shifted and turned until it covers the patch of its colour in the picture. Returns ((u, v), cover) of where it
    really stands, or (None, cover) if no such patch is there. Blocks of the same colour further down are taken out
    of the comparison, so a red block on a red block is still judged by what shows of it alone."""
    lv, size = tower[-1], (im.shape[1], im.shape[0])
    cam, table = setup.camera.scaled(size), setup.table
    T = eyes.Table(table.axes, table.w0 + lv["base_mm"])
    draw = lambda x: eyes.outline(cam, eyes.solid(lv["rest"], T, x[0], x[1], x[2], lv["kind"].shape))  # noqa: E731
    hull = draw((lv["uv"][0], lv["uv"][1], lv["angle"]))
    lo, hi = hull.min(0), hull.max(0)
    pad = 0.6 * float(max(hi - lo)) + 10.0
    x0, y0 = int(max(0, lo[0] - pad)), int(max(0, lo[1] - pad))
    x1, y1 = int(min(size[0], hi[0] + pad)), int(min(size[1], hi[1] + pad))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None, 0.0
    crop, origin = im[y0:y1, x0:x1], np.array([x0, y0], float)
    target = np.zeros(crop.shape[:2], bool)
    for b in eyes.find_blobs(crop):
        if lv["kind"].colour is None or b.colour == lv["kind"].colour:
            target[b.y0:b.y0 + b.mask.shape[0], b.x0:b.x0 + b.mask.shape[1]] |= b.mask
    below = np.zeros(crop.shape[:2], np.uint8)
    for other in tower[:-1]:
        if other["kind"].colour == lv["kind"].colour:
            h = eyes.outline(cam, eyes.solid(other["rest"], eyes.Table(table.axes, table.w0 + other["base_mm"]), other["uv"][0], other["uv"][1], other["angle"], other["kind"].shape))
            eyes.cv2.fillConvexPoly(below, np.round(h - origin).astype(np.int32), 1)
    keep = ~below.astype(bool)
    target &= keep

    def cover(x):
        canvas = np.zeros(target.shape, np.uint8)
        eyes.cv2.fillConvexPoly(canvas, np.round((draw(x) - origin) * 16).astype(np.int32), 1, shift=4)
        drawn = canvas.astype(bool) & keep
        union = int((drawn | target).sum())
        return 0.0 if union == 0 else float((drawn & target).sum()) / union

    x, s = eyes._climb(cover, np.array([lv["uv"][0], lv["uv"][1], lv["angle"]], float), np.array([4.0, 4.0, 0.0 if lv["rest"].round else 4.0]))
    if s < good or float(np.hypot(x[0] - lv["uv"][0], x[1] - lv["uv"][1])) > far_mm:
        return None, s
    return (float(x[0]), float(x[1])), s


# ---------------------------------------------------------------------------------------------------- the session
class Closed:
    def _closed_ready(self, on, aside):
        """What every job here needs. Returns (eyes, catalogue, setup, the spot to step aside to)."""
        if self.phase() != "ARMED":
            raise Refused(f"this can only be started when ARMED (now: {self.phase()})")
        eyes, cat, why = self._eyes(), self.catalogue(), self._setup_fits()
        if why:
            raise Refused(why)
        self.spots.grasp(on)
        if aside is None or aside == "":
            aside = max((k for k in self.spots.at if k != on), key=lambda k: float(np.linalg.norm(self.spots.grasp(k) - self.spots.grasp(on))))
        self.spots.grasp(aside)
        if aside == on:
            raise Refused("the arm cannot step aside to the spot the tower stands on")
        return eyes, cat, self.setup, aside

    def _turn_for(self, setup, angle):
        """How the gripper is turned to take a block whose long side points along `angle` (the shorter way round)."""
        return spin(setup.rpy_ref, self.spots.up, (angle - setup.angle_ref + 90.0) % 180.0 - 90.0)

    def _angle_on(self, setup, on):
        """Which way a bar's long side points once it was put down with the gripper turned as taught at `on`."""
        want = rpy_matrix(self.spots.turn(on))
        tw = twist_deg(rpy_matrix(setup.rpy_ref), want)[0]
        d = min((tw, -tw), key=lambda d: float(np.linalg.norm(rpy_matrix(spin(setup.rpy_ref, self.spots.up, d)) - want)))
        return float((setup.angle_ref + d) % 180.0)

    # ------------------------------------------------------------------ the jobs
    def build_watched(self, on, aside=None, push=True, step=False, rounds=1, scatter=False, seed=None, tower=None):
        """A tower on the taught spot `on` from the blocks on the table, with a look after every block.
        `aside`: the taught spot the arm waits above while the camera looks (the one farthest from `on` unless
        given). `push`: blocks the jaws cannot get at are pushed clear first. `scatter`: afterwards the tower is
        taken down and its blocks laid at places picked at random; with `rounds` the two are done in turn, that many
        times (data collection). `tower`: the levels that already stand there, as an earlier job left them."""
        with self.lock:
            eyes, cat, setup, aside = self._closed_ready(on, aside)
            rounds = int(rounds)
            if not 1 <= rounds <= 200:
                raise Refused("between 1 and 200 rounds")
            rng, levels = np.random.default_rng(seed), list(tower if tower is not None else [])
            self.stack, self.thing_at = dict(on=on, block_mm=None, n=len(levels), of=None, feeds=[]), None

            def body(job, g):
                for r in range(rounds):
                    again = f" (round {r + 1} of {rounds})" if rounds > 1 else ""
                    self._ep_begin("stack the blocks into a tower", kind="build", on=on, round=r + 1, rounds=rounds)
                    out = self._build(job, g, eyes, cat, setup, on, aside, bool(push), levels)
                    self._ep_end(out, success=out == "done" and len(levels) > 0, blocks=[lv["name"] for lv in levels])
                    if out != "done" or not (scatter or r + 1 < rounds):
                        return out if out != "done" else f"done: a tower of {len(levels)}{again}" + (f"; {self.left_out} left on the table" if self.left_out else "")
                    self._ep_begin("take the tower apart and spread the blocks over the table", kind="scatter", on=on, round=r + 1, rounds=rounds)
                    out = self._scatter(job, g, eyes, cat, setup, on, aside, levels, rng)
                    self._ep_end(out, success=out == "done")
                    if out != "done":
                        return out
                return "done"

            self._begin_job(f"a tower on {on}, with a look after every block", body, step=step, tower=True)

    def scatter(self, on, aside=None, step=False, seed=None):
        """Take down the tower the last watched build left on `on`, top first, and lay its blocks at places and
        angles picked at random."""
        with self.lock:
            eyes, cat, setup, aside = self._closed_ready(on, aside)
            levels = self.tower_levels if self.tower_levels is not None else []
            if not levels or (self.stack or {}).get("on") != on:
                raise Refused(f"no tower that this program built stands on {on}")
            rng = np.random.default_rng(seed)

            def body(job, g):
                self._ep_begin("take the tower apart and spread the blocks over the table", kind="scatter", on=on)
                out = self._scatter(job, g, eyes, cat, setup, on, aside, levels, rng)
                self._ep_end(out, success=out == "done")
                return out

            self._begin_job(f"take the tower on {on} down and spread its blocks", body, step=step, tower=True)

    # ------------------------------------------------------------------ pieces of a job (run in the worker's thread)
    def _steps(self, job, g, sp, steps, text):
        """Run a few steps through the guard as part of the job the page shows."""
        here = g.where()
        g.lim = sp.limits(ceil_w=float(here @ sp.up) + 1.0, **self.limit_kw)
        with self.lock:
            job.update(steps=steps, i=None, text=text)
        self._mark("leg", text=text)
        return execute(g, steps, self.P, before=self._before(job, g))

    def _look_aside(self, job, g, eyes, cat, setup, aside, levels, im=None):
        """Step aside (if the arm is not there already), take a picture, and say what lies on the table. The tower
        is hidden from the block finder. Returns (picture, blocks seen inside the area, [(blob, why)] not made out).
        `im`: a picture that was just taken, to be read again with another tower in mind."""
        sp = copy.deepcopy(self.spots)
        sp.extra = max([lv["top_mm"] + lv["hold_mm"] for lv in levels], default=0.0)
        here, there = g.where(), sp.above(sp.grasp(aside))
        if float(np.linalg.norm(here - there)) > 2.0:
            self._steps(job, g, sp, [dict(name="go to the travel height", kind="move", to=sp.above(here), speed=self.P.lift),
                                     dict(name=f"step aside to {aside} for the camera", kind="move", to=there, speed=self.P.travel)], "step aside and look")
        im = self._grab() if im is None else im
        hide = tower_mask(eyes, setup, (im.shape[1], im.shape[0]), levels) if levels else None
        sight = self._see(im, setup, cat, hide=hide)
        self._mark("look", seen=[dict(name=b.name, uv=[round(float(v), 1) for v in b.uv], angle=round(float(b.angle), 1), lies=b.rest.name) for b in sight["seen"]],
                   not_made_out=len(sight["odd"]), tower=[lv["name"] for lv in levels])
        return im, sight["seen"], sight["odd"]

    def _build(self, job, g, eyes, cat, setup, on, aside, push, levels):
        table, P = setup.table, self.P
        base, angle_on = np.array(table.coords(self.spots.grasp(on))[:2]), self._angle_on(setup, on)
        fails, pushes, left, again = 0, 0, [], None
        self.tower_levels = levels
        while True:
            if fails >= 3:
                return "gave up after three failures in a row: look at the table, then start again"
            im, seen, odd = self._look_aside(job, g, eyes, cat, setup, aside, levels, im=again)
            again = None
            if levels:  # 1. is the top of the tower where it was put?
                at, cover = on_tower(eyes, im, setup, levels)
                if at is None:
                    fell = levels.pop()
                    fails += 1
                    self.say(f"the camera does not see the {fell['name']} on the tower (it fits {100 * cover:.0f}%): it is taken for fallen")
                    self._mark("fell", name=fell["name"], cover=round(cover, 2))
                    with self.lock:
                        self.stack = {**self.stack, "n": len(levels)}
                    again = im  # the same picture, read once more without that block on the tower
                    continue
                levels[-1]["seen_uv"] = at
            top = levels[-1]["top_mm"] if levels else 0.0
            plan, left = plan_tower(seen, cat, max_mm=self._tower_max() - top)  # 2. which block goes next?
            if levels:  # nothing wider than what it would stand on
                wide = [lv for lv in plan if lv["block"].rest.short > levels[-1]["rest"].short + 1.0]
                plan, left = [lv for lv in plan if lv not in wide], left + [(lv["block"], "it is wider than the top of the tower") for lv in wide]
            with self.lock:
                job.update(done=len(levels), total=len(levels) + len(plan))
                self.stack = {**self.stack, "n": len(levels), "of": len(levels) + len(plan), "feeds": [lv["block"].name for lv in plan]}
            inside = lambda p: bool(setup.inside(p))  # noqa: E731
            cam, heaps = setup.camera.scaled((im.shape[1], im.shape[0])), []
            for blob, _ in odd:  # what was not made out is in the way all the same
                if not blob.cut:
                    ys, xs = np.nonzero(blob.mask)
                    pick = np.linspace(0, len(xs) - 1, min(len(xs), 300)).astype(int)
                    heaps.append(Heap([table.coords(p)[:2] for p in cam.on_plane(np.c_[xs[pick] + blob.x0, ys[pick] + blob.y0], table.axes[:, 2], table.w0 + 10.0)], blob.colour))
            if not plan:
                if push and heaps and pushes < 6:  # 3a. something is there that was not made out: push it apart and look again
                    line = push_apart(heaps[0].uvs, pushes, inside)
                    if line is not None:
                        pushes += 1
                        self._push(job, g, setup, levels, line, f"push apart what the camera does not make out ({heaps[0].colour})")
                        continue
                left = left + [(h, "the camera does not make it out") for h in heaps]
                break
            b = plan[0]["block"]
            a = jaws_room(b, seen + heaps, cat.jaw_mm)  # 3. can the jaws get at it?
            if a is None:
                line = push_clear(b, seen + heaps, cat.jaw_mm, inside) if push and pushes < 6 else None
                if line is None:
                    return f"the jaws have no room beside the {b.name}, and it cannot be pushed clear: move the blocks apart, then start again"
                pushes += 1
                self._push(job, g, setup, levels, line, f"push the {b.name} clear of its neighbour")
                continue
            hold = plan[0]["hold_mm"]
            aim = base if not levels else (base + np.array(levels[-1].get("seen_uv", levels[-1]["uv"]))) / 2.0  # half way back to the tower's own axis
            if float(np.hypot(*(aim - base))) > 15.0:
                return f"the tower leans: its top stands {np.hypot(*(aim - base)) * 2:.0f} mm off its foot. Take it down by hand"
            sp = copy.deepcopy(self.spots)
            src, dst = (b.name if b.name not in sp.at else b.name + " (seen)"), "the tower"
            sp.at[src], sp.rot[src] = table.point(b.uv[0], b.uv[1], setup.w_grasp), self._turn_for(setup, a)
            sp.at[dst], sp.rot[dst] = table.point(aim[0], aim[1], setup.w_grasp), np.asarray(self.spots.turn(on), float).copy()
            sp.extra = top + b.rest.height + hold
            text = f"pick up the {b.name} and put it on " + ("the tower" if levels else f"spot {on}") + f" (level {len(levels) + 1})"
            try:
                rec = self._leg(job, g, sp, src, dst, hold, top + hold + (P.drop_mm if levels else 0.0), text, width_mm=b.grip_mm, open_mm=opening(b, cat.jaw_mm))
            except NothingHeld as e:  # also a block of another width: it was let go where it stood
                fails += 1
                self.say(f"{e}: looking again")
                self._mark("missed", name=b.name, why=str(e), wrong_width=isinstance(e, WrongWidth))
                continue
            if rec["blocked"]:
                if rec["blocked"]["step"].startswith("lower around"):  # the jaws met something on the way down: nothing is held
                    fails += 1
                    self._mark("missed", name=b.name, why=rec["blocked"]["why"])
                    continue
                return f"blocked at '{rec['blocked']['step']}': lifted clear with the block still in the jaws. Take it out, then start again"
            more = max(0.0, rec["rested"]["short_mm"] + (P.drop_mm if levels else 0.0) - P.stall_mm) if rec["rested"] else 0.0  # it met the tower early: the tower is that much higher
            levels.append(dict(name=b.name, kind=b.kind, rest=b.rest, uv=(float(aim[0]), float(aim[1])), angle=float((angle_on + b.angle - a) % 180.0), base_mm=top + more,
                               hold_mm=hold, top_mm=top + more + b.rest.height, rot=sp.rot[dst].copy()))
            fails = pushes = 0
            self._mark("placed", name=b.name, level=len(levels), top_mm=levels[-1]["top_mm"], rested=rec["rested"])
        with self.lock:
            self.stack = {**self.stack, "n": len(levels), "of": len(levels), "feeds": []}
        self.left_out = len(left)
        if left:
            self.say("left on the table: " + "; ".join(f"{s.name} ({why})" for s, why in left))
        return "done"

    def _push(self, job, g, setup, levels, line, text):
        """One push along the table with the closed gripper, the jaws across the way it goes."""
        table = setup.table
        (su, sv), (eu, ev) = line
        w = setup.w_grasp + 2.0  # the fingertips a little higher than where they close on a low block
        a, b = table.point(su, sv, w), table.point(eu, ev, w)
        sp = copy.deepcopy(self.spots)
        sp.at["push from"], sp.at["push to"] = a, b  # the guard's box is the one around the taught spots and this push
        sp.rot["push from"] = sp.rot["push to"] = self._turn_for(setup, float(np.degrees(np.arctan2(ev - sv, eu - su))))
        sp.extra = max([lv["top_mm"] + lv["hold_mm"] for lv in levels], default=0.0)
        self.say(text)
        self._mark("push", text=text, start=[float(su), float(sv)], end=[float(eu), float(ev)])
        rec = self._steps(job, g, sp, plan_push(sp, a, b, g.where(), self.P, sp.rot["push from"]), text)
        if self.sight is not None:
            self.sight["stale"] = True
        return rec

    def _scatter(self, job, g, eyes, cat, setup, on, aside, levels, rng):
        table = setup.table
        if not levels:
            return "done"
        im, seen, odd = self._look_aside(job, g, eyes, cat, setup, aside, levels)
        if on_tower(eyes, im, setup, levels)[0] is None:
            return f"the camera does not see the {levels[-1]['name']} on the tower: look at the table, then start again"
        base = np.array(table.coords(self.spots.grasp(on))[:2])
        area = np.asarray(setup.area, float)
        places = scatter_places(len(levels), rng, lambda p: bool(setup.inside(p)), [base] + [np.asarray(b.uv, float) for b in seen], box=(area.min(0), area.max(0)))
        if len(places) < len(levels):
            return f"there is room on the table for only {len(places)} of the tower's {len(levels)} blocks: clear the table"
        full = levels[-1]["top_mm"] + max(lv["hold_mm"] for lv in levels)
        for u, v, angle in places:
            lv = levels[-1]
            sp = copy.deepcopy(self.spots)
            src, dst = "the tower", "a free place"
            sp.at[src], sp.rot[src] = table.point(lv["uv"][0], lv["uv"][1], setup.w_grasp), np.asarray(lv["rot"], float).copy()
            sp.at[dst], sp.rot[dst] = table.point(u, v, setup.w_grasp), self._turn_for(setup, 0.0 if lv["rest"].round else angle)
            sp.extra = full
            text = f"take the {lv['name']} off the tower and put it on the table"
            try:
                rec = self._leg(job, g, sp, src, dst, lv["base_mm"] + lv["hold_mm"], lv["hold_mm"], text, width_mm=min(lv["rest"].short, lv["rest"].long))
            except NothingHeld as e:
                return f"{e}. Look at the tower, then start again"
            if rec["blocked"]:
                return f"blocked at '{rec['blocked']['step']}': lifted clear. Look, then start again"
            levels.pop()
            self._mark("taken off", name=lv["name"], to=[u, v, angle])
            with self.lock:
                self.stack = {**self.stack, "n": len(levels)}
                job["done"] = job.get("done", 0) + 1
        return "done"

    def _leg(self, job, g, sp, src, dst, su, du, text, width_mm=None, open_mm=None):
        """Take the thing at `src` and put it down at `dst` (spots of `sp`), as one leg of the job the page shows."""
        here = g.where()
        steps = plan_move(sp, src, dst, here, self.P, src_up=su, dst_up=du, width_mm=width_mm if self.check_width else None,
                          open_counts=None if open_mm is None else int(min(self.P.open_counts, round(open_mm * self.P.counts_per_mm))))
        g.lim = sp.limits(ceil_w=float(here @ sp.up) + 1.0, **self.limit_kw)
        with self.lock:
            job.update(steps=steps, i=None, src=src, dst=dst, text=text)
        self.say(text)
        self._mark("leg", text=text, src=src, dst=dst)
        if self.sight is not None:
            self.sight["stale"] = True
        rec = execute(g, steps, self.P, before=self._before(job, g))
        with open(self.log_dir / "moves.jsonl", "a") as fh:
            fh.write(json.dumps(dict(t=time.time(), src=src, dst=dst, src_up=su, dst_up=du, rehearse=False, text=text, **rec), default=float) + "\n")
        return rec
