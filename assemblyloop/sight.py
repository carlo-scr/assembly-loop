"""What a session does with the camera (the methods the dashboard's Camera card ends up in). Mixed into `Session`.

  show_table   ARMED, one bar lying flat on a taught spot, nothing else on the mat: the arm carries that block to a
               dozen places between the taught spots and steps aside after each; the camera takes a picture every
               time. From "the arm put it there" and "the camera saw it here" the camera is fitted (`eyes.settle`)
               and kept in `camera.json`. This is the only calibration, and nobody measures or prints anything.
  look         one picture, and what the camera makes of the blocks in it. Moves nothing.
  build_seen   a tower on a taught spot from the blocks of the last look, in the order `blocks.plan_tower` gives.
               Every seen block becomes a spot for this one job, so the arm takes it exactly as it takes a block
               from a taught spot: through the guard, only while the grip is held, lowering watched for contact.

The camera never widens where the arm may go: blocks outside the part of the table it was shown are left where they
are, and the guard's box is the one around the taught spots and those blocks.
"""
from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path

import numpy as np

from .arms import matrix_rpy, rpy_matrix, twist_deg
from .blocks import Catalogue, plan_tower
from .guard import Refused


def spin(rpy_deg, up, deg):
    """The gripper turned by `deg` about `up` (seen from above: counter-clockwise), its tilt unchanged."""
    a, k = np.radians(deg), np.asarray(up, float)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return matrix_rpy((np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * K @ K) @ rpy_matrix(rpy_deg))


class Sight:
    def _sight_init(self, camera=None, camera_path="camera.json", blocks_path="blocks.json"):
        self.camera, self.camera_path, self.blocks_path = camera, Path(camera_path), Path(blocks_path)
        self.setup, self.sight, self._setup_read = None, None, False

    # ------------------------------------------------------------------ what is there to work with
    def _eyes(self):
        try:
            from . import eyes
        except ImportError:
            raise Refused('the camera needs OpenCV: pip install -e ".[eyes]"') from None
        if self.camera is None:
            self.camera = eyes.V4L2()
        if not self._setup_read:
            self._setup_read = True
            if self.camera_path.exists():
                try:
                    self.setup = eyes.Setup.load(self.camera_path)
                except (OSError, ValueError, KeyError, TypeError):
                    self.say(f"{self.camera_path} is not readable: show the camera the table again")
        return eyes

    def _grab(self):
        """One picture from the camera that looks at the table. If the page is showing that camera live, its owner
        lets go of it for the moment (a camera streams to one program at a time)."""
        feeds = getattr(self, "feeds", None)
        if feeds is None:
            return self.camera.grab()
        with feeds.borrow(getattr(self.camera, "device", None)):
            return self.camera.grab()

    def catalogue(self):
        if not self.blocks_path.exists():
            raise Refused(f"there is no {self.blocks_path}: measure the blocks and write them down (see blocks.example.json)")
        return Catalogue.load(self.blocks_path)

    def _setup_fits(self):
        """'' if the camera's setup belongs to the spots that are known now, otherwise why not."""
        if self.setup is None:
            return "the camera has not been shown the table yet"
        if self.setup.side != self.chosen:
            return f"the camera was shown the table with the {self.setup.side} arm"
        if twist_deg(rpy_matrix(self.setup.rpy_ref), rpy_matrix(self.spots.rpy))[1] > 2.0:
            return "the gripper is tilted differently than when the camera was shown the table: show it again"
        return ""

    # ------------------------------------------------------------------ look
    def look(self):
        """One picture and what is in it. The arm should be out of the camera's way: what it hides is not seen."""
        with self.lock:
            if self.worker is not None:
                raise Refused("look when the arm has finished")
            eyes, cat = self._eyes(), self.catalogue()
            if self.setup is None:
                raise Refused("the camera has not been shown the table yet")
            setup = self.setup
        im = self._grab()
        tall = max(r.height for k in cat.kinds for r in k.rests())
        seen, unknown, blobs = eyes.look(im, setup.camera, setup.table, cat, setup.region((im.shape[1], im.shape[0]), tall))
        outside = [s for s in seen if not setup.inside(s.uv)]
        seen = [s for s in seen if setup.inside(s.uv)]
        ok, jpg = eyes.cv2.imencode(".jpg", eyes.draw(im, blobs, seen, unknown, setup.camera, setup.table), [eyes.cv2.IMWRITE_JPEG_QUALITY, 80])
        with self.lock:
            self.sight = dict(t=time.time(), seen=seen, unknown=[why for _, why in unknown] + [f"{s.name}: outside the part of the table the camera was shown" for s in outside],
                              jpg=jpg.tobytes() if ok else b"", stale=False)
            self.say(f"the camera sees {len(seen)} block{'' if len(seen) == 1 else 's'}" + (f": {', '.join(s.name for s in seen)}" if seen else "")
                     + (f"; {len(self.sight['unknown'])} thing{'' if len(self.sight['unknown']) == 1 else 's'} not made out" if self.sight["unknown"] else ""))
        return self.sight

    def sight_jpg(self):
        return (self.sight or {}).get("jpg") or b""

    def sight_state(self):
        """For the page: what the camera can do now, and what it saw last."""
        out = dict(ready=False, why="", kinds=[], bars=[], shown=None, fit_mm=None, sight=None)
        try:
            cat = self.catalogue()
            out["kinds"], out["bars"] = [k.name for k in cat.kinds], [k.name for k in cat.kinds if k.rests()[0].turns]
            if not self._setup_read and self.camera_path.exists():
                self._eyes()
        except Refused as e:
            out["why"] = str(e)
            return out
        if self.setup is not None:
            out.update(shown=time.strftime("%d %b %H:%M", time.localtime(self.setup.when)), fit_mm=round(self.setup.fit_mm, 1))
        out["why"] = self._setup_fits() if self.calibrated else ("the camera has not been shown the table yet" if self.setup is None else "")
        out["ready"] = not out["why"]
        s = self.sight
        if s is not None:
            levels, left = plan_tower(s["seen"], cat, max_mm=self._tower_max()) if s["seen"] else ([], [])
            out["sight"] = dict(t=s["t"], age_s=round(time.time() - s["t"]), stale=s["stale"], unknown=s["unknown"],
                                seen=[dict(name=b.name, lies=b.rest.name, mm=[round(b.uv[0]), round(b.uv[1])], fit=round(100 * b.score)) for b in s["seen"]],
                                order=[lv["block"].name for lv in levels], height_mm=levels[-1]["top_mm"] if levels else 0.0, left_out=[f"{b.name}: {why}" for b, why in left])
        return out

    def _tower_max(self):
        return 300.0 - (self.spots.lift if self.spots is not None else 80.0)  # the travel height stays within what the guard has always allowed

    # ------------------------------------------------------------------ show the camera the table
    def show_table(self, kind, n=12, step=False, grid_mm=60.0):
        """Fit the camera: carry the block that lies on the spot `thing_at` to up to `n` places and photograph each.
        The places are not laid out beforehand. They go outwards from where the block lies, `grid_mm` apart, and a
        place is used only if the arm can reach it and, as far as the pictures taken so far tell, it is on the grey
        mat. So the area that comes out is as large as the arm and the mat allow, wherever the block was put."""
        with self.lock:
            if self.phase() != "ARMED":
                raise Refused(f"the camera can be shown the table when ARMED (now: {self.phase()})")
            eyes, block = self._eyes(), self.catalogue().kind(str(kind))
            if not block.rests()[0].turns:
                raise Refused("show the camera the table with a bar or a slab: a cylinder on its end has no direction")
            if self.thing_at is None:
                raise Refused("say first on which spot the block lies (One block: 'The block stands on')")
            self._empty_handed("showing the camera the table")
            n = int(n)
            if not 8 <= n <= 20:
                raise Refused("between 8 and 20 places")
            sp, home, up = copy.deepcopy(self.spots), self.thing_at, self.spots.up
            axes, rpy, base = sp.limits().axes, sp.turn(home).copy(), sp.grasp(home).copy()
            fwd = np.array([1.0, 0.0, 0.0]) - up[0] * up  # away from the arm's own base, along the table
            fwd = fwd / np.linalg.norm(fwd)
            left = np.cross(up, fwd)
            place = lambda c: base + grid_mm * (c[0] * fwd + c[1] * left)  # noqa: E731
            can = lambda p: not (self.arm.reach(p, rpy) or self.arm.reach(p + sp.lift * up, rpy))  # noqa: E731
            # Where the arm waits while a picture is taken. The rig's camera looks across the table from beside the arm, so
            # the arm hides much of the table from it almost wherever it waits (straight back, it hid the block in 8 of 13
            # pictures). Before the first picture the arm therefore tries a few waiting places, mostly out to its own right,
            # and keeps the one from which the camera sees the most of the grey mat.
            waits = [q for q in (base + f * fwd + side * left for f, side in ((-150.0, 0.0), (-60.0, -300.0), (60.0, -300.0), (150.0, -300.0), (-150.0, -300.0), (0.0, -400.0), (150.0, -400.0), (0.0, 300.0))) if can(q)]
            if not waits:
                raise Refused("the arm cannot step aside from here for the camera: put the area somewhere else")
            for k, q in enumerate(waits):  # for this job they are places like the others: inside what the guard allows
                sp.at[f"aside {k + 1}"], sp.rot[f"aside {k + 1}"] = q, rpy.copy()
            aside = [None]
            todo = []  # ring by ring outwards; the places next to the block go round in a circle, the rings further out are taken
            for ring in range(1, 6):  # ... a half turn, a quarter turn, ... apart, so that a ring that is not finished still lies all around
                cs = sorted((c for c in ((i, j) for i in range(-1, 5) for j in range(-5, 6)) if max(abs(c[0]), abs(c[1])) == ring), key=lambda c: np.arctan2(c[1], c[0]))
                spread = sorted(range(len(cs)), key=lambda k: int(format(k, "08b")[::-1], 2)) if ring > 1 else range(len(cs))
                todo += [cs[k] for k in spread]
            shots, cells, at, said = [], {home: (0, 0)}, [home], set()  # pictures; which place is where; where the block lies now

            def known():
                """(place in mm from the block's first place, pixel) for every picture in which the block is plain."""
                blobs = eyes.the_block([im for _, im in shots], block.colour)
                return [((grid_mm * cells[name][0], grid_mm * cells[name][1]), b.centre) for (name, _), b in zip(shots, blobs) if b is not None]

            def photo(name):
                """Step aside, take a picture, note where the block lies. The first time, find the best place to step to."""
                self.guard.move_to(sp.above(self.guard.where()), self.P.lift, "go to the travel height")
                if aside[0] is None:
                    tried = []
                    for q in waits:
                        self.guard.move_to(sp.above(q), self.P.travel, "try a place to wait for the camera")
                        im = self._grab()
                        tried.append((eyes.mat_seen(im), im))
                    best = max(range(len(tried)), key=lambda k: tried[k][0])
                    aside[0] = waits[best]
                    off = aside[0] - base
                    self.say(f"  the arm will wait {abs(float(off @ left)):.0f} mm to its {'left' if off @ left > 0 else 'right'} and {abs(float(off @ fwd)):.0f} mm {'ahead' if off @ fwd > 0 else 'back'}: "
                             f"the camera sees {100.0 * tried[best][0] / max(max(t[0] for t in tried), 1):.0f}% of the most mat there, {100.0 * tried[0][0] / max(tried[best][0], 1):.0f}% straight back")
                    shots.append((name, tried[best][1]))
                    return
                self.guard.move_to(sp.above(aside[0]), self.P.travel, "step aside for the camera")
                shots.append((name, self._grab()))

            def more():
                """The next move: to the nearest place not yet used that the arm reaches and that is on the mat; after
                the last one, back to where the block lay at first."""
                if at[0] is None:
                    return None
                pairs = known() if len(shots) >= 3 else []
                while todo and len(cells) - 1 < n:
                    c = todo.pop(0)
                    p = place(c)
                    if not can(p):
                        continue
                    px, scale = eyes.guess_pixel(pairs, (grid_mm * c[0], grid_mm * c[1]))
                    if px is not None and not eyes.on_mat(shots[-1][1], px, 0.6 * grid_mm * scale):
                        if "mat" not in said:
                            said.add("mat")
                            self.say("  a place that would be off the grey mat is left out (and so are the others like it)")
                        continue
                    name = f"place {len(cells)}"
                    cells[name], sp.at[name], sp.rot[name] = c, p, rpy.copy()
                    leg = dict(src=at[0], dst=name, su=0.0, du=0.0, grip_mm=block.rests()[0].short, text=f"the block to {name}", then=lambda name=name: photo(name),
                               first=(lambda: photo(home)) if not shots else None)
                    at[0] = name
                    return leg
                if at[0] == home:
                    at[0] = None
                    return None
                leg, at[0] = dict(src=at[0], dst=home, su=0.0, du=0.0, grip_mm=block.rests()[0].short, text=f"the block back to {home}"), None
                return leg

            def finish():
                keep = self.log_dir / "camera_shots"  # what the camera saw and where the arm had put the block: kept, so that a fit can be done again without the arm
                keep.mkdir(parents=True, exist_ok=True)
                xyz = [sp.grasp(name) for name, _ in shots]
                for i, (_, im) in enumerate(shots):
                    eyes.cv2.imwrite(str(keep / f"{i:02d}.jpg"), im, [eyes.cv2.IMWRITE_JPEG_QUALITY, 92])
                (keep / "places.json").write_text(json.dumps(dict(kind=block.name, axes=np.asarray(axes).tolist(), rpy=[float(v) for v in rpy], side=self.chosen,
                                                                  xyz=[[float(v) for v in p] for p in xyz], taught=[[float(v) for v in p] for p in self.spots.at.values()])))
                if len(shots) < 9:
                    raise Refused(f"only {len(shots) - 1} places could be used around the block (the arm's reach and the mat allow no more): put the area somewhere with more room")
                setup, line = eyes.settle(xyz, [im for _, im in shots], block, axes, rpy, extra_uv=[tuple((axes.T @ base)[:2])], side=self.chosen)
                setup.save(self.camera_path)
                with self.lock:
                    self.setup, self.sight, self.thing_at = setup, None, home
                self.say(f"{line}. Kept in {self.camera_path}")

            first = more()
            if first is None:
                raise Refused("there is no place next to the block that the arm can reach: put the area closer to the arm")
            self.job = dict(steps=[], i=None, waiting=False, outcome=None, done=0, total=n + 1, src=first["src"], dst=first["dst"], rehearse=False, step=bool(step),
                            fixed=None, kind="move", legs=[first], spots=sp, text=first["text"], finish=finish, more=more)
            self.stack = None
            self.say(f"showing the camera the table: the {block.name} goes to up to {n} places around where it lies, where the arm reaches and the mat is. Keep everything else off the mat")
            self._next.clear()
            self.worker = threading.Thread(target=self._work, daemon=True)
            self.worker.start()

    # ------------------------------------------------------------------ a tower from what the camera sees
    def build_seen(self, on, rehearse=False, step=False):
        """A tower on the taught spot `on` from the blocks of the last look, widest first."""
        with self.lock:
            if self.phase() != "ARMED":
                raise Refused(f"a tower can only be started when ARMED (now: {self.phase()})")
            self._eyes()
            cat, why = self.catalogue(), self._setup_fits()
            if why:
                raise Refused(why)
            s, setup = self.sight, self.setup
            if not rehearse:
                self._empty_handed("a tower")
            if s is None or s["stale"] or time.time() - s["t"] > 300.0:
                raise Refused("press Look first: the arm builds from a picture taken after the last move, at most five minutes old")
            self.spots.grasp(on)
            table, up = setup.table, self.spots.up
            base = table.coords(self.spots.grasp(on))[:2]
            free = [b for b in s["seen"] if np.hypot(b.uv[0] - base[0], b.uv[1] - base[1]) >= 60.0]
            if len(free) < len(s["seen"]):
                raise Refused(f"something stands where the tower goes ({on}): clear that spot, then Look again")
            levels, _ = plan_tower(free, cat, max_mm=self._tower_max())
            if not levels:
                raise Refused("the camera sees no block that can go onto a tower")
            sp, legs = copy.deepcopy(self.spots), []
            for k, lv in enumerate(levels):
                b = lv["block"]
                name = b.name if b.name not in self.spots.at else b.name + " (seen)"
                t = b.rest.turns
                turn = 0.0 if not t else (b.angle - setup.angle_ref + t / 2.0) % t - t / 2.0  # the shorter way round: a bar is the same turned by 180
                sp.at[name], sp.rot[name] = table.point(b.uv[0], b.uv[1], setup.w_grasp), spin(setup.rpy_ref, up, turn)
                legs.append(dict(src=name, dst=on, su=lv["hold_mm"], grip_mm=b.grip_mm, du=lv["base_mm"] + lv["hold_mm"] + (self.P.drop_mm if k else 0.0),
                                 text=f"{b.name} onto {on}, level {k + 1} of {len(levels)}"))
            sp.extra = levels[-1]["top_mm"] + max(lv["hold_mm"] for lv in levels)  # a carried block clears the finished tower, however low it hangs
            self._reach([sp.grasp(leg["src"]) for leg in legs] + [sp.above(sp.grasp(leg["src"])) for leg in legs] + [sp.above(sp.grasp(on))], sp.rpy, "this tower")
            self.stack, self.thing_at = dict(on=on, block_mm=None, n=0, of=len(levels), feeds=[leg["src"] for leg in legs]), None
            s["stale"] = True
            self.job = dict(steps=[], i=None, waiting=False, outcome=None, done=0, total=len(legs), src=legs[0]["src"], dst=on, rehearse=bool(rehearse), step=bool(step),
                            fixed=None, kind="move", legs=legs, spots=sp, text=legs[0]["text"], tower=True)
            self.say(f"a tower of {len(levels)} on {on}, {levels[-1]['top_mm']:.0f} mm high: " + ", ".join(lv["block"].name for lv in levels))
            self._next.clear()
            self.worker = threading.Thread(target=self._work, daemon=True)
            self.worker.start()
