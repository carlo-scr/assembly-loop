"""What a session does with the camera (the methods the dashboard's Camera card ends up in). Mixed into `Session`.

  show_table   ARMED, one bar lying flat on a taught spot, nothing else on the mat: the arm carries that block to a
               dozen places between the taught spots and steps aside after each; the camera takes a picture every
               time. From "the arm put it there" and "the camera saw it here" the camera is fitted (`eyes.settle`)
               and kept in `camera.json`. This is the only calibration, and nobody measures or prints anything.
  look         one picture, and what the camera makes of the blocks in it. Moves nothing.
  build_seen   a tower on a taught spot from the blocks of the last look, in the order `blocks.plan_tower` gives.
               Every seen block becomes a spot for this one job, so the arm takes it exactly as it takes a block
               from a taught spot: through the guard, only while the grip is held, lowering watched for contact.
  build_design a saved design (`design.py`, drawn in the designer) on a taught spot, from the blocks of the last
               look: every block of the design is matched to a seen block of the same kind lying the same way, and
               is put down where the design says, turned as it says, step by step in the design's order. Nothing is
               built unless `design.check` finds every step buildable and standing.

The camera never widens where the arm may go: blocks outside the part of the table it was shown are left where they
are, and the guard's box is the one around the taught spots and those blocks.
"""
from __future__ import annotations

import copy
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
    def _sight_init(self, camera=None, camera_path="camera.json", blocks_path="blocks.json", designs_dir="designs"):
        self.camera, self.camera_path, self.blocks_path, self.designs_dir = camera, Path(camera_path), Path(blocks_path), Path(designs_dir)
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
    def show_table(self, kind, n=12, step=False):
        """Fit the camera: carry the block that lies on the spot `thing_at` to `n` places and photograph each."""
        with self.lock:
            if self.phase() != "ARMED":
                raise Refused(f"the camera can be shown the table when ARMED (now: {self.phase()})")
            eyes, block = self._eyes(), self.catalogue().kind(str(kind))
            if not block.rests()[0].turns:
                raise Refused("show the camera the table with a bar or a slab: a cylinder on its end has no direction")
            if self.thing_at is None:
                raise Refused("say first on which spot the block lies (One block: 'The block stands on')")
            n = int(n)
            if not 8 <= n <= 20:
                raise Refused("between 8 and 20 places")
            sp, home = copy.deepcopy(self.spots), self.thing_at
            axes = sp.limits().axes
            C = np.array([axes.T @ p for p in sp.at.values()])
            lo, hi = C[:, :2].min(0), C[:, :2].max(0)
            if min(hi - lo) < 80.0:
                raise Refused("the taught spots are almost on one line: teach spots at opposite corners of the area the blocks will lie in (at least 80 mm apart each way)")
            nu = max(3, int(round(np.sqrt(n * (hi[0] - lo[0]) / (hi[1] - lo[1])))))
            nv = max(3, int(np.ceil(n / nu)))
            w, here, names = float(sp.grasp(home) @ sp.up), axes.T @ sp.grasp(home), []
            for i, u in enumerate(np.linspace(lo[0], hi[0], nu)):
                for v in (np.linspace(lo[1], hi[1], nv) if i % 2 == 0 else np.linspace(hi[1], lo[1], nv)):  # there and back, so that the block is never carried far
                    if np.hypot(u - here[0], v - here[1]) < 15.0:
                        continue  # the block's own spot is photographed before anything moves
                    names.append(f"place {len(names) + 1}")
                    sp.at[names[-1]], sp.rot[names[-1]] = axes @ np.array([u, v, w]), sp.turn(home).copy()
            corners = [k for k in self.spots.at]
            shots = []

            def photo(at):
                """Step aside (above the taught spot that is farthest away), take a picture, note where the block lies."""
                far = max(corners, key=lambda k: float(np.linalg.norm(sp.grasp(k) - sp.grasp(at))))
                self.guard.move_to(sp.above(self.guard.where()), self.P.lift, "go to the travel height")
                self.guard.move_to(sp.above(sp.grasp(far)), self.P.travel, "step aside for the camera")
                shots.append((sp.grasp(at).copy(), self._grab()))

            def finish():
                setup, line = eyes.settle([p for p, _ in shots], [im for _, im in shots], block, axes, sp.turn(home),
                                          extra_uv=[tuple((axes.T @ p)[:2]) for p in self.spots.at.values()], side=self.chosen)
                setup.save(self.camera_path)
                with self.lock:
                    self.setup, self.sight, self.thing_at = setup, None, home
                self.say(f"{line}. Kept in {self.camera_path}")

            stops = [home] + names + [home]
            legs = [dict(src=a, dst=b, su=0.0, du=0.0, text=f"the block to {b}", first=(lambda a=a: photo(a)) if i == 0 else None, then=(lambda b=b: photo(b)) if b != home else None)
                    for i, (a, b) in enumerate(zip(stops, stops[1:]))]
            self.job = dict(steps=[], i=None, waiting=False, outcome=None, done=0, total=len(legs), src=legs[0]["src"], dst=legs[0]["dst"], rehearse=False, step=bool(step),
                            fixed=None, kind="move", legs=legs, spots=sp, text=legs[0]["text"], finish=finish)
            self.stack = None
            self.say(f"showing the camera the table: the {block.name} goes to {len(names)} places. Keep everything else off the mat")
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
                legs.append(dict(src=name, dst=on, su=lv["hold_mm"], du=lv["base_mm"] + lv["hold_mm"] + (self.P.drop_mm if k else 0.0),
                                 text=f"{b.name} onto {on}, level {k + 1} of {len(levels)}"))
            sp.extra = levels[-1]["top_mm"] + max(lv["hold_mm"] for lv in levels)  # a carried block clears the finished tower, however low it hangs
            self.stack, self.thing_at = dict(on=on, block_mm=None, n=0, of=len(levels), feeds=[leg["src"] for leg in legs]), None
            s["stale"] = True
            self.job = dict(steps=[], i=None, waiting=False, outcome=None, done=0, total=len(legs), src=legs[0]["src"], dst=on, rehearse=bool(rehearse), step=bool(step),
                            fixed=None, kind="move", legs=legs, spots=sp, text=legs[0]["text"], tower=True)
            self.say(f"a tower of {len(levels)} on {on}, {levels[-1]['top_mm']:.0f} mm high: " + ", ".join(lv["block"].name for lv in levels))
            self._next.clear()
            self.worker = threading.Thread(target=self._work, daemon=True)
            self.worker.start()

    # ------------------------------------------------------------------ designs (the designer page; no arm needed)
    def design_catalogue(self):
        """What the designer may build with: the kinds of block, every way each can lie, and the gripper's sizes."""
        from . import design as D

        cat, path = D.catalogue_for_design(self.blocks_path)
        kinds = [dict(name=k.name, shape=k.shape, size=list(k.size), colour=k.colour,
                      rests=[dict(name=r.name, height=r.height, long=r.long, short=r.short, round=r.round, rolls=r.rolls) for r in k.rests()]) for k in cat.kinds]
        return dict(kinds=kinds, finger_mm=cat.finger_mm, jaw_mm=cat.jaw_mm, source=path.name, example=path.name != self.blocks_path.name, gripper=D.GRIPPER,
                    reach_mm=D.REACH_MM, height_mm=min(D.HEIGHT_MM, self._tower_max()), shaky_deg=D.SHAKY_DEG, max_tilt_deg=D.MAX_TILT_DEG)

    def design_list(self):
        from . import design as D

        return D.listing(self.designs_dir)

    def design_get(self, name):
        from . import design as D

        return D.load(self.designs_dir, name).to_dict()

    def design_check(self, d):
        from . import design as D

        cat, _ = D.catalogue_for_design(self.blocks_path)
        return D.check(D.Design.from_dict(d), cat, max_mm=min(D.HEIGHT_MM, self._tower_max()))

    def design_save(self, d):
        from . import design as D

        dz = D.Design.from_dict(d)
        dz.name = D.check_name(dz.name)
        f = D.save(self.designs_dir, dz)
        self.say(f"design {dz.name!r} saved in {f}")
        return dict(name=dz.name, file=str(f))

    def design_order(self, d):
        from . import design as D

        cat, _ = D.catalogue_for_design(self.blocks_path)
        return D.find_order(D.Design.from_dict(d), cat).to_dict()

    # ------------------------------------------------------------------ build a design from what the camera sees
    def build_design(self, name, on, rehearse=False, step=False):
        """The saved design `name`, built on the taught spot `on` from the blocks of the last look."""
        from . import design as D

        with self.lock:
            if self.phase() != "ARMED":
                raise Refused(f"a design can only be built when ARMED (now: {self.phase()})")
            self._eyes()
            cat, why = self.catalogue(), self._setup_fits()
            if why:
                raise Refused(why)
            s, setup = self.sight, self.setup
            if s is None or s["stale"] or time.time() - s["t"] > 300.0:
                raise Refused("press Look first: the arm builds from a picture taken after the last move, at most five minutes old")
            self.spots.grasp(on)
            dz = D.load(self.designs_dir, name)
            rep = D.check(dz, cat, max_mm=self._tower_max())
            if not rep["ok"]:
                first = next((f"step {k + 1}: {b['problems'][0]}" for k, b in enumerate(rep["blocks"]) if b["problems"]), None)
                raise Refused(f"the design {dz.name!r} cannot be built ({first or (rep['problems'] or ['it does not stand'])[0]}): open it in the designer")
            parts = D.resolve(dz, cat)
            table, up = setup.table, self.spots.up
            base = table.coords(self.spots.grasp(on))[:2]
            fwd = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])  # the design's forward: away from the base, level
            fwd = fwd - (fwd @ up) * up
            fwd /= np.linalg.norm(fwd)
            f2, l2 = (table.axes.T @ fwd)[:2], (table.axes.T @ np.cross(up, fwd))[:2]
            uv = lambda x, y: base + x * f2 + y * l2  # noqa: E731  a point of the design on the table
            to_design = lambda q: np.array([(np.asarray(q) - base) @ f2, (np.asarray(q) - base) @ l2])  # noqa: E731
            ang = lambda deg: float(np.degrees(np.arctan2(*(np.cos(np.radians(deg)) * f2 + np.sin(np.radians(deg)) * l2)[::-1])))  # noqa: E731  a direction of the design, as a direction of the table
            for b in s["seen"]:  # nothing may lie where the design goes, nor where the fingers reach beside it
                q, a = to_design(b.uv), np.radians(b.angle)
                d = np.cos(a) * table.axes[:, 0] + np.sin(a) * table.axes[:, 1]  # its long side, as a direction of the arm
                deg = float(np.degrees(np.arctan2(d @ np.cross(up, fwd), d @ fwd)))
                there = D.circle(q[0], q[1], b.rest.long) if b.rest.round else D.rect(q[0], q[1], b.rest.long, b.rest.short, deg)
                if any(D.gap(there, p.poly) < D.CLEAR_MM for p in parts):
                    raise Refused(f"the {b.name} lies where the design goes on {on}, or within {D.CLEAR_MM:.0f} mm of it: move it away, then Look again")
            pool, legs, plan, missing = sorted(s["seen"], key=lambda b: float(np.linalg.norm(np.asarray(b.uv) - base))), [], [], {}
            sp = copy.deepcopy(self.spots)
            for k, p in enumerate(parts):
                b = next((b for b in pool if b.kind.name == p.kind.name and b.rest.name == p.rest.name), None)
                if b is None:
                    missing[(p.kind.name, p.rest.name)] = missing.get((p.kind.name, p.rest.name), 0) + 1
                    continue
                pool.remove(b)
                src, dst = (b.name if b.name not in self.spots.at else b.name + " (seen)"), f"design place {k + 1}"
                t = b.rest.turns
                turn_src = 0.0 if not t else (b.angle - setup.angle_ref + t / 2.0) % t - t / 2.0
                turn_dst = (ang(rep["blocks"][k]["grip_deg"]) - setup.angle_ref + 90.0) % 180.0 - 90.0  # the gripper is the same turned by half a turn
                sp.at[src], sp.rot[src] = table.point(b.uv[0], b.uv[1], setup.w_grasp), spin(setup.rpy_ref, up, turn_src)
                sp.at[dst], sp.rot[dst] = table.point(*uv(p.block.x, p.block.y), setup.w_grasp), spin(setup.rpy_ref, up, turn_dst)
                legs.append(dict(src=src, dst=dst, su=p.hold, du=p.z + p.hold + (self.P.drop_mm if p.z > D.TOUCH_MM else 0.0), adds=True,
                                 release=int(round(10.0 * rep["blocks"][k]["release_mm"])), text=f"{b.name} to step {k + 1} of {len(parts)}"))
                a = np.radians(ang(p.block.angle))
                plan.append(dict(name=p.kind.name, xyz=[round(float(v), 1) for v in sp.at[dst]], base=round(p.z, 1), along=[float(v) for v in table.axes @ np.array([np.cos(a), np.sin(a), 0.0])],
                                 size=[p.rest.long, p.rest.short, p.rest.height], round=bool(p.rest.round), colour=p.kind.colour))
            if missing:
                raise Refused("lay these out on the mat, apart from each other, then Look again: " + ", ".join(f"{n} x {k} ({r})" for (k, r), n in sorted(missing.items())))
            sp.extra = rep["height_mm"] + max(p.hold for p in parts)  # a carried block clears what is built, however low it hangs
            self.stack, self.thing_at = dict(on=on, block_mm=None, n=0, of=len(legs), feeds=[leg["src"] for leg in legs], design=dz.name), None
            s["stale"] = True
            self.job = dict(steps=[], i=None, waiting=False, outcome=None, done=0, total=len(legs), src=legs[0]["src"], dst=legs[0]["dst"], rehearse=bool(rehearse), step=bool(step),
                            fixed=None, kind="move", legs=legs, spots=sp, text=legs[0]["text"], tower=True, plan=plan)
            low = rep["tilt_deg"]
            self.say(f"building the design {dz.name!r} on {on}: {len(legs)} blocks, {rep['height_mm']:.0f} mm high, holds a tilt of {low:.0f} degrees"
                     + (" (SHAKY: a bump may topple it)" if low < D.SHAKY_DEG else ""))
            self._next.clear()
            self.worker = threading.Thread(target=self._work, daemon=True)
            self.worker.start()
