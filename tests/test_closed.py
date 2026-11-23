"""A tower built with a look after every block: the stand-in arm, watched by a made-up camera that draws the stand-in
blocks where they are and turned as they are, so that what goes wrong between two looks shows in the next one."""
import threading
import time

import cv2
import numpy as np
import pytest

from assemblyloop.arms import FakeArm
from assemblyloop.blocks import Catalogue, Kind, Seen
from assemblyloop.closed import fingers, footprint, jaws_room, opening, push_apart, push_clear, rects_touch, scatter_places
from assemblyloop.eyes import PAINT, Camera, Setup, Table, outline, solid
from assemblyloop.guard import Refused
from assemblyloop.session import Session

CAT = Catalogue([Kind("green bar", "box", (75.0, 25.0, 25.0), "green"), Kind("blue slab", "box", (60.0, 40.0, 15.0), "blue"), Kind("yellow cube", "box", (30.0, 30.0, 30.0), "yellow"),
                 Kind("red bar", "box", (75.0, 25.0, 25.0), "red")]).check()
BAR = CAT.kind("green bar").rests()[0]
AREA = ((385.0, -125.0), (535.0, -125.0), (535.0, 125.0), (385.0, 125.0))  # the part of the table the camera was shown, in the arm's x and y
ANGLE_REF = 30.0  # the way a bar's long side points when the gripper holds it turned as at spot A


def until(cond, secs):
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        if cond():
            return True
        time.sleep(0.01)
    return False


class Scene:
    """Draws every stand-in block where the stand-in arm has it, the lower ones first, each turned as far as the arm
    has turned it. `then` is called before every picture (to let something go wrong between two looks)."""

    def __init__(self, ses, size=(1920, 1080)):
        eye, at = np.array([1350.0, -350.0, 900.0]), np.array([460.0, 0.0, 120.0])
        z = (at - eye) / np.linalg.norm(at - eye)
        x = np.cross(z, [0.0, 0.0, 1.0])
        x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z])
        self.ses, self.cam, self.what, self.n, self.then = ses, Camera(size, 0.7 * size[0], (-0.18, 0.04), cv2.Rodrigues(R)[0].ravel(), -R @ eye), {}, 0, None

    def grab(self):
        self.n += 1
        if self.then:
            self.then(self)
        a, axes = self.ses.arm, self.ses.spots.limits().axes
        im = np.full((self.cam.size[1], self.cam.size[0], 3), (110, 100, 95), np.uint8)
        for name, p in sorted(a.bricks.items(), key=lambda kv: float(axes[:, 2] @ kv[1])):
            kind, angle = self.what[name]
            rest, c = CAT.kind(kind).rests()[0], axes.T @ p
            bottom = c[2] - max(0.0, rest.height - CAT.finger_mm) - BAR.height / 2.0
            hull = outline(self.cam, solid(rest, Table(axes, bottom), c[0], c[1], angle + a.angles.get(name, 0.0), CAT.kind(kind).shape))
            cv2.fillConvexPoly(im, np.round(hull * 16).astype(np.int32), PAINT[CAT.kind(kind).colour], shift=4)
        return im


def lay(ses, blocks):
    """Put blocks on the stand-in's table: {name: (kind, x, y, angle)}. The jaws stall at each block's own width."""
    a, setup = ses.arm, ses.setup
    a.holding, a.bricks, a.angles, a.widths = None, {}, {}, {}
    for name, (kind, x, y, angle) in blocks.items():
        r = CAT.kind(kind).rests()[0]
        uv = setup.table.coords([x, y, 0.0])[:2]
        a.bricks[name] = setup.table.point(uv[0], uv[1], setup.w_grasp + max(0.0, r.height - CAT.finger_mm))
        a.widths[name] = int(round(10 * r.short))
        ses.camera.what[name] = (kind, angle)


def on_the_tower(ses, name, level_mm, within=3.0):
    sp, d = ses.spots, ses.arm.bricks[name] - ses.spots.grasp("T")
    return abs(float(d @ sp.up) - level_mm) < 0.8 and float(np.linalg.norm(d - (d @ sp.up) * sp.up)) < within


@pytest.fixture
def ses(tmp_path):
    s = Session({"stand-in": dict(label="Stand-in arm (no robot)", ip=None, make=FakeArm)}, spots_path=tmp_path / "spots.json", log_dir=tmp_path / "logs", probe=lambda ip: False,
                fake_speed=40.0, guard_kw=dict(poll_s=0.002), limit_kw=dict(settle_s=0.0), camera_path=tmp_path / "camera.json", blocks_path=tmp_path / "blocks.json")
    s.camera, s.alive, s.pressed = Scene(s), True, True
    CAT.save(s.blocks_path)

    def grip():
        while s.alive:
            if s.dm is not None:
                s.dm.feed(s.pressed)
            time.sleep(0.02)

    s.pressed = False
    threading.Thread(target=grip, daemon=True).start()
    s.start("stand-in")
    assert until(lambda: all(c["ok"] for c in s.state()["checks"]), 6)
    s.arm_on()
    s.use_demo()
    sp, axes = s.spots, s.spots.limits().axes
    w = float(sp.grasp("A") @ sp.up)  # the camera as if it had been shown the table: fitted exactly, the table at the taught height
    s._eyes()
    s.setup = Setup(s.camera.cam, Table(axes, w - BAR.height / 2.0), w, np.asarray(sp.turn("A"), float), ANGLE_REF, [tuple((axes.T @ np.array([x, y, 0.0]))[:2]) for x, y in AREA],
                    side="stand-in", fit_mm=0.5, when=time.time())
    s.pressed = True
    yield s
    s.alive = False
    s.close()


def finished(ses, secs=240):
    assert until(lambda: ses.worker is None, secs), ses.state()["job"]
    return ses.job["outcome"]


def test_the_flat_geometry_of_fingers_and_pushes():
    bar, cube = CAT.kind("green bar"), CAT.kind("yellow cube")
    b = Seen(bar, bar.rests()[0], (0.0, 0.0), 0.0, name="bar")
    assert rects_touch(footprint(b), footprint(Seen(cube, cube.rests()[0], (40.0, 20.0), 30.0))) and not rects_touch(footprint(b), footprint(Seen(cube, cube.rests()[0], (0.0, 60.0), 0.0)))
    f = fingers(b, 0.0, 85.0)  # the fingers come down across the bar's short side, half the jaws' opening away
    assert np.allclose(sorted(r.mean(0)[1] for r in f), [-48.5, 48.5]) and opening(b, 85.0) == 45.0 and np.allclose([r.mean(0)[0] for r in f], 0.0)
    near, far = Seen(cube, cube.rests()[0], (0.0, 50.0), 0.0, name="near"), Seen(cube, cube.rests()[0], (120.0, 0.0), 0.0, name="far")
    assert jaws_room(b, [b, far], 85.0) == 0.0 and jaws_room(b, [b, near], 85.0) is None  # a bar is only taken across its short side
    assert jaws_room(near, [b, near], 85.0) == 90.0  # a cube with a bar beside it is taken from the other side
    (start, end) = push_clear(b, [b, near], 85.0, lambda p: True)
    assert abs(start[1]) < 1e-6 and abs(end[1]) < 1e-6 and abs(abs(start[0]) - 59.5) < 1e-6 and abs(end[0] - start[0]) == 68.0  # along the bar, away from where the cube is in the way, starting beside it
    assert push_clear(b, [b, near], 85.0, lambda p: False) is None
    heap = np.array([(x, y) for x in np.linspace(-75, 75, 30) for y in np.linspace(-12, 12, 6)])  # two bars end to end
    lines = [push_apart(heap, k, lambda p: True) for k in range(4)]
    assert len({(round(s[0]), round(s[1])) for s, _ in lines}) == 4  # every attempt comes from somewhere else
    assert abs(abs(lines[0][0][0]) - 37.5) < 2.0 and abs(lines[0][0][1]) > 12.0 + 14.0  # the first: across, a quarter of the way from one end
    room, box = (lambda p: bool(abs(p[0]) < 250 and abs(p[1]) < 200)), (np.array([-300.0, -300.0]), np.array([300.0, 300.0]))
    places = scatter_places(4, np.random.default_rng(1), room, [np.zeros(2)], box=box)
    pts = np.array([(u, v) for u, v, _ in places] + [(0.0, 0.0)])
    assert len(places) == 4 and min(np.hypot(*(pts[i] - pts[j])) for i in range(5) for j in range(i)) >= 90.0 and np.all(np.abs(pts[:, 0]) <= 220.0)
    assert len(scatter_places(40, np.random.default_rng(1), room, [], box=box)) < 30  # a table that is full gives fewer


def test_a_tower_is_built_with_a_look_after_every_block(ses):
    lay(ses, {"slab": ("blue slab", 420.0, 90.0, 100.0), "bar": ("green bar", 470.0, -110.0, 75.0), "cube": ("yellow cube", 400.0, -40.0, 10.0)})
    ses.build_watched("T")
    assert finished(ses) == "done: a tower of 3", (ses.job["outcome"], list(ses.lines)[-8:])
    drop = ses.P.drop_mm
    assert on_the_tower(ses, "slab", 0.0) and on_the_tower(ses, "cube", 15.0 + drop) and on_the_tower(ses, "bar", 45.0 + drop), ses.arm.bricks  # widest first
    assert ses.camera.n == 4 and ses.stack["n"] == 3 and ses.stack["of"] == 3  # a look before each block and one at the finished tower
    assert [lv["name"] for lv in ses.tower_levels] == ["blue slab", "yellow cube", "green bar"]
    looks = sorted((ses.log_dir / "looks").glob("*.jpg"))
    assert len(looks) == 4 and len(list((ses.log_dir / "looks").glob("*.json"))) == 4  # every picture is kept, with what was made of it
    ses.scatter("T", seed=3)  # and taken apart again, the blocks laid somewhere else
    assert finished(ses) == "done", (ses.job["outcome"], list(ses.lines)[-8:])
    table, base = ses.setup.table, ses.setup.table.coords(ses.spots.grasp("T"))[:2]
    at = {k: table.coords(p) for k, p in ses.arm.bricks.items()}
    assert all(abs(c[2] - ses.setup.w_grasp) < 2.8 and np.hypot(*(c[:2] - base)) >= 88.0 and ses.setup.inside(c[:2]) for c in at.values()), at
    assert min(np.hypot(*(at[i][:2] - at[j][:2])) for i in at for j in at if i < j) >= 88.0 and ses.stack["n"] == 0
    with pytest.raises(Refused, match="no tower"):
        ses.scatter("T")


def test_a_block_that_falls_off_is_seen_and_taken_again_from_where_it_lies(ses):
    lay(ses, {"slab": ("blue slab", 420.0, 90.0, 100.0), "bar": ("green bar", 470.0, -110.0, 75.0), "cube": ("yellow cube", 400.0, -40.0, 10.0)})
    a, setup, fell = ses.arm, ses.setup, []

    def slip(scene):  # the cube slides off the slab once, and lies on the table somewhere else
        if not fell and on_the_tower(ses, "cube", 15.0 + ses.P.drop_mm) and a.holding is None:
            uv = setup.table.coords([400.0, -40.0, 0.0])[:2]
            a.bricks["cube"] = setup.table.point(uv[0], uv[1], setup.w_grasp)
            fell.append(scene.n)

    ses.camera.then = slip
    ses.build_watched("T")
    assert finished(ses) == "done: a tower of 3", (ses.job["outcome"], list(ses.lines)[-8:])
    assert fell and any("taken for fallen" in line for line in ses.lines)
    drop = ses.P.drop_mm
    assert on_the_tower(ses, "slab", 0.0) and on_the_tower(ses, "cube", 15.0 + drop) and on_the_tower(ses, "bar", 45.0 + drop), a.bricks
    assert ses.camera.n == 5  # one look more than without the slip


def test_the_next_block_is_aimed_back_towards_the_towers_own_axis(ses):
    lay(ses, {"slab": ("blue slab", 420.0, 90.0, 100.0), "cube": ("yellow cube", 400.0, -40.0, 10.0)})
    a, setup, moved = ses.arm, ses.setup, []

    def nudge(scene):  # the slab is 8 mm off where it was put
        if not moved and a.holding is None and np.linalg.norm(a.bricks["slab"] - ses.spots.grasp("T")) < 1.0:
            a.bricks["slab"] = a.bricks["slab"] + setup.table.point(8.0, 0.0, 0.0)
            moved.append(1)

    ses.camera.then = nudge
    ses.build_watched("T")
    assert finished(ses) == "done: a tower of 2", (ses.job["outcome"], list(ses.lines)[-8:])
    off = setup.table.coords(a.bricks["cube"] - ses.spots.grasp("T"))[:2]
    assert moved and abs(off[0] - 4.0) < 1.5 and abs(off[1]) < 1.5, off  # half way between where the slab stands and where the tower should stand


def test_a_block_of_another_width_than_the_camera_said_is_let_go_again(ses):
    lay(ses, {"slab": ("blue slab", 420.0, 90.0, 100.0), "cube": ("yellow cube", 400.0, -40.0, 10.0)})
    a = ses.arm
    a.widths["cube"], cube = 600, a.bricks["cube"].copy()  # the jaws stop 60 mm apart on what the camera took for a 30 mm cube
    ses.build_watched("T")
    out = finished(ses)
    assert out.startswith("gave up after three failures"), out
    assert on_the_tower(ses, "slab", 0.0) and np.allclose(a.bricks["cube"], cube) and a.holding is None and a.counts >= 800
    assert sum("the block that was meant is 30 mm across" in line for line in ses.lines) == 3
    assert ses.phase() == "ARMED"  # giving up is not a fault: the session goes on
    ses.check_width = False  # ... and the check can be switched off (a gripper whose counts are not millimetres)
    ses.build_watched("T", tower=ses.tower_levels)
    assert finished(ses) == "done: a tower of 2" and on_the_tower(ses, "cube", 15.0 + ses.P.drop_mm)


def test_a_block_the_jaws_cannot_get_at_is_pushed_clear_first(ses):
    lay(ses, {"bar": ("green bar", 430.0, 20.0, 0.0), "red": ("red bar", 472.0, 20.0, 0.0)})  # two bars side by side: a finger would come down on the other one
    a = ses.arm
    a.pushes = True
    ses.build_watched("T", push=False)
    assert "no room beside" in finished(ses) and ses.camera.n == 1 and len(a.bricks) == 2
    ses.build_watched("T")
    assert finished(ses) == "done: a tower of 2", (ses.job["outcome"], list(ses.lines)[-8:])
    notes = [e for e in ses.guard.events if e["kind"] == "move" and e["what"] == "push"]
    assert notes and any("clear of its neighbour" in line for line in ses.lines)
    assert all(e["speed"] <= ses.P.slow for e in notes) and max(float(np.linalg.norm(e["d"])) for e in notes) <= 3.0 + 1e-6  # a push is watched: 3 mm at a time
    drop = ses.P.drop_mm
    assert sorted(round(float((p - ses.spots.grasp("T")) @ ses.spots.up), 1) for p in a.bricks.values()) == [0.0, 25.0 + drop]


def test_a_push_that_meets_something_that_does_not_slide_ends_there(ses):
    lay(ses, {"bar": ("green bar", 430.0, 20.0, 0.0), "red": ("red bar", 472.0, 20.0, 0.0)})
    a = ses.arm
    a.blocked = lambda p: a.counts < 800 and float(p @ ses.spots.up) < float(ses.spots.grasp("A") @ ses.spots.up) + 5.0 and abs(p[1] - 20.0) < 50.0  # the bar is stuck down
    ses.build_watched("T")
    out = finished(ses)
    assert "no room beside" in out or out.startswith("gave up"), out  # pushed as often as allowed, to no avail: a person is asked
    assert ses.phase() == "ARMED" and ses.guard.latched is None and any(e["kind"] == "blocked" and e["what"] == "push" for e in ses.guard.events)
    assert float(a.p @ ses.spots.up) > float(ses.spots.grasp("A") @ ses.spots.up) + 50.0  # the arm is up in the air again
