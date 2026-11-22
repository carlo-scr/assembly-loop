"""The camera in a session: the stand-in arm, watched by a made-up camera that draws the stand-in blocks."""
import threading
import time

import cv2
import numpy as np
import pytest

from assemblyloop.arms import FakeArm, rpy_matrix, twist_deg
from assemblyloop.blocks import Catalogue, Kind
from assemblyloop.eyes import PAINT, Camera, Table, outline, solid
from assemblyloop.guard import Refused
from assemblyloop.session import Session
from assemblyloop.sight import spin

CAT = Catalogue([Kind("green bar", "box", (75.0, 25.0, 25.0), "green"), Kind("yellow cylinder", "cylinder", (25.0, 75.0), "yellow"),
                 Kind("blue slab", "box", (60.0, 40.0, 15.0), "blue"), Kind("yellow cube", "box", (30.0, 30.0, 30.0), "yellow")]).check()
BAR = CAT.kind("green bar").rests()[0]


def until(cond, secs):
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        if cond():
            return True
        time.sleep(0.01)
    return False


class Scene:
    """Draws every stand-in block where the stand-in arm has it. A block's place in the arm's world is where the
    gripper is when it holds that block: its bottom is that much lower (`hold`), and a low block's middle is there."""

    def __init__(self, ses, size=(1920, 1080)):
        eye, at = np.array([1350.0, -350.0, 900.0]), np.array([460.0, 0.0, 120.0])
        z = (at - eye) / np.linalg.norm(at - eye)
        x = np.cross(z, [0.0, 0.0, 1.0])
        x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z])
        self.ses, self.cam, self.what, self.n = ses, Camera(size, 0.7 * size[0], (-0.18, 0.04), cv2.Rodrigues(R)[0].ravel(), -R @ eye), {"thing": ("green bar", 0, 30.0)}, 0

    def grab(self):
        self.n += 1
        axes = self.ses.spots.limits().axes
        im = np.full((self.cam.size[1], self.cam.size[0], 3), (110, 100, 95), np.uint8)
        for name, p in self.ses.arm.bricks.items():
            kind, rest, angle = self.what[name]
            rest, c = CAT.kind(kind).rests()[rest], axes.T @ p
            bottom = c[2] - max(0.0, rest.height - CAT.finger_mm) - BAR.height / 2.0
            hull = outline(self.cam, solid(rest, Table(axes, bottom), c[0], c[1], angle, CAT.kind(kind).shape))
            cv2.fillConvexPoly(im, np.round(hull * 16).astype(np.int32), PAINT[CAT.kind(kind).colour], shift=4)
        return im


@pytest.fixture
def ses(tmp_path):
    s = Session({"stand-in": dict(label="Stand-in arm (no robot)", ip=None, make=FakeArm)}, spots_path=tmp_path / "spots.json", log_dir=tmp_path / "logs", probe=lambda ip: False,
                fake_speed=40.0, guard_kw=dict(poll_s=0.002), limit_kw=dict(settle_s=0.0), camera_path=tmp_path / "camera.json", blocks_path=tmp_path / "blocks.json")
    s.camera, s.alive, s.pressed = Scene(s), True, False

    def grip():
        while s.alive:
            if s.dm is not None:
                s.dm.feed(s.pressed)
            time.sleep(0.02)

    threading.Thread(target=grip, daemon=True).start()
    s.start("stand-in")
    assert until(lambda: all(c["ok"] for c in s.state()["checks"]), 6)
    s.arm_on()
    s.use_demo()
    yield s
    s.alive = False
    s.close()


def test_the_gripper_is_spun_about_up_and_keeps_its_tilt():
    rpy, up = np.array([180.0, 0.0, 20.0]), np.array([0.0, 0.0, 1.0])
    for deg in (-90.0, -30.0, 0.0, 45.0):
        R = rpy_matrix(spin(rpy, up, deg))
        assert twist_deg(rpy_matrix(rpy), R)[1] < 1e-6  # its axis has not moved
        want = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg)), 0.0])  # a direction fixed to the gripper, seen from above
        assert np.allclose(R @ rpy_matrix(rpy).T @ [1.0, 0.0, 0.0], want, atol=1e-9)


def test_without_a_catalogue_or_before_the_camera_was_shown_the_table_nothing_is_built(ses):
    assert "blocks.json" in ses.state()["eye"]["why"]
    with pytest.raises(Refused, match="blocks.json"):
        ses.show_table("green bar")
    CAT.save(ses.blocks_path)
    assert ses.state()["eye"] == dict(ready=False, why="the camera has not been shown the table yet", kinds=[k.name for k in CAT.kinds], bars=["green bar", "blue slab", "yellow cube"],
                                      shown=None, fit_mm=None, sight=None)
    for call, why in ((ses.look, "not been shown"), (lambda: ses.build_seen("T"), "not been shown"), (lambda: ses.show_table("yellow cylinder"), "a bar or a slab"),
                      (lambda: ses.show_table("red thing"), "no kind of block")):
        with pytest.raises(Refused, match=why):
            call()
    ses.thing_at = None
    with pytest.raises(Refused, match="on which spot"):
        ses.show_table("green bar")


def test_the_camera_is_shown_the_table_then_sees_blocks_and_a_tower_is_built_from_them(ses):
    CAT.save(ses.blocks_path)
    a, sp, scene = ses.arm, ses.spots, ses.camera
    ses.pressed = True
    ses.show_table("green bar")
    assert until(lambda: ses.worker is None, 120), ses.state()["job"]
    assert ses.job["outcome"] == "done", (ses.job["outcome"], list(ses.lines)[-4:])
    setup = ses.setup
    assert setup is not None and ses.camera_path.exists() and setup.fit_mm < 2.0 and scene.n >= 10, (setup, scene.n)
    assert ses.thing_at == "A" and np.linalg.norm(a.bricks["thing"] - sp.grasp("A")) < 0.6  # the block is back where it lay
    assert abs(setup.angle_ref - 30.0) < 2.0 and np.allclose(setup.rpy_ref, sp.turn("A"))
    assert all(setup.inside(setup.table.coords(p)[:2]) for p in sp.at.values())
    eye = ses.state()["eye"]
    assert eye["ready"] and eye["fit_mm"] < 2.0 and eye["shown"]

    # the mat as a person would set it: blocks lying anywhere between the taught spots, turned any way
    lay = {"slab": ("blue slab", 0, 430.0, 40.0, 100.0), "bar": ("green bar", 0, 445.0, -80.0, 75.0), "cyl": ("yellow cylinder", 0, 410.0, -30.0, 0.0), "lying": ("yellow cylinder", 1, 490.0, 70.0, 20.0)}
    table = setup.table
    a.holding, a.bricks = None, {}
    for name, (kind, rest, x, y, angle) in lay.items():
        r = CAT.kind(kind).rests()[rest]
        uv = table.coords([x, y, 0.0])[:2]
        a.bricks[name] = table.point(uv[0], uv[1], setup.w_grasp + max(0.0, r.height - CAT.finger_mm))
        scene.what[name] = (kind, rest, angle)
    with pytest.raises(Refused, match="Look first"):
        ses.build_seen("T")
    seen = {s.name: s for s in ses.look()["seen"]}
    assert sorted(seen) == ["blue slab", "green bar", "yellow cylinder 1", "yellow cylinder 2"] and not ses.sight["unknown"]
    for name, got in (("slab", "blue slab"), ("bar", "green bar")):
        want = table.coords(a.bricks[name])[:2]
        assert np.hypot(*(np.array(seen[got].uv) - want)) < 2.0, (name, seen[got].uv, want)
    assert abs(seen["green bar"].angle - 75.0) < 3.0
    eye = ses.state()["eye"]["sight"]
    assert eye["order"] == ["blue slab", "green bar", [n for n in seen if "cylinder" in n and seen[n].rest.round][0]] and eye["height_mm"] == 115.0
    assert len(eye["left_out"]) == 1 and "roll" in eye["left_out"][0]
    assert ses.sight_jpg()[:2] == b"\xff\xd8"

    turns = len([e for e in a.log if e[0] == "turn"])
    ses.build_seen("T")
    assert until(lambda: ses.worker is None, 120), ses.state()["job"]
    assert ses.job["outcome"] == "done" and ses.stack["n"] == 3 and ses.stack["of"] == 3, (ses.job["outcome"], ses.stack, list(ses.lines)[-6:])
    base, drop = sp.grasp("T"), ses.P.drop_mm
    for name, level_mm in (("slab", 0.0), ("bar", 15.0 + drop), ("cyl", 40.0 + 35.0 + drop)):  # the cylinder is held 35 mm above its bottom
        d = a.bricks[name] - base
        assert abs(float(d @ sp.up) - level_mm) < 0.6 and np.linalg.norm(d - (d @ sp.up) * sp.up) < 2.5, (name, d)
    assert np.linalg.norm(a.bricks["lying"] - table.point(*table.coords([490.0, 70.0, 0.0])[:2], setup.w_grasp)) < 1e-6  # the lying cylinder was not touched
    assert len([e for e in a.log if e[0] == "turn"]) > turns  # the gripper was turned to take the bar across its short side
    with pytest.raises(Refused, match="Look first"):  # what the camera saw is no longer what is there
        ses.build_seen("T")
    ses.look()
    with pytest.raises(Refused, match="where the tower goes"):
        ses.build_seen("T")


def test_the_pages_camera_buttons_and_picture(ses):
    import json
    import urllib.error
    import urllib.request

    from assemblyloop.dashboard import make_server

    CAT.save(ses.blocks_path)
    srv = make_server(ses, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(path, body, token=srv.token):
        req = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json", "X-Token": token})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        page = urllib.request.urlopen(url + "/").read().decode()
        assert all(x in page for x in ('id="lookbtn"', 'id="e_build"', 'id="e_show"', "/api/build_seen"))
        assert post("/api/look", {}, token="wrong")[0] == 403
        code, j = post("/api/look", {})
        assert code == 409 and "not been shown" in j["error"]
        code, j = post("/api/build_seen", dict(on="T"))
        assert code == 409 and "not been shown" in j["error"]
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(url + "/api/sight.jpg")  # no picture yet
        ses.pressed = True
        assert post("/api/show_table", dict(kind="green bar", n=9)) == (200, dict(ok=True))
        assert until(lambda: ses.worker is None, 120) and ses.job["outcome"] == "done", ses.job
        assert post("/api/look", {}) == (200, dict(ok=True))
        assert urllib.request.urlopen(url + "/api/sight.jpg?t=1").read()[:2] == b"\xff\xd8"
        eye = json.loads(urllib.request.urlopen(url + "/api/state").read())["eye"]
        assert eye["ready"] and [b["name"] for b in eye["sight"]["seen"]] == ["green bar"] and eye["sight"]["order"] == ["green bar"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_the_table_is_found_by_coming_down_with_a_bar_and_the_camera_is_shown_it_from_there(tmp_path):
    s = Session({"stand-in": dict(label="Stand-in arm (no robot)", ip=None, make=FakeArm)}, spots_path=tmp_path / "spots.json", log_dir=tmp_path / "logs", probe=lambda ip: False,
                fake_speed=40.0, guard_kw=dict(poll_s=0.002), limit_kw=dict(settle_s=0.0), camera_path=tmp_path / "camera.json", blocks_path=tmp_path / "blocks.json")
    s.camera, s.alive, s.pressed = Scene(s), True, False
    CAT.save(s.blocks_path)

    def grip():
        while s.alive:
            if s.dm is not None:
                s.dm.feed(s.pressed)
            time.sleep(0.02)

    threading.Thread(target=grip, daemon=True).start()
    try:
        s.start("stand-in")
        a = s.arm
        a.p = a.plan = np.array([460.0, 0.0, 300.0])
        a.blocked = lambda p: p[2] < 120.0  # the bar in the jaws meets the table when the gripper is 120 mm up
        assert until(lambda: all(c["ok"] for c in s.state()["checks"]), 6)
        s.arm_on()
        s.pressed = True
        with pytest.raises(Refused, match="100 to 400"):
            s.find_table(50.0, 140.0)
        s.find_table()  # nothing is between the jaws
        assert until(lambda: s.worker is None, 20) and "nothing is between the jaws" in s.note and s.phase() == "AWAITING CALIBRATION", s.note
        a.bricks = {"thing": a.p.copy()}
        s.find_table(200.0, 140.0)
        assert until(lambda: s.worker is None, 60), s.state()["calibration"]
        assert s.phase() == "ARMED" and s.thing_at == "T" and sorted(s.spots.at) == ["P", "Q", "T"], (s.phase(), s.note)
        T = s.spots.grasp("T")
        assert np.allclose(T[:2], [460.0, 0.0], atol=0.1) and 120.0 <= T[2] <= 122.5, T  # just resting on the table, not pressed into it
        assert a.holding is None and np.linalg.norm(a.bricks["thing"] - T) < 0.1 and a.counts == 850  # the bar was let go there
        assert np.allclose(s.spots.grasp("Q") - s.spots.grasp("P"), [140.0, 200.0, 0.0], atol=0.1) and s.spots_path.exists()
        assert max(e[3] for e in a.log if e[0] == "move" and e[2][2] < -0.5) <= 15.0  # it came down at pressing speed only
        with pytest.raises(Refused, match="arm first"):
            s.find_table()  # not while ARMED
        a.blocked = None
        s.show_table("green bar")
        assert until(lambda: s.worker is None, 120) and s.job["outcome"] == "done", (s.job, list(s.lines)[-3:])
        assert s.setup is not None and s.setup.fit_mm < 2.0 and s.thing_at == "T" and s.state()["eye"]["ready"]
    finally:
        s.alive = False
        s.close()


def test_coloured_things_that_never_move_do_not_confuse_the_camera(ses):
    CAT.save(ses.blocks_path)
    scene, grab = ses.camera, ses.camera.grab

    def cluttered():  # two green blocks lie on the bench beside the mat, in every picture
        im = grab()
        cv2.rectangle(im, (150, 200), (230, 240), PAINT["green"], -1)
        cv2.rectangle(im, (200, 700), (260, 790), PAINT["green"], -1)
        return im

    scene.grab = cluttered
    ses.pressed = True
    ses.show_table("green bar")
    assert until(lambda: ses.worker is None, 120) and ses.job["outcome"] == "done", (ses.job["outcome"], list(ses.lines)[-3:])
    assert ses.setup.fit_mm < 2.0


def test_a_gripper_brought_to_the_table_by_hand_is_taken_as_the_table(tmp_path):
    s = Session({"stand-in": dict(label="Stand-in arm (no robot)", ip=None, make=FakeArm)}, spots_path=tmp_path / "spots.json", log_dir=tmp_path / "logs", probe=lambda ip: False,
                fake_speed=40.0, guard_kw=dict(poll_s=0.002), limit_kw=dict(settle_s=0.0), camera_path=tmp_path / "camera.json", blocks_path=tmp_path / "blocks.json")
    alive = [True]
    threading.Thread(target=lambda: [(s.dm and s.dm.feed(True if s.gate and s.gate.armed else False), time.sleep(0.02)) for _ in iter(lambda: alive[0], False)], daemon=True).start()
    try:
        s.start("stand-in")
        a = s.arm
        a.p = a.plan = np.array([460.0, 0.0, 120.0])  # the fingertips touch the table here
        a.blocked = lambda p: p[2] < 120.0
        a.bricks = {"thing": a.p.copy()}
        assert until(lambda: all(c["ok"] for c in s.state()["checks"]), 6)
        s.arm_on()
        s.dm.feed(False)
        s.find_table(200.0, 140.0, here=True)
        assert until(lambda: s.worker is None, 30), s.state()["calibration"]
        assert s.phase() == "ARMED" and s.thing_at == "T" and "where the gripper was" in s.note, (s.phase(), s.note)
        assert np.allclose(s.spots.grasp("T"), [460.0, 0.0, 121.5], atol=0.05) and a.holding is None
        assert not [e for e in a.log if e[0] == "move" and e[2][2] < -1e-6]  # nothing came down
        assert s.state()["quest"]["gap_ms"] < 200
    finally:
        alive[0] = False
        s.close()
