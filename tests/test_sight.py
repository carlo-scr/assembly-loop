"""The camera in a session: the stand-in arm, watched by a made-up camera that draws the stand-in blocks."""
import json
import re
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
            angle += self.ses.arm.turned.get(name, 0.0)  # turned with the gripper while it was held
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
    assert setup.inside(setup.table.coords(sp.grasp("A"))[:2]) and not setup.inside(setup.table.coords(sp.grasp("A") + [-200.0, 0.0, 0.0])[:2])  # around where the block lay, not behind it
    eye = ses.state()["eye"]
    assert eye["ready"] and eye["fit_mm"] < 2.0 and eye["shown"]

    # the mat as a person would set it: blocks lying anywhere between the taught spots, turned any way
    lay = {"slab": ("blue slab", 0, 450.0, -30.0, 100.0), "bar": ("green bar", 0, 470.0, -150.0, 75.0), "cyl": ("yellow cylinder", 0, 400.0, -170.0, 0.0), "lying": ("yellow cylinder", 1, 550.0, -150.0, 20.0)}
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
    assert np.linalg.norm(a.bricks["lying"] - table.point(*table.coords([550.0, -150.0, 0.0])[:2], setup.w_grasp)) < 1e-6  # the lying cylinder was not touched
    assert len([e for e in a.log if e[0] == "turn"]) > turns  # the gripper was turned to take the bar across its short side
    # the camera checked every block once it was put down, from out of its way, and kept every picture
    checks = [json.loads(line) for line in open(ses.log_dir / "moves.jsonl") if '"check"' in line]
    assert [c["check"] for c in checks] == [1, 2, 3] and not any(c["problems"] for c in checks), checks
    assert all((ses.log_dir / "pictures" / c["picture"]).exists() for c in checks) and [len(c["blocks"]) for c in checks] == [1, 2, 3]
    assert all(b["fit"] > 0.9 and b["off_mm"] < 4.0 and abs(b["turn_deg"]) < 3.0 for c in checks for b in c["blocks"] if b["visible"] > 0.3), checks
    # its last picture was taken after the last block: the next job may start from it, and sees the tower in the way
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
        for path in ("/", "/classic"):  # the page that fits one screen (its buttons are wired in web/app.js) and the older one
            page = urllib.request.urlopen(url + path).read().decode() + (urllib.request.urlopen(url + "/web/app.js").read().decode() if path == "/" else "")
            assert all(x in page for x in ('id="lookbtn"', 'id="e_build"', 'id="e_show"', "/api/build_seen")), path
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
        with pytest.raises(Refused, match="the gripper is open"):
            s.find_table()
        a.bricks = {"thing": a.p.copy()}
        s.grip(True, 25.0)  # closed on the bar, to just under its width
        assert until(lambda: s.worker is None, 20) and "closed to 22 mm" in s.note and a.holding is not None and [e[2] for e in a.log if e[0] == "grip"] == [225], s.note
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
        a.bricks, a.holding, a.counts = {"thing": a.p.copy()}, ("thing", np.zeros(3)), 315
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


def test_a_wrong_first_picture_a_block_still_in_the_gripper_and_a_gentle_grip(ses):
    CAT.save(ses.blocks_path)
    a, scene, grab, n = ses.arm, ses.camera, ses.camera.grab, [0]
    a.counts = 160  # the jaws are closed on something
    with pytest.raises(Refused, match="not open"):
        ses.show_table("green bar")
    with pytest.raises(Refused, match="not open"):
        ses.move("B")
    a.counts = 850

    def first_wrong():  # in the first picture the block is not where the arm thinks it is
        n[0] += 1
        if n[0] > 1:
            return grab()
        keep = a.bricks["thing"].copy()
        a.bricks["thing"] = keep + np.array([60.0, 90.0, 0.0])
        try:
            return grab()
        finally:
            a.bricks["thing"] = keep

    scene.grab = first_wrong
    a.BRICK_COUNTS = 150  # these jaws bend: closed as hard as they go they read 15 mm on a 25 mm bar
    ses.pressed = True
    ses.show_table("green bar")
    assert until(lambda: ses.worker is None, 120) and ses.job["outcome"] == "done", (ses.job["outcome"], list(ses.lines)[-3:])
    assert ses.setup.fit_mm < 2.0 and "fitted to 12 places" in " ".join(ses.lines)  # 13 pictures, the wrong one left out
    shots = ses.log_dir / "camera_shots"
    assert len(list(shots.glob("*.jpg"))) == 13 and len(__import__("json").loads((shots / "places.json").read_text())["xyz"]) == 13
    grips = [e[2] for e in a.log if e[0] == "grip"]
    assert grips.count(225) >= 12 and 0 not in grips  # every pick closes to 2.5 mm under the bar's 25 mm, and never as hard as the jaws go


def test_the_area_is_put_somewhere_else_at_the_table_height_found_before(tmp_path):
    s = Session({"stand-in": dict(label="Stand-in arm (no robot)", ip=None, make=FakeArm)}, spots_path=tmp_path / "spots.json", log_dir=tmp_path / "logs", probe=lambda ip: False,
                fake_speed=40.0, guard_kw=dict(poll_s=0.002), limit_kw=dict(settle_s=0.0), camera_path=tmp_path / "camera.json", blocks_path=tmp_path / "blocks.json")
    alive = [True]
    threading.Thread(target=lambda: [(s.dm and s.dm.feed(bool(s.gate and s.gate.armed)), time.sleep(0.02)) for _ in iter(lambda: alive[0], False)], daemon=True).start()
    try:
        s.start("stand-in")
        a = s.arm
        a.p = a.plan = np.array([460.0, 0.0, 121.5])
        a.bricks, a.holding, a.counts = {"thing": a.p.copy()}, ("thing", np.zeros(3)), 315
        assert until(lambda: all(c["ok"] for c in s.state()["checks"]), 6)
        s.arm_on()
        s.dm.feed(False)
        with pytest.raises(Refused, match="no table height from before"):
            s.find_table(as_before=True)
        s.find_table(200.0, 140.0, here=True)  # the table is found once: T at 123 mm
        assert until(lambda: s.worker is None, 30) and s.phase() == "ARMED"
        w = float(s.spots.grasp("T")[2])
        s.calibrate_begin()  # ... and now the area is wanted 150 mm further on, the bar in the jaws, the gripper 60 mm up
        for name in list(s.draft.at):
            s.remove_spot(name)
        a.p = a.plan = np.array([610.0, -40.0, w + 60.0])
        a.bricks, a.holding, a.counts = {"thing": a.p.copy()}, ("thing", np.zeros(3)), 315
        s.dm.feed(False)
        s.find_table(200.0, 140.0, as_before=True)
        assert until(lambda: s.worker is None, 30), s.state()["calibration"]
        assert s.phase() == "ARMED" and "height from before" in s.note and s.thing_at == "T", (s.phase(), s.note)
        assert np.allclose(s.spots.grasp("T"), [610.0, -40.0, w], atol=0.05) and a.holding is None and np.linalg.norm(a.bricks["thing"] - s.spots.grasp("T")) < 0.1
    finally:
        alive[0] = False
        s.close()


def test_the_places_go_only_where_the_arm_reaches_and_the_mat_is(ses):
    CAT.save(ses.blocks_path)
    a, sp, scene, grab = ses.arm, ses.spots, ses.camera, ses.camera.grab
    home = sp.grasp("A").copy()  # (430, -90): forward is +x, left is +y

    def with_wood():  # the grey mat ends 75 mm to the left of the block: bare wood from there on
        im = grab()
        axes = sp.limits().axes
        far = [axes.T @ np.array([x, y, home[2] - BAR.height / 2.0]) for x, y in ((0.0, -15.0), (1200.0, -15.0), (1200.0, 900.0), (0.0, 900.0))]
        wood = np.round(scene.cam.project([axes @ q for q in far])).astype(np.int32)
        clean = im.copy()
        cv2.fillPoly(im, [wood], (114, 165, 183))
        coloured = cv2.cvtColor(clean, cv2.COLOR_BGR2HSV)[..., 1] > 150  # the block stays on top of whatever it lies on
        im[coloured] = clean[coloured]
        return im

    scene.grab = with_wood
    a.reach_mm = float(np.linalg.norm(home + [130.0, 0.0, 80.0]))  # ... and the arm reaches about two places further than the block
    ses.pressed = True
    ses.show_table("green bar", n=14)
    assert until(lambda: ses.worker is None, 180), ses.state()["job"]
    assert ses.job["outcome"] == "done", (ses.job["outcome"], list(ses.lines)[-4:])
    used = np.array([p - home for k, p in ses.job["spots"].at.items() if k.startswith("place")])
    assert 9 <= len(used) <= 14 and ses.setup.fit_mm < 2.5, (len(used), ses.setup.fit_mm)
    assert all(np.linalg.norm(home + d + [0.0, 0.0, 80.0]) <= a.reach_mm + 1e-6 for d in used)  # nowhere beyond the arm's reach
    assert used[:, 1].max() <= 60.0 + 1e-6 and (used[:, 1] > 1.0).sum() <= 2  # hardly anything on the wood (the first places are tried before the mat is known)
    assert "off the grey mat is left out" in " ".join(ses.lines)
    assert used[:, 0].min() >= -60.0 - 1e-6  # and nothing behind the block, where the arm waits for the camera
    assert np.linalg.norm(a.bricks["thing"] - home) < 0.6 and ses.thing_at == "A"


def test_only_the_tab_opened_last_is_the_switch(ses):
    import json
    import urllib.request

    from assemblyloop.dashboard import make_server

    srv = make_server(ses, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    hold = lambda page, pressed: urllib.request.urlopen(urllib.request.Request(url + "/api/hold", json.dumps(dict(pressed=pressed, page=page, t=0)).encode(), {"Content-Type": "application/json", "X-Token": srv.token})).read()  # noqa: E731
    try:
        assert "window.PAGE_ID = 1;" in urllib.request.urlopen(url + "/").read().decode() and "PAGE_ID=2," in urllib.request.urlopen(url + "/classic").read().decode()
        assert json.loads(urllib.request.urlopen(url + "/api/state").read())["page"] == 2
        ses.alive = False  # the fixture's own stand-in for the switch is silenced: only the two tabs report now
        time.sleep(0.1)
        hold(2, False)
        hold(2, True)
        assert ses.gate.held()
        hold(1, False)  # the old tab, still open somewhere, says "off": it is not listened to
        assert ses.gate.held()
        hold(2, False)
        assert not ses.gate.held()
    finally:
        srv.shutdown()
        srv.server_close()


def test_the_arm_finds_where_to_wait_so_that_it_does_not_hide_the_table(ses):
    CAT.save(ses.blocks_path)
    a, sp, scene, grab = ses.arm, ses.spots, ses.camera, ses.camera.grab
    home = sp.grasp("A").copy()

    def with_arm():  # the arm hides the table under and ahead of where it is, as it does from a camera that stands beside it
        im = grab()
        under = np.array([a.p[0] + 150.0, a.p[1], home[2] - BAR.height / 2.0])
        px = scene.cam.project([under])[0]
        cv2.circle(im, (int(px[0]), int(px[1])), 300, (245, 245, 245), -1)
        return im

    scene.grab = with_arm
    ses.pressed = True
    ses.show_table("green bar")
    assert until(lambda: ses.worker is None, 240), ses.state()["job"]
    assert ses.job["outcome"] == "done", (ses.job["outcome"], list(ses.lines)[-4:])
    line = next(x for x in ses.lines if "the arm will wait" in x)
    assert ("to its right" in line or "to its left" in line) and " 0 mm to its" not in line, line  # out to the side, not straight back
    assert "fitted to 1" in " ".join(ses.lines) and ses.setup.fit_mm < 2.5  # ... and from there the block is seen in (nearly) every place


def test_the_live_pictures_know_the_block_finders_camera_and_the_stand_in_gets_a_drawn_one():
    from assemblyloop import eyes, feeds

    assert feeds.TABLE_CAMERA == eyes.SCENE_CAMERA  # the tile that says "finds the blocks" is the camera `look` opens
    assert len({c["device"] for c in feeds.rig_cameras()}) == 6 and sum("right" in c["arms"] for c in feeds.rig_cameras()) == 4
    axes = np.eye(3)
    world = dict(axes=axes, centre=None, gripper=np.array([450.0, 0.0, 300.0]), open=True, spots={"A": np.array([430.0, -90.0, 120.0])}, things={"thing": np.array([430.0, -90.0, 120.0])})
    for w in (None, world):
        ok, im = feeds.MadeUp(lambda w=w: w, fps=200.0).read()
        assert ok and im.shape == (480, 640, 3) and im.dtype == np.uint8
    jpg, size = feeds.to_jpeg(np.zeros((1080, 1920, 3), np.uint8))  # a large picture is made smaller for the page
    assert jpg[:2] == b"\xff\xd8" and size == (960, 540)


def lay_out(ses, blocks):
    """Put stand-in blocks on the mat: {name: (kind, rest index, x, y, angle)} in the arm's base frame; the camera draws them."""
    a, setup = ses.arm, ses.setup
    a.holding, a.bricks, a.turned = None, {}, {}
    for name, (kind, rest, x, y, angle) in blocks.items():
        r = CAT.kind(kind).rests()[rest]
        uv = setup.table.coords([x, y, 0.0])[:2]
        a.bricks[name] = setup.table.point(uv[0], uv[1], setup.w_grasp + max(0.0, r.height - CAT.finger_mm))
        a.widths[name] = r.short
        ses.camera.what[name] = (kind, rest, angle)


def shown(ses):
    """The camera is shown the table from the spot in the middle (B), with enough places that every taught spot is in the area."""
    CAT.save(ses.blocks_path)
    ses.arm.bricks["thing"] = ses.spots.grasp("B").copy()
    ses.set_thing("B")
    ses.pressed = True
    ses.show_table("green bar", n=20)
    assert until(lambda: ses.worker is None, 120) and ses.job["outcome"] == "done"


def test_a_block_pushed_on_the_mat_is_taken_where_it_lies_now_and_turned_as_it_lies(ses):
    shown(ses)
    lay_out(ses, {"slab": ("blue slab", 0, 430.0, 70.0, 100.0), "bar": ("green bar", 0, 445.0, -95.0, 30.0)})
    ses.look()
    a, moved = ses.arm, []

    def push(k):  # after the slab is on the tower, someone pushes the bar 12 mm and turns it by 25 degrees
        moved.append(k)
        if k == 0:
            a.bricks["bar"] = a.bricks["bar"] + ses.setup.table.axes[:, 0] * 12.0
            a.turned["bar"] = 25.0

    real = ses._check_built
    ses._check_built = lambda job, k, at: (push(k), real(job, k, at))
    ses.build_seen("T")
    assert until(lambda: ses.worker is None, 120)
    assert ses.job["outcome"] == "done" and ses.stack["n"] == 2, (ses.job["outcome"], list(ses.lines)[-6:])
    assert any("12 mm from where it was seen before" in line for line in ses.lines), list(ses.lines)[-12:]
    d = a.bricks["bar"] - ses.spots.grasp("T")
    assert np.linalg.norm(d - (d @ ses.spots.up) * ses.spots.up) < 2.5  # it ended on the tower all the same


def test_a_block_that_fell_off_stops_the_job_and_a_wrong_width_stops_the_pick(ses):
    shown(ses)
    lay_out(ses, {"slab": ("blue slab", 0, 430.0, 70.0, 100.0), "bar": ("green bar", 0, 445.0, -95.0, 30.0), "cube": ("yellow cube", 0, 395.0, -20.0, 10.0)})
    ses.look()
    a = ses.arm
    real = ses._check_built

    def knock(job, k, at):
        if k == 1:  # the second block of the tower (widest first: slab, cube, bar) slides off once it is put down
            a.bricks["cube"] = a.bricks["cube"] + ses.setup.table.axes[:, 1] * 45.0
        return real(job, k, at)

    ses._check_built = knock
    ses.build_seen("T")
    assert until(lambda: ses.worker is None, 120)
    out = ses.job["outcome"]
    assert re.match(r"the camera: the yellow cube \(step 2\) is 4[3-9] mm", out) and ses.stack["n"] is None and ses.phase() == "ARMED", (out, ses.stack)  # found 45 mm off; stopped, still armed
    assert np.linalg.norm(a.bricks["bar"] - ses.spots.grasp("T")) > 50.0  # the bar was not put onto it
    checks = [json.loads(line) for line in open(ses.log_dir / "moves.jsonl") if '"check"' in line]
    assert [c["check"] for c in checks] == [1, 2] and checks[-1]["problems"] and not checks[0]["problems"]
    # the jaws close to another width than the block meant: taken the wrong way round, say
    ses._check_built = real
    lay_out(ses, {"slab": ("blue slab", 0, 430.0, 70.0, 100.0), "bar": ("green bar", 0, 445.0, -95.0, 30.0)})
    a.widths["slab"] = 60.0
    ses.look()
    ses.build_seen("T")
    assert until(lambda: ses.worker is None, 120)
    assert "the jaws closed to 60 mm, but the block is 40 mm across" in ses.job["outcome"] and a.holding is None, ses.job["outcome"]


def test_without_the_check_no_picture_is_taken_between_blocks(ses):
    shown(ses)
    lay_out(ses, {"slab": ("blue slab", 0, 430.0, 70.0, 100.0), "bar": ("green bar", 0, 445.0, -95.0, 30.0)})
    ses.look()
    n = ses.camera.n
    ses.build_seen("T", check=False)
    assert until(lambda: ses.worker is None, 120) and ses.job["outcome"] == "done"
    assert ses.camera.n == n and not (ses.log_dir / "pictures").exists()


def test_the_check_pictures_are_taken_from_where_the_arm_waited_when_the_camera_was_shown_the_table(ses):
    shown(ses)
    setup = ses.setup
    assert setup.wait is not None and eyes_setup_again(ses).wait == pytest.approx(setup.wait)  # kept in camera.json
    wait = setup.table.point(setup.wait[0], setup.wait[1], setup.w_grasp)
    lay_out(ses, {"slab": ("blue slab", 0, 430.0, 70.0, 100.0), "bar": ("green bar", 0, 445.0, -95.0, 30.0)})
    ses.look()
    ses.build_seen("T")
    assert until(lambda: ses.worker is None, 120) and ses.job["outcome"] == "done", ses.job["outcome"]
    sp = ses.job["spots"]
    assert np.allclose(sp.grasp("wait"), wait)
    checks = [json.loads(line) for line in open(ses.log_dir / "moves.jsonl") if '"check"' in line]
    assert len(checks) == 2 and all(np.linalg.norm(np.array(c["arm"]) - sp.above(wait)) < 1.0 for c in checks), [c["arm"] for c in checks]  # both pictures from there


def eyes_setup_again(ses):
    from assemblyloop.eyes import Setup

    return Setup.load(ses.camera_path)


def test_blocks_are_put_onto_the_widest_one_where_it_lies_without_any_spot_for_it(ses):
    shown(ses)
    lay_out(ses, {"slab": ("blue slab", 0, 440.0, 60.0, 100.0), "bar": ("green bar", 0, 445.0, -95.0, 30.0), "cube": ("yellow cube", 0, 395.0, -20.0, 10.0)})
    ses.look()
    a = ses.arm
    slab = a.bricks["slab"].copy()
    ses.build_seen("*")
    assert until(lambda: ses.worker is None, 180)
    assert ses.job["outcome"] == "done" and ses.stack["n"] == 3 and ses.stack["on"] == "the blue slab", (ses.job["outcome"], ses.stack, list(ses.lines)[-5:])
    assert np.allclose(a.bricks["slab"], slab)  # the slab was never touched
    up, drop = ses.spots.up, ses.P.drop_mm
    for name, level in (("cube", 15.0 + drop), ("bar", 45.0 + drop)):  # widest first: the slab (15 mm), the cube (30 mm), the bar
        d = a.bricks[name] - slab
        assert abs(float(d @ up) - level) < 0.6 and np.linalg.norm(d - (d @ up) * up) < 3.0, (name, d)
    assert abs((a.turned.get("bar", 0.0) + 30.0 - 100.0 + 90.0) % 180.0 - 90.0) < 3.0  # the bar was turned to lie along the slab (which lies at 100 degrees)
    checks = [json.loads(line) for line in open(ses.log_dir / "moves.jsonl") if '"check"' in line]
    assert len(checks) == 2 and not any(c["problems"] for c in checks) and [len(c["blocks"]) for c in checks] == [2, 3]  # the slab is looked at too
    lay_out(ses, {"slab": ("blue slab", 0, 440.0, 60.0, 100.0)})
    ses.look()
    with pytest.raises(Refused, match="at least two"):
        ses.build_seen("*")


class Wrist:
    """A stand-in for the camera on the gripper: it looks down past the fingers. A block that is held shows in the
    middle whatever the gripper does; one that lies on the table shows smaller, and further off, the higher the gripper is."""

    def __init__(self, ses):
        self.ses, self.n = ses, 0

    def grab(self):
        self.n += 1
        a, im = self.ses.arm, np.full((360, 640, 3), (110, 100, 95), np.uint8)
        for name, at in a.bricks.items():
            colour = PAINT[CAT.kind(self.ses.camera.what[name][0]).colour]
            if a.holding is not None and a.holding[0] == name:
                cv2.rectangle(im, (260, 150), (380, 210), colour, -1)
                continue
            d = at - a.p
            far = 200.0 / (200.0 - float(d @ a.up))  # 200 mm from the camera to the fingertips
            x, y = 320 + 2.0 * far * d[0], 180 + 2.0 * far * d[1]
            cv2.rectangle(im, (int(x - 60 * far), int(y - 30 * far)), (int(x + 60 * far), int(y + 30 * far)), colour, -1)
        return im


def test_the_grippers_camera_sees_whether_the_block_came_along_and_the_jaws_close_further_if_not(ses):
    from assemblyloop.eyes import same_patch

    held, left = np.full((360, 640, 3), (110, 100, 95), np.uint8), np.full((360, 640, 3), (110, 100, 95), np.uint8)
    cv2.rectangle(held, (260, 150), (380, 210), PAINT["blue"], -1)
    cv2.rectangle(left, (278, 158), (362, 200), PAINT["blue"], -1)  # the same block, 80 mm further from the camera
    assert same_patch(held, held, "blue")[0] > 0.95 and same_patch(held, left, "blue")[0] < 0.55 and same_patch(held, held, "green") == (0.0, 0)

    shown(ses)
    ses.wrist_camera = Wrist(ses)
    lay_out(ses, {"slab": ("blue slab", 0, 430.0, 70.0, 100.0), "bar": ("green bar", 0, 445.0, -95.0, 30.0)})
    ses.look()
    a = ses.arm
    a.needs = 350  # jaws that are not sent to 35 mm or closer hold nothing: the slab (40 mm, first sent to 37.5) slips twice
    n = len(a.log)
    ses.build_seen("T")
    assert until(lambda: ses.worker is None, 180)
    assert ses.job["outcome"] == "done" and ses.stack["n"] == 2, (ses.job["outcome"], list(ses.lines)[-8:])
    closes = [e[2] for e in a.log[n:] if e[0] == "grip" and 0 < e[2] < 800]
    assert closes == [375, 360, 345, 225], closes  # the slab three times, each 1.5 mm further; the bar at once
    seen = [json.loads(line)["grip_seen"] for line in open(ses.log_dir / "moves.jsonl") if '"grip_seen"' in line]
    assert len(seen) == 4 and max(seen[:2]) < 0.55 and min(seen[2:]) > 0.9, seen
    assert sum("once more, the jaws closed" in line for line in ses.lines) == 2 and sum("come along" in line for line in ses.lines) == 2
    d = a.bricks["slab"] - ses.spots.grasp("T")
    assert np.linalg.norm(d - (d @ ses.spots.up) * ses.spots.up) < 2.5  # ... and both ended on the tower
    # jaws that never hold: three tries, then the job stops and says so
    lay_out(ses, {"slab": ("blue slab", 0, 430.0, 70.0, 100.0)})
    a.needs = 50
    ses.look()
    ses.build_seen("T")
    assert until(lambda: ses.worker is None, 180)
    assert "stay behind" in ses.job["outcome"] and "three times over" in ses.job["outcome"] and ses.phase() == "ARMED", ses.job["outcome"]
    # without a camera on the gripper nothing changes
    ses.wrist_camera, a.needs = None, None
    lay_out(ses, {"slab": ("blue slab", 0, 430.0, 70.0, 100.0)})
    ses.look()
    ses.build_seen("T")
    assert until(lambda: ses.worker is None, 180) and ses.job["outcome"] == "done"
