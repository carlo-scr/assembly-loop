"""The page that fits one screen: its files, the arm's pose for the model, what is drawn around the arm, and the live
pictures. Everything on the stand-in arm, with made-up cameras; none of it needs OpenCV."""
import json
import re
import struct
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from assemblyloop.arms import FakeArm, _frames, joints_for, rpy_matrix
from assemblyloop.dashboard import PAGE, WEB, make_server, web_files
from assemblyloop.feeds import Feeds
from assemblyloop.session import TOOL_MM, Session, mount_down


def until(cond, secs):
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        if cond():
            return True
        time.sleep(0.01)
    return False


class Cam:
    """A made-up camera: every picture is a small array that says which camera and which picture it is."""

    opened, closed = [], []

    def __init__(self, name):
        self.name, self.n = name, 0
        Cam.opened.append(name)

    def read(self):
        time.sleep(0.005)
        self.n += 1
        return True, np.full((4, 6, 3), self.n % 250, np.uint8)

    def release(self):
        Cam.closed.append(self.name)


def pack(im):
    return b"JPG" + bytes([int(im[0, 0, 0])]), (int(im.shape[1]), int(im.shape[0]))


def open_cam(device, size):
    if "broken" in str(device):
        raise OSError("does not open: another program is using it")
    return Cam(str(device))


@pytest.fixture
def cams(tmp_path):
    Cam.opened, Cam.closed = [], []
    specs = [dict(id="scene", label="Scene", device=str(tmp_path / "scene"), arms=("right", "left", "stand-in")), dict(id="wrist", label="Wrist", device=str(tmp_path / "wrist"), arms=("stand-in",)),
             dict(id="other", label="Other arm's", device=str(tmp_path / "other"), arms=("left",)), dict(id="broken", label="Broken", device=str(tmp_path / "broken"), arms=("stand-in",))]
    for s in specs:
        (tmp_path / s["id"]).write_text("")
    f = Feeds(specs, fps=100.0, idle_s=0.4, open=open_cam, encode=pack)
    yield f
    f.close()


@pytest.fixture
def ses(tmp_path, cams):
    s = Session({"right": dict(label="Right arm", ip="192.0.2.1", make=FakeArm, down=mount_down("right")), "stand-in": dict(label="Stand-in arm (no robot)", ip=None, make=FakeArm)},
                spots_path=tmp_path / "spots.json", log_dir=tmp_path / "logs", probe=lambda ip: False, fake_speed=40.0, guard_kw=dict(poll_s=0.002), limit_kw=dict(settle_s=0.0),
                camera_path=tmp_path / "camera.json", blocks_path=tmp_path / "blocks.json", feeds=cams, record=False)  # these tests count who opens which camera: the recorder would open them too
    s.alive, s.pressed = True, False

    def grip():
        while s.alive:
            if s.dm is not None:
                s.dm.feed(s.pressed)
            time.sleep(0.02)

    threading.Thread(target=grip, daemon=True).start()
    yield s
    s.alive = False
    s.close()


def armed(s):
    s.start("stand-in")
    assert until(lambda: all(c["ok"] for c in s.state()["checks"]), 6)
    s.arm_on()
    s.use_demo()
    return s


def serve(s):
    srv = make_server(s, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def get(url, token=None):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"X-Token": token} if token else {}), timeout=5) as r:
            return r.status, r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), ""


def pictures(body):
    n = struct.unpack(">I", body[:4])[0]
    head, at, out = json.loads(body[4:4 + n]), 4 + n, {}
    for c in head["cams"]:
        out[c["id"]], at = body[at:at + c["len"]], at + c["len"]
    assert at == len(body)
    return head, out


# ------------------------------------------------------------------------------------------------ the page and its files
def test_the_page_is_served_with_every_file_it_asks_for_and_nothing_else(ses):
    srv, url = serve(ses)
    try:
        code, page, kind = get(url + "/")
        page = page.decode()
        assert code == 200 and "html" in kind and srv.token in page and "__TOKEN__" not in page and "STOP SESSION" in page and "Start session" in page
        asked = set(re.findall(r'"/web/([^"]+)"', page))
        for name in ("app.js", "scene.js"):
            asked |= set(re.findall(r'"/web/([^"`$]+)"', (WEB / name).read_text()))
        asked |= {f"xarm7/{m}.stl" for m in re.findall(r'"((?:link_base|end_tool|base_link))"', (WEB / "scene.js").read_text())} | {f"xarm7/link{i}.stl" for i in range(1, 7)}
        asked |= {f"xarm7/{side}_{part}.stl" for side in ("left", "right") for part in ("outer_knuckle", "finger", "inner_knuckle")}
        assert {"app.css", "app.js", "scene.js", "vendor/three.module.min.js", "vendor/OrbitControls.js", "vendor/STLLoader.js", "xarm7/link_base.stl"} <= asked
        for name in sorted(asked):
            code, body, kind = get(f"{url}/web/{name}")
            assert code == 200 and len(body) > 100, name
            assert ("javascript" in kind) == name.endswith(".js") and ("css" in kind) == name.endswith(".css"), (name, kind)
        assert (WEB / "xarm7" / "LICENSE").exists() and (WEB / "vendor" / "LICENSE-three.txt").exists()  # other people's work, passed on with their terms
        for bad in ("/web/../session.py", "/web/%2e%2e/session.py", "/web/..%2Fsession.py", "/web/index.html", "/web/", "/web/nothing.js", "/session.py"):
            assert get(url + bad)[0] == 404, bad
        assert all(".." not in k and not k.startswith("/") for k in web_files())
    finally:
        srv.shutdown()
        srv.server_close()


def test_the_older_page_is_still_there_and_the_new_one_has_all_its_buttons(ses):
    srv, url = serve(ses)
    try:
        code, classic, _ = get(url + "/classic")
        assert code == 200 and srv.token in classic.decode() and "Show it the places" in classic.decode()
    finally:
        srv.shutdown()
        srv.server_close()
    new = (WEB / "index.html").read_text() + (WEB / "app.js").read_text()
    calls = lambda text: set(re.findall(r'"(/api/[a-z_]+)', text))  # noqa: E731
    assert calls(PAGE) <= calls(new), calls(PAGE) - calls(new)  # everything the older page can ask of the program, the new one can too
    for part in ("KeyW", "KeyS", "KeyA", "KeyD", "KeyQ", "KeyE", "KeyZ", "KeyC", "KeyF", "KeyR", 'e.code !== "Space"', "setInterval(tell, 50)", 'addEventListener("blur"'):
        assert part in new, part  # the keys, and the space bar reported twenty times a second and let go of when the page loses the keyboard


# ------------------------------------------------------------------------------------------------ the model of the arm
def test_joint_angles_are_found_that_reach_a_pose():
    rng = np.random.default_rng(3)
    for k in range(60):
        xyz, R = np.array([rng.uniform(320, 580), rng.uniform(-180, 180), rng.uniform(90, 400)]), rpy_matrix([180.0, 0.0, rng.uniform(-170, 170)])
        if k % 3 == 0:  # as on the rig: straight down for an arm hung at 45 degrees
            down = mount_down("right")
            R = np.stack([[1.0, 0.0, 0.0], np.cross(down, [1.0, 0.0, 0.0]), down], 1)
        T = _frames(np.radians(joints_for(xyz, R)))[-1]
        assert np.linalg.norm(T[:3, 3] * 1e3 - xyz) < 0.5 and np.degrees(np.arccos(np.clip((np.trace(R @ T[:3, :3].T) - 1) / 2, -1, 1))) < 0.2, (k, xyz)


def test_the_model_follows_the_arm_and_asks_the_controller_nothing(ses):
    assert ses.view() == dict(connected=False) and ses.state()["scene"]["up"] is None
    armed(ses)
    a, n = ses.arm, len(ses.arm.log)
    v = ses.view()
    assert v["connected"] and len(v["joints"]) == 7 and v["grip"] == a.counts and v["moving"] is False
    T = _frames(np.radians(v["joints"]))[-1]
    assert np.linalg.norm(T[:3, 3] * 1e3 - a.p) < 0.5 and np.allclose(T[:3, :3], rpy_matrix(a.rpy), atol=2e-3)  # the drawn arm is where the stand-in is
    assert len(a.log) == n  # drawing sends the arm nothing
    ses.pressed = True
    ses.move("B", 1, False, False)
    seen = []
    while ses.worker is not None and len(seen) < 4000:
        seen.append(ses.view()["joints"])
        time.sleep(0.002)
    assert ses.job["outcome"] == "done"
    steps = np.abs(np.diff(np.array(seen), axis=0)).max(1)
    assert steps.max() < 25.0 and np.ptp(np.array(seen), axis=0).max() > 5.0  # it moved, and without a jump from one posture to another
    srv, url = serve(ses)
    try:
        code, body, _ = get(url + "/api/view")
        assert code == 200 and len(json.loads(body)["joints"]) == 7
    finally:
        srv.shutdown()
        srv.server_close()
    a.joints_view = lambda: 1 / 0
    assert "ZeroDivisionError" in ses.view()["error"]  # a drawing that fails says so and takes nothing down


def test_the_page_is_told_what_is_around_the_arm(ses):
    armed(ses)
    st = ses.state()
    sc = st["scene"]
    assert st["feeds"] is True and sc["tool_mm"] == TOOL_MM and np.allclose(sc["up"], [0, 0, 1]) and sc["path"] is None
    assert [a["key"] for a in st["arms"]] == ["right", "stand-in"] and np.allclose(st["arms"][0]["up"], -mount_down("right")) and np.allclose(st["arms"][1]["up"], [0, 0, 1])
    lim, box = ses.guard.lim, sc["box"]
    assert np.allclose(box["lo"], lim.lo) and np.allclose(box["hi"], lim.hi) and np.allclose(box["axes"], lim.axes) and np.allclose(box["origin"], lim.origin)
    assert [t["name"] for t in sc["things"]] == ["thing"] and np.allclose(sc["things"][0]["xyz"], ses.spots.grasp("A"), atol=0.1)
    ses.pressed = True
    ses.move("B", 1, False, True)  # one step at a time
    assert until(lambda: ses.state()["job"]["waiting"], 5)
    sc, job = ses.state()["scene"], ses.job
    assert len(sc["path"]) == len(job["steps"]) and sc["at"] == 0
    for got, step in zip(sc["path"], job["steps"]):
        assert (got is None) == (step["kind"] != "move") and (got is None or np.allclose(got, step["to"], atol=0.06))
    ses.stop()
    assert ses.state()["scene"]["box"] is None


# ------------------------------------------------------------------------------------------------ the live pictures
def test_each_camera_has_one_owner_and_is_let_go_when_nobody_watches(cams):
    seq, got = cams.frames("stand-in", 0, 2.0)
    assert seq > 0 and [c["id"] for c in got] == ["scene", "wrist", "broken"]  # the left arm's camera is not shown with the stand-in, and not opened
    assert until(lambda: all(c["jpg"] for c in cams.frames("stand-in", 0, 0.5)[1][:2]), 3)
    got = {c["id"]: c for c in cams.frames("stand-in", 0, 0.5)[1]}
    assert got["scene"]["jpg"][:3] == b"JPG" and got["scene"]["size"] == (6, 4) and got["scene"]["age_ms"] < 500 and got["scene"]["note"] == ""
    assert not got["broken"]["jpg"] and got["broken"]["age_ms"] is None and "another program" in got["broken"]["note"]  # says why, and the others go on
    assert not any("other" in name for name in Cam.opened)
    seq = max(c["seq"] for c in got.values())
    again = cams.frames("stand-in", seq, 1.0)
    assert again[0] > seq and any(c["jpg"] for c in again[1]) and all((c["seq"] > seq) == bool(c["jpg"]) for c in again[1])  # only what is new is sent again
    assert sorted(set(Cam.opened)) == sorted(Cam.opened) and not Cam.closed  # opened once each, still open
    assert until(lambda: len(Cam.closed) == 2, 3)  # nobody asked for 0.4 s: both are let go
    assert cams.frames("stand-in", 0, 2.0)[0] > 0 and until(lambda: len(Cam.opened) == 4, 3)  # and opened again when a page comes back
    cams.close()
    assert until(lambda: not any(o.is_alive() for o in cams.owners), 3)


def test_a_camera_is_lent_for_one_picture_and_taken_back(ses, cams, tmp_path):
    assert until(lambda: all(c["jpg"] for c in cams.frames("stand-in", 0, 0.5)[1][:2]), 3)
    scene = cams.owners[0]

    class Finder:  # the block finder's own camera: the same device as the live "scene" tile
        device, was_free = str(tmp_path / "scene"), None

        def grab(self):
            Finder.was_free = scene.cap is None and "lent" in scene.note
            return "a picture"

    ses.camera = Finder()
    assert ses._grab() == "a picture" and Finder.was_free  # while the finder had it, the live owner did not
    cams.frames("stand-in", 0, 0.5)
    assert until(lambda: scene.cap is not None and not scene.lent, 3)  # and the live picture comes back
    head, _ = ses.frames(0, "stand-in")
    assert [c["finds"] for c in head["cams"]][:2] == [True, False] and "device" not in head["cams"][0]  # the page marks that tile


def test_the_page_gets_all_new_pictures_in_one_answer(ses, cams):
    srv, url = serve(ses)
    try:
        assert get(url + "/api/frames?after=0&side=stand-in")[0] == 403  # pictures of the lab: for the dashboard's own page only
        assert get(url + "/api/frames?after=0&side=stand-in", "wrong")[0] == 403
        assert until(lambda: len(pictures(get(url + "/api/frames?after=0&side=stand-in", srv.token)[1])[1]["scene"]) > 0, 4)
        code, body, kind = get(url + "/api/frames?after=0&side=stand-in", srv.token)
        head, pics = pictures(body)
        assert code == 200 and "octet-stream" in kind and head["on"] and head["seq"] > 0
        assert [c["id"] for c in head["cams"]] == ["scene", "wrist", "broken", "made_up"] and pics["scene"][:3] == b"JPG" and pics["wrist"][:3] == b"JPG" and pics["broken"] == b""
        head, pics = pictures(get(url + "/api/frames?after=0&side=right", srv.token)[1])
        assert [c["id"] for c in head["cams"]] == ["scene"]  # before a session: the cameras of the arm picked on the page
        ses.start("stand-in")
        head, _ = pictures(get(url + "/api/frames?after=0&side=right", srv.token)[1])
        assert len(head["cams"]) == 4  # during a session: those of its arm, whatever the page says
        head, pics = pictures(get(url + f"/api/frames?after={10 ** 9}&side=stand-in", srv.token)[1])
        assert all(c["len"] == 0 for c in head["cams"]) and not any(pics.values())  # nothing new: an answer without pictures, within a second
    finally:
        srv.shutdown()
        srv.server_close()


def test_without_live_pictures_the_page_says_so(tmp_path):
    s = Session({"stand-in": dict(label="Stand-in arm (no robot)", ip=None, make=FakeArm)}, spots_path=tmp_path / "spots.json", log_dir=tmp_path / "logs", probe=lambda ip: False)
    try:
        head, pics = s.frames(0, "stand-in")
        assert s.state()["feeds"] is False and head["on"] is False and "switched off" in head["why"] and pics == []
        assert s._grab.__self__ is s and s.feeds is None
    finally:
        s.close()
