"""Folding the other arm out of the way (assemblyloop/park.py, Session.park_other), rehoming the session's own arm
(Session.rehome) and each arm's own files, on stand-in arms in real time. In the tests of the other arm the session's
own arm is the left one and the right one is folded away; in the rehome tests the session has the right one."""
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from assemblyloop.arms import FakeArm, rpy_matrix, run_fake_clock
from assemblyloop.dashboard import make_server
from assemblyloop.guard import Refused
from assemblyloop.park import DONE, PARK_DEG, VIA_MM, levelled
from assemblyloop.session import Session
from assemblyloop.spots import per_arm


def until(cond, secs=10.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        if cond():
            return True
        time.sleep(0.01)
    return False


class Switch:
    """The page's MOVING switch: reported fifty times a second, on or off."""

    def __init__(self, ses):
        self.ses, self.pressed, self.alive = ses, False, True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.alive:
            dm = self.ses.dm
            if dm is not None:
                dm.feed(self.pressed)
            time.sleep(0.02)

    def on(self):  # as on the page: the switch was seen off first (a press counts only after a release)
        n = self.ses.dm.n
        assert until(lambda: self.ses.dm.n >= n + 2)
        self.pressed = True

    def off(self):
        self.pressed = False


@pytest.fixture
def rig(tmp_path):
    """Two "real" arms as far as the session knows (they have addresses), both answering. Returns a function that
    makes the session; the right arm starts low over the table with its jaws shut."""
    made = []

    def make(right=None, left=None, probe=True, fake_speed=40.0):
        right = right or FakeArm(pose=(450.0, 50.0, 100.0, 180.0, 0.0, 0.0))
        left = left or FakeArm()
        right.counts = 0
        arms = {"right": dict(label="Right arm", ip="192.0.2.10", make=lambda: right, down=(0.0, 0.0, -1.0), park=PARK_DEG["right"]),
                "left": dict(label="Left arm", ip="192.0.2.11", make=lambda: left, down=(0.0, 0.0, -1.0), park=PARK_DEG["left"]),
                "stand-in": dict(label="Stand-in arm (no robot)", ip=None, make=FakeArm)}
        s = Session(arms, spots_path=tmp_path / "spots.json", log_dir=tmp_path / "logs", probe=lambda ip: probe, fake_speed=fake_speed, key_hold=True,
                    guard_kw=dict(poll_s=0.002), limit_kw=dict(settle_s=0.0), quest_port=0, camera_path=tmp_path / "camera.json")
        sw = Switch(s)
        made.append((s, sw))
        assert until(lambda: s.reach.get("right") is probe)
        return s, sw, right, left

    yield make
    for s, sw in made:
        sw.alive = False
        s.close()


def sent(arm, *kinds):
    return [e for e in arm.log if e[0] in kinds]


def test_the_other_arm_is_folded_away_the_labs_way_and_only_while_the_switch_is_on(rig):
    s, sw, right, left = rig()
    s.start("left", hold="page")
    o = s.state()["other"]
    assert o["key"] == "right" and o["answers"] and o["can"] and not o["busy"]
    s.park_other()
    time.sleep(0.3)
    assert not sent(right, "grip", "move", "joints"), right.log  # switched on, nothing moved: MOVING is off
    assert s.state()["other"]["busy"] and s.state()["hint"].startswith("Right arm: ready to move out of the way. Switch MOVING on")
    sw.on()
    assert until(lambda: not s.state()["other"]["busy"], 20), s.state()["other"]
    o = s.state()["other"]
    assert o["outcome"] == DONE, o
    assert o["steps"] == ["open the gripper", "come straight up 200 mm", "go to the lab's waypoint in front of it", "fold it to the lab's resting pose"]
    kinds = [e[0] for e in right.log]
    assert kinds.index("grip") < kinds.index("move") and right.log[kinds.index("grip")][2] == 850  # the jaws open before anything moves
    moves = [np.array(e[2]) for e in sent(right, "move")]
    rise = [d for d in moves if abs(d[0]) < 1e-6 and abs(d[1]) < 1e-6]
    assert len(rise) == 3 and np.isclose(sum(d[2] for d in rise), 200.0) and all(d[2] <= 80.0 for d in rise)  # straight up first, in pieces
    assert np.allclose(moves[-1], [-50.0, -50.0, 0.0]) and np.allclose(right.plan, VIA_MM)  # then across, at the waypoint's height
    folds = sent(right, "joints")
    assert kinds.index("joints") > max(i for i, k in enumerate(kinds) if k == "move")  # folded only from the waypoint
    assert len(folds) == 10 and np.allclose(folds[-1][2], PARK_DEG["right"]) and all(f[3] == 20.0 for f in folds)  # 5 deg pieces, at the lab's 20 deg/s
    assert np.allclose(right.joints(False), PARK_DEG["right"]) and right.stopped and not right.connected  # left stopped, connection closed
    assert left.log == []  # the session's own arm was never touched
    assert s.phase() == "DISARMED" and s.parking["outcome"] == DONE


def test_letting_go_pauses_it_and_stop_session_stops_it_where_it_is(rig):
    s, sw, right, left = rig(fake_speed=2.0)
    s.start("left", hold="page")
    s.park_other()
    sw.on()
    assert until(lambda: sent(right, "move"))
    sw.off()
    assert until(lambda: right.paused)
    at = right.plan.copy()
    time.sleep(0.2)
    assert np.allclose(right.plan, at)  # paused: it stands
    assert s.state()["hint"].startswith("Right arm: ready to move out of the way")
    s.stop()
    assert s.phase() == "STOPPED" and s.parking is None
    assert right.stopped and not right.connected and not sent(right, "joints")


def test_a_stop_before_its_guard_exists_still_stops_it(rig):
    s, sw, right, left = rig()
    slow = right.connect

    def connect():
        time.sleep(0.4)
        slow()

    right.connect = connect
    s.start("left", hold="page")
    sw.on()
    s.park_other()
    s.stop()  # while it is still connecting
    assert not sent(right, "grip", "move", "joints", "enable") and not right.connected


def test_what_is_refused_and_what_needs_nothing(rig):
    s, sw, right, left = rig()
    s.start("stand-in")
    assert s.state()["other"] is None  # the stand-in has no neighbour
    with pytest.raises(Refused, match="no other arm"):
        s.park_other()
    s.end()

    s.start("left", hold="page")
    right.err = 5
    s.park_other()
    assert until(lambda: not s.parking["busy"])
    assert s.parking["outcome"].startswith("not moved: controller error 5") and not sent(right, "enable")  # never cleared, never enabled
    right.err = 0

    right.q6, right.j7 = np.array(PARK_DEG["right"][:6]) + 0.3, PARK_DEG["right"][6]
    s.park_other()
    assert until(lambda: not s.parking["busy"])
    assert s.parking["outcome"].startswith("already out of the way") and not sent(right, "enable")

    right.q6 = np.array(PARK_DEG["right"][:6]) + np.r_[250.0, np.zeros(5)]
    s.park_other()
    assert until(lambda: not s.parking["busy"])
    assert s.parking["outcome"].startswith("not moved: joint 1 is at 254 deg") and not sent(right, "enable")

    right.q6, right.j7, right.reach_mm = np.zeros(6), 0.0, 300.0  # the waypoint is out of its reach
    s.park_other()
    assert until(lambda: not s.parking["busy"])
    assert "not within its reach" in s.parking["outcome"] and not sent(right, "enable")
    right.reach_mm = None

    s.dm.feed(False)
    assert until(lambda: all(c["ok"] for c in s.state()["checks"]))
    s.arm_on()  # the session's own arm, waiting to be nudged: then the other one may not move
    s.jog("up", 5)
    with pytest.raises(Refused, match="wait until the left arm has finished"):
        s.park_other()
    s.disarm()
    assert until(lambda: s.worker is None)
    s.end()

    s.start("left", hold="page")
    s.park_other()  # waits for the switch ...
    assert until(lambda: s.parking.get("guard") is not None)
    s.dm.feed(False)
    with pytest.raises(Refused, match="wait until the right arm is out of the way"):
        s.arm_on()  # ... and meanwhile the session's own arm stays as it is
    with pytest.raises(Refused, match="being moved already"):
        s.park_other()
    s.disarm()  # stops the other arm too
    assert until(lambda: not s.parking["busy"]) and s.parking["outcome"].startswith("stopped where it was")
    assert right.stopped and not sent(right, "move", "joints")
    s.end()


def test_an_arm_that_does_not_answer_is_not_offered(rig):
    s, sw, right, left = rig(probe=False)
    s.start("left", hold="page")
    o = s.state()["other"]
    assert o["answers"] is False and not o["can"] and "no answer" in o["why"]
    with pytest.raises(Refused, match="does not answer at 192.0.2.10"):
        s.park_other()


def test_nearly_folded_is_folded_from_where_it_is_and_a_tilted_gripper_is_pointed_down_first(rig):
    s, sw, right, left = rig()
    s.start("left", hold="page")
    sw.on()
    right.q6, right.j7, right.counts = np.array(PARK_DEG["right"][:6]) - 8.0, PARK_DEG["right"][6] + 8.0, 850
    s.park_other()
    assert until(lambda: not s.parking["busy"]) and s.parking["outcome"] == DONE, s.parking
    assert s.parking["steps"] == ["fold it to the lab's resting pose"] and not sent(right, "move", "grip")  # no waypoint, the jaws were open

    right.q6, right.j7, right.j7_real = np.zeros(6), 0.0, 0.0
    right.rpy = np.array([150.0, 0.0, 0.0])  # tilted 30 deg
    right.enabled, right.stopped = False, True
    s.park_other()
    assert until(lambda: not s.parking["busy"], 20) and s.parking["outcome"] == DONE, s.parking
    assert s.parking["steps"][0] == "point its gripper down (tilted 30 deg)"
    kinds = [e[0] for e in right.log]
    assert kinds.index("tilt") < kinds.index("move") and np.isclose(rpy_matrix(right.rpy)[2, 2], -1.0, atol=1e-6)
    assert np.allclose(rpy_matrix(levelled([150.0, 0.0, 0.0], (0.0, 0.0, -1.0)))[:, 2], [0.0, 0.0, -1.0])


def test_something_holding_it_stops_it_and_it_is_not_tried_again_in_that_session(rig):
    s, sw, right, left = rig()
    s.start("left", hold="page")
    sw.on()
    right.q6, right.j7, right.turn_stuck = np.array(PARK_DEG["right"][:6]) - 8.0, PARK_DEG["right"][6] + 8.0, True
    s.park_other()
    assert until(lambda: not s.parking["busy"])
    assert s.parking["outcome"].startswith("stopped where it was") and "holding the arm" in s.parking["outcome"] and right.stopped
    o = s.state()["other"]
    assert not o["can"] and "end this session" in o["why"]
    with pytest.raises(Refused, match="was stopped"):
        s.park_other()
    s.end()
    s.start("left", hold="page")
    assert s.state()["other"]["can"]  # a new session may try again, after a person has looked


def test_each_arm_keeps_its_own_places_and_camera_fit(rig, tmp_path):
    assert per_arm(tmp_path / "spots.json", "right") == tmp_path / "spots.json" and per_arm("spots.json", None).name == "spots.json"
    assert per_arm(tmp_path / "spots.json", "left") == tmp_path / "spots_left.json" and per_arm("camera.json", "stand-in").name == "camera_stand-in.json"
    s, sw, right, left = rig()
    s.start("left", hold="page")
    assert s.spots_path == tmp_path / "spots_left.json" and s.camera_path == tmp_path / "camera_left.json"
    s.dm.feed(False)
    assert until(lambda: all(c["ok"] for c in s.state()["checks"]))
    s.arm_on()
    s.save_spot("A")
    left.p = left.plan = left.p + np.array([0.0, 60.0, 0.0])
    s.save_spot("B")
    s.calibrate_done()
    assert (tmp_path / "spots_left.json").exists() and not (tmp_path / "spots.json").exists()  # the right arm's places are untouched
    assert json.loads((tmp_path / "spots_left.json").read_text())["side"] == "left"
    s.end()
    s.start("right", hold="page")
    assert s.state()["calibration"]["saved"] is None and s.state()["other"]["key"] == "left"  # nothing taught for the right arm yet


def test_the_page_has_the_button_and_it_needs_the_token(rig):
    s, sw, right, left = rig()
    srv = make_server(s, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(path, token):
        req = urllib.request.Request(base + path, data=b"{}", method="POST", headers={"Content-Type": "application/json", "X-Token": token or ""})
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        page = urllib.request.urlopen(base + "/", timeout=3).read().decode()
        assert 'id="o_btn"' in page
        assert post("/api/park", "wrong")[0] == 403
        code, body = post("/api/park", srv.token)
        assert code == 409 and "start a session first" in body["error"]
        s.start("left", hold="page")
        assert post("/api/park", srv.token) == (200, {"ok": True})
        assert until(lambda: s.parking.get("guard") is not None)
        st = json.loads(urllib.request.urlopen(base + "/api/state", timeout=3).read())
        assert st["other"]["busy"] and st["other"]["label"] == "Right arm"
    finally:
        srv.shutdown()
        srv.server_close()


# ---------------------------------------------------------------------- "Rehome": the session's own arm
def armed(s, arm, speed=40.0):
    """Arm the session's own arm. It counts as a real one here, so it gets its clock from the test."""
    run_fake_clock(arm, speed, alive=lambda: s.connected)
    s.dm.feed(False)
    assert until(lambda: all(c["ok"] for c in s.state()["checks"]))
    s.arm_on()


def test_rehome_folds_this_arm_the_labs_way_and_leaves_it_switched_on(rig):
    s, sw, right, left = rig()
    s.start("right", hold="page")
    h = s.state()["home"]
    assert h["real"] and not h["can"] and h["why"] == "arm first"
    with pytest.raises(Refused, match="arm first"):
        s.rehome()
    armed(s, right)
    assert s.state()["home"]["can"]
    s.rehome()
    time.sleep(0.3)
    assert not sent(right, "grip", "move", "joints"), right.log  # nothing moves while MOVING is off
    assert s.state()["home"]["busy"] and s.state()["hint"].startswith("Rehome: ready to fold into the lab's resting pose. Switch MOVING on")
    with pytest.raises(Refused, match="wait until the arm has finished"):
        s.rehome()
    with pytest.raises(Refused, match="wait until"):
        s.jog("up", 5)
    sw.on()
    assert until(lambda: not s.state()["home"]["busy"], 20), s.state()["home"]
    h = s.state()["home"]
    assert h["outcome"] == "rehomed: folded to the lab's resting pose", h
    assert h["steps"] == ["open the gripper", "come straight up 200 mm", "go to the lab's waypoint in front of it", "fold it to the lab's resting pose"]
    kinds = [e[0] for e in right.log]
    assert kinds.index("grip") < kinds.index("move") < kinds.index("joints") and right.log[kinds.index("grip")][2] == 850  # the jaws open first
    assert kinds.index("joints") > max(i for i, k in enumerate(kinds) if k == "move")  # folded only from the waypoint
    folds = sent(right, "joints")
    assert len(folds) == 10 and np.allclose(folds[-1][2], PARK_DEG["right"]) and all(f[3] == 20.0 for f in folds)
    assert np.allclose(right.joints(False), PARK_DEG["right"]) and np.allclose(right.plan, VIA_MM)
    assert s.phase() == "AWAITING CALIBRATION" and right.connected and right.enabled and not right.stopped  # still switched on
    assert s.state()["hint"].startswith("Folded into the lab's resting pose")
    assert np.allclose(s.guard.lim.origin, right.pose()[:3])  # nudging starts from where it ended
    assert left.log == []  # the other arm was never touched
    sw.off()
    s.jog("up", 5)
    sw.on()
    assert until(lambda: s.worker is None)
    assert s.state()["calibration"]["note"] == "" and np.isclose(right.plan[2], VIA_MM[2] + 5.0)
    assert not s.state()["hint"].startswith("Folded")

    right.q6 = np.array(PARK_DEG["right"][:6]) + 0.3
    n = len(sent(right, "joints"))
    s.rehome()  # every joint within 1 deg of the resting pose: nothing is sent
    assert s.state()["home"]["outcome"] == "already in the lab's resting pose: nothing was sent" and len(sent(right, "joints")) == n and s.worker is None


def test_rehome_from_armed_forgets_the_places_but_they_stay_saved(rig):
    s, sw, right, left = rig()
    s.start("right", hold="page")
    armed(s, right)
    s.save_spot("A")
    with pytest.raises(Refused, match="press Done first, or remove them"):  # they would be lost
        s.rehome()
    right.p = right.plan = right.p + np.array([0.0, 60.0, 0.0])
    s.save_spot("B")
    s.calibrate_done()
    assert s.phase() == "ARMED"
    sw.on()
    s.rehome()  # MOVING is switched off on the page first; here it stays on
    assert s.phase() == "AWAITING CALIBRATION" and s.spots is None and not s.calibrated  # the gripper leaves the places
    assert until(lambda: not s.state()["home"]["busy"], 20) and s.state()["home"]["outcome"].startswith("rehomed")
    assert s.state()["calibration"]["saved"]["names"] == ["A", "B"]  # still on disk: "Use saved spots" once it is back over them
    assert s.state()["calibration"]["spots"] == []


def test_rehome_stops_where_it_is_on_disarm_and_is_not_offered_for_the_stand_in(rig):
    s, sw, right, left = rig(fake_speed=2.0)
    s.start("right", hold="page")
    armed(s, right, 2.0)
    s.rehome()
    sw.on()
    assert until(lambda: sent(right, "move"))
    s.disarm()
    assert until(lambda: not s.state()["home"]["busy"])
    assert s.state()["home"]["outcome"] == "stopped: disarmed on the dashboard" and s.phase() == "DISARMED" and right.stopped
    assert not sent(right, "joints")
    s.end()
    s.start("stand-in")
    h = s.state()["home"]
    assert not h["real"] and not h["can"]
    with pytest.raises(Refused, match="the stand-in arm has no resting pose"):
        s.rehome()


def test_the_page_has_the_rehome_button_and_it_needs_the_token(rig):
    s, sw, right, left = rig()
    srv = make_server(s, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(path, token):
        req = urllib.request.Request(base + path, data=b"{}", method="POST", headers={"Content-Type": "application/json", "X-Token": token or ""})
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        page = urllib.request.urlopen(base + "/", timeout=3).read().decode()
        assert 'id="homebtn"' in page
        assert post("/api/rehome", "wrong")[0] == 403
        code, body = post("/api/rehome", srv.token)
        assert code == 409 and "start a session first" in body["error"]
        s.start("right", hold="page")
        armed(s, right)
        assert post("/api/rehome", srv.token) == (200, {"ok": True})
        st = json.loads(urllib.request.urlopen(base + "/api/state", timeout=3).read())
        assert st["home"]["busy"] and st["home"]["steps"][0] == "open the gripper"
    finally:
        srv.shutdown()
        srv.server_close()
