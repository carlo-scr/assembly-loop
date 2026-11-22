"""The dashboard's session and its web buttons (assemblyloop/session.py, dashboard.py), on a stand-in arm in real time."""
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from assemblyloop.arms import ArmFault, FakeArm, rpy_matrix, twist_deg
from assemblyloop.dashboard import make_server
from assemblyloop.guard import Refused
from assemblyloop.session import Session
from assemblyloop.spots import Spots, demo_spots, plan_move, tool_axis


def until(cond, secs=6.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        if cond():
            return True
        time.sleep(0.01)
    return False


class Grip:
    """A stand-in for the headset: reports the grip's state fifty times a second, pressed or not."""

    def __init__(self, ses, pressed=False):
        self.ses, self.pressed, self.alive = ses, pressed, True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.alive:
            dm = self.ses.dm
            if dm is not None:
                dm.feed(self.pressed)
            time.sleep(0.02)

    def hold(self):
        self.pressed = True

    def letgo(self):
        self.pressed = False
        time.sleep(0.06)


def boom():
    raise OSError("no route to host")


@pytest.fixture
def ses(tmp_path):
    arms = {"right": dict(label="Right arm", ip="192.0.2.10", make=lambda: type("Dead", (), {"connect": staticmethod(boom)})()),
            "stand-in": dict(label="Stand-in arm (no robot)", ip=None, make=FakeArm)}
    s = Session(arms, spots_path=tmp_path / "spots.json", log_dir=tmp_path / "logs", probe=lambda ip: False, fake_speed=25.0,
                guard_kw=dict(poll_s=0.002), limit_kw=dict(settle_s=0.0))  # the stand-in arm needs no settling time
    s.grip = Grip(s)  # shadows Session.grip (the gripper button) on this object: call that as Session.grip(s, close)
    yield s
    s.grip.alive = False
    s.close()


def ready(s):
    """Session on the stand-in arm, armed, made-up spots known: the label reads ARMED."""
    if not s.connected:
        s.start("stand-in")
    s.grip.letgo()
    assert until(lambda: all(c["ok"] for c in s.state()["checks"])), s.state()["checks"]
    s.arm_on()
    if s.teaching:
        s.use_demo()
    assert s.phase() == "ARMED"


def moves(s):
    return [e for e in s.arm.log if e[0] == "move"]


def test_the_ladder_off_disarmed_calibration_armed_running(ses):
    st = ses.state()
    assert st["phase"] == "OFF" and not st["connected"] and "checks" not in st  # nothing is on by default
    assert until(lambda: ses.state()["arms"][0]["answers"] is False, 3.0)  # the page shows that the right arm does not answer
    with pytest.raises(Refused, match="powered"):
        ses.start("right")  # an arm without power: the session does not start
    assert ses.phase() == "OFF"
    for ask in (ses.arm_on, lambda: ses.jog("up", 5), lambda: ses.move("B")):
        with pytest.raises(Refused):
            ask()
    ses.start("stand-in")
    assert ses.phase() == "DISARMED" and ses.arm.log == []  # connected, read only
    for ask in (lambda: ses.jog("up", 5), lambda: ses.save_spot("A"), lambda: ses.move("B")):
        with pytest.raises(Refused):
            ask()
    assert until(lambda: all(c["ok"] for c in ses.state()["checks"]))
    ses.arm_on()
    assert ses.phase() == "AWAITING CALIBRATION" and moves(ses) == []  # switched on; it knows no place yet; nothing moved

    def nudge(direction, mm, n=1):
        for _ in range(n):
            ses.jog(direction, mm)
            assert until(lambda: ses.worker is None), ses.state()["log"]

    z0 = ses.arm.p.copy()
    ses.jog("down", 20)
    time.sleep(0.25)
    assert moves(ses) == [] and np.allclose(ses.arm.p, z0)  # the grip is not held: a nudge does nothing
    with pytest.raises(Refused, match="finished"):
        ses.jog("down", 20)  # one thing at a time
    ses.grip.hold()
    assert until(lambda: ses.worker is None)
    assert np.allclose(ses.arm.p, z0 - [0, 0, 20.0], atol=1e-6)
    with pytest.raises(Refused):
        ses.jog("down", 61)  # never more than 60 mm at once
    nudge("down", 20, 4)  # 100 mm down: around the block
    ses.arm.blocked = lambda p: p[2] < 195.0  # the table
    nudge("down", 20)
    assert "blocked" in ses.state()["calibration"]["note"] and ses.phase() == "AWAITING CALIBRATION"  # it ends by itself; the session goes on
    ses.arm.blocked = None
    nudge("up", 5, 2)
    nudge("down", 5, 2)
    here = ses.arm.p.copy()
    assert 190.0 < here[2] < 201.0  # a few millimetres above where the table stopped it
    ses.arm.bricks = {"thing": here.copy()}
    Session.grip(ses, True)  # (`ses.grip` is the test's stand-in headset; this is the gripper button)
    assert until(lambda: ses.worker is None) and "holding something 32 mm wide" in ses.state()["calibration"]["note"]
    Session.grip(ses, False)
    assert until(lambda: ses.worker is None) and ses.arm.counts == 850
    ses.save_spot("A")
    nudge("left", 5)
    with pytest.raises(Refused, match="apart"):
        ses.save_spot("B")  # 5 mm beside A: too close
    nudge("up", 20, 2)
    nudge("left", 20, 2)
    nudge("fwd", 20)
    nudge("down", 20, 2)
    ses.save_spot("B")
    with pytest.raises(Refused, match="ARMED"):
        ses.move("B")
    ses.calibrate_done()
    assert ses.phase() == "ARMED" and ses.spots_path.exists() and sorted(ses.spots.at) == ["A", "B"]
    assert np.allclose(ses.spots.at["B"] - ses.spots.at["A"], [20.0, 45.0, 0.0], atol=0.6)  # forward = +x, left = +y for this stand-in
    with pytest.raises(Refused, match="which spot"):
        ses.move("B")  # it has not been told where the block stands
    ses.set_thing("A")
    n0 = len(moves(ses))
    ses.grip.letgo()
    ses.move("B")
    time.sleep(0.3)
    assert ses.phase() == "PAUSED" and len(moves(ses)) == n0  # move started, grip not held: nothing happens
    ses.grip.hold()
    assert until(lambda: ses.phase() == "RUNNING")
    assert until(lambda: ses.worker is None), ses.state()["log"]
    ses.grip.letgo()
    st = ses.state()
    assert st["phase"] == "ARMED" and st["job"]["outcome"] == "done" and st["thing_at"] == "B"
    assert np.linalg.norm(ses.arm.bricks["thing"] - ses.spots.grasp("B")) < 0.05
    ses.end()
    assert ses.phase() == "OFF" and ses.arm is None
    ses.start("stand-in")  # a new session from the page: it has to be armed and the saved spots confirmed again
    assert ses.phase() == "DISARMED" and ses.state()["calibration"]["saved"]["names"] == ["A", "B"]
    assert until(lambda: all(c["ok"] for c in ses.state()["checks"]))
    ses.arm_on()
    assert ses.phase() == "AWAITING CALIBRATION"
    ses.use_saved()
    assert ses.phase() == "ARMED" and sorted(ses.spots.at) == ["A", "B"]
    ses.calibrate_begin()  # "Teach again"
    assert ses.phase() == "AWAITING CALIBRATION" and sorted(ses.draft.at) == ["A", "B"]


def test_keep_moving_it_around_and_a_rehearsal_never_comes_down(ses):
    ready(ses)
    ses.grip.hold()
    ses.move(None, count=3, rehearse=True)
    assert until(lambda: ses.worker is None), ses.state()["log"]
    low = min(float(e[4] @ ses.spots.up) for e in moves(ses))
    assert low >= ses.spots.travel_w() - 1.0 - 1e-6 and ses.thing_at == "A"  # stayed at the travel height; the thing was not touched
    assert np.linalg.norm(ses.arm.bricks["thing"] - ses.spots.grasp("A")) < 1e-9 and ses.state()["job"]["done"] == 3
    ses.move(None, count=4)
    assert until(lambda: ses.worker is None, 12.0), ses.state()["log"]
    ses.grip.letgo()
    st = ses.state()
    assert st["job"]["outcome"] == "done" and st["job"]["done"] == 4 and st["thing_at"] in ses.spots.at
    assert np.linalg.norm(ses.arm.bricks["thing"] - ses.spots.grasp(st["thing_at"])) < 0.05
    assert len((ses.log_dir / "moves.jsonl").read_text().splitlines()) == 7


def test_arming_is_refused_while_the_grip_is_held_or_silent(ses):
    ses.start("stand-in")
    ses.grip.alive = False  # the headset is off
    time.sleep(0.9)
    with pytest.raises(Refused, match="signal"):
        ses.arm_on()
    ses.grip = Grip(ses, pressed=True)  # the headset comes on with the grip already squeezed
    time.sleep(0.2)
    assert not ses.dm.held()  # a squeeze that was never released does not count
    ses.grip.letgo()
    ses.grip.hold()
    time.sleep(0.1)
    with pytest.raises(Refused, match="Grip released"):
        ses.arm_on()
    assert ses.phase() == "DISARMED"
    ready(ses)


def test_letting_go_pauses_and_the_stop_button_ends_the_session(ses):
    ready(ses)
    ses.move("C", rehearse=True)
    ses.grip.hold()
    assert until(lambda: ses.arm.moving_now())
    ses.grip.letgo()
    assert until(lambda: ses.phase() == "PAUSED", 1.0)
    time.sleep(0.05)
    a = ses.arm.plan.copy()
    time.sleep(0.2)
    assert np.allclose(a, ses.arm.plan)  # paused: not moving
    ses.grip.hold()
    arm = ses.arm
    assert until(lambda: arm.moving_now(), 1.0)
    ses.stop()
    ses.grip.letgo()
    assert ses.phase() == "STOPPED" and arm.stopped and not ses.connected and "stopped" in ses.state()["job"]["outcome"]
    n = len(arm.log)
    for ask in (ses.arm_on, lambda: ses.move("B"), ses.calibrate_begin):
        with pytest.raises((Refused, ArmFault)):
            ask()
    time.sleep(0.2)
    assert len(arm.log) == n  # nothing more reaches the arm
    ses.start("stand-in")  # a fresh session starts again at DISARMED
    assert ses.phase() == "DISARMED" and ses.state()["ended"] is None


def test_disarming_mid_move_stops_the_arm_without_ending_the_session(ses):
    ready(ses)
    ses.move("C", rehearse=True)
    ses.grip.hold()
    assert until(lambda: ses.arm.moving_now())
    ses.disarm()
    assert until(lambda: ses.worker is None, 1.0)
    ses.grip.letgo()
    assert ses.phase() == "DISARMED" and ses.arm.stopped and ses.ended is None and ses.connected
    ses.grip.letgo()
    assert until(lambda: all(c["ok"] for c in ses.state()["checks"]))
    ses.arm_on()
    assert ses.phase() == "AWAITING CALIBRATION"  # armed again: the places have to be confirmed again
    ses.use_demo()
    assert ses.phase() == "ARMED"


def test_step_mode_waits_for_the_button_before_every_step(ses):
    ready(ses)
    ses.grip.hold()
    ses.move("B", rehearse=True, step=True)
    assert until(lambda: (ses.state()["job"] or {}).get("waiting"))
    time.sleep(0.2)
    assert ses.state()["job"]["i"] == 0 and moves(ses) == [] and ses.phase() == "PAUSED"  # held, but no step was allowed yet
    ses.next_step()
    ses.next_step()  # a double click is one step, not two
    assert until(lambda: ses.state()["job"]["i"] == 1 and ses.state()["job"]["waiting"])
    time.sleep(0.2)
    assert ses.state()["job"]["i"] == 1 and moves(ses) == []  # the first step was "rise", and the arm was already up
    ses.next_step()
    assert until(lambda: ses.state()["job"]["i"] == 2 and ses.state()["job"]["waiting"])
    assert len(moves(ses)) >= 1  # "go above A" has happened
    ses.stop()


def test_a_thing_that_is_not_there_and_a_blocked_put_down_are_reported(ses):
    ready(ses)
    ses.arm.bricks = {}  # nothing stands on A
    ses.arm_on()
    ses.grip.hold()
    ses.move("B")
    assert until(lambda: ses.worker is None), ses.state()["log"]
    st = ses.state()
    assert "no brick" in st["job"]["outcome"] and st["thing_at"] is None and st["phase"] == "ARMED" and ses.arm.counts == 850
    ses.arm.bricks = {"thing": ses.spots.grasp("A").copy()}
    ses.set_thing("A")
    gb, up = ses.spots.grasp("B"), ses.spots.up
    ses.arm.blocked = lambda p: np.linalg.norm((p - gb) - ((p - gb) @ up) * up) < 12.0 and (p - gb) @ up < 15.0  # something already stands on B
    ses.move("B")
    assert until(lambda: ses.worker is None), ses.state()["log"]
    ses.grip.letgo()
    st = ses.state()
    assert "blocked" in st["job"]["outcome"] and st["thing_at"] is None and st["phase"] == "ARMED"
    assert ses.arm.holding is not None  # still in the gripper, lifted clear


def test_the_quest_port_is_taken_as_soon_as_teleoperation_lets_go_of_it(tmp_path):
    import socket

    from assemblyloop.session import default_arms

    teleop = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # the teleoperation dashboard is still open: it has the Quest's port
    teleop.bind(("0.0.0.0", 0))
    port = teleop.getsockname()[1]
    s = Session({"stand-in": default_arms()["stand-in"]}, quest_port=port, spots_path=tmp_path / "spots.json", log_dir=tmp_path / "logs", fake_speed=25.0)
    try:
        s.start("stand-in", hold="quest")  # a stand-in arm with the real Quest switch
        assert s.phase() == "DISARMED" and "in use" in s.state()["quest"]["why"]  # the session starts, and says what is in the way
        with pytest.raises(Refused, match="teleoperation"):
            s.arm_on()
        teleop.close()
        assert until(lambda: s.dm.listening, 3.0)
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for seq in range(1, 12):
            tx.sendto(json.dumps(dict(seq=seq, right=dict(grip=0.0), left=dict(grip=0.0))).encode(), ("127.0.0.1", port))
            time.sleep(0.02)
        assert until(lambda: all(c["ok"] for c in s.state()["checks"]), 3.0), s.state()["checks"]
        s.arm_on()
        assert s.phase() == "AWAITING CALIBRATION"
        time.sleep(0.5)  # the frames have stopped: no grip, no nudge
        s.jog("up", 5)
        time.sleep(0.2)
        assert [e for e in s.arm.log if e[0] == "move"] == []
    finally:
        s.close()


def test_the_gripper_is_turned_as_it_was_taught_at_each_spot(ses):
    ses.start("stand-in")
    ses.grip.letgo()
    assert until(lambda: all(c["ok"] for c in ses.state()["checks"]))
    ses.arm_on()
    ses.grip.hold()

    def do(fn, *a):
        fn(*a)
        assert until(lambda: ses.worker is None), ses.state()["log"]

    with pytest.raises(Refused):
        ses.turn("sideways", 15)
    with pytest.raises(Refused):
        ses.turn("cw", 120)
    for _ in range(5):
        do(ses.jog, "down", 20)
    ses.save_spot("A")  # the block lies square to the gripper here
    for _ in range(2):
        do(ses.jog, "up", 20)
    for _ in range(3):
        do(ses.jog, "left", 20)
    do(ses.turn, "ccw", 45)
    do(ses.turn, "ccw", 45)  # ... and should be put down a quarter turn round
    assert moves(ses) and len([e for e in ses.arm.log if e[0] == "turn"]) >= 9
    for _ in range(2):
        do(ses.jog, "down", 20)
    ses.save_spot("B")
    st = ses.state()["calibration"]["spots"]
    assert abs(abs(st[0]["yaw"] - st[1]["yaw"]) - 90.0) < 1.0
    ses.calibrate_done()
    assert abs(abs(twist_deg(rpy_matrix(ses.spots.turn("A")), rpy_matrix(ses.spots.turn("B")))[0]) - 90.0) < 0.5
    ses.arm.bricks = {"thing": ses.spots.grasp("A").copy()}
    ses.set_thing("A")
    seen = []
    close = ses.arm.grip
    ses.arm.grip = lambda counts: (seen.append((counts, ses.arm.rpy.copy())), close(counts))[1]  # how is the gripper turned whenever it opens or closes?
    ses.move("B")
    assert until(lambda: ses.worker is None, 12.0), ses.state()["log"]
    assert ses.state()["job"]["outcome"] == "done" and ses.thing_at == "B"
    grasp = next(r for c, r in seen if c == 0)
    letgo = [r for c, r in seen if c == 850][-1]
    assert abs(twist_deg(rpy_matrix(grasp), rpy_matrix(ses.spots.turn("A")))[0]) <= 1.0  # picked up turned as at A
    assert abs(twist_deg(rpy_matrix(letgo), rpy_matrix(ses.spots.turn("B")))[0]) <= 1.0  # put down turned as at B
    ses.grip.letgo()


def test_spots_geometry_and_plan():
    assert np.allclose(tool_axis([180.0, 0.0, 0.0]), [0.0, 0.0, -1.0])  # the gripper points down
    assert np.allclose(tool_axis([180.0, 0.0, 77.0]), [0.0, 0.0, -1.0])  # whatever its yaw
    assert np.allclose(tool_axis([135.0, 0.0, 0.0]), [0.0, -np.sin(np.radians(135)), np.cos(np.radians(135))])
    s = demo_spots()
    assert np.allclose(s.up, [0, 0, 1]) and s.travel_w() == 200.0
    lim = s.limits()
    assert lim.why_outside(s.grasp("A")) == "" and lim.why_outside(s.above(s.grasp("C"))) == ""
    assert lim.why_outside(s.grasp("A") - 5.0 * s.up) and lim.why_outside(s.grasp("A") + [200.0, 0, 0])
    with pytest.raises(Refused):
        Spots(rpy=np.array([180.0, 0, 0])).check()  # no spots
    with pytest.raises(Refused, match="TILTED"):
        s.add("D", [400.0, 0.0, 120.0, 180.0, 10.0, 0.0])
    s.add("D", [400.0, 0.0, 120.0, 180.0, 0.0, 77.0])  # spun about its own axis: fine, and remembered
    assert np.allclose(s.turn("D"), [180.0, 0.0, 77.0]) and np.allclose(s.turn("B"), [180.0, 0.0, 90.0])
    steps = [x["name"] for x in plan_move(s, "A", "D", s.above(s.grasp("A")), __import__("assemblyloop.cycle", fromlist=["Params"]).Params())]
    assert steps.index("turn the gripper as at A") < steps.index("lower around the thing at A") < steps.index("turn the gripper as at D") < steps.index("lower it onto D")
    old = Spots.from_dict(dict(rpy_deg=[180.0, 0.0, 5.0], spots=dict(A=[1.0, 2.0, 3.0], B=[100.0, 2.0, 3.0])))  # a file from before spots had turns
    assert np.allclose(old.turn("B"), [180.0, 0.0, 5.0])
    with pytest.raises(Refused, match="same"):
        plan_move(s, "A", "A", s.grasp("A"), None)
    assert Spots.from_dict(json.loads(json.dumps(s.to_dict()))).to_dict() == s.to_dict()


def test_web_buttons_need_the_pages_token_and_the_quest_only_rule_holds(ses):
    srv = make_server(ses, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def post(path, body, token=None, host=None):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        if token:
            req.add_header("X-Token", token)
        if host:
            req.add_header("Host", host)
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        page = urllib.request.urlopen(base + "/", timeout=3).read().decode()
        assert srv.token in page and "STOP SESSION" in page and "Start session" in page
        st = json.loads(urllib.request.urlopen(base + "/api/state", timeout=3).read())
        assert st["phase"] == "OFF" and [a["key"] for a in st["arms"]] == ["right", "stand-in"]
        assert post("/api/session", dict(start=True, arm="stand-in"))[0] == 403  # no token: another page cannot press the buttons
        assert post("/api/session", dict(start=True, arm="stand-in"), token="wrong")[0] == 403
        assert post("/api/session", dict(start=True, arm="stand-in"), token=srv.token, host="evil.example")[0] == 403
        assert ses.phase() == "OFF"
        code, j = post("/api/session", dict(start=True, arm="right", hold="page"), token=srv.token)
        assert code == 409 and "Quest" in j["error"]  # a real arm never takes the space bar as its switch
        code, j = post("/api/session", dict(start=True, arm="right"), token=srv.token)
        assert code == 409 and ("powered" in j["error"] or "Quest" in j["error"])
        assert post("/api/session", dict(start=True, arm="stand-in", hold="page"), token=srv.token) == (200, dict(ok=True))
        assert ses.phase() == "DISARMED"
        code, j = post("/api/arm", dict(on=True), token=srv.token)
        assert code == 409 and "cannot arm" in j["error"]  # no hold-to-run signal yet
        code, j = post("/api/jog", dict(dir="up", mm=5), token=srv.token)
        assert code == 409 and "arm first" in j["error"]
        assert until(lambda: all(c["ok"] for c in ses.state()["checks"]))  # the test's stand-in headset reports "released"
        assert post("/api/arm", dict(on=True), token=srv.token)[0] == 200 and ses.phase() == "AWAITING CALIBRATION"
        assert post("/api/jog", dict(dir="sideways", mm=5), token=srv.token)[0] == 409
        assert post("/api/jog", dict(dir="up", mm=5), token=srv.token)[0] == 200
        ses.grip.hold()
        assert until(lambda: ses.worker is None) and len(moves(ses)) >= 1
        ses.grip.letgo()
        assert post("/api/calibrate", dict(do="demo"), token=srv.token)[0] == 200 and ses.phase() == "ARMED"
        assert post("/api/hold", dict(pressed=False), token=srv.token)[0] == 200
        ses.hold = "quest"  # as with a real arm
        code, j = post("/api/hold", dict(pressed=True), token=srv.token)
        assert code == 409 and "Quest" in j["error"]
        assert post("/api/stop", {}, token=srv.token)[0] == 200 and ses.phase() == "STOPPED"
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_tower_is_built_level_by_level_and_taken_down_again(ses):
    ready(ses)
    sp, a, h = ses.spots, ses.arm, 30.0
    with pytest.raises(Refused, match="how many"):
        ses.tower("T", h, "build", n_now=None)
    with pytest.raises(Refused, match="nothing to do"):
        ses.tower("T", h, "down", n_now=0)
    with pytest.raises(Refused, match="between 10 and 80"):
        ses.tower("T", 200.0)
    with pytest.raises(Refused, match="no spot"):
        ses.tower("nowhere", h)
    highest = []
    a.fault_when = lambda arm: highest.append(float(arm.p @ sp.up)) and 0  # only watches: how high the arm goes
    ses.grip.hold()
    ses.tower("T", h, "build")
    assert until(lambda: ses.worker is None, 60), ses.state()["job"]
    assert ses.job["outcome"] == "done" and ses.job["done"] == 3 and ses.stack["n"] == 3, (ses.job["outcome"], ses.stack)
    drop, base = ses.P.drop_mm, sp.grasp("T")
    for k, name in enumerate("ABC"):  # A on the taught spot, B and C a block height higher each, let go a little above
        want = base + (k * h + (drop if k else 0.0)) * sp.up
        assert np.linalg.norm(a.bricks[name] - want) < 0.6, (name, a.bricks[name], want)
    assert max(highest) >= float(base @ sp.up) + sp.lift + 2 * h - 1.0  # it travelled above the finished tower
    assert ses.state()["stack"] == dict(on="T", block_mm=h, n=3, of=3, feeds=["A", "B", "C"])
    with pytest.raises(Refused, match="complete"):
        ses.tower("T", h, "build", n_now=3)

    a.bricks = {k: v.copy() for k, v in a.bricks.items()}
    ses.fake = False  # from here on the stand-in world is NOT reset to what is said: the blocks are where the build left them
    ses.tower("T", h, "down", n_now=3)
    assert until(lambda: ses.worker is None, 60), ses.state()["job"]
    assert ses.job["outcome"] == "done" and ses.stack["n"] == 0, (ses.job["outcome"], ses.stack)
    for k, name in enumerate("ABC"):  # top first, every block back on its own spot
        # (the stand-in world has no gravity: a block let go 2 mm above the tower stayed there, so it comes back 2 mm high)
        assert np.linalg.norm(a.bricks[name] - sp.grasp(name) - (drop if k else 0.0) * sp.up) < 0.6, (name, a.bricks[name])
    lines = [json.loads(x) for x in (ses.log_dir / "moves.jsonl").read_text().splitlines()]
    assert [(m["src"], m["dst"]) for m in lines] == [("A", "T"), ("B", "T"), ("C", "T"), ("T", "C"), ("T", "B"), ("T", "A")]
    assert [m["src_up"] for m in lines[3:]] == [2 * h, h, 0.0]
    ses.fake = True

    ses.tower("T", h, "loop", rounds=2, rehearse=True)  # a rehearsal of two rounds: 12 moves in the air, no block touched
    assert until(lambda: ses.worker is None, 90), ses.state()["job"]
    assert ses.job["outcome"] == "done" and ses.job["done"] == 12 and ses.stack["n"] == 0


def test_a_block_that_meets_the_tower_early_is_let_go_there_but_a_far_obstacle_ends_the_run(ses):
    ready(ses)
    sp, a, h = ses.spots, ses.arm, 30.0
    top = sp.grasp("T") + h * sp.up  # the tower turns out 3 mm higher than the block height says (level 2 rests early)
    a.blocked = lambda p: np.linalg.norm((p - top) - ((p - top) @ sp.up) * sp.up) < 15.0 and float((p - top) @ sp.up) < 3.0 and a.holding is not None and a.holding[0] == "B"
    ses.grip.hold()
    ses.tower("T", h, "build")
    assert until(lambda: ses.worker is None, 60), ses.state()["job"]
    assert ses.job["outcome"] == "done" and ses.stack["n"] == 3, (ses.job["outcome"], ses.stack)
    assert any("came to rest" in line for line in ses.lines)
    assert float((a.bricks["B"] - top) @ sp.up) > 2.0  # it was let go where it met the tower, not pushed down to the planned height

    a.blocked = lambda p: np.linalg.norm((p - top) - ((p - top) @ sp.up) * sp.up) < 15.0 and float((p - top) @ sp.up) < 25.0 and a.holding is not None and a.holding[0] == "B"
    ses.tower("T", h, "build")  # something stands 25 mm above the place: that is not a put-down
    assert until(lambda: ses.worker is None, 60), ses.state()["job"]
    assert ses.job["outcome"].startswith("blocked at 'lower it onto T") and ses.stack["n"] is None, (ses.job["outcome"], ses.stack)
    assert a.holding is not None and a.holding[0] == "B"  # lifted clear with the block still in the gripper
    with pytest.raises(Refused, match="how many"):
        ses.tower("T", h, "build", n_now=ses.stack["n"])


def test_the_gripper_is_pointed_down_in_small_watched_pieces_before_any_spot(ses):
    ses.start("stand-in")
    a = ses.arm
    a.rpy = np.array([117.72, -41.15, 70.75])  # how the lab's teleoperation leaves the gripper
    tilt = lambda: float(np.degrees(np.arccos(-rpy_matrix(a.rpy)[2, 2])))  # noqa: E731
    was = tilt()
    assert 65.0 < was < 75.0
    ses.grip.letgo()
    assert until(lambda: all(c["ok"] for c in ses.state()["checks"]))
    with pytest.raises(Refused, match="arm first"):
        ses.level()
    ses.arm_on()
    place, heading = a.p.copy(), rpy_matrix(a.rpy)
    ses.level()  # nothing moves while the grip is released
    time.sleep(0.2)
    assert not [e for e in a.log if e[0] == "tilt"] and ses.worker is not None
    ses.grip.hold()
    assert until(lambda: ses.worker is None, 30), ses.state()["calibration"]
    pieces = [e for e in a.log if e[0] == "tilt"]
    assert tilt() < 0.3 and "points straight down now" in ses.note and len(pieces) == 14, (tilt(), ses.note, len(pieces))  # about 70 degrees, 5 at a time
    assert np.allclose(a.p, place) and not [e for e in a.log if e[0] == "move"]  # the flange stayed where it was
    assert twist_deg(heading, rpy_matrix(a.rpy))[1] == pytest.approx(was, abs=0.5)  # the shortest way down, no spin on top
    assert np.allclose(ses.guard.lim.axes[:, 2], [0.0, 0.0, 1.0], atol=1e-3)  # "up" for the nudges is straight up now
    ses.level()
    assert until(lambda: ses.worker is None, 10) and "already points straight down" in ses.note
    Session.save_spot(ses, "A")
    with pytest.raises(Refused, match="remove the saved spots"):
        ses.level()


def test_an_arm_that_re_poses_itself_while_tilting_is_stopped_at_once(ses):
    ses.start("stand-in")
    a = ses.arm
    a.rpy, a.swing_deg = np.array([150.0, 0.0, 0.0]), 40.0  # a joint jumps 40 degrees during a 5 degree tilt
    ses.grip.letgo()
    assert until(lambda: all(c["ok"] for c in ses.state()["checks"]))
    ses.arm_on()
    ses.grip.hold()
    ses.level()
    assert until(lambda: ses.phase() == "STOPPED", 10), ses.phase()
    assert "re-posing itself" in ses.ended and len([e for e in a.log if e[0] == "tilt"]) == 1 and a.stopped


def test_straight_down_follows_how_the_arm_is_mounted(ses):
    from assemblyloop.session import default_arms, mount_down

    arms = default_arms()
    assert np.allclose(arms["right"]["down"], [0.0, -0.7071, -0.7071], atol=1e-4) and np.allclose(arms["left"]["down"], [0.0, 0.7071, -0.7071], atol=1e-4)
    assert np.allclose(mount_down("right", 0.0), [0.0, 0.0, -1.0]) and "down" not in arms["stand-in"]
    ses.arms["stand-in"]["down"] = mount_down("right")  # the stand-in arm, mounted like the rig's right arm
    ses.start("stand-in")
    a = ses.arm
    a.rpy = np.array([180.0, 0.0, 14.5])  # pointing along the base's own -z: 45 degrees off for this mount
    ses.grip.letgo()
    assert until(lambda: all(c["ok"] for c in ses.state()["checks"]))
    ses.arm_on()
    assert until(lambda: ses.state()["arm"].get("tilt_deg") == pytest.approx(45.0, abs=0.1)), ses.state()["arm"]
    ses.grip.hold()
    ses.level()
    assert until(lambda: ses.worker is None, 30) and "it was tilted 45 deg" in ses.note, ses.note
    assert np.allclose(tool_axis(a.rpy), mount_down("right"), atol=2e-3)
    assert until(lambda: ses.state()["arm"].get("tilt_deg") == pytest.approx(0.0, abs=0.2))
    assert np.allclose(ses.guard.lim.axes[:, 2], -mount_down("right"), atol=2e-3) and np.allclose(ses.guard.lim.axes[:, 0], [1.0, 0.0, 0.0], atol=2e-3)  # up is up, forward is still forward


def test_the_space_bar_may_hold_a_real_arm_only_when_the_program_was_started_for_it(tmp_path):
    arms = {"right": dict(label="Right arm", ip="192.0.2.10", make=FakeArm)}  # a real arm as far as the session knows
    kw = dict(spots_path=tmp_path / "spots.json", log_dir=tmp_path / "logs", probe=lambda ip: False, guard_kw=dict(poll_s=0.002), limit_kw=dict(settle_s=0.0), quest_port=0)
    s = Session(arms, **kw)
    try:
        assert s.state()["key_hold"] is False
        with pytest.raises(Refused, match="key_hold"):
            s.start("right", hold="page")
    finally:
        s.close()
    s = Session(arms, key_hold=True, **kw)
    try:
        s.start("right", hold="page")
        assert s.state()["key_hold"] and s.hold == "page" and not s.fake and s.phase() == "DISARMED"
        assert not s.state()["checks"][1]["ok"]  # no signal yet: the page has not reported the key
        s.dm.feed(False)
        assert until(lambda: all(c["ok"] for c in s.state()["checks"]))
        s.arm_on()
        s.dm.feed(True)
        assert s.gate.held()
        time.sleep(2.6)  # the page stopped reporting (closed or frozen): that is "released"
        assert not s.gate.held() and s.dm.why().startswith("no signal")
        s.end()
        s.start("right")  # the Quest grip is still what a real arm gets when nothing is said
        assert s.hold == "quest"
    finally:
        s.close()


def test_a_short_silence_of_the_pages_key_is_forgiven_and_a_block_left_in_the_gripper_is_put_down(ses):
    from assemblyloop.deadman import Deadman

    t = [0.0]
    d = Deadman(timeout_s=0.6, clock=lambda: t[0], forgive_s=3.0)
    d.feed(False)
    d.feed(True)
    assert d.held()
    t[0] += 1.5  # the reports stalled for a second and a half: nothing is held meanwhile
    assert not d.held() and d.why().startswith("no signal")
    d.feed(True)  # ... and the key, still down, counts again without being let go first
    assert d.held()
    t[0] += 3.5  # a long silence is a new start: release, then press
    d.feed(True)
    assert not d.held() and d.why().startswith("release")
    strict = Deadman(timeout_s=0.25, clock=lambda: t[0])  # the Quest's rule is unchanged
    strict.feed(False)
    strict.feed(True)
    t[0] += 0.3
    strict.feed(True)
    assert not strict.held()

    ready(ses)
    a, sp = ses.arm, ses.spots
    with pytest.raises(Refused, match="no spot"):
        ses.put_held("nowhere")
    a.p = a.plan = sp.above(sp.grasp("C")) - 30.0 * sp.up  # stopped in the air on the way down, the block in the gripper
    a.bricks = {"thing": a.p.copy()}
    a.counts, a.holding = 315, ("thing", np.zeros(3))
    ses.thing_at = None
    ses.grip.hold()
    ses.put_held("B")
    assert until(lambda: ses.worker is None, 30), ses.state()["job"]
    assert ses.job["outcome"] == "done" and ses.thing_at == "B" and a.holding is None, (ses.job, ses.thing_at)
    assert np.linalg.norm(a.bricks["thing"] - sp.grasp("B")) < 0.6 and ses.phase() == "ARMED"
    ses.page_hold(False, 1000.0)
    time.sleep(0.5)
    ses.page_hold(False, 1050.0)  # sent on time, came in late
    assert "held up on the way into this program" in list(ses.lines)[-1]
    time.sleep(0.5)
    ses.page_hold(False, 1560.0)
    assert "the browser stalled" in list(ses.lines)[-1]


def test_nothing_is_started_towards_a_place_the_arm_cannot_reach(ses):
    ready(ses)
    a = ses.arm
    a.reach_mm = 525.0  # spot T (at 520, -40) and what is above it are just out of reach; A, B and C are not
    ses.grip.hold()
    n = len(moves(ses))
    ses.move("T")
    assert until(lambda: ses.worker is None, 10)
    assert "cannot reach that far" in ses.job["outcome"] and "Nothing was moved" in ses.job["outcome"] and len(moves(ses)) == n, ses.job["outcome"]
    assert ses.phase() == "ARMED" and ses.connected  # refused, not a fault: the session goes on
    ses.set_thing("A")
    ses.move("B")
    assert until(lambda: ses.worker is None, 30) and ses.job["outcome"] == "done", ses.job["outcome"]
    with pytest.raises(Refused, match="cannot reach"):
        ses.put_held("T")
    ses.calibrate_begin()
    a.reach_mm = float(np.linalg.norm(a.pose()[:3])) + 10.0
    with pytest.raises(Refused, match="cannot reach"):
        ses.jog("fwd", 50)  # a nudge that would leave the arm's reach is not started either
    ses.jog("back", 20)
    assert until(lambda: ses.worker is None, 10)
    assert ses.state()["hint"].startswith("Show it the places: switch MOVING on")  # with the page's key the hint says the key


def test_saved_spots_are_taken_from_a_little_beside_them_and_a_move_comes_back_in(ses):
    ready(ses)
    a, sp = ses.arm, ses.spots
    sp.save(ses.spots_path)
    ses.calibrate_begin()
    a.p = a.plan = sp.above(sp.grasp("A")) + np.array([-150.0, 0.0, 0.0])  # stopped while it stood aside: 90 mm outside the box around the spots
    assert sp.limits().why_outside(a.p) and not sp.limits(start=a.p).why_outside(a.p)
    ses.draft = Spots(side=ses.chosen)
    ses.use_saved()
    assert ses.phase() == "ARMED"
    ses.set_thing("A")
    ses.grip.hold()
    ses.move("B")
    assert until(lambda: ses.worker is None, 30) and ses.job["outcome"] == "done", ses.job["outcome"]
    ses.calibrate_begin()
    a.p = a.plan = sp.above(sp.grasp("A")) + np.array([-600.0, 0.0, 0.0])  # nowhere near
    with pytest.raises(Refused, match="not over the saved spots"):
        ses.use_saved()
