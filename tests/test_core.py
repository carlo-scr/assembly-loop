"""The assembly loop's safety rules (assemblyloop/), checked on the stand-in arm in simulated time. Nothing here needs the
robot or its SDK."""
import json
import socket
import sys

import numpy as np
import pytest

from assemblyloop.arms import ArmFault, FakeArm, flange_mm, matrix_rpy, rpy_matrix, twist_deg
from assemblyloop.cycle import NothingHeld, Params, execute, plan
from assemblyloop.deadman import Deadman, ManualDeadman, QuestDeadman, discovery_reply
from assemblyloop.guard import Blocked, Guard, Refused, Released
from assemblyloop.plate import Plate, demo


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def rig(src="C3", script=None, **kw):
    """Plate, stand-in arm with a brick seated on `src`, switch, guard. `script(t)` says whether the switch is pressed
    at simulated time t (default: always)."""
    plate = Plate.from_taught(demo())
    arm = FakeArm(pose=[*plate.origin, *plate.rpy], bricks={"b": plate.grasp(src)}, up=plate.n)
    dm = ManualDeadman(clock=lambda: arm.t)
    dm.press()
    trace = []

    def sleep(dt):
        arm.advance(dt)
        if script is not None:
            (dm.press if script(arm.t) else dm.release)()
        trace.append((arm.t, arm.plan.copy(), arm.p.copy()))

    g = Guard(arm, dm, plate.limits(), clock=lambda: arm.t, sleep=sleep, say=lambda *_: None, **kw)
    return plate, arm, dm, g, trace


def moves(arm):
    return [e for e in arm.log if e[0] == "move"]


# ------------------------------------------------------------------------------------------------ hold-to-run
def test_a_press_counts_only_after_a_release_and_only_while_fresh():
    c = Clock()
    d = Deadman(timeout_s=0.25, clock=c)
    assert not d.held() and "no signal" in d.why()
    d.feed(True)
    assert not d.held()  # squeezed before anything was seen released: does nothing
    d.feed(False)
    d.feed(True)
    assert d.held()
    c.t += 0.2
    assert d.held()
    c.t += 0.1
    assert not d.held()  # 0.3 s without a signal
    d.feed(True)
    assert not d.held()  # the signal came back already squeezed: release first
    d.feed(False)
    d.feed(True)
    assert d.held()
    d.feed(False)
    assert not d.held()


def test_quest_frames_grip_hand_sequence_and_rubbish():
    c = Clock()
    q = QuestDeadman(hand="right", clock=c)
    frame = lambda seq, right=0.0, left=0.0: json.dumps(dict(ver=1, seq=seq, left=dict(grip=left), right=dict(grip=right))).encode()  # noqa: E731
    assert q.on_packet(frame(1)) and not q.held()
    assert q.on_packet(frame(2, right=0.9)) and q.held()
    assert not q.on_packet(frame(2, right=0.9))  # the same frame again is not a fresh signal
    c.t += 0.3
    assert not q.held()  # only repeats arrived: stale
    assert q.on_packet(frame(3, left=1.0)) and not q.held()  # the other hand's grip does not count
    assert q.on_packet(frame(4, right=0.49)) and not q.held()
    assert q.on_packet(frame(5, right=0.5)) and q.held()
    for junk in (b"", b"not json", b"[1, 2]", b'{"right": 3}'):
        q.on_packet(junk)
    assert not q.on_packet(b"\xff\xfe")
    assert QuestDeadman(hand="either", clock=c).on_packet(frame(1, left=1.0))


def test_the_headsets_search_is_answered_and_is_never_taken_for_a_grip():
    r = json.loads(discovery_reply(b"DCOL-DISCOVER/1 abc\n", 6666, "pc (assembly-loop)"))
    assert (r["proto"], r["type"], r["teleop_port"], r["nonce"], r["name"]) == (1, "dcol-collector", 6666, "abc", "pc (assembly-loop)")
    assert json.loads(discovery_reply(b"DCOL-DISCOVER/1", 6666, "pc"))["nonce"] == ""
    for other in (b"DCOL-DISCOVER/11 abc", b"DCOL-DISCOVER/2", b'{"seq": 1, "right": {"grip": 1.0}}', b"", b"DCOL-DISCOVER/1 " + b"x" * 600):
        assert discovery_reply(other, 6666, "pc") is None
    q = QuestDeadman(port=0, host="127.0.0.1").start()
    port = q._sock.getsockname()[1]
    try:
        assert "open the connect screen" in q.why()
        quest = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        quest.settimeout(2.0)
        quest.sendto(b"DCOL-DISCOVER/1 n7", ("127.0.0.1", port))
        r = json.loads(quest.recvfrom(4096)[0])
        assert r["nonce"] == "n7" and r["name"] == q.name and r["busy"] is False
        assert q.n == 0 and q.searches == 1 and q.why().startswith("no signal") and q.name in q.why()
        quest.close()
    finally:
        q.stop()


# ------------------------------------------------------------------------------------------------ the guard
def test_a_target_outside_the_volume_or_too_fast_is_refused_before_anything_is_sent():
    plate, arm, dm, g, _ = rig()
    g.start()
    with pytest.raises(Refused):
        g.move_to(plate.grasp("C3") - 10.0 * plate.n, 30.0, "below the plate")
    with pytest.raises(Refused):
        g.move_to(plate.hover("C3") + 300.0 * plate.f(), 30.0, "off the plate")
    with pytest.raises(Refused):
        g.move_to(plate.hover("E3"), 121.0, "too fast")
    assert moves(arm) == []
    with pytest.raises(Refused):
        plate.hover("H8")  # a cell outside the taught area


def test_nothing_moves_while_the_switch_is_not_held():
    plate, arm, dm, g, _ = rig(start_s=2.0)
    g.start()
    dm.release()
    with pytest.raises(Released):
        g.move_to(plate.hover("E3"), 60.0, "travel")
    assert moves(arm) == [] and arm.stopped
    with pytest.raises(ArmFault):  # the run is over: nothing further is accepted, pressed or not
        dm.press()
        g.move_to(plate.hover("E3"), 60.0, "travel")
    assert moves(arm) == []


def test_releasing_mid_move_pauses_within_one_poll_and_pressing_resumes():
    plate, arm, dm, g, trace = rig(script=lambda t: not (0.5 <= t < 1.5))
    g.start()
    g.move_to(plate.hover("F3"), 60.0, "travel")  # 95 mm
    pauses = [e[1] for e in arm.log if e[0] == "pause"]
    resumes = [e[1] for e in arm.log if e[0] == "resume"]
    assert len(pauses) == 1 and 0.5 <= pauses[0] <= 0.5 + g.poll_s + 1e-9
    assert len(resumes) == 1 and 1.5 <= resumes[0] <= 1.5 + g.poll_s + 1e-9
    held_still = [p for t, p, _ in trace if pauses[0] < t <= resumes[0]]
    assert np.ptp(np.array(held_still), axis=0).max() == 0.0  # not a millimetre while released
    assert np.linalg.norm(arm.p - plate.hover("F3")) < 1e-6


def test_staying_released_stops_the_arm_and_ends_the_run():
    plate, arm, dm, g, _ = rig(script=lambda t: t < 0.5, resume_s=3.0)
    g.start()
    with pytest.raises(Released):
        g.move_to(plate.hover("F3"), 60.0, "travel")
    assert arm.stopped and g.latched and 3.5 <= arm.t <= 3.6
    assert np.linalg.norm(arm.p - plate.hover("F3")) > 50.0  # it did not go on
    n = len(arm.log)
    with pytest.raises(ArmFault):
        g.grip(0, "close")
    assert len(arm.log) == n


def test_a_controller_that_does_not_take_the_pause_or_the_resume_is_stopped():
    for broken, why in (("pause", "the pause was not taken"), ("resume", "could not be taken up again")):
        plate, arm, dm, g, _ = rig(script=lambda t: not (0.5 <= t < 1.5))

        def refuse(broken=broken):
            raise ArmFault(f"{broken} was refused by the controller (code 9)")

        setattr(arm, broken, refuse)
        g.start()
        with pytest.raises(ArmFault, match=why):
            g.move_to(plate.hover("F3"), 60.0, "travel")
        assert arm.stopped and arm.target is None and g.latched  # nothing is left waiting in the controller for whoever enables the arm next
        with pytest.raises(ArmFault):
            g.recover()


def test_a_cycle_is_never_lowered_below_where_it_was_taught():
    plate = Plate.from_taught(demo())
    for air in (-30.0, float("nan"), 1e9):
        with pytest.raises(Refused, match="never lowered"):
            plan(plate, "C3", "E3", air_mm=air)


def test_a_controller_error_stops_and_latches():
    plate, arm, dm, g, _ = rig()
    g.start()
    arm.fault_when = lambda a: 31 if a.t > 0.4 else 0
    with pytest.raises(ArmFault, match="31"):
        g.move_to(plate.hover("F3"), 60.0, "travel")
    assert arm.stopped and "31" in g.latched
    n = len(arm.log)
    for ask in (lambda: g.move_to(plate.hover("C3"), 30.0, "back"), lambda: g.grip(850, "open"), g.start, g.recover):
        with pytest.raises(ArmFault):
            ask()
    assert len(arm.log) == n  # not one more command after the fault


def test_the_run_does_not_start_if_measured_and_planned_joints_disagree_at_rest():
    plate, arm, dm, g, _ = rig()
    arm.p = arm.p + np.array([0.0, 0.0, 2.0])  # the stall watch's own sensor reads 2 mm on a free arm at rest
    with pytest.raises(ArmFault, match="at rest"):
        g.start()
    assert moves(arm) == [] and g.latched


# ------------------------------------------------------------------------------------------------ turning the gripper
def turns(arm):
    return [e for e in arm.log if e[0] == "turn"]


def test_turning_spins_the_last_joint_only_in_checked_pieces():
    plate, arm, dm, g, _ = rig()
    g.start()
    at = arm.p.copy()
    g.turn_by(35.0, "turn")
    assert moves(arm) == [] and np.allclose(arm.p, at)  # the gripper spun in place: no Cartesian move was sent
    assert len(turns(arm)) == 4 and max(abs(e[2]) for e in turns(arm)) <= 10.0 + 1e-9 and max(e[3] for e in turns(arm)) <= 20.0
    assert abs(arm.j7 - 35.0) < 1e-6 and abs(abs(twist_deg(rpy_matrix([180.0, 0.0, 0.0]), rpy_matrix(arm.rpy))[0]) - 35.0) < 1e-6
    n = len(arm.log)
    with pytest.raises(Refused, match="half a turn"):
        g.turn_by(200.0, "too far")
    arm.j7 = arm.j7_real = 330.0
    with pytest.raises(Refused, match="limit"):
        g.turn_by(20.0, "past the joint's limit")
    assert len(arm.log) == n


def test_letting_go_pauses_a_turn_and_a_gripper_that_is_held_ends_it():
    plate, arm, dm, g, trace = rig(script=lambda t: not (0.6 <= t < 1.6))
    g.start()
    g.turn_by(60.0, "turn")  # 3 s at 20 deg/s
    pauses = [e[1] for e in arm.log if e[0] == "pause"]
    assert len(pauses) == 1 and 0.6 <= pauses[0] <= 0.6 + g.poll_s + 1e-9 and abs(arm.j7 - 60.0) < 1e-6
    plate, arm, dm, g, trace = rig()
    g.start()
    arm.turn_stuck = True  # something holds the gripper: the joint's plan goes on, the joint does not
    with pytest.raises(Blocked, match="holding"):
        g.turn_by(60.0, "turn")
    assert arm.stopped and not g.latched and arm.j7 <= g.lim.turn_step_deg + 1e-6  # one piece into it, no further
    g.recover()


def test_turning_to_a_taught_orientation_whichever_way_the_joint_spins_it():
    for sign in (1.0, -1.0):
        plate, arm, dm, g, _ = rig()
        arm.turn_sign = sign
        g.start()
        g.turn_to([180.0, 0.0, 90.0], "turn as taught")
        assert abs(twist_deg(rpy_matrix(arm.rpy), rpy_matrix([180.0, 0.0, 90.0]))[0]) <= 1.0
        assert abs(turns(arm)[0][2]) <= 5.0 + 1e-9  # the first turn is a small one: which way it goes is looked at, not assumed
        assert g.turn_sign == sign and moves(arm) == []
        g.turn_to([180.0, 0.0, -45.0], "and back")
        assert abs(twist_deg(rpy_matrix(arm.rpy), rpy_matrix([180.0, 0.0, -45.0]))[0]) <= 1.0
        with pytest.raises(Refused, match="axis"):
            g.turn_to([170.0, 0.0, 0.0], "a tilt is not a turn")
    for sign in (1.0, -1.0):  # almost half a turn: which way the joint spins it must not be misread where 180 becomes -180
        for yaw in (178.0, -178.0):
            plate, arm, dm, g, _ = rig()
            arm.turn_sign = sign
            g.start()
            g.turn_to([180.0, 0.0, yaw], "end for end")
            assert abs(twist_deg(rpy_matrix(arm.rpy), rpy_matrix([180.0, 0.0, yaw]))[0]) <= 1.0 and g.turn_sign == sign and g.latched is None
            assert sum(abs(t[2]) for t in turns(arm)) < 200.0  # and not the long way round


def test_orientation_helpers():
    for rpy in ([180.0, 0.0, 0.0], [135.0, 10.0, -70.0], [-170.0, -25.0, 160.0]):
        assert np.allclose(rpy_matrix(matrix_rpy(rpy_matrix(rpy))), rpy_matrix(rpy), atol=1e-9)
    assert np.allclose(rpy_matrix([180.0, 0.0, 33.0])[:, 2], [0.0, 0.0, -1.0])  # pointing down whatever the yaw
    spin, tilt = twist_deg(rpy_matrix([135.0, 0.0, 20.0]), rpy_matrix([135.0, 0.0, 20.0]) @ rpy_matrix([0.0, 0.0, 40.0]))
    assert abs(spin - 40.0) < 1e-9 and tilt < 1e-6


# ------------------------------------------------------------------------------------------------ the cycle
def test_a_cycle_stays_inside_its_limits_and_puts_the_brick_where_the_aim_says():
    for corr, want in ((6.0, 0.0), (0.0, 6.0)):
        plate, arm, dm, g, _ = rig()
        P = Params()
        g.start()
        out = execute(g, plan(plate, "C3", "E3", 6.0, corr, P), P)
        g.finish()
        assert out["blocked"] is None and set(out["reads"]) == {"read the grasp", "read the carry", "read the seat"}
        mv = moves(arm)
        assert max(np.linalg.norm(e[2]) for e in mv) <= g.lim.max_step_mm + 1e-6
        assert max(e[3] for e in mv) <= P.travel == 100.0
        assert all(g.lim.why_outside(e[4]) == "" for e in mv)
        off = arm.bricks["b"] - plate.grasp("E3")
        assert abs(np.linalg.norm(off - (off @ plate.n) * plate.n) - want) < 0.05
        assert float(off @ plate.n) >= -P.overtravel_mm - 1e-6  # never pushed further than the planned overtravel
        assert arm.stopped  # left stopped


def test_closing_on_nothing_opens_lifts_clear_and_says_so():
    plate, arm, dm, g, _ = rig(src="G7")  # the brick is somewhere else
    g.start()
    with pytest.raises(NothingHeld):
        execute(g, plan(plate, "C3", "E3"), Params())
    assert arm.counts == 850 and np.linalg.norm(arm.p - plate.hover("C3")) < 1e-6


def test_a_press_onto_something_that_does_not_give_ends_within_a_few_millimetres():
    plate, arm, dm, g, trace = rig()
    P = Params()
    # at the target stud the brick lands on the stud tops, 19 mm above seated, and goes no further
    wall = lambda p: np.linalg.norm((p - plate.grasp("E3")) - ((p - plate.grasp("E3")) @ plate.n) * plate.n) < 12.0 and (p - plate.grasp("E3")) @ plate.n < 19.0  # noqa: E731
    arm.blocked = wall
    g.start()
    out = execute(g, plan(plate, "C3", "E3", 0.0, 0.0, P), P)
    assert out["blocked"] and out["blocked"]["step"] in ("lower onto the studs", "press home")
    ahead = max(float((p - q) @ plate.n) for _, q, p in trace)  # how far the plan ever got below the arm
    assert P.stall_mm < ahead <= g.lim.inch_mm + P.stall_mm + 1e-6
    assert arm.holding is not None  # the brick was not dropped
    assert abs(float((arm.p - plate.hover("E3")) @ plate.n)) < 1e-6  # lifted back to the hover height
    assert not g.latched  # a blocked move is not a fault


def test_a_brick_that_will_not_come_off_is_let_go():
    plate, arm, dm, g, trace = rig()
    P = Params()
    arm.blocked = lambda p: arm.holding is not None and (p - plate.grasp("C3")) @ plate.n > 1.0  # held down by its studs
    g.start()
    out = execute(g, plan(plate, "C3", "E3", 0.0, 0.0, P), P)
    assert out["blocked"]["step"].startswith("pull")
    assert arm.counts == 850 and np.linalg.norm(arm.bricks["b"] - plate.grasp("C3")) < 1.5
    ahead = max(float(np.linalg.norm(q - p)) for _, q, p in trace)
    assert ahead <= g.lim.inch_mm + P.pull_stall_mm + 1e-6
    assert not g.latched and abs(float((arm.p - plate.hover("C3")) @ plate.n)) < 1e-6  # let go, then lifted clear


def test_being_held_back_on_a_free_move_counts_as_a_collision_and_ends_the_run():
    plate, arm, dm, g, _ = rig()
    g.start()
    wall = plate.hover("C3") + 40.0 * plate.f()
    arm.blocked = lambda p: (p - wall) @ plate.f() > 0.0  # something stands in the way of the travel
    with pytest.raises(ArmFault, match="should meet nothing"):
        g.move_to(plate.hover("F3"), 60.0, "travel")
    assert arm.stopped and g.latched
    assert float(np.linalg.norm(arm.plan - arm.p)) <= g.lim.max_step_mm + 1e-6  # at most one segment into it


def test_a_rehearsal_in_the_air_starts_from_the_hover_height_and_cannot_go_below_it():
    plate, arm, dm, g, _ = rig()
    g.lim = plate.limits(air_mm=100.0)
    P = Params()
    g.start()
    steps = plan(plate, "C3", "E3", 4.0, 0.0, P, air_mm=100.0)
    out = execute(g, steps, P, air=True)  # the arm starts AT the hover height and climbs from there
    assert out["blocked"] is None and arm.holding is None
    assert min(float(g.lim.coords(e[4])[2]) for e in moves(arm)) >= -1e-6  # never below the hover height
    assert abs(min(float(g.lim.coords(x["to"])[2]) for x in steps if x["kind"] == "move") - (100.0 - plate.grasp_drop - P.overtravel_mm)) < 1e-6
    with pytest.raises(Refused):
        g.move_to(plate.hover("E3") - 5.0 * plate.n, 30.0, "below the hover height in a rehearsal")
    assert np.linalg.norm(arm.bricks["b"] - plate.grasp("C3")) < 1e-9  # the brick was never touched


def test_plans_refuse_what_they_should():
    plate = Plate.from_taught(demo())
    with pytest.raises(Refused):
        plan(plate, "C3", "E3", offset_mm=16.0)
    with pytest.raises(Refused):
        plan(plate, "C3", "E3", P=Params(travel=130.0))
    with pytest.raises(Refused):
        plan(plate, "C3", "E3", P=Params(overtravel_mm=3.0))
    bad = demo()
    bad["taught"]["col"]["hover_mm"][1] += 12.0  # a stud pitch of 35.75 mm
    with pytest.raises(Refused, match="pitch"):
        Plate.from_taught(bad)
    bad = demo()
    bad["taught"]["row"]["hover_mm"][1] += 8.0  # rows not square to the columns
    with pytest.raises(Refused, match="deg"):
        Plate.from_taught(bad)
    bad = demo()
    bad["grasp_mm"][0] += 6.0  # the grasp point is not straight below the hover point
    with pytest.raises(Refused, match="side"):
        Plate.from_taught(bad)


def test_the_dry_run_never_loads_the_arms_sdk(tmp_path):
    from assemblyloop.__main__ import main

    f = tmp_path / "plate.json"
    main(["demo-plate", "--out", str(f)])
    res = main(["plan", "--plate", str(f), "--src", "C3", "--dst", "E3", "--offset", "5", "--correct", "truth"])
    assert res["miss_mm"] < 0.05 and not res["xarm_loaded"] and "xarm" not in sys.modules
    with pytest.raises(SystemExit):  # the live command refuses without --live
        main(["run", "--plate", str(f), "--src", "C3", "--dst", "E3", "--ip", "192.0.2.1"])


def test_flange_position_from_joints_is_a_rigid_function():
    assert np.allclose(flange_mm([0.0] * 7), [206.0, 0.0, 120.5], atol=1e-6)  # the xArm 7's zero pose
    a = flange_mm([30.0, 10.0, 0.0, 40.0, 0.0, 20.0, 0.0])
    b = flange_mm([75.0, 10.0, 0.0, 40.0, 0.0, 20.0, 0.0])  # turning joint 1 turns the flange about the base's z axis
    assert abs(a[2] - b[2]) < 1e-9 and abs(np.hypot(*a[:2]) - np.hypot(*b[:2])) < 1e-9
