"""Two arms in one session's world, and what a session writes down: stand-in arms only. The made-up rig here (two
upright arms 700 mm apart) is not the lab's."""
import json
import threading
import time

import numpy as np
import pytest

from assemblyloop.arms import ArmFault, FakeArm
from assemblyloop.deadman import ManualDeadman, QuestDeadman
from assemblyloop.guard import Guard, Limits, Refused
from assemblyloop.record import episodes, load
from assemblyloop.session import Session
from assemblyloop.spots import Spots

APART = np.array([0.0, 700.0, 0.0])  # the left arm's base, in the right arm's
MARKS = {"M1": (400.0, 300.0), "M2": (520.0, 300.0), "M3": (400.0, 420.0), "M4": (520.0, 400.0), "M5": (460.0, 350.0)}  # on the table, in the right arm's x and y


def until(cond, secs=8.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def ses(tmp_path):
    arms = {"right": dict(label="Right arm", ip=None, make=FakeArm, down=(0.0, 0.0, -1.0)), "left": dict(label="Left arm", ip=None, make=FakeArm, down=(0.0, 0.0, -1.0))}
    s = Session(arms, spots_path=tmp_path / "spots.json", log_dir=tmp_path / "logs", probe=lambda ip: False, fake_speed=25.0, guard_kw=dict(poll_s=0.002), limit_kw=dict(settle_s=0.0),
                rig_path=tmp_path / "rig.json", camera_path=tmp_path / "camera.json", blocks_path=tmp_path / "blocks.json")
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


def armed(s, side):
    s.pressed = False
    s.start(side)
    assert until(lambda: all(c["ok"] for c in s.state()["checks"])), s.state()["checks"]
    s.arm_on()


def put(arm, xyz):
    arm.p = np.array(xyz, float)
    arm.plan = arm.p.copy()


def teach_marks(s, side, shift, names=MARKS):
    """Touch every mark with one arm (the stand-in arm is simply put there) and save it as a spot of that name."""
    armed(s, side)
    for name in names:
        x, y = MARKS[name]
        put(s.arm, np.array([x, y, 172.0]) - shift)  # the fingertips on the table, which is at height 0 in the right arm's frame
        s.save_spot(name)
    s.calibrate_done()
    s.end()


def test_each_arm_keeps_its_own_spots_and_never_takes_the_others(ses, tmp_path):
    teach_marks(ses, "right", np.zeros(3))
    teach_marks(ses, "left", APART)
    assert Spots.load(tmp_path / "spots.right.json").side == "right" and Spots.load(tmp_path / "spots.left.json").side == "left"
    assert Spots.load(tmp_path / "spots.json").side == "right"  # the plain file is still the first arm's: the second did not write over it
    armed(ses, "left")
    assert ses.state()["calibration"]["saved"]["side"] == "left"
    put(ses.arm, np.array([460.0, 350.0, 200.0]) - APART)
    ses.use_saved()
    assert ses.spots.side == "left" and np.allclose(ses.spots.at["M1"], np.array([400.0, 300.0, 172.0]) - APART)
    ses.end()
    (tmp_path / "spots.left.json").unlink()
    armed(ses, "left")
    assert ses.state()["calibration"]["saved"] is None
    with pytest.raises(Refused, match="no spots were saved with the left arm"):
        ses.use_saved()


def test_the_arms_are_fitted_to_each_other_from_marks_both_have_touched(ses, tmp_path):
    with pytest.raises(Refused, match="no spots were saved with the right arm"):
        ses.pair_fit()
    teach_marks(ses, "right", np.zeros(3))
    teach_marks(ses, "left", APART, names=("M1", "M2", "M3"))
    with pytest.raises(Refused, match="share only 3"):
        ses.pair_fit()
    teach_marks(ses, "left", APART)
    frame = ses.pair_fit()
    assert np.allclose(frame.t, APART, atol=1e-6) and np.allclose(frame.R, np.eye(3), atol=1e-9) and frame.rms_mm < 1e-6 and frame.n == 5
    assert np.allclose(ses.rig().point([10.0, 20.0, 30.0]), np.array([10.0, 20.0, 30.0]) + APART) and ses.state()["pair"]["fitted"]
    wrong = Spots.load(tmp_path / "spots.left.json")  # two marks mixed up with one arm: the fit says so instead of being kept
    wrong.at["M1"], wrong.at["M2"] = wrong.at["M2"], wrong.at["M1"]
    wrong.save(tmp_path / "spots.left.json")
    (tmp_path / "rig.json").unlink()
    with pytest.raises(Refused):
        ses.pair_fit()
    assert not (tmp_path / "rig.json").exists()


def test_one_arm_takes_over_what_the_other_was_taught(ses, tmp_path):
    teach_marks(ses, "right", np.zeros(3))
    armed(ses, "left")
    with pytest.raises(Refused, match="not been fitted"):
        ses.use_other()
    ses.end()
    (tmp_path / "rig.json").write_text(json.dumps(dict(left_in_right=dict(R=np.eye(3).tolist(), t=APART.tolist(), rms_mm=0.4, n=5, how="marks", when=0.0))))
    armed(ses, "left")
    put(ses.arm, np.array([460.0, 350.0, 260.0]) - APART - [0.0, 300.0, 0.0])
    with pytest.raises(Refused, match="not over those spots"):
        ses.use_other()
    put(ses.arm, np.array([460.0, 350.0, 200.0]) - APART)
    ses.use_other()
    assert ses.phase() == "ARMED" and ses.spots.side == "left" and sorted(ses.spots.at) == sorted(MARKS)
    assert all(np.allclose(ses.spots.at[n], np.array([x, y, 172.0]) - APART) for n, (x, y) in MARKS.items())  # the same places on the table, in the left arm's own terms
    assert Spots.load(tmp_path / "spots.left.json").side == "left" and Spots.load(tmp_path / "spots.json").side == "right"
    assert any("REHEARSE IN THE AIR FIRST" in line for line in ses.lines)


def test_the_other_arm_is_watched_and_this_one_keeps_away_from_it(ses, tmp_path):
    armed(ses, "right")
    assert ses.watch is None and not ses.state()["pair"]["watched"] and any("NOT watched" in line for line in ses.lines)  # nothing is known about the other arm: a person keeps it away
    ses.end()
    (tmp_path / "rig.json").write_text(json.dumps(dict(left_in_right=dict(R=np.eye(3).tolist(), t=APART.tolist(), rms_mm=0.4, n=5, how="marks", when=0.0))))
    armed(ses, "right")
    other, pair = ses.other, ses.state()["pair"]
    assert other is not None and ses.other_side == "left" and pair["watched"] and pair["mm"] > 400 and not other.enabled and not other.log  # connected, read only
    ses.pressed = True
    ses.jog("left", 20)
    assert until(lambda: ses.worker is None) and ses.job["outcome"] == "done"  # far apart: nothing is in the way
    put(other, np.array([450.0, 20.0 + 190.0 + 90.0, 300.0]) - APART)  # the other gripper is now 90 mm from this one (each is 95 mm thick)
    assert until(lambda: 80 < (ses.state()["pair"]["mm"] or 0) < 100), ses.state()["pair"]
    here = ses.arm.p.copy()
    ses.jog("left", 25)  # towards it
    assert until(lambda: ses.worker is None)
    assert "Refused" in ses.job["outcome"] and "could bring them closer than 60 mm" in ses.job["outcome"], ses.job["outcome"]
    assert float(np.linalg.norm(ses.arm.p - here)) < 25.0 and ses.watch.mm()[0] >= 60.0  # it ended short of where it was sent, still clear of the other arm
    assert ses.phase() == "AWAITING CALIBRATION" and ses.guard.latched is None  # a refusal, not a fault
    ses.jog("right", 5)  # a small step is still allowed
    assert until(lambda: ses.worker is None) and ses.job["outcome"] == "done", ses.job["outcome"]
    other.connected = False  # the other arm is lost: nothing is known about it any more, and that counts as too close
    ses.jog("up", 5)
    assert until(lambda: ses.worker is None) and "Refused" in ses.job["outcome"], ses.job["outcome"]
    assert ses.state()["pair"]["mm"] is None and ses.state()["pair"]["why"]
    ses.end()
    assert not other.enabled and not [e for e in other.log if e[0] in ("move", "turn", "tilt", "grip", "enable")]  # the other arm was never asked to do anything


def test_an_arm_that_comes_too_close_while_this_one_moves_ends_the_run():
    arm = FakeArm()
    dm = ManualDeadman(clock=lambda: arm.t)
    dm.press()

    class Near:
        gap = 300.0

        def why(self, step_mm):
            return "" if self.gap - 2 * step_mm >= 60.0 else f"the other arm is {self.gap:.0f} mm away"

        def max_step(self, mm):
            return min(mm, max(0.0, (self.gap - 60.0) / 2.0))

        def too_close(self):
            return "" if self.gap >= 60.0 else f"the other arm is only {self.gap:.0f} mm away"

    near, intrude = Near(), []

    def sleep(dt):
        arm.advance(dt)
        if intrude and arm.t > intrude[0] + 0.4:
            near.gap = 40.0  # it moves in while this arm is under way

    g = Guard(arm, dm, Limits(np.zeros(3), np.eye(3), np.array([0.0, -500.0, 0.0]), np.array([900.0, 500.0, 900.0]), max_step_mm=80.0), clock=lambda: arm.t, sleep=sleep, say=lambda *_: None, watch=near)
    g.start()
    near.gap = 100.0
    g.move_to(arm.pose()[:3] + [0.0, 60.0, 0.0], 100.0, "near the other arm")
    assert [round(float(np.linalg.norm(e[2])), 3) for e in arm.log if e[0] == "move"] == [20.0, 20.0, 20.0]  # cut shorter than the 80 mm it may otherwise take
    near.gap = 300.0
    intrude.append(arm.t)
    with pytest.raises(ArmFault, match="only 40 mm away"):
        g.move_to(arm.pose()[:3] + [0.0, 200.0, 0.0], 100.0, "across")
    assert g.latched and arm.stopped and ("stop", arm.t) in [(e[0], e[1]) for e in arm.log[-2:]]


def test_a_job_is_written_down_as_an_episode_a_policy_can_learn_from(ses, tmp_path):
    armed(ses, "right")
    for name, xyz in (("A", (430.0, -90.0, 120.0)), ("B", (470.0, 20.0, 120.0))):
        put(ses.arm, xyz)
        ses.save_spot(name)
    ses.calibrate_done()
    ses.arm.bricks = {"thing": np.array([430.0, -90.0, 120.0])}
    ses.set_thing("A")
    ses.pressed = True
    ses.move("B")
    assert until(lambda: ses.worker is None, 60) and ses.job["outcome"] == "done", ses.job["outcome"]
    eps = episodes(tmp_path / "logs")
    assert len(eps) == 1
    ep = load(eps[0])
    assert ep.finished and ep.success and ep.task == "move the block from spot A to spot B" and ep.meta["meta"]["arm"] == "right" and ep.meta["meta"]["fake"] is True
    rows = ep.rows
    assert len(rows) >= 5 and all(set(r["arms"]) == {"right"} and len(r["arms"]["right"]["joints"]) == 7 and len(r["arms"]["right"]["xyz"]) == 3 for r in rows)
    assert {r["arms"]["right"]["grip"] for r in rows} >= {850, 315} and any(r["arms"]["right"]["goal_xyz"] for r in rows) and any(r["held"] for r in rows)
    assert {0, 850} <= {r["arms"]["right"]["goal_grip"] for r in rows}  # what the gripper was asked for, not only where it came to rest
    steps = [e["name"] for e in ep.events if e["kind"] == "step"]
    assert steps[:3] == ["go to the travel height", "go above A", "turn the gripper as at A"] and steps[-1] == "lift clear" and [e["text"] for e in ep.events if e["kind"] == "leg"] == ["A to B"]
    assert ses.state()["rec"]["on"] and not ses.state()["rec"]["active"]
    ses.judge(False)  # a person's word counts for more than "the steps ran"
    assert load(eps[0]).success is False
    ses.set_recording(False)
    ses.move("A")
    assert until(lambda: ses.worker is None, 60) and len(episodes(tmp_path / "logs")) == 1
    ses.set_recording(True)
    ses.move("B", rehearse=True)
    assert until(lambda: ses.worker is None, 60)
    last = load(episodes(tmp_path / "logs")[-1])
    assert last.meta["meta"]["rehearse"] is True and last.success is False  # a rehearsal in the air is never taken for a demonstration


def test_only_the_first_headset_heard_is_the_switch():
    now = [0.0]
    dm = QuestDeadman(port=0, clock=lambda: now[0] + 0.01)
    def got(data, t, addr):
        now[0] = t
        return dm.on_packet(data, t=t, addr=addr)

    frame = lambda seq, grip: json.dumps(dict(seq=seq, right=dict(grip=grip))).encode()  # noqa: E731
    assert got(frame(1, 0.0), 0.00, ("192.0.2.7", 5000)) and got(frame(2, 1.0), 0.02, ("192.0.2.7", 5000)) and dm.held()
    assert got(frame(3, 0.0), 0.04, ("192.0.2.7", 5000)) and not dm.held()  # the operator lets go
    assert not got(frame(900, 1.0), 0.05, ("192.0.2.99", 5000)) and not dm.held() and dm.strangers == 1  # someone else's "held" does not count
    assert not got(json.dumps(dict(right=dict(grip=1.0))).encode(), 0.06, ("192.0.2.7", 5000)) and not dm.held()  # nor does a frame without a number
    assert got(frame(4, 1.0), 0.07, ("192.0.2.7", 5001)) and dm.held()  # the same headset from another port still is the switch
