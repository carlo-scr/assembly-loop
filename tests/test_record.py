"""Recording: episodes on disk, pictures without a page, and the numbers a policy is trained on. A made-up robot, a
made-up clock and made-up cameras; only the last test needs OpenCV, and none needs the `lerobot` library."""
import json
import threading
import time

import numpy as np
import pytest

from assemblyloop import export
from assemblyloop.export import convert, export_lerobot, export_npz, keep_mask
from assemblyloop.feeds import Feeds
from assemblyloop.record import Recorder, episodes, load, next_slot


def until(cond, secs):
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        if cond():
            return True
        time.sleep(0.01)
    return False


class Rig:
    """A made-up robot and its cameras, on a clock that moves only when told to."""

    def __init__(self, sides=("right",), cams=("scene", "wrist")):
        self.now, self.sides, self.cams, self.seq, self.said = 1000.0, sides, cams, {c: 1 for c in cams}, []
        self.joints, self.grip, self.goal, self.moving, self.held, self.rpy = np.zeros(7), 850, None, False, True, [180.0, 0.0, 0.0]
        self.fail_sample = self.fail_frames = False

    def clock(self):
        return 1.7e9 + self.now

    def mono(self):
        return self.now

    def sample(self):
        if self.fail_sample:
            raise OSError("the controller did not answer")
        arm = lambda k: dict(joints=(self.joints + 100.0 * k).tolist(), xyz=[450.0 + k, 0.0, 300.0], rpy=list(self.rpy), grip=self.grip, moving=self.moving, goal_xyz=None,  # noqa: E731
                             goal_grip=self.goal, what="lower")
        return dict(arms={s: arm(k) for k, s in enumerate(self.sides)}, held=self.held, phase="RUNNING", step="lower", leg="A to B")

    def frames(self):
        if self.fail_frames:
            raise RuntimeError("no cameras")
        return [dict(id=c, jpg=b"JPG" + c.encode() + bytes([self.seq[c] % 250]), seq=self.seq[c], age_ms=20) for c in self.cams]

    def recorder(self, root, **kw):
        return Recorder(root, self.sample, self.frames, hz=10.0, clock=self.clock, say=self.said.append, mono=self.mono, thread=False, **kw)

    def run(self, rec, script, dt=0.1):
        """One row per entry of `script`: {"joints": deg added to every joint, "grip", "goal", "moving", "held", "new":
        cameras with a new picture (default all)}."""
        for s in script:
            self.now += dt
            self.joints = self.joints + s.get("joints", 0.0)
            self.grip, self.goal, self.moving, self.held = s.get("grip", self.grip), s.get("goal", self.goal), s.get("moving", bool(s.get("joints"))), s.get("held", self.held)
            for c in s.get("new", self.cams):
                self.seq[c] += 1
            rec.tick()


def test_an_episode_is_a_folder_of_rows_events_and_pictures(tmp_path):
    rig = Rig()
    rec = rig.recorder(tmp_path)
    assert not rec.active and rec.stats()["active"] is False and rec.end("nothing") is None
    rec.mark("ignored")  # outside an episode: nothing happens, nothing is written
    assert not (tmp_path / "episodes").exists()
    eid = rec.begin("stack the blocks into a tower", arm="right", blocks=3)
    assert rec.active and eid.endswith("_001") and json.loads((tmp_path / "episodes" / eid / "episode.json").read_text())["finished"] is False
    rig.run(rec, [dict(joints=1.0), dict(joints=1.0, new=("scene",))])
    rec.mark("step", name="lower onto B", n=np.int64(2), at=np.array([1.0, 2.0]))
    rig.run(rec, [dict(joints=1.0, new=())])
    st = rec.stats()
    assert st["episode"] == eid and st["rows"] == 4 and st["pictures"] == 5 and st["dropped"] == 0 and st["bytes"] > 0 and st["cams"] == {"scene": 20, "wrist": 20}
    path = rec.end("done", success=True, blocks_placed=3)
    assert path == tmp_path / "episodes" / eid and not rec.active and rec.stats()["last"]["rows"] == 4

    ep = load(path)
    assert ep.finished and ep.success is True and ep.task == "stack the blocks into a tower" and ep.meta["meta"] == dict(arm="right", blocks=3)
    assert ep.meta["schema"] == 1 and ep.meta["hz"] == 10.0 and ep.meta["cams"] == ["scene", "wrist"] and ep.meta["outcome"] == "done" and ep.meta["end"] == dict(blocks_placed=3)
    assert ep.meta["rows"] == 4 and ep.meta["pictures"] == dict(scene=3, wrist=2) and ep.meta["dropped"] == 0 and abs(ep.meta["duration_s"] - 0.3) < 1e-6
    assert [r["i"] for r in ep.rows] == [0, 1, 2, 3] == [r["tick"] for r in ep.rows] and np.allclose(np.diff([r["mono"] for r in ep.rows]), 0.1)
    assert ep.rows[1]["arms"]["right"]["joints"] == [1.0] * 7 and ep.rows[1]["held"] is True and ep.rows[1]["phase"] == "RUNNING" and "err" not in ep.rows[1]
    # a camera whose picture is not new is not written again: the row points at the picture it had
    assert [r["cams"]["scene"]["file"] for r in ep.rows] == ["cam/scene/000000.jpg", "cam/scene/000001.jpg", "cam/scene/000002.jpg", "cam/scene/000002.jpg"]
    assert [r["cams"]["wrist"]["file"] for r in ep.rows] == ["cam/wrist/000000.jpg", "cam/wrist/000001.jpg", "cam/wrist/000001.jpg", "cam/wrist/000001.jpg"]
    assert sorted(p.name for p in (path / "cam" / "wrist").iterdir()) == ["000000.jpg", "000001.jpg"]
    assert ep.picture(2, "scene") == b"JPGscene" + bytes([3]) and ep.picture(ep.rows[3], "wrist") == b"JPGwrist" + bytes([2]) and ep.picture(0, "nobody's") is None
    assert [(e["kind"], e["i"]) for e in ep.events] == [("begin", -1), ("step", 2), ("end", 3)]
    assert ep.events[1]["name"] == "lower onto B" and ep.events[1]["n"] == 2 and ep.events[1]["at"] == [1.0, 2.0] and ep.events[2]["success"] is True
    assert episodes(tmp_path) == [path] == episodes(tmp_path / "episodes") == episodes(path)
    assert rec.begin("the next one").endswith("_002")
    rec.close()
    assert load(episodes(tmp_path)[1]).meta["outcome"] == "closed"


def test_a_late_tick_is_counted_as_dropped_and_not_made_up_for(tmp_path):
    assert [next_slot(4, x) for x in (4.2, 4.99, 5.0, 5.4, 5.6, 6.2, 9.51)] == [5, 5, 5, 5, 6, 6, 10]
    rig = Rig()
    rec = rig.recorder(tmp_path)
    rec.begin("wait")
    rig.run(rec, [{}, {}])
    rig.run(rec, [{}], dt=0.37)  # slots 3 and 4 went by while something was slow
    rig.run(rec, [{}])
    ep = load(rec.end("done"))
    assert [r["tick"] for r in ep.rows] == [0, 1, 2, 6, 7] and [r["i"] for r in ep.rows] == [0, 1, 2, 3, 4] and ep.meta["dropped"] == 3


def test_the_thread_keeps_its_schedule_and_a_mark_never_waits(tmp_path):
    slow, n = threading.Event(), [0]

    def sample():
        n[0] += 1
        if n[0] == 5:
            time.sleep(0.11)  # more than two periods
        if slow.is_set():
            raise ValueError("no")
        return dict(arms={}, held=True)

    rec = Recorder(tmp_path, sample, None, hz=25.0, say=lambda _m: None)
    rec.begin("keep time")
    assert until(lambda: rec.stats()["rows"] >= 12, 3)
    t0 = time.monotonic()
    for k in range(200):
        rec.mark("note", k=k)
    assert time.monotonic() - t0 < 0.2
    slow.set()
    assert until(lambda: rec.stats()["errors"] >= 3, 3) and rec.active  # a sample that fails does not end the thread
    ep = load(rec.end("done"))
    late = np.array([r["mono"] - ep.meta["mono0"] - r["tick"] / 25.0 for r in ep.rows])
    assert ep.meta["dropped"] >= 2 and ep.rows[-1]["tick"] == len(ep.rows) - 1 + ep.meta["dropped"]
    assert late.min() > -0.005 and np.median(late) < 0.01  # every row at its own slot: nothing drifts, nothing bunches
    assert np.diff([r["mono"] for r in ep.rows]).min() > 0.5 / 25.0 - 0.005
    assert len([e for e in ep.events if e["kind"] == "note"]) == 200 and ep.meta["events"] == 202 and "sample: ValueError: no" in ep.rows[-1]["err"]


def test_a_sample_or_a_camera_that_fails_is_written_down_and_recording_goes_on(tmp_path):
    rig = Rig()
    rec = rig.recorder(tmp_path)
    rec.begin("go on")
    rig.fail_sample = True
    rig.run(rec, [{}])
    rig.fail_sample, rig.fail_frames = False, True
    rig.run(rec, [{}])
    rig.fail_frames, rec.sample = False, lambda: dict(arms={}, odd={1, 2}, i="mine", t="mine")
    rig.run(rec, [{}])
    assert rec.stats()["errors"] == 2 and rec.stats()["error"].startswith("frames: RuntimeError")
    ep = load(rec.end("done"))
    assert "sample: OSError: the controller did not answer" in ep.rows[1]["err"] and "arms" not in ep.rows[1] and set(ep.rows[1]["cams"]) == {"scene", "wrist"}
    assert ep.rows[2]["err"] == "frames: RuntimeError: no cameras" and ep.rows[2]["cams"] == {} and "arms" in ep.rows[2]
    assert "err" not in ep.rows[3] and ep.rows[3]["i"] == 3 and sorted(ep.rows[3]["odd"]) == [1, 2] and ep.rows[3]["t"] > 1e9  # the recorder's own keys stay its own
    assert sum("OSError" in s for s in rig.said) == 1


def test_an_old_picture_is_not_passed_off_as_new(tmp_path):
    rig = Rig(cams=("scene",))
    old = [dict(id="scene", jpg=b"OLD", seq=7, age_ms=60000)]
    rec = Recorder(tmp_path, rig.sample, lambda: old, clock=rig.clock, mono=rig.mono, thread=False, say=None)
    rec.begin("a camera that was closed a minute ago")
    assert rec.tick()["cams"] == {}
    old[0] = dict(id="scene", jpg=b"NEW", seq=8, age_ms=30)
    assert rec.tick()["cams"]["scene"]["file"] == "cam/scene/000002.jpg"
    old[0] = dict(id="scene", jpg=b"NEW", seq=8, age_ms=5000)  # it stopped: the row still points at its last picture and says how old that is
    assert rec.tick()["cams"]["scene"] == dict(file="cam/scene/000002.jpg", seq=8, age_ms=5000)
    rec.end("done")


def test_an_episode_cut_short_by_a_crash_can_still_be_read(tmp_path):
    rig = Rig()
    rec = rig.recorder(tmp_path)
    rec.begin("crash")
    rig.run(rec, [dict(joints=1.0)] * 5)
    rec.mark("step", name="lift")
    path = episodes(tmp_path)[0]  # no end(): the program died here, in the middle of a line
    with open(path / "rows.jsonl", "a") as fh:
        fh.write('{"i": 6, "tick": 6, "t": 17000')
    ep = load(path)
    assert not ep.finished and ep.success is None and len(ep.rows) == 6 and ep.torn == 1 and ep.task == "crash" and [e["kind"] for e in ep.events] == ["begin", "step"]
    assert ep.picture(5, "scene") is not None
    (path / "episode.json").unlink()
    ep = load(path)
    assert not ep.finished and len(ep.rows) == 6 and ep.id == path.name and abs(ep.hz - 10.0) < 1e-6 and ep.sides() == ["right"] and ep.cams == ["scene", "wrist"]
    clip = convert(ep, trim=False)
    assert len(clip) == 6 and "cut short" in clip.notes[0]
    again = rig.recorder(tmp_path)  # the program is started again: the next episode gets the next number
    assert again.begin("after the crash").endswith("_002")
    again.close()


def test_a_new_episode_ends_the_one_that_is_open(tmp_path):
    rig = Rig()
    rec = rig.recorder(tmp_path)
    rec.begin("one")
    rec.begin("two")
    rec.close()
    with pytest.raises(RuntimeError):
        rec.begin("three")
    one, two = (load(p) for p in episodes(tmp_path))
    assert one.meta["outcome"].startswith("interrupted") and one.success is None and two.task == "two" and two.meta["outcome"] == "closed"


# ---------------------------------------------------------------------- pictures without a page
class Cam:
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


def open_cam(device, size):
    if "broken" in str(device):
        raise OSError("does not open: another program is using it")
    return Cam(str(device).rsplit("/", 1)[-1])


@pytest.fixture
def cams(tmp_path):
    Cam.opened, Cam.closed = [], []
    specs = [dict(id="scene", label="Scene", device=str(tmp_path / "scene"), arms=("right", "left", "stand-in")), dict(id="grip_right", label="Right", device=str(tmp_path / "grip_right"), arms=("right",)),
             dict(id="grip_left", label="Left", device=str(tmp_path / "grip_left"), arms=("left",)), dict(id="broken", label="Broken", device=str(tmp_path / "broken"), arms=("left",)),
             dict(id="wrist", label="Stand-in's", device=str(tmp_path / "wrist"), arms=("stand-in",))]
    f = Feeds(specs, fps=100.0, idle_s=0.4, open=open_cam, encode=lambda im: (b"JPG" + bytes([int(im[0, 0, 0])]), (int(im.shape[1]), int(im.shape[0]))))
    yield f
    f.close()


def test_a_recorder_gets_pictures_of_every_arm_without_a_page(cams):
    t0 = time.monotonic()
    got = cams.latest(["right", "left"])
    assert time.monotonic() - t0 < 0.2  # it never waits for a picture
    assert [c["id"] for c in got] == ["scene", "grip_right", "grip_left", "broken"] and all(c["jpg"] == b"" and c["seq"] == 0 and c["age_ms"] is None for c in got)
    assert until(lambda: all(c["jpg"] for c in cams.latest(("right", "left"))[:3]), 3)
    got = {c["id"]: c for c in cams.latest({"right", "left"})}
    assert got["scene"]["jpg"].startswith(b"JPG") and got["scene"]["age_ms"] < 500 and got["scene"]["size"] == (6, 4) and got["grip_left"]["seq"] > 0
    assert got["broken"]["jpg"] == b"" and "another program" in got["broken"]["note"]
    assert sorted(Cam.opened) == ["grip_left", "grip_right", "scene"]  # the stand-in's camera was not asked for
    assert [c["id"] for c in cams.latest("right", keep_awake=False)] == ["scene", "grip_right"]
    assert until(lambda: sorted(Cam.closed) == ["grip_left", "grip_right", "scene"], 3)  # let go a moment after the last call
    stale = cams.latest("right", keep_awake=False)
    assert len(Cam.opened) == 3 and stale[0]["jpg"] and stale[0]["age_ms"] >= 0  # looking without `keep_awake` opens nothing


def test_the_page_and_the_recorder_keep_cameras_open_for_their_own_reasons(cams):
    cams.frames("stand-in", 0, 0.5)
    assert until(lambda: sorted(Cam.opened) == ["scene", "wrist"], 3)
    end = time.monotonic() + 0.9  # the recorder keeps asking for the left arm's while the page has gone
    while time.monotonic() < end:
        cams.latest(["left"])
        time.sleep(0.02)
    assert "wrist" in Cam.closed and "grip_left" in Cam.opened and "grip_left" not in Cam.closed and "scene" not in Cam.closed and "grip_right" not in Cam.opened
    seq = max(c["seq"] for c in cams.latest(["left"]))
    assert cams.frames("stand-in", seq, 2.0)[0] > seq  # and a page that comes back gets its pictures as before
    assert until(lambda: Cam.opened.count("wrist") == 2, 3)


def test_pictures_from_the_cameras_end_up_in_the_episode(cams, tmp_path):
    rig = Rig()
    rec = Recorder(tmp_path, rig.sample, lambda: cams.latest(["right"]), hz=50.0, say=None)
    rec.begin("look")
    assert until(lambda: rec.stats()["pictures"] >= 6 and rec.stats()["rows"] >= 10, 4)
    ep = load(rec.end("done"))
    assert ep.cams == ["grip_right", "scene"] and ep.meta["pictures"]["scene"] >= 3
    last = ep.rows[-1]["cams"]["scene"]
    assert ep.picture(ep.rows[-1], "scene").startswith(b"JPG") and last["seq"] > 0 and last["age_ms"] < 500
    files = {r["cams"]["scene"]["file"] for r in ep.rows if "scene" in r["cams"]}
    assert len(files) == ep.meta["pictures"]["scene"] == len(list((ep.dir / "cam" / "scene").iterdir()))


# ---------------------------------------------------------------------- what a policy is trained on
def recorded(tmp_path, script, sides=("right",), success=True, end=True, task="put the block on B", rig=None):
    rig = rig or Rig(sides)
    rec = rig.recorder(tmp_path)
    rec.begin(task)
    rig.run(rec, script)
    if end:
        return rec.end("done" if success else "blocked", success=success)
    return episodes(tmp_path)[-1]


def test_state_and_action_are_joints_in_radians_and_the_gripper_from_shut_to_open(tmp_path):
    rig = Rig(("left", "right"))  # whatever order the session names them in: right comes first
    rig.rpy = [179.0, 0.0, 0.0]
    path = recorded(tmp_path, [dict(joints=2.0), dict(joints=2.0, goal=0), dict(joints=0.0, grip=425, goal=0, moving=True), dict(joints=1.0, grip=170)], rig=rig)
    rows = load(path).rows
    rows[3]["arms"]["right"]["rpy"] = [-179.0, 0.0, 0.0]  # the controller's roll goes round at 180 degrees
    (path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    clip = convert(path, trim=False)
    assert clip.sides == ["right", "left"] and clip.fps == 10.0 and clip.task == "put the block on B" and len(clip) == 5
    assert clip.state.shape == (5, 16) == clip.action.shape and clip.ee.shape == (5, 12) and clip.state.dtype == np.float32
    assert clip.names[:8] == [f"right_joint_{k}" for k in range(1, 8)] + ["right_gripper"] and clip.names[8] == "left_joint_1" and clip.ee_names[3:6] == ["right_roll", "right_pitch", "right_yaw"]
    # the made-up right arm is 100 degrees ahead of the left one in every joint
    assert np.allclose(clip.state[:, 0], np.radians([100, 102, 104, 104, 105])) and np.allclose(clip.state[:, 8], np.radians([0, 2, 4, 4, 5]), atol=1e-6)
    assert np.allclose(clip.state[:, 7], [1, 1, 1, 0.5, 0.2]) and np.allclose(clip.state[:, 15], clip.state[:, 7])
    # the action is where the joints are a tick later; the gripper's is what it was asked for, once it was asked
    assert np.allclose(clip.action[:-1, :7], clip.state[1:, :7]) and np.allclose(clip.action[-1, :7], clip.state[-1, :7])
    assert np.allclose(clip.action[:, 7], [1, 1, 0, 0, 0]) and np.allclose(clip.action[:, 15], [1, 1, 0, 0, 0])
    assert np.allclose(clip.ee[0, :3], [0.451, 0.0, 0.3]) and np.allclose(clip.ee[0, 6:9], [0.45, 0.0, 0.3])
    assert np.allclose(clip.ee[:, 3], np.radians([179, 179, 179, 181, 179])) and np.allclose(clip.ee[:, 9], np.radians(179))  # no jump to -179
    assert clip.row.tolist() == [0, 1, 2, 3, 4] and np.allclose(np.diff(clip.t), 0.1) and clip.pictures["scene"][4] == path / "cam" / "scene" / "000004.jpg"
    assert clip.dropped == dict(broken=0, repeated=0, start=0, end=0, pauses=0)


def test_rows_in_which_nothing_happens_are_cut(tmp_path):
    keep, cut = keep_mask([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], fps=2.0, still_s=1.0, pause_s=1.0)
    assert keep.astype(int).tolist() == [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0] and cut == dict(start=1, end=1, pauses=4)
    assert keep_mask([0, 0, 0], [1, 1, 1], 10.0)[0].sum() == 0 and keep_mask([0, 1, 0, 0, 0, 0, 0, 1], [1] * 8, 2.0)[0].all()  # waiting with the switch held is part of the task

    still, go = [dict()] * 30, [dict(joints=1.0)] * 10
    script = still + go + [dict(held=False)] * 40 + [dict(held=True)] + go + [dict(grip=400, goal=0)] + still
    path = recorded(tmp_path, script)
    clip = convert(path)  # 1 s of standing still is kept at either end, and 1 s of the pause
    assert clip.dropped == dict(broken=0, repeated=0, start=20, end=20, pauses=30) and len(clip) == len(script) + 1 - 70
    assert clip.row[0] == 20 and clip.row[-1] == len(script) - 20  # the last thing that happened is the gripper closing, a second before
    whole = convert(path, trim=False)
    assert len(whole) == len(script) + 1 and sum(whole.dropped.values()) == 0
    at = int(np.flatnonzero(clip.row == 40)[0])  # the last row of the first motion: its action is what truly came next
    assert np.allclose(clip.action[at], whole.action[40]) and np.allclose(clip.state[at], whole.state[40])
    assert len(convert(path, still_s=0.0, pause_s=0.0)) < len(clip) and len(clip.pictures["wrist"]) == len(clip)


def test_dropped_ticks_and_other_rates_are_put_on_an_even_grid(tmp_path):
    rig = Rig()
    rec = rig.recorder(tmp_path)
    rec.begin("uneven")
    rig.run(rec, [dict(joints=1.0)] * 3)
    rig.run(rec, [dict(joints=3.0)], dt=0.3)  # two ticks were dropped here
    rig.run(rec, [dict(joints=1.0)] * 4)
    path = rec.end("done", success=True)
    clip = convert(path, trim=False)
    assert len(clip) == 11 and clip.row.tolist() == [0, 1, 2, 3, 3, 4, 4, 5, 6, 7, 8] and clip.dropped["repeated"] == 2
    half = convert(path, fps=5.0, trim=False)
    assert half.fps == 5.0 and half.row.tolist() == [0, 2, 3, 4, 6, 8] and np.allclose(half.action[0, :7], half.state[1, :7])
    with pytest.raises(ValueError, match="recorded at 10"):
        convert(path, fps=30.0)
    rows = load(path).rows
    del rows[2]["arms"]["right"]["joints"]  # a row the sample failed in
    rows[5]["arms"]["right"]["grip"] = None
    (path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    clip = convert(path, trim=False)
    assert clip.dropped["broken"] == 1 and 2 not in clip.row and np.isfinite(clip.state).all() and np.allclose(clip.state[:, 7], 1.0)


def test_an_episode_becomes_arrays_with_numpy_alone(tmp_path):
    path = recorded(tmp_path, [dict()] * 15 + [dict(joints=1.0)] * 5 + [dict()] * 15)
    got = export_npz(path, tmp_path / "out" / "one.npz")
    assert got["rows"] == 26 and got["dropped"] == 10 and got["dropped_by"]["start"] == 5 and got["dropped_by"]["end"] == 5 and got["task"] == "put the block on B"
    z = np.load(tmp_path / "out" / "one.npz")  # no pickles: plain arrays and strings
    clip = convert(path)
    assert np.array_equal(z["state"], clip.state) and np.array_equal(z["action"], clip.action) and np.array_equal(z["ee"], clip.ee) and np.array_equal(z["t"], clip.t)
    assert str(z["task"]) == "put the block on B" and float(z["fps"]) == 10.0 and z["cams"].tolist() == ["scene", "wrist"] and z["sides"].tolist() == ["right"] and z["names"][7] == "right_gripper"
    assert z["pictures.scene"].shape == (26,) and (path / str(z["pictures.scene"][0])).read_bytes().startswith(b"JPGscene") and z["row"][0] == 5


class FakeDataset:
    """Stands in for lerobot's LeRobotDataset: it takes the same calls and checks what the real one checks."""

    made = None

    def __init__(self, **kw):
        self.kw, self.features, self.buffer, self.saved, self.finalized = kw, kw["features"], [], [], False

    @classmethod
    def create(cls, repo_id, fps, features, root=None, robot_type=None, use_videos=True, image_writer_threads=0):
        assert not root.exists()
        cls.made = cls(repo_id=repo_id, fps=fps, features=features, root=root, robot_type=robot_type, use_videos=use_videos)
        return cls.made

    def add_frame(self, frame):
        assert set(frame) == set(self.features) | {"task"} and isinstance(frame["task"], str)
        for key, ft in self.features.items():
            assert frame[key].shape == tuple(ft["shape"]) and frame[key].dtype == (np.uint8 if ft["dtype"] in ("image", "video") else np.dtype(ft["dtype"])), key
        self.buffer.append(frame)

    def save_episode(self):
        self.saved.append(self.buffer)
        self.buffer = []

    def finalize(self):
        self.finalized = True


def test_episodes_become_one_dataset_and_failed_ones_are_left_out(tmp_path):
    move = [dict(joints=1.0)] * 6
    good = recorded(tmp_path, move, task="stack the blocks")
    failed = recorded(tmp_path, move, success=False)
    unjudged = recorded(tmp_path, move + [dict()] * 30, success=None)
    crashed = recorded(tmp_path, move, end=False)
    slower = Rig()
    rec = Recorder(tmp_path, slower.sample, slower.frames, hz=5.0, clock=slower.clock, mono=slower.mono, thread=False, say=None)
    rec.begin("too slow")
    slower.run(rec, move, dt=0.2)
    rec.end("done", success=True)
    decode = lambda jpg: np.full((4, 6, 3), jpg[-1], np.uint8)  # noqa: E731
    said = []
    out = export_lerobot(tmp_path, tmp_path / "ds", "lab/blocks", decode=decode, dataset_cls=FakeDataset, say=said.append)
    ds = FakeDataset.made
    assert ds.finalized and ds.kw["repo_id"] == "lab/blocks" and ds.kw["fps"] == 10 and isinstance(ds.kw["fps"], int) and ds.kw["root"] == tmp_path / "ds" and ds.kw["robot_type"] == "xarm7"
    assert ds.features["observation.state"] == dict(dtype="float32", shape=(8,), names=[f"right_joint_{k}" for k in range(1, 8)] + ["right_gripper"]) and ds.features["action"]["shape"] == (8,)
    assert ds.features["observation.ee_pose"]["shape"] == (6,) and ds.features["observation.images.wrist"] == dict(dtype="image", shape=(4, 6, 3), names=["height", "width", "channels"])
    assert out["episodes"] == 2 and [len(e) for e in ds.saved] == [7, 17] == [c["rows"] for c in out["clips"]] and out["frames"] == 24 and out["dropped"] == 20
    assert {f["task"] for f in ds.saved[0]} == {"stack the blocks"} and np.allclose(ds.saved[0][2]["observation.state"][:7], np.radians(2.0)) and ds.saved[0][3]["observation.images.scene"][0, 0, 0] == 4
    assert [name for name, _ in out["skipped"]] == [failed.name, crashed.name, episodes(tmp_path)[-1].name] and "failed" in out["skipped"][0][1] and "cut short" in out["skipped"][1][1]
    assert "recorded at 5" in out["skipped"][2][1] and any(unjudged.name in s and "20 dropped" in s for s in said)
    with pytest.raises(FileExistsError):
        (tmp_path / "there").mkdir()
        (tmp_path / "there" / "something").write_text("")
        export_lerobot(tmp_path, tmp_path / "there", "lab/blocks", decode=decode, dataset_cls=FakeDataset, say=None)
    everything = export_lerobot([good, failed, crashed], tmp_path / "all", "lab/blocks", only_success=False, unfinished=True, use_videos=True, decode=decode, dataset_cls=FakeDataset, say=None)
    assert everything["episodes"] == 3 and FakeDataset.made.features["observation.images.scene"]["dtype"] == "video" and FakeDataset.made.kw["use_videos"] is True
    with pytest.raises(ValueError, match="no episode to export"):
        export_lerobot([failed], tmp_path / "none", "lab/blocks", decode=decode, dataset_cls=FakeDataset, say=None)


def test_without_the_library_the_exporter_says_what_is_missing(tmp_path, monkeypatch):
    import builtins

    real = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", lambda name, *a, **kw: (_ for _ in ()).throw(ImportError(name)) if name.startswith("lerobot") else real(name, *a, **kw))
    path = recorded(tmp_path, [dict(joints=1.0)] * 3)
    with pytest.raises(RuntimeError, match="needs the `lerobot` library"):
        export_lerobot(path, tmp_path / "ds", "lab/blocks", say=None)
    assert not (tmp_path / "ds").exists()


def test_recorded_jpegs_are_decoded_red_first(tmp_path):
    cv2 = pytest.importorskip("cv2")
    im = np.zeros((48, 64, 3), np.uint8)
    im[:, :, 2] = 250  # red, in OpenCV's order
    jpg = cv2.imencode(".jpg", im)[1].tobytes()
    rig = Rig(cams=("scene",))
    rig.frames = lambda: [dict(id="scene", jpg=jpg, seq=rig.seq["scene"], age_ms=5)]
    path = recorded(tmp_path, [dict(joints=1.0)] * 3, rig=rig)
    assert load(path).picture(0, "scene") == jpg  # the bytes exactly as they came
    got = export.decode_jpeg(jpg)
    assert got.shape == (48, 64, 3) and got[10, 10, 0] > 240 and got[10, 10, 2] < 15
    export_lerobot(path, tmp_path / "ds", "lab/blocks", image_size=(32, 24), dataset_cls=FakeDataset, say=None)
    frame = FakeDataset.made.saved[0][0]["observation.images.scene"]
    assert frame.shape == (24, 32, 3) and frame[5, 5, 0] > 240
