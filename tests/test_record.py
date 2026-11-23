"""Recording episodes, checking them, exporting them to LeRobot v2.1 and uploading them: on the stand-in arm with
made-up cameras. Needs the `data` extra (imageio + its ffmpeg, pyarrow) and OpenCV."""
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("imageio_ffmpeg")
pq = pytest.importorskip("pyarrow.parquet")

from assemblyloop.arms import FakeArm  # noqa: E402
from assemblyloop.feeds import Feeds  # noqa: E402
from assemblyloop.record.export import FEATURES, export_campaign  # noqa: E402
from assemblyloop.record.store import Campaign, read_json  # noqa: E402
from assemblyloop.record.upload import push, round_once  # noqa: E402
from assemblyloop.record.validate import count_frames, read_jsonl, validate_campaign  # noqa: E402
from assemblyloop.session import Session  # noqa: E402


def until(cond, secs):
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        if cond():
            return True
        time.sleep(0.02)
    return False


class Cam:
    """A made-up camera: 30 pictures a second, each a different shade, so that frames can be told apart."""

    def __init__(self, size=(160, 120)):
        self.size, self.n = size, 0

    def read(self):
        time.sleep(1 / 30)
        self.n += 1
        im = np.zeros((self.size[1], self.size[0], 3), np.uint8)
        im[:, :, 1] = (self.n * 7) % 256
        return True, im

    def release(self):
        pass


@pytest.fixture
def ses(tmp_path):
    feeds = Feeds([dict(id="cam_a", key="head_left", label="A", arms=("stand-in",), make=Cam), dict(id="cam_b", key="gripper_right_top", label="B", arms=("stand-in",), make=Cam)], fps=5.0)
    s = Session({"stand-in": dict(label="Stand-in arm (no robot)", ip=None, make=FakeArm)}, spots_path=tmp_path / "spots.json", log_dir=tmp_path / "logs", probe=lambda ip: False,
                fake_speed=8.0, guard_kw=dict(poll_s=0.002), limit_kw=dict(settle_s=0.0), blocks_path=tmp_path / "blocks.json", feeds=feeds, campaigns_root=tmp_path / "campaigns")
    s.feeds.owners = [o for o in s.feeds.owners if o.id != "made_up"]  # the two made-up cameras above, not the drawn one
    s.alive, s.pressed = True, True

    def grip():
        while s.alive:
            if s.dm is not None:
                s.dm.feed(s.pressed)
            time.sleep(0.02)

    threading.Thread(target=grip, daemon=True).start()
    s.start("stand-in")
    s.pressed = False
    assert until(lambda: all(c["ok"] for c in s.state()["checks"]), 6)
    s.arm_on()
    s.use_demo()
    s.pressed = True
    yield s
    s.alive = False
    s.close()


def record_tower(ses, name="Blocks v1"):
    ses.record_new_campaign(name)
    ses.record_on(name)
    ses.tower("T", 30.0, "build", n_now=0)
    assert until(lambda: ses.worker is None, 120) and ses.job["outcome"] == "done", ses.job
    ses.record_off()
    return Campaign.open(ses.campaigns_root, name)


def test_every_block_taken_and_put_down_is_one_episode_in_the_labs_layout(ses):
    c = record_tower(ses)
    runs = c.runs()
    assert len(runs) == 3 and all(r.name.startswith("recording_") for r in runs)
    for k, r in enumerate(runs):
        m = r.meta
        assert m["status"] == "ready" and m["validation"]["ok"], (r.name, [x for x in m["validation"]["checks"] if x["level"] != "ok"])
        assert m["outcome"]["success"] and m["outcome"]["kind"] == "placed" and not m["outcome"]["verified_by_camera"]  # taught spots: no camera checked it
        t = m["task"]
        assert t["leg"] == k and t["dst"] == "T" and t["action_source"] == "script" and t["arm"] == "stand-in"
        assert t["instruction"] == ("put the block down where the tower stands" if k == 0 else "put the block on top of the tower")
        assert m["cameras"] == ["head_left", "gripper_right_top"] and m["provenance"]["program"]["sha"] and m["provenance"]["arm"]["R_world_from_base"]
        arm = read_jsonl(r.path / "raw" / "arm_state.jsonl")
        cmd = read_jsonl(r.path / "raw" / "commanded.jsonl")
        ev = read_jsonl(r.path / "raw" / "events.jsonl")
        assert len(arm) > 10 and all(len(x["joints_measured_deg"]) == 7 and len(x["pose_base"]) == 6 for x in arm)
        assert {"move", "grip"} <= {x["kind"] for x in cmd} and all(0 <= x["t"] <= m["duration_s"] + 1 for x in cmd)
        assert ev[0]["kind"] == "episode_start" and ev[-1]["kind"] == "motion_end" and any(x["kind"] == "step" and x["name"] == "let go" for x in ev)
        for key in m["cameras"]:
            ft = read_json(r.path / "video" / f"{key}_frame_times.json")
            assert ft["frames"] == len(ft["t"]) == count_frames(r.path / "video" / f"{key}.mp4") > 5 and ft["label"] == key and not ft["stopped_early"]
            assert ft["t"] == sorted(ft["t"]) and ft["t"][0] < 0.5


def test_episodes_become_a_lerobot_dataset_added_to_as_more_are_recorded(ses):
    c = record_tower(ses)
    added, refused = export_campaign(c, say=lambda *_: None)
    assert len(added) == 3 and not refused
    out = c.path / "lerobot"
    info = read_json(out / "meta" / "info.json")
    eps = read_jsonl(out / "meta" / "episodes.jsonl")
    assert info["codebase_version"] == "v2.1" and info["total_episodes"] == 3 and info["fps"] == 30.0 and info["total_frames"] == sum(e["length"] for e in eps)
    assert set(FEATURES) | {"action.is_valid", "observation.images.head_left", "observation.images.gripper_right_top"} <= set(info["features"])
    assert info["features"]["observation.images.head_left"]["shape"] == [120, 160, 3]
    assert [e["run"] for e in eps] == [r.name for r in c.runs()] and all(e["success"] and e["action_source"] == "script" for e in eps)
    assert {t["task"] for t in read_jsonl(out / "meta" / "tasks.jsonl")} == {"put the block down where the tower stands", "put the block on top of the tower"}
    for e in eps:
        t = pq.read_table(out / f"data/chunk-000/episode_{e['episode_index']:06d}.parquet")
        assert t.num_rows == e["length"] and len(t.column("observation.state")[0]) == 17 and len(t.column("action")[0]) == 5
        assert max(t.column("sync_error_s").to_pylist())[0] < 0.1  # nearest in time, and near
        for key in ("head_left", "gripper_right_top"):
            assert count_frames(out / f"videos/chunk-000/observation.images.{key}/episode_{e['episode_index']:06d}.mp4") == e["length"]  # frame i is row i
        assert (out / "raw" / e["run"] / "raw" / "arm_state.jsonl").exists()
    first = (out / "data/chunk-000/episode_000000.parquet").stat().st_mtime_ns
    ses.record_on("Blocks v1")
    ses.tower("T", 30.0, "down", n_now=3)
    assert until(lambda: ses.worker is None, 120) and ses.job["outcome"] == "done"
    ses.record_off()
    added, _ = export_campaign(c, say=lambda *_: None)
    assert len(added) == 3 and read_json(out / "meta" / "info.json")["total_episodes"] == 6
    assert (out / "data/chunk-000/episode_000000.parquet").stat().st_mtime_ns == first  # what was exported stays as it was


def test_a_run_that_fails_its_checks_is_not_exported_and_rehearsals_are_not_recorded(ses):
    c = record_tower(ses)
    bad = c.runs()[1]
    ft = bad.path / "video" / "head_left_frame_times.json"
    d = json.loads(ft.read_text())
    d["t"] = d["t"][:-3]  # three frames written but not timed: every later time would be off
    ft.write_text(json.dumps(d))
    v = validate_campaign(c, again=True)
    assert not v[bad.name]["ok"] and sum(x["ok"] for x in v.values()) == 2
    added, refused = export_campaign(c, say=lambda *_: None)
    assert len(added) == 2 and refused and refused[0][0] == bad.name and "head_left" in refused[0][1]
    ses.record_on("Blocks v1")
    ses.tower("T", 30.0, "down", n_now=3, rehearse=True)
    assert until(lambda: ses.worker is None, 120)
    ses.record_off()
    assert len(c.runs()) == 3


class FakeHub:
    def __init__(self):
        self.calls = []

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create_repo", repo_id, kw))

    def upload_folder(self, **kw):
        self.calls.append(("upload_folder", kw["repo_id"], kw))

    def delete_tag(self, repo_id, **kw):
        self.calls.append(("delete_tag", repo_id, kw))

    def create_tag(self, repo_id, **kw):
        self.calls.append(("create_tag", repo_id, kw))


def test_the_upload_sends_only_what_is_new_one_commit_at_a_time_and_private(ses):
    c = record_tower(ses)
    hub = FakeHub()
    r = round_once(c, "lab/blocks", api=hub, say=lambda *_: None)
    assert r["added"] == 3 and r["sent"] > 0
    up = [x for x in hub.calls if x[0] == "upload_folder"]
    assert len(up) == 1 and up[0][2]["repo_type"] == "dataset" and up[0][2]["folder_path"].endswith("lerobot")
    sent = set(up[0][2]["allow_patterns"])
    assert "meta/info.json" in sent and "README.md" in sent and sum(p.endswith(".parquet") for p in sent) == 3
    assert ("create_repo", "lab/blocks", {"repo_type": "dataset", "private": True, "exist_ok": True}) in hub.calls and hub.calls[-1][0] == "create_tag" and hub.calls[-1][2]["tag"] == "v2.1"
    assert push(c, "lab/blocks", api=hub, say=lambda *_: None) == 0  # nothing new: nothing sent
    ses.record_on("Blocks v1")
    ses.tower("T", 30.0, "down", n_now=3)
    assert until(lambda: ses.worker is None, 120)
    ses.record_off()
    round_once(c, "lab/blocks", api=hub, say=lambda *_: None)
    second = set([x for x in hub.calls if x[0] == "upload_folder"][-1][2]["allow_patterns"])
    assert sum(p.endswith(".parquet") for p in second) == 3 and not (second & {p for p in sent if p.endswith(".parquet")})  # only the new episodes, and the meta files
    assert "meta/episodes.jsonl" in second and read_json(c.path / "upload.json")["repo"] == "lab/blocks"


def test_the_page_makes_campaigns_and_switches_recording(ses):
    from assemblyloop.dashboard import make_server

    srv = make_server(ses, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(path, body=None, token=srv.token):
        req = urllib.request.Request(url + path, None if body is None else json.dumps(body).encode(), {"Content-Type": "application/json", "X-Token": token})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    try:
        code, st = call("/api/record")
        assert code == 200 and st["on"] is False and st["campaigns"] == []
        assert call("/api/campaign", dict(name="Pick place"), token="wrong")[0] == 403
        code, c = call("/api/campaign", dict(name="Pick place"))
        assert code == 200 and c["slug"] == "pick-place"
        assert call("/api/campaign", dict(name="Pick place"))[0] == 409  # there is one of that name
        assert call("/api/record", dict(on=True, campaign="nothing"))[0] == 409
        assert call("/api/record", dict(on=True, campaign="pick-place")) == (200, dict(ok=True))
        code, st = call("/api/record")
        assert st["on"] and st["campaign"] == "Pick place" and st["cameras"] == ["head_left", "gripper_right_top"] and st["state"] == "idle"
        page = urllib.request.urlopen(url + "/").read().decode() + urllib.request.urlopen(url + "/web/app.js").read().decode()
        assert all(x in page for x in ('id="recbtn"', 'id="recpanel"', "/api/record", "/api/campaign"))
        assert call("/api/record", dict(on=False)) == (200, dict(ok=True)) and call("/api/record")[1]["on"] is False
    finally:
        srv.shutdown()
        srv.server_close()
