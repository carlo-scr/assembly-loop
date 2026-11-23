"""One session's recording: episodes opened and closed by the job loop, written as the lab's recorder writes runs.

An EPISODE is one block taken and put down (one leg of a job: a move, a level of a tower, a block of a design). It
opens when the leg starts and its streams close when the arm has let go and lifted clear; what is learnt after that
(the camera's check of where the block landed, a missed grip seen by the gripper's camera) is added to run.json as
its outcome and labels. Rehearsals in the air are not recorded.

While a session records, a thread reads the arm about 30 times a second (`arm_state.jsonl`): the joints the
controller plans and the ones it measures, the flange pose in the arm's base frame and in the lab's world frame
(x forward, y left, z up: R_WORLD_FROM_BASE of swoosh-collect for the right arm, mirrored for the left), the
gripper, the error code, and whether the switch allows moving. Every command the guard sends is already in its own
log; the part of it that belongs to the episode is written to `commanded.jsonl` when the episode closes. Steps,
grips seen and pauses go to `events.jsonl`. All times are seconds since the episode's t0, on time.monotonic, the
clock the camera times are on as well.

Reading the arm never waits on the arm's moves and never steers them: a slow read is written down as slow.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from .store import Campaign, merge_json

PROGRAM = Path(__file__).resolve().parents[2]


def world_from_base(down):
    """The lab's world frame for an arm whose 'straight down' is `down` in its base frame: x forward (the base's x,
    levelled), y left, z up. For the rig's right arm this is swoosh-collect's R_WORLD_FROM_BASE."""
    z = -np.asarray(down, float)
    z = z / np.linalg.norm(z)
    x = np.array([1.0, 0.0, 0.0]) - z[0] * z
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])  # rows: the world's axes in the base frame


def git_state():
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=PROGRAM, capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return dict(sha=run("rev-parse", "--short", "HEAD") or "unknown", branch=run("rev-parse", "--abbrev-ref", "HEAD"), dirty=bool(run("status", "--porcelain")))


class Lines:
    """Line-buffered JSONL, one file per stream (as the lab's StreamWriter)."""

    def __init__(self, path):
        self.fh, self.rows = open(path, "w", buffering=1, encoding="utf-8"), 0

    def write(self, row):
        self.fh.write(json.dumps(row, separators=(",", ":"), default=_plain) + "\n")
        self.rows += 1

    def close(self):
        try:
            self.fh.close()
        except OSError:
            pass


def _plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return str(v)


class Recorder:
    def __init__(self, ses, campaign, rate_hz=30.0, size=(640, 480), fps=30.0, quality=8, validate=True, cameras="rig"):
        self.ses, self.campaign = ses, campaign if isinstance(campaign, Campaign) else Campaign(campaign)
        self.rate, self.size, self.fps, self.quality, self.do_validate = float(rate_hz), tuple(size), float(fps), int(quality), bool(validate)
        self.lock = threading.RLock()
        self.run, self.state, self.t0, self.streams, self.labels, self._stop = None, "idle", 0.0, {}, {}, None
        self.counts = dict(episodes=0, succeeded=0, failed=0)
        self.last, self.pending = None, []
        side = ses.chosen
        self.R = world_from_base(ses.arms[side].get("down", (0.0, 0.0, -1.0)))
        feeds = getattr(ses, "feeds", None)
        # Every camera of the rig, whichever arm moves (cameras="rig"): one camera set for every episode, as in the lab's
        # datasets (lego_assemblies_v2 has all six), so that both arms' episodes, and later both at once, are one dataset.
        # cameras="arm": only the cameras of this session's arm. The stand-in always records its own.
        if feeds is None:
            self.cams = []
        elif cameras == "rig" and not ses.fake:
            self.cams = [dict(id=o.id, key=o.key, label=o.label, device=o.device) for o in feeds.owners if {"right", "left"} & set(o.arms)]
        else:
            self.cams = feeds.cameras_of(side)
        if feeds is not None and self.cams:
            feeds.record([c["id"] for c in self.cams], self.size, self.fps)
        self.provenance = self._provenance()

    # ------------------------------------------------------------------ what was true when this was recorded
    def _provenance(self):
        ses = self.ses
        out = dict(program=dict(name="assembly-loop", **git_state()), recorded_by="script", clock="time.monotonic; every t is seconds since the run's t0, cameras included",
                   arm=dict(side=ses.chosen, label=ses.arms[ses.chosen]["label"], ip=ses.arms[ses.chosen].get("ip"), stand_in=bool(ses.fake),
                            R_world_from_base=self.R.round(6).tolist(), world="x forward, y left, z up (swoosh-collect frames.py)"),
                   recording=dict(arm_rate_hz=self.rate, video_size=list(self.size), video_fps=self.fps, mp4_quality=self.quality, cameras=self.cams),
                   params=dataclasses.asdict(ses.P), limits=dict(getattr(ses, "limit_kw", {}) or {}), guard=dict(getattr(ses, "guard_kw", {}) or {}))
        try:
            out["controller"] = ses.arm.status()  # error, mode, tcp offset and load, collision sensitivity: as the controller says
        except Exception as e:  # noqa: BLE001  the stand-in, or a controller that does not answer
            out["controller"] = dict(error=f"{type(e).__name__}: {e}")
        try:
            out["catalogue"] = ses.catalogue().to_dict()
        except Exception:  # noqa: BLE001  no blocks.json: a taught-spot job needs none
            out["catalogue"] = None
        setup = getattr(ses, "setup", None)
        if setup is not None:
            out["camera_fit"] = dict(fit_mm=setup.fit_mm, when=setup.when, side=setup.side)
        if ses.spots is not None:
            out["spots"] = ses.spots.to_dict()
        return out

    # ------------------------------------------------------------------ an episode
    def begin(self, job, k, leg, text):
        """A leg of a job starts: a new run, its streams, its clips, the arm read from now on."""
        with self.lock:
            if self.state != "idle":
                self.finish("ended by the next episode starting")
            ses, sp = self.ses, job.get("spots") or self.ses.spots
            up = sp.up
            src, dst = leg["src"], leg["dst"]
            source = leg.get("source") or {}
            kind, rest = source.get("kind"), source.get("rest")
            task = dict(job=job.get("what") or ("tower" if job.get("tower") else "move"), leg=k, of=job.get("total"), text=text, instruction=instruction(job, k, leg),
                        src=src, dst=dst, src_up_mm=leg.get("su", 0.0), dst_up_mm=leg.get("du", 0.0),
                        src_flange_base_mm=(sp.grasp(src) + float(leg.get("su", 0.0)) * up).round(2).tolist(), dst_flange_base_mm=(sp.grasp(dst) + float(leg.get("du", 0.0)) * up).round(2).tolist(),
                        src_rpy_deg=np.asarray(sp.turn(src)).round(3).tolist(), dst_rpy_deg=np.asarray(sp.turn(dst)).round(3).tolist(),
                        block=None if kind is None else dict(kind=kind.name, shape=kind.shape, size_mm=list(kind.size), colour=kind.colour, rest=getattr(rest, "name", None)),
                        width_mm=leg.get("width"), release_counts=leg.get("release"), grasp_offset_mm=float(leg.get("grasp_offset_mm", 0.0)), design=job.get("design"),
                        action_source="script", arm=ses.chosen)
            task["dst_flange_world_mm"] = (self.R @ np.asarray(task["dst_flange_base_mm"])).round(2).tolist()
            self.run = self.campaign.new_run(task=task, provenance=self.provenance, cameras=[c["key"] for c in self.cams])
            self.t0 = time.monotonic()
            raw = self.run.path / "raw"
            self.streams = {n: Lines(raw / f"{n}.jsonl") for n in ("arm_state", "commanded", "events")}
            self.labels, self.state = dict(grips=[], checks=[], grip_seen=[]), "recording"
            feeds = getattr(ses, "feeds", None)
            if feeds is not None and self.cams:
                feeds.start_clip(self.run.path / "video", self.t0, self.quality)
            self._stop = threading.Event()
            self._poller = threading.Thread(target=self._poll, args=(self._stop, self.t0, self.streams["arm_state"]), daemon=True, name="record:arm")
            self._poller.start()
            self.event("episode_start", **{k_: task[k_] for k_ in ("instruction", "src", "dst", "leg")})

    def event(self, kind, **kw):
        """Something that happened. During the motion it goes to events.jsonl; after it, into the run's labels."""
        with self.lock:
            if self.state == "recording":
                self.streams["events"].write(dict(t=round(time.monotonic() - self.t0, 4), kind=kind, **kw))
            elif self.state == "labelling":
                pic = kw.get("picture")
                if pic and Path(pic).is_file():  # the check's picture (the block where it landed) is kept with the run
                    (self.run.path / "stills").mkdir(exist_ok=True)
                    dst = self.run.path / "stills" / Path(pic).name
                    dst.write_bytes(Path(pic).read_bytes())
                    kw = {**kw, "picture": f"stills/{dst.name}"}
                self.labels.setdefault(kind + "s" if kind in ("check", "grip_seen") else kind, []).append(dict(t=round(time.monotonic() - self.t0, 4), **kw))

    def motion_done(self, rec=None):
        """The arm has let go and lifted clear: the streams and clips of this episode close; labels may still come."""
        with self.lock:
            if self.state != "recording":
                return
            t_end = time.monotonic() - self.t0
            self.event("motion_end")
            self._stop.set()
            self._poller.join(timeout=2.0)
            feeds = getattr(self.ses, "feeds", None)
            if feeds is not None and self.cams:
                feeds.stop_clip()
            g = self.ses.guard
            rows = []
            for row in reversed(getattr(g, "events", []) or []):  # the guard's own log of what it was told, this episode's part
                if row.get("t", -1e9) < self.t0:
                    break
                rows.append({**row, "t": round(float(row["t"]) - self.t0, 4)})
            for row in reversed(rows):
                self.streams["commanded"].write(row)
            for s in self.streams.values():
                s.close()
            if rec:
                self.labels["grips"] = [dict(step=k, got_counts=v) for k, v in (rec.get("grips") or {}).items()]
                self.labels["blocked"], self.labels["rested"] = rec.get("blocked"), rec.get("rested")
            self.labels["motion_s"] = round(t_end, 3)
            self.state = "labelling"

    def finish(self, outcome="done"):
        """The episode is over: its outcome and labels into run.json, then it is checked (in the background)."""
        with self.lock:
            if self.state == "idle":
                return
            if self.state == "recording":
                self.motion_done()
            run, labels = self.run, self.labels
            kind, success = classify(outcome, labels)
            leg = (run.meta.get("task") or {}).get("leg")
            mine = [r for c in labels.get("checks", []) for r in c.get("rows", []) if r.get("step") == (leg or 0) + 1 + (c.get("offset") or 0)]
            placed = mine[-1] if mine else None
            merge_json(run.path / "run.json", status="processing", wall_end_unix=time.time(), duration_s=round(time.monotonic() - self.t0, 3),
                       outcome=dict(text=str(outcome), kind=kind, success=success, verified_by_camera=bool(labels.get("checks")), placed=placed),
                       labels=labels, stream_rows={n: s.rows for n, s in self.streams.items()})
            self.counts["episodes"] += 1
            self.counts["succeeded" if success else "failed"] += 1
            self.last = dict(run=run.name, outcome=kind, success=success, instruction=(run.meta.get("task") or {}).get("instruction"))
            self.run, self.state, self.streams = None, "idle", {}
        if self.do_validate:
            t = threading.Thread(target=self._validate, args=(run,), daemon=True, name="record:validate")
            t.start()
            self.pending.append(t)

    def _validate(self, run):
        from .validate import validate_run

        try:
            validate_run(run.path)
        except Exception as e:  # noqa: BLE001  a check that fails to run leaves the run "processing", and says why
            merge_json(run.path / "run.json", validation=dict(ok=False, checks=[dict(check="validate", level="fail", detail=f"{type(e).__name__}: {e}")]))

    def close(self, why="recording switched off"):
        with self.lock:
            if self.state != "idle":
                self.finish(f"stopped: {why}")
            feeds = getattr(self.ses, "feeds", None)
            if feeds is not None and self.cams:
                feeds.record(None)
        for t in self.pending:
            t.join(timeout=30.0)

    def status(self):
        with self.lock:
            return dict(on=True, campaign=self.campaign.name, slug=self.campaign.path.name, folder=str(self.campaign.path), state=self.state, episode=self.run.name if self.run else None,
                        cameras=[c["key"] for c in self.cams], **self.counts, last=self.last)

    # ------------------------------------------------------------------ the arm, about 30 times a second
    def _poll(self, stop, t0, out):
        ses, a, R = self.ses, self.ses.arm, self.R
        period, nxt, k = 1.0 / self.rate, time.monotonic(), 0
        while not stop.is_set():
            k += 1
            row, t_read = dict(t=round(time.monotonic() - t0, 4)), time.monotonic()
            try:
                xyz, rpy = np.asarray(a.xyz_now(), float), np.asarray(a.rpy_now(), float)
                row.update(pose_base=np.r_[xyz, rpy].round(3).tolist(), pose_world_xyz_mm=(R @ xyz).round(3).tolist())
                planned = np.asarray(a.joints_view() if hasattr(a, "joints_view") else a.joints_now(), float)
                row["joints_planned_deg"] = planned.round(3).tolist()
                if ses.fake:  # the stand-in has no joints of its own: what is drawn is what there is
                    row["joints_measured_deg"], row["measured_source"] = row["joints_planned_deg"], "stand-in"
                else:
                    row["joints_measured_deg"] = np.asarray(a.joints(True), float).round(3).tolist()
                if k % 3 == 1:  # the gripper ten times a second is plenty
                    row["gripper_counts"] = _gripper(a)
                row["error"], row["moving"] = int(a.fault_now()), bool(a.moving_now())
                gate = ses.gate
                row["switch_held"] = bool(gate.held()) if gate is not None else False
            except Exception as e:  # noqa: BLE001  written down, never raised: reading must not disturb the arm
                row["read_error"] = f"{type(e).__name__}: {e}"
            row["read_s"] = round(time.monotonic() - t_read, 4)
            out.write(row)
            nxt += period
            stop.wait(max(0.0, nxt - time.monotonic()))
            if time.monotonic() - nxt > 1.0:  # far behind (a slow controller): start again from now, the gap is in the times
                nxt = time.monotonic()


def _gripper(a):
    api = getattr(a, "api", None)
    if api is not None:
        code, pos = api.get_gripper_position()
        return None if code != 0 or pos is None else float(pos)
    return float(getattr(a, "counts", 0))


def instruction(job, k, leg):
    """What this episode asks for, in words: the task a language-conditioned policy is given."""
    source = leg.get("source") or {}
    kind = source.get("kind")
    what = f"the {kind.name}" if kind is not None else "the block"
    if job.get("plan") is not None:  # a design
        name = job.get("design") or (job.get("what") or "the design")
        return f"put {what} in its place in {name}" if name.startswith("the design") else f"put {what} in its place in the design {name}"
    if job.get("tower"):
        return f"put {what} on top of the tower" if float(leg.get("du", 0.0)) > 1.0 else f"put {what} down where the tower stands"
    return f"move {what} from {leg['src']} to {leg['dst']}"


def classify(outcome, labels):
    """(kind, success) of an episode from how its job ended and what was seen after."""
    t = str(outcome).lower()
    checks = labels.get("checks") or []
    if t == "done":
        bad = any(c.get("problems") for c in checks)
        return ("placed" if not bad else "misplaced"), not bad
    for key, kind in (("the camera", "misplaced"), ("blocked", "blocked"), ("closed to", "wrong_grip"), ("no brick", "nothing_held"), ("stayed behind", "nothing_held"),
                      ("stopped", "stopped"), ("armfault", "arm_fault"), ("refused", "refused"), ("next episode", "interrupted")):
        if key in t:
            return kind, False
    return "failed", False
