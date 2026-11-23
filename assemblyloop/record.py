"""What the robot did and what the cameras saw while it did it, written down so that a policy can be trained on it.
Nothing here moves the arm, decides a move or knows the session: the recorder is handed two functions and calls them.

    rec = Recorder("assembly_logs", sample, frames)      sample() -> the robot NOW (the row below); frames() -> the newest
    rec.begin("stack the blocks into a tower")           JPEG of every camera, as `feeds.Feeds.latest` gives them
    rec.mark("step", name="lower onto B")                anything worth knowing later, from any thread
    rec.end("done", success=True)

An EPISODE is one attempt at one task, said in words (`task` is what a vision-language-action policy is told). While
one is open a thread asks `sample()` and `frames()` `hz` times a second, on a schedule that does not drift: tick k is
due k / hz after the start whatever the ticks before it cost. A tick that comes more than half a period late is not
made up for by two in a row: its slot is left out and counted as dropped (`tick` in the rows shows the gap).

On disk, under <root>/episodes/<YYYYmmdd_HHMMSS>_<nnn>/ :

    episode.json   written at begin: {"schema": 1, "id", "task", "meta": {...}, "hz", "cams": [ids], "t0", "mono0",
                   "started", "finished": false}; written again at end with "finished": true, "outcome", "success",
                   "t1", "duration_s", "rows", "events", "pictures": {cam: how many files}, "dropped", "bytes",
                   "errors", "end": {...}. An episode whose file still says "finished": false (or has none) was cut
                   short by a crash: everything up to the last flushed line is there and `load` reads it.
    rows.jsonl     one line per tick: {"i": row number, "tick": slot number, "t": wall clock [s], "mono": monotonic
                   clock [s], ...what sample() gave..., "cams": {cam: {"file": "cam/<cam>/000123.jpg", "seq",
                   "age_ms"}}, "err": only if something failed}
    events.jsonl   one line per mark: {"t", "mono", "i": the number of the newest row (-1: none yet), "kind", ...}.
                   The recorder itself marks "begin" and "end".
    cam/<cam>/<6 digits>.jpg   the JPEG bytes exactly as they came, named after the row they were first seen in. A
                   camera whose `seq` has not moved since the tick before is NOT written again: its entry repeats the
                   file, with the age the picture has by now, so every row points at the newest picture there is.

THE ROW that `sample()` returns (the contract between the session and everything that reads the data; millimetres and
degrees in each arm's base frame, as the controller reports them). Arms are keyed by side, so that data from one arm
and from two have one format:

    {"arms": {"right": {"joints": [7 floats, deg], "xyz": [3, mm], "rpy": [3, deg],
                        "grip": int counts 0 (shut) .. 850 (open) or None, "moving": bool,
                        "goal_xyz": [3] or None, "goal_grip": int or None,    what the arm was last asked for
                        "what": "name of the step being executed" or ""},
              "left": {...}},                                                  only with two arms
     "held": bool,              the hold-to-run switch is held
     "phase": "RUNNING" | ...,  the session's label
     "step": str, "leg": str}   the current step / the current sub-task in words

`sample()` only reads what is published anyway and must not wait for the controller. If it or `frames()` raises, the row
is written without that part and says so in "err"; the thread goes on. Lines are appended and flushed one at a time and
pictures are plain files, by one writer thread: neither `mark` nor a slow disk holds up whoever calls.
"""
from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from pathlib import Path

import numpy as np

SCHEMA = 1
SIDES = ("right", "left")  # the order in which arms are put side by side wherever they become one vector
GRIP_OPEN = 850  # the gripper's counts when fully open (`cycle.Params.open_counts`)
OWN_KEYS = ("i", "tick", "t", "mono", "cams", "err")  # a row's keys that are the recorder's, not sample()'s


def plain(o):
    """For json.dumps: numpy arrays and numbers, paths and whatever else turns up in a row or a mark."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    if isinstance(o, (bytes, bytearray)):
        return f"<{len(o)} bytes>"
    return str(o)


def line(row):
    """One JSON line. A row that cannot be packed is replaced by one that says so: a line is never half written."""
    try:
        return json.dumps(row, default=plain, separators=(",", ":")) + "\n"
    except Exception as e:  # noqa: BLE001  a key that is not a string, a loop in the data ...
        keep = {k: row[k] for k in ("i", "tick", "t", "mono", "kind") if isinstance(row.get(k), (int, float, str))}
        return json.dumps(dict(keep, err=f"could not be written: {type(e).__name__}: {e}")) + "\n"


def safe_name(cam):
    return re.sub(r"[^\w.-]", "_", str(cam)) or "_"


def next_slot(last, periods, slack=0.5):
    """The slot the next tick falls in: `last` is the slot of the tick before, `periods` the time since the start of
    the episode in periods (1 / hz). Usually last + 1; later if more than `slack` of a period was lost."""
    return max(int(last) + 1, int(np.floor(float(periods) + 1.0 - slack)))


def write_json(path, data):
    """Written beside and moved into place, so that a crash leaves the old file or the new one, never half of one."""
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=plain, indent=1))
    os.replace(tmp, path)


class _Episode:
    def __init__(self, path, eid, task, meta, hz, t0, mono0):
        self.dir, self.id, self.task, self.meta, self.hz, self.t0, self.mono0 = path, eid, task, meta, hz, t0, mono0
        self.rows_fh, self.events_fh = open(path / "rows.jsonl", "a"), open(path / "events.jsonl", "a")
        self.stop, self.thread, self.lock = threading.Event(), None, threading.Lock()
        self.i, self.slot, self.dropped, self.events, self.bytes, self.errors, self.skipped = -1, -1, 0, 0, 0, 0, 0
        self.seen, self.pictures, self.made, self.said, self.last_err = {}, {}, set(), set(), ""  # seen: cam -> (seq, file, age at last look)

    def head(self):
        return dict(schema=SCHEMA, id=self.id, task=self.task, meta=self.meta, hz=self.hz, cams=sorted(self.seen), t0=self.t0, mono0=self.mono0,
                    started=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.t0)), finished=False)


class Recorder:
    """See the module's text. Beyond what the session calls: `mono` is the clock the schedule runs on, `thread=False`
    starts no thread at all (every row is then asked for with `tick()` and written at once: for tests), `stale_ms` is
    how old a picture may be when it is first written (a camera that was closed still has its last picture), and
    `backlog` is how many writes may wait for the disk before pictures are left out rather than kept in memory."""

    def __init__(self, root, sample, frames=None, hz=10.0, clock=time.time, say=print, mono=time.monotonic, thread=True, stale_ms=2000, backlog=300):
        if not float(hz) > 0:
            raise ValueError("hz must be above zero")
        self.root, self.sample, self.frames, self.hz, self.clock, self.say, self.mono = Path(root), sample, frames, float(hz), clock, say or (lambda _m: None), mono
        self.threaded, self.stale_ms, self.backlog = bool(thread), stale_ms, int(backlog)
        self._lock, self._ep, self._q, self._writer, self._closed, self.last = threading.RLock(), None, queue.SimpleQueue(), None, False, None

    # ------------------------------------------------------------------ what the session calls
    @property
    def active(self):
        return self._ep is not None

    def begin(self, task, **meta):
        """A new episode: its folder, its first row at once, and the thread that samples from then on. `task` is the
        instruction in words; `meta` is anything that describes the attempt (which arm, which blocks, a rehearsal ...).
        An episode that is still open is ended first, as interrupted."""
        with self._lock:
            if self._closed:
                raise RuntimeError("the recorder was closed")
            if self._ep is not None:
                self.end("interrupted: a new episode began")
            t0, mono0 = float(self.clock()), float(self.mono())
            home = self.root / "episodes"
            home.mkdir(parents=True, exist_ok=True)
            n = 1 + max([int(m.group(1)) for m in (re.search(r"_(\d+)$", p.name) for p in home.iterdir()) if m] + [0])
            while True:
                eid = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime(t0))}_{n:03d}"
                try:
                    (home / eid).mkdir()
                    break
                except FileExistsError:
                    n += 1
            ep = _Episode(home / eid, eid, str(task), meta, self.hz, t0, mono0)
            write_json(ep.dir / "episode.json", ep.head())
            if self.threaded and (self._writer is None or not self._writer.is_alive()):
                self._writer = threading.Thread(target=self._write_loop, daemon=True, name="record:write")
                self._writer.start()
            self._ep = ep
            self._event(ep, "begin", dict(task=ep.task))
            self._tick(ep)
            write_json(ep.dir / "episode.json", ep.head())  # now with the cameras that answered
            if self.threaded:
                ep.thread = threading.Thread(target=self._run, args=(ep,), daemon=True, name="record:sample")
                ep.thread.start()
            self.say(f"recording {eid}: {ep.task}")
            return eid

    def mark(self, kind, **kw):
        """Something happened now: a step began, a grasp held or did not, the block finder saw this. Costs a clock
        reading and a queue entry; outside an episode it does nothing."""
        ep = self._ep
        if ep is not None:
            self._event(ep, kind, kw)

    def tick(self):
        """One row now (what the thread does `hz` times a second). Returns the row, or None outside an episode."""
        ep = self._ep
        return None if ep is None else self._tick(ep)

    def end(self, outcome, success=None, **kw):
        """Stops sampling and writes the summary. `outcome` is how it ended in words, `success` whether the task was
        done (None: nobody said). Returns the episode's folder, or None outside an episode."""
        with self._lock:
            ep, self._ep = self._ep, None
            if ep is None:
                return None
            ep.stop.set()
            if ep.thread is not None and ep.thread is not threading.current_thread():
                ep.thread.join(5.0)
            self._event(ep, "end", dict(outcome=outcome, success=success, **kw))
            done = threading.Event()
            self._put(("close", ep, done))
            if not done.wait(10.0):
                self.say(f"recording {ep.id}: the disk did not keep up, the last lines may be missing")
            t1 = float(self.clock())
            summary = dict(ep.head(), finished=True, outcome=outcome, success=success, t1=t1, duration_s=round(t1 - ep.t0, 3), rows=ep.i + 1, events=ep.events,
                           pictures=dict(sorted(ep.pictures.items())), dropped=ep.dropped, bytes=ep.bytes, errors=ep.errors, end=kw)
            try:
                write_json(ep.dir / "episode.json", summary)
            except Exception as e:  # noqa: BLE001  a full disk must not take the session down: the episode stays "unfinished"
                self.say(f"recording {ep.id}: the summary could not be written ({type(e).__name__}: {e})")
            self.last = dict(episode=ep.id, dir=str(ep.dir), outcome=outcome, success=success, rows=ep.i + 1, dropped=ep.dropped, bytes=ep.bytes, duration_s=round(t1 - ep.t0, 3))
            self.say(f"recorded {ep.id}: {ep.i + 1} rows, {sum(ep.pictures.values())} pictures, {ep.dropped} ticks dropped ({outcome})")
            return ep.dir

    def close(self):
        with self._lock:
            self.end("closed")
            self._closed = True
            self._put(("quit",))

    def stats(self):
        """For the dashboard. `cams` is the age [ms] each camera's newest picture had at the last tick."""
        ep = self._ep
        if ep is None:
            return dict(active=False, episode=None, last=self.last)
        return dict(active=True, episode=ep.id, task=ep.task, dir=str(ep.dir), hz=self.hz, seconds=round(float(self.mono()) - ep.mono0, 1), rows=ep.i + 1, events=ep.events,
                    pictures=sum(ep.pictures.values()), dropped=ep.dropped, bytes=ep.bytes, errors=ep.errors, error=ep.last_err, waiting=self._q.qsize(),
                    cams={cam: v[2] for cam, v in sorted(ep.seen.items())}, last=self.last)

    # ------------------------------------------------------------------ sampling
    def _run(self, ep):
        while not ep.stop.is_set():
            now = float(self.mono())  # the slot that is next from here (not simply the one after the last), and never before it is due
            wait = ep.mono0 + next_slot(ep.slot, (now - ep.mono0) * ep.hz) / ep.hz - now
            if wait > 0 and ep.stop.wait(wait):
                break
            try:
                self._tick(ep)
            except Exception as e:  # noqa: BLE001  whatever it is, the next tick is tried again
                self._trouble(ep, f"tick: {type(e).__name__}: {e}")
                ep.stop.wait(1.0 / ep.hz)

    def _trouble(self, ep, err):
        ep.errors, ep.last_err = ep.errors + 1, err
        kind = err.split(":")[0]
        if kind not in ep.said:  # said once per episode and kind; every row that has it says so in "err"
            ep.said.add(kind)
            self.say(f"recording {ep.id}: {err}")

    def _tick(self, ep):
        with ep.lock:  # one tick at a time, whoever asks
            if ep.stop.is_set():
                return None
            mono, errs, jobs = float(self.mono()), [], []
            slot = next_slot(ep.slot, (mono - ep.mono0) * ep.hz) if ep.slot >= 0 else 0
            ep.dropped += slot - ep.slot - 1
            i = ep.i + 1
            row = {}
            try:
                got = self.sample()
                if isinstance(got, dict):
                    row = {str(k): v for k, v in got.items() if k not in OWN_KEYS}
                else:
                    errs.append(f"sample: gave {type(got).__name__}, not a dict")
            except Exception as e:  # noqa: BLE001
                errs.append(f"sample: {type(e).__name__}: {e}")
            cams = {}
            try:
                behind = self._q.qsize() > self.backlog
                for f in (self.frames() if self.frames is not None else None) or ():
                    cam, seq, jpg, age = safe_name(f["id"]), f.get("seq") or 0, f.get("jpg") or b"", f.get("age_ms")
                    was = ep.seen.get(cam)
                    if jpg and (was is None or seq != was[0]) and not (age is not None and self.stale_ms is not None and age > self.stale_ms):
                        if behind:
                            ep.skipped += 1
                        else:
                            was = (seq, f"cam/{cam}/{i:06d}.jpg", age)
                            jobs.append((cam, was[1], bytes(jpg)))
                    if was is not None:
                        ep.seen[cam] = was = (was[0], was[1], age)
                        cams[cam] = dict(file=was[1], seq=was[0], age_ms=age)
                if behind:
                    errs.append("frames: the disk is behind, new pictures were left out")
            except Exception as e:  # noqa: BLE001
                errs.append(f"frames: {type(e).__name__}: {e}")
            row = dict(i=i, tick=slot, t=round(float(self.clock()), 4), mono=round(mono, 4), **row, cams=cams)
            if errs:
                row["err"] = "; ".join(errs)
                for e in errs:
                    self._trouble(ep, e)
            ep.i, ep.slot = i, slot
            self._put(("row", ep, row, jobs))
            return row

    def _event(self, ep, kind, kw):
        row = dict(t=round(float(self.clock()), 4), mono=round(float(self.mono()), 4), i=ep.i, kind=str(kind), **{k: v for k, v in kw.items() if k not in ("t", "mono", "i", "kind")})
        ep.events += 1
        self._put(("event", ep, row))

    # ------------------------------------------------------------------ writing (one thread; in the caller's with thread=False)
    def _put(self, job):
        if self.threaded and self._writer is not None and self._writer.is_alive():
            self._q.put(job)
        else:
            self._write(job)

    def _write_loop(self):
        while True:
            job = self._q.get()
            if job[0] == "quit":
                return
            self._write(job)

    def _write(self, job):
        kind, ep = job[0], job[1] if len(job) > 1 else None
        try:
            if kind == "row":
                for cam, name, jpg in job[3]:
                    if cam not in ep.made:
                        (ep.dir / "cam" / cam).mkdir(parents=True, exist_ok=True)
                        ep.made.add(cam)
                    (ep.dir / name).write_bytes(jpg)
                    ep.pictures[cam], ep.bytes = ep.pictures.get(cam, 0) + 1, ep.bytes + len(jpg)
                text = line(job[2])
                ep.rows_fh.write(text)
                ep.rows_fh.flush()
                ep.bytes += len(text)
            elif kind == "event":
                text = line(job[2])
                ep.events_fh.write(text)
                ep.events_fh.flush()
                ep.bytes += len(text)
        except Exception as e:  # noqa: BLE001  a full disk, a folder that was removed: said, counted, and the program goes on
            self._trouble(ep, f"write: {type(e).__name__}: {e}")
        if kind == "close":
            for fh in (ep.rows_fh, ep.events_fh):
                try:
                    fh.flush()
                    os.fsync(fh.fileno())
                    fh.close()
                except Exception:  # noqa: BLE001
                    pass
            job[2].set()


# ---------------------------------------------------------------------- reading it back (nothing below needs a Recorder)
def read_lines(path):
    """The rows of a .jsonl file, and how many lines could not be read (a crash can leave the last one cut off)."""
    rows, torn = [], 0
    try:
        with open(path, errors="replace") as fh:
            for text in fh:
                if not text.strip():
                    continue
                try:
                    row = json.loads(text)
                except ValueError:
                    torn += 1
                    continue
                if isinstance(row, dict):
                    rows.append(row)
                else:
                    torn += 1
    except OSError:
        pass
    return rows, torn


class Episode:
    """One recorded episode, read back. `meta` is episode.json (made up from the folder if a crash left none),
    `finished` whether its summary was written, `rows` and `events` the lines that can be read, `torn` how many
    could not."""

    def __init__(self, path):
        self.dir = Path(path)
        try:
            self.meta = json.loads((self.dir / "episode.json").read_text())
            if not isinstance(self.meta, dict):
                raise ValueError("not an episode")
        except (OSError, ValueError):
            self.meta = dict(schema=SCHEMA, id=self.dir.name, task="", meta={}, hz=None, cams=[], finished=False)
        self.rows, torn_rows = read_lines(self.dir / "rows.jsonl")
        self.events, torn_events = read_lines(self.dir / "events.jsonl")
        self.torn = torn_rows + torn_events

    id = property(lambda self: str(self.meta.get("id") or self.dir.name))
    task = property(lambda self: str(self.meta.get("task") or ""))
    finished = property(lambda self: bool(self.meta.get("finished")))
    success = property(lambda self: self.meta.get("success") if self.finished else None)

    @property
    def hz(self):
        """How often it was sampled: what episode.json says, else what the rows' own clock says."""
        if self.meta.get("hz"):
            return float(self.meta["hz"])
        t = row_times(self.rows)
        return float(1.0 / np.median(np.diff(t))) if len(t) > 2 and np.median(np.diff(t)) > 0 else 10.0

    @property
    def cams(self):
        return sorted({cam for r in self.rows for cam in (r.get("cams") or {})})

    def sides(self):
        """The arms in the rows, right before left, any other after them."""
        found = {s for r in self.rows for s in (r.get("arms") or {})}
        return [s for s in SIDES if s in found] + sorted(found.difference(SIDES))

    def picture_path(self, row, cam):
        """Where the picture is that `row` (a row or its number) points at for camera `cam`, or None."""
        row = self.rows[row] if isinstance(row, int) else row
        name = ((row.get("cams") or {}).get(cam) or {}).get("file")
        return None if not name else self.dir / name

    def picture(self, row, cam):
        """The JPEG bytes of that picture, or None if the row has none for this camera or the file is not there."""
        path = self.picture_path(row, cam)
        try:
            return None if path is None else path.read_bytes()
        except OSError:
            return None


def load(episode_dir):
    return Episode(episode_dir)


def is_episode(path):
    path = Path(path)
    return path.is_dir() and ((path / "rows.jsonl").exists() or (path / "episode.json").exists())


def episodes(root):
    """The episode folders under `root`, oldest first. `root` may be what the Recorder was given, its `episodes`
    folder, or one episode."""
    root = Path(root)
    if is_episode(root):
        return [root]
    home = root / "episodes" if (root / "episodes").is_dir() else root
    return sorted(p for p in home.iterdir() if is_episode(p)) if home.is_dir() else []


def row_times(rows):
    """Seconds since the first row, by the clock that cannot jump (`mono`; the wall clock for rows without it)."""
    if not rows:
        return np.zeros(0)
    key = "mono" if all(isinstance(r.get("mono"), (int, float)) for r in rows) else "t"
    t = np.array([float(r.get(key) or 0.0) for r in rows])
    return t - t[0]


def arm_ok(arm):
    """Whether one arm's part of a row has what a training example needs: seven joints, a position, an orientation."""
    try:
        return len(arm["joints"]) == 7 and len(arm["xyz"]) == 3 and len(arm["rpy"]) == 3 and bool(np.all(np.isfinite(np.array(arm["joints"] + arm["xyz"] + arm["rpy"], float))))
    except (KeyError, TypeError, ValueError):
        return False
