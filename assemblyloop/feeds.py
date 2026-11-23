"""Live pictures from the rig's cameras, for the dashboard. Nothing here moves the arm and nothing here decides a move:
the pictures are for the person at the stop button.

A V4L2 camera streams to ONE program at a time. So every camera has one owner here (a thread that keeps its newest
picture) and the page asks the owners. The block finder takes its own, larger picture from one of the same cameras:
for that moment the owner lets go of it (`borrow`), and the page says so.

Cameras are opened only while a page is watching, and are let go a few seconds after the last page stops asking, so
that the lab's other tools find them free.

Cameras are known by the USB port they are plugged into, as in the lab's own tools: /dev/videoN is renumbered at every
boot, and all six cameras report the same serial number. Move a cable to another port and `RIG_CAMERAS` is wrong.

A picture that stopped arriving must not look live: every picture carries its age, and the page shows it.

RECORDING (`record.Recorder` asks for it): while `recording` is set, the cameras it names stay open whether or not a
page watches, at the recording size and rate, and every picture is read and unpacked. Between `start_clip` and
`stop_clip` each one is also written to `<folder>/<key>.mp4`, and its time to `<key>_frame_times.json`, as the
lab's teleoperation recorder does (swoosh-collect, cameras.py): the time is the kernel's own stamp of the buffer
(the same clock as time.monotonic) when that is sane, the time after the read otherwise, and every field of that
file means what it means there. A camera lent to the block finder in the middle of a clip leaves a gap in its times;
the gap is in the file, not hidden. `key` is the lab's dataset name of the camera (head_left, gripper_right_top, ...).
"""
from __future__ import annotations

import contextlib
import os
import threading
import time

import numpy as np

RIG_PATH = "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:{port}:1.0-video-index0"
# the names are the lab's own (PO-Assembly, pi05/deploy/cameras.json, confirmed there by an operator on 2 Sep 2026)
# `key`: the camera's name in the lab's datasets (lego_assemblies_v2: observation.images.<key>)
RIG_CAMERAS = (dict(id="scene_left", key="head_left", label="Scene, left", port="4.4.1", arms=("right", "left")),
               dict(id="scene_right", key="head_right", label="Scene, right", port="4.3", arms=("right", "left")),
               dict(id="grip_right_upper", key="gripper_right_top", label="Right gripper, upper", port="4.4.4.2", arms=("right",)),
               dict(id="grip_right_lower", key="gripper_right_bottom", label="Right gripper, lower", port="4.4.4.1", arms=("right",)),
               dict(id="grip_left_upper", key="gripper_left_top", label="Left gripper, upper", port="4.4.3", arms=("left",)),
               dict(id="grip_left_lower", key="gripper_left_bottom", label="Left gripper, lower", port="4.4.2", arms=("left",)))
TABLE_CAMERA = RIG_PATH.format(port="4.3")  # the one the block finder looks through unless told otherwise (`eyes.SCENE_CAMERA`)


def rig_cameras():
    return [dict(id=c["id"], key=c["key"], label=c["label"], device=RIG_PATH.format(port=c["port"]), arms=c["arms"]) for c in RIG_CAMERAS]


def same_device(a, b):
    return bool(a) and bool(b) and os.path.realpath(str(a)) == os.path.realpath(str(b))


def open_v4l2(device, size):
    """A camera opened for a stream of small pictures. Raises OSError with a reason a person can act on."""
    import cv2

    real = os.path.realpath(str(device))
    if not os.path.exists(real):
        raise OSError("not found on this computer")
    cap = cv2.VideoCapture(real, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        raise OSError("does not open: another program is using it")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    cap.set(cv2.CAP_PROP_FPS, 30.0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # the newest picture, not one that has waited in a queue
    return cap


def mp4_writer(path, fps, quality=8):
    """The lab's video writer: H.264 through imageio's ffmpeg, frames given as RGB. Raises ImportError without it."""
    import imageio.v2 as imageio

    return imageio.get_writer(str(path), fps=float(fps), codec="libx264", quality=int(quality), macro_block_size=1)


def to_jpeg(im, wide=960, quality=72):
    """A picture as JPEG bytes for the page (made smaller if the camera gave a large one), and its size."""
    import cv2

    if im.shape[1] > wide:
        im = cv2.resize(im, (wide, int(round(im.shape[0] * wide / im.shape[1]))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("the picture could not be packed")
    return buf.tobytes(), (int(im.shape[1]), int(im.shape[0]))


class Owner(threading.Thread):
    """The one thread that has a camera. It keeps the newest picture, packed for the page."""

    def __init__(self, hub, spec):
        super().__init__(daemon=True, name=f"feed:{spec['id']}")
        self.hub, self.id, self.label, self.device, self.arms, self.make = hub, str(spec["id"]), str(spec["label"]), spec.get("device"), tuple(spec.get("arms", ())), spec.get("make")
        self.cap, self.jpg, self.seq, self.at, self.size, self.note, self.lent, self._shot = None, b"", 0, 0.0, None, "", False, 0.0
        self.busy = False  # the device is open, being opened or being closed: nobody else can have it yet (at the rig, closing takes a moment)
        self.key = str(spec.get("key") or spec["id"])
        self.opened_at = None  # the size the device was opened at
        self.latest, self.latest_t = None, 0.0  # while recording: the newest picture itself (for the gripper's camera check)
        self.clip, self.clip_done = None, threading.Event()  # the clip being written, and "it is closed"
        self._lags = []

    def _let_go(self):
        cap, self.cap = self.cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:  # noqa: BLE001  whatever a camera driver raises while closing
                pass
        self.busy = False

    def _recording(self):
        r = self.hub.recording
        return r if r is not None and self.id in r["ids"] else None

    def _stamp(self, now):
        """The kernel's time of this picture if it is on our clock and sane, else the time after the read."""
        try:
            pos = float(self.cap.get(5))  # cv2.CAP_PROP_POS_MSEC
        except Exception:  # noqa: BLE001  a camera that is not a device
            pos = 0.0
        if pos > 0.0 and 0.0 <= now - pos / 1000.0 < 0.5:
            if len(self._lags) < 900:
                self._lags.append(now - pos / 1000.0)
            return pos / 1000.0, "v4l2_buffer"
        return now, "post_read"

    def _clip_frame(self, im, t, source):
        """Write one picture into the open clip (opening it at its first picture)."""
        c = self.clip
        if c is None:
            return
        try:
            if c.get("writer") is None and not c.get("error"):
                c["writer"], c["actual"] = mp4_writer(c["path"], c["fps"], c.get("quality", 8)), dict(width=int(im.shape[1]), height=int(im.shape[0]))
            if c.get("writer") is not None:
                import cv2

                c["writer"].append_data(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                c["t"].append(round(t - c["t0"], 6))  # only after the write succeeded: frames and times can never disagree
                c["source"] = source
        except Exception as e:  # noqa: BLE001  a writer that failed ends this clip for this camera; it says so in its file
            c["error"] = f"{type(e).__name__}: {e}"
            c["stopped_early"], c["stopped_at_s"] = True, (c["t"][-1] if c["t"] else 0.0)

    def _close_clip(self):
        c, self.clip = self.clip, None
        if c is None:
            return
        try:
            if c.get("writer") is not None:
                c["writer"].close()
        except Exception as e:  # noqa: BLE001
            c["error"] = c.get("error") or f"{type(e).__name__}: {e}"
        ts = c["t"]
        span = (ts[-1] - ts[0]) if len(ts) > 1 else 0.0
        info = dict(label=self.key, device=self.device, frames=len(ts), measured_fps=round((len(ts) - 1) / span, 3) if span > 0 else 0.0, error=c.get("error"),
                    stopped_early=bool(c.get("stopped_early")), stopped_at_s=round(float(c.get("stopped_at_s") or 0.0), 3), timestamp_source=c.get("source", "post_read"),
                    read_lag_s=round(float(sorted(self._lags)[len(self._lags) // 2]), 4) if self._lags else 0.0, flipped_vertical=False, actual=c.get("actual") or {},
                    lent_s=[[round(a - c["t0"], 3), round(b - c["t0"], 3)] for a, b in c["lent"]], t=ts)
        try:
            import json

            (c["folder"] / f"{self.key}_frame_times.json").write_text(json.dumps(info))
        finally:
            c["done"].set()

    def run(self):
        hub, wait = self.hub, 0.5
        while not hub.closed:
            rec = self._recording()
            if self.clip is not None and (self.clip.get("stop") or rec is None):
                self._close_clip()
            if self.lent or not hub.wants(self):
                if self.lent and self.clip is not None and (not self.clip["lent"] or self.clip["lent"][-1][1] is not None):
                    self.clip["lent"].append([time.monotonic(), None])  # the gap this picture leaves in the clip
                self._let_go()
                if self.lent:
                    self.note = "lent to the block finder for one picture"
                time.sleep(0.03)
                continue
            if self.clip is not None and self.clip["lent"] and self.clip["lent"][-1][1] is None:
                self.clip["lent"][-1][1] = time.monotonic()
            want_size = tuple(rec["size"]) if rec is not None and self.make is None else hub.size
            if self.cap is not None and self.make is None and self.opened_at is not None and tuple(self.opened_at) != want_size:
                self._let_go()  # recording started or ended: opened again at the other size
            if self.cap is None:
                self.busy = True
                try:
                    self.cap = self.make() if self.make is not None else hub.open(self.device, want_size)
                    self.opened_at = want_size
                    self.note, wait = "", 0.5
                except Exception as e:  # noqa: BLE001  not there, busy, no OpenCV: say so and try again later
                    self.busy = False
                    self.note = str(e) or type(e).__name__
                    hub.nap(wait)
                    wait = min(wait * 2.0, 8.0)
                    continue
            try:
                if rec is None and time.monotonic() - self._shot < 1.0 / hub.fps and hasattr(self.cap, "grab"):
                    # A picture the page will not get is taken off the camera's queue WITHOUT being unpacked: unpacking
                    # every picture of four cameras used most of a processor in the program that also moves the arm.
                    # (While recording every picture is unpacked: it goes into the clip.)
                    if self.cap.grab():
                        continue
                ok, im = self.cap.read()
            except Exception:  # noqa: BLE001
                ok, im = False, None
            if not ok or im is None:
                self.note = "gives no picture"
                if self.clip is not None and self.clip.get("t"):
                    self.clip["lent"].append([time.monotonic(), None])  # a camera that went away: a gap, said so
                self._let_go()
                hub.nap(wait)
                wait = min(wait * 2.0, 8.0)
                continue
            now = time.monotonic()
            if rec is not None:
                t, source = self._stamp(now)
                self.latest, self.latest_t = im, t
                self._clip_frame(im, t, source)
            if now - self._shot < 1.0 / hub.fps:
                continue  # read and dropped: the camera's own queue stays empty, the page gets `fps` pictures a second
            self._shot = now
            try:
                jpg, size = hub.encode(im)
            except Exception as e:  # noqa: BLE001
                self.note = f"{type(e).__name__}: {e}"
                continue
            with hub.cond:
                hub.seq += 1
                self.jpg, self.seq, self.at, self.size, self.note = jpg, hub.seq, now, size, ""
                hub.cond.notify_all()
        if self.clip is not None:
            self._close_clip()
        self._let_go()


class Feeds:
    def __init__(self, cameras=None, size=(640, 480), fps=12.0, idle_s=4.0, open=open_v4l2, encode=to_jpeg):
        self.size, self.fps, self.idle_s, self.open, self.encode = tuple(size), float(fps), float(idle_s), open, encode
        self.cond, self.seq, self.closed, self.watched, self.side = threading.Condition(), 0, False, -1e9, None
        self.recording = None  # {"ids": cameras kept open and unpacked for recording, "size", "fps"}
        self.owners = []
        for spec in cameras if cameras is not None else rig_cameras():
            self.add(spec)

    def add(self, spec):
        """One more camera: {"id", "label", "arms": which arms it is shown with, "device" or "make" (a function that
        gives something with read() -> (ok, picture) and release(): a camera that is not a device)}."""
        self.owners.append(Owner(self, spec))

    def why_not(self):
        """'' if pictures can be packed for the page, otherwise what is missing."""
        if self.encode is to_jpeg:
            try:
                import cv2  # noqa: F401
            except ImportError:
                return 'the live pictures need OpenCV: pip install -e ".[eyes]"'
        return ""

    def wants(self, owner):
        if self.recording is not None and owner.id in self.recording["ids"]:
            return True
        return (time.monotonic() - self.watched) < self.idle_s and self.side in owner.arms

    # ------------------------------------------------------------------ recording
    def cameras_of(self, side):
        """The cameras that belong to the arm `side`: [{"id", "key", "label", "device"}]."""
        return [dict(id=o.id, key=o.key, label=o.label, device=o.device) for o in self.owners if side in o.arms]

    def record(self, ids, size=(1280, 720), fps=30.0):
        """Keep these cameras open at this size and unpack every picture (None: stop)."""
        if ids is None:
            self.recording = None
            return
        self.recording = dict(ids=set(ids), size=tuple(size), fps=float(fps))
        for o in self.owners:
            if o.id in self.recording["ids"] and not o.is_alive() and not o.ident and not self.closed:
                o.start()

    def start_clip(self, folder, t0, quality=8):
        """Every recorded camera writes its pictures to `<folder>/<key>.mp4` from now on; times are seconds since t0."""
        r = self.recording
        if r is None:
            return []
        out = []
        for o in self.owners:
            if o.id in r["ids"]:
                o.clip = dict(folder=folder, path=folder / f"{o.key}.mp4", t0=float(t0), fps=r["fps"], quality=quality, t=[], lent=[], done=threading.Event(), writer=None)
                out.append(o.key)
        return out

    def stop_clip(self, timeout=10.0):
        """Close every clip; returns once each camera has written its file (or after `timeout`)."""
        clips = [(o, o.clip) for o in self.owners if o.clip is not None]
        for o, c in clips:
            c["stop"] = True
            if (o.cap is None and not o.busy) or not o.is_alive():  # not streaming (a camera that does not open): closed here, nobody writes to it
                o._close_clip()
        end = time.monotonic() + timeout
        for _, c in clips:
            if not c["done"].wait(max(0.0, end - time.monotonic())):
                c["error"] = "the camera did not close its clip in time"
        return [o.key for o, c in clips]

    def latest(self, device, max_age_s=0.5):
        """While recording: the newest picture of this camera itself (not a copy for the page), or None."""
        for o in self.owners:
            if same_device(o.device, device) and o.latest is not None and time.monotonic() - o.latest_t <= max_age_s:
                return o.latest.copy()
        return None

    def nap(self, secs):
        end = time.monotonic() + float(secs)
        while not self.closed and time.monotonic() < end:
            time.sleep(0.05)

    def frames(self, side, after=0, wait_s=1.0):
        """The cameras that belong to the arm `side`, each with its newest picture if that is newer than number
        `after`. Waits up to `wait_s` for a new one. Returns (the highest number given out, the cameras)."""
        with self.cond:
            self.watched, self.side = time.monotonic(), side
            mine = [o for o in self.owners if side in o.arms]
            for o in mine:
                if not o.is_alive() and not o.ident and not self.closed:
                    o.start()
            end = time.monotonic() + float(wait_s)
            while not self.closed and not any(o.seq > after for o in mine):
                left = end - time.monotonic()
                if left <= 0:
                    break
                self.cond.wait(left)
            now = self.watched = time.monotonic()
            return max([after] + [o.seq for o in mine]), [dict(id=o.id, label=o.label, device=o.device, seq=o.seq, note=o.note, size=o.size, age_ms=None if not o.seq else int(1000 * (now - o.at)),
                                                               jpg=o.jpg if o.seq > after else b"") for o in mine]

    @contextlib.contextmanager
    def borrow(self, device):
        """While this is open the owner of `device` has let go of it, so that someone else can take a picture."""
        mine = [o for o in self.owners if same_device(o.device, device)]
        for o in mine:
            o.lent = True
        try:
            end = time.monotonic() + 3.0
            while any(o.busy or o.cap is not None for o in mine) and time.monotonic() < end:  # until it is really closed, not only being closed
                time.sleep(0.01)
            yield
        finally:
            for o in mine:
                o.lent = False

    def close(self):
        with self.cond:
            self.closed = True
            self.cond.notify_all()
        for o in self.owners:  # each owner lets go of its camera before the program ends (ended in the middle of a read, the program aborted on its way out)
            if o.is_alive():
                o.join(timeout=3.0)


class MadeUp:
    """What a camera straight above the stand-in arm's made-up table would see, drawn: the spots, the stand-in's
    blocks and the gripper. `world()` gives {"axes": forward / left / up as columns, "centre": the middle of the view,
    "gripper": xyz, "open": bool, "spots": {name: xyz}, "things": {name: xyz}} in the arm's base frame [mm]."""

    def __init__(self, world, size=(640, 480), fps=15.0, px_per_mm=1.5):
        self.world, self.size, self.fps, self.k, self.c0 = world, tuple(size), float(fps), float(px_per_mm), None

    def read(self):
        import cv2

        time.sleep(1.0 / self.fps)
        w, (W, H) = self.world(), self.size
        if w is None:  # no stand-in session yet
            self.c0 = None
            im = np.full((H, W, 3), (104, 100, 96), np.uint8)
            cv2.putText(im, "made-up picture: start a session on the stand-in arm", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 196, 192), 1, cv2.LINE_AA)
            return True, im
        if self.c0 is None:
            self.c0 = np.asarray(w["gripper"], float).copy()  # before any spot is known the view stays where the gripper started
        A, c = np.asarray(w["axes"], float), np.asarray(self.c0 if w.get("centre") is None else w["centre"], float)

        def px(p):  # forward is up in the picture, the robot's left is left
            d = A.T @ (np.asarray(p, float) - c)
            return int(round(W / 2 - self.k * d[1])), int(round(H / 2 - self.k * d[0]))

        im = np.full((H, W, 3), (104, 100, 96), np.uint8)
        for g in range(-600, 601, 50):
            cv2.line(im, px(c + A @ [g, -900.0, 0.0]), px(c + A @ [g, 900.0, 0.0]), (110, 106, 102), 1)
            cv2.line(im, px(c + A @ [-900.0, g, 0.0]), px(c + A @ [900.0, g, 0.0]), (110, 106, 102), 1)
        for name, p in sorted(w.get("spots", {}).items()):
            cv2.circle(im, px(p), int(21 * self.k), (150, 146, 142), 1, cv2.LINE_AA)
            cv2.putText(im, str(name), (px(p)[0] + int(24 * self.k), px(p)[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 186, 182), 1, cv2.LINE_AA)
        for i, (_, p) in enumerate(sorted(w.get("things", {}).items())):
            x, y, h = *px(p), int(15 * self.k)
            cv2.rectangle(im, (x - h, y - h), (x + h, y + h), ((70, 170, 90), (60, 150, 210), (190, 120, 60), (60, 190, 215))[i % 4], -1, cv2.LINE_AA)
        x, y = px(w["gripper"])
        gap = int((30 if w.get("open", True) else 16) * self.k)
        for side in (-1, 1):
            cv2.rectangle(im, (x + side * gap - 4, y - int(14 * self.k)), (x + side * gap + 4, y + int(14 * self.k)), (40, 38, 36), -1, cv2.LINE_AA)
        cv2.line(im, (x - gap, y), (x + gap, y), (40, 38, 36), 2, cv2.LINE_AA)
        cv2.putText(im, "made-up picture: there is no camera on the stand-in arm", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 196, 192), 1, cv2.LINE_AA)
        return True, im

    def release(self):
        pass
