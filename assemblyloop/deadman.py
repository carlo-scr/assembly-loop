"""Hold-to-run. The arm may move only while a POSITIVE signal keeps arriving: stopping is the absence of that signal,
so a released grip, a dead headset, a dropped network or a crashed sender all stop the arm the same way.

Rules (all tested in tests/test_core.py):
  - "held" needs a pressed signal younger than `timeout_s`.
  - A press only counts after a RELEASE has been seen: a grip that is already squeezed when the signal appears (or
    reappears after a gap) does nothing until it is let go and pressed again.
  - A frame that repeats or goes back in sequence number is not a fresh signal.
This is a convenience layer for the operator. It does not replace the red emergency-stop button.
"""
from __future__ import annotations

import json
import socket
import threading
import time


class Deadman:
    def __init__(self, timeout_s=0.25, clock=time.monotonic):
        self.timeout_s, self.clock = float(timeout_s), clock
        self._lock = threading.Lock()
        self._last_any = None  # when a signal of any kind last arrived
        self._last_press = None  # when a counted "pressed" last arrived
        self._armed = False  # a release has been seen since the signal (re)appeared
        self.n = 0

    def feed(self, pressed, t=None):
        t = self.clock() if t is None else t
        with self._lock:
            if self._last_any is None or t - self._last_any > self.timeout_s:
                self._armed, self._last_press = False, None  # the signal was gone: start over
            self._last_any = t
            self.n += 1
            if not pressed:
                self._armed, self._last_press = True, None
            elif self._armed:
                self._last_press = t

    def held(self):
        t = self.clock()
        with self._lock:
            return bool(self._armed and self._last_press is not None and t - self._last_press <= self.timeout_s)

    def why(self):
        t = self.clock()
        with self._lock:
            if self._last_any is None:
                return "no signal has arrived yet"
            if t - self._last_any > self.timeout_s:
                return f"no signal for {t - self._last_any:.2f} s"
            if not self._armed:
                return "release the grip once, then press it"
            return "held" if self._last_press is not None and t - self._last_press <= self.timeout_s else "released"


class ManualDeadman(Deadman):
    """Pressed and released by the program itself: the step mode (one key press allows one step) and the tests."""

    def __init__(self, clock=time.monotonic):
        super().__init__(timeout_s=float("inf"), clock=clock)
        self.feed(False)

    def press(self):
        self.feed(True)

    def release(self):
        self.feed(False)


class QuestDeadman(Deadman):
    """The Quest controller's grip as the hold-to-run switch. The headset app sends one JSON frame per tick by UDP
    (port 6666 on this rig): {"seq": n, "left": {"grip": 0..1, ...}, "right": {"grip": 0..1, ...}, ...}. Held = the
    chosen hand's grip is at or above `threshold` in fresh, advancing frames. The lab's dashboard owns that port while
    it runs: close it, or have the headset send to another port, before starting this."""

    def __init__(self, port=6666, host="0.0.0.0", hand="right", threshold=0.5, timeout_s=0.25, clock=time.monotonic):
        super().__init__(timeout_s=timeout_s, clock=clock)
        if hand not in ("right", "left", "either"):
            raise ValueError("hand must be right, left or either")
        self.port, self.host, self.hand, self.threshold = int(port), host, hand, float(threshold)
        self._seq, self._sock, self._thread, self._run = None, None, None, False
        self.last_grip = None

    def on_packet(self, data, t=None):
        """Take one datagram. Returns True if it counted as a fresh signal."""
        try:
            f = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return False
        if not isinstance(f, dict):
            return False
        seq = f.get("seq")
        if isinstance(seq, (int, float)) and not isinstance(seq, bool):
            if self._seq is not None and seq <= self._seq and self._seq - seq < 1000:  # a restart of the app counts down by a lot
                return False
            self._seq = seq
        try:
            grips = [float((f.get(h) or {}).get("grip", 0.0) or 0.0) for h in (("left", "right") if self.hand == "either" else (self.hand,))]
        except (TypeError, ValueError, AttributeError):
            return False
        self.last_grip = max(grips)
        self.feed(self.last_grip >= self.threshold, t)
        return True

    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((self.host, self.port))
        except OSError:
            sock.close()
            raise
        self._sock, self._seq = sock, None
        self._sock.settimeout(0.2)
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self._run:
            try:
                data, _ = self._sock.recvfrom(65535)
            except (socket.timeout, TimeoutError):  # noqa: UP041  two different classes before Python 3.10, and the lab laptop's version is not known
                continue
            except OSError:
                break
            self.on_packet(data)

    def stop(self):
        self._run = False
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    @property
    def listening(self):
        return bool(self._run and self._sock is not None)
