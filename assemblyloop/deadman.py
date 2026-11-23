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
import uuid

DISCOVER = b"DCOL-DISCOVER/1"  # the lab's headset app looks for a server with this before it sends any frame


def discovery_reply(data, port, name, busy=False):
    """The answer to the headset's search, or None if `data` is not one. The app broadcasts
    b"DCOL-DISCOVER/1 <nonce>" to the port it streams to, lists whoever answers, and sends its frames to the address
    the chosen answer came from (the wire format is in the lab's retriever-data-collection, docs/discovery.md)."""
    if len(data) > 512 or not data.startswith(DISCOVER) or data[len(DISCOVER):len(DISCOVER) + 1] not in (b"", b" ", b"\n", b"\r"):
        return None
    try:
        nonce = data[len(DISCOVER):].strip().decode("ascii")[:64]
    except UnicodeDecodeError:
        return None
    return json.dumps(dict(proto=1, type="dcol-collector", name=name, id=uuid.uuid5(uuid.NAMESPACE_DNS, name).hex, teleop_port=int(port),
                           dashboard_port=0, busy=bool(busy), session_running=True, mode="assembly", nonce=nonce), separators=(",", ":")).encode()


class Deadman:
    def __init__(self, timeout_s=0.25, clock=time.monotonic):
        self.timeout_s, self.clock = float(timeout_s), clock
        self._lock = threading.Lock()
        self._last_any = None  # when a signal of any kind last arrived
        self._last_press = None  # when a counted "pressed" last arrived
        self._armed = False  # a release has been seen since the signal (re)appeared
        self.n = 0
        self._gaps = []  # (when, how long since the signal before): the last few seconds, for the page

    def feed(self, pressed, t=None):
        t = self.clock() if t is None else t
        with self._lock:
            if self._last_any is not None:
                self._gaps = [(a, d) for a, d in self._gaps if t - a <= 5.0] + [(t, t - self._last_any)]
            if self._last_any is None or t - self._last_any > self.timeout_s:
                self._armed, self._last_press = False, None  # the signal was gone: start over
            self._last_any = t
            self.n += 1
            if not pressed:
                self._armed, self._last_press = True, None
            elif self._armed:
                self._last_press = t

    def worst_gap(self):
        """The longest silence between two signals in the last five seconds [s]."""
        with self._lock:
            return max((d for _, d in self._gaps), default=0.0)

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
        self.last_grip, self.sender, self.strangers = None, None, 0  # whose frames count (the first headset heard), and how many frames of others were dropped
        self.name, self.searches = f"{socket.gethostname() or 'lab-pc'} (assembly-loop)", 0  # as listed in the headset; searches answered

    def why(self):
        w = super().why()
        if w != "no signal has arrived yet":
            return w
        if self.searches:
            return f'{w}: the headset is searching, choose "{self.name}" in its server list'
        return f"{w}: open the connect screen of the headset app, on the same network as this computer"

    def on_packet(self, data, t=None, addr=None):
        """Take one datagram. Returns True if it counted as a fresh signal. Only ONE headset is the switch: the first
        one heard (`addr`, its network address) until the session ends. A second sender that reports "held" must not
        keep the arm moving after the person at the stop button has let go. A frame without a running number cannot
        be told from a repeated one and does not count."""
        if addr is not None:
            if self.sender is None:
                self.sender = addr[0]
            elif addr[0] != self.sender:
                self.strangers += 1
                return False
        try:
            f = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            return False
        if not isinstance(f, dict):
            return False
        seq = f.get("seq")
        if not isinstance(seq, (int, float)) or isinstance(seq, bool) or seq != seq:
            return False
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
        self._sock, self._seq, self.sender = sock, None, None
        self._sock.settimeout(0.2)
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self._run:
            try:
                data, addr = self._sock.recvfrom(65535)
            except (socket.timeout, TimeoutError):  # noqa: UP041  two different classes before Python 3.10, and the lab laptop's version is not known
                continue
            except OSError:
                break
            reply = discovery_reply(data, self.port, self.name, busy=not super().why().startswith("no signal"))
            if reply is None:
                self.on_packet(data, addr=addr)
                continue
            self.searches += 1
            try:
                self._sock.sendto(reply, addr)
            except OSError:
                pass

    def stop(self):
        self._run = False
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    @property
    def listening(self):
        return bool(self._run and self._sock is not None)
