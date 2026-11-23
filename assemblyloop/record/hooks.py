"""What a session does to record: mixed into `Session`. The job loop calls the `_rec_*` hooks; each is a no-op unless
recording is on, so a session that does not record behaves exactly as before."""
from __future__ import annotations

from pathlib import Path

from ..guard import Refused
from .recorder import Recorder
from .store import Campaign


class Recording:
    def _rec_init(self, campaigns_root="campaigns", record_kw=None):
        self.campaigns_root, self.rec, self.record_kw = Path(campaigns_root), None, dict(record_kw or {})

    # ------------------------------------------------------------------ the page's buttons
    def record_campaigns(self, max_age_s=5.0):
        """Every campaign with its counts (read from the runs' files, so kept for a few seconds)."""
        import time

        cached = getattr(self, "_rec_campaigns", None)
        if cached is None or time.monotonic() - cached[0] > max_age_s:
            cached = (time.monotonic(), [c.summary() for c in Campaign.all(self.campaigns_root)])
            self._rec_campaigns = cached
        return cached[1]

    def record_new_campaign(self, name, notes=""):
        try:
            c = Campaign.create(self.campaigns_root, name, notes)
        except ValueError as e:
            raise Refused(str(e)) from None
        self.say(f"campaign {c.name!r} made in {c.path}")
        self._rec_campaigns = None
        return c.summary()

    def record_on(self, campaign):
        with self.lock:
            if not self.connected:
                raise Refused("start a session first: a recording belongs to the arm of a session")
            if self.worker is not None:
                raise Refused("switch recording on or off when the arm has finished")
            try:
                c = Campaign.open(self.campaigns_root, campaign)
            except ValueError as e:
                raise Refused(str(e)) from None
            if self.rec is not None:
                self.rec.close("switched to another campaign")
            self.rec = Recorder(self, c, **self.record_kw)
            self.say(f"recording into {c.name!r}: every block taken and put down is one episode ({c.path})")

    def record_off(self, why="recording switched off"):
        with self.lock:
            if self.worker is not None:
                raise Refused("switch recording off when the arm has finished")
            rec, self.rec = self.rec, None
        if rec is not None:
            rec.close(why)
            self.say(f"recording off: {rec.counts['episodes']} episodes in {rec.campaign.name!r}")

    def record_state(self):
        rec = self.rec
        out = dict(on=False, root=str(self.campaigns_root.resolve()), campaigns=self.record_campaigns())
        if rec is not None:
            out.update(rec.status())
        return out

    def _rec_close(self, why):
        """The session ends: an open episode is finished as stopped, the cameras let go."""
        rec, self.rec = self.rec, None
        if rec is not None:
            rec.close(why)

    # ------------------------------------------------------------------ the job loop's hooks
    def _rec_begin(self, job, k, leg, text):
        if self.rec is not None and not job.get("rehearse"):
            self.rec.begin(job, k, leg, text)

    def _rec_event(self, kind, **kw):
        if self.rec is not None:
            self.rec.event(kind, **kw)

    def _rec_motion_done(self, rec=None):
        if self.rec is not None:
            self.rec.motion_done(rec)

    def _rec_finish(self, outcome="done"):
        if self.rec is not None:
            self.rec.finish(outcome)
