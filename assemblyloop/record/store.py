"""Campaigns and runs on disk, as the lab's recorder keeps them (swoosh_collect/campaign.py).

A CAMPAIGN is a named collection (a folder with campaign.json). A RUN is one episode in it: a folder named by the
time it started, `recording_YYYY_MM_DD_HH_MM_SS`, whose run.json is written at once (status "recording") so that a
crash leaves a run that says it never finished, and is merged into, never overwritten, as the run goes on.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

STATUSES = ("recording", "processing", "ready")


def slugify(name):
    s = re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")
    if not s or len(s) > 60:
        raise ValueError("a campaign's name is 1 to 60 letters, digits, spaces or dashes")
    return s


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def merge_json(path, **fields):
    """Merge fields into a JSON file (a run's run.json): what was there stays unless named here."""
    path = Path(path)
    cur = read_json(path, {}) or {}
    cur.update(fields)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur, indent=2, default=_plain))
    tmp.replace(path)  # never a half-written run.json
    return cur


def _plain(v):
    try:
        import numpy as np

        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
    except ImportError:
        pass
    return str(v)


class Run:
    def __init__(self, path):
        self.path = Path(path)

    @property
    def meta(self):
        return read_json(self.path / "run.json", {}) or {}

    @property
    def name(self):
        return self.path.name

    @property
    def status(self):
        return self.meta.get("status", "unknown")

    @property
    def validation(self):
        return self.meta.get("validation") or {}

    def __repr__(self):
        return f"Run({self.name}, {self.status})"


class Campaign:
    def __init__(self, path):
        self.path = Path(path)

    @classmethod
    def create(cls, root, name, notes="", repo=None):
        path = Path(root) / slugify(name)
        if (path / "campaign.json").exists():
            raise ValueError(f"there is a campaign {path.name!r} already")
        path.mkdir(parents=True, exist_ok=True)
        (path / "campaign.json").write_text(json.dumps(dict(name=str(name).strip(), slug=path.name, created=time.strftime("%Y-%m-%d %H:%M:%S"), created_unix=time.time(),
                                                            notes=str(notes), program="assembly-loop", layout="swoosh-collect campaigns (recording_<ts>/raw, video)", repo=repo), indent=2))
        return cls(path)

    @property
    def repo(self):
        """The Hugging Face dataset this campaign is uploaded to (one dataset holds one campaign)."""
        return self.meta.get("repo")

    def set_repo(self, repo):
        merge_json(self.path / "campaign.json", repo=str(repo))

    @classmethod
    def open(cls, root, name):
        path = Path(root) / slugify(name)
        if not (path / "campaign.json").is_file():
            raise ValueError(f"there is no campaign {path.name!r} in {Path(root)}")
        return cls(path)

    @classmethod
    def all(cls, root):
        root = Path(root)
        if not root.is_dir():
            return []
        return sorted((cls(p) for p in root.iterdir() if (p / "campaign.json").is_file()), key=lambda c: c.meta.get("created_unix", 0.0))

    @property
    def meta(self):
        return read_json(self.path / "campaign.json", {}) or {}

    @property
    def name(self):
        return self.meta.get("name", self.path.name)

    def runs(self):
        return [Run(p) for p in sorted(self.path.glob("recording_*")) if p.is_dir()]

    def new_run(self, **meta):
        """A new run folder, its run.json written NOW with status "recording" (so a crash is visible as such)."""
        stamp = datetime.now().strftime("recording_%Y_%m_%d_%H_%M_%S")
        path, n = self.path / stamp, 1
        while path.exists():  # two episodes in one second: the second gets a suffix
            n += 1
            path = self.path / f"{stamp}_{n}"
        (path / "raw").mkdir(parents=True)
        (path / "video").mkdir()
        merge_json(path / "run.json", status="recording", started=time.strftime("%Y-%m-%d %H:%M:%S"), wall_start_unix=time.time(), **meta)
        return Run(path)

    def summary(self):
        runs = self.runs()
        ok = [r for r in runs if r.status == "ready" and r.validation.get("ok")]
        good = [r for r in ok if (r.meta.get("outcome") or {}).get("success")]
        return dict(name=self.name, slug=self.path.name, repo=self.repo, runs=len(runs), ready=len(ok), succeeded=len(good), failed_validation=sum(1 for r in runs if r.status == "ready" and not r.validation.get("ok")),
                    unfinished=sum(1 for r in runs if r.status != "ready"), upload=read_json(self.path / "upload.json"))
