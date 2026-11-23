"""Checks of one run, as the lab's swoosh-validate does them; the verdict goes into run.json so that the exporter
can refuse a bad run (it does, by default). Each check is ok, warn or fail; a run passes if nothing fails.

  finished        the run was closed (status not "recording") and has an outcome
  streams         arm_state, commanded and events are there and not empty
  arm rate        the arm was read about 30 times a second: median interval, longest gap
  measured joints the measured joints are in (almost) every row
  video <key>     for every camera: the file and its times are there, as many frames as times, as many frames in the
                  mp4 as were written, the camera did not stop early, a sane rate, no gap that is not declared (a
                  camera lent to the block finder leaves a declared gap), and it covers the episode
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .store import merge_json, read_json


def read_jsonl(path):
    out = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        out.append(dict(_bad=line[:80]))
    except OSError:
        pass
    return out


def count_frames(path):
    """Frames really in an mp4 (decoded, as the lab does: a dropped frame shifts every later time)."""
    try:
        import imageio.v2 as imageio

        n = 0
        with imageio.get_reader(str(path)) as rd:
            for _ in rd:
                n += 1
        return n
    except Exception:  # noqa: BLE001  no reader, or a broken file: counted as none
        return -1


def _check(name, level, detail="", **extra):
    return dict(check=name, level=level, detail=detail, **extra)


def validate_run(run_dir, count_mp4=True):
    run_dir = Path(run_dir)
    meta = read_json(run_dir / "run.json", {}) or {}
    checks = []
    status, outcome = meta.get("status"), meta.get("outcome")
    checks.append(_check("finished", "ok" if status in ("processing", "ready") and outcome else "fail",
                         "" if status in ("processing", "ready") and outcome else f"status {status!r}, outcome {'there' if outcome else 'missing'}: the episode never closed"))
    rows = {n: read_jsonl(run_dir / "raw" / f"{n}.jsonl") for n in ("arm_state", "commanded", "events")}
    empty = [n for n, r in rows.items() if not r]
    checks.append(_check("streams", "fail" if "arm_state" in empty or "events" in empty else ("warn" if empty else "ok"), ", ".join(f"{n} is empty" for n in empty)))
    arm = [r for r in rows["arm_state"] if "t" in r]
    motion = float((meta.get("labels") or {}).get("motion_s") or (arm[-1]["t"] if arm else 0.0))
    if len(arm) >= 2:
        t = np.array([r["t"] for r in arm], float)
        dt = np.diff(t)
        med, gap = float(np.median(dt)), float(dt.max())
        lvl = "fail" if med > 1 / 15 or gap > 1.0 else ("warn" if med > 1 / 25 or gap > 0.25 else "ok")
        checks.append(_check("arm rate", lvl, f"median {1 / med if med > 0 else 0:.1f} per second, longest gap {gap * 1000:.0f} ms", median_hz=round(1 / med, 2) if med > 0 else 0.0, max_gap_s=round(gap, 3)))
        have = sum(1 for r in arm if r.get("joints_measured_deg") is not None) / len(arm)
        errs = sum(1 for r in arm if r.get("read_error"))
        checks.append(_check("measured joints", "ok" if have >= 0.9 else "warn", f"in {100 * have:.0f}% of the rows" + (f"; {errs} reads failed" if errs else "")))
    else:
        checks.append(_check("arm rate", "fail", "fewer than two rows"))
    for key in meta.get("cameras") or []:
        checks.append(_video(run_dir, key, motion, count_mp4))
    ok = not any(c["level"] == "fail" for c in checks)
    verdict = dict(ok=ok, checks=checks)
    merge_json(run_dir / "run.json", status="ready", validation=verdict)
    return verdict


def _video(run_dir, key, motion_s, count_mp4):
    name = f"video {key}"
    mp4, times = run_dir / "video" / f"{key}.mp4", run_dir / "video" / f"{key}_frame_times.json"
    if not mp4.is_file() or not times.is_file():
        return _check(name, "fail", f"{'the mp4' if not mp4.is_file() else 'the frame times'} missing")
    ft = read_json(times, {}) or {}
    t = np.asarray(ft.get("t") or [], float)
    problems, warns = [], []
    if ft.get("error"):
        problems.append(f"error: {ft['error']}")
    if ft.get("stopped_early"):
        problems.append(f"stopped early at {ft.get('stopped_at_s')} s")
    if len(t) != int(ft.get("frames", -1)):
        problems.append(f"{ft.get('frames')} frames but {len(t)} times")
    if count_mp4:
        n = count_frames(mp4)
        if n != len(t):
            problems.append(f"{n} frames in the mp4 but {len(t)} times")
    if len(t) < 2:
        problems.append("fewer than two frames")
    else:
        dt = np.diff(t)
        fps = (len(t) - 1) / (t[-1] - t[0]) if t[-1] > t[0] else 0.0
        if fps < 10:
            problems.append(f"{fps:.1f} frames a second")
        elif fps < 24:
            warns.append(f"{fps:.1f} frames a second")
        declared = [(a, b if b is not None else 1e9) for a, b in (ft.get("lent_s") or [])]
        big = [(float(t[i]), float(dt[i])) for i in np.nonzero(dt > 0.25)[0] if not any(a - 0.2 <= t[i] <= b + 0.2 for a, b in declared)]
        if any(d > 1.0 for _, d in big):
            problems.append(f"{len(big)} undeclared gap(s), longest {max(d for _, d in big):.2f} s")
        elif big:
            warns.append(f"{len(big)} undeclared gap(s) over 0.25 s")
        if t[0] > 1.0 or (motion_s and t[-1] < motion_s - 1.0):
            warns.append(f"covers {t[0]:.1f} to {t[-1]:.1f} s of a {motion_s:.1f} s episode")
    level = "fail" if problems else ("warn" if warns else "ok")
    return _check(name, level, "; ".join(problems + warns), frames=len(t), timestamp_source=ft.get("timestamp_source"))


def validate_campaign(campaign, again=False, stale_s=600.0):
    """Every run that has closed but not been checked (or all, with `again`). Returns {run name: verdict}. A run still
    "recording" whose files have not changed for `stale_s` was left by a program that ended: it is checked (and
    fails "finished"), so that it shows as such; one that may still be recording is left alone."""
    import time

    out = {}
    for run in campaign.runs():
        if run.status == "recording":
            newest = max((f.stat().st_mtime for f in run.path.rglob("*") if f.is_file()), default=0.0)
            if time.time() - newest < stale_s:
                continue
        if run.status == "processing" or again or not run.validation:
            out[run.name] = validate_run(run.path)
    return out
