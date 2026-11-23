"""A campaign's export -> a Hugging Face dataset, once or watching: the step after every session, made automatic.

    python -m assemblyloop upload --campaign "blocks pickplace v1" --repo <org>/<name>            once
    python -m assemblyloop upload --campaign "blocks pickplace v1" --repo <org>/<name> --watch    every few minutes

Each round: runs that have closed are checked (validate), the ready and passed ones are added to the export, and the
files that are new or changed since the last upload are sent as ONE commit. The tag `v2.1` (LeRobot's codebase
version, which its loader looks for) is moved to that commit. What was sent is kept in `<campaign>/upload.json`, so a
round that was cut off is simply done again; the dataset on the hub only ever gets whole episodes.

The uploader is its own process: it never slows the program that moves the arm, and a network that is gone only
delays the upload. Log in once on the machine (`huggingface-cli login`, or HF_TOKEN in the environment); no token is
ever written into a file here. Datasets are made PRIVATE unless --public is given: the cameras may show people.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .export import export_campaign
from .store import read_json
from .validate import validate_campaign

TAG = "v2.1"


def manifest(folder):
    """Every file of the export with its size and time, by its path in the dataset."""
    folder = Path(folder)
    return {p.relative_to(folder).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(folder.rglob("*"))
            if p.is_file() and ".cache" not in p.parts and not p.name.startswith(".")}


def push(campaign, repo_id, private=True, api=None, say=print):
    """Send what is new in `<campaign>/lerobot/` to `repo_id` (a dataset). Returns the number of files sent."""
    out = campaign.path / "lerobot"
    state_path = campaign.path / "upload.json"
    state = read_json(state_path, {}) or {}
    if state.get("repo") not in (None, repo_id):
        state = {}  # another repo: everything is new there
    sent = state.get("files") or {}
    now = manifest(out)
    changed = sorted(k for k, v in now.items() if sent.get(k) != v)
    if not changed:
        return 0
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    eps = sum(1 for k in changed if k.endswith(".parquet"))
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(out), allow_patterns=changed,
                      commit_message=f"assembly-loop: {eps} new episode(s), {len(changed)} file(s)")
    try:  # LeRobot's loader looks for the codebase version as a tag: moved to the commit just made
        api.delete_tag(repo_id, tag=TAG, repo_type="dataset")
    except Exception:  # noqa: BLE001  there was none yet
        pass
    api.create_tag(repo_id, tag=TAG, repo_type="dataset")
    state.update(repo=repo_id, files=now, last_unix=time.time(), last=time.strftime("%Y-%m-%d %H:%M:%S"), last_files=len(changed), last_episodes=eps, error=None)
    state_path.write_text(json.dumps(state, indent=1))
    say(f"[upload] {eps} episode(s), {len(changed)} file(s) -> https://huggingface.co/datasets/{repo_id}")
    return len(changed)


def round_once(campaign, repo_id=None, private=True, api=None, fps=30.0, say=print):
    """Check, export, upload. Without `repo_id` only check and export."""
    checked = validate_campaign(campaign)
    bad = [n for n, v in checked.items() if not v["ok"]]
    if checked:
        say(f"[check] {len(checked)} run(s): {len(checked) - len(bad)} passed" + (f", failed: {', '.join(bad)}" if bad else ""))
    added, refused = export_campaign(campaign, fps=fps, say=say)
    if added:
        say(f"[export] {len(added)} episode(s) added to {campaign.path / 'lerobot'}")
    n = push(campaign, repo_id, private, api, say) if repo_id else 0
    return dict(checked=len(checked), failed=bad, added=len(added), refused=len(refused), sent=n)


def watch(campaign, repo_id, every_s=300.0, private=True, api=None, say=print, rounds=None):
    """`round_once` every `every_s` seconds, for ever (or `rounds` times). A failing round is written down and tried
    again next time: the network at the rig comes and goes."""
    k = 0
    while rounds is None or k < rounds:
        k += 1
        try:
            r = round_once(campaign, repo_id, private, api, say=say)
            _status(campaign, ok=True, **r)
        except Exception as e:  # noqa: BLE001  said, kept, tried again
            say(f"[upload] round failed: {type(e).__name__}: {e} (tried again in {every_s:.0f} s)")
            _status(campaign, ok=False, error=f"{type(e).__name__}: {e}")
        if rounds is not None and k >= rounds:
            break
        time.sleep(every_s)


def _status(campaign, **kw):
    p = campaign.path / "upload.json"
    cur = read_json(p, {}) or {}
    cur.update(round=dict(when=time.strftime("%Y-%m-%d %H:%M:%S"), **kw))
    if kw.get("error"):
        cur["error"] = kw["error"]
    p.write_text(json.dumps(cur, indent=1))
