"""Recorded episodes (`record.py`) turned into what a policy is trained on. Nothing here talks to the arm.

One conversion, `convert`, and two ways to write its result: `export_npz` (numpy only: for looking at the numbers and
for tests) and `export_lerobot` (a LeRobot dataset, through the `lerobot` library, which is imported only there).

What a training example is, per row that is kept:

    observation.state     per arm: its 7 joint angles [rad], then how far the gripper is open, 0 (shut) .. 1 (open,
                          counts / 850). Arms side by side, right before left: 8 numbers for one arm, 16 for two.
    action                the same quantities ONE TICK LATER: where the joints are next, as an absolute target (what
                          ALOHA-style datasets hold). The gripper's entry is what the gripper was ASKED for at this
                          tick (`goal_grip`) when the row has it, otherwise where it is one tick later.
    observation.ee_pose   per arm: the flange's position [m], then roll / pitch / yaw [rad], unwrapped along the
                          episode so that an angle near 180 degrees does not jump to -180. Not what the policy is
                          trained on by default: there to try other ways of saying where the arm is.
    observation.images.<camera>   the newest picture that camera had at that tick.
    task                  the episode's instruction.

Rows are put on an even grid of `fps` (the rate it was recorded at unless told otherwise; never faster): a grid point
takes the row nearest to it, so a tick the recorder dropped repeats its neighbour. Then, unless `trim=False`, the
rows in which nothing happens are cut: all but `still_s` seconds before the first motion and after the last, and
all but `pause_s` seconds of every stretch in the middle during which the hold-to-run switch was released and nothing
moved. "Moves" means the controller says so, or a joint or the gripper is somewhere else than a tick before or after.
The action of a row is worked out BEFORE rows are cut, so it is always what truly came next.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .record import GRIP_OPEN, Episode, arm_ok, episodes, row_times

JOINT_EPS_DEG, GRIP_EPS = 0.05, 3  # less than this from one tick to the next counts as standing still
EE = "observation.ee_pose"


@dataclass
class Clip:
    """One episode as arrays, row for row: N rows, A arms."""

    task: str
    fps: float
    sides: list
    cams: list
    state: np.ndarray  # (N, 8 A) float32
    action: np.ndarray  # (N, 8 A) float32
    ee: np.ndarray  # (N, 6 A) float32
    t: np.ndarray  # (N,) the wall clock of the row each one was taken from
    row: np.ndarray  # (N,) that row's number "i" in rows.jsonl
    pictures: dict  # camera -> N paths (None where that camera had no picture at all)
    names: list  # what each entry of state / action is
    ee_names: list
    dropped: dict = field(default_factory=dict)  # how many rows were left out, by reason
    notes: list = field(default_factory=list)  # what a person should know about this episode's data
    dir: Path | None = None

    def __len__(self):
        return int(len(self.state))


def state_names(sides):
    return [f"{s}_{n}" for s in sides for n in [f"joint_{k}" for k in range(1, 8)] + ["gripper"]]


def ee_names(sides):
    return [f"{s}_{n}" for s in sides for n in ("x", "y", "z", "roll", "pitch", "yaw")]


def _filled(a):
    """Gaps (nan) take the value before them; gaps at the very start the first value there is. All gaps stay nan."""
    a = np.array(a, float)
    for col in a.T if a.ndim == 2 else [a]:
        ok = np.flatnonzero(np.isfinite(col))
        if len(ok):
            col[:] = col[ok[np.clip(np.searchsorted(ok, np.arange(len(col)), side="right") - 1, 0, None)]]
    return a


def keep_mask(active, held, fps, still_s=1.0, pause_s=1.0):
    """Which rows stay, and how many go for which reason. `active`: something moves in that row; `held`: the
    hold-to-run switch is held (rows that do not say count as held)."""
    active, held = np.asarray(active, bool), np.asarray(held, bool)
    n, keep = len(active), np.ones(len(active), bool)
    dropped = dict(start=0, end=0, pauses=0)
    if not n:
        return keep, dropped
    on = np.flatnonzero(active)
    if not len(on):  # nothing ever moves: there is nothing to learn from
        return np.zeros(n, bool), dict(dropped, start=n)
    lead, pause = int(round(still_s * fps)), int(round(pause_s * fps))
    first, last = max(0, on[0] - lead), min(n - 1, on[-1] + lead)
    keep[:first], keep[last + 1:] = False, False
    dropped["start"], dropped["end"] = int(first), int(n - 1 - last)
    idle = ~active & ~held
    k = first
    while k <= last:
        if not idle[k]:
            k += 1
            continue
        end = k
        while end <= last and idle[end]:
            end += 1
        if end - k > pause:  # its first and last moments stay: the arm coming to rest and setting off again
            cut = slice(k + (pause + 1) // 2, end - pause // 2)
            dropped["pauses"] += int(keep[cut].sum())
            keep[cut] = False
        k = end
    return keep, dropped


def convert(ep, fps=None, trim=True, still_s=1.0, pause_s=1.0, cams=None, sides=None):
    """An episode (a folder or a `record.Episode`) as a `Clip`. `cams` / `sides`: only these cameras / arms, in this
    order (default: all there are; arms right before left). Raises ValueError for a `fps` above the recorded rate."""
    ep = ep if isinstance(ep, Episode) else Episode(ep)
    sides = list(sides) if sides else ep.sides()
    cams = list(cams) if cams is not None else ep.cams
    notes, A = [], len(sides)
    good = [r for r in ep.rows if sides and all(arm_ok((r.get("arms") or {}).get(s)) for s in sides)]
    dropped = dict(broken=len(ep.rows) - len(good), repeated=0, start=0, end=0, pauses=0)
    fps = float(fps or ep.hz)
    if fps > ep.hz * 1.01:
        raise ValueError(f"{ep.id} was recorded at {ep.hz:g} rows a second: it cannot be exported at {fps:g}")
    if not ep.finished:
        notes.append("the episode has no summary: it was cut short")
    if ep.torn:
        notes.append(f"{ep.torn} lines could not be read")
    if not good:
        return Clip(ep.task, fps, sides, cams, np.zeros((0, 8 * A), np.float32), np.zeros((0, 8 * A), np.float32), np.zeros((0, 6 * A), np.float32), np.zeros(0), np.zeros(0, int),
                    {c: [] for c in cams}, state_names(sides), ee_names(sides), dropped, notes + ["no row has every arm's joints and pose"], ep.dir)

    arm = lambda r, s: r["arms"][s]  # noqa: E731
    num = lambda v: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else np.nan  # noqa: E731
    joints = np.array([[arm(r, s)["joints"] for s in sides] for r in good], float)  # (R, A, 7) deg
    xyz = np.array([[arm(r, s)["xyz"] for s in sides] for r in good], float)  # mm
    rpy = np.unwrap(np.radians(np.array([[arm(r, s)["rpy"] for s in sides] for r in good], float)), axis=0)
    grip = np.array([[num(arm(r, s).get("grip")) for s in sides] for r in good])  # counts, nan where the row has none
    goal = np.array([[num(arm(r, s).get("goal_grip")) for s in sides] for r in good])
    moving = np.array([any(bool(arm(r, s).get("moving")) for s in sides) for r in good])
    held = np.array([r.get("held") is not False for r in good])
    if np.isnan(grip).any():
        grip = _filled(grip)
        for a, s in enumerate(sides):
            if np.isnan(grip[:, a]).any():  # never measured: the best there is is what it was asked for, else open
                grip[:, a] = np.where(np.isfinite(_filled(goal[:, a])), _filled(goal[:, a]), GRIP_OPEN)
                notes.append(f"the {s} gripper's opening was never measured: what it was asked for stands in")

    # an even grid: each point takes the row nearest to it
    tm = row_times(good)
    grid = np.arange(int(np.floor(tm[-1] * fps + 1e-6)) + 1) / fps
    hi = np.clip(np.searchsorted(tm, grid), 1, len(tm) - 1) if len(tm) > 1 else np.zeros(len(grid), int)
    idx = np.where(np.abs(tm[hi] - grid) < np.abs(tm[hi - 1] - grid), hi, hi - 1) if len(tm) > 1 else hi
    dropped["repeated"] = int((np.diff(idx) == 0).sum())
    joints, xyz, rpy, grip, goal, moving, held = joints[idx], xyz[idx], rpy[idx], grip[idx], goal[idx], moving[idx], held[idx]

    opening = np.clip(grip / GRIP_OPEN, 0.0, 1.0)
    state = np.concatenate([np.radians(joints), opening[:, :, None]], axis=2)  # (N, A, 8)
    action = np.concatenate([state[1:], state[-1:]])  # one tick later; the last row has no later: it stays where it is
    action[:, :, 7] = np.where(np.isfinite(goal), np.clip(goal / GRIP_OPEN, 0.0, 1.0), action[:, :, 7])
    ee = np.concatenate([xyz / 1000.0, rpy], axis=2)

    step = (np.abs(np.diff(joints, axis=0)).max(axis=(1, 2)) > JOINT_EPS_DEG) | (np.abs(np.diff(grip, axis=0)).max(axis=1) > GRIP_EPS) if len(idx) > 1 else np.zeros(0, bool)
    active = moving | np.concatenate([step, [False]]) | np.concatenate([[False], step])
    keep = np.ones(len(idx), bool)
    if trim:
        keep, cut = keep_mask(active, held, fps, still_s, pause_s)
        dropped.update(cut)
        if not keep.any():
            notes.append("nothing moves in this episode")

    pictures = {}
    for cam in cams:
        paths = [ep.picture_path(good[k], cam) for k in idx[keep]]
        if any(p is None for p in paths) and any(p is not None for p in paths):  # the camera woke up after the first rows: they get its first picture
            have = [k for k, p in enumerate(paths) if p is not None]
            late = sum(p is None for p in paths)
            paths = [paths[have[max(0, int(np.searchsorted(have, k, side="right")) - 1)]] for k in range(len(paths))]
            notes.append(f"camera {cam} had no picture yet in {late} rows: they point at its first")
        pictures[cam] = paths
    shape = lambda a: a[keep].reshape(int(keep.sum()), -1).astype(np.float32)  # noqa: E731
    return Clip(ep.task, fps, sides, cams, shape(state), shape(action), shape(ee), np.array([float(good[k].get("t") or 0.0) for k in idx[keep]]),
                np.array([int(good[k].get("i", k)) for k in idx[keep]], int), pictures, state_names(sides), ee_names(sides), dropped, notes, ep.dir)


def summary(clip):
    cut = {k: v for k, v in clip.dropped.items() if k != "repeated"}
    return dict(episode=None if clip.dir is None else clip.dir.name, task=clip.task, fps=clip.fps, rows=len(clip), dropped=sum(cut.values()), dropped_by=dict(clip.dropped), sides=list(clip.sides),
                cams=list(clip.cams), notes=list(clip.notes))


def export_npz(episode_dir, out_path, fps=None, trim=True, still_s=1.0, pause_s=1.0, cams=None, sides=None):
    """One episode as one .npz, with numpy alone: state, action, ee, t, row (the row in rows.jsonl each line came
    from), names, ee_names, sides, cams, fps, task, episode_dir, and per camera `pictures.<camera>`: the files the
    rows point at, relative to the episode's folder ("" where there is none). Returns what was kept and dropped."""
    clip = convert(episode_dir, fps=fps, trim=trim, still_s=still_s, pause_s=pause_s, cams=cams, sides=sides)
    rel = lambda p: "" if p is None else str(Path(p).relative_to(clip.dir))  # noqa: E731
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, state=clip.state, action=clip.action, ee=clip.ee, t=clip.t, row=clip.row, names=np.array(clip.names), ee_names=np.array(clip.ee_names),
                        sides=np.array(clip.sides, dtype=str), cams=np.array(clip.cams, dtype=str), fps=np.float64(clip.fps), task=np.array(clip.task), episode_dir=np.array(str(clip.dir)),
                        **{f"pictures.{cam}": np.array([rel(p) for p in paths], dtype=str) for cam, paths in clip.pictures.items()})
    return dict(summary(clip), out=str(out_path))


# ---------------------------------------------------------------------- LeRobot
def decode_jpeg(jpg):
    """JPEG bytes as an (H, W, 3) picture, red first (OpenCV gives blue first). Needs OpenCV."""
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError('reading the recorded pictures needs OpenCV: pip install -e ".[eyes]"') from e
    im = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError("not a picture")
    return np.ascontiguousarray(im[:, :, ::-1])


def lerobot_dataset_class():
    """`LeRobotDataset`, from wherever this version of the library keeps it."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # 0.3.4 and later (dataset format v3.0)
    except ImportError as new:
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # before: format v2.x, the commit openpi pins
        except ImportError as e:
            raise RuntimeError("exporting to LeRobot needs the `lerobot` library with its dataset part, which cannot be imported here "
                               f"({new}). Install it with: pip install 'lerobot[dataset]' (it brings torch). "
                               "Without it, `export_npz` writes the same numbers with numpy alone.") from e
    return LeRobotDataset


def features_for(clip, sizes, use_videos=False):
    """The feature table `LeRobotDataset.create` wants. `sizes`: camera -> (height, width)."""
    out = {"observation.state": dict(dtype="float32", shape=(len(clip.names),), names=list(clip.names)),
           "action": dict(dtype="float32", shape=(len(clip.names),), names=list(clip.names)),
           EE: dict(dtype="float32", shape=(len(clip.ee_names),), names=list(clip.ee_names))}
    for cam in clip.cams:
        out[f"observation.images.{cam}"] = dict(dtype="video" if use_videos else "image", shape=(int(sizes[cam][0]), int(sizes[cam][1]), 3), names=["height", "width", "channels"])
    return out


def export_lerobot(src, out_dir, repo_id, fps=None, only_success=True, trim=True, still_s=1.0, pause_s=1.0, cams=None, sides=None, unfinished=False, use_videos=False,
                   image_size=None, robot_type=None, min_rows=2, say=print, decode=decode_jpeg, dataset_cls=None):
    """Recorded episodes as one LeRobot dataset in `out_dir` (which must not exist yet: the library makes it).

    `src`: what the Recorder was given as root, or one episode's folder, or a list of episode folders. `repo_id`: the
    dataset's name, "someone/assembly_blocks" (nothing is uploaded). `fps`: default the first episode's own rate.
    `only_success`: leave out episodes whose summary says the task failed (`success` false); episodes nobody judged
    stay. `unfinished`: also take episodes that were cut short by a crash. `cams` / `sides`: default those of the
    first episode that is taken; an episode that lacks one of them is left out and said. `use_videos`: store the
    pictures as video files instead of pictures (smaller; openpi's own examples store pictures). `image_size`:
    (width, height) to bring every picture to, default the size they have. `dataset_cls`: something that stands in
    for `lerobot`'s LeRobotDataset (tests).

    Returns {"out", "repo_id", "fps", "episodes", "frames", "dropped", "skipped": [(episode, why)], "clips": [...]}.
    """
    say = say or (lambda _m: None)
    dirs = episodes(src) if isinstance(src, (str, Path)) else [Path(p) for p in src]
    out_dir = Path(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(f"{out_dir} is already there: a LeRobot dataset is written to a folder of its own")
    cls = dataset_cls or lerobot_dataset_class()
    clips, skipped = [], []
    for d in dirs:
        ep = Episode(d)
        why = ""
        if not ep.finished and not unfinished:
            why = "it has no summary (cut short by a crash)"
        elif only_success and ep.success is False:
            why = "its summary says the task failed"
        elif not ep.task:
            why = "it has no task"
        if not why:
            try:
                clip = convert(ep, fps=fps, trim=trim, still_s=still_s, pause_s=pause_s, cams=cams, sides=sides)
            except ValueError as e:
                why = str(e)
        if not why:
            if len(clip) < min_rows:
                why = "no rows are left" if not clip.notes else f"no rows are left ({clip.notes[-1]})"
            elif clips and (clip.sides != clips[0].sides or clip.cams != clips[0].cams):
                why = f"its arms and cameras ({', '.join(clip.sides + clip.cams)}) are not those of the first episode"
            elif [c for c in clip.cams if clip.pictures[c][0] is None]:
                why = "it has no picture from camera " + ", ".join(c for c in clip.cams if clip.pictures[c][0] is None)
        if why:
            skipped.append((d.name, why))
            say(f"left out {d.name}: {why}")
            continue
        if not clips:  # the first episode taken says what the dataset holds
            fps, cams, sides = clip.fps, clip.cams, clip.sides
        clips.append(clip)
    if not clips:
        raise ValueError("no episode to export" + "".join(f"\n  {name}: {why}" for name, why in skipped))

    first = clips[0]
    sizes = {}
    for cam in first.cams:
        h, w = decode(Path(first.pictures[cam][0]).read_bytes()).shape[:2]
        sizes[cam] = (int(image_size[1]), int(image_size[0])) if image_size else (h, w)
    if out_dir.exists():
        out_dir.rmdir()  # empty (checked above); the library refuses a folder that is there
    asked = dict(repo_id=repo_id, fps=int(round(first.fps)) if abs(first.fps - round(first.fps)) < 1e-9 else first.fps, root=out_dir, features=features_for(first, sizes, use_videos),
                 robot_type=robot_type or ("xarm7" if len(first.sides) == 1 else "xarm7_bimanual"), use_videos=bool(use_videos), image_writer_threads=4)
    takes = inspect.signature(cls.create).parameters
    ds = cls.create(**{k: v for k, v in asked.items() if k in takes or any(p.kind is p.VAR_KEYWORD for p in takes.values())})

    def picture(path, size, cache):
        if cache.get("path") != path:  # cameras are slower than the rows now and then: the same file twice in a row
            im = decode(Path(path).read_bytes())
            if im.shape[:2] != size:
                import cv2

                im = cv2.resize(im, (size[1], size[0]), interpolation=cv2.INTER_AREA)
            cache["path"], cache["im"] = path, im
        return cache["im"]

    frames = 0
    for clip in clips:
        caches = {cam: {} for cam in clip.cams}
        for k in range(len(clip)):
            frame = {"observation.state": clip.state[k], "action": clip.action[k], EE: clip.ee[k], "task": clip.task}
            for cam in clip.cams:
                frame[f"observation.images.{cam}"] = picture(clip.pictures[cam][k], sizes[cam], caches[cam])
            ds.add_frame(frame)
        ds.save_episode()
        frames += len(clip)
        cut = sum(v for k, v in clip.dropped.items() if k != "repeated")
        say(f"{clip.dir.name}: {len(clip)} rows kept, {cut} dropped (" + ", ".join(f"{k} {v}" for k, v in clip.dropped.items() if v) + ")" + "".join(f"; {n}" for n in clip.notes))
    for last in ("finalize", "consolidate"):  # v3.0 must be finalized or its files cannot be read; v2.0 was consolidated; v2.1 needs neither
        if callable(getattr(ds, last, None)):
            getattr(ds, last)()
            break
    out = dict(out=str(out_dir), repo_id=repo_id, fps=first.fps, episodes=len(clips), frames=frames, dropped=sum(summary(c)["dropped"] for c in clips), skipped=skipped,
               sides=first.sides, cams=first.cams, clips=[summary(c) for c in clips])
    say(f"{out['episodes']} episodes, {frames} rows written to {out_dir} ({out['dropped']} rows dropped, {len(skipped)} episodes left out)")
    return out


def main(argv=None):
    """python -m assemblyloop.export assembly_logs out/blocks someone/assembly_blocks"""
    import argparse

    ap = argparse.ArgumentParser(prog="assemblyloop.export", description="Recorded episodes as a LeRobot dataset (or, with --npz, one .npz per episode, which needs numpy only)")
    ap.add_argument("src", help="the folder that was recorded into, or one episode's folder")
    ap.add_argument("out", help="the dataset's folder (must not exist yet); with --npz, a folder for the .npz files")
    ap.add_argument("repo_id", nargs="?", default="local/assembly_loop", help="the dataset's name, as in someone/assembly_blocks")
    ap.add_argument("--fps", type=float, default=None, help="rows a second (default: as recorded)")
    ap.add_argument("--all", action="store_true", help="also episodes whose summary says the task failed")
    ap.add_argument("--unfinished", action="store_true", help="also episodes that were cut short by a crash")
    ap.add_argument("--no_trim", action="store_true", help="keep the rows in which nothing moves")
    ap.add_argument("--still_s", type=float, default=1.0)
    ap.add_argument("--pause_s", type=float, default=1.0)
    ap.add_argument("--videos", action="store_true", help="store pictures as video files")
    ap.add_argument("--npz", action="store_true")
    a = ap.parse_args(argv)
    kw = dict(fps=a.fps, trim=not a.no_trim, still_s=a.still_s, pause_s=a.pause_s)
    if a.npz:
        for d in episodes(a.src):
            got = export_npz(d, Path(a.out) / f"{d.name}.npz", **kw)
            print(f"{d.name}: {got['rows']} rows kept, {got['dropped']} dropped -> {got['out']}")
        return 0
    export_lerobot(a.src, a.out, a.repo_id, only_success=not a.all, unfinished=a.unfinished, use_videos=a.videos, **kw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
