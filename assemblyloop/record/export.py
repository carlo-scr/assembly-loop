"""Runs -> LeRobot v2.1 at a fixed rate, ADDED to what was exported before (one episode per run).

The lab's rules (swoosh_collect/export_lerobot.py), kept:
  - nearest-in-time, NEVER interpolated: every row takes the nearest sample of every stream and says how far away it
    was (`sync_error_s`); interpolating would invent states and commands that never were;
  - every video is re-encoded onto the grid, so that frame i of the mp4 IS row i of the parquet;
  - only runs that are ready and passed validation are taken; the others are listed with the reason;
  - the raw logs stay authoritative: a small copy of each run's raw files goes along (raw/<run>/), so the dataset
    can be re-derived from the hub alone.

THE ACTION. This data is made by a script, not an operator, so there is no controller input. `action` is what the
arm was commanded to do over the next row: the change of the controller's PLANNED flange position (lab world frame,
mm), the change of the planned last joint (the gripper's turn, deg), and the gripper target (gripper counts / 10 =
its own mm). The planned joints of the next row (`action.joints_planned_next`) and the goal of the segment being
driven (`action.target_world_mm`) are separate columns, so that a consumer chooses deliberately. `action.is_valid` is
false where the switch did not allow moving (the arm stood paused). Every episode says where its actions came from
(`action_source`: script; later teleop, policy, correction), so data of other kinds can join the same dataset.

THE POSE IS THE FLANGE, not the fingertips: about 172 mm short of where the jaws meet.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .store import read_json
from .validate import read_jsonl

CHUNK = 1000
JOINTS = [f"joint{i}" for i in range(1, 8)]
FEATURES = {  # name: (width, names)
    "observation.state": (17, [f"{j}_measured_deg" for j in JOINTS] + ["gripper_mm", "flange_base_x_mm", "flange_base_y_mm", "flange_base_z_mm",
                                                                       "flange_base_roll_deg", "flange_base_pitch_deg", "flange_base_yaw_deg",
                                                                       "flange_world_x_mm", "flange_world_y_mm", "flange_world_z_mm"]),
    "observation.joints_planned": (7, [f"{j}_planned_deg" for j in JOINTS]),
    "action": (5, ["dx_world_mm", "dy_world_mm", "dz_world_mm", "djoint7_deg", "gripper_target_mm"]),
    "action.joints_planned_next": (7, [f"{j}_planned_next_deg" for j in JOINTS]),
    "action.target_world_mm": (3, ["x_mm", "y_mm", "z_mm"]),
    "sync_error_s": (1, ["seconds"]),
}


def pick(times, grid):
    """Nearest-in-time indices of `times` for every `grid` point, and how far each was."""
    times = np.asarray(times, float)
    if len(times) == 0:
        return np.zeros(len(grid), int), np.full(len(grid), np.inf)
    i = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
    prev = np.clip(i - 1, 0, len(times) - 1)
    take = np.abs(times[prev] - grid) < np.abs(times[i] - grid)
    sel = np.where(take, prev, i)
    return sel, np.abs(times[sel] - grid)


def _carry(times, values, grid, dim):
    """The last value given at or before each grid time (for commands: a target holds until the next one)."""
    out = np.full((len(grid), dim), np.nan)
    if not len(times):
        return out
    idx = np.searchsorted(np.asarray(times, float), grid, side="right") - 1
    ok = idx >= 0
    out[ok] = np.asarray(values, float).reshape(len(times), dim)[idx[ok]]
    return out


def reencode(src, dst, idx, fps, quality=8):
    """One output frame per row, frame idx[i] of the source; read once, front to back (idx never goes back)."""
    import imageio.v2 as imageio

    w = imageio.get_writer(str(dst), fps=float(fps), codec="libx264", quality=int(quality), macro_block_size=1)
    n, cur, k = 0, None, -1
    try:
        with imageio.get_reader(str(src)) as rd:
            it = iter(rd)
            for j in idx:
                while k < j:
                    try:
                        cur, k = next(it), k + 1
                    except StopIteration:
                        break
                if cur is None:
                    break
                w.append_data(cur)
                n += 1
    finally:
        w.close()
    return n


def episode_arrays(run_dir, fps):
    """The parquet columns of one run, and for every camera which of its frames each row shows."""
    run_dir = Path(run_dir)
    meta = read_json(run_dir / "run.json", {}) or {}
    R = np.asarray(((meta.get("provenance") or {}).get("arm") or {}).get("R_world_from_base") or np.eye(3), float)
    arm = [r for r in read_jsonl(run_dir / "raw" / "arm_state.jsonl") if "t" in r and "pose_base" in r]
    if len(arm) < 2:
        raise ValueError("fewer than two readings of the arm")
    t_end = float((meta.get("labels") or {}).get("motion_s") or arm[-1]["t"])
    grid = np.arange(0.0, t_end + 1e-9, 1.0 / fps)
    ta = np.array([r["t"] for r in arm], float)
    sel, err = pick(ta, grid)
    nan7 = [np.nan] * 7
    meas = np.array([r.get("joints_measured_deg") or nan7 for r in arm], float)[sel]
    plan = np.array([r.get("joints_planned_deg") or nan7 for r in arm], float)[sel]
    pose = np.array([r["pose_base"] for r in arm], float)[sel]
    world = pose[:, :3] @ R.T
    g_rows = [(r["t"], r["gripper_counts"]) for r in arm if r.get("gripper_counts") is not None]
    grip = _carry([a for a, _ in g_rows], [[b / 10.0] for _, b in g_rows], grid, 1)
    held = np.array([bool(r.get("switch_held", True)) for r in arm])[sel]
    cmd = read_jsonl(run_dir / "raw" / "commanded.jsonl")
    gr = [(c["t"], c["asked"]) for c in cmd if c.get("kind") == "grip" and "asked" in c]
    grip_target = _carry([a for a, _ in gr], [[b / 10.0] for _, b in gr], grid, 1)
    mv = [(c["t"], c["goal"]) for c in cmd if c.get("kind") == "move" and c.get("goal") is not None]
    target = _carry([a for a, _ in mv], [list(np.asarray(b, float) @ R.T) for _, b in mv], grid, 3)
    nxt = np.r_[np.arange(1, len(grid)), len(grid) - 1]
    action = np.c_[world[nxt] - world, plan[nxt, 6] - plan[:, 6], grip_target[:, 0]]
    cols = {"observation.state": np.c_[meas, grip, pose, world], "observation.joints_planned": plan, "action": action, "action.joints_planned_next": plan[nxt],
            "action.target_world_mm": target, "sync_error_s": err[:, None]}
    cams = {}
    for key in meta.get("cameras") or []:
        ft = read_json(run_dir / "video" / f"{key}_frame_times.json", {}) or {}
        ci, ce = pick(ft.get("t") or [], grid)
        cams[key] = ci
        cols["sync_error_s"] = np.maximum(cols["sync_error_s"], ce[:, None])
    return cols, held, cams, meta


def export_campaign(campaign, fps=30.0, quality=8, include_failed=False, say=print):
    """Add every ready, passed run that is not yet in `<campaign>/lerobot/` to it. Returns (added, refused)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = campaign.path / "lerobot"
    (out / "meta").mkdir(parents=True, exist_ok=True)
    episodes = read_jsonl(out / "meta" / "episodes.jsonl")
    stats = read_jsonl(out / "meta" / "episodes_stats.jsonl")
    tasks = {t["task"]: t["task_index"] for t in read_jsonl(out / "meta" / "tasks.jsonl")}
    info = read_json(out / "meta" / "info.json", {}) or {}
    done = {e.get("run") for e in episodes}
    cams_fixed = info.get("assembly_loop", {}).get("cameras")
    shapes = dict(info.get("assembly_loop", {}).get("video_shapes") or {})  # per camera: [width, height] as its frames are
    added, refused = [], []
    for run in campaign.runs():
        if run.name in done:
            continue
        v, meta = run.validation, run.meta
        if run.status != "ready":
            refused.append((run.name, f"status {run.status!r}: not closed and checked yet"))
            continue
        if not v.get("ok") and not include_failed:
            refused.append((run.name, "failed its checks: " + ", ".join(c["check"] for c in v.get("checks", []) if c["level"] == "fail")))
            continue
        cams = list(meta.get("cameras") or [])
        if cams_fixed is not None and cams != cams_fixed:
            refused.append((run.name, f"its cameras {cams} are not the dataset's {cams_fixed} (every episode needs the same)"))
            continue
        try:
            cols, held, cam_idx, meta = episode_arrays(run.path, fps)
        except (ValueError, KeyError, OSError) as e:
            refused.append((run.name, f"could not be read: {e}"))
            continue
        idx = len(episodes)
        chunk = idx // CHUNK
        n = len(cols["action"])
        task_text = (meta.get("task") or {}).get("instruction") or "pick and place a block"
        if task_text not in tasks:
            tasks[task_text] = len(tasks)
        table = {k: pa.array(np.asarray(v, np.float32).tolist(), type=pa.list_(pa.float32(), FEATURES[k][0])) for k, v in cols.items()}
        table.update({"action.is_valid": pa.array(held.tolist(), type=pa.bool_()), "timestamp": pa.array((np.arange(n) / fps).astype(np.float32)),
                      "frame_index": pa.array(np.arange(n, dtype=np.int64)), "episode_index": pa.array(np.full(n, idx, np.int64)),
                      "index": pa.array(np.arange(n, dtype=np.int64) + int(sum(e["length"] for e in episodes))), "task_index": pa.array(np.full(n, tasks[task_text], np.int64))})
        d = out / f"data/chunk-{chunk:03d}"
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(table), d / f"episode_{idx:06d}.parquet")
        for key in cams:
            vd = out / f"videos/chunk-{chunk:03d}/observation.images.{key}"
            vd.mkdir(parents=True, exist_ok=True)
            written = reencode(run.path / "video" / f"{key}.mp4", vd / f"episode_{idx:06d}.mp4", cam_idx[key], fps, quality)
            if written != n:
                say(f"  [warn] {run.name} {key}: {written} frames for {n} rows")
        oc, task = meta.get("outcome") or {}, meta.get("task") or {}
        placed = oc.get("placed") or {}
        episodes.append(dict(episode_index=idx, tasks=[task_text], length=n, run=run.name, success=bool(oc.get("success")), outcome=oc.get("kind"),
                             verified_by_camera=bool(oc.get("verified_by_camera")), placed_off_mm=placed.get("off_mm"), placed_turn_deg=placed.get("turn_deg"),
                             action_source=task.get("action_source", "script"), arm=task.get("arm"), block=task.get("block"), grasp_offset_mm=task.get("grasp_offset_mm"),
                             job=task.get("job"), design=task.get("design")))
        st = {}
        for k, v in cols.items():
            a = np.asarray(v, float)
            with np.errstate(all="ignore"):
                st[k] = dict(min=np.nanmin(a, 0).tolist(), max=np.nanmax(a, 0).tolist(), mean=np.nanmean(a, 0).tolist(), std=np.nanstd(a, 0).tolist(), count=[n])
        stats.append(dict(episode_index=idx, stats=st))
        raw = out / "raw" / run.name  # the authoritative logs go along, small
        (raw / "raw").mkdir(parents=True, exist_ok=True)
        (raw / "video").mkdir(exist_ok=True)
        shutil.copy2(run.path / "run.json", raw / "run.json")
        for f in (run.path / "raw").glob("*.jsonl"):
            shutil.copy2(f, raw / "raw" / f.name)
        for f in (run.path / "video").glob("*_frame_times.json"):
            shutil.copy2(f, raw / "video" / f.name)
        if cams_fixed is None:
            cams_fixed = cams
        for key in cams:
            if key not in shapes:
                act = (read_json(run.path / "video" / f"{key}_frame_times.json", {}) or {}).get("actual") or {}
                shapes[key] = [int(act.get("width") or 0), int(act.get("height") or 0)]
        added.append(run.name)
        say(f"  [ok]   {run.name}: episode {idx}, {n} rows, worst sync {1000 * float(np.max(cols['sync_error_s'])):.0f} ms, {oc.get('kind')}")
    for name, why in refused:
        say(f"  [REFUSED] {name}: {why}")
    if not added:
        return added, refused
    features = {k: dict(dtype="float32", shape=[w], names=names) for k, (w, names) in FEATURES.items()}
    features["action.is_valid"] = dict(dtype="bool", shape=[1], names=None)
    for k in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        features[k] = dict(dtype="float32" if k == "timestamp" else "int64", shape=[1], names=None)
    for key in cams_fixed:
        features[f"observation.images.{key}"] = dict(dtype="video", shape=[shapes[key][1], shapes[key][0], 3], names=["height", "width", "channel"],
                                                     info={"video.fps": fps, "video.codec": "h264", "video.pix_fmt": "yuv420p", "has_audio": False})
    total = int(sum(e["length"] for e in episodes))
    info = dict(codebase_version="v2.1", robot_type="xarm7_" + str(episodes[0].get("arm") or "right"), total_episodes=len(episodes), total_frames=total, total_tasks=len(tasks),
                total_videos=len(episodes) * len(cams_fixed), total_chunks=(len(episodes) - 1) // CHUNK + 1, chunks_size=CHUNK, fps=fps, splits={"train": f"0:{len(episodes)}"},
                data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4", features=features,
                assembly_loop=dict(cameras=cams_fixed, video_shapes=shapes, campaign=campaign.name, action="change of the PLANNED flange position (world, mm), of joint 7 (deg), gripper target (counts/10)",
                                   pose="flange, not fingertips", world="x forward, y left, z up (swoosh-collect R_WORLD_FROM_BASE)"))
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    (out / "meta" / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    (out / "meta" / "episodes_stats.jsonl").write_text("".join(json.dumps(s) + "\n" for s in stats))
    (out / "meta" / "tasks.jsonl").write_text("".join(json.dumps(dict(task_index=i, task=t)) + "\n" for t, i in sorted(tasks.items(), key=lambda kv: kv[1])))
    _card(out, campaign, info, episodes)
    return added, refused


def _card(out, campaign, info, episodes):
    """The dataset's README on the hub: what it is, in a few lines, and how many episodes worked."""
    ok = sum(1 for e in episodes if e.get("success"))
    (out / "README.md").write_text(f"""---
tags:
- LeRobot
- robotics
- xarm7
- assembly
configs:
- config_name: default
  data_files: data/*/*.parquet
---
# {campaign.name}

Recorded by assembly-loop on the lab's xArm 7 cell: every episode is one block taken and put down by a script, with
the camera's check of where it landed. LeRobot v2.1 at {info['fps']:g} Hz, {info['total_episodes']} episodes ({ok} placed as planned),
{info['total_frames']} frames, cameras: {', '.join(info['assembly_loop']['cameras'])}.

- `action`: {info['assembly_loop']['action']}. `action.is_valid` is false while the arm stood paused.
- Poses are the {info['assembly_loop']['pose']}; world frame: {info['assembly_loop']['world']}.
- Per episode (meta/episodes.jsonl): success, outcome, where the block landed against the plan (`placed_off_mm`),
  the block, the arm, `action_source`.
- `raw/<run>/`: the authoritative logs every row was derived from (nearest in time, never interpolated).
""")
