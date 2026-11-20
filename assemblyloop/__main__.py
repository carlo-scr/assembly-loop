"""Assembly loop for the lab's xArm 7.

  python -m assemblyloop dashboard                         the dashboard: start, calibrate, arm, move and stop from one page
                                                           (it opens at OFF; choose the right arm, the left arm or a stand-in there)
  python -m assemblyloop status --ip 192.168.1.199         READ-ONLY: connects, prints the arm's state, disconnects
  python -m assemblyloop deadman --port 6666               shows whether the Quest grip arrives (no arm involved)
  python -m assemblyloop look --out look.jpg               one picture from the camera with what it makes out drawn in (no arm involved)

Press-fit bricks on a taught stud plate (command line only; older than the dashboard):
  python -m assemblyloop demo-plate --out plate_demo.json  a made-up taught file, for trying `plan` anywhere
  python -m assemblyloop plan --plate plate_demo.json --src C3 --dst E3 --offset 6 --correct truth
                                                           DRY: prints every step and runs it on a stand-in arm
  python -m assemblyloop teach --ip 192.168.1.199 --out plate.json
                                                           READ-ONLY: records poses while YOU move the arm with the lab's tools
  python -m assemblyloop run --plate plate.json --ip 192.168.1.199 --src C3 --dst E3 --correct truth --air 100 --step --live
                                                           LIVE: hold the Quest grip to let it move; --air lifts the whole
                                                           motion above the bricks; --step asks for a key before every step

From the dashboard nothing moves unless the session is armed AND the grip is held; from the command line nothing moves
unless `run --live` is given and confirmed. The red emergency-stop button stays within reach at all times; the
hold-to-run switch is not a substitute for it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .arms import ArmFault, FakeArm, XArm
from .cycle import NothingHeld, Params, execute, plan
from .deadman import ManualDeadman, QuestDeadman
from .guard import Guard, Refused, Released
from .plate import Plate, demo


def _describe(steps, lim, log=print):
    for i, s in enumerate(steps, 1):
        if s["kind"] == "move":
            u, v, w = lim.coords(s["to"])
            log(f"  {i:2d}. {s['name']:34s} to {u:7.1f} along the columns, {v:7.1f} along the rows, {w:7.1f} mm from the hover height, at {s['speed']:.0f} mm/s")
        elif s["kind"] == "grip":
            log(f"  {i:2d}. {s['name']:34s} gripper to {s['counts']} counts")
        else:
            log(f"  {i:2d}. {s['name']:34s} read pose and joints")


def cmd_plan(a):
    plate = Plate.load(a.plate)
    P = Params()
    corr = a.offset if a.correct == "truth" else 0.0
    steps = plan(plate, a.src, a.dst, a.offset, corr, P, a.air)
    lim = plate.limits(air_mm=a.air)
    print(f"DRY RUN on a stand-in arm ({a.plate}): brick from {a.src} to {a.dst}, gripped {a.offset:+.1f} mm off-centre, aim corrected by {corr:+.1f} mm" + (f", lifted {a.air:.0f} mm into the air" if a.air else ""))
    _describe(steps, lim)
    arm = FakeArm(pose=[*(plate.origin + a.air * plate.n), *plate.rpy], bricks={"brick": plate.grasp(a.src)}, up=plate.n)
    dm = ManualDeadman(clock=lambda: arm.t)
    dm.press()
    g = Guard(arm, dm, lim, clock=lambda: arm.t, sleep=arm.advance, say=lambda *_: None)
    g.start()
    try:
        execute(g, steps, P, air=bool(a.air))
    finally:
        g.finish()
    moves = [e for e in arm.log if e[0] == "move"]
    low = min(float(lim.coords(e[4])[2]) for e in moves)
    off = arm.bricks["brick"] - plate.grasp(a.dst)
    miss = float(np.linalg.norm(off - (off @ plate.n) * plate.n))  # along the plate
    print(f"  {len(moves)} segments, longest {max(np.linalg.norm(e[2]) for e in moves):.1f} mm, fastest {max(e[3] for e in moves):.0f} mm/s, lowest point {low:.1f} mm from the hover height "
          f"(taught grasp height: {-plate.grasp_drop:.1f}); simulated time {arm.t:.0f} s")
    print(f"  the brick ends {miss:.1f} mm from stud {a.dst}" if not a.air else "  in the air: the brick was not touched")
    return dict(segments=len(moves), miss_mm=miss, time_s=arm.t, xarm_loaded="xarm" in sys.modules)


def cmd_deadman(a):
    dm = QuestDeadman(port=a.port, hand=a.hand).start()
    print(f"listening on UDP {a.port} for {a.seconds:.0f} s: squeeze and release the {a.hand} grip")
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < a.seconds:
            print(f"  frames {dm.n:6d}   grip {dm.last_grip if dm.last_grip is not None else float('nan'):.2f}   {'HELD' if dm.held() else dm.why()}        ", end="\r", flush=True)
            time.sleep(0.1)
    finally:
        dm.stop()
    print(f"\n{dm.n} frames in {a.seconds:.0f} s" + ("" if dm.n else ": nothing arrived (is the lab's dashboard holding the port, or the headset sending elsewhere?)"))


def cmd_status(a):
    arm = XArm(a.ip)
    arm.connect()
    try:
        st = arm.status()
    finally:
        arm.disconnect()
    for k, v in st.items():
        print(f"  {k:22s} {v}")
    print("  (read only: nothing was enabled, cleared or changed)")
    return st


def cmd_look(a):
    """One picture. Before the camera was shown the table (no camera.json) it can only outline what is coloured."""
    from . import eyes
    from .blocks import Catalogue

    im = eyes.cv2.imread(a.picture) if a.picture else eyes.grab(a.camera or eyes.SCENE_CAMERA)
    if im is None:
        raise SystemExit(f"{a.picture} is not a picture")
    if Path(a.setup).exists() and Path(a.blocks).exists():
        setup, cat = eyes.Setup.load(a.setup), Catalogue.load(a.blocks)
        tall = max(r.height for k in cat.kinds for r in k.rests())
        seen, unknown, blobs = eyes.look(im, setup.camera, setup.table, cat, setup.region((im.shape[1], im.shape[0]), tall))
        for s in seen:
            print(f"  {s.name:22s} {s.rest.name:12s} at {s.uv[0]:7.1f}, {s.uv[1]:7.1f} mm, long side at {s.angle:5.1f} deg, outline fits {100 * s.score:.0f}%")
        for _, why in unknown:
            print(f"  not made out: {why}")
        out = eyes.draw(im, blobs, seen, unknown, setup.camera, setup.table)
    else:
        blobs = eyes.find_blobs(im)
        print(f"  {len(blobs)} coloured patches: " + ", ".join(f"{b.colour} ({b.area} px)" for b in blobs))
        print(f"  (outlines only: {a.setup if not Path(a.setup).exists() else a.blocks} is missing)")
        out = eyes.draw(im, blobs)
    eyes.cv2.imwrite(a.out, out)
    print(f"wrote {a.out}")


def cmd_teach(a):
    arm = XArm(a.ip)
    arm.connect()
    ask = [("origin", f"the open gripper centred over the seated brick on stud {a.origin}, 60 to 80 mm above the brick, fingers either side of it"),
           ("col", f"the same over the brick on stud {a.col} (same row)"), ("row", f"the same over the brick on stud {a.row} (same column)")]
    d = dict(side=a.side, fingers=a.fingers, taught={})
    try:
        for key, what in ask:
            input(f"Move the arm with the lab's tools: {what}. Then press Enter here. ")
            p = arm.pose()
            d["taught"][key] = dict(cell=getattr(a, key), hover_mm=p[:3].round(3).tolist())
            if "rpy_deg" not in d:
                d["rpy_deg"] = p[3:].round(3).tolist()
            elif float(np.max(np.abs((p[3:] - np.array(d["rpy_deg"]) + 180.0) % 360.0 - 180.0))) > 1.0:
                raise Refused("the gripper's orientation changed between taught poses: keep it fixed and start again")
            print(f"  {key}: {p.round(2).tolist()}")
        input(f"Back over stud {a.origin}: lower the open gripper STRAIGHT DOWN until the jaws are where they should close on the brick. Then press Enter. ")
        d["grasp_mm"] = arm.pose()[:3].round(3).tolist()
    finally:
        arm.disconnect()
    plate = Plate.from_taught(d)  # refuses a pitch, angle or height that cannot be right
    Path(a.out).write_text(json.dumps(d, indent=1))
    print(f"taught: pitch {np.linalg.norm(plate.ec):.2f} / {np.linalg.norm(plate.er):.2f} mm, jaws close {plate.grasp_drop:.1f} mm below the hover height -> {a.out}")


def cmd_run(a):
    plate = Plate.load(a.plate)
    P = Params()
    lim = plate.limits(air_mm=a.air)
    if not a.live:
        raise SystemExit("this command moves the arm: add --live when you mean it (use `plan` for a dry run)")
    arm = XArm(a.ip)
    arm.connect()
    dm = ManualDeadman() if a.deadman == "step" else QuestDeadman(port=a.port, hand=a.hand).start()
    log_dir = Path(a.log) / time.strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    g = Guard(arm, dm, lim, log_path=log_dir / "guard.jsonl")
    rng = np.random.default_rng(a.seed)
    try:
        st = arm.status()
        print("\n".join(f"  {k:22s} {v}" for k, v in st.items()))
        turn = float(np.max(np.abs((np.array(st["pose"][3:]) - plate.rpy + 180.0) % 360.0 - 180.0)))
        if turn > 1.5:
            raise Refused(f"the gripper is turned {turn:.1f} deg away from the taught orientation: bring it back with the lab's tools")
        why = lim.why_outside(st["pose"][:3])
        if why:
            raise Refused(f"the arm is not inside the taught volume ({why}): bring it over the plate with the lab's tools")
        if input(f"\nThis will MOVE the {plate.side} arm at {a.ip}. Workspace clear, emergency stop in reach? Type MOVE to continue: ").strip() != "MOVE":
            raise SystemExit("not confirmed, nothing moved")
        g.start()
        src, dst = a.src, a.dst
        for i in range(a.cycles):
            off = float(a.offset if a.offset is not None else rng.uniform(-a.max_offset, a.max_offset))
            corr = off if a.correct == "truth" else 0.0
            steps = plan(plate, src, dst, off, corr, P, a.air)
            print(f"cycle {i + 1}/{a.cycles}: {src} -> {dst}, gripped {off:+.1f} mm off-centre, aim corrected by {corr:+.1f} mm")

            def before(s, i=i):
                if a.step or a.deadman == "step":
                    if input(f"   Enter = {s['name']}   (q + Enter = stop) ").strip().lower() == "q":
                        raise Released("stopped at the keyboard")
                if a.deadman == "step":
                    dm.press()

            def after(s):
                if a.deadman == "step":
                    dm.release()

            try:
                rec = execute(g, steps, P, air=bool(a.air), before=before, after=after)
            except NothingHeld as e:
                print(f"   {e}")
                rec = dict(nothing_held=True)
            if rec.get("blocked"):
                print(f"   BLOCKED at '{rec['blocked']['step']}': {rec['blocked']['why']}. Lifted clear; look before going on.")
                with open(log_dir / "cycles.jsonl", "a") as fh:
                    fh.write(json.dumps(dict(cycle=i, src=src, dst=dst, offset_mm=off, correction_mm=corr, air_mm=a.air, **rec)) + "\n")
                break
            with open(log_dir / "cycles.jsonl", "a") as fh:
                fh.write(json.dumps(dict(cycle=i, src=src, dst=dst, offset_mm=off, correction_mm=corr, air_mm=a.air, **rec)) + "\n")
            if not a.air and not rec.get("nothing_held"):
                src, dst = dst, src  # the brick is now on the other stud
    except (Refused, Released, ArmFault) as e:
        print(f"\nSTOPPED: {type(e).__name__}: {e}")
    finally:
        g.finish()
        if hasattr(dm, "stop"):
            dm.stop()
        arm.disconnect()
        print(f"log: {log_dir}")


def cmd_dashboard(a):
    from .dashboard import make_server
    from .session import Session, default_arms

    camera = None
    if a.camera:
        from .eyes import V4L2

        camera = V4L2(a.camera)
    ses = Session(default_arms(a.right, a.left, a.mount_deg), quest_port=a.quest_port, hand=a.hand, spots_path=a.spots, log_dir=a.log, say=print, camera=camera)
    srv = make_server(ses, a.port)
    print(f"dashboard: http://127.0.0.1:{srv.server_address[1]}/   (nothing is connected until you start a session on the page; Ctrl-C here stops the arm and closes)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ses.close()
        srv.server_close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m assemblyloop", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("demo-plate")
    s.add_argument("--out", default="plate_demo.json")
    for name in ("plan", "run"):
        s = sub.add_parser(name)
        s.add_argument("--plate", required=True)
        s.add_argument("--src", required=True)
        s.add_argument("--dst", required=True)
        s.add_argument("--offset", type=float, default=0.0 if name == "plan" else None, help="mm along the fingers; run: random within --max_offset if not given")
        s.add_argument("--correct", choices=("none", "truth"), default="truth")
        s.add_argument("--air", type=float, default=0.0, help="lift the whole motion by this many mm (a rehearsal above the bricks)")
        if name == "run":
            s.add_argument("--ip", required=True)
            s.add_argument("--live", action="store_true")
            s.add_argument("--step", action="store_true")
            s.add_argument("--deadman", choices=("quest", "step"), default="quest")
            s.add_argument("--port", type=int, default=6666)
            s.add_argument("--hand", default="right")
            s.add_argument("--cycles", type=int, default=1)
            s.add_argument("--max_offset", type=float, default=10.0)
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--log", default="assembly_logs")
    s = sub.add_parser("deadman")
    s.add_argument("--port", type=int, default=6666)
    s.add_argument("--hand", default="right")
    s.add_argument("--seconds", type=float, default=20.0)
    s = sub.add_parser("look")
    s.add_argument("--camera", default=None, help="the camera's device (default: the rig's left scene camera)")
    s.add_argument("--picture", default=None, help="look at this picture file instead of taking one")
    s.add_argument("--out", default="look.jpg")
    s.add_argument("--setup", default="camera.json")
    s.add_argument("--blocks", default="blocks.json")
    s = sub.add_parser("status")
    s.add_argument("--ip", required=True)
    s = sub.add_parser("teach")
    s.add_argument("--ip", required=True)
    s.add_argument("--out", default="plate.json")
    s.add_argument("--side", default="right")
    s.add_argument("--origin", default="C3")
    s.add_argument("--col", default="F3")
    s.add_argument("--row", default="C6")
    s.add_argument("--fingers", choices=("col", "row"), required=True, help="the plate direction along which the brick could slide between the jaws")
    s = sub.add_parser("dashboard")
    s.add_argument("--right", default="192.168.1.199", help="address of the right arm")
    s.add_argument("--left", default="192.168.1.219", help="address of the left arm")
    s.add_argument("--spots", default="spots.json", help="where taught spots are kept")
    s.add_argument("--mount_deg", type=float, default=45.0, help="how far the arms are rolled about their forward axis where they are mounted (0: upright)")
    s.add_argument("--camera", default=None, help="the camera that looks at the table (default: the rig's left scene camera)")
    s.add_argument("--quest_port", type=int, default=6666)
    s.add_argument("--hand", default="right")
    s.add_argument("--port", type=int, default=8777)
    s.add_argument("--log", default="assembly_logs")
    a = ap.parse_args(argv)
    if a.cmd == "demo-plate":
        Path(a.out).write_text(json.dumps(demo(), indent=1))
        print(f"wrote {a.out} (made up, for `plan` only)")
        return None
    return dict(plan=cmd_plan, deadman=cmd_deadman, status=cmd_status, teach=cmd_teach, run=cmd_run, dashboard=cmd_dashboard, look=cmd_look)[a.cmd](a)


if __name__ == "__main__":
    main()
