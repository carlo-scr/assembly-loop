"""Designs: the statics, the arm's own limits, the order, the files, the designer's calls, and a design built by the
stand-in arm from what a made-up camera sees."""
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from assemblyloop import design as D
from assemblyloop.blocks import Catalogue

CAT = Catalogue.load(D.Path(__file__).parent.parent / "blocks.example.json")


def make(*blocks, name="t"):
    return D.Design(name, [D.Block(k, r, x, y, a, z, f"b{i + 1}") for i, (k, r, x, y, a, z) in enumerate(blocks)])


def steps(d):
    return [(s["stands"], s["tilt_deg"]) for s in D.check(d, CAT)["steps"]]


# ------------------------------------------------------------------------------------------------ the statics
def test_the_simplex_agrees_with_a_known_solver_and_with_hand_worked_answers():
    rng = np.random.default_rng(0)
    for _ in range(40):  # feasible by construction: b = A x0 for some x0 >= 0
        A = rng.normal(size=(6, 14))
        x = D._simplex(A, A @ rng.uniform(0, 1, 14), np.zeros(14))
        assert x is not None and np.all(x >= -1e-9) and np.allclose(A @ x, A @ x if x is None else A @ x)
    assert D._simplex(np.array([[1.0, 1.0]]), np.array([-1.0]), np.zeros(2)) is None  # x >= 0 cannot sum to -1
    x = D._simplex(np.array([[1.0, 1.0, 1.0]]), np.array([4.0]), np.array([-1.0, -2.0, 0.0]))  # min -x - 2y with x + y + z = 4
    assert np.allclose(x, [0.0, 4.0, 0.0])
    scipy = pytest.importorskip("scipy.optimize")
    for _ in range(60):
        A = rng.normal(size=(5, 12))
        b = rng.normal(size=5)
        ours = D._simplex(A, b, np.zeros(12)) is not None
        assert ours == (scipy.linprog(np.zeros(12), A_eq=A, b_eq=b, bounds=(0, None), method="highs").status == 0)


def test_how_far_a_stack_can_tilt_matches_the_hand_worked_answer():
    # a cube 10 mm off the middle of the cube under it: its middle is 2.7 mm inside the edge, 12.7 mm above it
    assert steps(make(("red cube", "flat", 0, 0, 0, 0), ("blue cube", "flat", 10, 0, 0, 25.4)))[1] == (True, pytest.approx(np.degrees(np.arctan(2.7 / 12.7)), abs=0.1))
    # six cubes: the whole tower about its bottom edge, middle 76.2 mm up, 12.7 mm inside
    assert steps(make(*[("red cube", "flat", 0, 0, 0, 25.4 * k) for k in range(6)]))[-1][1] == pytest.approx(np.degrees(np.arctan(12.7 / 76.2)), abs=0.1)
    # a single block on the mat is solid (at least the 20 degrees looked for)
    assert steps(make(("red cube", "flat", 0, 0, 0, 0))) == [(True, D.MAX_TILT_DEG)]


def test_an_overhang_falls_and_a_counterweight_only_helps_if_it_could_be_put_there_first():
    assert steps(make(("red cube", "flat", 0, 0, 0, 0), ("blue cube", "flat", 15, 0, 0, 25.4)))[1][0] is False
    r = D.check(make(("red cube", "flat", 0, 0, 0, 0), ("green plank", "flat", 20, 0, 0, 25.4), ("blue cube", "flat", -5, 0, 0, 38.1)), CAT)
    assert [s["stands"] for s in r["steps"]] == [True, False, True] and not r["ok"]  # standing in the end is not enough: the arm builds it step by step
    assert "falls over" in r["blocks"][1]["problems"][-1]
    with pytest.raises(D.Refused, match="no order"):
        D.find_order(make(("red cube", "flat", 0, 0, 0, 0), ("green plank", "flat", 20, 0, 0, 25.4), ("blue cube", "flat", -5, 0, 0, 38.1)), CAT)


def test_a_plank_across_two_cubes_stands_on_both():
    r = D.check(make(("red cube", "flat", 0, 28, 0, 0), ("red cube", "flat", 0, -28, 0, 0), ("blue plank", "flat", 0, 0, 90, 25.4)), CAT)
    assert r["ok"] and r["blocks"][2]["rests_on"] == ["b1", "b2"] and r["tilt_deg"] == D.MAX_TILT_DEG
    # one cube taken away: the plank's middle is beyond the cube that is left
    r = D.check(make(("red cube", "flat", 0, 28, 0, 0), ("blue plank", "flat", 0, 0, 90, 25.4)), CAT)
    assert r["steps"][1]["stands"] is False and r["blocks"][1]["held_by_mm"] < 0 and "beyond" in r["blocks"][1]["warnings"][0]


# ------------------------------------------------------------------------------------------------ what the arm can do
def test_what_the_arm_cannot_do_is_named():
    def problems(d, k):
        return " ".join(D.check(d, CAT)["blocks"][k]["problems"])

    assert "overlaps" in problems(make(("red cube", "flat", 0, 0, 0, 0), ("blue cube", "flat", 10, 0, 0, 0)), 1)
    assert "floats" in problems(make(("red cube", "flat", 0, 0, 0, 40)), 0)
    assert "comes later" in problems(make(("blue cube", "flat", 0, 0, 0, 25.4), ("red cube", "flat", 0, 0, 0, 0)), 0)
    assert "stands over it" in problems(make(("red cube", "flat", 0, 28, 0, 0), ("red cube", "flat", 0, -28, 0, 0), ("blue plank", "flat", 0, 0, 90, 25.4), ("green cube", "flat", 0, 0, 0, 0)), 3)
    assert "rolls" in problems(make(("red cylinder", "lying", 0, 0, 0, 0)), 0)
    assert "from the spot" in problems(make(("red cube", "flat", 180, 0, 0, 0)), 0)
    tall = make(*[("red bar", "on end", 0, 0, 0, 57.0 * k) for k in range(5)])
    assert "builds up to" in problems(tall, 4)
    # two planks side by side along their long faces: the fingers of the second one land in the first
    assert "fingers would hit step 1" in problems(make(("red plank", "flat", 0, 0, 0, 0), ("blue plank", "flat", 0, 30, 0, 0)), 1)
    assert not problems(make(("red plank", "flat", 0, 0, 0, 0), ("blue plank", "flat", 0, 55, 0, 0)), 1)  # far enough apart
    assert not problems(make(("red plank", "flat", 0, 0, 0, 0), ("blue plank", "flat", 62, 0, 0, 0)), 1)  # end to end: the fingers are on the long faces
    # the body of the gripper over a short block next to a tall one
    assert "gripper would hit" in problems(make(*[("red cube", "flat", 0, 0, 0, 25.4 * k) for k in range(5)], ("blue cube", "flat", 0, 40, 0, 0)), 5)


def test_a_square_block_is_taken_the_other_way_round_when_that_leaves_room():
    r = D.check(make(("red cube", "flat", 0, 22, 0, 0), ("red cube", "flat", 0, -22, 0, 0)), CAT)  # 18.6 mm apart: no room for a finger between them
    assert r["ok"] and r["blocks"][1]["grip_deg"] == 90.0  # so its jaws go across x, away from the other cube
    r = D.check(make(("red plank", "flat", 0, 22, 0, 0), ("red plank", "flat", 0, -22, 0, 0)), CAT)
    assert not r["ok"] and "fingers" in r["blocks"][1]["problems"][0]  # a plank has only one way


def test_an_order_that_works_is_found():
    d = make(("blue plank", "flat", 0, 0, 90, 25.4), ("red cube", "flat", 0, 28, 0, 0), ("red cube", "flat", 0, -28, 0, 0))
    assert not D.check(d, CAT)["ok"]
    o = D.find_order(d, CAT)
    assert [b.id for b in o.blocks] == ["b2", "b3", "b1"] and D.check(o, CAT)["ok"]


def test_settle_drops_every_block_onto_what_is_under_it():
    d = D.settle(make(("red cube", "flat", 0, 0, 0, 0), ("blue plank", "flat", 5, 0, 0, 30), ("green cube", "flat", 5, 0, 0, 60)), CAT)
    assert [b.z for b in d.blocks] == [0.0, 25.4, 38.1]  # lowest first: the plank lands on the cube, the green cube on the plank
    d.blocks[0].x = 100.0
    assert [b.z for b in D.settle(d, CAT).blocks] == [0.0, 0.0, 12.7]  # take the cube away and the rest comes down


def test_the_examples_that_come_with_the_program_are_all_buildable():
    names = [d["name"] for d in D.listing("/nonexistent")]
    assert len(names) >= 4
    for n in names:
        r = D.check(D.load("/nonexistent", n), CAT)
        assert r["ok"] and r["tilt_deg"] >= D.SHAKY_DEG, (n, r["problems"], [b["problems"] for b in r["blocks"]])


# ------------------------------------------------------------------------------------------------ files and the designer's calls
def test_designs_are_saved_listed_and_loaded_and_names_cannot_reach_outside(tmp_path):
    d = make(("red cube", "flat", 0, 0, 0, 0), name="My tower 1")
    f = D.save(tmp_path, d)
    assert f == tmp_path / "my_tower_1.json" and D.load(tmp_path, "My tower 1").to_dict() == d.to_dict()
    assert {"name": "My tower 1", "n": 1, "example": False} == {k: v for k, v in D.listing(tmp_path)[0].items() if k != "saved"}
    for bad in ("../x", "a/b", "", " ", ".hidden", "x" * 41, "..", "con\0"):
        with pytest.raises(D.Refused):
            D.check_name(bad)
    with pytest.raises(D.Refused, match="no design"):
        D.load(tmp_path, "nothing")
    with pytest.raises(D.Refused, match="not readable"):
        D.Design.from_dict({"blocks": [{"kind": "red cube"}]})
    with pytest.raises(D.Refused, match="cannot lie"):
        D.check(make(("red cube", "on end", 0, 0, 0, 0)), CAT)


@pytest.fixture
def served(tmp_path):
    from assemblyloop.arms import FakeArm
    from assemblyloop.dashboard import make_server
    from assemblyloop.session import Session

    s = Session({"stand-in": dict(label="Stand-in arm (no robot)", ip=None, make=FakeArm)}, spots_path=tmp_path / "spots.json", log_dir=tmp_path / "logs", probe=lambda ip: False,
                blocks_path=tmp_path / "blocks.json", designs_dir=tmp_path / "designs")
    srv = make_server(s, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(path, body=None, token=srv.token):
        req = urllib.request.Request(url + path, None if body is None else json.dumps(body).encode(), {"Content-Type": "application/json", "X-Token": token})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    yield s, srv, url, call, tmp_path
    srv.shutdown()
    srv.server_close()
    s.close()


def test_the_designer_page_and_its_calls(served):
    s, srv, url, call, tmp = served
    page = urllib.request.urlopen(url + "/build").read().decode()
    assert srv.token in page and "/web/build.js" in page and urllib.request.urlopen(url + "/web/build.js").status == 200
    assert call("/web/build.html")[0] == 404  # the page only with its token
    code, cat = call("/api/design/catalogue")
    assert code == 200 and cat["example"] and len(cat["kinds"]) == 16 and {k["colour"] for k in cat["kinds"]} == {"green", "red", "blue", "yellow"}
    assert cat["gripper"] == D.GRIPPER and cat["height_mm"] <= D.HEIGHT_MM
    d = make(("red cube", "flat", 0, 0, 0, 0), ("blue cube", "flat", 0, 0, 0, 25.4), name="Two").to_dict()
    code, r = call("/api/design/check", dict(design=d))
    assert code == 200 and r["ok"] and len(r["steps"]) == 2
    assert call("/api/design/check", dict(design=d), token="wrong")[0] == 403
    assert call("/api/design/check", dict(design="no"))[0] == 409
    assert call("/api/design/save", dict(design={**d, "name": "../../etc/x"}))[0] == 409
    code, r = call("/api/design/save", dict(design=d))
    assert code == 200 and r["name"] == "Two" and (tmp / "designs" / "two.json").exists()
    code, r = call("/api/designs")
    assert code == 200 and [x["name"] for x in r["designs"] if not x["example"]] == ["Two"] and any(x["example"] for x in r["designs"])
    code, r = call("/api/design?name=Two")
    assert code == 200 and r["design"]["blocks"][1]["z"] == 25.4
    assert call("/api/design?name=..%2Fx")[0] == 409
    swapped = {**d, "blocks": d["blocks"][::-1]}
    code, r = call("/api/design/order", dict(design=swapped))
    assert code == 200 and [b["id"] for b in r["design"]["blocks"]] == ["b1", "b2"]


# ------------------------------------------------------------------------------------------------ the stand-in arm builds a design
def test_a_design_is_built_from_the_blocks_the_camera_sees(tmp_path):
    pytest.importorskip("cv2")
    from test_sight import CAT as SCAT
    from test_sight import Scene, until

    from assemblyloop.arms import FakeArm
    from assemblyloop.session import Session

    s = Session({"stand-in": dict(label="Stand-in arm (no robot)", ip=None, make=FakeArm)}, spots_path=tmp_path / "spots.json", log_dir=tmp_path / "logs", probe=lambda ip: False,
                fake_speed=40.0, guard_kw=dict(poll_s=0.002), limit_kw=dict(settle_s=0.0), camera_path=tmp_path / "camera.json", blocks_path=tmp_path / "blocks.json",
                designs_dir=tmp_path / "designs")
    s.camera, alive, pressed = Scene(s), [True], [False]

    def grip():
        while alive[0]:
            if s.dm is not None:
                s.dm.feed(pressed[0])
            time.sleep(0.02)

    threading.Thread(target=grip, daemon=True).start()
    try:
        s.start("stand-in")
        assert until(lambda: all(c["ok"] for c in s.state()["checks"]), 6)
        s.arm_on()
        s.use_demo()
        SCAT.save(s.blocks_path)
        pressed[0] = True
        s.show_table("green bar")
        assert until(lambda: s.worker is None, 120) and s.job["outcome"] == "done"
        setup, a = s.setup, s.arm
        # what to build: the slab across, the bar on it turned by 90 degrees and 10 mm forward, the cube on the bar
        d = D.Design("Test", [D.Block("blue slab", "flat", 0.0, 0.0, 0.0, 0.0, "s"), D.Block("green bar", "flat", 10.0, 0.0, 90.0, 0.0, "b"), D.Block("yellow cube", "flat", 10.0, 0.0, 0.0, 0.0, "c")])
        D.settle(d, SCAT)
        assert D.check(d, SCAT)["ok"]
        D.save(s.designs_dir, d)
        # the mat: the three blocks lying apart, away from T, turned any way
        lay = {"slab": ("blue slab", 0, 430.0, 70.0, 100.0), "bar": ("green bar", 0, 445.0, -95.0, 30.0), "cube": ("yellow cube", 0, 395.0, -20.0, 10.0)}
        a.holding, a.bricks = None, {}
        for name, (kind, rest, x, y, angle) in lay.items():
            r = SCAT.kind(kind).rests()[rest]
            uv = setup.table.coords([x, y, 0.0])[:2]
            a.bricks[name] = setup.table.point(uv[0], uv[1], setup.w_grasp + max(0.0, r.height - SCAT.finger_mm))
            s.camera.what[name] = (kind, rest, angle)
        with pytest.raises(D.Refused, match="Look first"):
            s.build_design("Test", "T")
        s.look()
        bad = D.Design("Bad", [D.Block("yellow cube", "flat", 0.0, 0.0, 0.0, 0.0, "c"), D.Block("blue slab", "flat", 0.0, 0.0, 0.0, 25.0, "s")])
        D.save(s.designs_dir, bad)
        with pytest.raises(D.Refused, match="cannot be built"):
            s.build_design("Bad", "T")
        more = D.Design("More", d.blocks + [D.Block("yellow cube", "flat", 0.0, 90.0, 0.0, 0.0, "c2")])
        D.save(s.designs_dir, more)
        with pytest.raises(D.Refused, match="lay these out.*1 x yellow cube"):
            s.build_design("More", "T")
        s.build_design("Test", "T")
        st = s.state()
        assert [p["name"] for p in st["scene"]["plan"]] == ["blue slab", "green bar", "yellow cube"] and st["scene"]["plan"][0]["next"]
        assert until(lambda: s.worker is None, 180), s.state()["job"]
        assert s.job["outcome"] == "done" and s.stack["n"] == 3 and s.stack["design"] == "Test", (s.job["outcome"], s.stack, list(s.lines)[-5:])
        # where each block ended up, in the design's own frame (forward, left, up from the taught spot T)
        up = s.spots.up
        fwd = np.array([1.0, 0.0, 0.0]) - up[0] * up
        fwd /= np.linalg.norm(fwd)
        left, base = np.cross(up, fwd), s.spots.grasp("T")
        for name, blk in (("slab", d.blocks[0]), ("bar", d.blocks[1]), ("cube", d.blocks[2])):
            q = a.bricks[name] - base
            r = SCAT.kind(blk.kind).rests()[0]
            want_up = blk.z + max(0.0, r.height - SCAT.finger_mm) + (s.P.drop_mm if blk.z > 0 else 0.0)  # the stand-in has no gravity: a block stays where it is let go
            assert abs(q @ fwd - blk.x) < 1.5 and abs(q @ left - blk.y) < 1.5 and abs(q @ up - want_up) < 0.6, (name, q @ fwd, q @ left, q @ up)
        grips = [e[2] for e in a.log if e[0] == "grip"]
        assert [g for g in grips if 0 < g < 850] == [600, 450, 500], grips  # let go opening 10 mm past each side of the block: 40, 25 and 30 mm across
        assert s.state()["scene"]["plan"] == []
    finally:
        alive[0] = False
        s.close()
