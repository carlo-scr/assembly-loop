"""The camera and the blocks, on pictures drawn by a made-up camera whose truth is known."""
import cv2
import numpy as np
import pytest

from assemblyloop.blocks import Catalogue, Kind, Seen, plan_tower
from assemblyloop.eyes import PAINT, Camera, Table, calibrate, draw, find_blobs, look, outline, solid
from assemblyloop.guard import Refused

CAT = Catalogue([Kind("green bar", "box", (75.0, 25.0, 25.0), "green"), Kind("yellow cylinder", "cylinder", (25.0, 75.0), "yellow"),
                 Kind("blue slab", "box", (60.0, 40.0, 15.0), "blue"), Kind("yellow cube", "box", (30.0, 30.0, 30.0), "yellow")]).check()
TABLE = Table(np.eye(3), 100.0)  # a level table 100 mm up


def camera(size=(1920, 1080)):
    """A camera 1.1 m from the middle of the mat, looking down at it at a slant, with a lens that bends."""
    eye, at = np.array([-250.0, 1250.0, 950.0]), np.array([0.0, 450.0, 100.0])
    z = (at - eye) / np.linalg.norm(at - eye)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])
    return Camera(size, 0.62 * size[0], (-0.21, 0.05), cv2.Rodrigues(R)[0].ravel(), -R @ eye)


def picture(cam, blocks):
    im = np.full((cam.size[1], cam.size[0], 3), (110, 100, 95), np.uint8)
    for kind, rest, u, v, a in blocks:
        hull = outline(cam, solid(CAT.kind(kind).rests()[rest], TABLE, u, v, a, CAT.kind(kind).shape))
        cv2.fillConvexPoly(im, np.round(hull * 16).astype(np.int32), PAINT[CAT.kind(kind).colour], shift=4)
    return im


def test_the_table_frame_is_right_handed_and_up_is_up():
    assert np.allclose(np.cross(TABLE.axes[:, 0], TABLE.axes[:, 1]), TABLE.axes[:, 2]) and np.allclose(TABLE.axes[:, 2], [0, 0, 1])
    assert np.allclose(TABLE.coords(TABLE.point(3.0, 4.0, 5.0)), [3.0, 4.0, 5.0])


def test_a_camera_is_fitted_to_places_on_one_plane_and_then_knows_other_heights():
    true, rng = camera(), np.random.default_rng(1)
    uv = [(u, v) for u in (-150.0, -50.0, 50.0, 150.0) for v in (330.0, 430.0, 530.0, 630.0)]
    xyz = np.array([TABLE.point(u, v, 112.5) for u, v in uv])
    cam = calibrate(true.project(xyz) + rng.normal(0.0, 0.4, (len(xyz), 2)), xyz, true.size)
    assert cam.rms < 1.0
    for w in (100.0, 137.5, 175.0):  # the table itself, and the tops of things up to 75 mm high
        probe = np.array([TABLE.point(u, v, w) for u, v in ((-120.0, 350.0), (0.0, 480.0), (130.0, 600.0))])
        back = cam.on_plane(true.project(probe), TABLE.axes[:, 2], w)
        assert np.abs(back - probe).max() < 2.5, (w, np.abs(back - probe).max())
    again = Camera.from_dict(cam.to_dict())
    assert np.allclose(again.project(xyz), cam.project(xyz))
    assert np.allclose(cam.scaled((3840, 2160)).project(xyz), 2.0 * cam.project(xyz), atol=1e-6)


def test_too_few_places_or_places_on_a_line_are_refused():
    true = camera()
    line = np.array([TABLE.point(u, 400.0 + 0.01 * u, 112.5) for u in np.linspace(-150, 150, 10)])
    with pytest.raises(Refused):
        calibrate(true.project(line), line, true.size)
    with pytest.raises(Refused):
        calibrate(true.project(line[:5]), line[:5], true.size)


def test_blocks_are_made_out_where_they_are_whatever_way_they_lie():
    cam = camera()
    truth = [("green bar", 0, -120.0, 380.0, 20.0), ("green bar", 1, 60.0, 420.0, 0.0), ("yellow cylinder", 0, -20.0, 560.0, 0.0),
             ("blue slab", 0, 120.0, 580.0, 130.0), ("yellow cube", 0, -140.0, 600.0, 40.0), ("yellow cylinder", 1, 140.0, 360.0, 75.0)]
    seen, unknown, blobs = look(picture(cam, truth), cam, TABLE, CAT)
    assert len(blobs) == len(truth) and not unknown
    for kind, rest, u, v, a in truth:
        want = CAT.kind(kind).rests()[rest]
        s = min(seen, key=lambda s: np.hypot(s.uv[0] - u, s.uv[1] - v))
        assert np.hypot(s.uv[0] - u, s.uv[1] - v) < 1.5 and s.score > 0.93, (kind, s)
        assert abs(s.rest.height - want.height) < 0.5 and s.rest.round == want.round, (kind, want.name, s.kind.name, s.rest.name)
        if want.turns:
            d = abs(s.angle - a % want.turns) % want.turns
            assert min(d, want.turns - d) < 3.0, (kind, s.angle, a)
    assert sorted(s.name for s in seen if s.kind.name == "green bar") == ["green bar 1", "green bar 2"]
    assert draw(picture(cam, truth), blobs, seen, unknown, cam, TABLE).shape[1] == 1280


def test_what_is_not_in_the_catalogue_or_cut_off_is_said_so_and_never_guessed():
    cam = camera()
    im = picture(cam, [("green bar", 0, 0.0, 450.0, 0.0)])
    cv2.circle(im, (300, 300), 40, PAINT["purple"], -1)  # nothing purple is in the catalogue
    cv2.rectangle(im, (0, 600), (30, 700), PAINT["blue"], -1)  # at the edge of the picture
    big = outline(cam, solid(CAT.kind("blue slab").rests()[0], TABLE, 150.0, 600.0, 0.0, "box")) * [1.6, 1.6] - [500, 250]
    cv2.fillConvexPoly(im, np.round(big).astype(np.int32), PAINT["blue"])  # blue, but far too large for a slab
    seen, unknown, _ = look(im, cam, TABLE, CAT)
    assert [s.kind.name for s in seen] == ["green bar"]
    whys = " | ".join(w for _, w in unknown)
    assert len(unknown) == 3 and "no purple block" in whys and "cut off" in whys and "fits only" in whys
    region = np.zeros(im.shape[:2], bool)
    region[:, 700:] = True
    assert all(b.x0 >= 700 for b in find_blobs(im, region))


def test_the_tower_is_widest_first_and_says_what_it_leaves_out():
    k = CAT.kind
    seen = [Seen(k("green bar"), k("green bar").rests()[0], (0.0, 0.0), name="bar"), Seen(k("blue slab"), k("blue slab").rests()[0], (50.0, 0.0), name="slab"),
            Seen(k("yellow cylinder"), k("yellow cylinder").rests()[0], (100.0, 0.0), name="cyl up"), Seen(k("yellow cylinder"), k("yellow cylinder").rests()[1], (150.0, 0.0), name="cyl lying"),
            Seen(k("yellow cube"), k("yellow cube").rests()[0], (200.0, 0.0), name="cube")]
    levels, out = plan_tower(seen, CAT)
    assert [lv["block"].name for lv in levels] == ["slab", "cube", "bar", "cyl up"]
    assert [lv["base_mm"] for lv in levels] == [0.0, 15.0, 45.0, 70.0] and levels[-1]["top_mm"] == 145.0
    assert [lv["hold_mm"] for lv in levels] == [0.0, 0.0, 0.0, 35.0]  # the tall cylinder is taken near its top
    assert [(s.name, "roll" in why) for s, why in out] == [("cyl lying", True)]
    levels, out = plan_tower(seen, CAT, max_mm=60.0)
    assert [lv["block"].name for lv in levels] == ["slab", "cube"] and len(out) == 3
    wide = Catalogue([Kind("plank", "box", (120.0, 90.0, 10.0))]).check()
    assert plan_tower([Seen(wide.kinds[0], wide.kinds[0].rests()[0], (0.0, 0.0))], wide)[0] == []
    with pytest.raises(Refused):
        Catalogue([Kind("x", "box", (10.0, 10.0))]).check()
    assert Catalogue.from_dict(CAT.to_dict()).kinds == CAT.kinds


def test_fingers_whose_gap_is_wider_than_the_gripper_says_are_closed_further():
    from assemblyloop.cycle import Params
    from assemblyloop.spots import demo_spots, plan_move

    cat = Catalogue.from_dict({**CAT.to_dict(), "jaw_offset_mm": 4.0})
    assert cat.jaw_offset_mm == 4.0 and Catalogue.from_dict(cat.to_dict()).jaw_offset_mm == 4.0 and CAT.jaw_offset_mm == 0.0
    with pytest.raises(Refused):
        Catalogue.from_dict({**CAT.to_dict(), "jaw_offset_mm": 40.0})
    sp = demo_spots()
    close = lambda P: next(s for s in plan_move(sp, "A", "B", sp.above(sp.grasp("A")), P, grip_mm=31.0) if s["name"].startswith("close on it"))  # noqa: E731
    assert close(Params())["counts"] == 285  # 2.5 mm under 31 mm, as the gripper counts
    assert close(Params(jaw_offset_mm=4.0))["counts"] == 245  # the same gap between fingers that stand 4 mm wider apart than it says


def test_a_block_whose_colour_spills_over_its_edges_is_still_made_out_with_the_spill_allowed():
    """On the rig the camera saw 25.4 x 12.7 mm planks as about 31 x 17.5: their colour spills ~2.5 mm over each edge.
    With the exact size in the catalogue the outline then covers too little of the patch, unless the spill is allowed."""
    from assemblyloop.blocks import Catalogue, Kind
    from assemblyloop.eyes import PAINT, Blob, Camera, Table, make_out, outline, solid

    R = cv2.Rodrigues(np.array([np.pi, 0.0, 0.0]))[0]  # looking straight down from 600 mm
    cam = Camera((1280, 960), 1400.0, (0.0, 0.0), cv2.Rodrigues(R)[0].ravel(), -R @ np.array([0.0, 0.0, 600.0]))
    table = Table(np.eye(3), 0.0)
    kind = Kind("blue plank", "box", (88.9, 25.4, 12.7), "blue")
    wide = Kind("blue plank", "box", (88.9 + 5.0, 25.4 + 5.0, 12.7 + 2.5), "blue")  # what the camera sees: 2.5 mm more on each side
    im = np.full((960, 1280, 3), 100, np.uint8)
    hull = outline(cam, solid(wide.rests()[0], table, 0.0, 0.0, 30.0, "box"))
    cv2.fillConvexPoly(im, np.round(hull).astype(np.int32), PAINT["blue"])
    mask = np.all(im == np.array(PAINT["blue"], np.uint8), axis=2)
    ys, xs = np.nonzero(mask)
    blob = Blob("blue", mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1], int(xs.min()), int(ys.min()), (float(xs.mean()), float(ys.mean())))
    exact, allowed = Catalogue([kind]), Catalogue([kind], seen_wider_mm=2.5)
    s0, why = make_out(blob, cam, table, exact)
    s1, _ = make_out(blob, cam, table, allowed)
    assert s0 is None and "fits only" in why  # the exact outline alone covers too little of what is seen
    assert s1 is not None and s1.score > 0.9 and abs(s1.uv[0]) < 1.0 and abs(s1.uv[1]) < 1.0 and abs((s1.angle - 30.0 + 90) % 180 - 90) < 2.0
