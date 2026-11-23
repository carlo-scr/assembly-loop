"""The camera on pictures that are harder than flat paint: blocks with a light top and dark sides, shadows, grain,
JPEG, another white balance, a dim room, a lamp mirrored in a glossy top, blocks that touch. Still made-up pictures
(nobody has shown this code a real one), but each of these broke the camera as it was."""
import cv2
import numpy as np
import pytest

from assemblyloop.blocks import Catalogue, Kind, Seen
from assemblyloop.eyes import (
    V4L2,
    Camera,
    Setup,
    Table,
    _hue_groups,
    colour_of,
    crowd,
    find_blobs,
    gap_mm,
    look,
    make_out,
    make_out_all,
    measure_hues,
    settle,
    steady,
)

CAT = Catalogue([Kind(f"{c} {n}", shape, size, c) for c in ("red", "yellow", "green", "blue")
                 for n, shape, size in (("cube", "box", (25.4, 25.4, 25.4)), ("bar", "box", (57.0, 25.4, 25.4)), ("plank", "box", (57.0, 25.4, 12.7)), ("cylinder", "cylinder", (25.4, 57.0)))]).check()
TABLE = Table(np.eye(3), 100.0)
PAINTS = dict(red=(38, 42, 205), yellow=(35, 195, 230), green=(62, 150, 45), blue=(170, 95, 30))  # BGR, fully lit
LAMP = np.array([0.35, -0.45, 1.0]) / np.linalg.norm([0.35, -0.45, 1.0])
EYE = np.array([-250.0, 1250.0, 950.0])


def camera(size=(1920, 1080)):
    at = np.array([0.0, 450.0, 100.0])
    z = (at - EYE) / np.linalg.norm(at - EYE)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])
    return Camera(size, 0.62 * size[0], (-0.21, 0.05), cv2.Rodrigues(R)[0].ravel(), -R @ EYE)


def faces(rest, shape, u, v, a):
    """[(corners, the way it faces)] of every face of a block, a round side as many narrow ones."""
    a = np.radians(a)
    al, ac, up, c = np.array([np.cos(a), np.sin(a), 0.0]), np.array([-np.sin(a), np.cos(a), 0.0]), np.array([0.0, 0.0, 1.0]), np.array([u, v, TABLE.w0])
    if shape == "cylinder":
        r, ts = rest.short / 2, np.linspace(0.0, 2 * np.pi, 36, endpoint=False)
        e1, e2, axis, base = (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), up * rest.height, c) if rest.round else (ac, up, al * rest.long, c - al * rest.long / 2 + up * r)
        ring = np.array([r * (np.cos(t) * e1 + np.sin(t) * e2) for t in ts])
        out = [(base + ring, -axis), (base + axis + ring, axis)]
        return out + [(np.array([base + ring[i], base + ring[i - 1], base + ring[i - 1] + axis, base + ring[i] + axis]), ring[i] + ring[i - 1]) for i in range(len(ts))]
    P = lambda i, j, k: c + i * rest.long / 2 * al + j * rest.short / 2 * ac + k * rest.height * up  # noqa: E731
    out = [(np.array([P(-1, -1, 1), P(1, -1, 1), P(1, 1, 1), P(-1, 1, 1)]), up)]
    for s in (-1, 1):
        out += [(np.array([P(s, -1, 0), P(s, 1, 0), P(s, 1, 1), P(s, -1, 1)]), al * s), (np.array([P(-1, s, 0), P(1, s, 0), P(1, s, 1), P(-1, s, 1)]), ac * s)]
    return out


def picture(cam, blocks, seed=0, shadow=0.45, grain=4.0, jpeg=80, white=(1.0, 1.0, 1.0), gain=1.0, shine=0.0, mat=(110, 100, 95), stains=()):
    """`blocks` [(kind, which rest, u, v, angle)] as a camera would see them: the top of a block as bright as its
    paint, its sides darker the more they face away from the lamp, a soft shadow beside it, a mat that is brighter on
    one side than on the other, the grain of the sensor, the camera's own white balance, exposure and JPEG."""
    rng = np.random.default_rng(seed)
    w, h = cam.size
    im = np.empty((h, w, 3), np.float32)
    im[:] = mat
    im *= np.linspace(0.9, 1.1, w, dtype=np.float32)[None, :, None]
    for (x, y, r), tint in stains:
        blot = np.zeros((h, w), np.float32)
        cv2.circle(blot, (x, y), r, 1.0, -1)
        im += cv2.GaussianBlur(blot, (0, 0), 4.0)[..., None] * np.array(tint, np.float32)
    solids = [(CAT.kind(k), faces(CAT.kind(k).rests()[ri], CAT.kind(k).shape, u, v, a)) for k, ri, u, v, a in sorted(blocks, key=lambda b: -np.hypot(b[2] - EYE[0], b[3] - EYE[1]))]
    dark = np.zeros((h, w), np.float32)
    for _, fs in solids:
        pts = np.concatenate([f for f, _ in fs])
        down = pts - LAMP * ((pts[:, 2] - TABLE.w0) / LAMP[2])[:, None]
        cv2.fillConvexPoly(dark, np.round(cv2.convexHull(cam.project(np.concatenate([down, pts])).astype(np.float32))).astype(np.int32), 1.0, cv2.LINE_AA)
    im *= (1.0 - shadow * cv2.GaussianBlur(dark, (0, 0), 3.0))[..., None]
    for kind, fs in solids:
        for pts, n in fs:
            n = n / np.linalg.norm(n)
            if n @ (EYE - pts.mean(0)) <= 0:
                continue
            lit = min(1.0, 0.45 + 0.55 * max(0.0, float(n @ LAMP)) / LAMP[2])
            px = cam.project(pts)
            cv2.fillConvexPoly(im, np.round(px * 16).astype(np.int32), tuple(float(c) * lit for c in PAINTS[kind.colour]), cv2.LINE_AA, 4)
            if shine and n[2] > 0.9:  # the lamp mirrored in the top: a patch of it is white
                glow = np.zeros((h, w), np.uint8)
                cv2.circle(glow, tuple(int(c) for c in px.mean(0) + rng.uniform(-3, 3, 2)), max(3, int(shine * np.sqrt(cv2.contourArea(px.astype(np.float32))))), 1, -1)
                top = np.zeros((h, w), np.uint8)
                cv2.fillConvexPoly(top, np.round(px).astype(np.int32), 1)
                im[(glow & top) > 0] = 248.0
    im = (im + 8.0) * np.array(white, np.float32) * gain
    im += rng.normal(0.0, 1.0, im.shape).astype(np.float32) * grain
    im = np.clip(im, 0, 255).astype(np.uint8)
    return cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, jpeg])[1], cv2.IMREAD_COLOR) if jpeg else im


def check(seen, truth, mm=2.5, deg=6.0, square_deg=12.0):
    """Every block of the truth was seen as what it is, where it is; nothing else was seen."""
    assert len(seen) == len(truth), ([(s.name, s.rest.name) for s in seen], truth)
    for kind, ri, u, v, a in truth:
        want = CAT.kind(kind).rests()[ri]
        s = min(seen, key=lambda s: np.hypot(s.uv[0] - u, s.uv[1] - v))
        assert (s.kind.name, s.rest.name) == (kind, want.name) and np.hypot(s.uv[0] - u, s.uv[1] - v) < mm, (kind, want.name, s.kind.name, s.rest.name, s.uv, s.margin, s.rival)
        if want.turns:
            d = abs(s.angle - a) % want.turns
            assert min(d, want.turns - d) < (deg if want.turns == 180.0 else square_deg), (kind, s.angle, a)  # which way a square footprint is turned shows only faintly


APART = [("red bar", 0, -120.0, 380.0, 20.0), ("yellow cube", 0, 60.0, 420.0, 35.0), ("green plank", 0, -20.0, 560.0, 110.0), ("blue bar", 1, 120.0, 580.0, 0.0),
         ("yellow cylinder", 0, -140.0, 600.0, 0.0), ("blue cube", 0, 140.0, 360.0, 75.0)]


def test_shaded_blocks_with_shadows_grain_and_jpeg_are_made_out_and_none_is_called_crowded():
    cam = camera()
    seen, unknown, blobs = look(picture(cam, APART), cam, TABLE, CAT)
    assert not unknown and len(blobs) == len(APART)
    check(seen, APART)
    assert not any(s.crowded for s in seen) and all(s.score > 0.9 and s.rival for s in seen)


@pytest.mark.parametrize("how", [dict(gain=0.5), dict(gain=1.4), dict(white=(0.78, 1.0, 1.22)), dict(white=(1.15, 1.0, 0.87)), dict(mat=(128, 104, 88))],
                         ids=["a dim room", "a bright room", "warm light", "cool light", "a mat that is not grey"])
def test_the_thresholds_follow_the_mat_into(how):
    cam = camera()
    seen, unknown, _ = look(picture(cam, APART, **how), cam, TABLE, CAT)
    assert not unknown
    check(seen, APART, deg=9.0, square_deg=25.0)  # 27 pixels of cube do not say much about how it is turned (a 4K picture says twice as much)


def test_with_fixed_thresholds_a_dim_room_loses_the_blocks_as_it_used_to():
    cam = camera()
    im = picture(cam, APART, gain=0.5)
    assert len([b for b in find_blobs(im) if not b.cut]) == len(APART)
    assert sum(b.area for b in find_blobs(im, s_min=115, v_min=70)) < 0.5 * sum(b.area for b in find_blobs(im))


def test_in_a_small_cut_out_half_filled_by_a_block_the_mat_is_still_told_from_the_paint():
    cam = camera()
    im = picture(cam, [("red bar", 0, 0.0, 450.0, 30.0)])
    whole = find_blobs(im)[0]
    crop = im[whole.y0 - 9:whole.y0 + whole.mask.shape[0] + 9, whole.x0 - 9:whole.x0 + whole.mask.shape[1] + 9]
    small = find_blobs(crop)
    assert whole.area > 0.3 * crop.shape[0] * crop.shape[1] and len(small) == 1 and small[0].colour == "red" and abs(small[0].area - whole.area) < 0.06 * whole.area
    assert find_blobs(np.full((200, 300, 3), (40, 40, 220), np.uint8)) != []  # nothing but paint: no mat to go by, the fixed thresholds decide


def test_one_paint_is_one_blob_wherever_the_borders_of_the_colour_names_are():
    hist = np.zeros(180)
    hist[18:27] = 100.0  # one paint, across the border between orange (up to 22) and yellow
    group, mids = _hue_groups(hist)
    assert len(mids) == 1 and 20 <= mids[0] <= 24 and len(set(group[18:27])) == 1
    hist[100:108] = 60.0  # and another, far away
    group, mids = _hue_groups(hist)
    assert len(mids) == 2 and group[20] != group[104]
    hist = np.zeros(180)
    hist[[176, 177, 178, 179, 0, 1, 2, 3]] = 50.0  # red lies on both sides of where the hues start again
    assert len(_hue_groups(hist)[1]) == 1
    # a yellow bar in warm light: its top is yellow, its shaded sides are orange by the table of names
    cam = camera()
    im = picture(cam, [("yellow bar", 0, 0.0, 450.0, 30.0)], white=(0.78, 1.0, 1.22))
    plain, named = find_blobs(im), find_blobs(im, colours={"red", "yellow", "green", "blue"})
    assert len(plain) == len(named) == 1 and named[0].colour == "yellow" and abs(named[0].area - plain[0].area) < 5
    assert colour_of(20.0) == "orange" and colour_of(20.0, colours={"red", "yellow"}) == "yellow" and colour_of(12.0, colours={"red", "yellow"}) == "red"
    assert colour_of(145.0, colours={"blue", "red"}) == "purple"  # far from every colour there is: it keeps its own name
    assert colour_of(20.0, hues={"yellow": 21.0, "red": 3.0}) == "yellow" and colour_of(60.0, hues={"yellow": 21.0, "red": 3.0}) == "green"


def test_the_hues_of_the_paints_can_be_measured_and_are_kept_with_the_camera(tmp_path):
    cam = camera()
    one_each = [("red cube", 0, -120.0, 400.0, 0.0), ("yellow cube", 0, -40.0, 400.0, 0.0), ("green cube", 0, 40.0, 400.0, 0.0), ("blue cube", 0, 120.0, 400.0, 0.0)]
    hues = measure_hues(picture(cam, one_each))
    assert sorted(hues) == ["blue", "green", "red", "yellow"] and 20 < hues["yellow"] < 36 and 95 < hues["blue"] < 125
    setup = Setup(cam, TABLE, 112.7, np.zeros(3), 0.0, [(-170.0, 340.0), (170.0, 340.0), (170.0, 620.0), (-170.0, 620.0)])
    old = setup.to_dict()
    assert "hues" not in old and Setup.from_dict(old).hues == {}  # a camera.json written before there were hues
    setup.hues = hues
    setup.save(tmp_path / "camera.json")
    assert Setup.load(tmp_path / "camera.json").hues == pytest.approx(hues)
    seen, unknown, _ = look(picture(cam, APART, white=(0.78, 1.0, 1.22)), cam, TABLE, CAT, hues=measure_hues(picture(cam, one_each, white=(0.78, 1.0, 1.22)), colours={"red", "yellow", "green", "blue"}))
    assert not unknown
    check(seen, APART, deg=9.0, square_deg=25.0)


def test_a_lamp_mirrored_in_a_glossy_top_does_not_cost_the_block():
    cam = camera((3840, 2160))
    truth = [("red bar", 0, -60.0, 420.0, 20.0), ("blue cube", 0, 60.0, 520.0, 40.0), ("green cylinder", 0, 0.0, 600.0, 0.0)]
    im = picture(cam, truth, shine=0.3)
    blobs = find_blobs(im)
    assert len(blobs) == 3 and all(b.area > 0.97 * cv2.contourArea(cv2.convexHull(cv2.findNonZero(b.mask.astype(np.uint8)))) for b in blobs)  # mended: no hole, no bite
    seen, unknown, _ = look(im, cam, TABLE, CAT)
    assert not unknown
    check(seen, truth)


def test_a_stain_on_the_mat_is_not_a_block():
    cam = camera()
    im = picture(cam, APART[:2], stains=[((900, 500, 22), (30, -8, -18)), ((1100, 620, 18), (-20, 12, 26))])
    seen, unknown, blobs = look(im, cam, TABLE, CAT)
    assert len(blobs) == 2 and not unknown
    check(seen, APART[:2])


TOUCHING = [("red bar", 0, -100.0, 420.0, 20.0), ("red cube", 0, -100.0 + 41.3 * np.cos(np.radians(20.0)) + 6.0 * np.sin(np.radians(20.0)), 420.0 + 41.3 * np.sin(np.radians(20.0)) - 6.0 * np.cos(np.radians(20.0)), 50.0),
            ("green plank", 0, 60.0, 520.0, 100.0), ("green cube", 0, 87.0, 532.0, 40.0), ("blue cube", 0, 100.0, 380.0, 0.0)]


def test_two_blocks_of_one_colour_that_touch_are_made_out_as_two_and_marked_crowded():
    cam = camera()
    im = picture(cam, TOUCHING)
    seen, unknown, blobs = look(im, cam, TABLE, CAT)
    assert len(blobs) == 3 and not unknown
    check(seen, TOUCHING, mm=3.0, square_deg=25.0)
    assert sorted(s.name for s in seen if s.crowded) == ["green cube", "green plank", "red bar", "red cube"] and not [s for s in seen if s.kind.name == "blue cube"][0].crowded
    red = [b for b in blobs if b.colour == "red"][0]
    one, why = make_out(red, cam, TABLE, CAT)  # as ONE block it is still not guessed at
    assert one is None and "fits only" in why
    two, why = make_out_all(red, cam, TABLE, CAT, picture=im)
    assert [s.kind.name for s in sorted(two, key=lambda s: s.uv[0])] == ["red bar", "red cube"] and all(s.crowded and s.score > 0.84 for s in two) and why == ""
    assert not make_out_all(red, cam, TABLE, CAT, most=1)[0] and not look(im, cam, TABLE, CAT, most=1)[0][1:]  # told to see one block per blob, it sees the blue cube only


def test_three_blocks_of_one_colour_in_a_heap_are_made_out_or_said_not_to_be_but_never_fewer_are_guessed():
    cam = camera()
    heap = [("yellow bar", 0, 0.0, 450.0, 0.0), ("yellow cube", 0, 10.0, 477.0, 30.0), ("yellow cube", 0, -45.0, 440.0 - 27.0, 60.0)]
    seen, unknown, _ = look(picture(cam, heap), cam, TABLE, CAT)
    assert (len(seen) == 3 and not unknown) or (not seen and len(unknown) == 1)
    if seen:
        check(seen, heap, mm=4.0, deg=12.0)
        assert all(s.crowded for s in seen)


def test_blocks_within_reach_of_the_jaws_of_one_another_are_marked_crowded():
    bar, cube = CAT.kind("red bar"), CAT.kind("blue cube")
    a = Seen(bar, bar.rests()[0], (0.0, 0.0), 0.0)
    b = Seen(cube, cube.rests()[0], (0.0, 12.7 + 12.7 + 20.0), 0.0)  # 20 mm beside the bar
    c = Seen(cube, cube.rests()[0], (28.5 + 12.7 + 55.0, 0.0), 45.0)  # 55 mm beyond its end, turned: its corner is nearer
    assert not (a.crowded or b.crowded or c.crowded) and a.margin == 1.0 and a.rival == ""  # what a `Seen` made by older code says
    assert gap_mm(a, b) == pytest.approx(20.0, abs=0.01) and gap_mm(a, c) == pytest.approx(55.0 - 12.7 * (np.sqrt(2.0) - 1.0), abs=0.01)
    assert gap_mm(a, Seen(cube, cube.rests()[0], (20.0, 5.0), 10.0)) == 0.0  # in one another: no gap
    close = crowd([a, b, c], clear_mm=30.0)
    assert [(p.kind.name, q.kind.name, round(d)) for p, q, d in close] == [("red bar", "blue cube", 20)] and a.crowded and b.crowded and not c.crowded
    assert len(crowd([a, b, c], clear_mm=60.0)) == 2 and c.crowded
    cam = camera()
    near = [("red bar", 0, -60.0, 450.0, 0.0), ("blue cube", 0, -60.0, 450.0 + 25.4 + 12.0, 0.0), ("green cube", 0, 100.0, 520.0, 0.0)]
    seen, _, _ = look(picture(cam, near), cam, TABLE, CAT)
    assert sorted(s.kind.name for s in seen if s.crowded) == ["blue cube", "red bar"]
    seen, _, _ = look(picture(cam, near), cam, TABLE, CAT, clear_mm=5.0)
    assert not any(s.crowded for s in seen)
    im = picture(cam, near[::2])
    at = cam.project([TABLE.point(-60.0, 450.0 + 12.7 + 14.0, TABLE.w0), TABLE.point(100.0, 440.0, TABLE.w0)]).astype(int)
    cv2.circle(im, tuple(int(v) for v in at[0]), 13, (170, 60, 150), -1)  # something purple, which no block is, 14 mm beside the bar
    cv2.circle(im, tuple(int(v) for v in at[1]), 13, (170, 60, 150), -1)  # and another far from everything
    seen, unknown, _ = look(im, cam, TABLE, CAT)
    assert len(unknown) == 2 and [(s.kind.name, s.crowded) for s in sorted(seen, key=lambda s: s.uv[0])] == [("red bar", True), ("green cube", False)]  # what was not made out is in the way too


def test_a_block_partly_behind_one_of_another_colour_is_still_found_where_it_is():
    cam = camera()
    truth = [("green bar", 0, 0.0, 430.0, 0.0), ("red bar", 1, 24.0, 430.0 + 12.7 + 12.7 + 1.0, 0.0)]  # the red one stands on its end in front of one end of the green one, as the camera sees it
    im = picture(cam, truth)
    seen, unknown, blobs = look(im, cam, TABLE, CAT)
    assert not unknown
    check(seen, truth, mm=3.0)
    assert all(s.crowded for s in seen)
    green = [b for b in blobs if b.colour == "green"][0]
    blind, _ = make_out(green, cam, TABLE, CAT)  # without being told that something stands in front of it, the bite counts against every outline
    told = [s for s in seen if s.colour == "green"][0]
    assert blind is None or blind.score < told.score - 0.03


def test_a_bar_on_its_end_is_told_from_one_lying_towards_the_camera_by_its_faces_and_says_how_sure_it_is():
    cam = camera((2560, 1440))
    towards = float(np.degrees(np.arctan2(EYE[1] - 450.0, EYE[0] - 0.0)))  # the direction from the block to the camera, on the table
    for truth in ([("blue bar", 1, 0.0, 450.0, towards)], [("blue bar", 0, 0.0, 450.0, towards)], [("green cylinder", 0, 0.0, 450.0, 0.0)], [("green cylinder", 1, 0.0, 450.0, towards)]):
        im = picture(cam, truth)
        seen, unknown, blobs = look(im, cam, TABLE, CAT)
        assert not unknown
        check(seen, truth)
        outline_only, _ = make_out(blobs[0], cam, TABLE, CAT)  # the same blob without the picture: nothing but its outline
        assert outline_only is not None and outline_only.rival and 0.0 <= outline_only.margin < 0.1, (truth, outline_only)
    apart = look(picture(cam, [("red plank", 0, 0.0, 450.0, 30.0)]), cam, TABLE, CAT)[0][0]
    assert apart.margin > outline_only.margin and apart.rest.name == "flat"  # a plank lying flat looks like nothing else


def test_several_frames_make_one_steady_picture():
    cam = camera((960, 540))
    frames = [picture(cam, APART[:1], seed=i, grain=12.0, jpeg=0) for i in range(5)]
    clean = picture(cam, APART[:1], grain=0.0, jpeg=0).astype(float)
    one, five = np.abs(frames[0] - clean).mean(), np.abs(steady(frames) - clean).mean()
    assert five < 0.6 * one and steady(frames[:1]) is frames[0] and steady(frames).dtype == np.uint8
    assert V4L2("/dev/null", frames=3).frames == 3 and V4L2("/dev/null").frames == 1


def test_the_camera_is_fitted_twice_and_the_second_time_is_kept_only_if_it_is_no_worse():
    cam, bar = camera((1280, 720)), CAT.kind("green bar")
    rng = np.random.default_rng(3)
    places = [(u + rng.uniform(-8, 8), v + rng.uniform(-8, 8)) for u in (-150.0, -50.0, 50.0, 150.0) for v in (350.0, 480.0, 610.0)]
    pictures = [picture(cam, [("green bar", 0, u, v, 30.0)], seed=i, white=(0.9, 1.0, 1.1)) for i, (u, v) in enumerate(places)]
    xyz = [TABLE.point(u, v, TABLE.w0 + 12.7) for u, v in places]
    twice, line = settle(xyz, pictures, bar, np.eye(3), np.zeros(3))
    once, _ = settle(xyz, pictures, bar, np.eye(3), np.zeros(3), refine=False)
    assert twice.fit_mm <= once.fit_mm + 1e-9 and twice.fit_mm < 2.0 and "fitted to 12 places" in line and abs(twice.angle_ref - 30.0) < 3.0
    probe = np.array([TABLE.point(u, v, TABLE.w0 + h) for u in (-150.0, 0.0, 150.0) for v in (350.0, 480.0, 610.0) for h in (0.0, 57.0)])  # the table, and the top of a bar on its end
    for h in (0.0, 57.0):
        sub = probe[np.isclose(probe[:, 2], TABLE.w0 + h)]
        assert np.linalg.norm(twice.camera.on_plane(cam.project(sub), TABLE.axes[:, 2], TABLE.w0 + h) - sub, axis=1).max() < 3.0
