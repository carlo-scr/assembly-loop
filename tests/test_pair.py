"""Two arms in one space, on a made-up rig whose truth is known: two bases 600 mm apart, rolled 45 degrees each way
like shoulders, over a table. Nothing here has run on the arms."""
import time
from pathlib import Path

import numpy as np
import pytest

from assemblyloop import pair
from assemblyloop.arms import flange_mm, joints_for, matrix_rpy, rpy_matrix
from assemblyloop.guard import Refused
from assemblyloop.pair import Frame, Unknown, Watch, arm_capsules, capsule_distance, clearance, fit_frames, split_plane, why_across
from assemblyloop.session import mount_down
from assemblyloop.spots import Spots


def rx(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


# the room: x forward, y to the left, z up. Each base: its orientation in the room and where it is.
ROOM = {"right": (rx(45.0), np.array([0.0, -300.0, 0.0])), "left": (rx(-45.0), np.array([0.0, 300.0, 0.0]))}
DOWN = {side: mount_down(side) for side in ROOM}
TRUE = Frame(ROOM["left"][0].T @ ROOM["right"][0], ROOM["left"][0].T @ (ROOM["right"][1] - ROOM["left"][1]), 0.0, 0, "marks")  # right to left
TOOL_DOWN = np.diag([1.0, -1.0, -1.0])  # a gripper that points straight down, in the room


def in_base(side, p_room):
    R, o = ROOM[side]
    return (np.asarray(p_room, float) - o) @ R


def in_room(side, p_base):
    R, o = ROOM[side]
    return np.asarray(p_base, float) @ R.T + o


def marks(n=6, rng=None, noise=0.0):
    """Marks on the table (420 mm under the bases) as each arm's fingertips would report them."""
    grid = np.array([(x, y, -420.0) for x in (330.0, 560.0) for y in (-170.0, 20.0, 190.0)])[:n]
    a, b = in_base("right", grid), in_base("left", grid)
    if rng is not None:
        a, b = a + rng.normal(0.0, noise, a.shape), b + rng.normal(0.0, noise, b.shape)
    return a, b


_POSTURES = {}


def posture(side, flange_room):
    """Joint angles that put this arm's flange at a point of the room, the gripper pointing straight down."""
    key = (side, tuple(flange_room))
    if key not in _POSTURES:
        p, R, q = in_base(side, flange_room), ROOM[side][0].T @ TOOL_DOWN, None
        for _ in range(3):
            q = joints_for(p, R, q)
        assert np.linalg.norm(flange_mm(q) - p) < 0.5
        _POSTURES[key] = q
    return _POSTURES[key]


FAR = {"right": (420.0, -330.0, -200.0), "left": (420.0, 330.0, -200.0)}  # each over its own side of the table


# ---------------------------------------------------------------------------------------------------- the frame
def test_the_made_up_rig_agrees_with_how_the_program_says_the_arms_are_mounted():
    for side in ROOM:
        assert np.allclose(ROOM[side][0].T @ [0.0, 0.0, -1.0], DOWN[side])


def test_a_frame_moves_points_directions_and_orientations_and_can_be_undone_chained_and_saved():
    p = np.array([[400.0, -50.0, 120.0], [10.0, 20.0, 30.0]])
    assert np.allclose(in_room("left", TRUE.point(p)), in_room("right", p))
    assert np.allclose(TRUE.inverse().point(TRUE.point(p)), p) and np.allclose(TRUE.direction([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0])
    rpy = np.array([170.0, 12.0, -40.0])
    assert np.allclose(ROOM["left"][0] @ rpy_matrix(TRUE.rpy(rpy)), ROOM["right"][0] @ rpy_matrix(rpy))  # the same orientation in the room
    pose = TRUE.pose(np.r_[p[0], rpy])
    assert np.allclose(pose[:3], TRUE.point(p[0])) and np.allclose(rpy_matrix(pose[3:]), TRUE.R @ rpy_matrix(rpy))
    other = Frame(rpy_matrix([10.0, 20.0, 30.0]), [5.0, 6.0, 7.0], 3.0, 4, "cameras")
    both = TRUE.then(other)
    assert np.allclose(both.point(p), other.point(TRUE.point(p))) and both.how == "marks, then cameras" and both.rms_mm == pytest.approx(3.0)
    assert np.allclose(TRUE.then(TRUE.inverse()).point(p), p)
    again = Frame.from_dict(fit_frames(*marks(), names=list("abcdef")).to_dict())
    assert np.allclose(again.R, TRUE.R) and np.allclose(again.t, TRUE.t) and again.n == 6 and again.how == "marks" and again.names == list("abcdef")
    assert np.isnan(Frame.from_dict(Frame(np.eye(3), np.zeros(3)).to_dict()).rms_mm)


def test_a_saved_frame_that_mirrors_or_stretches_is_refused():
    good = TRUE.to_dict()
    for R in (np.diag([1.0, 1.0, -1.0]), 1.01 * np.eye(3), np.ones((3, 3)), np.eye(2)):
        with pytest.raises(Refused):
            Frame.from_dict({**good, "R": R.tolist()})
    with pytest.raises(Refused):
        Frame.from_dict({"t": [0, 0, 0]})


def test_the_frame_is_recovered_from_noisy_marks_that_all_lie_on_the_table():
    a, b = marks(6, np.random.default_rng(3), noise=0.3)
    f = fit_frames(a, b)
    assert pair.turn_between_deg(f.R, TRUE.R) < 0.25 and f.rms_mm < 1.0 and f.n == 6 and len(f.residuals_mm) == 6 and f.how == "marks"
    probe = in_base("right", [(300.0, -300.0, -420.0), (600.0, 300.0, -420.0), (450.0, 0.0, -100.0)])  # also well above the table
    assert np.abs(f.point(probe) - TRUE.point(probe)).max() < 1.5
    exact = fit_frames(*(m[[0, 2, 4]] for m in marks()))  # three marks are enough when they are right
    assert np.allclose(exact.R, TRUE.R) and np.allclose(exact.t, TRUE.t) and exact.rms_mm < 1e-9


def test_too_few_marks_marks_on_a_line_and_marks_close_together_are_refused():
    a, b = marks()
    with pytest.raises(Refused, match="at least 3"):
        fit_frames(a[:2], b[:2])
    line = np.array([(300.0 + 60.0 * i, -200.0 + 80.0 * i, -420.0 + 0.5 * (i % 2)) for i in range(5)])
    with pytest.raises(Refused, match="one line"):
        fit_frames(in_base("right", line), in_base("left", line))
    near = np.array([(400.0, 0.0, -420.0), (430.0, 0.0, -420.0), (400.0, 30.0, -420.0), (430.0, 30.0, -420.0)])
    with pytest.raises(Refused, match="close together"):
        fit_frames(in_base("right", near), in_base("left", near))
    with pytest.raises(Refused):
        fit_frames(a, b[:5])
    with pytest.raises(Refused):
        fit_frames(a, b, names=["only one"])


def test_one_wrong_mark_is_named_and_marks_touched_in_another_order_are_refused():
    a, b = marks(6, np.random.default_rng(5), noise=0.3)
    wrong = b.copy()
    wrong[4] += [6.0, -8.0, 0.0]  # 10 mm off: within the whole fit's limit, but not like the others
    with pytest.raises(Refused, match="E does not fit the others"):
        fit_frames(a, wrong, names=list("ABCDEF"))
    swapped = b.copy()
    swapped[[1, 2]] = swapped[[2, 1]]
    with pytest.raises(Refused):
        fit_frames(a, swapped)
    with pytest.raises(Refused, match="do not fit"):
        fit_frames(a[:4], swapped[:4])
    assert fit_frames(a, b, names=list("ABCDEF")).rms_mm < 1.0  # and the same marks, right, pass


def test_fingertips_and_not_flanges_are_what_two_arms_have_in_common_at_a_mark():
    mark = np.array([450.0, 30.0, -420.0])
    tilts = {"right": rpy_matrix([180.0, 0.0, 20.0]), "left": rpy_matrix([165.0, 10.0, -30.0])}  # in the room: each arm leans its own way
    pose = {}
    for side, T in tilts.items():
        flange = mark - 172.0 * T[:, 2]
        pose[side] = np.r_[in_base(side, flange), matrix_rpy(ROOM[side][0].T @ T)]
    assert np.allclose(TRUE.point(pair.fingertip(pose["right"])), pair.fingertip(pose["left"]))
    assert np.linalg.norm(TRUE.point(pose["right"][:3]) - pose["left"][:3]) > 40.0
    assert np.allclose(pair.fingertip(pose["right"], 0.0), pose["right"][:3])


# ---------------------------------------------------------------------------------------------------- the mounts
def test_the_mounts_of_this_rig_say_a_quarter_turn_about_forward_and_a_right_fit_agrees():
    R = pair.mount_turn(DOWN["right"], DOWN["left"])
    assert np.allclose(R, rx(90.0)) and np.allclose(R, TRUE.R) and np.allclose(pair.mount_turn(DOWN["left"], DOWN["right"]), R.T)
    assert pair.turn_between_deg(np.eye(3), rx(90.0)) == pytest.approx(90.0) and pair.turn_between_deg(R, R) == pytest.approx(0.0, abs=1e-5)
    assert np.allclose(pair.mount_turn(mount_down("right", 0.0), mount_down("left", 0.0)), np.eye(3))  # two upright arms
    f = fit_frames(*marks(6, np.random.default_rng(7), noise=0.3))
    assert pair.check_against_mounts(f, DOWN["right"], DOWN["left"]) < 1.0
    with pytest.raises(Refused):
        pair.mount_turn([1.0, 0.0, 0.0], DOWN["left"])


def test_marks_in_a_rectangle_touched_the_other_way_round_fit_perfectly_and_only_the_mounts_notice():
    rect = np.array([(350.0, -150.0, -420.0), (350.0, 150.0, -420.0), (550.0, 150.0, -420.0), (550.0, -150.0, -420.0)])
    a, b = in_base("right", rect), in_base("left", rect)[[1, 0, 3, 2]]  # the left arm went round the other way
    f = fit_frames(a, b)
    assert f.rms_mm < 1e-6  # nothing in the marks themselves gives it away
    with pytest.raises(Refused, match="different order"):
        pair.check_against_mounts(f, DOWN["right"], DOWN["left"])


# ---------------------------------------------------------------------------------------------------- handing over
def right_spots():
    s = Spots(side="right", lift=70.0)
    for name, (x, y, yaw) in dict(A=(400.0, -60.0, 0.0), B=(470.0, 40.0, 90.0), T=(520.0, -20.0, 30.0)).items():
        T = TOOL_DOWN @ rpy_matrix([0.0, 0.0, yaw])  # pointing down, turned about its own axis
        s.add(name, np.r_[in_base("right", (x, y, -420.0 + 172.0 + 12.0)), matrix_rpy(ROOM["right"][0].T @ T)])
    return s


def test_spots_taught_with_one_arm_are_the_same_places_for_the_other_and_come_back_unchanged():
    s = right_spots()
    left = pair.spots_in(TRUE, s, "left")
    assert left.side == "left" and left.lift == 70.0 and list(left.at) == list(s.at)
    assert np.allclose(left.up, -DOWN["left"]) and np.allclose(s.up, -DOWN["right"])  # the gripper still points straight down
    for name in s.at:
        assert np.allclose(in_room("left", left.at[name]), in_room("right", s.at[name]))
        assert np.allclose(ROOM["left"][0] @ rpy_matrix(left.turn(name)), ROOM["right"][0] @ rpy_matrix(s.turn(name)))
        assert np.allclose(in_room("left", pair.fingertip(np.r_[left.at[name], left.turn(name)])), in_room("right", pair.fingertip(np.r_[s.at[name], s.turn(name)])))
    assert left.travel_w() - max(p @ left.up for p in left.at.values()) == pytest.approx(70.0)
    back = pair.spots_in(TRUE.inverse(), left, "right")
    for name in s.at:
        assert np.allclose(back.at[name], s.at[name]) and np.allclose(rpy_matrix(back.turn(name)), rpy_matrix(s.turn(name)))
    assert np.allclose(rpy_matrix(back.rpy), rpy_matrix(s.rpy))
    again = Spots.from_dict(left.to_dict())  # and it is an ordinary spots file
    assert again.side == "left" and np.allclose(again.at["B"], left.at["B"], atol=1e-3)


def right_setup():
    cv2 = pytest.importorskip("cv2")
    from assemblyloop.eyes import Camera, Setup, Table

    eye, at = in_base("right", (-250.0, 800.0, 500.0)), in_base("right", (450.0, 0.0, -420.0))
    z = (at - eye) / np.linalg.norm(at - eye)
    x = np.cross(z, -DOWN["right"])
    x /= np.linalg.norm(x)
    Rc = np.stack([x, np.cross(z, x), z])
    cam = Camera((1920, 1080), 0.62 * 1920, (-0.21, 0.05), cv2.Rodrigues(Rc)[0].ravel(), -Rc @ eye, 0.4)
    table = Table(ROOM["right"][0].T.copy(), -420.0 + 172.0)  # u forward, v left, w up, in the right arm's base frame
    rpy_ref = matrix_rpy(ROOM["right"][0].T @ TOOL_DOWN @ rpy_matrix([0.0, 0.0, 25.0]))
    return Setup(cam, table, table.w0 + 12.0, rpy_ref, 25.0, [(320.0, 180.0), (600.0, 180.0), (600.0, 480.0), (320.0, 480.0)], "right", 1.5, 123.0)


def test_a_table_point_shows_at_the_same_pixel_through_the_setup_handed_to_the_other_arm():
    a = right_setup()
    b = pair.setup_in(TRUE, a, "left")
    assert b.side == "left" and b.angle_ref == a.angle_ref and b.when == a.when and b.fit_mm == pytest.approx(a.fit_mm)
    assert np.allclose(b.table.axes[:, 2], -DOWN["left"]) and np.allclose(b.table.axes.T @ b.table.axes, np.eye(3))
    assert np.allclose(rpy_matrix(b.rpy_ref), TRUE.R @ rpy_matrix(a.rpy_ref))
    shift = b.table.coords(TRUE.point(a.table.point(0.0, 0.0, 0.0)))  # where the right arm's (0, 0, 0) of the table is for the left
    assert np.linalg.norm(shift) > 100.0 and b.w_grasp - b.table.w0 == pytest.approx(a.w_grasp - a.table.w0) and b.table.w0 == pytest.approx(a.table.w0 + shift[2])
    for u, v, h in ((350.0, 200.0, 0.0), (590.0, 470.0, 12.0), (460.0, 330.0, 75.0), (200.0, 100.0, 30.0)):
        p_a = a.table.point(u, v, a.table.w0 + h)
        p_b = TRUE.point(p_a)
        assert np.allclose(b.table.coords(p_b), np.array([u, v, a.table.w0 + h]) + shift)
        assert np.allclose(b.table.point(u + shift[0], v + shift[1], b.table.w0 + h), p_b)  # the same point, said the left arm's way
        assert np.allclose(b.camera.project([p_b]), a.camera.project([p_a]), atol=1e-6)
        back = b.camera.on_plane(a.camera.project([p_a]), b.table.axes[:, 2], b.table.w0 + h)[0]  # and from the pixel back onto the table
        assert np.allclose(back, p_b, atol=1e-4)
    for uv in ((330.0, 190.0), (460.0, 330.0), (310.0, 330.0), (610.0, 470.0), (590.0, 490.0), (0.0, 0.0)):
        assert a.inside(uv) == b.inside((uv[0] + shift[0], uv[1] + shift[1]))
    assert a.inside((460.0, 330.0)) and not a.inside((310.0, 330.0))
    assert np.array_equal(a.region((960, 540), tall_mm=75.0), b.region((960, 540), tall_mm=75.0)) and a.region((960, 540)).any()


def test_a_setup_handed_over_and_back_is_the_one_that_was_handed_over():
    from assemblyloop.eyes import Setup

    a = right_setup()
    noisy = Frame(TRUE.R, TRUE.t, 2.0, 6, "marks")
    assert pair.setup_in(noisy, a, "left").fit_mm == pytest.approx(3.5)  # the frame's own error is counted in
    back = Setup.from_dict(pair.setup_in(TRUE.inverse(), Setup.from_dict(pair.setup_in(TRUE, a, "left").to_dict()), "right").to_dict())
    assert np.allclose(back.table.axes, a.table.axes) and back.table.w0 == pytest.approx(a.table.w0) and back.w_grasp == pytest.approx(a.w_grasp)
    assert np.allclose(back.area, a.area) and np.allclose(rpy_matrix(back.rpy_ref), rpy_matrix(a.rpy_ref)) and back.side == "right"
    probe = [a.table.point(400.0, 300.0, a.table.w0 + 20.0)]
    assert np.allclose(back.camera.project(probe), a.camera.project(probe), atol=1e-6) and np.allclose(back.camera.tvec, a.camera.tvec, atol=1e-6)


def test_one_camera_fitted_from_both_arms_gives_the_frame_and_says_when_the_two_fits_disagree():
    from assemblyloop.eyes import Camera

    a = right_setup()
    b = pair.setup_in(TRUE, a, "left")
    f = pair.frame_from_cameras(a, b)
    assert np.allclose(f.R, TRUE.R, atol=1e-9) and np.allclose(f.t, TRUE.t, atol=1e-6) and f.how == "cameras" and f.n == 5 and f.rms_mm < 1e-3
    b.camera = Camera(b.camera.size, b.camera.f * 1.03, b.camera.k, b.camera.rvec, b.camera.tvec)  # the second fit got the lens 3% wrong
    assert pair.frame_from_cameras(a, b).rms_mm > 3.0
    b.camera = Camera((1000, 1000), b.camera.f, b.camera.k, b.camera.rvec, b.camera.tvec)  # not even the same shape of picture: no estimate
    assert np.isnan(pair.frame_from_cameras(a, b).rms_mm)


# ---------------------------------------------------------------------------------------------------- capsules
def test_the_distance_between_capsules_agrees_with_points_sampled_along_them():
    rng = np.random.default_rng(11)
    s = np.linspace(0.0, 1.0, 301)[:, None]
    for i in range(150):
        a1, a2 = rng.uniform(-200.0, 200.0, (2, 3))
        b1, b2 = a1 + rng.uniform(-150.0, 150.0, 3), a2 + rng.uniform(-150.0, 150.0, 3)
        if i % 5 == 1:
            b1 = a1.copy()  # a ball
        if i % 5 == 2:
            b2 = a2 + (b1 - a1) * rng.uniform(-1.5, 1.5)  # parallel, also to a ball
        if i % 25 == 3:
            b1, b2 = a1.copy(), a2.copy()  # two balls
        r1, r2 = rng.uniform(0.0, 60.0, 2)
        P, Q = a1 + s * (b1 - a1), a2 + s * (b2 - a2)
        brute = float(np.linalg.norm(P[:, None] - Q[None], axis=2).min()) - r1 - r2
        got = capsule_distance(("one", a1, b1, r1), ("two", a2, b2, r2))
        assert -1e-9 <= brute - got <= 0.75, (i, brute, got)  # samples can only be further apart than the nearest points
        assert got == pytest.approx(capsule_distance(("two", a2, b2, r2), ("one", a1, b1, r1)))


def test_capsules_that_cross_touch_or_run_side_by_side_have_the_distance_one_can_work_out_by_hand():
    o = np.zeros(3)
    assert capsule_distance(("a", o, [100.0, 0, 0], 10.0), ("b", [50.0, -50.0, 30.0], [50.0, 50.0, 30.0], 5.0)) == pytest.approx(15.0)  # crossing, 30 apart
    assert capsule_distance(("a", o, [100.0, 0, 0], 10.0), ("b", [20.0, 40.0, 0], [300.0, 40.0, 0], 10.0)) == pytest.approx(20.0)  # side by side
    assert capsule_distance(("a", o, [100.0, 0, 0], 10.0), ("b", [130.0, 40.0, 0], [300.0, 40.0, 0], 0.0)) == pytest.approx(40.0)  # in line, past the end
    assert capsule_distance(("a", o, o, 10.0), ("b", [0, 0, 50.0], [0, 0, 50.0], 15.0)) == pytest.approx(25.0)  # two balls
    assert capsule_distance(("a", o, [100.0, 0, 0], 30.0), ("b", [50.0, 0, 20.0], [50.0, 0, 90.0], 30.0)) == pytest.approx(-40.0)  # overlapping
    mm, na, nb = clearance([("near", o, [0, 0, 100.0], 10.0), ("far", [0, 900.0, 0], [0, 950.0, 0], 10.0)], [("x", [200.0, 0, 0], [200.0, 0, 50.0], 20.0), ("y", [60.0, 0, 50.0], [90.0, 0, 50.0], 20.0)])
    assert mm == pytest.approx(30.0) and (na, nb) == ("near", "y")


def test_an_arm_is_eight_capsules_from_its_fixed_base_to_its_fingertips():
    q = posture("right", FAR["right"])
    caps = arm_capsules(q)
    assert [c[0] for c in caps] == ["base", "shoulder", "upper arm", "elbow", "forearm", "wrist", "hand", "gripper"]
    again = arm_capsules(q + np.array([20.0, -10.0, 5.0, 15.0, 0.0, -20.0, 40.0]))
    assert np.allclose(caps[0][1], again[0][1]) and np.allclose(caps[0][2], again[0][2])  # the base stays where it is
    flange, tips = flange_mm(q), in_base("right", np.array(FAR["right"]) + [0.0, 0.0, -172.0])
    assert np.allclose(caps[-1][1], flange, atol=1e-6) and np.allclose(caps[-1][2], tips, atol=0.6) and caps[-1][3] >= 87.0
    assert np.allclose(arm_capsules(q, tool_mm=100.0)[-1][2], flange + (tips - flange) * 100.0 / 172.0, atol=0.6)
    assert arm_capsules(q, jaw_mm=140.0)[-1][3] > caps[-1][3]
    spun = arm_capsules(q + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 90.0]))  # turning the gripper by the last joint moves no capsule
    assert all(np.allclose(c[1], d[1], atol=1e-6) and np.allclose(c[2], d[2], atol=1e-6) for c, d in zip(caps, spun))
    with pytest.raises(ValueError):
        arm_capsules([0.0] * 6)


def _stl(path):
    raw = path.read_bytes()
    n = int(np.frombuffer(raw[80:84], "<u4")[0])
    return np.frombuffer(raw[84:84 + 50 * n], np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))["v"].reshape(-1, 3).astype(float) * 1e3


def test_every_part_of_the_drawn_arm_and_its_open_gripper_is_inside_its_capsule_with_room_to_spare():
    folder = Path(pair.__file__).parent / "web" / "xarm7"
    if not (folder / "link1.stl").exists():
        pytest.skip("the arm's meshes are not here")

    def spare(V, a, b, r):
        return r - (pair._segments_apart(V, V, np.asarray(a, float), np.asarray(b, float))).max()

    for (name, _, a, b, r), mesh in zip(pair.CAPSULES, ["link_base"] + [f"link{i}" for i in range(1, 7)]):
        assert spare(_stl(folder / f"{mesh}.stl"), a, b, r) >= 5.0, name  # each mesh is drawn in the frame its capsule is fixed in
    turn = np.diag([-1.0, -1.0, 1.0])  # the hand is put on the flange half a turn round (as `web/scene.js` does), jaws open
    hand = [_stl(folder / "end_tool.stl"), _stl(folder / "base_link.stl") @ turn]
    for side, s in (("left", 1.0), ("right", -1.0)):
        hand += [(_stl(folder / f"{side}_outer_knuckle.stl") + [0.0, s * 35.0, 59.098]) @ turn, (_stl(folder / f"{side}_inner_knuckle.stl") + [0.0, s * 20.0, 74.098]) @ turn,
                 (_stl(folder / f"{side}_finger.stl") + [0.0, s * 35.0, 59.098] + [0.0, s * 35.465, 42.039]) @ turn]
    assert spare(np.vstack(hand), (0.0, 0.0, 0.0), (0.0, 0.0, 172.0), pair._gripper_radius(85.0)) >= 5.0


# ---------------------------------------------------------------------------------------------------- the watch
def watch(side, at, **kw):
    """The watch of one arm of the made-up rig; `at`: where each arm's flange is in the room."""
    other = "left" if side == "right" else "right"
    frame = TRUE if other == "right" else TRUE.inverse()
    return Watch(lambda: posture(side, at[side]), lambda: posture(other, at[other]), frame, **kw)


def test_two_arms_over_their_own_sides_are_clear_and_the_watch_sees_what_the_capsules_say():
    w = watch("right", FAR)
    mm, mine, theirs = w.mm()
    caps_r, caps_l = arm_capsules(posture("right", FAR["right"])), arm_capsules(posture("left", FAR["left"]))
    in_right = [(n, TRUE.inverse().point(a), TRUE.inverse().point(b), r) for n, a, b, r in caps_l]
    slow = clearance(caps_r, in_right)
    assert mm == pytest.approx(slow[0]) and (mine, theirs) == slow[1:] and mm > 150.0
    assert mm == pytest.approx(min(capsule_distance(c, d) for c in caps_r for d in in_right if (c[0], d[0]) != ("base", "base")))
    assert capsule_distance(caps_r[0], in_right[0]) < mm  # the two bases are nearer than anything that moves, and are not counted: they never move
    back = watch("left", FAR).mm()  # the other arm's watch sees the same gap, from its side
    assert back[0] == pytest.approx(mm) and back[1:] == (theirs, mine)
    assert w.too_close() == "" and w.why(10.0) == "" and w.pace() == 1.0


def test_two_grippers_at_the_same_place_are_not_clear_and_no_step_may_start():
    both = {"right": (420.0, 0.0, -200.0), "left": (420.0, 0.0, -200.0)}
    w = watch("right", both)
    mm, mine, theirs = w.mm()
    assert mm < 0.0 and (mine, theirs) == ("gripper", "gripper")
    assert "gripper" in w.too_close() and "touching" in w.too_close() and w.why(1.0) and w.why(0.0) and w.max_step(80.0) == 0.0 and w.pace() == 0.0


def test_the_step_that_would_close_the_gap_is_refused_and_the_longest_step_that_may_start_is_offered_instead():
    w = watch("right", FAR)
    gap = w.mm()[0]
    longest = w.max_step()
    assert longest == pytest.approx((gap - 60.0) / 4.0) and 20.0 < longest < gap  # reach 2, and the other arm may move as far as mine
    assert w.why(longest - 0.01) == "" and "could bring them closer than 60 mm" in w.why(longest + 0.01)
    assert w.why(gap) and w.max_step(10.0) == 10.0 and w.max_step(1e6) == pytest.approx(longest)
    still = watch("right", FAR, other_step_mm=0.0)  # the other arm is known to stand still: twice as far
    assert still.max_step() == pytest.approx((gap - 60.0) / 2.0) and still.why(still.max_step() - 0.01) == "" and still.why(still.max_step() + 0.01)
    careful = watch("right", FAR, stop_mm=gap - 10.0, slow_mm=gap + 30.0, reach=3.0)
    assert careful.too_close() == "" and careful.max_step() == pytest.approx(10.0 / 6.0) and careful.pace() == pytest.approx(0.25)
    assert watch("right", FAR, stop_mm=gap + 1.0, slow_mm=gap + 50.0).too_close()
    assert w.why(float("nan")) and w.why(-1.0)
    with pytest.raises(Refused):
        watch("right", FAR, reach=0.5)


def test_an_arm_that_cannot_be_read_is_unknown_and_unknown_refuses():
    good = posture("left", FAR["left"])

    def fails():
        raise OSError("no route to the arm")

    for read in (fails, lambda: None, lambda: [0.0] * 7, lambda: [float("nan")] * 7, lambda: good[:5], lambda: "joints", lambda: (good, 90.0)):
        for w in (Watch(lambda: posture("right", FAR["right"]), read, TRUE.inverse(), clock=lambda: 100.0), Watch(read, lambda: good, TRUE, clock=lambda: 100.0)):
            with pytest.raises(Unknown):
                w.mm()
            assert "not known" in w.why(1.0) and "not known" in w.too_close() and w.max_step(50.0) == 0.0 and w.pace() == 0.0
    fresh = Watch(lambda: (posture("right", FAR["right"]), 99.9), lambda: (list(good), 99.8), TRUE.inverse(), clock=lambda: 100.0)  # readings that say when they were made
    assert fresh.too_close() == "" and fresh.mm()[0] == pytest.approx(watch("right", FAR).mm()[0])
    stale = Watch(lambda: posture("right", FAR["right"]), lambda: (good, 99.0), TRUE.inverse(), clock=lambda: 100.0)
    assert "1.0 s old" in stale.why(1.0)


def test_asking_the_watch_is_cheap_enough_for_every_poll_of_the_guard():
    rng = np.random.default_rng(2)
    q = [posture("right", FAR["right"]), posture("left", FAR["left"])]
    w = Watch(lambda: q[0], lambda: q[1], TRUE.inverse())
    n, t0 = 300, time.perf_counter()
    for _ in range(n):
        q[0], q[1] = q[0] + rng.normal(0.0, 0.05, 7), q[1] + rng.normal(0.0, 0.05, 7)  # both arms moving: nothing is reused
        w.mm()
    each_ms = (time.perf_counter() - t0) / n * 1e3
    assert each_ms < 4.0, f"one mm() took {each_ms:.2f} ms"  # the guard polls every 20 ms; on the laptop it was written on, 0.2 to 0.3 ms


def test_in_most_short_straight_moves_no_part_of_the_arm_goes_much_further_than_the_flange():
    """What `Watch.reach` stands on, MEASURED with the stand-in's solver (`joints_for`), which is not the controller's.
    A capsule lies within the hull of its two ends, so no point of it moves further than the end that moves most.
    Over 4,500 such moves sampled when this was written the ratio was 1.01 in the middle, under 1.3 for 90% and under
    2.0 for 99%, and the worst were 2.9 (the arm nearly stretched out) and 9 (the solver flipping the elbow over
    during a 30 mm move). So `reach` = 2 covers the ordinary move and NOT the worst one: `too_close`, asked at every
    poll while a move runs, is what catches the rest."""
    rng, ratios = np.random.default_rng(4), []
    for _ in range(24):
        p_room = np.array([rng.uniform(300.0, 550.0), rng.uniform(-550.0, -50.0), rng.uniform(-400.0, -150.0)])
        R = ROOM["right"][0].T @ TOOL_DOWN @ rpy_matrix([0.0, 0.0, rng.uniform(-90.0, 90.0)])
        p, q = in_base("right", p_room), None
        for _ in range(3):
            q = joints_for(p, R, q)
        d = rng.normal(size=3)
        d *= 30.0 / np.linalg.norm(d)
        q2 = joints_for(p + d, R, q)
        if np.linalg.norm(flange_mm(q) - p) > 0.5 or np.linalg.norm(flange_mm(q2) - p - d) > 0.5:
            continue  # out of reach
        ends = [np.r_[a, b] for a, b, _ in (pair._arm_arrays(q), pair._arm_arrays(q2))]
        ratios.append(float(np.linalg.norm((ends[1] - ends[0]).reshape(-1, 3), axis=1).max()) / 30.0)
    assert len(ratios) >= 15 and min(ratios) >= 0.999  # the gripper goes exactly as far as the flange
    assert np.median(ratios) < 1.3 and np.mean(np.array(ratios) <= 2.0) >= 0.9, sorted(ratios)[-3:]


# ---------------------------------------------------------------------------------------------------- each arm on its own side
def test_a_wall_half_way_between_the_bases_gives_each_arm_its_own_side_with_a_gap_between():
    half = dict(zip(("right", "left"), split_plane(TRUE, gap_mm=100.0)))
    assert np.allclose(in_room("right", half["right"][0]) - ROOM["right"][1], [0.0, 1.0, 0.0])  # the right arm's wall faces left, in the room
    for side in ROOM:
        assert np.linalg.norm(half[side][0]) == pytest.approx(1.0) and why_across(np.zeros(3), half[side]) == ""  # its own base is on its own side
    for y, allowed in ((-200.0, {"right"}), (-51.0, {"right"}), (-49.0, set()), (0.0, set()), (49.0, set()), (51.0, {"left"}), (300.0, {"left"})):
        for x, z in ((400.0, -300.0), (100.0, 50.0)):
            ok = {side for side in ROOM if why_across(in_base(side, (x, y, z)), half[side]) == ""}
            assert ok == allowed, (y, ok)
    assert "50 mm on the other arm's side" in why_across(in_base("right", (400.0, 0.0, -300.0)), half["right"])
    moved = split_plane(TRUE, base_a=in_base("right", (0.0, -100.0, 0.0)), base_b=in_base("left", (0.0, 300.0, 0.0)), gap_mm=0.0)  # the wall at y = +100
    assert why_across(in_base("right", (400.0, 99.0, 0.0)), moved[0]) == "" and why_across(in_base("right", (400.0, 101.0, 0.0)), moved[0])
    assert why_across(in_base("left", (400.0, 101.0, 0.0)), moved[1]) == "" and why_across(in_base("left", (400.0, 99.0, 0.0)), moved[1])
    with pytest.raises(Refused):
        split_plane(Frame(np.eye(3), np.zeros(3)))
    with pytest.raises(Refused):
        split_plane(TRUE, gap_mm=-1.0)
