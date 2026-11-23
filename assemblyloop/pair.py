"""Two arms in one space: where one arm's base is in the other's terms, and how far the two arms are apart.

Every position in this program (a taught spot, the camera's fit, the table) is in the base frame of ONE arm. With two
arms there are two such frames, and four things are needed, all of them here:

  1. a `Frame`: the rigid transform that turns a point of arm A's base frame into the same point of the room in arm
     B's base frame. It is fitted (`fit_frames`) to a few marks on the table that BOTH arms touched with their
     fingertips, checked against how the arms are mounted (`check_against_mounts`), or worked out from one camera
     that was fitted from each arm (`frame_from_cameras`, coarser);
  2. with it, what one arm was taught is handed to the other: `spots_in`, `setup_in`;
  3. a model of an arm as a few fat line segments ("capsules"), `arm_capsules`, and the distance between two such
     models, `clearance`;
  4. a `Watch`, which a guard asks before and while its arm moves: is the other arm too close, and how far may I go.
     And `split_plane`, the simplest way of sharing: each arm keeps to its own side of a wall between the two bases.

Lengths are millimetres, angles degrees, orientations the controller's roll / pitch / yaw, as everywhere else.

NOTHING HERE HAS RUN ON THE ARMS. The maths is checked by tests against made-up arms whose truth is known
(`tests/test_pair.py`); the capsules are checked against the arm's drawing meshes, not against the real covers,
cables or the cameras on the grippers. What must be confirmed on the rig is listed where it is assumed:
`fingertip`, `mount_turn`, `spots_in`, `CAPSULES`, `Watch`.

Only numpy is needed to load this module. The camera (`eyes`, which needs OpenCV) is loaded inside the three
functions that deal with a `Setup`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .arms import _frames, matrix_rpy, rpy_matrix
from .guard import Refused
from .spots import Spots

TOOL_MM = 172.0  # from the flange to the fingertips of UFACTORY's gripper


# ---------------------------------------------------------------------------------------------------- the frame
@dataclass(eq=False)
class Frame:
    """Where arm A's base frame is in arm B's: the same point of the room is `p_b = R @ p_a + t` [mm].
    `rms_mm`: how well it fitted the points it was made from (NaN: not known). `n`: how many points. `how`: "marks"
    (both arms touched the same marks), "cameras" (one camera fitted from each arm) or "mounts" (assumed from how the
    arms are mounted). `when`: `time.time()` of the fit. `residuals_mm` / `names`: how far each mark was off."""
    R: np.ndarray
    t: np.ndarray
    rms_mm: float = float("nan")
    n: int = 0
    how: str = "marks"
    when: float = 0.0
    residuals_mm: list = field(default_factory=list)
    names: list = field(default_factory=list)

    def __post_init__(self):
        self.R, self.t = np.array(self.R, float), np.array(self.t, float).reshape(-1)
        if self.R.shape != (3, 3) or self.t.shape != (3,) or not (np.isfinite(self.R).all() and np.isfinite(self.t).all()):
            raise Refused("a frame is a 3 x 3 rotation and a shift of three numbers")
        if np.abs(self.R.T @ self.R - np.eye(3)).max() > 1e-6 or np.linalg.det(self.R) < 0.0:
            raise Refused("the frame between the arms is not a rotation (it stretches or mirrors): it was not written by this program, or the file is damaged")

    def point(self, p):
        """A point (or rows of points) of A's base frame, in B's."""
        return np.asarray(p, float) @ self.R.T + self.t

    def direction(self, d):
        """A direction of A's base frame, in B's: turned, not shifted."""
        return np.asarray(d, float) @ self.R.T

    def rpy(self, rpy_deg):
        """The SAME orientation of a gripper in the room, as the roll / pitch / yaw arm B's controller would report."""
        return matrix_rpy(self.R @ rpy_matrix(rpy_deg))

    def pose(self, pose6):
        pose6 = np.asarray(pose6, float)
        return np.concatenate([self.point(pose6[:3]), self.rpy(pose6[3:6])])

    def inverse(self):
        """B's base frame in A's."""
        return Frame(self.R.T, -self.R.T @ self.t, self.rms_mm, self.n, self.how, self.when, list(self.residuals_mm), list(self.names))

    def then(self, other):
        """A to B (this frame), then B to C (`other`): A to C. The two fits' errors add up (root of the squares)."""
        how = self.how if self.how == other.how else f"{self.how}, then {other.how}"
        return Frame(other.R @ self.R, other.R @ self.t + other.t, float(np.hypot(self.rms_mm, other.rms_mm)), min(self.n, other.n), how, max(self.when, other.when))

    def to_dict(self):
        return dict(R=self.R.tolist(), t=self.t.tolist(), rms_mm=None if not np.isfinite(self.rms_mm) else float(self.rms_mm), n=int(self.n), how=self.how,
                    when=float(self.when), residuals_mm=[round(float(v), 3) for v in self.residuals_mm], names=[str(v) for v in self.names])

    @classmethod
    def from_dict(cls, d):
        try:
            return cls(np.array(d["R"], float), np.array(d["t"], float), float("nan") if d.get("rms_mm") is None else float(d["rms_mm"]), int(d.get("n", 0)),
                       str(d.get("how", "marks")), float(d.get("when", 0.0)), [float(v) for v in d.get("residuals_mm") or []], [str(v) for v in d.get("names") or []])
        except (KeyError, TypeError, ValueError) as e:
            raise Refused(f"the frame between the arms cannot be read ({e})") from None


def fingertip(pose6, tool_mm=TOOL_MM):
    """Where the fingertips are [mm] when the controller reports `pose6`: `tool_mm` along the gripper's own axis from
    the reported point. This is what goes into `fit_frames` for a mark touched with the fingertips: two arms touch
    one mark with their grippers tilted differently, so their FLANGES are then in different places.
    ASSUMES the controller reports the flange (no tool offset set in it), as the rest of the program does. With a tool
    offset of the fingertips set in a controller, pass `tool_mm=0` for that arm."""
    pose6 = np.asarray(pose6, float)
    return pose6[:3] + float(tool_mm) * rpy_matrix(pose6[3:6])[:, 2]


def _kabsch(A, B):
    """The rotation and shift that bring the points A onto the points B with the least squared distance left."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0]) @ U.T  # the last sign: a rotation, never a mirror image
    return R, cb - R @ ca


def _spread(P):
    """How far the points reach along their longest, second and third direction [mm, rms]."""
    return np.linalg.svd(P - P.mean(0), compute_uv=False) / np.sqrt(len(P))


def fit_frames(p_a, p_b, max_rms_mm=3.0, names=None, min_spread_mm=40.0, odd_mm=3.0, odd_times=4.0, how="marks"):
    """The `Frame` from arm A to arm B from the SAME points of the room given in each base frame: row i of `p_a` and
    row i of `p_b` are one mark on the table, touched with the fingertips of each arm (`fingertip`). At least three,
    not on one line; four or more show a mistake, and the further apart they are the better the fit holds far away.

    Marks that all lie on the table (in one plane) ARE enough. Three points that are not on one line already fix a
    rigid body: two directions within the plane are known in both frames, and the third, out of the plane, follows
    because a rotation keeps right-handed things right-handed. (A plane also fits its own mirror image; the sign in
    the fit rules that out.) Marks on one line are not enough: the arms could be turned about that line by any angle.

    Refused, with the reason, when: there are fewer than three marks; they are (nearly) on a line or all close
    together; one mark does not fit the others (it is named, so that it can be taught again); or the fit as a whole
    is worse than `max_rms_mm`. The odd mark is looked for by leaving each mark out in turn: it is the one without
    which the rest agree best, and it is refused if it lies more than `odd_mm` and more than `odd_times` their own
    disagreement from where they put it. (Measured in the fit of ALL marks, a wrong mark hides: it pulls the fit
    towards itself, and no mark of n can be further off than root n times the rms.) This takes five marks or more;
    with four, the three that are left always agree well, so a wrong one is named only once the whole fit is refused.

    `rms_mm` of the result is over the marks themselves. With three or four marks that flatters the fit (six of the
    nine or twelve numbers are used up by the fit); what counts in the end is a mark that was NOT used: touch one more
    with both arms and compare. Each mark's distance from the fit is in `residuals_mm`, in the order given."""
    A, B = np.asarray(p_a, float), np.asarray(p_b, float)
    if A.ndim != 2 or A.shape != B.shape or A.shape[1] != 3:
        raise Refused("the marks must be given as the same number of points (x, y, z) for each arm")
    n = len(A)
    names = [str(v) for v in names] if names is not None else [f"mark {i + 1}" for i in range(n)]
    if len(names) != n:
        raise Refused("there must be one name for every mark")
    if n < 3:
        raise Refused(f"the frame between the arms takes at least 3 marks touched by both arms, not {n}")
    if not (np.isfinite(A).all() and np.isfinite(B).all()):
        raise Refused("a mark has a position that is not a number")
    for P, who in ((A, "first"), (B, "second")):
        s = _spread(P)
        if s[1] < 0.1 * s[0] or s[0] < 1e-9:
            raise Refused(f"the marks are almost on one line (as the {who} arm touched them): that leaves the arms free to turn about that line. Add a mark well off the line")
        if s[1] < min_spread_mm:
            raise Refused(f"the marks are too close together (they spread only {2 * s[1]:.0f} mm across): a small error at the marks becomes a large one further away. Spread them over the table")
    R, t = _kabsch(A, B)
    res = np.linalg.norm(A @ R.T + t - B, axis=1)
    rms = float(np.sqrt(np.mean(res ** 2)))
    worst = int(np.argmax(res))
    odd = None
    for i in range(n if n >= 4 else 0):  # one mark against the fit of all the OTHERS: in the fit of all, a wrong mark drags the fit towards itself and hides
        rest = np.arange(n) != i
        s = _spread(A[rest])
        if s[1] < 0.1 * s[0] or s[1] < 0.5 * min_spread_mm:
            continue  # without this mark the others are on a line: nothing can be said about it
        Ri, ti = _kabsch(A[rest], B[rest])
        others = float(np.sqrt(np.sum((A[rest] @ Ri.T + ti - B[rest]) ** 2) / (n - 3)))  # how far one mark is off, typically (the six numbers the fit used up taken off)
        if odd is None or others < odd[2]:  # the odd one out is the mark without which the rest agree best
            odd = (i, float(np.linalg.norm(Ri @ A[i] + ti - B[i])), others)
    if odd is not None and odd[1] > max(odd_times * odd[2], odd_mm) and (n >= 5 or rms > max_rms_mm):  # four marks: the other three always agree quite well, no evidence alone
        raise Refused(f"{names[odd[0]]} does not fit the others: it is {odd[1]:.1f} mm from where the other {n - 1} marks say it should be (they agree within "
                      f"{odd[2]:.1f} mm). Touch it again with both arms, or leave it out")
    if rms > max_rms_mm:
        raise Refused(f"the marks do not fit one another: {rms:.1f} mm off on average (at most {max_rms_mm:.1f} is allowed), {names[worst]} worst with {res[worst]:.1f} mm. "
                      "Were they touched in the same order by both arms, and with the fingertips?")
    return Frame(R, t, rms, n, how, time.time(), [float(v) for v in res], names)


# ---------------------------------------------------------------------------------------------------- the mounts
def mount_turn(down_a, down_b):
    """The rotation between two bases (`v_b = R @ v_a`) that follows from how they are mounted. `down_a`, `down_b`:
    straight down in each arm's own base frame (`session.mount_down`).
    ASSUMES the two bases point the same way: their forward (x) axes are parallel, as for two shoulders. Then one
    direction (forward) and a second (down) are known in both frames, which fixes the rotation. Two arms that face
    each other, or are toed in, do not satisfy this. On this rig (right and left rolled 45 degrees each way) it is a
    quarter turn about x. It says nothing about where the bases ARE: that takes marks."""
    x = np.array([1.0, 0.0, 0.0])

    def basis(down):
        d = np.asarray(down, float)
        d = d - (d @ x) * x
        if np.linalg.norm(d) < 0.2:
            raise Refused("an arm whose forward axis points up or down has no roll about it: the mounts say nothing here")
        d = d / np.linalg.norm(d)
        return np.stack([x, d, np.cross(x, d)], 1)

    return basis(down_b) @ basis(down_a).T


def turn_between_deg(R1, R2):
    """The angle [deg] of the one turn that takes the orientation R1 to R2."""
    return float(np.degrees(np.arccos(np.clip((np.trace(np.asarray(R1, float).T @ np.asarray(R2, float)) - 1.0) / 2.0, -1.0, 1.0))))


def check_against_mounts(frame, down_a, down_b, max_deg=8.0):
    """Refuse a fitted frame whose rotation is not the one the mounts give (`mount_turn`), within `max_deg`. Returns
    the difference [deg]. Marks touched in a different order by the two arms can fit one another perfectly (four
    marks in a rectangle, two pairs swapped: the same rectangle, turned over) and only this notices."""
    off = turn_between_deg(mount_turn(down_a, down_b), frame.R)
    if off > float(max_deg):
        raise Refused(f"the fitted frame turns one arm against the other {off:.0f} degrees differently than the way they are mounted says (at most {max_deg:.0f}). "
                      "The usual cause is marks touched in a different order by the two arms; otherwise the mount angle given to the program is not the rig's")
    return off


# ---------------------------------------------------------------------------------------------------- one camera, two arms
def frame_from_cameras(setup_a, setup_b):
    """The `Frame` from arm A to arm B from ONE fixed camera that was fitted from each arm: `setup_a` and `setup_b`
    (`eyes.Setup`) each hold where the camera is in that arm's base frame, so a point the camera sees is
    X_cam = Ra p_a + ta = Rb p_b + tb, and p_b = Rb' (Ra p_a + ta - tb).

    This is COARSER than marks. Each side is a lens fit of its own from a dozen points in one plane, in which the
    focal length and the camera's distance can stand in for each other: each fit is good where its points were (on
    the table) and less good as a statement of where the camera IS, and it is the two cameras' places that are
    compared here. Use it to check a frame from marks, or to start with; do not let two arms come close on it.
    It is only meaningful if the camera did not move between the two fits.

    `rms_mm` is an estimate of how far the two fits disagree ON THE TABLE: the corners and the middle of A's area, at
    the height the camera was fitted at, go into the picture through A's camera and back onto the table through B's,
    and are compared with where this frame puts them. That counts the two lens fits' disagreement and the tables'
    (both arms are taken to have held the block at the same height above the table). It cannot see an error both fits
    share, so the truth may be worse. NaN if it cannot be worked out. `n` is the number of points compared."""
    import cv2

    Ra, Rb = cv2.Rodrigues(np.asarray(setup_a.camera.rvec, float).reshape(3))[0], cv2.Rodrigues(np.asarray(setup_b.camera.rvec, float).reshape(3))[0]
    ta, tb = np.asarray(setup_a.camera.tvec, float).reshape(3), np.asarray(setup_b.camera.tvec, float).reshape(3)
    frame = Frame(Rb.T @ Ra, Rb.T @ (ta - tb), float("nan"), 0, "cameras", max(float(setup_a.when), float(setup_b.when)))
    try:
        uv = [tuple(p) for p in setup_a.area] + [tuple(np.mean(setup_a.area, 0))]
        pts = np.array([setup_a.table.point(u, v, setup_a.w_grasp) for u, v in uv])
        back = setup_b.camera.scaled(setup_a.camera.size).on_plane(setup_a.camera.project(pts), setup_b.table.axes[:, 2], setup_b.w_grasp)
        res = np.linalg.norm(back - frame.point(pts), axis=1)
        if np.isfinite(res).all():
            frame.rms_mm, frame.n, frame.residuals_mm = float(np.sqrt(np.mean(res ** 2))), len(pts), [float(v) for v in res]
    except (Refused, ValueError, cv2.error):
        pass
    return frame


# ---------------------------------------------------------------------------------------------------- handing over what one arm knows
def spots_in(frame, spots, side):
    """The spots taught with arm A, for arm B (`side`): every position through `frame.point`, how the gripper was
    turned at each spot and the common tilt through `frame.rpy`; the lift stays.
    ASSUMES both arms carry the same gripper the same way: a spot is where the controller's reported point (the
    flange) is while the fingertips are at the thing, so it carries over only if flange to fingertips is the same
    length on both arms. Whether arm B can REACH a spot, and get there with that tilt, is not looked at here."""
    out = Spots(side=str(side), rpy=None if spots.rpy is None else frame.rpy(spots.rpy), lift=float(spots.lift))
    for name, p in spots.at.items():
        out.at[name], out.rot[name] = frame.point(p), frame.rpy(spots.turn(name))
    return out


def setup_in(frame, setup, side):
    """What the camera was shown with arm A (`eyes.Setup`), for arm B (`side`), without showing it again.

      - the camera: X_cam = Rc p_a + tc with p_a = R' (p_b - t) gives Rc_b = Rc R', tc_b = tc - Rc_b t;
      - the table's directions are turned: axes_b = R axes;
      - the table has no origin of its own (`Table.point(u, v, w) = axes @ (u, v, w)`, counted from the BASE), so the
        shift between the bases turns up in the numbers (u, v, w): p_b = axes_b @ ((u, v, w) + axes_b' t). The area's
        corners move by the first two of `axes_b' t`; the table's height `w0` and the grasp height `w_grasp` by the third;
      - `rpy_ref` (how the gripper is turned for a block along `angle_ref`) goes through `frame.rpy`;
      - `angle_ref` stays: it is an angle within the table, counted from the table's own first direction, and that
        direction is turned along with everything else; shifting where (u, v) starts changes no angle.
    `fit_mm` grows by the frame's own error (unknown: unchanged). The same assumption as `spots_in` holds: the same
    gripper, the same length, on both arms."""
    import cv2

    from .eyes import Camera, Setup, Table

    axes = np.asarray(setup.table.axes, float)
    if np.abs(axes.T @ axes - np.eye(3)).max() > 1e-6:
        raise Refused("the table's three directions are not at right angles: this setup cannot be handed over")
    Rc = cv2.Rodrigues(np.asarray(setup.camera.rvec, float).reshape(3))[0] @ frame.R.T
    axes_b = frame.R @ axes
    du, dv, dw = axes_b.T @ frame.t
    cam = Camera(tuple(setup.camera.size), float(setup.camera.f), tuple(setup.camera.k), cv2.Rodrigues(Rc)[0].ravel(),
                 np.asarray(setup.camera.tvec, float).reshape(3) - Rc @ frame.t, float(setup.camera.rms))
    return Setup(cam, Table(axes_b, float(setup.table.w0) + dw), float(setup.w_grasp) + dw, frame.rpy(setup.rpy_ref), float(setup.angle_ref),
                 [(float(u) + du, float(v) + dv) for u, v in setup.area], str(side), float(setup.fit_mm) + (frame.rms_mm if np.isfinite(frame.rms_mm) else 0.0), float(setup.when))


# ---------------------------------------------------------------------------------------------------- an arm as capsules
# One capsule per part of the arm: (name, which joint's frame it is fixed in (index into `arms._frames`; None: the
# base), one end, the other end [mm, in that frame], radius [mm]).
#
# Where the numbers come from. Each segment was fitted to that part's drawing mesh (`web/xarm7/`, MuJoCo Menagerie's
# xArm 7) as the short segment around which the mesh needs the smallest radius; the radius is what the mesh needs
# (in brackets) plus 8 mm or more, rounded up to 5. The segments are NOT the lines between the joints' origins: the
# xArm's parts are offset sideways from those lines (the forearm by 78 mm), and capsules along them would need radii
# of 105 to 180 mm. A test holds every mesh vertex at least 5 mm inside its capsule.
# NOT covered, to be confirmed on the rig: cables and their loops, anything bolted to the arm, and the two cameras on
# each gripper if they stick out further than 95 mm from the gripper's axis.
CAPSULES = (("base", None, (-14.0, 3.0, 22.0), (0.0, 0.0, 90.0), 90.0),  # (80.3) with the connector panel at its back
            ("shoulder", 0, (0.0, 17.0, 0.0), (0.0, 0.0, -54.0), 85.0),  # (75.2)
            ("upper arm", 1, (0.0, -137.0, 14.0), (0.0, -26.0, 102.0), 90.0),  # (78.7)
            ("elbow", 2, (5.0, 7.0, -45.0), (52.0, -4.0, 0.0), 85.0),  # (76.2)
            ("forearm", 3, (88.0, -120.0, 9.0), (23.0, -18.0, 84.0), 85.0),  # (74.4)
            ("wrist", 4, (0.0, 45.0, 0.0), (-1.0, 10.0, -156.0), 65.0),  # (55.7)
            ("hand", 5, (76.0, 21.0, 0.0), (18.0, 7.0, 18.0), 70.0))  # (61.3)
GRIPPER_MM = 95.0  # (86.7: the linkage of the OPEN jaws, 85 mm apart, reaches that far from the gripper's axis)
_FIXED = (np.array([c[2] for c in CAPSULES] + [(0.0, 0.0, 0.0)]), np.array([c[3] for c in CAPSULES] + [(0.0, 0.0, 0.0)]),
          np.array([c[4] for c in CAPSULES] + [0.0]), tuple(c[0] for c in CAPSULES) + ("gripper",), [6 if c[1] is None else c[1] for c in CAPSULES] + [6])


def _gripper_radius(jaw_mm):
    return max(GRIPPER_MM, float(jaw_mm) / 2.0 + 52.5)  # jaws that open wider than this gripper's carry their linkage further out


def _arm_arrays(joints_deg, tool_mm=TOOL_MM, jaw_mm=85.0):
    """The capsules of `arm_capsules` as arrays: one end each, the other end each, the radii."""
    q = np.asarray(joints_deg, float).reshape(-1)[:7]
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError("seven joint angles are needed")
    F = np.array(_frames(np.radians(q)))
    lo, hi, rad, _, idx = _FIXED
    Rm, o = F[idx, :3, :3], F[idx, :3, 3] * 1e3
    Rm[0], o[0] = np.eye(3), 0.0  # the base does not move
    a, b, rad = np.einsum("nij,nj->ni", Rm, lo) + o, np.einsum("nij,nj->ni", Rm, hi) + o, rad.copy()
    b[-1], rad[-1] = b[-1] + float(tool_mm) * Rm[-1][:, 2], _gripper_radius(jaw_mm)
    return a, b, rad


def arm_capsules(joints_deg, tool_mm=TOOL_MM, jaw_mm=85.0):
    """The arm as a list of capsules `(name, a, b, radius)` in its own base frame [mm], from seven joint angles [deg]:
    everything within `radius` of the segment from `a` to `b`. One per part of the arm (`CAPSULES`, which says where
    the numbers come from), and one for the gripper: from the flange along the gripper's axis to the fingertips
    (`tool_mm`), wide enough for the open jaws (`jaw_mm` apart) whichever way the gripper is turned. Its round end
    reaches a radius beyond the fingertips, which also covers a block that is being carried.
    The model is generous on purpose and it is a model: see what `CAPSULES` does not cover."""
    a, b, rad = _arm_arrays(joints_deg, tool_mm, jaw_mm)
    return [(name, a[i].copy(), b[i].copy(), float(rad[i])) for i, name in enumerate(_FIXED[3])]


def _segments_apart(p1, q1, p2, q2):
    """The distance between the segments p1-q1 and p2-q2, for arrays of segments at once (the last axis is x, y, z).
    A segment may be a single point, and the two may be parallel."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = np.sum(d1 * d1, -1), np.sum(d2 * d2, -1), np.sum(d2 * r, -1)
    b, c = np.sum(d1 * d2, -1), np.sum(d1 * r, -1)
    has1, has2 = a > 1e-12, e > 1e-12  # a segment of no length is a point
    a1, e1, den = np.where(has1, a, 1.0), np.where(has2, e, 1.0), a * e - b * b
    cross = has1 & has2 & (den > 1e-9 * a * e)  # not parallel: the two lines have one nearest pair of points
    s = np.where(cross, np.clip((b * f - c * e) / np.where(cross, den, 1.0), 0.0, 1.0), 0.0)
    t = np.where(has2, (b * s + f) / e1, 0.0)
    s = np.where((t < 0.0) | ~has2, np.clip(-c / a1, 0.0, 1.0), np.where(t > 1.0, np.clip((b - c) / a1, 0.0, 1.0), s))  # past an end of the second: nearest to that end
    s, t = np.where(has1, s, 0.0), np.clip(t, 0.0, 1.0)
    return np.linalg.norm(p1 + s[..., None] * d1 - p2 - t[..., None] * d2, axis=-1)


def capsule_distance(c1, c2):
    """The free distance [mm] between two capsules `(name, a, b, radius)`: between their segments, less both radii.
    Negative: they overlap by that much."""
    return float(_segments_apart(np.asarray(c1[1], float), np.asarray(c1[2], float), np.asarray(c2[1], float), np.asarray(c2[2], float))) - float(c1[3]) - float(c2[3])


def _nearest(a1, b1, r1, a2, b2, r2, fixed=None):
    D = _segments_apart(a1[:, None], b1[:, None], a2[None], b2[None]) - r1[:, None] - r2[None]
    if fixed is not None:
        D[fixed] = np.inf  # two parts that never move: how far they are apart is the rig's business, not a move's
    i, j = np.unravel_index(int(np.argmin(D)), D.shape)
    return float(D[i, j]), int(i), int(j)


def clearance(caps_a, caps_b):
    """The smallest free distance [mm] between any capsule of one list and any of the other (both in ONE frame), and
    which two they are: `(mm, name_a, name_b)`. The two capsules called "base" are not compared with each other:
    neither ever moves, so their distance says nothing about a move (and would hide every other pair on a rig whose
    bases are close together)."""
    if not len(caps_a) or not len(caps_b) or (len(caps_a) == 1 and len(caps_b) == 1 and caps_a[0][0] == caps_b[0][0] == "base"):
        raise ValueError("both arms need at least one capsule that can move")
    arr = [(np.array([c[1] for c in caps], float), np.array([c[2] for c in caps], float), np.array([c[3] for c in caps], float)) for caps in (caps_a, caps_b)]
    fixed = np.array([[c[0] == d[0] == "base" for d in caps_b] for c in caps_a])
    mm, i, j = _nearest(*arr[0], *arr[1], fixed)
    return mm, caps_a[i][0], caps_b[j][0]


# ---------------------------------------------------------------------------------------------------- the watch
class Unknown(RuntimeError):
    """How far the arms are apart cannot be said (an arm's joints could not be read). Never to be taken as "clear"."""


class Watch:
    """What a guard asks about the OTHER arm. One `Watch` per guard (it is not made for two threads at once).

    `mine()` and `other()` return the seven joint angles [deg] each arm's controller publishes (`joints_now`, which
    costs nothing) or raise. `other_in_mine`: the `Frame` from the other arm's base to mine. From these the two arms
    are drawn as capsules (`arm_capsules`) and the smallest free distance between them is the clearance (base against
    base left out: neither moves).

    The rule for a move of `step_mm` by my arm (`why`): the clearance NOW, less `reach` times the step, less `reach`
    times what the other arm may move meanwhile, must stay above `stop_mm`. Behind it is this: in a short straight
    move at a fixed orientation the flange goes `step_mm`, the gripper goes exactly as far, and the rest of the arm
    is taken to go no further than `reach` times that. THAT IS ASSUMED, NOT PROVEN: how the elbow moves is the
    controller's choice, and close to a stretched-out or folded posture a short move of the flange can swing the
    elbow a long way. A test samples short moves with the stand-in's own solver (not the controller's) and reports
    the worst ratio it saw; `reach` should be set with that in mind, and confirmed on the arm by logging joints over
    real moves. The rule looks only at distance, not direction: once the arms are closer than the rule allows, a
    move AWAY is refused as well, and a person separates them.

    `other_step_mm`: how far the other arm's flange may travel while my step is under way, i.e. the longest segment
    its guard sends. None (the default) takes it to be as long as my own step, which is right when both guards ask a
    `Watch` like this one; 0 only when the other arm is known to stand still (not enabled, or its session is not
    armed). Turning the gripper by the last joint moves no capsule (the gripper's is round about its axis). Tilting
    the gripper (`Guard.level`) is not a straight move: ask with the distance the fingertips swing, 3 mm per degree.

    An arm that cannot be read, reads nonsense, reads all zeros (what the controller publishes before its first
    report), or whose reading is older than `max_age_s`, makes the clearance UNKNOWN, and unknown refuses: `why` and
    `too_close` give the reason, `max_step` gives 0. Age can only be judged if the reader says when its reading was
    made: `mine` / `other` may return `(joints, stamp)` with `stamp` on `clock`. A bare list is believed to be
    fresh; the SDK goes on returning the last angles of an arm it has lost, so give a stamp, or raise when the arm
    is not connected.

    `slow_mm` is for `pace`: below it a guard may go slower. One `mm()` costs about 0.2 ms."""

    def __init__(self, mine, other, other_in_mine, stop_mm=60.0, slow_mm=150.0, reach=2.0, other_step_mm=None, tool_mm=TOOL_MM, jaw_mm=85.0,
                 max_age_s=0.5, clock=time.monotonic):
        if not (float(reach) >= 1.0 and float(stop_mm) >= 0.0 and float(slow_mm) >= float(stop_mm)):
            raise Refused("a watch needs reach of at least 1 (the gripper itself moves as far as the flange), and 0 <= stop_mm <= slow_mm")
        self.mine, self.other, self.frame = mine, other, other_in_mine
        self.stop_mm, self.slow_mm, self.reach = float(stop_mm), float(slow_mm), float(reach)
        self.other_step_mm = None if other_step_mm is None else float(other_step_mm)
        self.tool_mm, self.jaw_mm, self.max_age_s, self.clock = float(tool_mm), float(jaw_mm), float(max_age_s), clock
        self._kept = [None, None]  # the last joints of each arm and their capsules: an arm that stands still is not worked out again

    def _read(self, read, who):
        try:
            got = read()
        except Exception as e:  # whatever went wrong: the arm's place is not known
            raise Unknown(f"{who} could not be read ({e})") from None
        if isinstance(got, tuple) and len(got) == 2 and np.ndim(got[0]) == 1:
            got, stamp = got
            try:
                age = float(self.clock()) - float(stamp)
            except (TypeError, ValueError):
                raise Unknown(f"{who} gave a reading without a usable time") from None
            if not age <= self.max_age_s:  # also when the age is not a number
                raise Unknown(f"the last reading of {who} is {age:.1f} s old")
        try:
            q = np.asarray(got, float).reshape(-1)
        except (TypeError, ValueError):
            raise Unknown(f"{who} gave no joint angles") from None
        if q.size < 7 or not np.isfinite(q[:7]).all() or np.abs(q[:7]).max() > 400.0:
            raise Unknown(f"{who} gave no usable joint angles")
        if not q[:7].any():
            raise Unknown(f"{who} reports all joints at exactly zero: its controller has not published its state yet")
        return q[:7]

    def _capsules(self, k, q):
        kept = self._kept[k]
        if kept is None or not np.array_equal(kept[0], q):
            a, b, rad = _arm_arrays(q, self.tool_mm, self.jaw_mm)
            if k == 1:
                a, b = self.frame.point(a), self.frame.point(b)
            kept = self._kept[k] = (q.copy(), a, b, rad)
        return kept[1:]

    def mm(self):
        """The clearance now: `(mm, part of my arm, part of the other arm)`. Raises `Unknown` if it cannot be said."""
        mine, other = self._capsules(0, self._read(self.mine, "this arm")), self._capsules(1, self._read(self.other, "the other arm"))
        gap, i, j = _nearest(*mine, *other, (0, 0))
        if not np.isfinite(gap):
            raise Unknown("the distance between the arms did not come out as a number")
        return gap, _FIXED[3][i], _FIXED[3][j]

    def _room(self, gap, step_mm):
        """What is left of the clearance above `stop_mm` after my step and the other arm's."""
        other = float(step_mm) if self.other_step_mm is None else self.other_step_mm
        return gap - self.reach * (float(step_mm) + other) - self.stop_mm

    def too_close(self):
        """'' while the arms are at least `stop_mm` apart; otherwise why not (a guard stops BOTH arms on this). An arm
        that cannot be read counts as too close."""
        try:
            gap, a, b = self.mm()
        except Unknown as e:
            return f"where the arms are is not known: {e}"
        return "" if gap >= self.stop_mm else f"this arm's {a} is {max(gap, 0.0):.0f} mm from the other arm's {b} (less than {self.stop_mm:.0f} mm)" + (" and, by the model, touching it" if gap < 0.0 else "")

    def why(self, step_mm):
        """'' if a move of `step_mm` by my arm may start now; otherwise why not."""
        if not float(step_mm) >= 0.0:
            return "the length of the step is not a number"
        try:
            gap, a, b = self.mm()
        except Unknown as e:
            return f"where the arms are is not known: {e}"
        if self._room(gap, step_mm) >= 0.0:
            return ""
        return (f"this arm's {a} is {max(gap, 0.0):.0f} mm from the other arm's {b}: a step of {float(step_mm):.0f} mm could bring them closer than {self.stop_mm:.0f} mm "
                f"(from here at most {self._longest(gap):.0f} mm)")

    def _longest(self, gap):
        spare = gap - self.stop_mm
        return max(0.0, spare / (2.0 * self.reach) if self.other_step_mm is None else (spare - self.reach * self.other_step_mm) / self.reach)

    def max_step(self, mm=float("inf")):
        """The longest step [mm] that may start now, at most `mm`: 0 if none may, or if the arms' places are not
        known. A guard shortens its segments to this near the other arm instead of refusing the whole move."""
        try:
            gap = self.mm()[0]
        except Unknown:
            return 0.0
        return float(min(max(float(mm), 0.0), self._longest(gap)))

    def pace(self):
        """How fast to go, from 0 to 1: 1 with the arms `slow_mm` or more apart, falling evenly to 0 at `stop_mm`;
        0 if their places are not known."""
        try:
            gap = self.mm()[0]
        except Unknown:
            return 0.0
        return float(np.clip((gap - self.stop_mm) / max(self.slow_mm - self.stop_mm, 1e-9), 0.0, 1.0))


# ---------------------------------------------------------------------------------------------------- each arm on its own side
def split_plane(frame, base_a=(0.0, 0.0, 0.0), base_b=(0.0, 0.0, 0.0), gap_mm=100.0):
    """The simplest way for two arms to share a table: a wall half way between the two bases, at right angles to the
    line from one to the other, and each arm's flange stays `gap_mm / 2` on its own side of it.
    `frame`: from arm A's base to arm B's. `base_a` / `base_b`: the point of each base to measure from, in its own
    frame (the origin; another point if the wall should stand elsewhere).
    Returns `(half_a, half_b)`, each `(normal, offset)` in that arm's OWN base frame: the arm is on its side where
    `normal @ p <= offset` (`why_across`). The normal points at the other arm.
    This keeps the two FLANGES `gap_mm` apart and nothing else: an elbow or a gripper can still cross the wall, so
    `gap_mm` must be generous (a gripper reaches `tool_mm` beyond the flange) or a `Watch` used as well."""
    if not float(gap_mm) >= 0.0:
        raise Refused("the gap between the two sides cannot be negative")
    oa, ob = np.asarray(base_a, float), frame.inverse().point(base_b)  # both in A's frame
    apart = float(np.linalg.norm(ob - oa))
    if apart < 1.0:
        raise Refused("the two bases are in the same place by this frame: there is no wall to put between them")
    n = (ob - oa) / apart
    mid = (oa + ob) / 2.0
    nb = -frame.direction(n)
    return (n, float(n @ mid) - float(gap_mm) / 2.0), (nb, float(nb @ frame.point(mid)) - float(gap_mm) / 2.0)


def why_across(p, half_space):
    """'' if the point `p` is on the arm's own side (`half_space` from `split_plane`); otherwise how far it is over."""
    normal, offset = half_space
    over = float(np.asarray(normal, float) @ np.asarray(p, float)) - float(offset)
    return "" if over <= 1e-6 else f"it is {over:.0f} mm on the other arm's side of the line between the two arms"
