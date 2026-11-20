"""Eyes: one fixed camera that looks at the table, and what it makes of the blocks on it. Nothing here moves the arm.

  1. `grab` takes one picture.
  2. `find_blobs` cuts out everything strongly coloured (painted blocks on a grey mat), one blob per patch of one
     colour. Two blocks of one colour that touch are one blob and are not made out: keep blocks apart.
  3. A `Camera` says where in the picture a point of the table shows. It is fitted (`calibrate`) to pairs of
     "this point of the table showed here in the picture", which the arm makes itself by putting one block down in
     a dozen places.
  4. `make_out` tries every kind of block of the catalogue, every way it can lie, on every blob: it draws the outline
     that block would have in the picture and shifts and turns it until it covers the blob. The best cover names the
     block and says where its middle is and which way it points. This is what makes a camera that looks at the table
     from the side usable: a tall block's outline is its top AND its sides, and only the drawn solid accounts for that.

Lengths are millimetres. Points are in the arm's base frame, shifted along the gripper's axis so that a block stands
where the gripper goes to hold it (see `Table`): the camera is fitted to where the ARM put a block, so everything
the camera says is already in the arm's own terms.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .blocks import Seen, name_all
from .guard import Refused

HUES = (("red", 0, 8), ("orange", 8, 22), ("yellow", 22, 39), ("green", 39, 89), ("blue", 89, 131), ("purple", 131, 170), ("red", 170, 181))  # OpenCV hue, 0 to 180
PAINT = dict(red=(40, 40, 220), orange=(30, 140, 240), yellow=(40, 220, 240), green=(60, 190, 60), blue=(200, 110, 40), purple=(170, 60, 150))  # BGR, for drawing


SCENE_CAMERA = "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:4.3:1.0-video-index0"  # the rig's left scene camera, by the port it is plugged into


class V4L2:
    """A camera that is opened for one picture and let go again, so that other programs can use it in between."""

    def __init__(self, device=SCENE_CAMERA, size=(3840, 2160)):
        self.device, self.size = str(device), tuple(size)

    def grab(self):
        return grab(self.device, self.size)


def grab(device, size=(3840, 2160), warm=8):
    """One picture (BGR) from a V4L2 camera. The first frames after opening are dark: they are thrown away."""
    cap = cv2.VideoCapture(os.path.realpath(str(device)), cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise Refused(f"the camera {device} does not open: is another program using it?")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        ok, im = False, None
        for _ in range(max(1, warm)):
            ok, im = cap.read()
        if not ok or im is None:
            raise Refused(f"the camera {device} gives no picture")
        return im
    finally:
        cap.release()


# ---------------------------------------------------------------------------------------------------- blobs
@dataclass
class Blob:
    colour: str
    mask: np.ndarray  # True where the blob is, in its own bounding box
    x0: int
    y0: int
    centre: tuple  # in the whole picture
    cut: bool = False  # it touches the edge of the picture or of the region looked at: part of it is missing

    @property
    def area(self):
        return int(self.mask.sum())


def find_blobs(bgr, region=None, s_min=115, v_min=70):
    """Every patch of strong colour, largest first. `region`: a mask of where to look (the mat)."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (0, 0), max(0.8, w / 2000.0)), cv2.COLOR_BGR2HSV)
    strong = (hsv[..., 1] >= s_min) & (hsv[..., 2] >= v_min)
    if region is not None:
        strong &= region
    k = max(3, int(round(w / 640.0)) | 1)
    box = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = []
    for colour in dict.fromkeys(n for n, _, _ in HUES):
        m = np.zeros((h, w), bool)
        for n, lo, hi in HUES:
            if n == colour:
                m |= (hsv[..., 0] >= lo) & (hsv[..., 0] < hi)
        m = cv2.morphologyEx(cv2.morphologyEx((m & strong).astype(np.uint8), cv2.MORPH_OPEN, box), cv2.MORPH_CLOSE, box)
        n, lab, stats, cen = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            x, y, bw, bh, area = (int(v) for v in stats[i])
            if area < 1.2e-4 * w * h:
                continue
            piece = lab[y:y + bh, x:x + bw] == i
            edge = x <= 0 or y <= 0 or x + bw >= w or y + bh >= h
            if region is not None and not edge:
                ring = cv2.dilate(piece.astype(np.uint8), box).astype(bool) & ~region[y:y + bh, x:x + bw]
                edge = bool(ring.any())
            out.append(Blob(colour, piece, x, y, (float(cen[i][0]), float(cen[i][1])), cut=edge))
    return sorted(out, key=lambda b: -b.area)


# ---------------------------------------------------------------------------------------------------- the table
@dataclass
class Table:
    """The table in the arm's terms. `axes`: its two directions and `up` (away from it, along the gripper), as the
    columns of a matrix. `w0`: how far along `up` its surface is, in the shifted frame of the module's docstring: a
    block that stands on it, held at its taught height, has its middle at w0 + height / 2."""
    axes: np.ndarray
    w0: float

    def point(self, u, v, w):
        return self.axes @ np.array([u, v, w], float)

    def coords(self, p):
        return self.axes.T @ np.asarray(p, float)


# ---------------------------------------------------------------------------------------------------- the camera
@dataclass
class Camera:
    size: tuple  # width, height of the picture it was fitted to
    f: float
    k: tuple = (0.0, 0.0)  # how much the lens bends straight lines
    rvec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tvec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rms: float = 0.0  # how far [pixels] the fitted points are from where the camera says they should be

    @property
    def K(self):
        return np.array([[self.f, 0.0, self.size[0] / 2.0], [0.0, self.f, self.size[1] / 2.0], [0.0, 0.0, 1.0]])

    @property
    def dist(self):
        return np.array([self.k[0], self.k[1], 0.0, 0.0, 0.0])

    def project(self, pts):
        px, _ = cv2.projectPoints(np.asarray(pts, float).reshape(-1, 1, 3), self.rvec, self.tvec, self.K, self.dist)
        return px.reshape(-1, 2)

    def on_plane(self, px, up, w):
        """Where the line of sight through a pixel meets the plane `up . X = w`."""
        n = cv2.undistortPoints(np.asarray(px, float).reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)
        R, _ = cv2.Rodrigues(np.asarray(self.rvec, float))
        eye = -R.T @ np.asarray(self.tvec, float).reshape(3)
        rays = (R.T @ np.c_[n, np.ones(len(n))].T).T
        s = (float(w) - eye @ up) / (rays @ up)
        return eye + rays * s[:, None]

    def scaled(self, size):
        """The same camera for a picture of another size (the same view, more or fewer pixels)."""
        r = size[0] / float(self.size[0])
        if abs(size[1] / float(self.size[1]) - r) > 0.01:
            raise Refused("this picture has another shape than the ones the camera was fitted to")
        return Camera(tuple(size), self.f * r, self.k, self.rvec, self.tvec, self.rms * r)

    def to_dict(self):
        return dict(size=list(self.size), f=self.f, k=list(self.k), rvec=[float(v) for v in np.ravel(self.rvec)], tvec=[float(v) for v in np.ravel(self.tvec)], rms=self.rms)

    @classmethod
    def from_dict(cls, d):
        return cls(tuple(d["size"]), float(d["f"]), tuple(d["k"]), np.array(d["rvec"], float), np.array(d["tvec"], float), float(d.get("rms", 0.0)))


def _residual(p, size, xyz, px):
    return (Camera(size, p[0], (p[1], p[2]), p[3:6], p[6:9]).project(xyz) - px).ravel()


def calibrate(px, xyz, size):
    """Fit a camera to points of the table (`xyz`) and where each showed in the picture (`px`). The points may all lie
    in one plane. Starts from a lens that bends nothing, tried at a range of focal lengths, then lets everything go."""
    px, xyz = np.asarray(px, float).reshape(-1, 2), np.asarray(xyz, float).reshape(-1, 3)
    if len(px) < 8 or len(px) != len(xyz):
        raise Refused("fitting the camera takes at least 8 places")
    spread = np.linalg.svd(xyz - xyz.mean(0), compute_uv=False)
    if spread[1] < 0.15 * spread[0]:
        raise Refused("the places are almost on one line: spread them over the whole mat")
    best = None
    for f in size[0] * np.geomspace(0.25, 2.5, 24):
        cam = Camera(tuple(size), float(f))
        ok, rvec, tvec = cv2.solvePnP(xyz.reshape(-1, 1, 3), px.reshape(-1, 1, 2), cam.K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or float(tvec[2, 0]) <= 0:
            continue
        p = np.r_[f, 0.0, 0.0, rvec.ravel(), tvec.ravel()]
        err = float(np.sqrt(np.mean(_residual(p, size, xyz, px) ** 2)))
        if best is None or err < best[0]:
            best = (err, p)
    if best is None:
        raise Refused("no camera fits these places: were the blocks found where the arm put them?")
    p = best[1]
    free = np.array([1, 1, len(px) >= 12, 1, 1, 1, 1, 1, 1], bool)  # the second bend of the lens only with enough places
    scale = np.array([size[0] * 1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2])
    lam, r = 1e-3, _residual(p, size, xyz, px)
    for _ in range(80):
        J = np.zeros((len(r), 9))
        for i in np.flatnonzero(free):
            d = np.zeros(9)
            d[i] = scale[i]
            J[:, i] = (_residual(p + d, size, xyz, px) - r) / scale[i]
        Jf = J[:, free]
        H, g = Jf.T @ Jf, Jf.T @ r
        step = np.zeros(9)
        step[free] = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), g)
        r2 = _residual(p + step, size, xyz, px)
        if r2 @ r2 < r @ r:
            better = (r @ r - r2 @ r2) < 1e-10 * max(r @ r, 1e-12)
            p, r, lam = p + step, r2, lam * 0.3
            if better:
                break
        else:
            lam *= 5.0
            if lam > 1e8:
                break
    return Camera(tuple(size), float(p[0]), (float(p[1]), float(p[2])), p[3:6].copy(), p[6:9].copy(), float(np.sqrt(np.mean(r ** 2))))


@dataclass
class Setup:
    """What showing the camera the table found, kept in `camera.json`: the camera, the table, at which height the
    gripper holds a low block (`w_grasp`), and how the gripper is turned (`rpy_ref`) to take a block whose long side
    points along `angle_ref`. `area`: the corners [(u, v)] of the part of the table the camera was shown; nothing
    outside it is looked at or reached for."""
    camera: Camera
    table: Table
    w_grasp: float
    rpy_ref: np.ndarray
    angle_ref: float
    area: list
    side: str = "right"
    fit_mm: float = 0.0  # how far the camera was off, at worst, when it looked for the block the arm had just put down
    when: float = 0.0

    def region(self, size, tall_mm=0.0, around_mm=60.0):
        """A mask of where to look in a picture of this size: the shown part of the table and a little around it
        (a block whose middle is just inside must be seen whole), and the room above it up to `tall_mm` (a block
        that stands there reaches higher in the picture than the table under it)."""
        cam = self.camera.scaled(size)
        mid = np.mean(self.area, 0)
        wide = [p + (p - mid) / max(np.linalg.norm(p - mid), 1e-6) * around_mm for p in np.asarray(self.area, float)]
        px = cam.project([self.table.point(u, v, self.table.w0 + h) for u, v in wide for h in (0.0, float(tall_mm))])
        m = np.zeros((size[1], size[0]), np.uint8)
        cv2.fillConvexPoly(m, np.round(cv2.convexHull(px.astype(np.float32))).astype(np.int32), 1)
        return m.astype(bool)

    def inside(self, uv):
        return cv2.pointPolygonTest(np.array(self.area, np.float32).reshape(-1, 1, 2), (float(uv[0]), float(uv[1])), False) >= 0

    def to_dict(self):
        return dict(camera=self.camera.to_dict(), axes=self.table.axes.tolist(), w0=self.table.w0, w_grasp=self.w_grasp, rpy_ref=[float(v) for v in self.rpy_ref],
                    angle_ref=self.angle_ref, area=[[float(u), float(v)] for u, v in self.area], side=self.side, fit_mm=self.fit_mm, when=self.when)

    @classmethod
    def from_dict(cls, d):
        return cls(Camera.from_dict(d["camera"]), Table(np.array(d["axes"], float), float(d["w0"])), float(d["w_grasp"]), np.array(d["rpy_ref"], float), float(d["angle_ref"]),
                   [tuple(p) for p in d["area"]], str(d.get("side", "right")), float(d.get("fit_mm", 0.0)), float(d.get("when", 0.0)))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))


def area_around(uvs, margin_mm=40.0):
    """The outline around these points of the table, pushed out by a margin."""
    hull = cv2.convexHull(np.asarray(uvs, np.float32)).reshape(-1, 2).astype(float)
    mid = hull.mean(0)
    return [tuple(p + (p - mid) / max(np.linalg.norm(p - mid), 1e-6) * margin_mm) for p in hull]


def mean_angle(angles, turns):
    """The middle of directions that repeat every `turns` degrees (a bar looks the same turned by 180)."""
    a = np.radians(np.asarray(angles, float)) * (360.0 / turns)
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) * turns / 360.0 % turns)


def settle(xyz, pictures, kind, axes, rpy_ref, extra_uv=(), side="right"):
    """From the places the arm put ONE block (`xyz`: where the gripper held it, lying flat) and a picture of each,
    work out the whole `Setup`. Returns (setup, a line that says how well it went). Places where the block was not
    plainly seen (hidden behind the arm, or two blobs) are left out; too few left, or a poor fit, is refused."""
    rest = kind.rests()[0]
    if not rest.turns:
        raise Refused("show the camera the table with a bar or a slab: a cylinder on its end has no direction")
    per = [[b for b in find_blobs(im) if (kind.colour is None or b.colour == kind.colour) and not b.cut] for im in pictures]
    near, found = 0.012 * pictures[0].shape[1], []
    for i, (p, blobs) in enumerate(zip(xyz, per)):
        # the block is the one patch that is somewhere else in the other pictures: what stays put is not the block
        moved = [b for b in blobs if sum(any(np.hypot(b.centre[0] - o.centre[0], b.centre[1] - o.centre[1]) < near for o in other)
                                         for j, other in enumerate(per) if j != i) < 0.5 * (len(per) - 1)]
        if len(moved) == 1:
            found.append((np.asarray(p, float), moved[0]))
    if found:
        middle = float(np.median([b.area for _, b in found]))
        found = [(p, b) for p, b in found if 0.5 * middle <= b.area <= 2.0 * middle]
    if len(found) < 8:
        raise Refused(f"the block was plainly seen in only {len(found)} of {len(pictures)} places (8 are needed): is it the only {kind.colour or 'coloured'} thing on the mat, "
                      "and is the arm out of the camera's way?")
    size = (pictures[0].shape[1], pictures[0].shape[0])
    pts = np.array([p for p, _ in found])
    cam = calibrate([b.centre for _, b in found], pts, size)
    w = float(np.mean(pts @ axes[:, 2]))
    table = Table(np.asarray(axes, float), w - rest.height / 2.0)
    from .blocks import Catalogue  # the one block alone, as it lies: nothing else may be fitted here
    alone, off, angles = Catalogue([kind]), [], []
    for p, b in found:
        s, _ = make_out(b, cam, table, alone, good=0.0)
        if s is not None and abs(s.rest.height - rest.height) < 0.5:
            off.append(float(np.hypot(*(np.array(s.uv) - table.coords(p)[:2]))))
            angles.append(s.angle)
    if len(off) < 8 or max(off) > 6.0:
        raise Refused(f"the camera does not find the block where the arm put it (off by up to {max(off, default=0):.0f} mm): is its size in blocks.json right, and did it lie flat?")
    uvs = [tuple(table.coords(p)[:2]) for p, _ in found] + [tuple(uv) for uv in extra_uv]
    setup = Setup(cam, table, w, np.asarray(rpy_ref, float), mean_angle(angles, rest.turns), area_around(uvs), side, max(off), time.time())
    return setup, f"the camera was fitted to {len(found)} places and finds the block within {max(off):.1f} mm (typically {float(np.median(off)):.1f} mm)"


# ---------------------------------------------------------------------------------------------------- making blocks out
def solid(rest, table, u, v, angle_deg, shape):
    """Corner points of a block lying this way with the middle of its footprint at (u, v): enough of them that the
    outline of the block in a picture is the outline around these points."""
    a = np.radians(angle_deg)
    along, across = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    pts = []
    if shape == "cylinder" and rest.round:
        for t in np.linspace(0.0, 2 * np.pi, 24, endpoint=False):
            for w in (0.0, rest.height):
                pts.append((u + rest.short / 2 * np.cos(t), v + rest.short / 2 * np.sin(t), table.w0 + w))
    elif shape == "cylinder":  # lying: a circle at each end, standing on the table
        for end in (-0.5, 0.5):
            for t in np.linspace(0.0, 2 * np.pi, 16, endpoint=False):
                q = np.array([u, v]) + end * rest.long * along + rest.short / 2 * np.cos(t) * across
                pts.append((q[0], q[1], table.w0 + rest.short / 2 * (1 + np.sin(t))))
    else:
        for i in (-0.5, 0.5):
            for j in (-0.5, 0.5):
                q = np.array([u, v]) + i * rest.long * along + j * rest.short * across
                for w in (0.0, rest.height):
                    pts.append((q[0], q[1], table.w0 + w))
    return (table.axes @ np.array(pts, float).T).T


def outline(cam, pts):
    """The outline in the picture of the solid with these corner points."""
    return cv2.convexHull(cam.project(pts).astype(np.float32)).reshape(-1, 2)


def _cover(cam, table, rest, shape, x, target, origin):
    hull = outline(cam, solid(rest, table, x[0], x[1], x[2], shape)) - origin
    canvas = np.zeros(target.shape, np.uint8)
    cv2.fillConvexPoly(canvas, np.round(hull * 16).astype(np.int32), 1, shift=4)
    drawn = canvas.astype(bool)
    union = int((drawn | target).sum())
    return 0.0 if union == 0 else float((drawn & target).sum()) / union


def _climb(score, x, steps, rounds=7):
    best = score(x)
    for _ in range(rounds):
        moved = True
        while moved:
            moved = False
            for i in range(len(x)):
                if steps[i] == 0.0:
                    continue
                for sign in (1.0, -1.0):
                    y = x.copy()
                    y[i] += sign * steps[i]
                    s = score(y)
                    if s > best + 1e-7:
                        x, best, moved = y, s, True
        steps = steps * 0.5
    return x, best


def make_out(blob, cam, table, cat, good=0.80):
    """The block this blob is: the kind and rest whose drawn outline covers it best, or None with the reason."""
    if blob.cut:
        return None, "it is cut off at the edge of what the camera looks at"
    bh, bw = blob.mask.shape
    pad = int(0.5 * max(bw, bh)) + 8
    x0, y0 = max(0, blob.x0 - pad), max(0, blob.y0 - pad)
    x1, y1 = min(cam.size[0], blob.x0 + bw + pad), min(cam.size[1], blob.y0 + bh + pad)
    target = np.zeros((y1 - y0, x1 - x0), bool)
    target[blob.y0 - y0:blob.y0 - y0 + bh, blob.x0 - x0:blob.x0 - x0 + bw] = blob.mask
    origin, up = np.array([x0, y0], float), table.axes[:, 2]
    tried = []
    for kind in cat.kinds:
        if kind.colour is not None and kind.colour != blob.colour:
            continue
        for rest in kind.rests():
            mid = table.coords(cam.on_plane([blob.centre], up, table.w0 + rest.height / 2.0)[0])
            score = lambda x, rest=rest, kind=kind: _cover(cam, table, rest, kind.shape, x, target, origin)  # noqa: E731
            starts = [np.array([mid[0], mid[1], 0.0])] if rest.round else [np.array([mid[0], mid[1], a]) for a in np.arange(0.0, 180.0, 15.0)]
            starts = sorted(starts, key=score, reverse=True)[:2]
            for x in starts:
                x, s = _climb(score, x, np.array([6.0, 6.0, 0.0 if rest.round else 8.0]))
                tried.append((s, kind, rest, x))
    if not tried:
        return None, f"no {blob.colour} block is in the catalogue"
    tried.sort(key=lambda t: -t[0])
    s, kind, rest, x = tried[0]
    if s < good:
        return None, f"it looks most like a {kind.name} ({rest.name}) but fits only {100 * s:.0f}%: is it touching another block, or missing from the catalogue?"
    turns = rest.turns
    return Seen(kind, rest, (float(x[0]), float(x[1])), float(x[2] % turns) if turns else 0.0, score=s, colour=blob.colour), ""


def look(bgr, cam, table, cat, region=None):
    """Everything the camera makes of one picture: (blocks seen, [(blob, why it was not made out)], blobs)."""
    cam = cam.scaled((bgr.shape[1], bgr.shape[0]))
    blobs = find_blobs(bgr, region)
    seen, unknown = [], []
    for b in blobs:
        s, why = make_out(b, cam, table, cat)
        if s is None:
            unknown.append((b, why))
        else:
            seen.append(s)
    return name_all(seen), unknown, blobs


def draw(bgr, blobs=(), seen=(), unknown=(), cam=None, table=None, width=1280):
    """The picture with what was found drawn into it, scaled down for the page."""
    im = bgr.copy()
    t = max(2, im.shape[1] // 640)
    for b in blobs:
        cs, _ = cv2.findContours(b.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cs, -1, (255, 255, 255), t, offset=(b.x0, b.y0))
    if cam is not None and table is not None:
        cam = cam.scaled((im.shape[1], im.shape[0]))
        for s in seen:
            hull = outline(cam, solid(s.rest, table, s.uv[0], s.uv[1], s.angle, s.kind.shape))
            cv2.polylines(im, [np.round(hull).astype(np.int32)], True, (0, 0, 0), 3 * t)
            cv2.polylines(im, [np.round(hull).astype(np.int32)], True, PAINT.get(s.colour, (255, 255, 255)), t)
            p = hull.min(0)
            cv2.putText(im, s.name, (int(p[0]), int(p[1]) - 4 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * t, (0, 0, 0), 3 * t)
            cv2.putText(im, s.name, (int(p[0]), int(p[1]) - 4 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * t, (255, 255, 255), t)
    for b, _ in unknown:
        cv2.putText(im, "?", (int(b.centre[0]), int(b.centre[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * t, (0, 0, 255), 2 * t)
    if width and im.shape[1] > width:
        im = cv2.resize(im, (width, int(round(im.shape[0] * width / im.shape[1]))), interpolation=cv2.INTER_AREA)
    return im
