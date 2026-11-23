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
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .blocks import Seen, name_all
from .guard import Refused

HUES = (("red", 0, 8), ("orange", 8, 22), ("yellow", 22, 39), ("green", 39, 89), ("blue", 89, 131), ("purple", 131, 170), ("red", 170, 181))  # OpenCV hue, 0 to 180
PAINT = dict(red=(40, 40, 220), orange=(30, 140, 240), yellow=(40, 220, 240), green=(60, 190, 60), blue=(200, 110, 40), purple=(170, 60, 150))  # BGR, for drawing


SCENE_CAMERA = "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:4.3:1.0-video-index0"  # the rig's left scene camera, by the port it is plugged into


class V4L2:
    """A camera that is opened for one picture and let go again, so that other programs can use it in between.
    `frames`: how many frames one picture is made of (see `grab`)."""

    def __init__(self, device=SCENE_CAMERA, size=(3840, 2160), frames=1):
        self.device, self.size, self.frames = str(device), tuple(size), int(frames)

    def grab(self):
        return grab(self.device, self.size, frames=self.frames)


def grab(device, size=(3840, 2160), warm=8, frames=1):
    """One picture (BGR) from a V4L2 camera. The first frames after opening are dark: they are thrown away.
    `frames`: take this many and give the middle value of every pixel: the grain of the sensor and the flicker of
    the lamps are different in every frame, the blocks are not (3 to 5 are plenty; nothing may move meanwhile)."""
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
        more = []
        for _ in range(max(1, int(frames)) - 1):
            ok, extra = cap.read()
            if ok and extra is not None and extra.shape == im.shape:
                more.append(extra)
        return steady([im] + more)
    finally:
        cap.release()


def steady(pictures):
    """One picture out of several of the same scene: the middle value of every pixel."""
    pictures = list(pictures)
    return pictures[0] if len(pictures) < 2 else np.median(np.stack(pictures), axis=0).astype(np.uint8)


# ---------------------------------------------------------------------------------------------------- blobs
@dataclass
class Blob:
    colour: str
    mask: np.ndarray  # True where the blob is, in its own bounding box
    x0: int
    y0: int
    centre: tuple  # in the whole picture
    cut: bool = False  # it touches the edge of the picture or of the region looked at: part of it is missing
    hue: float = -1.0  # the hue in the middle of its paint (OpenCV, 0 to 180), or -1 if nobody measured it

    @property
    def area(self):
        return int(self.mask.sum())


def _hue_range(colour):
    return [(lo, hi) for n, lo, hi in HUES if n == colour]


def hue_apart(a, b):
    """How far apart two hues are, the short way round the circle (OpenCV hues: 180 is 0 again)."""
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def colour_of(hue, hues=None, colours=None, slack=10.0):
    """The name of a hue. Plainly, the name the table `HUES` gives it. `hues` {colour: hue}: what the paints of
    this rig were measured to be (`measure_hues`); the nearest of them within 16 names it. `colours`: the colours
    there are (those of the catalogue): a hue up to `slack` outside one of them still takes its name, so that a
    yellow block in warm light is not called orange when nothing orange exists."""
    hue = float(hue) % 180.0
    if hues:
        near = min(hues, key=lambda c: hue_apart(hue, hues[c]))
        if hue_apart(hue, hues[near]) <= 16.0:
            return near
    plain = next(n for n, lo, hi in HUES if lo <= hue < hi)
    if colours and plain not in colours:
        off = {c: min(min(hue_apart(hue, lo), hue_apart(hue, hi - 1e-6)) for lo, hi in _hue_range(c)) for c in colours if _hue_range(c)}
        if off and min(off.values()) <= slack:
            return min(off, key=off.get)
    return plain


def _mat(hsv, region):
    """What the mat looks like in this picture: (how strong its colour is, how much that varies, how bright it is,
    how much that varies). The mat is taken to be the least colourful two fifths of what is looked at: on the whole
    mat that is the mat, and in a small cut-out half filled by a block it still is."""
    step = max(1, hsv.shape[1] // 480)
    sub = hsv[::step, ::step]
    px = sub[region[::step, ::step]] if region is not None else sub.reshape(-1, 3)
    if len(px) < 64:
        return None
    s, v = px[:, 1].astype(float), px[:, 2].astype(float)
    p10, ms, p40 = np.percentile(s, (10, 30, 40))
    v = v[s <= p40]
    mv = float(np.median(v))
    if region is None and ms >= 90.0:
        return None  # nothing here is as dull as a mat: no telling what the paint should stand out from
    return float(ms), float(ms - p10) / 0.76, mv, 1.4826 * float(np.median(np.abs(v - mv)))


def _hue_groups(hist, floor=0.04, gap=4):
    """Sort the hues of one patch of paint into groups, one per paint: hues that many pixels have and that follow
    one another without a break are one group, wherever the borders of the colour names are. Returns for each of
    the 180 hues the number of its group (the nearest one, for a hue hardly any pixel has), and the middle hue of
    each group."""
    hist = np.asarray(hist, float)
    sm = np.convolve(np.r_[hist[-2:], hist, hist[:2]], np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0, "valid")
    has = sm > floor * sm.max()
    if has.all():
        return np.zeros(180, int), [float(np.argmax(sm))]
    start = int(np.flatnonzero(~has)[0])  # walk the circle from a hue nobody has
    order = (np.arange(180) + start) % 180
    runs, last = [], -99
    for i, h in enumerate(order):
        if has[h]:
            if runs and i - last <= gap:
                runs[-1][1] = i
            else:
                runs.append([i, i])
            last = i
    split = []
    for i0, i1 in runs:  # two paints whose hues run into each other: cut at the dip between the two heaps
        h = sm[order[i0:i1 + 1]]
        cut = None
        if len(h) > 16:
            top = int(np.argmax(h))
            for other in range(len(h)):
                if abs(other - top) >= 9 and h[other] >= 0.15 * h[top] and h[other] == h[max(0, other - 4):other + 5].max():
                    lo, hi = sorted((top, other))
                    dip = lo + int(np.argmin(h[lo:hi + 1]))
                    if h[dip] < 0.4 * h[other] and (cut is None or h[dip] < h[cut]):
                        cut = dip
        split += [[i0, i1]] if cut is None else [[i0, i0 + cut], [i0 + cut + 1, i1]]
    group, mids = np.zeros(180, int), []
    for i0, i1 in split:
        w = hist[order[i0:i1 + 1]] + 1e-9
        mids.append(float((start + i0 + np.searchsorted(np.cumsum(w), 0.5 * w.sum())) % 180))
    for i, h in enumerate(order):
        group[h] = min(range(len(split)), key=lambda g: 0 if split[g][0] <= i <= split[g][1] else min(abs(i - split[g][0]), abs(i - split[g][1]), 180 - abs(i - split[g][0]), 180 - abs(i - split[g][1])))
    return group, mids


def _filled(mask):
    """A patch with its holes filled in."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(out, cs, -1, 1, -1)
    return out.astype(bool)


def _hull_mask(mask):
    pts = cv2.findNonZero(mask.astype(np.uint8))
    out = np.zeros(mask.shape, np.uint8)
    if pts is not None:
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out.astype(bool)


def find_blobs(bgr, region=None, s_min=None, v_min=None, hues=None, colours=None, glare=None):
    """Every patch of paint, largest first. `region`: a mask of where to look (the mat).

    Paint is what is plainly more colourful than the mat, and not much darker: the thresholds are set from the mat
    in this very picture, so that a darker room, another white balance or a mat that is not quite grey shift them
    along. (`s_min`, `v_min`: fixed thresholds on OpenCV's saturation and value instead, as it used to be.)
    One patch of paint is cut into blobs by hue, where its hues fall apart into groups (`_hue_groups`), not at the
    borders of the colour names: a yellow block whose shaded side drifts towards orange stays one blob.
    A blob is then mended: holes in it are filled, and where it is dented by something brighter than the mat (the
    lamp mirrored in a glossy face) the dent is filled up to the blob's own convex outline.
    `hues`, `colours`: see `colour_of`. `glare`: a list; if given, the mask of what is too bright to be the mat and
    has no colour (highlights, the white of the arm) is appended to it."""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (0, 0), max(0.8, w / 2000.0)), cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    mat = _mat(hsv, region) if s_min is None or v_min is None else None
    if mat is None:
        s_thr, v_thr, c_thr, bright = (115 if s_min is None else s_min), (70 if v_min is None else v_min), 0.0, 256.0
    else:
        ms, ss, mv, sv = mat
        s_thr = min(215.0, max(70.0, ms + max(0.2 * (255.0 - ms), 6.0 * ss))) if s_min is None else s_min
        v_thr = min(70.0, 0.25 * mv) if v_min is None else v_min
        c_thr, bright = 0.15 * mv, max(1.25 * mv, mv + 5.0 * sv)
    strong = (S >= s_thr) & (V >= v_thr)
    if c_thr > 0:
        strong &= S.astype(np.float32) * V >= 255.0 * c_thr  # dark pixels have a hue only by chance: paint is also this much more one colour than another, in grey levels
    shine = (V >= bright) & ~strong
    if region is not None:
        strong &= region
        shine &= region
    if glare is not None:
        glare.append(shine)
    k = max(3, int(round(w / 640.0)) | 1)
    box = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    paint = cv2.morphologyEx(cv2.morphologyEx(strong.astype(np.uint8), cv2.MORPH_OPEN, box), cv2.MORPH_CLOSE, box)
    least = 1.2e-4 * w * h
    n, lab, stats, _ = cv2.connectedComponentsWithStats(paint, 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < least:
            continue
        x, y, x1, y1 = max(0, x - 2 * k), max(0, y - 2 * k), min(w, x + bw + 2 * k), min(h, y + bh + 2 * k)  # room around it for what shines next to it
        here = lab[y:y1, x:x1] == i
        hue, sat = hsv[y:y1, x:x1, 0], strong[y:y1, x:x1] & here
        sat_of = S[y:y1, x:x1]
        group, mids = _hue_groups(np.bincount(hue[sat], minlength=180)[:180])
        near = (V[y:y1, x:x1] >= bright) & ~here if bright < 256.0 else np.zeros(here.shape, bool)  # what shines beside it or in it
        pieces = []
        for g, mid in enumerate(mids):
            mine = here & (sat & (group[hue] == g) if len(mids) > 1 else True)
            m = cv2.morphologyEx((mine | near).astype(np.uint8), cv2.MORPH_OPEN, box) if len(mids) > 1 or near.any() else mine.astype(np.uint8)
            cn, clab = cv2.connectedComponents(m, connectivity=8)
            for j in range(1, cn):
                core = (clab == j) & mine
                if core.sum() < least:
                    continue
                if mat is not None:  # its edge is where the colour is half-way between the mat's and its own, whatever its own is
                    own = float(np.median(sat_of[core]))
                    if own < s_thr + 0.16 * (255.0 - ms):
                        continue  # a stain: more colourful than the mat, but not paint
                    fine = cv2.morphologyEx((core & (sat_of >= ms + 0.5 * (own - ms))).astype(np.uint8), cv2.MORPH_OPEN, box).astype(bool)
                    core = fine if fine.sum() >= max(least, 0.6 * core.sum()) else core
                pieces.append((mid, core, (clab == j) & near & ~core))
        for j, (mid, core, lit) in enumerate(pieces):
            piece = core | (lit & _hull_mask(core)) if lit.any() else core
            piece = _filled(piece)
            for jj, (_, other, _) in enumerate(pieces):
                if jj != j:
                    piece &= ~other  # a small block in front of a large one is not a hole in it
            ys, xs = np.nonzero(piece)
            if len(xs) < least:
                continue
            px, py, pw, ph = x + xs.min(), y + ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
            piece = piece[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            edge = px <= 0 or py <= 0 or px + pw >= w or py + ph >= h
            if region is not None and not edge:
                ring = cv2.dilate(piece.astype(np.uint8), box).astype(bool) & ~region[py:py + ph, px:px + pw]
                edge = bool(ring.any())
            out.append(Blob(colour_of(mid, hues, colours), piece, int(px), int(py), (float(px + xs.mean() - xs.min()), float(py + ys.mean() - ys.min())), cut=bool(edge), hue=mid))
    return sorted(out, key=lambda b: -b.area)


def measure_hues(bgr, region=None, colours=None):
    """The hue of every paint in this picture, {colour: hue}: lay one block of each colour on the mat, well apart,
    and keep the answer in `Setup.hues`. From then on a blob is named by the nearest of these hues, not by the
    borders of the table `HUES`, which no lamp and no pot of paint keeps to. `colours`: the colours there are
    (see `colour_of`), so that the blocks are given their right names here already."""
    by = {}
    for b in find_blobs(bgr, region, colours=colours):
        if not b.cut:
            by.setdefault(b.colour, []).append(b)
    return {c: float(max(bs, key=lambda b: b.area).hue) for c, bs in by.items()}


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
    hues: dict = field(default_factory=dict)  # {colour: hue} of the paints as this camera sees them under these lamps (`measure_hues`); empty: the table `HUES` names them

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
                    angle_ref=self.angle_ref, area=[[float(u), float(v)] for u, v in self.area], side=self.side, fit_mm=self.fit_mm, when=self.when,
                    **({"hues": {str(c): float(h) for c, h in self.hues.items()}} if self.hues else {}))

    @classmethod
    def from_dict(cls, d):
        return cls(Camera.from_dict(d["camera"]), Table(np.array(d["axes"], float), float(d["w0"])), float(d["w_grasp"]), np.array(d["rpy_ref"], float), float(d["angle_ref"]),
                   [tuple(p) for p in d["area"]], str(d.get("side", "right")), float(d.get("fit_mm", 0.0)), float(d.get("when", 0.0)),
                   {str(c): float(h) for c, h in (d.get("hues") or {}).items()})

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


def settle(xyz, pictures, kind, axes, rpy_ref, extra_uv=(), side="right", refine=True):
    """From the places the arm put ONE block (`xyz`: where the gripper held it, lying flat) and a picture of each,
    work out the whole `Setup`. Returns (setup, a line that says how well it went). Places where the block was not
    plainly seen (hidden behind the arm, or two blobs) are left out; too few left, or a poor fit, is refused.
    `refine`: fit the camera a second time, to where the block's middle must have shown rather than to the middle of
    its blob (kept only if the block is then found at least as well)."""
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
    alone = Catalogue([kind])

    def check(cam):
        """Look for the block in every picture with this camera: how far off it is found, which way it points, and
        by how much the middle of its outline in the picture is not the picture of its own middle."""
        off, angles, shift = [], [], {}
        for i, (p, b) in enumerate(found):
            s, _ = make_out(b, cam, table, alone, good=0.0)
            if s is not None and abs(s.rest.height - rest.height) < 0.5:
                off.append(float(np.hypot(*(np.array(s.uv) - table.coords(p)[:2]))))
                angles.append(s.angle)
                m = cv2.moments(outline(cam, solid(s.rest, table, s.uv[0], s.uv[1], s.angle, kind.shape)))
                shift[i] = np.array([m["m10"], m["m01"]]) / max(m["m00"], 1e-9) - cam.project([table.point(s.uv[0], s.uv[1], table.w0 + rest.height / 2.0)])[0]
        return off, angles, shift

    off, angles, shift = check(cam)
    if refine and len(shift) >= 8:
        # the middle of a blob is not exactly where the middle of the block shows (the near end of a block is larger in the picture than the far end):
        # now that there is a camera, it says by how much, and the camera is fitted again to the corrected middles
        again = calibrate([np.array(b.centre) - shift.get(i, 0.0) for i, (_, b) in enumerate(found)], pts, size)
        off2, angles2, _ = check(again)
        if len(off2) >= len(off) and off2 and max(off2) <= max(off):
            cam, off, angles = again, off2, angles2
    if len(off) < 8 or max(off) > 6.0:
        raise Refused(f"the camera does not find the block where the arm put it (off by up to {max(off, default=0):.0f} mm): is its size in blocks.json right, and did it lie flat?")
    uvs = [tuple(table.coords(p)[:2]) for p, _ in found] + [tuple(uv) for uv in extra_uv]
    setup = Setup(cam, table, w, np.asarray(rpy_ref, float), mean_angle(angles, rest.turns), area_around(uvs), side, max(off), time.time())
    return setup, f"the camera was fitted to {len(found)} places and finds the block within {max(off):.1f} mm (typically {float(np.median(off)):.1f} mm)"


# ---------------------------------------------------------------------------------------------------- making blocks out
@lru_cache(maxsize=256)
def _corners(rest, shape):
    """The corner points of a block lying this way, in its own terms: (along its long side, across it, up)."""
    pts = []
    if shape == "cylinder" and rest.round:
        for t in np.linspace(0.0, 2 * np.pi, 24, endpoint=False):
            for w in (0.0, rest.height):
                pts.append((rest.short / 2 * np.cos(t), rest.short / 2 * np.sin(t), w))
    elif shape == "cylinder":  # lying: a circle at each end, standing on the table
        for end in (-0.5, 0.5):
            for t in np.linspace(0.0, 2 * np.pi, 16, endpoint=False):
                pts.append((end * rest.long, rest.short / 2 * np.cos(t), rest.short / 2 * (1 + np.sin(t))))
    else:
        for i in (-0.5, 0.5):
            for j in (-0.5, 0.5):
                for w in (0.0, rest.height):
                    pts.append((i * rest.long, j * rest.short, w))
    return np.array(pts, float)


def solid(rest, table, u, v, angle_deg, shape):
    """Corner points of a block lying this way with the middle of its footprint at (u, v): enough of them that the
    outline of the block in a picture is the outline around these points."""
    a = 0.0 if rest.round else np.radians(angle_deg)  # what is round has no direction
    c, s = np.cos(a), np.sin(a)
    own = _corners(rest, shape)
    pts = np.stack([u + own[:, 0] * c - own[:, 1] * s, v + own[:, 0] * s + own[:, 1] * c, table.w0 + own[:, 2]])
    return (table.axes @ pts).T


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


LOST = 0.35  # a part of an outline that lies behind something else counts this much as if it lay on bare mat: it may be there, but nobody saw it
FACES = 0.25  # how much the faces of a block count beside its outline when two explanations cover a blob about equally well
CLOSE = 0.08  # explanations whose covers are this close are told apart by their faces


class _Patch:
    """One blob cut out of the picture with room around it, ready for outlines to be laid on it. A large blob is
    shrunk to `side` pixels (the search does not need more, and a 4K picture then costs what a small one does).
    `hidden`: where in the picture something else is that may hide a block (other blobs, the arm): an outline that
    reaches in there is not counted as wrong there, only as less certain (`LOST`)."""

    def __init__(self, blob, cam, table, hidden=None, picture=None, side=96):
        bh, bw = blob.mask.shape
        pad = int(0.5 * max(bw, bh)) + 8
        x0, y0 = max(0, blob.x0 - pad), max(0, blob.y0 - pad)
        x1, y1 = min(cam.size[0], blob.x0 + bw + pad), min(cam.size[1], blob.y0 + bh + pad)
        target = np.zeros((y1 - y0, x1 - x0), np.uint8)
        target[blob.y0 - y0:blob.y0 - y0 + bh, blob.x0 - x0:blob.x0 - x0 + bw] = blob.mask
        self.cam, self.table, self.origin, self.up = cam, table, np.array([x0, y0], float), table.axes[:, 2]
        self.scale = 1.0 if not side or max(bw, bh) <= side else side / float(max(bw, bh))
        small = (lambda m: m) if self.scale == 1.0 else (lambda m: cv2.resize(m, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA))
        self.T = (small(target * 255) >= 128).astype(np.uint8)
        self.H = None
        if hidden is not None:
            H = (small(hidden[y0:y1, x0:x1].astype(np.uint8) * 255) >= 128).astype(np.uint8) & (1 - self.T)
            self.H = H if H.any() else None
        self.grey = None if picture is None else small(cv2.cvtColor(picture[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)).astype(np.float32)
        self.n, self.room = max(1, cv2.countNonZero(self.T)), None
        R, _ = cv2.Rodrigues(np.asarray(cam.rvec, float))
        self.eye = table.coords(-R.T @ np.asarray(cam.tvec, float).reshape(3))

    def px(self, pts):
        """Where points of the arm's world show in this patch."""
        return (self.cam.project(pts) - self.origin + 0.5) * self.scale - 0.5

    def draw(self, rest, shape, x, into=None):
        canvas = np.zeros(self.T.shape, np.uint8) if into is None else into
        hull = cv2.convexHull(self.px(solid(rest, self.table, x[0], x[1], x[2], shape)).astype(np.float32)).reshape(-1, 2)
        cv2.fillConvexPoly(canvas, np.round(hull * 16).astype(np.int32), 1, shift=4)
        return canvas

    def cover(self, D, T=None, H=None):
        """How well what is drawn covers the blob: what they share over what either takes up (1 = exactly)."""
        T, H = self.T if T is None else T, self.H if H is None else H
        union = cv2.countNonZero(cv2.bitwise_or(D, T))
        if H is not None:
            union -= (1.0 - LOST) * cv2.countNonZero(cv2.bitwise_and(D, H))
        return 0.0 if union <= 0 else cv2.countNonZero(cv2.bitwise_and(D, T)) / union

    def within(self, D):
        """How much of the blob one block takes up if it is one of several in it: its outline must lie inside
        (a pixel over the rim is forgiven: the rim is not that exact)."""
        if self.room is None:
            wide = cv2.dilate(self.T, np.ones((3, 3), np.uint8))
            self.room = wide if self.H is None else cv2.bitwise_or(wide, self.H)
        out = cv2.countNonZero(D) - cv2.countNonZero(cv2.bitwise_and(D, self.room))
        return (cv2.countNonZero(cv2.bitwise_and(D, self.T)) - 2.0 * out) / self.n

    def spot(self, p, height):
        """Where on the table a block of this height stands whose middle shows at the point `p` of the patch."""
        full = (np.asarray(p, float) + 0.5) / self.scale - 0.5 + self.origin
        return self.table.coords(self.cam.on_plane([full], self.up, self.table.w0 + height / 2.0)[0])[:2]

    def faces(self, blocks):
        """How much of the light and dark inside the blob the faces of these blocks [(rest, shape, x)] explain (0 to
        1): a box shows its top and one or two sides, each lit differently, and where the borders between them run
        tells a bar standing on its end from one lying towards the camera, which have nearly the same outline.
        0 if no picture was given, or the blob is evenly bright (then the faces say nothing)."""
        if self.grey is None:
            return 0.0
        label, at = np.zeros(self.T.shape, np.uint8), 1
        for rest, shape, x in sorted(blocks, key=lambda b: -np.hypot(b[2][0] - self.eye[0], b[2][1] - self.eye[1])):  # the nearest block last: it hides the others
            a = np.radians(x[2])
            mid, al, ac = np.array([x[0], x[1]]), np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
            w0, h, flat = self.table.w0, rest.height, []
            if shape == "cylinder":  # a round side has no borders: it is cut into three strips as the camera sees them, so that it can be light on one side and dark on the other
                r, n = rest.short / 2, 18
                if rest.round:
                    c, e1, e2, e3, ends = np.array([x[0], x[1], w0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, h]), [1.0]
                else:
                    c, e1, e2, e3 = np.r_[mid - 0.5 * rest.long * al, w0 + r], np.r_[ac, 0.0], np.array([0, 0, 1.0]), np.r_[rest.long * al, 0.0]
                    ends = [e for e in (0.0, 1.0) if (2 * e - 1) * al @ (self.eye[:2] - mid) > 0]
                see = self.eye - (c + 0.5 * e3)
                t0 = np.arctan2(see @ e2, see @ e1)
                for i in range(n):
                    t = t0 - np.pi / 2 + np.pi * np.array([i, i + 1]) / n
                    q = [c + r * (np.cos(v) * e1 + np.sin(v) * e2) for v in t]
                    flat.append((i * 3 // n, [q[0], q[1], q[1] + e3, q[0] + e3]))
                for e in ends:
                    flat.append((3, [c + e * e3 + r * (np.cos(v) * e1 + np.sin(v) * e2) for v in np.linspace(0.0, 2 * np.pi, 20, endpoint=False)]))
            else:
                q = lambda i, j, w, mid=mid, al=al, ac=ac, rest=rest, w0=w0: np.r_[mid + i * 0.5 * rest.long * al + j * 0.5 * rest.short * ac, w0 + w]  # noqa: E731
                for sgn in (-1, 1):
                    if sgn * al @ (self.eye[:2] - (mid + sgn * 0.5 * rest.long * al)) > 0:
                        flat.append((len(flat), [q(sgn, -1, 0), q(sgn, 1, 0), q(sgn, 1, h), q(sgn, -1, h)]))
                    if sgn * ac @ (self.eye[:2] - (mid + sgn * 0.5 * rest.short * ac)) > 0:
                        flat.append((len(flat), [q(-1, sgn, 0), q(1, sgn, 0), q(1, sgn, h), q(-1, sgn, h)]))
                flat.append((len(flat), [q(-1, -1, h), q(1, -1, h), q(1, 1, h), q(-1, 1, h)]))
            hull = cv2.convexHull(self.px(solid(rest, self.table, x[0], x[1], x[2], shape)).astype(np.float32)).reshape(-1, 2)
            cv2.fillConvexPoly(label, np.round(hull * 16).astype(np.int32), at + 4, shift=4)
            for i, f in flat:
                cv2.fillConvexPoly(label, np.round(self.px((self.table.axes @ np.array(f, float).T).T) * 16).astype(np.int32), at + i, shift=4)
            at += 5
        label[self.T == 0] = 0
        inside = label > 0
        if inside.sum() < 30:
            return 0.0
        g = self.grey[inside]
        g = np.clip(g, *np.percentile(g, (3, 97)))
        total = float(((g - g.mean()) ** 2).sum())
        if total < 16.0 * len(g):  # evenly bright to within 4 grey levels
            return 0.0
        lab = label[inside]
        left = sum(float(((g[lab == i] - g[lab == i].mean()) ** 2).sum()) for i in np.unique(lab))
        return max(0.0, 1.0 - left / total)


def _rests(blob, cat):
    return [(kind, rest) for kind in cat.kinds if kind.colour is None or kind.colour == blob.colour for rest in kind.rests()]


def _turned(n=12):
    return np.arange(0.0, 180.0, 180.0 / n)


def _alone(patch, blob, cat):
    """Every kind of block and every way it can lie, laid on the blob as well as it goes: [(cover, kind, rest, x)], best first."""
    tried = []
    for kind, rest in _rests(blob, cat):
        mid = patch.table.coords(patch.cam.on_plane([blob.centre], patch.up, patch.table.w0 + rest.height / 2.0)[0])
        score = lambda x, rest=rest, kind=kind: patch.cover(patch.draw(rest, kind.shape, x))  # noqa: E731
        starts = [np.array([mid[0], mid[1], 0.0])] if rest.round else [np.array([mid[0], mid[1], a]) for a in _turned()]
        best = None
        for x in sorted(starts, key=score, reverse=True)[:2]:
            x, s = _climb(score, x, np.array([6.0, 6.0, 0.0 if rest.round else 8.0]))
            if best is None or s > best[0]:
                best = (s, kind, rest, x)
        tried.append(best)
    return sorted(tried, key=lambda t: -t[0])


def _by_faces(patch, tried):
    """Put the explanations that cover the blob about as well as the best one in the order their faces give.
    Returns [(cover, kind, rest, x, cover and faces together)], best first."""
    if not tried:
        return []
    near = [t for t in tried if t[0] >= tried[0][0] - CLOSE]
    if len(near) > 1 and patch.grey is not None:
        near = [t + (t[0] + FACES * patch.faces([(t[2], t[1].shape, t[3])]),) for t in near]
        if max(t[4] - t[0] for t in near) < 0.02:  # the faces say nothing (an evenly painted picture): the covers decide alone
            near = [t[:4] + (t[0],) for t in near]
    else:
        near = [t + (t[0],) for t in near]
    return sorted(near, key=lambda t: -t[4]) + [t + (t[0],) for t in tried if t[0] < tried[0][0] - CLOSE]


def _foot(rest, x):
    """The footprint of a block on the table, as a polygon [(u, v)]."""
    if rest.round:
        return np.array([[x[0] + rest.short / 2 * np.cos(t), x[1] + rest.short / 2 * np.sin(t)] for t in np.linspace(0.0, 2 * np.pi, 16, endpoint=False)], np.float32)
    a = np.radians(x[2])
    al, ac = np.array([np.cos(a), np.sin(a)]) * rest.long / 2, np.array([-np.sin(a), np.cos(a)]) * rest.short / 2
    return np.array([np.array(x[:2]) + i * al + j * ac for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1))], np.float32)


def gap_mm(a, b):
    """How far apart two seen blocks are on the table, footprint to footprint (0 if they touch or overlap)."""
    f, g = _foot(a.rest, (a.uv[0], a.uv[1], a.angle)), _foot(b.rest, (b.uv[0], b.uv[1], b.angle))
    if cv2.intersectConvexConvex(f, g)[0] > 0:
        return 0.0
    return float(min(-cv2.pointPolygonTest(p.reshape(-1, 1, 2), (float(q[0]), float(q[1])), True) for p, other in ((f, g), (g, f)) for q in other))


def crowd(seen, clear_mm=30.0):
    """Mark as `crowded` every seen block that has another one within `clear_mm` of its footprint: the jaws that
    come down beside it would meet the neighbour. Returns the close pairs [(block, block, gap in mm)]. A mark that
    is already set (two blocks made out of one blob) stays."""
    close = []
    for i, a in enumerate(seen):
        for b in seen[i + 1:]:
            if np.hypot(a.uv[0] - b.uv[0], a.uv[1] - b.uv[1]) > clear_mm + a.rest.long + b.rest.long:
                continue
            d = gap_mm(a, b)
            if d < clear_mm:
                a.crowded = b.crowded = True
                close.append((a, b, d))
    return close


def _centres(mask, k):
    """The middles of k heaps of the blob's pixels (a plain k-means from starts spread along the blob's long side)."""
    ys, xs = np.nonzero(mask)
    pts = np.c_[xs, ys].astype(float)
    pts = pts[::max(1, len(pts) // 1500)]
    d = pts - pts.mean(0)
    along = d @ np.linalg.svd(d, full_matrices=False)[2][0]
    c = np.array([pts[np.argsort(along)[int((i + 0.5) / k * len(pts))]] for i in range(k)])
    for _ in range(8):
        own = np.argmin(((pts[:, None] - c[None]) ** 2).sum(2), 1)
        c = np.array([pts[own == i].mean(0) if (own == i).any() else c[i] for i in range(k)])
    return c


def _together(patch, blob, cat, most=3, keep=8):
    """Explain the blob as two blocks of the catalogue that touch (or one hides part of the other), or as three:
    the union of their outlines must cover it. Found one block at a time (the one that takes up most of the blob
    while staying inside it, then the one that covers best what is left), then all shifted and turned together.
    Returns [(cover, [(kind, rest, x), ...], cover and faces together)], best first: explanations with different blocks in them."""
    rests = _rests(blob, cat)
    firsts = []
    spots = [c for k in (2, 3) for c in _centres(patch.T, k)]
    for kind, rest in rests:
        score = lambda x, rest=rest, kind=kind: patch.within(patch.draw(rest, kind.shape, x))  # noqa: E731
        starts = []
        for c in spots:  # from each place the way it lies best there: one place must not crowd out the others
            u, v = patch.spot(c, rest.height)
            starts.append(max([np.array([u, v, 0.0])] if rest.round else [np.array([u, v, a]) for a in _turned()], key=score))
        for x in sorted(starts, key=score, reverse=True)[:4]:
            x, s = _climb(score, x, np.array([4.0, 4.0, 0.0 if rest.round else 6.0]), rounds=5)
            if s > 0.15:
                firsts.append((s, kind, rest, x))
    grown = []
    for _, kind, rest, x in sorted(firsts, key=lambda t: -t[0]):  # the best few, but not all of them the same block in the same place under different names
        same = sum(np.hypot(*(y[:2] - x[:2])) < 8.0 and (r is rest or abs(r.long - rest.long) < 8.0 and hue_apart(y[2], x[2]) % 180.0 < 20.0) for _, [(_, r, y)] in grown)
        if same < 2 and len(grown) < keep:
            grown.append((patch.cover(patch.draw(rest, kind.shape, x)), [(kind, rest, x)]))
    best = []
    for _ in range(2, most + 1):
        nxt = []
        for _, blocks in grown:
            D = np.zeros(patch.T.shape, np.uint8)
            for kind, rest, x in blocks:
                patch.draw(rest, kind.shape, x, D)
            left = cv2.bitwise_and(patch.T, 1 - D)
            core = cv2.erode(left, np.ones((3, 3), np.uint8))
            core = core if cv2.countNonZero(core) >= 0.04 * patch.n else left
            if cv2.countNonZero(core) < 0.04 * patch.n:
                continue
            n, lab, stats, cen = cv2.connectedComponentsWithStats(core, 8)
            c = cen[1 + int(np.argmax(stats[1:, 4]))]
            H = D if patch.H is None else cv2.bitwise_or(D, patch.H)
            found = []
            for kind, rest in rests:
                score = lambda x, rest=rest, kind=kind, left=left, H=H: patch.cover(patch.draw(rest, kind.shape, x), left, H)  # noqa: E731
                u, v = patch.spot(c, rest.height)
                starts = [np.array([u, v, 0.0])] if rest.round else [np.array([u, v, a]) for a in _turned()]
                x = max(starts, key=score)
                found.append((score(x), kind, rest, x, score))
            for _, kind, rest, x, score in sorted(found, key=lambda t: -t[0])[:4]:
                x, _ = _climb(score, x, np.array([4.0, 4.0, 0.0 if rest.round else 6.0]), rounds=4)
                both = blocks + [(kind, rest, x)]
                nxt.append((_cover_all(patch, both, np.concatenate([b[2] for b in both])), both))
        nxt.sort(key=lambda t: -t[0])
        done, names = [], set()
        for _, blocks in nxt:  # the best of each different set of blocks, moved together until they fit
            name = tuple(sorted((k.name, r.name, round(x[0] / 12.0), round(x[1] / 12.0)) for k, r, x in blocks))  # the same blocks the other way round are another explanation
            if name in names or len(done) >= 4:
                continue
            names.add(name)
            steps = np.concatenate([[3.0, 3.0, 0.0 if r.round else 4.0] for _, r, _ in blocks])
            x, s = _climb(lambda x, blocks=blocks: _cover_all(patch, blocks, x), np.concatenate([b[2] for b in blocks]), steps, rounds=6)
            done.append((s, [(k, r, x[3 * i:3 * i + 3]) for i, (k, r, _) in enumerate(blocks)]))
        best = sorted(best + done, key=lambda t: -t[0])
        grown = done[:3]
        if best and best[0][0] >= 0.9:
            break
    best = [t + (t[0],) for t in best]
    if len(best) > 1 and patch.grey is not None:  # as for one block: what covers about equally well is told apart by its faces
        near = [t for t in best if t[0] >= best[0][0] - CLOSE]
        both = [t[0] + FACES * patch.faces([(r, k.shape, x) for k, r, x in t[1]]) for t in near]
        if max(b - t[0] for b, t in zip(both, near)) >= 0.02:
            best = sorted([t[:2] + (b,) for b, t in zip(both, near)], key=lambda t: -t[2]) + best[len(near):]
    return best


def _cover_all(patch, blocks, x):
    """The cover of several blocks drawn together; blocks that stand in one another on the table lose by it."""
    D = np.zeros(patch.T.shape, np.uint8)
    feet = []
    for i, (kind, rest, _) in enumerate(blocks):
        patch.draw(rest, kind.shape, x[3 * i:3 * i + 3], D)
        feet.append(_foot(rest, x[3 * i:3 * i + 3]))
    inside = sum(cv2.intersectConvexConvex(f, g)[0] for i, f in enumerate(feet) for g in feet[i + 1:])
    return patch.cover(D) - 0.002 * inside


def _polish(blob, cam, table, hidden, blocks):
    """The last touch on the picture as it is, not shrunk: [(kind, rest, x)] moved a little until they cover best."""
    patch = _Patch(blob, cam, table, hidden, side=None)
    steps = np.concatenate([[0.8, 0.8, 0.0 if r.round else 1.0] for _, r, _ in blocks])
    x, s = _climb(lambda x: _cover_all(patch, blocks, x), np.concatenate([b[2] for b in blocks]), steps, rounds=4)
    return s, [(k, r, x[3 * i:3 * i + 3]) for i, (k, r, _) in enumerate(blocks)]


def _seen(blob, kind, rest, x, score, margin=1.0, rival="", crowded=False):
    return Seen(kind, rest, (float(x[0]), float(x[1])), float(x[2] % rest.turns) if rest.turns else 0.0, score=float(score), colour=blob.colour, crowded=crowded, margin=float(margin), rival=rival)


def make_out_all(blob, cam, table, cat, good=0.80, hidden=None, picture=None, most=3, sure=0.93):
    """The block or blocks this blob is: ([Seen, ...], '') or ([], the reason).

    First as ONE block, as `make_out` does. If no single block covers the blob well (`sure`), also as two blocks of
    the catalogue that touch, then as three (`most`): several are believed only if together they cover the blob
    well and clearly better than one does. A blob with no dent in its outline is more likely ONE thing that is
    not in the catalogue than two blocks lying neatly side by side: there, several must cover it almost exactly.
    Blocks made out together are marked `crowded`.
    `hidden`: a mask of the whole picture, True where something else is that may hide part of this blob's block
    (`look` passes the other blobs and what glares). `picture`: the picture itself; with it, explanations that
    cover the blob equally well are told apart by where the borders between the block's faces run.
    Every `Seen` says in `margin` how far ahead it was of the next best explanation (`rival`)."""
    if blob.cut:
        return [], "it is cut off at the edge of what the camera looks at"
    patch = _Patch(blob, cam, table, hidden, picture)
    tried = _by_faces(patch, _alone(patch, blob, cat))
    if not tried:
        return [], f"no {blob.colour} block is in the catalogue"
    s, kind, rest, x, both = tried[0]
    other = next((t for t in tried[1:] if t[1] is not kind or t[2] is not rest), None)
    margin, rival = (both - other[4], f"{other[1].name} ({other[2].name})") if other is not None else (1.0, "")
    dented = patch.n < 0.95 * cv2.countNonZero(_hull_mask(patch.T).astype(np.uint8))  # blocks that touch mostly leave a dent in the outline where they meet
    many = _together(patch, blob, cat, most) if s < sure and most > 1 else []
    if many and many[0][0] >= (max(good, 0.84) if dented else 0.94) and many[0][0] >= s + 0.04:
        s2, blocks, rank = many[0]
        if patch.scale < 1.0:
            s2, blocks = _polish(blob, cam, table, hidden, blocks)
        if len(many) > 1 and many[1][2] > s:
            margin, rival = rank - many[1][2], " and ".join(f"{k.name} ({r.name})" for k, r, _ in many[1][1])
        else:
            margin, rival = many[0][0] - s, f"{kind.name} ({rest.name})"
        return [_seen(blob, k, r, y, s2, margin, rival, crowded=True) for k, r, y in blocks], ""
    if s < good:
        return [], f"it looks most like a {kind.name} ({rest.name}) but fits only {100 * s:.0f}%: is it touching another block, or missing from the catalogue?"
    if patch.scale < 1.0:
        s, ((_, _, x),) = _polish(blob, cam, table, hidden, [(kind, rest, x)])
    return [_seen(blob, kind, rest, x, s, margin, rival)], ""


def make_out(blob, cam, table, cat, good=0.80, hidden=None, picture=None):
    """The block this blob is: the kind and rest whose drawn outline covers it best, or None with the reason.
    One block only: `make_out_all` also makes out blocks that touch. `hidden`, `picture`: see there."""
    seen, why = make_out_all(blob, cam, table, cat, good, hidden, picture, most=1)
    return (seen[0], "") if seen else (None, why)


def look(bgr, cam, table, cat, region=None, clear_mm=30.0, hues=None, most=3):
    """Everything the camera makes of one picture: (blocks seen, [(blob, why it was not made out)], blobs).
    Blocks that touch are made out together where that explains their blob (up to `most` in one blob); a seen
    block with another within `clear_mm` of it is marked `crowded` (see `crowd`). `hues`: see `colour_of`."""
    cam = cam.scaled((bgr.shape[1], bgr.shape[0]))
    glare = []
    blobs = find_blobs(bgr, region, hues=hues, colours={k.colour for k in cat.kinds if k.colour}, glare=glare)
    taken = glare[0].copy()
    for b in blobs:
        taken[b.y0:b.y0 + b.mask.shape[0], b.x0:b.x0 + b.mask.shape[1]] |= b.mask
    seen, unknown = [], []
    for b in blobs:
        found, why = make_out_all(b, cam, table, cat, hidden=taken, picture=bgr, most=most)
        seen += found
        if not found:
            unknown.append((b, why))
    crowd(seen, clear_mm)
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
