// The designer: lay out what the arm should build. The program (design.py) decides whether it can be built and
// whether every step stands; this page draws, lets blocks be dragged in, turned and laid differently, and asks.
//
// Frames. A design is in millimetres: x forward (away from the robot's base), y to the robot's left, z up, the
// origin on the spot it will be built on. three.js has y up, so a design point (x, y, z) is drawn at (x, z, -y).
// Blocks fall straight down until they meet something (`settle`), which is also how the arm puts them down.
import * as THREE from "three";
import { OrbitControls } from "/web/vendor/OrbitControls.js";
import { RoomEnvironment } from "/web/vendor/RoomEnvironment.js";

const TOKEN = window.TOKEN, $ = id => document.getElementById(id);
const PAINT = {green: "#3e9f58", red: "#d8453a", blue: "#3a72cf", yellow: "#efc12f", orange: "#ee8a2e", purple: "#8f4bb5"};
const ORDER = ["green", "red", "blue", "yellow", "orange", "purple"];
const MAT = 200, GRID = 5, CENTRE_SNAP = 9, KEY = "assemblyloop.designer.v1";
const W = (x, y, z) => new THREE.Vector3(x, z, -y);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

// ---------------------------------------------------------------------------------------------------- state
let cat = null, KIND = new Map(), GROUPS = [];
let design = {name: "", note: "", blocks: []}, sel = null, report = null, upto = null, savedAs = null, dirty = false;
const hist = [], fut = [];
let checkSeq = 0, checkTimer = null, busy = false;

function say(text, good) { const m = $("msg"); m.textContent = text || ""; m.hidden = !text; m.style.borderColor = good ? "var(--ok)" : ""; m.style.color = good ? "var(--ok)" : ""; clearTimeout(say.t); if (text) say.t = setTimeout(() => { m.hidden = true; }, good ? 3500 : 9000); }
$("msg").onclick = () => { $("msg").hidden = true; };
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json", "X-Token": TOKEN}, body: JSON.stringify(body)});
  const j = await r.json();
  if (j.error) throw new Error(j.error);
  return j;
}
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const byId = id => design.blocks.find(b => b.id === id);
const idx = id => design.blocks.findIndex(b => b.id === id);
const kindOf = b => KIND.get(b.kind);
const restOf = b => { const k = kindOf(b); return k && (k.rests.find(r => r.name === b.rest) || k.rests[0]); };
const word = k => k.name.replace(k.colour || "", "").trim() || k.name;
function nextId() { let n = 0; design.blocks.forEach(b => { const m = /^b(\d+)$/.exec(b.id); if (m) n = Math.max(n, +m[1]); }); return "b" + (n + 1); }

// ---------------------------------------------------------------------------------------------------- footprints (as design.py)
function rect(cx, cy, long, short, deg) {
  const a = deg * Math.PI / 180, u = [Math.cos(a), Math.sin(a)], v = [-Math.sin(a), Math.cos(a)];
  return [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([s, t]) => [cx + s * long / 2 * u[0] + t * short / 2 * v[0], cy + s * long / 2 * u[1] + t * short / 2 * v[1]]);
}
function circle(cx, cy, d, n = 16) { return Array.from({length: n}, (_, i) => { const t = 2 * Math.PI * i / n; return [cx + d / 2 * Math.cos(t), cy + d / 2 * Math.sin(t)]; }); }
function area(p) { let s = 0; for (let i = 0; i < p.length; i++) { const [x1, y1] = p[i], [x2, y2] = p[(i + 1) % p.length]; s += x1 * y2 - x2 * y1; } return p.length < 3 ? 0 : s / 2; }
function clip(subject, clipper) {
  let out = subject.slice();
  for (let k = 0; k < clipper.length && out.length; k++) {
    const a = clipper[k], b = clipper[(k + 1) % clipper.length], e = [b[0] - a[0], b[1] - a[1]];
    const inside = p => e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0]) >= -1e-9;
    const cut = (p, q) => { const d = [q[0] - p[0], q[1] - p[1]], den = e[0] * d[1] - e[1] * d[0]; const t = Math.abs(den) < 1e-12 ? 0 : (e[0] * (a[1] - p[1]) - e[1] * (a[0] - p[0])) / den; const c = Math.min(1, Math.max(0, t)); return [p[0] + c * d[0], p[1] + c * d[1]]; };
    const pts = out; out = [];
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i], q = pts[(i + 1) % pts.length];
      if (inside(q)) { if (!inside(p)) out.push(cut(p, q)); out.push(q); } else if (inside(p)) out.push(cut(p, q));
    }
  }
  return out;
}
function foot(b) { const r = restOf(b); return r.round ? circle(b.x, b.y, r.long) : rect(b.x, b.y, r.long, r.short, b.angle); }
const overlap = (a, b) => area(clip(foot(a), foot(b)));
const top = b => b.z + restOf(b).height;

function dropZ(b, without) {  // where b comes to rest if it is let down from above here
  let z = 0;
  for (const o of design.blocks) if (o.id !== without && o.id !== b.id && overlap(b, o) > 1) z = Math.max(z, top(o));
  return z;
}
function settle() {  // everything falls until it meets something, lowest first (design.settle)
  const done = [];
  for (const b of design.blocks.slice().sort((p, q) => p.z - q.z || idx(p.id) - idx(q.id))) {
    let z = 0; for (const o of done) if (overlap(b, o) > 1) z = Math.max(z, top(o));
    b.z = Math.round(z * 100) / 100; done.push(b);
  }
}
function fixOrder() {  // every block after the blocks under it: the arm brings each one down from above
  const B = design.blocks, before = new Map(B.map(b => [b.id, B.filter(o => o !== b && overlap(b, o) > 1 && b.z >= top(o) - 0.5).map(o => o.id)]));
  const out = [], left = B.slice();
  while (left.length) {
    const k = left.findIndex(b => before.get(b.id).every(id => out.some(o => o.id === id)));
    out.push(...left.splice(k < 0 ? 0 : k, 1));
  }
  const moved = out.some((b, i) => b !== B[i]);
  design.blocks = out;
  return moved;
}

// ---------------------------------------------------------------------------------------------------- changes, undo, check
function snapshot() { return JSON.stringify(design); }
function change(fn, note) {
  hist.push(snapshot()); if (hist.length > 150) hist.shift(); fut.length = 0;
  fn(); settle();
  if (fixOrder() && !note) note = "The order was changed so that every block comes after what is under it.";
  if (note) say(note, true);
  upto = null; dirty = true; afterChange();
}
function undo() { if (!hist.length) return; fut.push(snapshot()); design = JSON.parse(hist.pop()); if (sel && !byId(sel)) sel = null; dirty = true; afterChange(); }
function redo() { if (!fut.length) return; hist.push(snapshot()); design = JSON.parse(fut.pop()); if (sel && !byId(sel)) sel = null; dirty = true; afterChange(); }
function afterChange() {
  try { localStorage.setItem(KEY, JSON.stringify({design, savedAs, dirty})); } catch (e) { /* a private window: nothing kept, nothing lost */ }
  draw(); renderUI(); askCheck();
}
function askCheck() {
  clearTimeout(checkTimer);
  if (!design.blocks.length) { report = null; busy = false; renderUI(); return; }
  busy = true; renderVerdict();
  checkTimer = setTimeout(async () => {
    const me = ++checkSeq;
    try { const r = await api("/api/design/check", {design}); if (me === checkSeq) { report = r; busy = false; draw(); renderUI(); } }
    catch (e) { if (me === checkSeq) { busy = false; report = null; say("The check did not answer: " + e.message); renderUI(); } }
  }, 140);
}
const rep = id => report && report.blocks[idx(id)] && report.blocks[idx(id)].id === id ? report.blocks[idx(id)] : null;
const stepRep = k => report && report.steps[k] && report.steps[k].id === (design.blocks[k] || {}).id ? report.steps[k] : null;
function statusOf(k) {
  const b = design.blocks[k], r = rep(b.id), s = stepRep(k);
  if (!r) return "";
  if (r.problems.length) return "bad";
  if (r.warnings.length || (s && s.stands && s.tilt_deg < cat.shaky_deg)) return "warn";
  return "ok";
}

// ---------------------------------------------------------------------------------------------------- the 3D view
const host = $("view"), canvas = $("gl");
let renderer, scene, camera, controls, need = true, meshes = new Map(), ghost = null, zoneGroup, labelEls = [];
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2(), ground = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const geoCache = new Map(), matCache = new Map();
function geometry(r) {
  const key = `${r.round}:${r.long}:${r.short}:${r.height}`;
  if (!geoCache.has(key)) geoCache.set(key, r.round ? new THREE.CylinderGeometry(r.long / 2, r.long / 2, r.height, 40) : new THREE.BoxGeometry(r.long, r.height, r.short));
  return geoCache.get(key);
}
function material(colour, opacity = 1) {
  const key = colour + ":" + opacity;
  if (!matCache.has(key)) matCache.set(key, new THREE.MeshStandardMaterial({color: PAINT[colour] || "#9aa2af", roughness: 0.55, metalness: 0.0, transparent: opacity < 1, opacity, depthWrite: opacity >= 0.99}));
  return matCache.get(key);
}
function blockMesh(b, opacity) {
  const r = restOf(b), m = new THREE.Mesh(geometry(r), material(kindOf(b).colour, opacity));
  m.position.copy(W(b.x, b.y, b.z + r.height / 2)); m.rotation.y = b.angle * Math.PI / 180;
  m.castShadow = opacity >= 0.99; m.receiveShadow = true; m.userData.id = b.id;
  return m;
}

function start3D() {
  renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: true});
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace; renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  scene = new THREE.Scene();
  scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
  scene.add(new THREE.HemisphereLight(0xeef2ff, 0x1a1d24, 0.4));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(-180, 520, 260); sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048); Object.assign(sun.shadow.camera, {left: -260, right: 260, top: 260, bottom: -260, near: 50, far: 1400}); sun.shadow.bias = -0.0005;
  scene.add(sun);
  camera = new THREE.PerspectiveCamera(34, 1, 5, 6000);
  controls = new OrbitControls(camera, canvas);
  Object.assign(controls, {enableDamping: true, dampingFactor: 0.14, minDistance: 140, maxDistance: 1600, maxPolarAngle: Math.PI / 2 - 0.04, zoomSpeed: 0.9});
  controls.addEventListener("change", () => { need = true; });

  // the mat, its grid, how far the arm may build, the spot, where the robot is
  const mat = new THREE.Mesh(new THREE.BoxGeometry(2 * MAT, 4, 2 * MAT), new THREE.MeshStandardMaterial({color: css("--mat"), roughness: 0.95}));
  mat.position.y = -2; mat.receiveShadow = true; scene.add(mat); scene.userData.mat = mat;
  const fine = new THREE.GridHelper(2 * MAT, 2 * MAT / 10, 0xffffff, 0xffffff), coarse = new THREE.GridHelper(2 * MAT, 2 * MAT / 50, 0xffffff, 0xffffff);
  fine.material.transparent = coarse.material.transparent = true; fine.material.opacity = 0.07; coarse.material.opacity = 0.16; fine.position.y = coarse.position.y = 0.15;
  scene.add(fine, coarse);
  const R = cat.reach_mm, reach = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints([W(-R, -R, 0.3), W(R, -R, 0.3), W(R, R, 0.3), W(-R, R, 0.3)]), new THREE.LineDashedMaterial({color: css("--accent"), dashSize: 6, gapSize: 5, transparent: true, opacity: 0.55}));
  reach.computeLineDistances(); scene.add(reach);
  const ring = new THREE.Mesh(new THREE.RingGeometry(9, 11, 40).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({color: css("--accent")})); ring.position.y = 0.4; scene.add(ring);
  const robot = new THREE.Mesh(new THREE.CircleGeometry(55, 48, Math.PI / 2, Math.PI).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({color: css("--faint"), transparent: true, opacity: 0.35}));
  robot.position.copy(W(-MAT - 18, 0, 0.2)); scene.add(robot);
  zoneGroup = new THREE.Group(); scene.add(zoneGroup);
  scene.userData.places = [{text: "spot", p: W(0, -16, 0)}, {text: "robot this side", p: W(-MAT + 12, 0, 0)}, {text: "left", p: W(0, MAT - 14, 0)}, {text: "right", p: W(0, -MAT + 14, 0)}];

  new ResizeObserver(size).observe(host); size(); view("iso");
  requestAnimationFrame(frame);
}
let vw = 0, vh = 0;
function size() { const r = host.getBoundingClientRect(); if (r.width === vw && r.height === vh) return; vw = r.width; vh = r.height; renderer.setSize(Math.max(1, vw), Math.max(1, vh), false); camera.aspect = Math.max(1, vw) / Math.max(1, vh); camera.updateProjectionMatrix(); need = true; }
function view(name) {
  const t = W(0, 0, 25), at = {iso: W(-330, -250, 300), robot: W(-560, 0, 230), top: W(-0.5, 0, 720), side: W(0, 560, 180)}[name] || W(-330, -250, 300);
  controls.target.copy(t); camera.position.copy(at); controls.update(); need = true;
  document.querySelectorAll("#views button").forEach(b => b.classList.toggle("on", b.dataset.view === name));
}
document.querySelectorAll("#views button").forEach(b => { b.onclick = () => view(b.dataset.view); });

let anim = null;
function frame(now) {
  requestAnimationFrame(frame);
  if (controls.update()) need = true;
  if (anim) {
    const t = Math.min(1, (now - anim.t0) / 380), m = meshes.get(anim.id);
    if (m) m.position.y = anim.y + 70 * (1 - t) * (1 - t);
    if (t >= 1) anim = null; need = true;
  }
  if (!need || !vw) return;
  need = false;
  renderer.render(scene, camera);
  const v = new THREE.Vector3();
  for (const l of labelEls) { v.copy(l.p).project(camera); const on = v.z < 1 && Math.abs(v.x) < 1.05 && Math.abs(v.y) < 1.05; l.e.style.display = on ? "" : "none"; if (on) { l.e.style.left = `${(v.x + 1) / 2 * vw}px`; l.e.style.top = `${(1 - v.y) / 2 * vh}px`; } }
}

function draw() {
  if (!scene) return;
  for (const m of meshes.values()) { scene.remove(m); m.children.forEach(c => c.geometry.dispose()); }
  meshes.clear(); zoneGroup.clear(); $("labels").replaceChildren(); labelEls = [];
  const N = design.blocks.length, shown = upto == null ? N : upto, edgeCol = {bad: css("--no"), warn: css("--cal"), ok: "#000000", "": "#000000"};
  design.blocks.forEach((b, k) => {
    const later = k >= shown, m = blockMesh(b, later ? 0.1 : 1), st = statusOf(k), on = b.id === sel;
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry, 25), new THREE.LineBasicMaterial({color: on ? css("--accent") : edgeCol[st], transparent: true, opacity: on ? 1 : (st === "bad" || st === "warn" ? 0.95 : 0.22)}));
    m.add(edges);
    if (on) { m.material = m.material.clone(); m.material.emissive = new THREE.Color(css("--accent")); m.material.emissiveIntensity = 0.16; }
    if (later) m.raycast = () => {};
    scene.add(m); meshes.set(b.id, m);
    if (!later) label(String(k + 1), W(b.x, b.y, top(b) + 9), on ? "sel" : st === "ok" ? "" : st);
  });
  const r = sel && rep(sel);
  if (r && (upto == null || idx(sel) < upto)) for (const z of r.zones) {  // where the gripper is when it lets go of the selected block
    const shape = new THREE.Shape(z.poly.map(([x, y]) => new THREE.Vector2(x, y)));
    const h = z.what === "body" ? 40 : z.z1 - z.z0, g = new THREE.ExtrudeGeometry(shape, {depth: h, bevelEnabled: false}).rotateX(-Math.PI / 2).translate(0, z.z0, 0);
    const hit = z.hit.length > 0, col = hit ? css("--no") : (z.what === "body" ? css("--faint") : css("--dis"));
    const mesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({color: col, transparent: true, opacity: hit ? 0.35 : (z.what === "body" ? 0.08 : 0.2), depthWrite: false}));
    const wire = new THREE.LineSegments(new THREE.EdgesGeometry(g), new THREE.LineBasicMaterial({color: col, transparent: true, opacity: hit ? 0.9 : (z.what === "body" ? 0.18 : 0.45)}));
    zoneGroup.add(mesh, wire);
  }
  for (const p of scene.userData.places) label(p.text, p.p, "place");
  if (ghost) scene.add(ghost.mesh);
  need = true;
}
function label(text, p, cls) { const e = el("span", cls || "", text); $("labels").append(e); labelEls.push({e, p}); }

// ---------------------------------------------------------------------------------------------------- dragging blocks
function pointerRay(e) { const r = canvas.getBoundingClientRect(); ndc.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1); ray.setFromCamera(ndc, camera); }
function overCanvas(e) { const r = canvas.getBoundingClientRect(); return e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom; }
function placeAt(e, b, without) {
  pointerRay(e);
  const hits = ray.intersectObjects([...meshes.values()].filter(m => m.userData.id !== without), false);
  let p = null, under = null;
  if (hits.length) { p = hits[0].point; under = byId(hits[0].object.userData.id); }
  else { p = ray.ray.intersectPlane(ground, new THREE.Vector3()); }
  if (!p) return null;
  let x = p.x, y = -p.z;
  if (!e.altKey) {
    x = Math.round(x / GRID) * GRID; y = Math.round(y / GRID) * GRID;
    if (under && Math.abs(p.x - under.x) < CENTRE_SNAP && Math.abs(-p.z - under.y) < CENTRE_SNAP) { x = under.x; y = under.y; }  // right over the middle of the block under it
  }
  x = Math.max(-MAT + 10, Math.min(MAT - 10, x)); y = Math.max(-MAT + 10, Math.min(MAT - 10, y));
  const q = {...b, x, y}; q.z = dropZ(q, without);
  return q;
}
function showGhost(q) {
  if (ghost) scene.remove(ghost.mesh);
  if (!q) { ghost = null; need = true; return; }
  const m = blockMesh(q, 0.6); m.raycast = () => {};
  m.add(new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry, 25), new THREE.LineBasicMaterial({color: css("--accent")})));
  ghost = {mesh: m, q}; scene.add(m); need = true;
}
let carry = null, press = null;
function beginCarry(e, kind) {
  const r = kind.rests.find(x => !x.rolls) || kind.rests[0];
  carry = {b: {id: "", kind: kind.name, rest: r.name, angle: 0, x: 0, y: 0, z: 0}, img: kind.img};
  $("carry").replaceChildren(Object.assign(el("img"), {src: kind.img})); $("carry").hidden = false; document.body.classList.add("carrying");
  moveCarry(e); e.preventDefault();
}
function moveCarry(e) {
  const c = $("carry"); c.style.left = e.clientX + "px"; c.style.top = e.clientY + "px";
  if (overCanvas(e)) { c.style.opacity = 0; const q = placeAt(e, carry.b, null); showGhost(q); dropNote(q); }
  else { c.style.opacity = 0.9; showGhost(null); dropNote(null); }
}
function endCarry(e) {
  const q = ghost && ghost.q, inside = overCanvas(e);
  showGhost(null); dropNote(null); $("carry").hidden = true; document.body.classList.remove("carrying");
  const b = carry.b; carry = null;
  if (!inside || !q) return;
  const id = nextId();
  change(() => { design.blocks.push({...q, id}); });
  sel = id; draw(); renderUI();
}
function dropNote(q) {
  const n = $("dropnote");
  if (!q) { n.hidden = true; return; }
  const under = design.blocks.filter(o => o.id !== q.id && overlap(q, o) > 1 && Math.abs(top(o) - q.z) < 0.5);
  n.textContent = q.z < 0.5 ? `on the mat · ${q.x.toFixed(0)}, ${q.y.toFixed(0)} mm` : `on ${under.map(o => "step " + (idx(o.id) + 1)).join(" and ")} · ${q.z.toFixed(1)} mm up`;
  n.hidden = false;
}
addEventListener("pointermove", e => {
  if (carry) return moveCarry(e);
  if (press && press.id) {
    if (!press.moving && Math.hypot(e.clientX - press.x0, e.clientY - press.y0) < 5) return;
    press.moving = true;
    const b = byId(press.id), m = meshes.get(press.id);
    if (m) m.visible = false;
    const q = placeAt(e, b, press.id); showGhost(q); dropNote(q); document.body.classList.add("carrying");
  }
});
addEventListener("pointerup", e => {
  if (carry) return endCarry(e);
  if (!press) return;
  const p = press; press = null; controls.enabled = true; document.body.classList.remove("carrying");
  if (p.id && p.moving) {
    const q = ghost && ghost.q; showGhost(null); dropNote(null);
    if (q && overCanvas(e)) change(() => { const b = byId(p.id); b.x = q.x; b.y = q.y; b.z = q.z; });
    else draw();
  } else if (!p.id && Math.hypot(e.clientX - p.x0, e.clientY - p.y0) < 5) { sel = null; draw(); renderUI(); }
});
canvas.addEventListener("pointerdown", e => {
  if (e.button !== 0 || carry) return;
  pointerRay(e);
  const hit = ray.intersectObjects([...meshes.values()], false)[0];
  press = {id: hit ? hit.object.userData.id : null, x0: e.clientX, y0: e.clientY, moving: false};
  if (hit) { controls.enabled = false; sel = press.id; upto = null; draw(); renderUI(); }
});

// ---------------------------------------------------------------------------------------------------- editing the selected block
function turnSel(deg) { if (carry) { carry.b.angle = (carry.b.angle + deg + 360) % 360; return; } const b = byId(sel); if (b) change(() => { b.angle = ((b.angle + deg) % 360 + 360) % 360; }); }
function moveSel(dx, dy) { const b = byId(sel); if (b) change(() => { b.x = Math.max(-MAT + 10, Math.min(MAT - 10, b.x + dx)); b.y = Math.max(-MAT + 10, Math.min(MAT - 10, b.y + dy)); b.z = dropZ(b, b.id); }); }
function recolour(colour) {
  const b = byId(sel); if (!b) return;
  const k = kindOf(b), to = cat.kinds.find(o => o.colour === colour && o.shape === k.shape && JSON.stringify(o.size) === JSON.stringify(k.size));
  if (to && to.name !== b.kind) change(() => { b.kind = to.name; if (!to.rests.some(r => r.name === b.rest)) b.rest = to.rests[0].name; });
}
function relay(rest) { const b = byId(sel); if (b && b.rest !== rest) change(() => { b.rest = rest; b.z = dropZ(b, b.id); }); }
function nextColour() { const b = byId(sel); if (!b) return; const k = kindOf(b), cols = ORDER.filter(c => cat.kinds.some(o => o.colour === c && o.shape === k.shape && JSON.stringify(o.size) === JSON.stringify(k.size))); recolour(cols[(cols.indexOf(k.colour) + 1) % cols.length]); }
function nextRest() { const b = byId(sel); if (!b) return; const rs = kindOf(b).rests.filter(r => !r.rolls); relay(rs[(rs.findIndex(r => r.name === b.rest) + 1) % rs.length].name); }
function removeSel() { const b = byId(sel); if (!b) return; change(() => { design.blocks = design.blocks.filter(o => o.id !== b.id); }); sel = null; renderUI(); draw(); }
function copySel() {
  const b = byId(sel); if (!b) return;
  const r = restOf(b), id = nextId(), q = {...b, id, y: b.y - (r.short + 25)}; q.z = dropZ(q, null);
  change(() => { design.blocks.push(q); }); sel = id; draw(); renderUI();
}
function shiftOrder(id, by) {
  const k = idx(id), j = k + by;
  if (k < 0 || j < 0 || j >= design.blocks.length) return;
  hist.push(snapshot()); fut.length = 0;
  const B = design.blocks; [B[k], B[j]] = [B[j], B[k]]; dirty = true; afterChange();
}
document.querySelectorAll("[data-turn]").forEach(b => { b.onclick = () => turnSel(+b.dataset.turn); });
document.querySelectorAll("[data-move]").forEach(b => { b.onclick = () => { const [dx, dy] = b.dataset.move.split(",").map(Number); moveSel(dx, dy); }; });
$("delbtn").onclick = removeSel; $("dupbtn").onclick = copySel;
addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" && e.target.type === "text") { if (e.key === "Enter" || e.key === "Escape") e.target.blur(); return; }
  const k = e.key, mod = e.metaKey || e.ctrlKey;
  if (mod && k.toLowerCase() === "z") { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
  if (mod && k.toLowerCase() === "y") { e.preventDefault(); redo(); return; }
  if (mod && k.toLowerCase() === "s") { e.preventDefault(); save(); return; }
  if (mod) return;
  if (k === "Escape") { closePops(); if (carry) { carry = null; showGhost(null); $("carry").hidden = true; document.body.classList.remove("carrying"); } else { sel = null; draw(); renderUI(); } return; }
  if (k === "r" || k === "R") { e.preventDefault(); turnSel(e.shiftKey ? -90 : 90); if (carry) { const q = ghost && ghost.q; if (q) showGhost({...q, angle: carry.b.angle, z: dropZ({...q, angle: carry.b.angle}, null)}); } return; }
  if (!sel) return;
  const step = e.shiftKey ? 1 : 5;
  const acts = {q: () => turnSel(15), e: () => turnSel(-15), l: nextRest, c: nextColour, d: copySel, Delete: removeSel, Backspace: removeSel,
                ArrowUp: () => moveSel(step, 0), ArrowDown: () => moveSel(-step, 0), ArrowLeft: () => moveSel(0, step), ArrowRight: () => moveSel(0, -step)};
  const f = acts[k] || acts[k.toLowerCase()];
  if (f) { e.preventDefault(); f(); }
});

// ---------------------------------------------------------------------------------------------------- the panels
function renderVerdict() {
  const v = $("verdict"), N = design.blocks.length;
  let cls = "", t = "Empty", s = "drag blocks in from the right";
  if (N && busy && !report) { cls = "busy"; t = "Checking…"; s = ""; }
  else if (N && report) {
    const bad = report.blocks.findIndex(b => b.problems.length);
    if (bad >= 0) { cls = "bad"; t = "Cannot be built"; s = `step ${bad + 1}: ${report.blocks[bad].problems[0]}`; }
    else if (report.tilt_deg != null && report.tilt_deg < cat.shaky_deg) { cls = "warn"; t = "Shaky"; s = `holds a tilt of ${report.tilt_deg.toFixed(1)}° at step ${report.weakest + 1}`; }
    else if (report.ok) { cls = "ok"; t = "Buildable"; s = `every step stands · holds ${report.tilt_deg >= cat.max_tilt_deg ? "more than " + cat.max_tilt_deg : report.tilt_deg.toFixed(0)}° of tilt`; }
    if (busy) cls += " busy";
  }
  v.className = "verdict " + cls; $("vtitle").textContent = t; $("vsub").textContent = s; v.title = s;
}
function renderUI() {
  const N = design.blocks.length, shown = upto == null ? N : upto;
  renderVerdict();
  $("dirty").hidden = !dirty; $("undobtn").disabled = !hist.length; $("redobtn").disabled = !fut.length;
  if (document.activeElement !== $("dname")) $("dname").value = design.name || "";
  // steps
  $("stepcount").textContent = N ? `${N} block${N === 1 ? "" : "s"}` : ""; show("steps_none", !N); $("orderbtn").disabled = N < 2;
  $("dsteps").replaceChildren(...design.blocks.map((b, k) => {
    const li = el("li", (b.id === sel ? "on " : "") + (k >= shown ? "later" : "")), k_ = kindOf(b), st = statusOf(k), s = stepRep(k);
    const dot = el("span", "dot"); dot.style.background = PAINT[k_.colour] || "#999";
    const nm = el("span", "nm", word(k_)); nm.append(el("small", "", b.rest));
    const mark = el("span", "st " + st, {ok: "✓", warn: "!", bad: "✗", "": "·"}[st]);
    mark.title = st === "bad" ? rep(b.id).problems.join("; ") : (s && s.stands ? `after this step it holds a tilt of ${s.tilt_deg}°` : "");
    const mv = el("span", "mv"), up = el("button", "", "▲"), dn = el("button", "", "▼");
    up.title = "earlier"; dn.title = "later"; up.onclick = ev => { ev.stopPropagation(); shiftOrder(b.id, -1); }; dn.onclick = ev => { ev.stopPropagation(); shiftOrder(b.id, 1); };
    up.disabled = k === 0; dn.disabled = k === N - 1; mv.append(up, dn);
    li.append(el("span", "k", String(k + 1)), dot, nm, mark, mv);
    li.onclick = () => { sel = b.id; draw(); renderUI(); };
    return li;
  }));
  const onRow = $("dsteps").querySelector("li.on"); if (onRow) onRow.scrollIntoView({block: "nearest"});
  // the chart of every step
  const max = cat.max_tilt_deg, chart = $("chart");
  chart.replaceChildren(...design.blocks.map((b, k) => {
    const s = stepRep(k), bar = el("div", "bar");
    if (!s || s.stands == null) bar.classList.add("none"), bar.style.height = "8%";
    else if (!s.stands) bar.classList.add("bad");
    else { bar.style.height = `${Math.max(6, 100 * Math.min(s.tilt_deg, max) / max)}%`; if (s.tilt_deg < cat.shaky_deg) bar.classList.add("warn"); }
    if (k >= shown) bar.classList.add("later");
    bar.title = !s ? "" : (s.stands == null ? "not checked yet" : (s.stands ? `step ${k + 1}: holds a tilt of ${s.tilt_deg}°` : `step ${k + 1}: falls over`));
    bar.onclick = () => { upto = k + 1; draw(); renderUI(); };
    return bar;
  }));
  const line = el("div", "line"); line.style.bottom = `${100 * cat.shaky_deg / max}%`; chart.append(line);
  let note = "";
  if (!N) note = "Nothing to check yet.";
  else if (!report) note = busy ? "Checking…" : "";
  else if (report.stands === false) note = `It falls over at step ${report.steps.findIndex(s => s.stands === false) + 1}.`;
  else if (report.stands == null) note = "";
  else if (report.stands) note = `Every step stands. Weakest: step ${report.weakest + 1}, which holds ${report.steps[report.weakest].tilt_deg}° of tilt.`;
  if (report && report.problems.length) note += " " + report.problems.join(" ");
  $("checknote").textContent = note;
  const needs = report ? report.needs : [];
  $("needs").replaceChildren(...(needs.length ? [el("b", "", "On the mat ")] : []), ...needs.map(n => { const c = el("span", "need"), i = el("i"); i.style.background = PAINT[(KIND.get(n.kind) || {}).colour] || "#999"; c.append(i, `${n.n} × ${n.kind}`, el("small", "", " " + n.rest)); return c; }));
  $("needs").title = needs.map(n => `${n.n} × ${n.kind} (${n.rest})`).join(", ");
  // the selected block
  const b = byId(sel); show("sel_none", !b); show("sel_on", !!b);
  if (b) {
    const k = idx(b.id), kind = kindOf(b), r = rep(b.id);
    $("seltitle").textContent = `Step ${k + 1} · ${kind.name}`;
    const same = cat.kinds.filter(o => o.shape === kind.shape && JSON.stringify(o.size) === JSON.stringify(kind.size));
    $("sel_col").replaceChildren(...ORDER.filter(c => same.some(o => o.colour === c)).map(c => { const s = el("button", c === kind.colour ? "on" : ""); s.style.background = PAINT[c]; s.title = c; s.onclick = () => recolour(c); return s; }));
    $("sel_rest").replaceChildren(...kind.rests.map(rr => { const s = el("button", rr.name === b.rest ? "on" : "", rr.name); s.disabled = rr.rolls; s.title = rr.rolls ? "it would roll: the arm cannot build with it" : `${rr.long.toFixed(0)} × ${rr.short.toFixed(0)} mm footprint, ${rr.height.toFixed(0)} mm high`; s.onclick = () => relay(rr.name); return s; }));
    $("sel_ang").textContent = `${b.angle.toFixed(0)}°`;
    const under = design.blocks.filter(o => o !== b && overlap(b, o) > 1 && Math.abs(top(o) - b.z) < 0.5);
    $("sel_pos").textContent = `${b.x.toFixed(0)}, ${b.y.toFixed(0)} mm · ` + (b.z < 0.5 ? "on the mat" : `on ${under.map(o => "step " + (idx(o.id) + 1)).join(" + ")}, ${b.z.toFixed(1)} mm up`);
    const notes = [];
    if (r) {
      r.problems.forEach(t => notes.push(["bad", "✗ " + t])); r.warnings.forEach(t => notes.push(["warn", "! " + t]));
      if (!r.problems.length) notes.push(["info", `The gripper takes it ${Math.abs(((r.grip_deg - b.angle) % 180 + 180) % 180) > 1 ? "turned " + r.grip_deg.toFixed(0) + "°, " : ""}fingers across its short side, and lets go opening to ${r.release_mm.toFixed(0)} mm.`]);
    }
    $("sel_notes").replaceChildren(...notes.map(([c, t]) => el("div", c, t)));
  } else $("seltitle").textContent = "Block";
  // the view
  const H = report ? report.height_mm : Math.max(0, ...design.blocks.map(top));
  $("t_h").textContent = N ? `${H.toFixed(0)} mm high` : "empty mat"; $("t_n").textContent = N ? `${N} blocks` : ""; $("t_n").hidden = !N;
  $("t_ms").textContent = report ? `checked in ${report.ms} ms` : ""; $("t_ms").hidden = !report;
  const up = $("upto"); up.max = N; up.value = shown; up.disabled = N < 2;
  $("uptotext").textContent = !N ? "no steps" : (shown === N ? `all ${N} steps` : `step ${shown} of ${N}`);
  $("playbtn").disabled = N < 1;
}
const show = (id, on) => { $(id).hidden = !on; };
$("upto").oninput = () => { const v = +$("upto").value; upto = v >= design.blocks.length ? null : v; draw(); renderUI(); };
$("playbtn").onclick = async () => {
  const N = design.blocks.length;
  for (let k = 1; k <= N; k++) {
    upto = k === N ? null : k; draw(); renderUI();
    const b = design.blocks[k - 1], m = meshes.get(b.id); if (m) anim = {id: b.id, y: m.position.y, t0: performance.now()};
    await new Promise(r => setTimeout(r, 520));
  }
};

// ---------------------------------------------------------------------------------------------------- files
$("dname").oninput = () => { design.name = $("dname").value; dirty = true; renderUI(); try { localStorage.setItem(KEY, JSON.stringify({design, savedAs, dirty})); } catch (e) { /* nothing kept */ } };
async function save() {
  const name = ($("dname").value || "").trim();
  if (!name) { say("Give the design a name first (top left)."); $("dname").focus(); return; }
  design.name = name;
  try { const r = await api("/api/design/save", {design}); savedAs = r.name; dirty = false; renderUI(); say(`Saved as "${r.name}". On the dashboard: Camera → A design.`, true); afterSaveStore(); }
  catch (e) { say(e.message); }
}
function afterSaveStore() { try { localStorage.setItem(KEY, JSON.stringify({design, savedAs, dirty})); } catch (e) { /* nothing kept */ } }
$("savebtn").onclick = save;
$("newbtn").onclick = () => {
  if (design.blocks.length && dirty && !confirm("Start a new design? The one shown has changes that are not saved (Undo brings it back).")) return;
  change(() => { design = {name: "", note: "", blocks: []}; }); sel = null; savedAs = null; dirty = false; renderUI();
};
$("undobtn").onclick = undo; $("redobtn").onclick = redo;
$("orderbtn").onclick = async () => {
  try {
    const r = await api("/api/design/order", {design}), ids = r.design.blocks.map(b => b.id);
    if (ids.every((id, i) => id === design.blocks[i].id)) { say("The order already works: every step stands and the gripper is clear.", true); return; }
    change(() => { design.blocks = ids.map(byId); }, "New order: every step stands and the gripper is clear.");
  } catch (e) { say(e.message); }
};
$("openbtn").onclick = async ev => {
  ev.stopPropagation();
  const menu = $("openlist");
  if (!menu.hidden) { menu.hidden = true; return; }
  try {
    const {designs} = await api("/api/designs");
    const mine = designs.filter(d => !d.example), ex = designs.filter(d => d.example);
    const item = d => { const b = el("button"); b.append(el("span", "", d.name), el("small", "", `${d.n} blocks` + (d.example ? "" : ` · ${d.saved}`))); b.onclick = () => openDesign(d.name); return b; };
    menu.replaceChildren(...(mine.length ? [el("div", "grp", "Saved"), ...mine.map(item)] : [el("div", "grp", "Nothing saved yet")]), ...(ex.length ? [el("div", "grp", "Examples"), ...ex.map(item)] : []));
    menu.hidden = false;
  } catch (e) { say(e.message); }
};
async function openDesign(name) {
  $("openlist").hidden = true;
  if (design.blocks.length && dirty && !confirm(`Open "${name}"? The design shown has changes that are not saved (Undo brings it back).`)) return;
  try {
    const {design: d} = await api("/api/design?name=" + encodeURIComponent(name));
    const unknown = d.blocks.filter(b => !KIND.has(b.kind)).map(b => b.kind);
    if (unknown.length) { say(`It uses blocks this catalogue does not have: ${[...new Set(unknown)].join(", ")}.`); return; }
    change(() => { design = d; }); sel = null; savedAs = d.name; dirty = false; renderUI(); view("iso");
  } catch (e) { say(e.message); }
}
function closePops() { document.querySelectorAll(".pop.on").forEach(p => p.classList.remove("on")); $("openlist").hidden = true; }
addEventListener("click", e => {
  const b = e.target.closest && e.target.closest("[data-pop]"), inside = e.target.closest && e.target.closest(".pop,.menu");
  if (b) { const p = $(b.dataset.pop), was = p.classList.contains("on"); closePops(); p.classList.toggle("on", !was); }
  else if (!inside) closePops();
});
addEventListener("beforeunload", e => { if (dirty && design.blocks.length && !savedAs) { e.preventDefault(); e.returnValue = ""; } });

// ---------------------------------------------------------------------------------------------------- the palette
function thumbnails() {  // every kind drawn once, as a picture for the palette
  const r = new THREE.WebGLRenderer({antialias: true, alpha: true, preserveDrawingBuffer: true});
  r.setSize(160, 120); r.outputColorSpace = THREE.SRGBColorSpace; r.toneMapping = THREE.ACESFilmicToneMapping;
  const s = new THREE.Scene(); s.environment = new THREE.PMREMGenerator(r).fromScene(new RoomEnvironment(), 0.04).texture;
  s.add(new THREE.HemisphereLight(0xffffff, 0x333333, 0.5)); const d = new THREE.DirectionalLight(0xffffff, 1.5); d.position.set(-1, 2, 1.4); s.add(d);
  const c = new THREE.PerspectiveCamera(28, 4 / 3, 1, 2000);
  for (const k of cat.kinds) {
    const rest = k.rests.find(x => !x.rolls) || k.rests[0], m = new THREE.Mesh(geometry(rest), material(k.colour));
    m.rotation.y = -0.5; s.add(m);
    const R = Math.hypot(rest.long, rest.short, rest.height) / 2;
    c.position.set(-1, 0.8, 1.15).normalize().multiplyScalar(R / Math.sin(14 * Math.PI / 180) * 1.05); c.lookAt(0, 0, 0);
    r.render(s, c); k.img = r.domElement.toDataURL("image/png"); s.remove(m);
  }
  r.dispose();
}
function palette() {
  const groups = new Map();
  for (const k of cat.kinds) { const key = k.shape + JSON.stringify(k.size); if (!groups.has(key)) groups.set(key, []); groups.get(key).push(k); }
  GROUPS = [...groups.values()];
  $("kinds").replaceChildren(...GROUPS.map(ks => {
    const k0 = ks[0], dims = k0.shape === "cylinder" ? `⌀${k0.size[0].toFixed(0)} × ${k0.size[1].toFixed(0)}` : k0.size.slice().sort((a, b) => b - a).map(v => v.toFixed(0)).join(" × ");
    const box = el("div", "kind"), h = el("h3", "", word(k0)); h.append(el("small", "", dims + " mm")); box.append(h);
    const row = el("div", "row4");
    ks.sort((a, b) => ORDER.indexOf(a.colour) - ORDER.indexOf(b.colour)).forEach(k => {
      const t = el("div", "thumb"); t.title = `${k.name}: drag it onto the mat`; t.append(Object.assign(el("img"), {src: k.img, alt: k.name, draggable: false}));
      t.addEventListener("pointerdown", e => { if (e.button === 0) beginCarry(e, k); });
      row.append(t);
    });
    box.append(row); return box;
  }));
  $("catsrc").textContent = cat.example ? "example sizes" : "from " + cat.source;
  $("catsrc").title = cat.example ? "blocks.json is not there: the sizes are those of blocks.example.json" : "the rig's own blocks.json";
}

// ---------------------------------------------------------------------------------------------------- start
(async () => {
  try { cat = await api("/api/design/catalogue"); }
  catch (e) { say("No catalogue of blocks: " + e.message); return; }
  cat.kinds.forEach(k => KIND.set(k.name, k)); $("shaky_deg").textContent = cat.shaky_deg;
  try { start3D(); thumbnails(); }
  catch (e) { $("v_fail").hidden = false; $("v_fail").textContent = "The build area cannot be drawn here (" + e.message + ")."; }
  palette();
  try {
    const kept = JSON.parse(localStorage.getItem(KEY) || "null");
    if (kept && kept.design && Array.isArray(kept.design.blocks) && kept.design.blocks.every(b => KIND.has(b.kind))) { design = kept.design; savedAs = kept.savedAs; dirty = !!kept.dirty; }
  } catch (e) { /* nothing kept */ }
  const want = new URLSearchParams(location.search).get("open");
  if (want) await openDesign(want);
  settle(); draw(); renderUI(); askCheck();
})();
