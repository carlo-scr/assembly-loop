// The model of the arm on the dashboard. A drawing only: it shows the joint angles the controller publishes and what
// the program knows around the arm (the spots, the box the guard keeps the arm in, where the current move leads). It
// sends nothing and decides nothing.
//
// Frames. The program speaks in the arm's base frame, in millimetres. The arm is drawn in that frame (in metres,
// inside `rig`), and `rig` is turned so that "up" of the room is up on the screen: on this rig the arms are mounted
// at 45 degrees, so the base's own z axis is not up. Meshes and joint offsets: UFACTORY's xArm 7 with its gripper,
// as in MuJoCo Menagerie (the same chain as `arms._CHAIN`).
import * as THREE from "three";
import { OrbitControls } from "/web/vendor/OrbitControls.js";
import { STLLoader } from "/web/vendor/STLLoader.js";
import { toCreasedNormals } from "/web/vendor/BufferGeometryUtils.js";
import { RoomEnvironment } from "/web/vendor/RoomEnvironment.js";

const CHAIN = [[0, 0, 0.267, 0], [0, 0, 0, -Math.PI / 2], [0, -0.293, 0, Math.PI / 2], [0.0525, 0, 0, Math.PI / 2], [0.0775, -0.3425, 0, Math.PI / 2], [0, 0, 0, Math.PI / 2], [0.076, 0.097, 0, -Math.PI / 2]];
const REST = [0, 0, 0, 60, 0, 60, 0];  // degrees: what is drawn while no session says otherwise
const PAINT = {red: 0xdc3c3c, orange: 0xf08a24, yellow: 0xf0d43a, green: 0x3fbf5a, blue: 0x3a78d8, purple: 0x9a48b8};
const THINGS = [0x4fae62, 0xd98a3c, 0x3f7fcf, 0xd9c23c];

export async function start(host) {
  const canvas = host.querySelector("canvas"), labelBox = host.querySelector("#v_labels");
  const renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: true, powerPreference: "high-performance"});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
  scene.add(new THREE.HemisphereLight(0xe6ecff, 0x1a1d24, 0.35));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  sun.position.set(1.0, 2.6, 1.4); sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048); sun.shadow.bias = -0.0004; sun.shadow.normalBias = 0.01;
  Object.assign(sun.shadow.camera, {left: -1.3, right: 1.3, top: 1.3, bottom: -1.3, near: 0.1, far: 7});
  scene.add(sun, sun.target);
  const cam = new THREE.PerspectiveCamera(32, 1, 0.05, 30);
  const controls = new OrbitControls(cam, canvas);
  Object.assign(controls, {enableDamping: true, dampingFactor: 0.14, minDistance: 0.35, maxDistance: 5, zoomSpeed: 0.8});
  let need = true, moved = false;
  controls.addEventListener("start", () => { moved = true; });
  const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // ---- the arm
  const rig = new THREE.Group(); rig.matrixAutoUpdate = false; scene.add(rig);
  const arm = new THREE.Group(); rig.add(arm);
  const paint = {white: new THREE.MeshStandardMaterial({color: 0xeceff3, roughness: 0.36, metalness: 0.04, envMapIntensity: 0.85}),
                 gray: new THREE.MeshStandardMaterial({color: 0xb6bac2, roughness: 0.45, metalness: 0.3, envMapIntensity: 0.9}),
                 black: new THREE.MeshStandardMaterial({color: 0x202329, roughness: 0.55, metalness: 0.1, envMapIntensity: 0.7})};
  const loader = new STLLoader(), waits = [];
  const part = (to, name, mat) => waits.push(loader.loadAsync(`/web/xarm7/${name}.stl`).then(g => {
    const m = new THREE.Mesh(toCreasedNormals(g, 0.65), mat); m.castShadow = m.receiveShadow = true; to.add(m); need = true; }));
  const mount = new THREE.Mesh(new THREE.CylinderGeometry(0.085, 0.085, 0.02, 48).rotateX(Math.PI / 2).translate(0, 0, -0.01), paint.black);
  mount.receiveShadow = true; arm.add(mount);
  part(arm, "link_base", paint.white);
  const J = []; let at = arm;
  CHAIN.forEach(([x, y, z, a], i) => {
    const fixed = new THREE.Group(); fixed.position.set(x, y, z); fixed.rotation.x = a; at.add(fixed);
    const joint = new THREE.Group(); fixed.add(joint); J.push(joint); at = joint;
    part(joint, i < 6 ? `link${i + 1}` : "end_tool", i < 6 ? paint.white : paint.gray);
  });
  const hand = new THREE.Group(); hand.rotation.z = Math.PI; J[6].add(hand); part(hand, "base_link", paint.white);
  const fingers = ["left", "right"].map(side => {
    const s = side === "left" ? 1 : -1, outer = new THREE.Group(), finger = new THREE.Group(), inner = new THREE.Group();
    outer.position.set(0, s * 0.035, 0.059098); finger.position.set(0, s * 0.035465, 0.042039); inner.position.set(0, s * 0.02, 0.074098);
    hand.add(outer, inner); outer.add(finger);
    part(outer, `${side}_outer_knuckle`, paint.black); part(finger, `${side}_finger`, paint.black); part(inner, `${side}_inner_knuckle`, paint.black);
    return {s, outer, finger, inner};
  });
  const tip = new THREE.Object3D(); tip.position.set(0, 0, 0.172); hand.add(tip);  // between the fingertips
  const cur = REST.slice(), goal = REST.slice(); let grip = 0, gripGoal = 0;
  const pose = () => {
    J.forEach((j, i) => { j.rotation.z = cur[i] * Math.PI / 180; });
    fingers.forEach(f => { f.outer.rotation.x = f.s * grip; f.finger.rotation.x = -f.s * grip; f.inner.rotation.x = f.s * grip; });
  };
  pose();

  // ---- the room: which way is up, a floor to stand the picture on
  const world = new THREE.Vector3(0, 0, 1);  // up of the room, in the base frame
  const floor = new THREE.GridHelper(4, 40, 0xffffff, 0xffffff); floor.material.transparent = true; floor.material.depthWrite = false; scene.add(floor);
  const catcher = new THREE.Mesh(new THREE.PlaneGeometry(6, 6).rotateX(-Math.PI / 2), new THREE.ShadowMaterial({opacity: 0.28})); catcher.receiveShadow = true; scene.add(catcher);
  let floorY = 0, tableKnown = false; const tableAt = new THREE.Vector3();
  function setUp(up) {
    const u = new THREE.Vector3(...up).normalize();
    if (u.distanceTo(world) < 1e-6 && rig.userData.set) return false;
    world.copy(u); rig.userData.set = true;
    const f = new THREE.Vector3(1, 0, 0).addScaledVector(u, -u.x).normalize(), l = new THREE.Vector3().crossVectors(u, f);  // forward and the robot's left, level
    rig.matrix.set(l.x, l.y, l.z, 0, u.x, u.y, u.z, 0, f.x, f.y, f.z, 0, 0, 0, 0, 1);  // the screen's right / up / towards the viewer
    rig.updateMatrixWorld(true);
    return true;
  }
  const toScreenWorld = mm => new THREE.Vector3(mm.x, mm.y, mm.z).multiplyScalar(0.001).applyMatrix4(rig.matrix);
  function theme() {
    floor.material.color.set(css("--grid")); floor.material.opacity = 0.6; need = true;
  }
  theme();
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { theme(); sigA = sigB = null; });

  // ---- what is drawn around the arm, in millimetres of the base frame
  const deco = new THREE.Group(); deco.scale.setScalar(0.001); rig.add(deco);
  const still = new THREE.Group(), live = new THREE.Group(); deco.add(still, live);
  let labels = [], sigA = null, sigB = null;
  const V = a => new THREE.Vector3(a[0], a[1], a[2]);
  const clear = g => { g.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material && o.material.dispose) o.material.dispose(); }); g.clear(); };
  const basis = (x, y, z, o) => new THREE.Matrix4().makeBasis(x, y, z).setPosition(o);
  const put = (mesh, m) => { mesh.matrixAutoUpdate = false; mesh.matrix.copy(m); return mesh; };
  function label(text, p, cls) { const e = document.createElement("span"); e.textContent = text; if (cls) e.className = cls; labelBox.append(e); labels.push({e, p}); }
  function block(size, colour, centre, x, y, z, opacity, round) {
    const g = round ? new THREE.CylinderGeometry(size[0] / 2, size[0] / 2, size[2], 32).rotateX(Math.PI / 2) : new THREE.BoxGeometry(size[0], size[1], size[2]);
    const m = put(new THREE.Mesh(g, new THREE.MeshStandardMaterial({color: colour, roughness: 0.6, transparent: opacity < 1, opacity, depthWrite: opacity >= 1})), basis(x, y, z, centre));
    m.castShadow = opacity >= 1; m.receiveShadow = true;
    return m;
  }

  function frame() {  // the table's own directions and height, from the spots
    const s = state.scene, spots = state.calibration.spots;
    const up = V(s.up), a = Math.abs(up.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
    const u = a.addScaledVector(up, -a.dot(up)).normalize(), v = new THREE.Vector3().crossVectors(up, u);
    const tipOf = p => V(p).addScaledVector(up, -s.tool_mm);
    const w = spots.length ? Math.min(...spots.map(x => tipOf(x.mm).dot(up))) - 2 : null;  // the fingertips stop a little above the table
    const onTable = p => { const q = tipOf(p); return w == null ? q : q.addScaledVector(up, w - q.dot(up)); };
    return {up, u, v, w, tipOf, onTable};
  }

  function drawStill() {  // the table, the spots, the box
    clear(still); labels.filter(l => !l.live).forEach(l => l.e.remove()); labels = labels.filter(l => l.live);
    const s = state.scene, spots = state.calibration.spots;
    tableKnown = false;
    if (!state.connected || !s.up) return;
    const F = frame(), accent = css("--accent");
    if (spots.length) {
      const pts = spots.map(x => F.onTable(x.mm)), us = pts.map(p => p.dot(F.u)), vs = pts.map(p => p.dot(F.v));
      const cu = (Math.min(...us) + Math.max(...us)) / 2, cv = (Math.min(...vs) + Math.max(...vs)) / 2;
      const su = Math.max(Math.max(...us) - Math.min(...us) + 200, 380), sv = Math.max(Math.max(...vs) - Math.min(...vs) + 200, 300);
      const centre = new THREE.Vector3().addScaledVector(F.u, cu).addScaledVector(F.v, cv).addScaledVector(F.up, F.w - 3);
      const mat = put(new THREE.Mesh(new THREE.BoxGeometry(su, sv, 6), new THREE.MeshStandardMaterial({color: css("--mat"), roughness: 0.95})), basis(F.u, F.v, F.up, centre));
      mat.receiveShadow = true; still.add(mat);
      tableKnown = true; tableAt.copy(toScreenWorld(centre)); floorY = tableAt.y + 0.0035;
      const tower = state.stack ? state.stack.on : null;
      spots.forEach((x, i) => {
        const p = pts[i].clone().addScaledVector(F.up, 1.5);
        still.add(put(new THREE.Mesh(new THREE.RingGeometry(17, 20, 48), new THREE.MeshBasicMaterial({color: x.name === tower ? accent : css("--ink"), transparent: true, opacity: x.name === tower ? 0.95 : 0.55, side: THREE.DoubleSide})), basis(F.u, F.v, F.up, p)));
        label(x.name, p.clone().addScaledVector(F.up, 16), x.name === tower ? "t" : "");
      });
    }
    if (s.box) {
      const B = s.box, ax = [0, 1, 2].map(k => new THREE.Vector3(B.axes[0][k], B.axes[1][k], B.axes[2][k]));
      const size = [0, 1, 2].map(k => B.hi[k] - B.lo[k]), mid = [0, 1, 2].map(k => (B.hi[k] + B.lo[k]) / 2);
      const centre = V(B.origin).addScaledVector(ax[0], mid[0]).addScaledVector(ax[1], mid[1]).addScaledVector(ax[2], mid[2]).addScaledVector(F.up, -s.tool_mm);
      const geo = new THREE.BoxGeometry(size[0], size[1], size[2]), m = basis(ax[0], ax[1], ax[2], centre);
      still.add(put(new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({color: accent, transparent: true, opacity: 0.7})), m));
      still.add(put(new THREE.Mesh(geo, new THREE.MeshBasicMaterial({color: accent, transparent: true, opacity: 0.045, depthWrite: false, side: THREE.DoubleSide})), m));
    }
  }

  function drawLive() {  // where the move leads, and the blocks
    clear(live); labels.filter(l => l.live).forEach(l => l.e.remove()); labels = labels.filter(l => !l.live);
    const s = state.scene;
    if (!state.connected || !s.up) return;
    const F = frame(), accent = css("--accent"), spots = state.calibration.spots, at = n => spots.find(x => x.name === n);
    if (s.path) {  // from the fingertips through every place the remaining steps go to
      const from = s.at == null ? 0 : s.at, pts = [deco.worldToLocal(tip.getWorldPosition(new THREE.Vector3()))];
      s.path.forEach((p, i) => { if (p && i >= from) pts.push(F.tipOf(p)); });
      if (pts.length > 1) {
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineDashedMaterial({color: accent, dashSize: 9, gapSize: 6}));
        line.computeLineDistances(); live.add(line);
        pts.slice(1).forEach((p, i) => { const dot = new THREE.Mesh(new THREE.SphereGeometry(i ? 3 : 5.5, 16, 12), new THREE.MeshBasicMaterial({color: accent, transparent: true, opacity: i ? 0.55 : 1})); dot.position.copy(p); live.add(dot); });
      }
    }
    const cube = (p, h, colour, opacity) => block([30, 30, h], colour, p.clone().addScaledVector(F.up, h / 2 - 2), F.u, F.v, F.up, opacity);
    if (state.fake) s.things.forEach((t, i) => live.add(cube(F.tipOf(t.xyz), 30, THINGS[i % THINGS.length], 1)));  // the stand-in's blocks: where they really are in its made-up world
    else {  // a real arm: only what the program BELIEVES, drawn faintly
      const st = state.stack, h = st && st.block_mm ? st.block_mm : 30;
      if (st && st.n != null && at(st.on)) {
        for (let k = 0; k < st.n; k++) live.add(cube(F.tipOf(at(st.on).mm).addScaledVector(F.up, k * h), h, accent, 0.4));
        st.feeds.slice(st.n).forEach(n => { if (at(n)) live.add(cube(F.tipOf(at(n).mm), h, accent, 0.4)); });
      } else if (state.thing_at && at(state.thing_at)) live.add(cube(F.tipOf(at(state.thing_at).mm), 30, accent, 0.4));
    }
    (s.plan || []).forEach(t => {  // a design being built: where each block still goes, the next one clearest
      if (t.done) return;
      const x = V(t.along).normalize(), y = new THREE.Vector3().crossVectors(F.up, x), p = F.onTable(t.xyz).addScaledVector(F.up, t.base + t.size[2] / 2 + 1);
      const m = block(t.round ? [t.size[0], t.size[0], t.size[2]] : t.size, PAINT[t.colour] || 0x9aa2af, p, x, y, F.up, t.next ? 0.45 : 0.16, t.round);
      m.castShadow = false; live.add(m);
      if (t.next) live.add(put(new THREE.LineSegments(new THREE.EdgesGeometry(m.geometry), new THREE.LineBasicMaterial({color: accent})), m.matrix));
    });
    s.seen.forEach(b => {  // what the camera made out at the last look
      const x = V(b.along).normalize(), y = new THREE.Vector3().crossVectors(F.up, x), p = F.onTable(b.xyz).addScaledVector(F.up, b.size[2] / 2 + 2);
      const solid = block(b.round ? [b.size[0], b.size[0], b.size[2]] : b.size, PAINT[b.colour] || 0x9aa2af, p, x, y, F.up, b.stale ? 0.3 : 0.9, b.round);
      if (!state.fake) { live.add(solid); return; }
      const edge = put(new THREE.LineSegments(new THREE.EdgesGeometry(solid.geometry, 30), new THREE.LineBasicMaterial({color: 0xffffff, transparent: true, opacity: b.stale ? 0.25 : 0.8})), solid.matrix);
      live.add(edge);  // the stand-in's blocks are already there, solid: what the camera made of them goes around them as an outline
    });
  }

  // ---- the camera of this picture
  let look = "iso";
  function aim(force) {
    if (moved && !force) return;
    const t = toScreenWorld(new THREE.Vector3(250, 0, 300));  // about the middle of the arm
    if (tableKnown) t.lerp(tableAt, 0.35);  // once the table is known: towards it, and a little closer
    const dir = {iso: [0.95, 0.62, 1.15], front: [0, 0.22, 1], side: [1, 0.22, 0.02], top: [0.001, 1, 0.02]}[look] || [1, 0.6, 1];
    controls.target.copy(t);
    cam.position.copy(t).addScaledVector(new THREE.Vector3(...dir).normalize(), tableKnown ? 2.0 : 2.15);
    controls.update(); need = true;
  }

  // ---- what the page tells this picture
  let state = {connected: false, scene: {}, calibration: {spots: []}}, sideNow = null;
  function setState(s, side) {
    state = s;
    const a = s.arms.find(x => x.key === side) || s.arms[0];
    const turned = setUp(a.up) || side !== sideNow; sideNow = side;
    if (!s.connected) { goal.splice(0, 7, ...REST); gripGoal = 0; }
    const A = JSON.stringify([s.connected, s.scene.up, s.scene.box, s.calibration.spots, s.scene.tool_mm, s.stack && s.stack.on]);
    const B = JSON.stringify([s.connected, s.scene.path, s.scene.at, s.scene.things, s.scene.seen, s.scene.plan, s.thing_at, s.stack, s.calibration.spots]);
    const hadTable = tableKnown;
    if (A !== sigA || turned) { sigA = A; drawStill(); need = true; }
    if (B !== sigB || turned || (s.scene.path && s.arm && s.arm.moving)) { sigB = B; drawLive(); need = true; }
    if (!tableKnown) floorY = toScreenWorld(new THREE.Vector3(0, 0, 0)).y - (Math.abs(world.z) > 0.98 ? 0.0 : 0.75);  // an upright arm stands on the floor of the picture; a shoulder-mounted one hangs above it
    floor.position.y = floorY; catcher.position.y = floorY + 0.001;
    if (turned || hadTable !== tableKnown) aim(false);
  }
  function setPose(v) {
    if (v && v.joints) goal.splice(0, 7, ...v.joints.slice(0, 7));
    if (v && v.grip != null) gripGoal = 0.85 * (1 - Math.min(Math.max(v.grip / 850, 0), 1));
  }
  function view(name) { look = name; moved = false; aim(true); }

  // ---- drawing, only when something changed
  let then = performance.now(), w = 0, h = 0;
  function size() {
    const r = host.getBoundingClientRect();
    if (r.width === w && r.height === h) return;
    w = r.width; h = r.height;
    renderer.setSize(Math.max(1, w), Math.max(1, h), false); cam.aspect = Math.max(1, w) / Math.max(1, h); cam.updateProjectionMatrix(); need = true;
  }
  new ResizeObserver(size).observe(host);
  const spot = new THREE.Vector3();
  function tickFrame(now) {
    requestAnimationFrame(tickFrame);
    const k = 1 - Math.exp(-(now - then) / 90); then = now;
    let moving = false;
    for (let i = 0; i < 7; i++) { const d = goal[i] - cur[i]; if (Math.abs(d) > 0.01) { cur[i] += d * k; moving = true; } else cur[i] = goal[i]; }
    if (Math.abs(gripGoal - grip) > 0.002) { grip += (gripGoal - grip) * k; moving = true; } else grip = gripGoal;
    if (moving) { pose(); need = true; }
    if (controls.update()) need = true;
    if (!need || !w) return;
    need = false;
    sun.target.position.copy(controls.target); sun.position.copy(controls.target).add(new THREE.Vector3(0.9, 2.4, 1.3));
    renderer.render(scene, cam);
    for (const l of labels) {  // names, written over the picture where their place shows
      spot.copy(l.p).multiplyScalar(0.001).applyMatrix4(rig.matrix).project(cam);
      const on = spot.z < 1 && Math.abs(spot.x) < 1.05 && Math.abs(spot.y) < 1.05;
      l.e.style.display = on ? "" : "none";
      if (on) { l.e.style.left = `${(spot.x + 1) / 2 * w}px`; l.e.style.top = `${(1 - spot.y) / 2 * h}px`; }
    }
  }
  size(); setUp([0, 0, 1]); aim(true);
  requestAnimationFrame(tickFrame);
  Promise.allSettled(waits).then(() => { need = true; });
  return {setState, setPose, view};
}
