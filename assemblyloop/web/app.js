// The dashboard's page: everything except the model of the arm (scene.js). The same buttons as the classic page, laid
// out on one screen. The program decides; this page only shows /api/state four times a second and posts button presses.
"use strict";
const TOKEN = window.TOKEN, $ = id => document.getElementById(id);
const COL = {"OFF": "--off", "STOPPED": "--stop", "DISARMED": "--dis", "AWAITING CALIBRATION": "--cal", "ARMED": "--accent", "RUNNING": "--ok", "PAUSED": "--cal"};
const sleep = ms => new Promise(r => setTimeout(r, ms));
let viewer = null, last = null, sig = {}, lastThing, lastN, tab = "blocks";

function say(text) { $("msg").textContent = text || ""; $("msg").hidden = !text; }
$("msg").onclick = () => say("");
async function post(path, body) {
  say("");
  try {
    const r = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json", "X-Token": TOKEN}, body: JSON.stringify(body || {})});
    const j = await r.json();
    if (j.error) say(j.error);
    tick();
    return j;
  } catch (e) { say("no answer from the program: " + e); }
}
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
function chip(mark, cls, text, detail, btn) { const c = el("span", "chip " + cls); c.append(el("b", "", mark), el("span", "", text)); if (detail) c.append(el("small", "", detail)); if (btn) c.append(btn); return c; }
function once(key, value, fn) { const v = JSON.stringify(value); if (sig[key] !== v) { sig[key] = v; fn(); } }
function options(sel, names, first) {
  const keep = sel.value;
  sel.replaceChildren(...first.concat(names.map(n => [n, n])).map(([v, t]) => { const o = el("option", "", t); o.value = v; return o; }));
  if ([...sel.options].some(o => o.value === keep)) sel.value = keep;
}
const f1 = v => v == null ? "…" : Number(v).toFixed(1), show = (id, on) => { $(id).hidden = !on; };
const seg = id => +($(id).querySelector("button.on") || {dataset: {v: 0}}).dataset.v;
document.querySelectorAll(".seg[id^=jog]").forEach(s => s.addEventListener("click", e => { const b = e.target.closest("button"); if (b) s.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b)); }));
const picked = () => (document.querySelector("input[name=arm]:checked") || {}).value;
const side = () => (last && last.connected) ? last.chosen : picked();

function render(s) {
  last = s;
  const ph = s.phase, c = s.calibration, teachOn = ph === "AWAITING CALIBRATION", busy = ["RUNNING", "PAUSED"].includes(ph), armed = ph === "ARMED" || busy, j = s.job, q = s.quest;
  const p = $("phase"); p.textContent = ph + (s.fake ? " · stand-in arm" : ""); p.style.color = `var(${COL[ph]})`;
  $("hint").textContent = s.hint; $("hint").title = s.hint;
  $("stop").disabled = !s.connected;

  // ---- hold-to-run (top bar)
  $("hold").classList.toggle("on", !!(q && q.held));
  if (q) {
    $("held").textContent = q.held ? "GRIP HELD" : (q.why === "released" ? "grip released" : (q.why === "held" ? "grip held, not armed" : q.why));
    $("holdsub").textContent = (s.hold === "page" ? "space bar" : "Quest grip" + (q.grip == null ? "" : " " + Number(q.grip).toFixed(2))) + (q.frames ? ` · ${q.hz}/s · longest gap ${q.gap_ms} ms` : " · no signal yet");
    $("hold").title = s.hold === "page" ? (s.fake ? "Hold the space bar to stand in for the grip." : "Hold the SPACE BAR to let the arm move; let go to pause it. It counts only while this page is in front and has the keyboard. With the space bar held: F closes the gripper, R opens it. The red emergency stop stays in reach.") : "Squeeze past halfway to hold. The red emergency stop stays in reach.";
  } else { $("held").textContent = "no session"; $("holdsub").textContent = "hold-to-run starts with the session"; $("hold").title = ""; }
  show("spacekey", s.hold === "page"); $("spacekey").classList.toggle("on", !!(q && q.held) || (down && s.hold === "page"));

  // ---- the sequence
  const ready = (s.checks || []).length > 0 && s.checks.every(k => k.ok);
  const rung = {"OFF": [1, 0, 0, 0, 0], "STOPPED": [1, 0, 0, 0, 0], "DISARMED": [2, ready ? 2 : 1, ready ? 1 : 0, 0, 0], "AWAITING CALIBRATION": [2, 2, 2, 1, 0]}[ph] || [2, 2, 2, 2, 1];
  rung.forEach((v, i) => { const li = $("r" + (i + 1)); li.className = ["todo", "cur", "done"][v]; li.querySelector(".n").textContent = v === 2 ? "✓" : i + 1; });
  const sums = [(s.arms.find(x => x.key === s.chosen) || {}).label || "", ready ? "all three hold" : "", "switched on", c.done ? `${c.spots.map(x => x.name).join(", ")} · lift ${c.lift_mm} mm` : "", ""];
  sums.forEach((t, i) => { $("r" + (i + 1) + "sum").textContent = t ? "· " + t : ""; });
  show("s_off", !s.connected); show("s_on", s.connected); show("startbtn", !s.connected); show("endbtn", s.connected);
  once("arms", s.arms.map(a => [a.key, a.label, a.ip, a.answers]), () => {
    const keep = picked();
    $("arms").replaceChildren(...s.arms.map(a => {
      const l = el("label"), i = el("input"), t = el("span", "t"), dot = el("i", a.ip ? (a.answers == null ? "" : (a.answers ? "y" : "n")) : "y");
      i.type = "radio"; i.name = "arm"; i.value = a.key; i.checked = keep ? keep === a.key : a.key === "stand-in"; i.onchange = pickArm;
      t.append(dot, a.label.replace(" (no robot)", ""));
      l.append(i, t, el("span", "d", a.ip ? (a.answers == null ? "checking…" : (a.answers ? "answers at " + a.ip : "no answer at " + a.ip)) : "no robot needed"));
      l.title = a.ip ? (a.answers ? "powered: answers at " + a.ip : "off or unplugged: no answer at " + a.ip) : "for trying things without a robot";
      return l;
    }));
    pickArm();
  });
  if (s.connected) { const a = s.arms.find(x => x.key === s.chosen) || {}; $("s_on").textContent = a.label + (a.ip ? " at " + a.ip : "") + (ph === "DISARMED" ? " · connected, read only" : " · connected, switched on"); }
  $("ended").textContent = ph === "STOPPED" ? "The last session was stopped: " + s.ended + ". Look at the arm before starting again." : "";
  show("k_none", !s.connected);
  $("checks").replaceChildren(...(s.checks || []).map(k => chip(k.ok ? "✓" : "✗", k.ok ? "y" : "n", k.text, k.ok ? "" : k.detail)));
  show("armbtn", !(teachOn || armed)); show("disbtn", teachOn || armed);
  $("armbtn").disabled = !(ph === "DISARMED" && ready); $("disbtn").disabled = !armed;
  $("c_status").textContent = c.done ? `Known: ${c.spots.map(x => x.name).join(", ")} · lift ${c.lift_mm} mm` : (teachOn ? (c.spots.length ? `${c.spots.length} saved so far: ${c.spots.map(x => x.name).join(", ")}` : "Nudge the gripper to each place below and save it.") : "Not known yet.");
  show("calbtn", c.done && !teachOn); $("calbtn").disabled = ph !== "ARMED";
  $("runline").textContent = !armed ? "Starts once the places are known." : (busy && j ? `Move ${Math.min(j.done + 1, j.total)} of ${j.total}: ${j.text || (j.src + " to " + j.dst)}` + (j.rehearse ? " (rehearsal)" : "") + (j.waiting ? " · waiting for Next step" : "") : "Ready: say below what to move, then hold the grip.");

  // ---- the controls
  show("p_none", !(teachOn || armed)); show("p_teach", teachOn); show("p_run", armed);
  $("p_none_text").textContent = !s.connected ? "The controls appear here once a session is started and armed." : "The controls appear here once the arm is armed.";
  document.querySelectorAll("#ctl .act").forEach(b => { b.disabled = c.busy; });
  $("jognote").textContent = c.busy ? "moving while you hold the grip…" : (c.note || "");
  const tilt = (s.arm || {}).tilt_deg == null ? null : s.arm.tilt_deg;
  $("levelbtn").disabled = c.busy || tilt == null || tilt < 0.5 || c.spots.length > 0;
  $("tiltnote").textContent = tilt == null ? "" : (tilt < 0.5 ? "The gripper points straight down." : `Tilted ${tilt.toFixed(0)}° from straight down.` + (c.spots.length ? " Remove the saved spots first." : ""));
  const sv = $("savedbtn"); show("savedbtn", !!c.saved && teachOn); if (c.saved) sv.textContent = `Use saved spots (${c.saved.names.join(", ")}; ${c.saved.when})`;
  show("demobtn", s.fake && teachOn);
  once("edit", teachOn ? c.spots : null, () => {
    $("spots_edit").replaceChildren(...(teachOn ? c.spots : []).map(x => {
      const b = el("button", "", "×"); b.title = "remove this spot"; b.onclick = () => post("/api/spot", {name: x.name, remove: true});
      const ch = chip("✓", "y", x.name, "", b); ch.title = x.mm.join(", ") + " mm, turned " + x.yaw + "°"; return ch;
    }));
    if (teachOn) { const used = c.spots.map(x => x.name), nxt = "ABCDEFGHIJKLMNOP".split("").find(l => !used.includes(l)); if (nxt && (used.includes($("spotname").value) || !$("spotname").value)) $("spotname").value = nxt; }
  });
  once("lift", teachOn ? c.lift_mm : null, () => { if (teachOn && document.activeElement !== $("lift")) $("lift").value = c.lift_mm; });

  // ---- moving blocks
  const names = c.done ? c.spots.map(x => x.name) : [];
  once("names", names, () => { options($("thing"), names, [["", "— say where —"]]); options($("dst"), names, [["", "a spot picked at random"]]); options($("tw_on"), names, [["", "— which spot —"]]); options($("e_on"), names, [["", "— which spot —"]]); lastThing = undefined; });
  const st = s.stack, nNow = st ? st.n : undefined; if (nNow !== lastN) { lastN = nNow; if (st) { $("tw_n").value = st.n == null ? "" : st.n; $("tw_on").value = st.on; } }
  const on = $("tw_on").value, feeds = names.filter(n => n !== on).slice(0, 6); ["tw_build", "tw_down", "tw_loop"].forEach(id => { $(id).disabled = ph !== "ARMED"; });
  const note = !c.done ? "" : (!on ? "Choose the spot the tower stands on. One block stands on every other spot." : `One block stands on each of ${feeds.join(", ")}. They go onto ${on} in that order: the first at the taught height, each next one a block height higher, let go ${s.drop_mm} mm above the one below.`) + (st && st.n == null ? " The last run did not end as planned: say how many blocks stand on the tower now." : "");
  $("tw_note").textContent = note; $("tw_note").title = note; show("tw_note", !busy);  // during a run the room is for the steps
  if (s.thing_at !== lastThing) { lastThing = s.thing_at; $("thing").value = s.thing_at || ""; }
  $("gobtn").disabled = ph !== "ARMED"; $("nextbtn").disabled = !(j && j.waiting && busy); show("nextbtn", busy);
  $("outcome").textContent = !j ? "" : (busy ? `Move ${Math.min(j.done + 1, j.total)} of ${j.total}: ${j.text || (j.src + " to " + j.dst)}` + (j.rehearse ? " (rehearsal)" : "") + (j.waiting ? " · waiting for Next step" : "") : `Last: ${j.done} of ${j.total} done. ${j.outcome || ""}`);
  $("steps").replaceChildren(...(j && busy ? j.steps : []).map((n, i) => { const e = el("li", j.i != null ? (i < j.i ? "past" : (i === j.i ? "cur" : "")) : "", n); return e; }));

  // ---- the camera that finds the blocks
  const e = s.eye || {}, g = e.sight;
  $("e_why").textContent = e.why ? e.why.charAt(0).toUpperCase() + e.why.slice(1) + "." : (e.shown ? `The camera was shown the table on ${e.shown} and found the block within ${e.fit_mm} mm.` : "");
  once("bars", e.bars || [], () => options($("e_kind"), e.bars || [], []));
  $("lookbtn").disabled = !e.shown || busy; $("e_show").disabled = ph !== "ARMED" || !(e.bars || []).length;
  $("e_build").disabled = !(ph === "ARMED" && e.ready && g && !g.stale && g.order.length);
  once("seen", g ? [g.t, g.stale] : null, () => { $("e_seen").replaceChildren(...(g ? g.seen.map(b => chip("✓", "y", b.name, `${b.lies}, fits ${b.fit}%`)).concat(g.unknown.map(w => chip("?", "n", "not made out", w))) : [])); });
  const plan = !g ? "" : ((g.stale ? "This picture is from before the last move: press Look again. " : "") + (g.order.length ? `Tower, bottom first: ${g.order.join(", ")} (${g.height_mm} mm high).` : "Nothing in this picture can go onto a tower.") + (g.left_out.length ? " Left out: " + g.left_out.join("; ") + "." : ""));
  $("e_plan").textContent = plan; $("e_plan").title = plan;
  lookTile(g);

  // ---- what is written over the model
  const a = s.arm, armInfo = s.arms.find(x => x.key === side()) || {};
  $("view").classList.toggle("idle", !s.connected);
  $("v_arm").textContent = (armInfo.label || "no arm") + (s.connected ? "" : " · no session");
  const tag = (id, text, cls) => { $(id).textContent = text; $(id).hidden = !text; $(id).className = "tag" + (cls ? " " + cls : ""); };
  tag("v_err", !a ? "" : (a.error == null ? "controller …" : (a.error ? "CONTROLLER ERROR " + a.error : "no controller error")), a && a.error ? "bad" : "");
  tag("v_h", !a || a.height_mm == null ? "" : f1(a.height_mm) + " mm above the lowest spot");
  tag("v_lag", !a || a.lag_mm == null ? "" : "measured against planned: " + f1(a.lag_mm) + " mm", a && a.lag_mm > 1.5 ? "bad" : "");
  tag("v_mov", a && a.moving ? "moving" : "", "live");
  tag("v_read", a && a.read_error ? "read problem: " + a.read_error : "", "bad");
  const stepName = j && busy && j.i != null ? j.steps[j.i] : "";
  $("v_caption").textContent = stepName ? `Step ${j.i + 1} of ${j.steps.length}: ${stepName}` : "";
  $("v_legend").innerHTML = s.connected && s.scene.box ? '<span><i style="background:var(--accent)"></i>where the fingertips may go</span>' + (c.spots.length ? '<span><i style="background:var(--mat)"></i>the table, from the spots</span>' : "") : "";

  // ---- log
  const lg = $("log"), atEnd = lg.scrollTop + lg.clientHeight >= lg.scrollHeight - 8;
  lg.textContent = s.log.join("\n"); if (atEnd) lg.scrollTop = lg.scrollHeight;
  $("logline").textContent = s.log.length ? s.log[s.log.length - 1] : "";
  window.HOLD = s.connected ? s.hold : null; window.PHASE = ph; window.KEYHOLD = !!s.key_hold; window.BUSY = !!c.busy;
  if (viewer) viewer.setState(s, side());
}
function pickArm() { const k = picked(); show("questrow", k === "stand-in" || window.KEYHOLD); if (viewer && last) viewer.setState(last, side()); }
async function tick() {
  try { render(await (await fetch("/api/state")).json()); }
  catch (e) { $("phase").textContent = "NO ANSWER"; $("phase").style.color = "var(--stop)"; $("hint").textContent = "The program is not answering. Assume nothing about the arm."; }
}
setInterval(tick, 250); tick();

// ---------------------------------------------------------------- buttons
$("stop").onclick = () => post("/api/stop");
$("startbtn").onclick = () => { const k = picked(); post("/api/session", {start: true, arm: k, hold: (k === "stand-in" || window.KEYHOLD) && !$("usequest").checked ? "page" : "quest"}); };
$("endbtn").onclick = () => post("/api/session", {start: false});
$("calbtn").onclick = () => post("/api/calibrate", {do: "begin"}); $("savedbtn").onclick = () => post("/api/calibrate", {do: "saved"}); $("demobtn").onclick = () => post("/api/calibrate", {do: "demo"});
$("donebtn").onclick = () => post("/api/calibrate", {do: "done"});
let waiting = null;
const jog = dir => { if (window.BUSY) { waiting = () => jog(dir); return; } window.BUSY = true; return post("/api/jog", {dir: dir, mm: seg("jogmm")}); };
const turn = dir => { if (window.BUSY) { waiting = () => turn(dir); return; } window.BUSY = true; return post("/api/turn", {dir: dir, deg: seg("jogdeg")}); };
setInterval(() => { if (waiting && !window.BUSY && window.PHASE === "AWAITING CALIBRATION") { const w = waiting; waiting = null; w(); } }, 60);
document.querySelectorAll("button[data-j]").forEach(b => { b.onclick = () => jog(b.dataset.j); });
document.querySelectorAll("button[data-t]").forEach(b => { b.onclick = () => turn(b.dataset.t); });
$("levelbtn").onclick = () => post("/api/level");
const table = here => { closePops(); return post("/api/find_table", {wide: +$("ft_w").value, deep: +$("ft_d").value, here: here}); };
$("ftbtn").onclick = () => table(false); $("fthere").onclick = () => table(true);
$("gopen").onclick = () => post("/api/grip", {close: false}); $("gclose").onclick = () => post("/api/grip", {close: true});
$("spotbtn").onclick = () => post("/api/spot", {name: $("spotname").value}); $("lift").onchange = () => post("/api/lift", {mm: +$("lift").value});
$("armbtn").onclick = () => post("/api/arm", {on: true}); $("disbtn").onclick = () => post("/api/arm", {on: false});
$("thing").onchange = () => { if ($("thing").value) post("/api/thing", {spot: $("thing").value}); };
$("gobtn").onclick = () => post("/api/move", {dst: $("dst").value || null, count: +$("count").value, rehearse: $("rehearse").checked, step: $("step").checked});
$("nextbtn").onclick = () => post("/api/next");
const tower = what => post("/api/tower", {on: $("tw_on").value, block_mm: +$("tw_h").value, n_now: $("tw_n").value === "" ? null : +$("tw_n").value, what: what, rounds: +$("tw_r").value, rehearse: $("rehearse").checked, step: $("step").checked});
$("tw_build").onclick = () => tower("build"); $("tw_down").onclick = () => tower("down"); $("tw_loop").onclick = () => tower("loop");
$("lookbtn").onclick = () => post("/api/look"); $("e_show").onclick = () => post("/api/show_table", {kind: $("e_kind").value, n: +$("e_n").value, step: $("step").checked});
$("e_build").onclick = () => post("/api/build_seen", {on: $("e_on").value, rehearse: $("rehearse").checked, step: $("step").checked});
const setTab = t => { tab = t; $("tab_blocks").classList.toggle("on", t === "blocks"); $("tab_cam").classList.toggle("on", t === "cam"); show("pane_blocks", t === "blocks"); show("pane_cam", t === "cam"); };
$("tab_blocks").onclick = () => setTab("blocks"); $("tab_cam").onclick = () => setTab("cam");
$("logbtn").onclick = () => { show("logdrawer", $("logdrawer").hidden); const lg = $("log"); lg.scrollTop = lg.scrollHeight; };
// the small sheets (help, find the table): a button with data-pop opens one over the controls; a click elsewhere closes it
function closePops() { document.querySelectorAll(".pop.on").forEach(p => p.classList.remove("on")); }
addEventListener("click", e => {
  const b = e.target.closest && e.target.closest("[data-pop]"), inside = e.target.closest && e.target.closest(".pop");
  if (b) { const p = $(b.dataset.pop), was = p.classList.contains("on"); closePops(); p.classList.toggle("on", !was); }
  else if (!inside) closePops();
  if (!(e.target.closest && e.target.closest("#logdrawer,#logbtn"))) show("logdrawer", false);
});
addEventListener("keydown", e => { if (e.code === "Escape") { closePops(); show("logdrawer", false); } });
document.querySelectorAll("#v_views button").forEach(b => { b.onclick = () => { document.querySelectorAll("#v_views button").forEach(x => x.classList.toggle("on", x === b)); if (viewer) viewer.view(b.dataset.view); }; });

// ---------------------------------------------------------------- keys
// the lab's own nudge keys: W/S away from and towards the base, A/D left and right, Q/E up and down; Z/C turn the gripper
const KEYS = {KeyW: "fwd", KeyS: "back", KeyA: "left", KeyD: "right", KeyQ: "up", KeyE: "down"};
const TURN = {KeyZ: "ccw", KeyC: "cw"};
const typing = e => e.target.tagName === "TEXTAREA" || (e.target.tagName === "INPUT" && ["text", "number"].includes(e.target.type));
const flash = code => { const b = document.querySelector(`[data-k=${code}]`); if (b) { b.classList.add("down"); setTimeout(() => b.classList.remove("down"), 140); } };
addEventListener("keydown", e => { if (!(KEYS[e.code] || TURN[e.code]) || window.PHASE !== "AWAITING CALIBRATION" || e.repeat || e.metaKey || e.ctrlKey || typing(e)) return; e.preventDefault(); flash(e.code); KEYS[e.code] ? jog(KEYS[e.code]) : turn(TURN[e.code]); });
addEventListener("keydown", e => { if (!["KeyF", "KeyR"].includes(e.code) || window.PHASE !== "AWAITING CALIBRATION" || e.repeat || e.metaKey || e.ctrlKey || typing(e)) return; e.preventDefault(); flash(e.code); post("/api/grip", {close: e.code === "KeyF"}); });
// The space bar as the hold-to-run switch (the stand-in arm; a real arm only if the program was started with --key_hold).
// Like the headset, the page reports its state ALL the time (pressed or not), so that a closed or frozen page reads as
// "no signal", not as "released and fine".
// A button or a tick box that was just clicked keeps the keyboard, and the space bar would press it again: it is let go of, so that the space bar is only ever the switch.
let down = false;
addEventListener("change", e => { if (e.target.tagName === "SELECT" || (e.target.tagName === "INPUT" && e.target.type === "number")) e.target.blur(); });
addEventListener("keydown", e => { if (e.code === "Enter" && e.target.tagName === "INPUT") e.target.blur(); });
addEventListener("click", e => { const b = e.target.closest && e.target.closest("button,input[type=checkbox],input[type=radio]"); if (b) b.blur(); });
const tell = () => { if (window.HOLD === "page") fetch("/api/hold", {method: "POST", headers: {"Content-Type": "application/json", "X-Token": TOKEN}, body: JSON.stringify({pressed: down}), keepalive: true}).catch(() => {}); };
setInterval(tell, 50);
addEventListener("keydown", e => { if (e.code !== "Space" || window.HOLD !== "page" || typing(e)) return; e.preventDefault(); if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur(); down = true; $("spacekey").classList.add("on"); });
let upTimer = null;
addEventListener("keydown", e => { if (e.code === "Space" && upTimer) { clearTimeout(upTimer); upTimer = null; } }, true);
addEventListener("keyup", e => { if (e.code !== "Space") return; if (window.HOLD === "page" && !typing(e)) e.preventDefault();
  clearTimeout(upTimer); upTimer = setTimeout(() => { upTimer = null; down = false; tell(); }, 40); }); addEventListener("blur", () => { down = false; tell(); });

// ---------------------------------------------------------------- the model of the arm
// Loaded apart from everything above: if the model cannot be drawn (no WebGL, a file missing) every button still works.
import("/web/scene.js").then(m => m.start($("view"))).then(v => { viewer = v; if (last) viewer.setState(last, side()); })
  .catch(e => { $("v_fail").hidden = false; $("v_fail").textContent = "The model of the arm cannot be drawn here (" + e + "). Everything else works."; });
async function poseLoop() {  // ten times a second: the joint angles the controller publishes
  for (;;) {
    await sleep(100);
    if (!viewer || document.hidden || !(last && last.connected)) continue;
    try { viewer.setPose(await (await fetch("/api/view")).json()); } catch (e) { /* the state poll says what is wrong */ }
  }
}
poseLoop();

// ---------------------------------------------------------------- the live pictures
// One request is kept waiting for the next pictures of all cameras, so that the pictures never use up the few
// connections a browser allows: the hold-to-run reports must not wait behind them. Every picture comes with its age;
// a picture that stopped arriving turns grey and says for how long.
const tiles = new Map(); let camSeq = 0, focus = null;
function tile(id, label) {
  let t = tiles.get(id);
  if (!t) {
    const f = el("figure", "cam"), cv = el("canvas"), cap = el("figcaption"), nm = el("span", "nm"), stt = el("span", "st"), why = el("div", "why");
    cap.append(nm, stt); f.append(cv, why, cap); f.onclick = () => { focus = focus === id ? null : id; layoutCams(); };
    t = {f, cv, nm, stt, why, at: null, note: "", got: false}; tiles.set(id, t); $("camgrid").append(f); layoutCams();
  }
  return t;
}
function layoutCams() {
  if (focus && !tiles.has(focus)) focus = null;
  const g = $("camgrid"); g.classList.toggle("focus", !!focus); g.style.setProperty("--rest", Math.max(1, tiles.size - 1));
  tiles.forEach((t, id) => t.f.classList.toggle("big", id === focus));
}
function dropTile(id) { const t = tiles.get(id); if (t) { t.f.remove(); tiles.delete(id); layoutCams(); } }
function lookTile(g) {  // what the block finder made of its last picture: one more tile next to the live ones
  if (!g) { dropTile("look"); sig.lookimg = null; return; }
  const t = tile("look");
  if (!t.img) { t.img = el("img"); t.cv.replaceWith(t.img); t.img.alt = "What the camera saw, with the blocks it made out drawn in"; }
  once("lookimg", g.t, () => { t.img.src = "/api/sight.jpg?t=" + g.t; });
  t.nm.textContent = "What it made out"; t.f.classList.toggle("old", !!g.stale); t.stt.textContent = g.stale ? "before the last move" : "last look"; t.why.textContent = "";
}
async function camLoop() {
  for (;;) {
    const sd = side();
    if (!last || !last.feeds) { [...tiles.keys()].filter(k => k !== "look").forEach(dropTile); show("cam_none", !tiles.size); $("cam_none").textContent = last ? "The live pictures are switched off." : ""; await sleep(600); continue; }
    if (!sd || document.hidden) { await sleep(400); continue; }
    try {
      const r = await fetch(`/api/frames?after=${camSeq}&side=${encodeURIComponent(sd)}`, {headers: {"X-Token": TOKEN}});
      if (!r.ok) throw new Error("refused (" + r.status + ")");
      const buf = await r.arrayBuffer(), n = new DataView(buf).getUint32(0), head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
      let off = 4 + n; camSeq = head.seq;
      [...tiles.keys()].filter(k => k !== "look" && !head.cams.some(c => c.id === k)).forEach(dropTile);
      for (const c of head.cams) {
        const t = tile(c.id);
        t.nm.textContent = c.label; if (c.finds) t.nm.append(el("em", "", "finds the blocks"));
        t.note = c.note || ""; if (c.age_ms != null) t.at = performance.now() - c.age_ms;
        if (c.len) {
          const blob = new Blob([new Uint8Array(buf, off, c.len)], {type: "image/jpeg"}); off += c.len;
          createImageBitmap(blob).then(bmp => { if (t.cv.width !== bmp.width || t.cv.height !== bmp.height) { t.cv.width = bmp.width; t.cv.height = bmp.height; } t.cv.getContext("2d").drawImage(bmp, 0, 0); bmp.close(); t.got = true; }).catch(() => {});
        }
      }
      show("cam_none", !tiles.size); $("cam_none").textContent = !head.on ? head.why : (head.cams.length ? "" : "No camera belongs to this arm.");
      if (!head.on) await sleep(2000);
    } catch (e) { tiles.forEach((t, id) => { if (id !== "look") t.note = "no pictures: " + (e.message || e); }); await sleep(1000); }
  }
}
setInterval(() => {  // a picture's age, said honestly
  const now = performance.now();
  tiles.forEach((t, id) => {
    if (id === "look") return;
    const age = t.at == null ? null : (now - t.at) / 1000, live = age != null && age < 1.5 && !t.note;
    t.f.classList.toggle("old", !live && t.got); t.f.classList.toggle("bad", !!t.note);
    t.stt.textContent = live ? "live" : (t.note ? (t.got ? "stopped" : "no picture") : (age == null ? "waiting…" : `${age < 10 ? age.toFixed(1) : Math.round(age)} s old`));
    t.why.textContent = !live && (t.note || !t.got) ? (t.note || "waiting for the first picture…") : "";
  });
}, 300);
camLoop();
