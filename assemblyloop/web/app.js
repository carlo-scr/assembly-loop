// The dashboard's controls and feedback. The program owns the session and every motion decision;
// this page polls its state and sends explicit operator actions. The model lives separately in scene.js.
"use strict";
const TOKEN = window.TOKEN, $ = id => document.getElementById(id);
const COL = {"OFF": "--off", "STOPPED": "--stop", "DISARMED": "--dis", "AWAITING CALIBRATION": "--cal", "ARMED": "--accent", "RUNNING": "--ok", "PAUSED": "--cal"};
const sleep = ms => new Promise(r => setTimeout(r, ms));
let viewer = null, last = null, sig = {}, lastThing, lastN, tab = "blocks";
let stateOnline = false, polling = false, pollAgain = false, waiting = null, stateEpoch = 0;
const pending = new Map();
const COMMAND_TIMEOUT_MS = 8000, STOP_TIMEOUT_MS = 3000;

function say(text) { $("msg_text").textContent = text || ""; $("msg").hidden = !text; }
$("msg_close").onclick = () => say("");
function disable(id, off, reason = "") {
  const b = typeof id === "string" ? $(id) : id;
  if (!("readyTitle" in b.dataset)) b.dataset.readyTitle = b.title;
  b.disabled = !!off;
  b.title = off && reason ? reason : b.dataset.readyTitle;
}
function post(path, body) {
  const stopping = path === "/api/stop";
  // Each explicit STOP gets its own request, even if an earlier acknowledgement is missing.
  // Other actions are deduplicated until their acknowledgement or timeout; none are retried automatically.
  if (!stopping && pending.has(path)) return pending.get(path);
  if (!stateOnline && !stopping) {
    const error = "Wait for the dashboard to reconnect before sending another command.";
    say(error); return Promise.resolve({error});
  }
  say("");
  stateEpoch += 1;
  const key = stopping ? Symbol("stop") : path, controller = new AbortController();
  const deadline = stopping ? STOP_TIMEOUT_MS : COMMAND_TIMEOUT_MS;
  const request = (async () => {
    let timeout, timedOut = false;
    try {
      const answer = async () => {
        const r = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json", "X-Token": TOKEN}, body: JSON.stringify(body || {}), signal: controller.signal});
        const j = await r.json();
        if (!r.ok && !j.error) j.error = "The command was refused (" + r.status + ").";
        return j;
      };
      const j = await Promise.race([answer(), new Promise((resolve, reject) => {
        timeout = setTimeout(() => { timedOut = true; controller.abort(); reject(new Error("Command acknowledgement timed out")); }, deadline);
      })]);
      if (j.error) say(j.error);
      return j;
    } catch (e) {
      const error = (stopping ? "STOP was not acknowledged" : "The command was not acknowledged") +
        (timedOut ? ` within ${deadline / 1000} seconds. ` : ". ") +
        "Its outcome is unknown. Refreshing robot state; check the activity log before trying again.";
      offline(); say(error); return {error};
    } finally {
      clearTimeout(timeout);
      pending.delete(key);
      stateEpoch += 1;
      tick();
    }
  })();
  pending.set(key, request);
  if (last && stateOnline) render(last);
  return request;
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
document.querySelectorAll(".seg[id^=jog]").forEach(s => {
  s.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.classList.contains("on"))));
  s.addEventListener("click", e => { const b = e.target.closest("button"); if (b) s.querySelectorAll("button").forEach(x => { x.classList.toggle("on", x === b); x.setAttribute("aria-pressed", String(x === b)); }); });
});
const picked = () => (document.querySelector("input[name=arm]:checked") || {}).value;
const side = () => (last && last.connected) ? last.chosen : picked();

function availability(s) {
  const ph = s.phase, c = s.calibration, teaching = ph === "AWAITING CALIBRATION", busy = ["RUNNING", "PAUSED"].includes(ph);
  const commandPending = pending.size > 0, waitingReason = commandPending ? "Waiting for the current command." : "Wait until the movement finishes.";
  const checks = s.checks || [], ready = checks.length > 0 && checks.every(k => k.ok), e = s.eye || {}, g = e.sight;
  document.querySelectorAll("#ctl .act").forEach(b => disable(b, !teaching || c.busy || commandPending, !teaching ? "Arm the session and open Teach places first." : waitingReason));
  disable("startbtn", s.connected || !picked() || commandPending, commandPending ? waitingReason : "Choose an arm first.");
  disable("endbtn", !s.connected || pending.has("/api/session"));
  disable("armbtn", ph !== "DISARMED" || !ready || commandPending, commandPending ? waitingReason : (checks.find(k => !k.ok) || {}).detail || "All three checks must pass before arming.");
  disable("disbtn", !(teaching || ph === "ARMED" || busy) || pending.has("/api/arm"), "The session is already disarmed.");
  disable("calbtn", ph !== "ARMED" || commandPending, "Finish the current task before teaching places again.");
  disable("donebtn", !teaching || c.busy || commandPending || c.spots.length < 2, c.spots.length < 2 ? "Save at least two places to continue." : waitingReason);
  const tilt = (s.arm || {}).tilt_deg;
  disable("levelbtn", !teaching || c.busy || commandPending || tilt == null || tilt < 0.5 || c.spots.length > 0,
    c.spots.length ? "Remove the taught places before changing the gripper tilt." : tilt == null ? "Waiting for the gripper's measured tilt." : tilt < 0.5 ? "The gripper already points down." : waitingReason);
  disable("lift", !teaching || c.busy || commandPending, waitingReason);
  ["thing", "e_source"].forEach(id => disable(id, ph !== "ARMED" || commandPending, "Wait for the session to be ready before changing the source spot."));
  const source = $("thing").value, destination = $("dst").value;
  disable("gobtn", ph !== "ARMED" || commandPending || !source || destination === source,
    !source ? "Choose where the block stands." : destination === source ? "Choose a different destination." : waitingReason);
  ["tw_build", "tw_down", "tw_loop"].forEach(id => disable(id, ph !== "ARMED" || commandPending || !$("tw_on").value,
    !$("tw_on").value ? "Choose the tower spot." : waitingReason));
  disable("nextbtn", !(s.job && s.job.waiting && busy) || commandPending, "Available when the task is waiting for its next step.");
  disable("lookbtn", !e.shown || busy || commandPending, !e.shown ? "Calibrate the camera before detecting blocks." : waitingReason);
  disable("e_show", ph !== "ARMED" || commandPending || !(e.bars || []).length || !$("e_source").value,
    !(e.bars || []).length ? (e.why || "No calibration block is available.") : !$("e_source").value ? "Choose the spot holding the calibration block." : waitingReason);
  disable("e_build", !(ph === "ARMED" && e.ready && g && !g.stale && g.order.length && $("e_on").value) || commandPending,
    !$("e_on").value ? "Choose the tower spot." : !g || g.stale ? "Detect the blocks again to use a current picture." : !e.ready ? (e.why || "Calibrate the camera first.") : !g.order.length ? "No detected blocks can form a tower." : waitingReason);
  ["rehearse", "step"].forEach(id => disable(id, busy || commandPending, "These options are fixed for the current task."));
}

function runFeedback(s) {
  const j = s.job, busy = ["RUNNING", "PAUSED"].includes(s.phase), waitingStep = !!(busy && j && j.waiting);
  const calibrating = busy && !j;
  const names = {blocks: "Move a block", tower: "Build a tower", cam: "Camera tasks"};
  const teaching = s.phase === "AWAITING CALIBRATION", ready = s.phase === "ARMED";
  $("workspace_title").textContent = teaching ? "Teach places" : calibrating ? "Calibration in progress" : busy ? "Task in progress" : ready ? names[tab] : s.phase === "STOPPED" ? "Session stopped" : "Prepare your session";
  $("workspace_subtitle").textContent = teaching ? "Position the gripper, save each place, then confirm your workspace." : busy ? "Release the grip to pause. STOP SESSION ends the session." : ready ? "Choose a task, check its settings, and hold the grip to move." : "Choose an arm and complete the session checks to unlock the controls.";
  $("run_mode").textContent = calibrating ? "Calibration · physical motion" : busy ? (j && j.rehearse ? "Current task: air rehearsal" : "Current task: physical motion") : ($("rehearse").checked ? "Air rehearsal" : "Physical placement") + " · " + ($("step").checked ? "One step at a time" : "Continuous sequence");
  show("job_empty", !j && !busy); show("jobbox", !!j || busy); show("run_progress", !!j || calibrating);
  show("step_details", !!(j && j.steps && j.steps.length));
  if (!j) {
    $("run_status").textContent = calibrating ? (s.phase === "PAUSED" ? "Calibration paused" : "Calibration in progress") : "Ready for a task";
    $("run_step").textContent = calibrating ? "Hold the grip to continue calibration. Step details will appear when available." : "";
    if (calibrating) { $("run_progress").removeAttribute("value"); $("run_progress").setAttribute("aria-valuetext", "Calibration progress is not available"); }
    return;
  }
  const steps = j.steps || [], current = busy && j.i != null ? steps[j.i] : "";
  $("run_status").textContent = s.phase === "PAUSED" ? "Task paused" : waitingStep ? "Ready for the next step" : busy ? "Running" : j.outcome === "done" ? "Task complete" : "Last task";
  $("run_step").textContent = current ? `Step ${j.i + 1} of ${steps.length}: ${current}` : (!busy ? (j.outcome || "Task ended.") : "Waiting for task details…");
  const progress = $("run_progress"), total = Math.max(1, j.total || 1);
  progress.max = total;
  progress.value = Math.min(total, Math.max(0, j.done || 0) + (busy && steps.length && j.i != null ? j.i / steps.length : 0));
  progress.setAttribute("aria-label", "Task progress");
  progress.setAttribute("aria-valuetext", `${j.done || 0} of ${j.total || 1} moves complete${current ? ", " + current : ""}`);
}

function interfaceHint(s) {
  if (s.phase === "AWAITING CALIBRATION") return s.hint.replace("Then press Done.", "Then choose Finish teaching.");
  if (s.phase === "ARMED") return "Choose Move a block, Tower, or Camera, check the run options, then hold the grip to move.";
  if (s.phase === "PAUSED" && s.quest && s.quest.why === "released") return "Paused: hold the grip" + (s.job && s.job.waiting ? " and choose Next step to continue." : " to continue.");
  return s.hint;
}

function render(s) {
  last = s;
  const ph = s.phase, c = s.calibration, teachOn = ph === "AWAITING CALIBRATION", busy = ["RUNNING", "PAUSED"].includes(ph), armed = ph === "ARMED" || busy, j = s.job, q = s.quest;
  const p = $("phase"); p.textContent = ph + (s.fake ? " · stand-in arm" : ""); p.style.color = `var(${COL[ph]})`;
  const hint = interfaceHint(s); $("hint").textContent = hint; $("hint").title = hint;
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
  $("seq").classList.toggle("setup-armed", teachOn || armed);
  $("seq").classList.toggle("setup-complete", armed);
  const rung = {"OFF": [1, 0, 0, 0, 0], "STOPPED": [1, 0, 0, 0, 0], "DISARMED": [2, ready ? 2 : 1, ready ? 1 : 0, 0, 0], "AWAITING CALIBRATION": [2, 2, 2, 1, 0]}[ph] || [2, 2, 2, 2, 1];
  rung.forEach((v, i) => { const li = $("r" + (i + 1)); li.className = ["todo", "cur", "done"][v]; li.querySelector(".n").textContent = v === 2 ? "✓" : i + 1; if (v === 1) li.setAttribute("aria-current", "step"); else li.removeAttribute("aria-current"); });
  const sums = [(s.arms.find(x => x.key === s.chosen) || {}).label || "", teachOn || armed ? "verified before arming" : ready ? "3 checks passed" : "", teachOn || armed ? "enabled" : "", c.done ? `${c.spots.map(x => x.name).join(", ")} · lift ${c.lift_mm} mm` : "", ""];
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
  $("checks").replaceChildren(...(s.checks || []).map((k, i) => {
    // Releasing the grip is a prerequisite to arming, not a fault while an armed operator holds it.
    // Controller health and incoming hold signal remain live checks; the header shows the current grip.
    if (i === 2 && (teachOn || armed)) return chip("✓", "y", "Grip released before arming", "current grip shown above");
    return chip(k.ok ? "✓" : "✗", k.ok ? "y" : "n", k.text, k.ok ? "" : k.detail);
  }));
  show("armbtn", !(teachOn || armed)); show("disbtn", teachOn || armed);
  $("c_status").textContent = c.done ? `Known: ${c.spots.map(x => x.name).join(", ")} · lift ${c.lift_mm} mm` : (teachOn ? (c.spots.length ? `${c.spots.length} saved so far: ${c.spots.map(x => x.name).join(", ")}` : "Nudge the gripper to each place below and save it.") : "Not known yet.");
  show("calbtn", c.done && !teachOn);
  $("runline").textContent = !armed ? "Starts once the places are known." : (busy && j ? `Move ${Math.min(j.done + 1, j.total)} of ${j.total}: ${j.text || (j.src + " to " + j.dst)}` + (j.rehearse ? " (rehearsal)" : "") + (j.waiting ? " · waiting for Next step" : "") : busy ? "Calibration in progress; task details are not available yet." : "Ready: choose a task below, then hold the grip.");

  // ---- the controls
  show("p_none", !(teachOn || armed)); show("p_teach", teachOn); show("p_run", armed);
  $("p_none_text").textContent = !s.connected ? "The controls appear here once a session is started and armed." : "The controls appear here once the arm is armed.";
  $("jognote").textContent = c.busy ? (q && q.held ? "Command active while the grip is held." : "Command waiting. Hold the grip to move.") : (c.note || "");
  const tilt = (s.arm || {}).tilt_deg == null ? null : s.arm.tilt_deg;
  $("tiltnote").textContent = tilt == null ? "" : (tilt < 0.5 ? "The gripper points straight down." : `Tilted ${tilt.toFixed(0)}° from straight down.` + (c.spots.length ? " Remove the saved spots first." : ""));
  const sv = $("savedbtn"); show("savedbtn", !!c.saved && teachOn); if (c.saved) sv.textContent = `Use saved spots (${c.saved.names.join(", ")}; ${c.saved.when})`;
  show("demobtn", s.fake && teachOn);
  once("edit", teachOn ? c.spots : null, () => {
    $("spots_edit").replaceChildren(...(teachOn ? c.spots : []).map(x => {
      const b = el("button", "act", "×"); b.title = "Remove spot " + x.name; b.setAttribute("aria-label", b.title); b.onclick = () => post("/api/spot", {name: x.name, remove: true});
      const ch = chip("✓", "y", x.name, "", b); ch.title = x.mm.join(", ") + " mm, turned " + x.yaw + "°"; return ch;
    }));
    if (teachOn) { const used = c.spots.map(x => x.name), nxt = "ABCDEFGHIJKLMNOP".split("").find(l => !used.includes(l)); if (nxt && (used.includes($("spotname").value) || !$("spotname").value)) $("spotname").value = nxt; }
  });
  once("lift", teachOn ? c.lift_mm : null, () => { if (teachOn && document.activeElement !== $("lift")) $("lift").value = c.lift_mm; });

  // ---- moving blocks
  const names = c.done ? c.spots.map(x => x.name) : [];
  once("names", names, () => { ["thing", "e_source", "tw_on", "e_on"].forEach(id => options($(id), names, [["", "Choose a spot"]])); options($("dst"), names, [["", "Random destination"]]); lastThing = undefined; });
  const st = s.stack, nNow = st ? st.n : undefined; if (nNow !== lastN) { lastN = nNow; if (st) { $("tw_n").value = st.n == null ? "" : st.n; $("tw_on").value = st.on; } }
  const on = $("tw_on").value, feeds = names.filter(n => n !== on).slice(0, 6);
  const note = !c.done ? "" : (!on ? "Choose the spot the tower stands on. One block stands on every other spot." : `One block stands on each of ${feeds.join(", ")}. They go onto ${on} in that order: the first at the taught height, each next one a block height higher, let go ${s.drop_mm} mm above the one below.`) + (st && st.n == null ? " The last run did not end as planned: say how many blocks stand on the tower now." : "");
  $("tw_note").textContent = note; $("tw_note").title = note; show("tw_note", !busy);  // during a run the room is for the steps
  if (!pending.has("/api/thing") && s.thing_at !== lastThing) { lastThing = s.thing_at; $("thing").value = $("e_source").value = s.thing_at || ""; }
  show("nextbtn", !!(busy && j));
  $("outcome").textContent = !j ? "" : (busy ? `Move ${Math.min(j.done + 1, j.total)} of ${j.total}: ${j.text || (j.src + " to " + j.dst)}` + (j.rehearse ? " (rehearsal)" : "") + (j.waiting ? " · waiting for Next step" : "") : `Last: ${j.done} of ${j.total} done. ${j.outcome || ""}`);
  once("steps", j ? [j.steps, j.i, busy, j.outcome] : null, () => {
    $("steps").replaceChildren(...(j ? j.steps : []).map((n, i) => {
      const state = busy && j.i != null ? (i < j.i ? "past" : i === j.i ? "cur" : "") : j.outcome === "done" ? "past" : "";
      const item = el("li", state, n); if (state === "cur") item.setAttribute("aria-current", "step"); return item;
    }));
  });

  // ---- the camera that finds the blocks
  const e = s.eye || {}, g = e.sight;
  once("camera-calibrated", !!e.shown, () => { document.querySelector(".camera-setup").open = !e.shown; });
  $("e_why").textContent = e.why ? e.why.charAt(0).toUpperCase() + e.why.slice(1) + "." : (e.shown ? `The camera was shown the table on ${e.shown} and found the block within ${e.fit_mm} mm.` : "");
  once("bars", e.bars || [], () => options($("e_kind"), e.bars || [], []));
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
  window.HOLD = s.connected ? s.hold : null; window.PHASE = ph; window.KEYHOLD = !!s.key_hold;
  // Keep an optimistic movement busy until a fresh post-acknowledgement state is rendered.
  // Otherwise a queued key can drain against the last idle snapshot before the worker becomes visible.
  window.BUSY = !!c.busy || ["/api/jog", "/api/turn", "/api/grip", "/api/level", "/api/find_table"].some(path => pending.has(path));
  if (!teachOn) waiting = null;
  document.body.dataset.connection = "online";
  availability(s); runFeedback(s);
  if (viewer) viewer.setState(s, side());
}
function pickArm() { const k = picked(); show("questrow", k === "stand-in" || window.KEYHOLD); if (last && stateOnline) availability(last); if (viewer && last) viewer.setState(last, side()); }
function offline() {
  stateOnline = false; waiting = null; releaseHold();
  window.PHASE = "NO ANSWER";
  document.body.dataset.connection = "offline";
  $("phase").textContent = "NO ANSWER"; $("phase").style.color = "var(--stop)";
  $("hint").textContent = "The program is not answering. Robot state is unknown; waiting to reconnect.";
  $("held").textContent = "signal unknown"; $("holdsub").textContent = "last state is no longer current";
  $("hold").classList.remove("on");
  ["startbtn", "endbtn", "armbtn", "disbtn", "calbtn", "gobtn", "tw_build", "tw_down", "tw_loop", "nextbtn", "lookbtn", "e_show", "e_build", "thing", "e_source", "lift"].forEach(id => disable(id, true, "Waiting for the dashboard to reconnect."));
  document.querySelectorAll("#ctl .act").forEach(b => disable(b, true, "Waiting for the dashboard to reconnect."));
  $("stop").disabled = !(last && last.connected);
  $("v_read").textContent = "Connection lost · last known pose"; $("v_read").className = "tag bad"; show("v_read", true);
}
async function tick() {
  if (polling) { pollAgain = true; return; }
  polling = true;
  const epoch = stateEpoch, controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 2000);
  try {
    const response = await fetch("/api/state", {signal: controller.signal});
    if (!response.ok) throw new Error("State request failed (" + response.status + ")");
    const s = await response.json();
    if (epoch === stateEpoch) { stateOnline = true; render(s); }
    else pollAgain = true;
  } catch (e) { offline(); }
  finally {
    clearTimeout(timeout); polling = false;
    if (pollAgain) { pollAgain = false; setTimeout(tick, 0); }
  }
}
setInterval(tick, 250); tick();

// ---------------------------------------------------------------- buttons
$("stop").onclick = () => { releaseHold(); post("/api/stop"); };
$("startbtn").onclick = () => { const k = picked(); post("/api/session", {start: true, arm: k, hold: (k === "stand-in" || window.KEYHOLD) && !$("usequest").checked ? "page" : "quest"}); };
$("endbtn").onclick = () => post("/api/session", {start: false});
$("calbtn").onclick = () => post("/api/calibrate", {do: "begin"}); $("savedbtn").onclick = () => post("/api/calibrate", {do: "saved"}); $("demobtn").onclick = () => post("/api/calibrate", {do: "demo"});
$("donebtn").onclick = () => post("/api/calibrate", {do: "done"});
const jog = dir => { if (!stateOnline || window.PHASE !== "AWAITING CALIBRATION") return; if (window.BUSY || pending.size) { waiting = () => jog(dir); return; } window.BUSY = true; return post("/api/jog", {dir: dir, mm: seg("jogmm")}); };
const turn = dir => { if (!stateOnline || window.PHASE !== "AWAITING CALIBRATION") return; if (window.BUSY || pending.size) { waiting = () => turn(dir); return; } window.BUSY = true; return post("/api/turn", {dir: dir, deg: seg("jogdeg")}); };
setInterval(() => { if (waiting && stateOnline && !pending.size && !window.BUSY && window.PHASE === "AWAITING CALIBRATION") { const w = waiting; waiting = null; w(); } }, 60);
document.querySelectorAll("button[data-j]").forEach(b => { b.onclick = () => jog(b.dataset.j); });
document.querySelectorAll("button[data-t]").forEach(b => { b.onclick = () => turn(b.dataset.t); });
$("levelbtn").onclick = () => post("/api/level");
const table = here => { closePops(); return post("/api/find_table", {wide: +$("ft_w").value, deep: +$("ft_d").value, here: here}); };
$("ftbtn").onclick = () => table(false); $("fthere").onclick = () => table(true);
$("gopen").onclick = () => post("/api/grip", {close: false}); $("gclose").onclick = () => post("/api/grip", {close: true});
$("spotbtn").onclick = () => post("/api/spot", {name: $("spotname").value}); $("lift").onchange = () => post("/api/lift", {mm: +$("lift").value});
$("armbtn").onclick = () => post("/api/arm", {on: true}); $("disbtn").onclick = () => { releaseHold(); post("/api/arm", {on: false}); };
function chooseSource(select) {
  const spot = select.value;
  $("thing").value = $("e_source").value = spot;
  if (spot) { lastThing = undefined; post("/api/thing", {spot}); }
  else if (last && stateOnline) availability(last);
}
$("thing").onchange = () => chooseSource($("thing")); $("e_source").onchange = () => chooseSource($("e_source"));
$("gobtn").onclick = () => post("/api/move", {dst: $("dst").value || null, count: +$("count").value, rehearse: $("rehearse").checked, step: $("step").checked});
$("nextbtn").onclick = () => post("/api/next");
const tower = what => post("/api/tower", {on: $("tw_on").value, block_mm: +$("tw_h").value, n_now: $("tw_n").value === "" ? null : +$("tw_n").value, what: what, rounds: +$("tw_r").value, rehearse: $("rehearse").checked, step: $("step").checked});
$("tw_build").onclick = () => tower("build"); $("tw_down").onclick = () => tower("down"); $("tw_loop").onclick = () => tower("loop");
$("lookbtn").onclick = () => post("/api/look"); $("e_show").onclick = () => post("/api/show_table", {kind: $("e_kind").value, n: +$("e_n").value, step: $("step").checked});
$("e_build").onclick = () => post("/api/build_seen", {on: $("e_on").value, rehearse: $("rehearse").checked, step: $("step").checked});
const taskTabs = ["blocks", "tower", "cam"];
function setTab(t, focusTab = false) {
  if (!taskTabs.includes(t)) return;
  tab = t;
  taskTabs.forEach(name => {
    const b = $("tab_" + name), active = name === t;
    b.classList.toggle("on", active); b.setAttribute("aria-selected", String(active)); b.tabIndex = active ? 0 : -1;
    show("pane_" + name, active);
  });
  if (focusTab) $("tab_" + t).focus();
  if (last && stateOnline) runFeedback(last);
}
taskTabs.forEach((t, i) => {
  const b = $("tab_" + t); b.onclick = () => setTab(t);
  b.addEventListener("keydown", e => {
    const next = e.key === "ArrowRight" ? (i + 1) % taskTabs.length : e.key === "ArrowLeft" ? (i + taskTabs.length - 1) % taskTabs.length : e.key === "Home" ? 0 : e.key === "End" ? taskTabs.length - 1 : null;
    if (next != null) { e.preventDefault(); setTab(taskTabs[next], true); }
  });
});
$("setup-toggle").onclick = () => {
  const expanded = $("seq").classList.toggle("expanded");
  $("setup-toggle").setAttribute("aria-expanded", String(expanded));
  $("setup-toggle").textContent = expanded ? "Hide details" : "Show details";
};
["dst", "tw_on", "e_on", "rehearse", "step"].forEach(id => $(id).addEventListener("change", () => { if (last && stateOnline) { availability(last); runFeedback(last); } }));
function showLog(open) { show("logdrawer", open); $("logbtn").setAttribute("aria-expanded", String(open)); }
$("logbtn").onclick = () => { showLog($("logdrawer").hidden); const lg = $("log"); lg.scrollTop = lg.scrollHeight; };
// the small sheets (help, find the table): a button with data-pop opens one over the controls; a click elsewhere closes it
let popTrigger = null;
function closePops(restore = false) {
  document.querySelectorAll(".pop.on").forEach(p => p.classList.remove("on"));
  document.querySelectorAll("[data-pop]").forEach(b => b.setAttribute("aria-expanded", "false"));
  if (restore && popTrigger && !popTrigger.disabled) popTrigger.focus();
  popTrigger = null;
}
document.querySelectorAll("[data-pop]").forEach(b => { b.setAttribute("aria-controls", b.dataset.pop); b.setAttribute("aria-expanded", "false"); });
document.querySelectorAll(".pop-close").forEach(b => { b.onclick = () => closePops(true); });
addEventListener("click", e => {
  const b = e.target.closest && e.target.closest("[data-pop]"), inside = e.target.closest && e.target.closest(".pop");
  if (b && !b.disabled) { const p = $(b.dataset.pop), was = p.classList.contains("on"); closePops(); if (!was) { p.classList.add("on"); b.setAttribute("aria-expanded", "true"); popTrigger = b; p.focus(); } }
  else if (!inside) closePops();
  if (!(e.target.closest && e.target.closest("#logdrawer,#logbtn"))) showLog(false);
});
addEventListener("keydown", e => { if (e.code === "Escape") { closePops(true); showLog(false); } });
document.querySelectorAll("#v_views button").forEach(b => { b.onclick = () => { document.querySelectorAll("#v_views button").forEach(x => { x.classList.toggle("on", x === b); x.setAttribute("aria-pressed", String(x === b)); }); if (viewer) viewer.view(b.dataset.view); }; });

// ---------------------------------------------------------------- keys
// the lab's own nudge keys: W/S away from and towards the base, A/D left and right, Q/E up and down; Z/C turn the gripper
const KEYS = {KeyW: "fwd", KeyS: "back", KeyA: "left", KeyD: "right", KeyQ: "up", KeyE: "down"};
const TURN = {KeyZ: "ccw", KeyC: "cw"};
const typing = e => !!(e.target.closest && e.target.closest("input,select,textarea,[contenteditable]:not([contenteditable=false]),.pop.on,[role=dialog]"));
const flash = code => { const b = document.querySelector(`[data-k=${code}]`); if (b) { b.classList.add("down"); setTimeout(() => b.classList.remove("down"), 140); } };
addEventListener("keydown", e => { if (!(KEYS[e.code] || TURN[e.code]) || window.PHASE !== "AWAITING CALIBRATION" || e.repeat || e.metaKey || e.ctrlKey || typing(e)) return; e.preventDefault(); flash(e.code); KEYS[e.code] ? jog(KEYS[e.code]) : turn(TURN[e.code]); });
addEventListener("keydown", e => { if (!["KeyF", "KeyR"].includes(e.code) || window.PHASE !== "AWAITING CALIBRATION" || e.repeat || e.metaKey || e.ctrlKey || typing(e)) return; e.preventDefault(); if (window.BUSY || pending.size) return; flash(e.code); post("/api/grip", {close: e.code === "KeyF"}); });
// The space bar as the hold-to-run switch (the stand-in arm; a real arm only if the program was started with --key_hold).
// Like the headset, the page reports its state ALL the time (pressed or not), so that a closed or frozen page reads as
// "no signal", not as "released and fine".
// A button or a tick box that was just clicked keeps the keyboard, and the space bar would press it again: it is let go of, so that the space bar is only ever the switch.
let down = false;
addEventListener("change", e => { if (e.target.tagName === "SELECT" || (e.target.tagName === "INPUT" && e.target.type === "number")) e.target.blur(); });
addEventListener("keydown", e => { if (e.code === "Enter" && e.target.tagName === "INPUT") e.target.blur(); });
addEventListener("click", e => { const b = e.target.closest && e.target.closest("button,input[type=checkbox],input[type=radio]"); if (b && e.detail > 0) b.blur(); });
const tell = () => { if (window.HOLD === "page") fetch("/api/hold", {method: "POST", headers: {"Content-Type": "application/json", "X-Token": TOKEN}, body: JSON.stringify({pressed: down && stateOnline && !document.hidden}), keepalive: true}).catch(() => {}); };
setInterval(tell, 50);
addEventListener("keydown", e => { if (e.code !== "Space" || window.HOLD !== "page" || typing(e)) return; e.preventDefault(); if (!stateOnline || document.hidden || (e.repeat && !down)) return; if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur(); down = true; $("spacekey").classList.add("on"); });
let upTimer = null;
function releaseHold() { clearTimeout(upTimer); upTimer = null; down = false; waiting = null; $("spacekey").classList.remove("on"); tell(); }
addEventListener("keydown", e => { if (e.code === "Space" && upTimer) { clearTimeout(upTimer); upTimer = null; } }, true);
addEventListener("keyup", e => { if (e.code !== "Space") return; if (window.HOLD === "page" && !typing(e)) e.preventDefault();
  clearTimeout(upTimer); upTimer = setTimeout(releaseHold, 40); }); addEventListener("blur", releaseHold);
document.addEventListener("visibilitychange", () => { if (document.hidden) releaseHold(); });
addEventListener("pagehide", releaseHold);
document.addEventListener("focusin", e => { if (down && typing(e)) releaseHold(); });

// ---------------------------------------------------------------- the model of the arm
// Loaded apart from everything above: if the model cannot be drawn (no WebGL, a file missing) every button still works.
import("/web/scene.js").then(m => m.start($("view"))).then(v => { viewer = v; if (last) viewer.setState(last, side()); })
  .catch(e => { $("v_fail").hidden = false; $("v_fail").textContent = "The model of the arm cannot be drawn here (" + e + "). Everything else works."; });
async function poseLoop() {  // ten times a second: the joint angles the controller publishes
  for (;;) {
    await sleep(100);
    if (!viewer || document.hidden || !stateOnline || !(last && last.connected)) continue;
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
    cap.append(nm, stt); f.append(cv, why, cap);
    f.tabIndex = 0; f.setAttribute("role", "button"); f.setAttribute("aria-keyshortcuts", "Enter");
    const expand = () => { focus = focus === id ? null : id; layoutCams(); };
    f.onclick = expand;
    f.onkeydown = e => { if (e.key === "Enter") { e.preventDefault(); expand(); } };
    t = {f, cv, nm, stt, why, at: null, note: "", got: false}; tiles.set(id, t); $("camgrid").append(f); layoutCams();
  }
  return t;
}
function describeTile(t, id) {
  const expanded = id === focus;
  t.f.setAttribute("aria-pressed", String(expanded));
  t.f.setAttribute("aria-label", `${expanded ? "Restore" : "Enlarge"} ${t.nm.textContent || "camera feed"}${t.stt.textContent ? ", " + t.stt.textContent : ""}`);
  t.f.title = expanded ? "Select to restore all feeds" : "Select or press Enter to enlarge";
}
function layoutCams() {
  if (focus && !tiles.has(focus)) focus = null;
  const g = $("camgrid"); g.classList.toggle("focus", !!focus); g.style.setProperty("--rest", Math.max(1, tiles.size - 1));
  tiles.forEach((t, id) => { t.f.classList.toggle("big", id === focus); describeTile(t, id); });
}
function dropTile(id) { const t = tiles.get(id); if (t) { t.f.remove(); tiles.delete(id); layoutCams(); } }
function lookTile(g) {  // what the block finder made of its last picture: one more tile next to the live ones
  if (!g) { dropTile("look"); sig.lookimg = null; return; }
  const t = tile("look");
  if (!t.img) { t.img = el("img"); t.cv.replaceWith(t.img); t.img.alt = "What the camera saw, with the blocks it made out drawn in"; }
  once("lookimg", g.t, () => { t.img.src = "/api/sight.jpg?t=" + g.t; });
  t.nm.textContent = "What it made out"; t.f.classList.toggle("old", !!g.stale); t.stt.textContent = g.stale ? "before the last move" : "last look"; t.why.textContent = "";
  describeTile(t, "look");
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
  const now = performance.now(); let count = 0, liveCount = 0;
  tiles.forEach((t, id) => {
    if (id === "look") return;
    const age = t.at == null ? null : (now - t.at) / 1000, live = age != null && age < 1.5 && !t.note;
    count += 1; if (live) liveCount += 1;
    t.f.classList.toggle("old", !live && t.got); t.f.classList.toggle("bad", !!t.note);
    t.stt.textContent = live ? "live" : (t.note ? (t.got ? "stopped" : "no picture") : (age == null ? "waiting…" : `${age < 10 ? age.toFixed(1) : Math.round(age)} s old`));
    t.why.textContent = !live && (t.note || !t.got) ? (t.note || "waiting for the first picture…") : "";
    describeTile(t, id);
  });
  if ($("cams-status")) $("cams-status").textContent = count ? `${liveCount} of ${count} live · select to expand` : tiles.has("look") ? "Latest detection · select to expand" : "No live feeds";
}, 300);
camLoop();
