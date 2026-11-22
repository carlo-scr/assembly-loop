"""The dashboard: one local web page from which a session at the robot is started, calibrated, armed, run and stopped.
Standard library only on this side, no build step, like the lab's.

    python -m assemblyloop dashboard            then open http://127.0.0.1:8777/ and choose an arm on the page

Two pages over the same buttons:
  /          the page that fits one screen: the arming sequence and the controls on the left, a model of the arm and
             the live pictures of the cameras on the right. Its files are in `web/` (the model is drawn with three.js,
             which is kept in `web/vendor/` so that the page needs no internet).
  /classic   the older page, one long column of cards (`PAGE` below). Kept as it was: if the new page misbehaves at
             the rig, this one does everything except the model and the live pictures.
  /build     the designer: lay out what the arm should build, block by block, and see at once whether every step
             stands and the gripper fits (`design.py` decides; the page draws). Needs no arm and no session.

The page starts at OFF and shows which arms answer on the network. "Start session" connects (read only) -> DISARMED.
"Arm" switches the arm on -> AWAITING CALIBRATION: the page's own buttons nudge the gripper to each place, which is
saved as a spot -> ARMED. A move then runs only while the grip is held: one block between two spots, or a tower on one
spot from the blocks on the others (and down again). Teleoperation is not used at the same time.

It listens on 127.0.0.1 only, so nothing on the network can reach it, and every button needs a token that only the
served page has, so another web page open in the same browser cannot press them either. The page can NOT make the arm
move by itself: moving also needs the grip held, and with a real arm the grip is the Quest's (the space-bar grip exists
only with the stand-in arm).
"""
from __future__ import annotations

import json
import secrets
import struct
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .arms import ArmFault, run_fake_clock  # noqa: F401  (run_fake_clock is re-exported for older callers)
from .guard import Refused, Released

WEB = Path(__file__).parent / "web"
KINDS = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".stl": "model/stl", ".txt": "text/plain; charset=utf-8"}


def web_files():
    """Every file of the page, by the name it is asked for. Only these names are ever served: nothing asked for by
    a path can reach outside `web/`."""
    return {f.relative_to(WEB).as_posix(): f for f in sorted(WEB.rglob("*")) if f.is_file() and not f.name.startswith(".")}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Assembly loop</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--ink:#16181d;--mute:#5d6470;--line:#dfe2e8;--ok:#1a7f4b;--no:#b3261e;
--off:#6b7280;--dis:#3b5b8c;--cal:#9a5b00;--armed:#c2410c;--run:#1a7f4b;--stop:#b3261e;--btn:#e9ebf0}
@media (prefers-color-scheme:dark){:root{--bg:#111318;--card:#1a1d24;--ink:#eceef2;--mute:#9aa2af;--line:#2c313b;--ok:#4ade80;
--no:#f87171;--off:#9aa2af;--dis:#8fb0e6;--cal:#fbbf24;--armed:#fb923c;--run:#4ade80;--stop:#f87171;--btn:#262b35}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:14px 20px 12px;border-bottom:1px solid var(--line);background:var(--card);position:sticky;top:0;z-index:2}
.top{display:flex;gap:16px;align-items:center;flex-wrap:wrap}h1{font-size:17px;margin:0;font-weight:600}
h2{font-size:13px;margin:0 0 10px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:var(--mute)}
h3{font-size:15px;margin:14px 0 0;font-weight:600}
#phase{font-size:22px;font-weight:700;padding:6px 16px;border-radius:999px;border:2px solid currentColor;letter-spacing:.03em}
#hint{flex:1;min-width:220px;font-size:16px;height:2.9em;overflow:hidden}
#ladder{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px;font-size:12.5px;font-weight:600;letter-spacing:.03em}
#ladder span{padding:3px 10px;border-radius:999px;border:1px solid var(--line);color:var(--mute)}#ladder span.on{color:var(--card);background:var(--ink);border-color:var(--ink)}
button{font:inherit;font-weight:600;border:1px solid var(--line);background:var(--btn);color:var(--ink);padding:9px 16px;border-radius:8px;cursor:pointer}
button:disabled{opacity:.45;cursor:not-allowed}button.x{padding:2px 9px;font-weight:400}
#stop{background:var(--stop);color:#fff;border-color:var(--stop);font-size:17px;padding:12px 26px}
.go{background:var(--armed);color:#fff;border-color:var(--armed)}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px;padding:16px 20px;max-width:1500px;margin:0 auto}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;min-width:0}
.wide{grid-column:1/-1}@media (min-width:1100px){.two{grid-column:span 2}}
dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;margin:0}dt{color:var(--mute)}dd{margin:0;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
ul{list-style:none;margin:0;padding:0}li{padding:6px 0;display:flex;gap:10px;align-items:baseline;border-bottom:1px solid var(--line)}li:last-child{border-bottom:0}
.t{width:1.2em;flex:none;font-weight:700}.y{color:var(--ok)}.n{color:var(--no)}.w{color:var(--mute)}.c{display:flex;flex-direction:column;min-width:0;flex:1}
.d,.note{color:var(--mute);font-size:13px;overflow-wrap:anywhere}.note{margin-top:8px}
.big{font-size:20px;font-weight:700}.held{color:var(--run)}.rel{color:var(--mute)}
label{display:flex;flex-direction:column;gap:3px;font-size:13px;color:var(--mute)}label.chk,label.pick{flex-direction:row;align-items:center;gap:8px;font-size:15px;color:var(--ink)}
input[type=text],input[type=number],select{font:inherit;padding:7px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink);min-width:0}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end;margin:8px 0}.row label{flex:1;min-width:110px}.row label.chk{flex:none}
ol{margin:10px 0 0;padding-left:22px;columns:2;column-gap:28px}ol li{display:list-item;border:0;padding:2px 0;color:var(--mute)}ol li.cur{color:var(--ink);font-weight:700}ol li.past{color:var(--ok)}
.pad{display:grid;grid-template-columns:repeat(3,minmax(84px,1fr)) minmax(84px,1fr);gap:6px;max-width:520px;margin:8px 0}.pad button:nth-child(4n){margin-left:18px}.pad button{padding:12px 6px}.pad i,button[data-t] i{font-style:normal;color:var(--mute);font-weight:400;margin-left:6px}
pre{margin:0;font:12.5px/1.5 ui-monospace,Menlo,monospace;white-space:pre-wrap;max-height:230px;overflow:auto;color:var(--mute)}
#e_img{width:100%;border-radius:8px;margin-top:10px;border:1px solid var(--line)}
#msg{color:var(--no);height:1.45em;font-weight:600;margin-top:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}[hidden]{display:none!important}
@media (max-width:700px){ol{columns:1}}
</style></head><body>
<header><div class="top"><h1>Assembly loop</h1><div id="phase">…</div><div id="hint"></div>
<button id="stop" title="Stops the arm and ends this session">STOP SESSION</button></div>
<div id="ladder"><span data-p="OFF">OFF</span><span data-p="DISARMED">DISARMED</span><span data-p="AWAITING CALIBRATION">AWAITING CALIBRATION</span><span data-p="ARMED">ARMED</span><span data-p="RUNNING">RUNNING</span></div>
<div id="msg"></div></header>
<main>
<section><h2>Session</h2>
<div id="s_off"><ul id="arms"></ul><label class="chk" id="questrow" hidden><input type="checkbox" id="usequest"> use the Quest grip instead of the space bar</label>
<div class="row"><button id="startbtn" class="go">Start session</button></div><div class="note" id="ended"></div></div>
<div id="s_on" hidden><dl id="sess"></dl><div class="row"><button id="endbtn">End session</button></div></div></section>

<section><h2>Hold-to-run</h2><div id="held" class="big rel">starts with the session</div><dl id="quest" style="margin-top:8px"></dl><div class="note" id="holdnote"></div></section>

<section><h2>Before arming</h2><div id="k_none" class="note">Starts with the session.</div><ul id="checks"></ul>
<div class="row"><button id="armbtn" class="go">Arm</button><button id="disbtn">Disarm</button></div>
<div class="note">Arming moves nothing. The arm moves only during a move, and only while the grip is held.</div></section>

<section><h2>Arm</h2><dl id="arm"></dl></section>

<section class="two"><h2>Show it the places</h2>
<div id="c_none" class="note">Starts with the session.</div>
<div id="c_dis" class="note" hidden>Arm first. Then you nudge the gripper to each place from here, in small steps, while holding the grip.</div>
<div id="c_teach" hidden>
<div class="pad"><span></span><button data-j="fwd">Forward<i>W</i></button><span></span><button data-j="up">Up<i>Q</i></button>
<button data-j="left">Left<i>A</i></button><span></span><button data-j="right">Right<i>D</i></button><span></span>
<span></span><button data-j="back">Back<i>S</i></button><span></span><button data-j="down">Down<i>E</i></button></div>
<div class="row"><label>Step<select id="jogmm"><option value="1">1 mm</option><option value="5" selected>5 mm</option><option value="20">20 mm</option><option value="50">50 mm</option></select></label>
<button id="gopen">Open gripper</button><button id="gclose">Close gripper</button></div>
<div class="row"><label>Turn by<select id="jogdeg"><option value="5">5°</option><option value="15" selected>15°</option><option value="45">45°</option><option value="90">90°</option></select></label>
<button data-t="ccw">Turn ⟲<i>Z</i></button><button data-t="cw">Turn ⟳<i>C</i></button></div>
<div class="row"><button id="levelbtn">Point the gripper down</button><span class="note" id="tiltnote"></span></div>
<div class="note" id="jognote"></div>
<div class="note">Forward is away from the robot's base. Up and down follow the gripper's own axis, so it should point at the table. Turning spins the gripper in place (only its last joint moves). Every nudge is slow and ends by itself if something is in the way.</div>
<h3>Find the table by itself</h3>
<div class="row"><label>Area across (mm)<input type="number" id="ft_w" value="200" min="100" max="400" step="10"></label><label>Area deep (mm)<input type="number" id="ft_d" value="140" min="100" max="400" step="10"></label><button id="ftbtn" class="go">Find the table</button><button id="fthere">The gripper is at the table now</button></div>
<div class="note">Instead of teaching spots by hand. Bring the gripper over the middle of the area the blocks will lie in, pointing straight down. Hold a green or blue bar between the open jaws by its ends, lying level, the jaws across its short side, and press Close gripper. Then press Find the table and hold the grip: the gripper comes straight down slowly until the bar meets the mat, lets go, and the label reads ARMED. Let go of the grip if it visibly pushes into the table. The tower will stand where the bar was put down. If you have already brought the fingertips down onto the mat yourself, press "The gripper is at the table now" instead: nothing comes down, the bar is let go there.</div>
<h3>Or teach spots by hand</h3>
<div class="row"><label>Name of this spot<input type="text" id="spotname" value="A" size="6" maxlength="12"></label><button id="spotbtn" class="go">Save spot here</button></div>
<ul id="spots_edit"></ul>
<div class="row"><label>Lift between spots (mm)<input type="number" id="lift" value="80" min="30" max="200" step="5"></label><button id="donebtn" class="go">Done</button>
<button id="savedbtn" hidden></button><button id="demobtn" hidden>Use made-up spots</button></div></div>
<div id="c_view" hidden><div id="c_status"></div><ul id="spots_view"></ul><div class="row"><button id="calbtn">Teach again</button></div></div></section>

<section class="two"><h2>Move blocks</h2>
<div class="row"><label class="chk"><input type="checkbox" id="rehearse" checked> rehearse in the air (does not come down)</label>
<label class="chk"><input type="checkbox" id="step" checked> one step at a time</label><button id="nextbtn">Next step</button></div>
<h3>One block</h3>
<div class="row"><label>The block stands on<select id="thing"></select></label><label>Put it on<select id="dst"></select></label>
<label>How many moves<input type="number" id="count" value="1" min="1" max="500"></label><button id="gobtn" class="go">Go</button></div>
<h3>A tower</h3>
<div class="row"><label>Tower stands on<select id="tw_on"></select></label><label>Block height (mm)<input type="number" id="tw_h" value="30" min="10" max="80" step="0.5"></label>
<label>Blocks on it now<input type="number" id="tw_n" value="0" min="0" max="6"></label><label>Rounds<input type="number" id="tw_r" value="1" min="1" max="50"></label></div>
<div class="row"><button id="tw_build" class="go">Build</button><button id="tw_down">Take down</button><button id="tw_loop">Build and take down, for the rounds</button></div>
<div class="note" id="tw_note"></div>
<div id="outcome" class="note"></div><ol id="steps"></ol></section>

<section class="two"><h2>Camera</h2>
<div class="note" id="e_why"></div>
<div class="row"><button id="lookbtn">Look</button><label>Tower stands on<select id="e_on"></select></label><button id="e_build" class="go">Build a tower from what it sees</button></div>
<div class="note" id="e_plan"></div>
<img id="e_img" alt="What the camera saw, with the blocks it made out drawn in" hidden>
<ul id="e_seen"></ul>
<h3>Show the camera the table</h3>
<div class="row"><label>With this block<select id="e_kind"></select></label><label>Places<input type="number" id="e_n" value="12" min="8" max="20"></label><button id="e_show">Show the camera the table</button></div>
<div class="note">Once, and again whenever the camera or the table was moved. Teach spots at opposite corners of the area the blocks will lie in. Lay ONE bar or slab flat on a spot, jaws across its short side, say under "The block stands on" which spot that is, and keep everything else off the mat. The arm carries the block to each place and steps aside for a picture; hold the grip throughout. Blocks must lie on the grey mat and must not touch each other.</div></section>

<section class="wide"><h2>Log</h2><pre id="log"></pre></section>
</main>
<script>
const TOKEN="__TOKEN__", $=id=>document.getElementById(id);
const COL={"OFF":"--off","STOPPED":"--stop","DISARMED":"--dis","AWAITING CALIBRATION":"--cal","ARMED":"--armed","RUNNING":"--run","PAUSED":"--cal"};
async function post(path, body){ $("msg").textContent="";
  try{ const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json","X-Token":TOKEN},body:JSON.stringify(body||{})});
       const j=await r.json(); if(j.error) $("msg").textContent=j.error; tick(); return j; }catch(e){ $("msg").textContent="no answer from the program: "+e; } }
function dl(el, rows){ el.replaceChildren(...rows.flatMap(([k,v])=>{const a=document.createElement("dt"),b=document.createElement("dd");a.textContent=k;b.textContent=v;return [a,b];})); }
function li(mark, cls, text, detail, btn){ const e=document.createElement("li"),m=document.createElement("span"),c=document.createElement("span"),t=document.createElement("span");
  m.className="t "+cls; m.textContent=mark; c.className="c"; t.textContent=text; c.append(t);
  if(detail){const d=document.createElement("span"); d.className="d"; d.textContent=detail; c.append(d);} e.append(m,c); if(btn) e.append(btn); return e; }
const f1=v=>v==null?"…":Number(v).toFixed(1), show=(id,on)=>{ $(id).hidden=!on; };
let sig={}, lastThing, lastN;
function once(key, value, fn){ const v=JSON.stringify(value); if(sig[key]!==v){ sig[key]=v; fn(); } }
function options(sel, names, first){ const keep=sel.value; sel.replaceChildren(...first.concat(names.map(n=>[n,n])).map(([v,t])=>{const o=document.createElement("option");o.value=v;o.textContent=t;return o;})); if([...sel.options].some(o=>o.value===keep)) sel.value=keep; }
function render(s){
  const p=$("phase"); p.textContent=s.phase+(s.fake?" · stand-in arm":""); p.style.color=`var(${COL[s.phase]})`;
  $("hint").textContent=s.hint;
  const rung=s.phase==="PAUSED"?"RUNNING":(s.phase==="STOPPED"?"OFF":s.phase);
  document.querySelectorAll("#ladder span").forEach(e=>{ e.className=e.dataset.p===rung?"on":""; e.textContent=(e.dataset.p==="RUNNING"&&s.phase==="PAUSED")?"PAUSED":(e.dataset.p==="OFF"&&s.phase==="STOPPED"?"STOPPED":e.dataset.p); });
  $("stop").disabled=!s.connected;
  // session
  show("s_off",!s.connected); show("s_on",s.connected);
  once("arms", s.arms, ()=>{ const keep=(document.querySelector("input[name=arm]:checked")||{}).value;
    $("arms").replaceChildren(...s.arms.map(a=>{ const l=document.createElement("label"), i=document.createElement("input"), c=document.createElement("span"), t=document.createElement("span"), d=document.createElement("span");
      l.className="pick"; i.type="radio"; i.name="arm"; i.value=a.key; i.checked=(keep? keep===a.key : a.key==="stand-in"); c.className="c"; t.textContent=a.label;
      d.className="d"; d.textContent=a.ip? (a.answers==null?"checking…":(a.answers?"answers at "+a.ip+" (powered)":"no answer at "+a.ip+" (off or unplugged)")):"for trying things without a robot";
      c.append(t,d); l.append(i,c); const e=document.createElement("li"); e.append(l); i.onchange=pickArm; return e; })); pickArm(); });
  $("ended").textContent=s.phase==="STOPPED"?"The last session was stopped: "+s.ended+". Look at the arm before starting again.":"";
  if(s.connected){ const a=s.arms.find(x=>x.key===s.chosen)||{};
    dl($("sess"),[["Arm",a.label+(a.ip?" at "+a.ip:"")],["Connection",s.phase==="DISARMED"?"connected, read only":"connected, switched on"]]); }
  // hold-to-run
  const q=s.quest, h=$("held");
  if(q){ h.textContent=q.held?"GRIP HELD":(q.why==="released"?"grip released":(q.why==="held"?"grip held, not armed":q.why)); h.className="big "+(q.held?"held":"rel");
    dl($("quest"),[["Switch",s.hold==="page"?"space bar on this page":"Quest controller grip"],["Signal",q.frames? `${q.hz} per second, longest gap ${q.gap_ms} ms`:"none yet"]].concat(s.hold==="page"?[]:[["Grip value",q.grip==null?"…":Number(q.grip).toFixed(2)]]));
    $("holdnote").textContent=s.hold==="page"?(s.fake?"Hold the space bar to stand in for the grip.":"Hold the SPACE BAR to let the arm move; let go to pause it. It counts only while this page is in front and has the keyboard. With the space bar held: F closes the gripper, R opens it. The red emergency stop stays in reach."):"Squeeze past halfway to hold. The red emergency stop stays in reach.";
  } else { h.textContent="starts with the session"; h.className="big rel"; $("quest").replaceChildren(); $("holdnote").textContent=""; }
  // calibration
  const c=s.calibration, teachOn=s.phase==="AWAITING CALIBRATION";
  show("c_none",!s.connected); show("c_dis",s.connected&&s.phase==="DISARMED"); show("c_teach",teachOn); show("c_view",s.connected&&c.done&&!teachOn);
  document.querySelectorAll("#c_teach button").forEach(b=>{ b.disabled=c.busy; });
  $("jognote").textContent=c.busy?"moving while you hold the grip…":(c.note||"");
  const tilt=(s.arm||{}).tilt_deg==null? null : s.arm.tilt_deg;
  $("levelbtn").disabled=c.busy||tilt==null||tilt<0.5||c.spots.length>0;
  $("tiltnote").textContent=tilt==null?"":(tilt<0.5?"The gripper points straight down.":`The gripper is tilted ${tilt.toFixed(0)}° from straight down. Tilting keeps the flange in place and swings the fingertips sideways: keep the space around the gripper free.`+(c.spots.length?" Remove the saved spots first.":""));
  $("c_status").textContent=c.done? `Known: ${c.spots.length} spots, lift ${c.lift_mm} mm.`:"";
  $("spots_view").replaceChildren(...(c.done&&!teachOn?c.spots:[]).map(x=>li("•","w","Spot "+x.name,"")));
  $("calbtn").disabled=s.phase!=="ARMED";
  const sv=$("savedbtn"); show("savedbtn",!!c.saved&&teachOn); if(c.saved) sv.textContent=`Use saved spots (${c.saved.names.join(", ")}; ${c.saved.when})`;
  show("demobtn",s.fake&&teachOn);
  once("edit", teachOn?c.spots:null, ()=>{ $("spots_edit").replaceChildren(...(teachOn?c.spots:[]).map(x=>{ const b=document.createElement("button"); b.className="x"; b.textContent="remove"; b.onclick=()=>post("/api/spot",{name:x.name,remove:true});
      return li("✓","y","Spot "+x.name,x.mm.join(", ")+" mm, turned "+x.yaw+"°",b); }));
    if(teachOn){ const used=c.spots.map(x=>x.name), nxt="ABCDEFGHIJKLMNOP".split("").find(l=>!used.includes(l)); if(nxt && (used.includes($("spotname").value)||!$("spotname").value)) $("spotname").value=nxt; } });
  once("lift", teachOn?c.lift_mm:null, ()=>{ if(teachOn && document.activeElement!==$("lift")) $("lift").value=c.lift_mm; });
  // before arming
  show("k_none",!s.connected); $("checks").replaceChildren(...(s.checks||[]).map(k=>li(k.ok?"✓":"✗",k.ok?"y":"n",k.text,k.ok?"":k.detail)));
  const ready=(s.checks||[]).length>0&&s.checks.every(k=>k.ok); $("armbtn").disabled=!(s.phase==="DISARMED"&&ready); $("disbtn").disabled=!["ARMED","RUNNING","PAUSED"].includes(s.phase);
  // move
  const names=c.done?c.spots.map(x=>x.name):[];
  once("names", names, ()=>{ options($("thing"),names,[["","— say where —"]]); options($("dst"),names,[["","a spot picked at random"]]); options($("tw_on"),names,[["","— which spot —"]]); options($("e_on"),names,[["","— which spot —"]]); lastThing=undefined; });
  const st=s.stack, nNow=st?st.n:undefined; if(nNow!==lastN){ lastN=nNow; if(st){ $("tw_n").value=st.n==null?"":st.n; $("tw_on").value=st.on; } }
  const on=$("tw_on").value, feeds=names.filter(n=>n!==on).slice(0,6); ["tw_build","tw_down","tw_loop"].forEach(id=>{ $(id).disabled=s.phase!=="ARMED"; });
  $("tw_note").textContent=!c.done?"":(!on?"Choose the spot the tower stands on. One block stands on every other spot.":`One block stands on each of ${feeds.join(", ")}. They go onto ${on} in that order: the first at the taught height, each next one a block height higher, let go ${s.drop_mm} mm above the one below. Taking down goes top first, every block back to its own spot.`+(st&&st.n==null?" The last run did not end as planned: say how many blocks stand on the tower now.":""));
  if(s.thing_at!==lastThing){ lastThing=s.thing_at; $("thing").value=s.thing_at||""; }
  const j=s.job, busy=["RUNNING","PAUSED"].includes(s.phase); $("gobtn").disabled=s.phase!=="ARMED"; $("nextbtn").disabled=!(j&&j.waiting&&busy);
  $("outcome").textContent=!j?"":(busy? `Move ${Math.min(j.done+1,j.total)} of ${j.total}: ${j.text||(j.src+" to "+j.dst)}`+(j.rehearse?" (rehearsal)":"")+(j.waiting?" · waiting for Next step":"") : `Last: ${j.done} of ${j.total} done. ${j.outcome||""}`);
  $("steps").replaceChildren(...(j&&busy?j.steps:[]).map((n,i)=>{const e=document.createElement("li"); e.textContent=n; if(j.i!=null){ if(i<j.i)e.className="past"; if(i===j.i)e.className="cur"; } return e;}));
  // camera
  const e=s.eye||{}, g=e.sight;
  $("e_why").textContent=e.why? e.why.charAt(0).toUpperCase()+e.why.slice(1)+"." : (e.shown? `The camera was shown the table on ${e.shown} and found the block within ${e.fit_mm} mm.`:"");
  once("bars", e.bars||[], ()=>options($("e_kind"),e.bars||[],[]));
  $("lookbtn").disabled=!e.shown||busy; $("e_show").disabled=s.phase!=="ARMED"||!(e.bars||[]).length;
  $("e_build").disabled=!(s.phase==="ARMED"&&e.ready&&g&&!g.stale&&g.order.length);
  once("img", g?g.t:null, ()=>{ show("e_img",!!g); if(g) $("e_img").src="/api/sight.jpg?t="+g.t; });
  once("seen", g?[g.t,g.stale]:null, ()=>{ $("e_seen").replaceChildren(...(g? g.seen.map(b=>li("✓","y",b.name,`${b.lies}, outline fits ${b.fit}%`)).concat(g.unknown.map(w=>li("?","n","Not made out",w))) : [])); });
  $("e_plan").textContent=!g?"":((g.stale?"This picture is from before the last move: press Look again. ":"")+(g.order.length? `Tower, bottom first: ${g.order.join(", ")} (${g.height_mm} mm high).`:"Nothing in this picture can go onto a tower.")+(g.left_out.length?" Left out: "+g.left_out.join("; ")+".":""));
  // arm
  const a=s.arm; dl($("arm"), a? [["Controller error",a.error==null?"…":(a.error? "ERROR "+a.error:"none")],["Height above the lowest spot",a.height_mm==null?"known once calibrated":f1(a.height_mm)+" mm"],
    ["Measured against planned",a.lag_mm==null?"…":f1(a.lag_mm)+" mm apart"],["Moving",a.moving?"yes":"no"]].concat(a.read_error?[["Read problem",a.read_error]]:[]) : [["Status","no session"]]);
  const lg=$("log"), atEnd=lg.scrollTop+lg.clientHeight>=lg.scrollHeight-8; lg.textContent=s.log.join("\n"); if(atEnd) lg.scrollTop=lg.scrollHeight;
  window.HOLD=s.connected?s.hold:null; window.PHASE=s.phase; window.KEYHOLD=!!s.key_hold; window.BUSY=!!c.busy;
}
function pickArm(){ const k=(document.querySelector("input[name=arm]:checked")||{}).value; show("questrow",k==="stand-in"||window.KEYHOLD); }
async function tick(){ try{ render(await (await fetch("/api/state")).json()); }catch(e){ $("phase").textContent="NO ANSWER"; $("hint").textContent="The program is not answering. Assume nothing about the arm."; } }
setInterval(tick,250); tick();
$("stop").onclick=()=>post("/api/stop");
$("startbtn").onclick=()=>{ const k=(document.querySelector("input[name=arm]:checked")||{}).value; post("/api/session",{start:true,arm:k,hold:(k==="stand-in"||window.KEYHOLD)&&!$("usequest").checked?"page":"quest"}); };
$("endbtn").onclick=()=>post("/api/session",{start:false});
$("calbtn").onclick=()=>post("/api/calibrate",{do:"begin"}); $("savedbtn").onclick=()=>post("/api/calibrate",{do:"saved"}); $("demobtn").onclick=()=>post("/api/calibrate",{do:"demo"});
$("donebtn").onclick=()=>post("/api/calibrate",{do:"done"});
let waiting=null;
const jog=dir=>{ if(window.BUSY){ waiting=()=>jog(dir); return; } window.BUSY=true; return post("/api/jog",{dir:dir,mm:+$("jogmm").value}); };
setInterval(()=>{ if(waiting&&!window.BUSY&&window.PHASE==="AWAITING CALIBRATION"){ const w=waiting; waiting=null; w(); } },60);
document.querySelectorAll("button[data-j]").forEach(b=>{ b.onclick=()=>jog(b.dataset.j); });
const turn=dir=>{ if(window.BUSY){ waiting=()=>turn(dir); return; } window.BUSY=true; return post("/api/turn",{dir:dir,deg:+$("jogdeg").value}); };
document.querySelectorAll("button[data-t]").forEach(b=>{ b.onclick=()=>turn(b.dataset.t); });
$("levelbtn").onclick=()=>post("/api/level"); $("ftbtn").onclick=()=>post("/api/find_table",{wide:+$("ft_w").value,deep:+$("ft_d").value}); $("fthere").onclick=()=>post("/api/find_table",{wide:+$("ft_w").value,deep:+$("ft_d").value,here:true}); $("gopen").onclick=()=>post("/api/grip",{close:false}); $("gclose").onclick=()=>post("/api/grip",{close:true});
// the lab's own nudge keys: W/S away from and towards the base, A/D left and right, Q/E up and down
const KEYS={KeyW:"fwd",KeyS:"back",KeyA:"left",KeyD:"right",KeyQ:"up",KeyE:"down"};
const TURN={KeyZ:"ccw",KeyC:"cw"};
addEventListener("keydown",e=>{ if(!(KEYS[e.code]||TURN[e.code])||window.PHASE!=="AWAITING CALIBRATION"||e.repeat||e.metaKey||e.ctrlKey||e.target.tagName==="TEXTAREA"||(e.target.tagName==="INPUT"&&["text","number"].includes(e.target.type))) return; e.preventDefault(); KEYS[e.code]? jog(KEYS[e.code]) : turn(TURN[e.code]); });
$("spotbtn").onclick=()=>post("/api/spot",{name:$("spotname").value}); $("lift").onchange=()=>post("/api/lift",{mm:+$("lift").value});
$("armbtn").onclick=()=>post("/api/arm",{on:true}); $("disbtn").onclick=()=>post("/api/arm",{on:false});
$("thing").onchange=()=>{ if($("thing").value) post("/api/thing",{spot:$("thing").value}); };
$("gobtn").onclick=()=>post("/api/move",{dst:$("dst").value||null,count:+$("count").value,rehearse:$("rehearse").checked,step:$("step").checked});
$("nextbtn").onclick=()=>post("/api/next");
const tower=what=>post("/api/tower",{on:$("tw_on").value,block_mm:+$("tw_h").value,n_now:$("tw_n").value===""?null:+$("tw_n").value,what:what,rounds:+$("tw_r").value,rehearse:$("rehearse").checked,step:$("step").checked});
$("lookbtn").onclick=()=>post("/api/look"); $("e_show").onclick=()=>post("/api/show_table",{kind:$("e_kind").value,n:+$("e_n").value,step:$("step").checked});
$("e_build").onclick=()=>post("/api/build_seen",{on:$("e_on").value,rehearse:$("rehearse").checked,step:$("step").checked});
$("tw_build").onclick=()=>tower("build"); $("tw_down").onclick=()=>tower("down"); $("tw_loop").onclick=()=>tower("loop");
// With the stand-in arm the space bar stands in for the grip. Like the headset, the page reports its state ALL the time
// (pressed or not), so that a closed or frozen page reads as "no signal", not as "released and fine".
// A button or a tick box that was just clicked keeps the keyboard, and the space bar would press it again: it is let go of, so that the space bar is only ever the switch.
let down=false; const typing=e=>e.target.tagName==="TEXTAREA"||(e.target.tagName==="INPUT"&&["text","number"].includes(e.target.type));
addEventListener("change",e=>{ if(e.target.tagName==="SELECT"||(e.target.tagName==="INPUT"&&e.target.type==="number")) e.target.blur(); });
addEventListener("keydown",e=>{ if(e.code==="Enter"&&e.target.tagName==="INPUT") e.target.blur(); });
addEventListener("click",e=>{ const b=e.target.closest&&e.target.closest("button,input[type=checkbox],input[type=radio]"); if(b) b.blur(); });
const tell=()=>{ if(window.HOLD==="page") fetch("/api/hold",{method:"POST",headers:{"Content-Type":"application/json","X-Token":TOKEN},body:JSON.stringify({pressed:down}),keepalive:true}).catch(()=>{}); };
setInterval(tell,50);
addEventListener("keydown",e=>{ if(e.code!=="Space"||window.HOLD!=="page"||typing(e)) return; e.preventDefault(); if(document.activeElement&&document.activeElement!==document.body) document.activeElement.blur(); down=true; });
addEventListener("keydown",e=>{ if(!["KeyF","KeyR"].includes(e.code)||window.PHASE!=="AWAITING CALIBRATION"||e.repeat||e.metaKey||e.ctrlKey||typing(e)) return; e.preventDefault(); post("/api/grip",{close:e.code==="KeyF"}); });
let upTimer=null;
addEventListener("keydown",e=>{ if(e.code==="Space"&&upTimer){ clearTimeout(upTimer); upTimer=null; } },true);
addEventListener("keyup",e=>{ if(e.code!=="Space") return; if(window.HOLD==="page"&&!typing(e)) e.preventDefault();
  clearTimeout(upTimer); upTimer=setTimeout(()=>{ upTimer=null; down=false; tell(); },40); }); addEventListener("blur",()=>{ down=false; tell(); });
</script></body></html>
"""


def make_server(session, port=8777):
    """The HTTP server for a session (not started). `server.token` is what the page's buttons must send."""
    token, files = secrets.token_urlsafe(18), web_files()

    def act(path, b):
        if path == "/api/stop":
            session.stop()
        elif path == "/api/session":
            session.start(str(b.get("arm")), b.get("hold")) if b.get("start") else session.end()
        elif path == "/api/calibrate":
            dict(begin=session.calibrate_begin, done=session.calibrate_done, saved=session.use_saved, demo=session.use_demo)[str(b.get("do"))]()
        elif path == "/api/jog":
            session.jog(str(b.get("dir")), b.get("mm", 5))
        elif path == "/api/turn":
            session.turn(str(b.get("dir")), b.get("deg", 15))
        elif path == "/api/level":
            session.level()
        elif path == "/api/find_table":
            session.find_table(b.get("wide", 200), b.get("deep", 140), bool(b.get("here")))
        elif path == "/api/grip":
            session.grip(bool(b.get("close")))
        elif path == "/api/spot":
            session.remove_spot(str(b.get("name"))) if b.get("remove") else session.save_spot(b.get("name"))
        elif path == "/api/lift":
            session.set_lift(b.get("mm"))
        elif path == "/api/arm":
            session.arm_on() if b.get("on") else session.disarm()
        elif path == "/api/thing":
            session.set_thing(str(b.get("spot")))
        elif path == "/api/move":
            session.move(b.get("dst") or None, b.get("count", 1), bool(b.get("rehearse")), bool(b.get("step")))
        elif path == "/api/tower":
            session.tower(str(b.get("on")), b.get("block_mm"), str(b.get("what", "build")), b.get("rounds", 1), b.get("n_now"), bool(b.get("rehearse")), bool(b.get("step")))
        elif path == "/api/next":
            session.next_step()
        elif path == "/api/look":
            session.look()
        elif path == "/api/show_table":
            session.show_table(str(b.get("kind")), b.get("n", 12), bool(b.get("step")))
        elif path == "/api/build_seen":
            session.build_seen(str(b.get("on")), bool(b.get("rehearse")), bool(b.get("step")), bool(b.get("check", True)))
        elif path == "/api/build_design":
            session.build_design(str(b.get("name")), str(b.get("on")), bool(b.get("rehearse")), bool(b.get("step")), bool(b.get("check", True)))
        elif path == "/api/hold":
            if session.hold != "page":
                raise Refused("with this arm the only hold-to-run switch is the Quest grip")
            session.dm.feed(bool(b.get("pressed")))
        else:
            return False
        return True

    def ask(path, b):
        """Buttons that answer with something (the designer's). None: not one of them."""
        d = b.get("design")
        if path in ("/api/design/check", "/api/design/save", "/api/design/order") and not isinstance(d, dict):
            raise ValueError("send the design as an object")
        if path == "/api/design/check":
            return session.design_check(d)
        if path == "/api/design/save":
            return session.design_save(d)
        if path == "/api/design/order":
            return dict(design=session.design_order(d))
        return None

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # connections are kept open: the page reports its key twenty times a second

        def log_message(self, *a):  # quiet: the page polls four times a second
            pass

        def _send(self, body, code=HTTPStatus.OK, kind="application/json", keep=False):
            data = body if isinstance(body, bytes) else json.dumps(body, default=float).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=86400" if keep else "no-store")  # kept by the browser: three.js and the arm's meshes, which never change
            self.end_headers()
            self.wfile.write(data)

        def _token(self):
            return secrets.compare_digest(self.headers.get("X-Token") or "", token)

        def _local(self):
            return (self.headers.get("Host") or "").split(":")[0] in ("127.0.0.1", "localhost")

        def do_GET(self):
            if not self._local():
                return self._send(dict(error="this page answers on 127.0.0.1 only"), HTTPStatus.FORBIDDEN)
            url = urlsplit(self.path)
            if url.path in ("/", "/build"):
                page = WEB / ("index.html" if url.path == "/" else "build.html")
                return self._send(page.read_text().replace("__TOKEN__", token).encode(), kind="text/html; charset=utf-8")
            try:
                if url.path == "/api/designs":
                    return self._send(dict(designs=session.design_list()))
                if url.path == "/api/design":
                    return self._send(dict(design=session.design_get(parse_qs(url.query).get("name", [""])[0])))
                if url.path == "/api/design/catalogue":
                    return self._send(session.design_catalogue())
            except (Refused, ValueError, KeyError, TypeError) as e:
                return self._send(dict(error=f"{e}"), HTTPStatus.CONFLICT)
            if url.path == "/classic":
                return self._send(PAGE.replace("__TOKEN__", token).encode(), kind="text/html; charset=utf-8")
            if url.path.startswith("/web/"):
                f = files.get(url.path[5:])
                if f is None or f.suffix == ".html":  # the pages themselves only with their token, at / and /build
                    return self._send(dict(error="not found"), HTTPStatus.NOT_FOUND)
                return self._send(f.read_bytes(), kind=KINDS.get(f.suffix, "application/octet-stream"), keep=f.parent.name in ("vendor", "xarm7"))
            if url.path == "/api/state":
                return self._send(session.state())
            if url.path == "/api/view":
                return self._send(session.view())
            if url.path == "/api/frames":
                if not self._token():  # pictures of the lab: for the dashboard's own page only
                    return self._send(dict(error="refused: not asked by the dashboard's own page"), HTTPStatus.FORBIDDEN)
                q = parse_qs(url.query)
                try:
                    after = int(q.get("after", ["0"])[0])
                except ValueError:
                    after = 0
                head, pictures = session.frames(after, q.get("side", [None])[0])
                head = json.dumps(head, default=float).encode()
                return self._send(struct.pack(">I", len(head)) + head + b"".join(pictures), kind="application/octet-stream")
            if url.path == "/api/sight.jpg" and session.sight_jpg():
                return self._send(session.sight_jpg(), kind="image/jpeg")
            return self._send(dict(error="not found"), HTTPStatus.NOT_FOUND)

        def do_POST(self):
            try:  # the body is read whatever happens next: the connection stays open for the next request
                raw = self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 1 << 20))
            except ValueError:
                raw = b""
            if not self._local() or not self._token():
                return self._send(dict(error="refused: this button was not pressed on the dashboard's own page"), HTTPStatus.FORBIDDEN)
            try:
                body = json.loads(raw or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("the request is not an object")
                got = ask(self.path, body)
                if got is not None:
                    return self._send(got)
                if not act(self.path, body):
                    return self._send(dict(error="not found"), HTTPStatus.NOT_FOUND)
            except (Refused, Released, ArmFault, KeyError, ValueError, TypeError) as e:
                return self._send(dict(error=f"{e}"), HTTPStatus.CONFLICT)
            return self._send(dict(ok=True))

    class Server(ThreadingHTTPServer):
        request_queue_size, daemon_threads = 128, True  # the default of 5 waiting connections drops some when the page is busy

    server = Server(("127.0.0.1", int(port)), Handler)
    server.token = token
    return server
