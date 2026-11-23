"""Opt-in browser checks against a real server with only a stand-in arm.

    pip install -e ".[dev,ui]"
    playwright install chromium
    ASSEMBLY_UI_TESTS=1 pytest -q tests/test_dashboard_ui.py

ASSEMBLY_CHROMIUM can point to an existing Chromium binary. No robot addresses,
camera devices, or saved lab calibration are used by these tests.
"""
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from assemblyloop.arms import FakeArm
from assemblyloop.dashboard import make_server
from assemblyloop.feeds import Feeds
from assemblyloop.session import Session

pytestmark = pytest.mark.skipif(os.environ.get("ASSEMBLY_UI_TESTS") != "1", reason="set ASSEMBLY_UI_TESTS=1 to run browser tests")


@pytest.fixture(scope="module")
def browser():
    playwright = pytest.importorskip("playwright.sync_api")
    executable = os.environ.get("ASSEMBLY_CHROMIUM")
    with playwright.sync_playwright() as p:
        instance = p.chromium.launch(executable_path=executable, headless=True,
                                     args=["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        yield instance
        instance.close()


class NoCamera:
    def grab(self):
        raise AssertionError("browser tests must never open a physical camera")


@pytest.fixture
def dashboard(browser, tmp_path, request):
    blocks = tmp_path / "blocks.json"
    blocks.write_text((Path(__file__).parents[1] / "blocks.example.json").read_text())
    session = Session({"stand-in": dict(label="Stand-in arm (no robot)", ip=None, make=FakeArm)},
                      spots_path=tmp_path / "spots.json", log_dir=tmp_path / "logs", probe=lambda _: False,
                      camera=NoCamera(), camera_path=tmp_path / "camera.json", blocks_path=blocks,
                      fake_speed=10.0, guard_kw=dict(poll_s=0.002), limit_kw=dict(settle_s=0.0),
                      feeds=Feeds(cameras=[]) if getattr(request, "param", False) else None)
    server = make_server(session, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, color_scheme="light")
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    try:
        page.goto(f"http://127.0.0.1:{server.server_address[1]}/")
        page.wait_for_function("window.PHASE === 'OFF'")
        yield SimpleNamespace(page=page, session=session)
    finally:
        context.close()
        session.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert not errors, errors


def until(page, predicate, timeout=8, detail=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        page.wait_for_timeout(25)
    assert predicate(), detail() if detail else "condition did not become true"


def hold_space(dashboard):
    """Establish a fresh release before pressing; honor a requested release after a rendering stall."""
    page, session = dashboard.page, dashboard.session
    page.bring_to_front()

    def details():
        return {"deadman": session.dm.why(), "frames": session.dm.n,
                "page": page.evaluate("({phase: window.PHASE, hold: window.HOLD, hidden: document.hidden, focused: document.activeElement?.tagName, notice: document.getElementById('held').textContent})")}

    for _ in range(2):
        frames = session.dm.n
        page.keyboard.up("Space")
        until(page, lambda frames=frames: session.dm.n > frames and session.dm.why() == "released", detail=details)
        page.keyboard.down("Space")
        until(page, lambda: session.dm.held() or session.dm.why().startswith("release the grip"), detail=details)
        if session.dm.held():
            return
    assert session.dm.held(), details()


def teaching(dashboard):
    page = dashboard.page
    page.locator("#startbtn").click()
    page.wait_for_function("window.PHASE === 'DISARMED'")
    until(page, lambda: page.locator("#armbtn").is_enabled())
    page.locator("#armbtn").click()
    page.wait_for_function("window.PHASE === 'AWAITING CALIBRATION'")


def ready(dashboard):
    teaching(dashboard)
    demo_spots(dashboard)


def demo_spots(dashboard):
    dashboard.page.locator("details.teaching-options > summary").click()
    dashboard.page.locator("#demobtn").click()
    dashboard.page.wait_for_function("window.PHASE === 'ARMED'")


def test_setup_rehearsal_next_step_and_stop(dashboard):
    """The default task rehearses in the air and waits for both the grip and Next."""
    page, session = dashboard.page, dashboard.session
    ready(dashboard)
    assert page.locator("#rehearse").is_checked()
    assert page.locator("#step").is_checked()
    page.locator("#dst").select_option("B")
    page.locator("#count").fill("2")
    with page.expect_request("**/api/move") as request:
        page.locator("#gobtn").click()
    assert request.value.post_data_json == {"dst": "B", "count": 2, "rehearse": True, "step": True}
    page.wait_for_function("window.PHASE === 'PAUSED'")
    until(page, lambda: page.locator("#nextbtn").is_enabled())
    page.wait_for_function("last?.quest?.why === 'released' && last?.job?.waiting")
    assert page.locator("#run_status").inner_text() == "Task paused"
    hint = page.locator("#hint").inner_text()
    assert "lost" not in hint.lower() and "hold the grip" in hint and "Next step" in hint
    assert not any(action[0] == "move" for action in session.arm.log)
    assert page.locator("#run_progress").is_visible()
    page.locator("#step_details > summary").click()
    assert page.locator("#steps").is_visible()
    assert page.locator("#steps li").count() > 0
    with page.expect_request("**/api/next"):
        page.locator("#nextbtn").click()
    with page.expect_request("**/api/stop"):
        page.locator("#stop").click()
    page.wait_for_function("window.PHASE === 'STOPPED'")
    assert not session.connected
    assert page.locator("#startbtn").is_visible()


def test_keyboard_motion_hold_release_and_disarm_while_teaching(dashboard):
    page, session = dashboard.page, dashboard.session
    teaching(dashboard)
    assert page.locator("#disbtn").is_enabled()
    before = session.arm.p.copy()
    page.keyboard.press("q")
    until(page, lambda: session.worker is not None)
    assert not any(action[0] == "move" for action in session.arm.log)
    hold_space(dashboard)
    until(page, lambda: session.worker is None)
    assert session.arm.p[2] > before[2]
    page.evaluate("window.dispatchEvent(new Event('blur'))")
    until(page, lambda: not session.dm.held())
    page.keyboard.up("Space")
    page.locator("#disbtn").click()
    page.wait_for_function("window.PHASE === 'DISARMED'")
    assert not session.gate.armed


def test_task_tabs_keyboard_navigation_and_camera_source(dashboard):
    page, session = dashboard.page, dashboard.session
    ready(dashboard)
    page.locator("#tab_blocks").focus()
    page.keyboard.press("ArrowRight")
    assert page.locator("#tab_tower").get_attribute("aria-selected") == "true"
    assert page.locator("#pane_tower").is_visible()
    page.keyboard.press("End")
    assert page.locator("#tab_cam").get_attribute("aria-selected") == "true"
    assert page.locator("#pane_cam").is_visible()
    assert page.locator("details.camera-setup").get_attribute("open") is not None
    page.locator("#e_source").select_option("C")
    until(page, lambda: session.thing_at == "C")
    until(page, lambda: page.locator("#thing").input_value() == "C")
    page.locator("#tab_cam").focus()
    page.keyboard.press("Home")
    assert page.locator("#pane_blocks").is_visible()
    page.locator("#thing").select_option("A")
    until(page, lambda: page.locator("#e_source").input_value() == "A")
    page.locator("#setup-toggle").click()
    assert page.locator("#setup-toggle").get_attribute("aria-expanded") == "true"
    assert page.locator("#checks").is_visible()


@pytest.mark.parametrize(("button", "action"), [("tw_build", "build"), ("tw_down", "down"), ("tw_loop", "loop")])
def test_tower_actions_keep_every_parameter(dashboard, button, action):
    """All three tower actions carry the operator's inputs to the existing API."""
    page = dashboard.page
    ready(dashboard)
    page.locator("#tab_tower").click()
    page.locator("#tw_on").select_option("T")
    page.locator("#tw_h").fill("35.5")
    page.locator("#tw_n").fill("2")
    page.locator("#tw_r").fill("3")
    page.route("**/api/tower", lambda route: route.fulfill(json={"ok": True}))
    with page.expect_request("**/api/tower") as request:
        page.locator(f"#{button}").click()
    assert request.value.post_data_json == {"on": "T", "block_mm": 35.5, "n_now": 2, "what": action,
                                            "rounds": 3, "rehearse": True, "step": True}


def test_lost_state_releases_hold_locks_motion_and_keeps_stop(dashboard):
    page, session = dashboard.page, dashboard.session
    ready(dashboard)
    hold_space(dashboard)
    page.route("**/api/state", lambda route: route.abort("failed"))
    until(page, lambda: "NO ANSWER" in page.locator("#phase").inner_text())
    until(page, lambda: not session.dm.held())
    assert page.locator("#gobtn").is_disabled()
    assert page.locator("#tw_build").is_disabled()
    assert page.locator("#e_show").is_disabled()
    assert page.locator("#stop").is_enabled()
    with page.expect_request("**/api/stop"):
        page.locator("#stop").click()
    until(page, lambda: not session.connected)
    page.keyboard.up("Space")


def test_queued_nudge_waits_for_fresh_state_after_command_ack(dashboard):
    """An old idle snapshot must not drain a queued key before the first move finishes."""
    page = dashboard.page
    teaching(dashboard)
    state_routes, jog_routes = [], []
    page.route("**/api/state", lambda route: state_routes.append(route))
    page.route("**/api/jog", lambda route: jog_routes.append(route))
    page.keyboard.press("q")
    until(page, lambda: len(jog_routes) == 1)
    page.keyboard.press("d")
    with page.expect_response("**/api/jog"):
        jog_routes[0].continue_()
    page.wait_for_timeout(150)  # longer than the 60 ms key queue, shorter than the state timeout
    assert len(jog_routes) == 1, "a second nudge escaped before a fresh busy state"
    assert page.evaluate("window.PHASE") == "AWAITING CALIBRATION", "the queue must wait without relying on an offline reset"
    page.unroute("**/api/state")
    page.unroute("**/api/jog")
    # Let any in-flight request reach the dashboard's normal state timeout. It may
    # already have been cancelled, so continuing the stored route can handle it twice.
    page.locator("#stop").click()
    page.wait_for_function("window.PHASE === 'STOPPED'")


@pytest.mark.parametrize("width", [390, 768, 1440])
def test_responsive_controls_and_stop_remain_reachable(dashboard, width):
    page = dashboard.page
    page.set_viewport_size({"width": width, "height": 844})
    ready(dashboard)
    for tab in ("tab_blocks", "tab_tower", "tab_cam"):
        page.locator(f"#{tab}").click()
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
        stop = page.locator("#stop").bounding_box()
        assert stop and 0 <= stop["x"] < width and stop["x"] + stop["width"] <= width
    page.mouse.wheel(0, 1200)
    until(page, lambda: (page.locator("#stop").bounding_box() or {}).get("y", -1) >= 0)
    stop = page.locator("#stop").bounding_box()
    assert stop["y"] + stop["height"] <= 844


@pytest.mark.skipif(not os.environ.get("ASSEMBLY_UI_ARTIFACTS"), reason="set ASSEMBLY_UI_ARTIFACTS to capture review images")
@pytest.mark.parametrize("dashboard", [True], indirect=True)
def test_capture_dashboard_review_images(dashboard):
    """Produce reviewable renders with the real stand-in backend and its made-up feed."""
    page = dashboard.page
    output = Path(os.environ["ASSEMBLY_UI_ARTIFACTS"])
    output.mkdir(parents=True, exist_ok=True)
    page.wait_for_function("viewer !== null || !document.getElementById('v_fail').hidden")
    assert page.locator("#v_fail").is_hidden(), page.locator("#v_fail").inner_text()
    page.wait_for_function("performance.getEntriesByType('resource').filter(r => r.name.includes('/web/xarm7/') && r.name.endsWith('.stl')).length >= 15")
    page.wait_for_function("document.querySelector('#camgrid .cam canvas')?.width > 300")

    def capture(stage):
        for name, width, height, scheme in [("desktop-light", 1440, 1000, "light"), ("desktop-dark", 1440, 1000, "dark"),
                                             ("mobile-light", 390, 844, "light")]:
            page.set_viewport_size({"width": width, "height": height})
            page.emulate_media(color_scheme=scheme, reduced_motion="reduce")
            page.evaluate("window.scrollTo(0, 0); document.activeElement.blur(); document.querySelectorAll('.left, #ctl').forEach(e => e.scrollTop = 0)")
            page.wait_for_timeout(150)  # let the viewer redraw after the theme/viewport changes
            page.screenshot(path=str(output / f"{name}-{stage}.png"), full_page=True, animations="disabled")

    capture("session")
    teaching(dashboard)
    capture("teach")
    demo_spots(dashboard)
    for tab in ("blocks", "tower", "cam"):
        page.locator(f"#tab_{tab}").click()
        if tab == "tower":
            page.locator("#tw_on").select_option("T")
        capture(tab)
    page.locator("#tab_blocks").click()
    page.locator("#dst").select_option("B")
    page.locator("#gobtn").click()
    page.wait_for_function("window.PHASE === 'PAUSED'")
    capture("paused")
