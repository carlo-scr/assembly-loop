"""Hold-to-run assembly on the lab's xArm 7: a local dashboard from which a session at the robot is started, calibrated,
armed, run and stopped, and a guard through which every motion goes. For now it moves and stacks blocks between spots
taught by hand; the same guard and step runner are meant to carry later tasks (press-fit bricks, cables).
Self-contained: numpy only; the arm's SDK is imported only when a real arm is connected.

    python -m assemblyloop dashboard      then open http://127.0.0.1:8777/

Layers, from the arm outwards:
  arms.py       the few calls made to the arm (XArm), and a stand-in with the same calls (FakeArm) for trying and tests
  deadman.py    hold-to-run: motion is allowed only while a fresh "pressed" signal keeps arriving (the Quest grip)
  guard.py      the ONLY door to the arm: limits, short segments, pause on release, stop and latch on any fault, a log
  cycle.py      a task as a list of named steps, and their execution through the guard
  spots.py      places taught by hand; moving a block between them; the levels of a tower
  plate.py      a stud plate taught from three poses (press-fit bricks; command line only for now)
  session.py    one session at the robot: OFF -> DISARMED -> AWAITING CALIBRATION -> ARMED -> RUNNING
  dashboard.py  the web page of a session (this machine only)
"""
__version__ = "0.1.0"
