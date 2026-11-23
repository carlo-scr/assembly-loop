"""Recording what the arm does, as data to learn from: the lab's teleoperation workflow, for the scripted cell.

The same shape as the lab's recorder (xarm7-data-collection, swoosh-collect), so that the two kinds of data can be
mixed and the lab's tools read ours:

    campaigns/                              <- gitignored; anywhere with --campaigns (an external drive, say)
      blocks-pickplace-v1/                  <- python -m assemblyloop campaign new "blocks pickplace v1"
        campaign.json                       <- name, created-at, notes
        recording_2026_11_24_14_05_12/      <- ONE EPISODE: one block taken and put down
          run.json                          <- status, the task, the outcome and its labels, provenance, validation
          raw/                              <- authoritative, every stream at its own rate, ONE monotonic clock
            arm_state.jsonl                 <- what the ROBOT reports (~30 Hz): planned and measured joints, pose, gripper
            commanded.jsonl                 <- what WE told it: every command the guard sent, and why
            events.jsonl                    <- what happened: steps, grips, pauses, the switch, cameras lent
          video/<key>.mp4 + <key>_frame_times.json   <- one per camera, the lab's names and format
        lerobot/                            <- the export: LeRobot v2.1 at 30 Hz (python -m assemblyloop export)
        upload.json                         <- what the uploader last did (python -m assemblyloop upload)

A run's status goes recording -> processing (closed) -> ready (validated, ok or not; the verdict is in run.json).
The exporter takes only runs that are ready and passed; the uploader exports what is new and sends it to a Hugging
Face dataset, one commit per batch, resuming where it stopped. Nothing here moves the arm or decides a move.

  store.py      campaigns and runs on disk
  recorder.py   one session's recording: episodes opened and closed by the job loop, the arm read ~30 Hz, the cameras
  validate.py   checks of one run, the verdict into run.json
  export.py     runs -> LeRobot v2.1, added to what was exported before
  upload.py     the export -> a (private) Hugging Face dataset, once or watching
"""
