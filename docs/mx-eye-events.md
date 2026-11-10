# Stimulus events for MX Eye

The three stages report fixation and target visibility through the framework's
`stimulus_event()` API. The shared helper checks the sprite group during drawing;
the framework timestamps the queued events immediately after the display flip.
Unchanged displays produce no repeated events. Trial termination hides any
previously reported stimulus, including on interruption.

Events include the stimulus ID (`fixation` or `target`), type, center position,
and bounding-box size in Pygame screen pixels. GIF targets also include their
filename in `metadata.asset`. Animation frames are not logged. Framework context
adds animal/RFID, visit, trial, stage, run, and screen dimensions.

In the existing experiment's `mxbi.json`, enable these fields on its configured
`mx_eye` eyetracker, preserving the current host, ports, and other settings:

```json
{"send_experiment_events": true, "record_on_animal": true}
```

Start acquisition in MX Eye before the task. The framework forwards animal
entry/identification/exit automatically and requests recording on entry/exit.
MX Eye saves received stimulus events in its event journal and, for the owned
visit recording, in `events.jsonl` beside the video. Existing local target-onset
CSV files remain separate; their pre-flip timestamps are not presentation times.

Quick manual check: show fixation, wait through its delay, show the target, then
let it disappear or interrupt the trial. Check `events.jsonl` for one onset and
offset per visible stimulus, with the expected position and GIF filename. Verify
animal entry starts recording and animal exit ends it. Separate computers need
synchronized clocks to align event and eye-camera timestamps.
