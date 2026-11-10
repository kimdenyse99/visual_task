# visual_task

Visual fixation and gaze/touch tasks for marmosets, built on
[`mxbiflow`](https://github.com/PJendritza/mxbiflow). The stages accept both
touchscreen input and live gaze from MX Eye, and report what was on screen to
MX Eye so stimulus timing can be aligned with eye-camera recordings.

## Stages

Registered in `main.py` and run in this order by `mxbiflow`'s scene manager:

| Stage | Class | Purpose |
| --- | --- | --- |
| `visual_stability_training` | `VisualStabilityTraining` | Fixation point, then a single target (static image or animated marmoset GIF) at a fixed, known position. Rewards stable, engaged posture. |
| `visual_3pt_calibration` | `Visual3ptCalibration` | Fixation → randomized blank gap → target at one of three positions. Used to collect calibration data. |
| `visual_3pt_calibration_v2` | `Visual3ptCalibrationV2` | Fixation point at one of three positions; gaze dwelling within a configurable radius of it gives an immediate reward while a random marmoset GIF plays in its place. No gaze within `fixation_timeout_ms` (default 5 s) is a timeout. |
| `visual_tracing` | `VisualTracing` | Fixation → a circular target placed at a random position in the right half of the screen. |

Each stage has its own `config.json` (per-animal levels, timings, sizes,
reward durations, level-up thresholds) and `models.py` (Pydantic config and
result models).

## Project layout

```
main.py                   Entry point: setup wizard → game loop → session summary
stages/                   One package per stage (scene.py, models.py, config.json)
models.py                 Shared types: results, screen, touch events, level tables
dwell.py                  Dwell tracking: "has the input stayed inside the shape long enough"
gaze_input.py             Feeds MX Eye gaze samples into the touch/dwell logic as a synthetic touch
stimulus_events.py        Reports fixation/target onset and offset to MX Eye
trial_events.py           Per-trial CSV sidecar files (e.g. target shown) with CLOCK_REALTIME timestamps
animated_gif_sprite.py    Looping animated-GIF sprite (decoded with Pillow)
assets/                   Visual (marmoset GIFs, synthetic images) and auditory stimuli
analysis/                 Plotting helpers for trial results
scripts/                  chrony NTP server install/check scripts for clock sync with MX Eye
docs/mx-eye-events.md     How stimulus events are sent to and stored by MX Eye
tests/                    unittest test suite
```

## Requirements

- Python 3.14+
- [uv](https://docs.astral.sh/uv/)
- PortAudio system library (needed by the audio stack, e.g.
  `sudo apt install libportaudio2` on Debian/Raspberry Pi OS)
- Access to the `mxbiflow` and `mx_eye` forks referenced in
  `pyproject.toml` (installed from Git by uv)

## Setup

```bash
uv sync --locked
```

## Running

```bash
uv run python main.py
```

On the Windows lab machine, `run_visual_task.bat` does the same from the
checkout directory (edit the hard-coded path in it if your checkout lives
elsewhere).

The run starts the `mxbiflow` setup wizard, runs the session at 30 FPS, then
shows a session summary and runs post-processing. Results are written by
`mxbiflow`'s data logger in the session's data directory.

## MX Eye integration

To send stimulus events and record visits in MX Eye, enable these fields on
the `mx_eye` eyetracker entry in `mxbi.json`:

```json
{"send_experiment_events": true, "record_on_animal": true}
```

When the task and MX Eye run on separate machines, their clocks must be
synchronized. This machine acts as the NTP server:

```bash
sudo scripts/install-chrony-server.sh <mx-eye-ip-or-subnet>
```

```bash
scripts/check-chrony-server.sh
```

See [`docs/mx-eye-events.md`](docs/mx-eye-events.md) for event contents and a
manual verification checklist.

## Development

Run the tests:

```bash
uv run python -m unittest discover tests
```

Type checking and linting:

```bash
uv run pyright
```

```bash
uv run ruff check .
```
