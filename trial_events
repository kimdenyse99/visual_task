"""Per-trial sidecar event files, for marking a moment on a video timeline
after the fact.

This is deliberately NOT written into the existing per-session
result.jsonl/.json files (which accumulate every trial's full result in
one file) - each event is its own small, separate CSV file per trial,
written at the exact moment the event actually happens, using the same
time.time_ns() / CLOCK_REALTIME clock base mx-eye's own recordings use.
See AGENTS.md's "Clock synchronization" and "Delay readouts" sections in
the mx_eye repo: every timestamp in this whole system is a raw
CLOCK_REALTIME value, directly comparable across machines only because
chrony keeps their system clocks aligned - no application-level offset is
computed or applied here either.

The file is intentionally minimal and separate from mxbiflow's own
DataLogger-based per-session results, so that joining it against a video
or a tracking.csv later is a simple, self-contained read - one CSV file,
one header row, one data row, directly openable in a spreadsheet.

Which event to log, and why it matters for calibration
---------------------------------------------------------
"Gaze registered correct" and "reward dispensed" are both *gated on an
existing, pre-calibration guess* at what counts as "on target" - for a
stage whose fixation phase rewards ANY touch/gaze position (e.g.
visual_stability_training's fixation phase: "the goal here is encouraging
a stable, engaged posture, not precise fixation-point targeting"),
neither of those events actually tells you where the animal was looking
when there's no calibration yet to judge that in the first place.

"Target shown" (write_target_shown_event) has no such problem: it fires
purely on stimulus presentation, independent of any gaze judgment. For a
stage with ONE FIXED, KNOWN target position (e.g.
visual_stability_training), this is the event to use for from-scratch,
no-prior-calibration center-point estimation - mark the moment the target
appears, find mx-eye's raw gaze reading at that same real-clock moment,
and pair it with the already-known target position. This reasoning does
not straightforwardly extend to a stage with MULTIPLE distinct candidate
positions shown across trials (e.g. visual_3pt_calibration): "target
shown" alone does not tell you WHICH of several positions was shown on
any one trial without also recording that position per event, which this
module already does via target_center.
"""

from pathlib import Path
from time import time_ns
from typing import TYPE_CHECKING

from mxbiflow.infra import DataLogger, DataLoggerType

if TYPE_CHECKING:
    from mxbiflow.core import MXBIFlow
    from mxbiflow.models import Animal


def _write_trial_event(
    mxbiflow: "MXBIFlow",
    animal: "Animal",
    stage_name: str,
    trial_id: int,
    event_name: str,
    target_center: tuple[int, int] | None,
) -> Path:
    logger = DataLogger(
        session=mxbiflow.session,
        animal=animal.name,
        stage=stage_name,
        filename=f"trial_{trial_id}_{event_name}",
        type=DataLoggerType.CSV,
    )
    logger.save(
        {
            "trial_id": trial_id,
            "event": event_name,
            "timestamp_ns": time_ns(),
            "target_x": target_center[0] if target_center is not None else "",
            "target_y": target_center[1] if target_center is not None else "",
        }
    )
    return logger.path


def write_gaze_correct_event(
    mxbiflow: "MXBIFlow",
    animal: "Animal",
    stage_name: str,
    trial_id: int,
    target_center: tuple[int, int] | None = None,
) -> Path:
    """Write a one-trial sidecar CSV file recording the moment this call
    is made as "gaze/touch registered correctly." Call this at the exact
    point a stage's _on_correct() fires - not earlier, not later. See this
    module's docstring for when this event is (and is not) usable as
    pre-calibration evidence.

    target_center: the (x, y) screen-pixel center of the target this
    trial's touch/gaze was resolved against, if the caller has one handy.
    None is written as an empty cell for a stage/trial where this doesn't
    apply.

    Returns the path written, mainly so a caller can log it for
    troubleshooting (see gaze_input.py's own logging for the pattern).
    """
    return _write_trial_event(
        mxbiflow, animal, stage_name, trial_id, "gaze_correct", target_center
    )


def write_target_shown_event(
    mxbiflow: "MXBIFlow",
    animal: "Animal",
    stage_name: str,
    trial_id: int,
    target_center: tuple[int, int],
) -> Path:
    """Write a one-trial sidecar CSV file recording the moment this call is
    made as "the target stimulus was just shown on screen." Call this at
    the exact point a stage adds the target sprite to the screen (e.g.
    right where visual_stability_training's _show_visual_stimulus() does
    self._sprites.add([self._correct_target])) - not earlier, not later.

    Unlike write_gaze_correct_event, this event does not depend on any
    gaze judgment at all, which is exactly why it's the right event to use
    for from-scratch, no-prior-calibration data collection - see this
    module's docstring.

    target_center: the (x, y) screen-pixel center of the target being
    shown. Required (not optional) here, since a "target shown" event with
    no target position would be useless for calibration.

    Returns the path written, mainly so a caller can log it for
    troubleshooting (see gaze_input.py's own logging for the pattern).
    """
    return _write_trial_event(
        mxbiflow, animal, stage_name, trial_id, "target_shown", target_center
    )
