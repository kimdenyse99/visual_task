"""Simple, direct dwell tracking: "is this position currently inside the
active stimulus's shape, and has it been there long enough."

This replaces the earlier tolerance-circle design (_ActiveTouch with a
sliding reference_pos/reference_ms), which was built around touch/finger
drag semantics ("settle somewhere, then hold") and did not behave well for
continuous gaze input - a stale reference point could lag behind the
live position, causing a dwell to lock onto the wrong spot or never
re-settle onto the actual target.

The replacement is deliberately direct: every frame, test whether the
current position is inside the current stimulus's shape (a plain circular
distance check against its center and acceptance radius - no tolerance
circle, no re-anchoring, no historical reference point). Accumulate
continuous time spent inside; fire the instant that reaches the
configured duration, using the *current* position, not a remembered one.

A short, configurable grace period absorbs brief exits (a single noisy
frame, a momentary dropout) without discarding progress already made:
leaving the shape starts a grace countdown rather than resetting
immediately, and returning before the grace period elapses resumes the
existing dwell exactly where it left off. Only an exit that outlasts the
grace period actually resets the accumulated dwell time to zero.
"""

from dataclasses import dataclass, field
from time import monotonic


@dataclass
class DwellTracker:
    """Tracks one input's (touch_id's) continuous dwell against whatever
    shape is currently being tested (the fixation point, or the target -
    the caller decides which shape to test against on each call, since
    that depends on the scene's current phase)."""

    touch_id: int

    # Set while a continuous (grace-tolerant) dwell is in progress; None
    # when no dwell is currently accumulating.
    inside_since_ms: float | None = None

    # Set the moment the position is found outside the shape; cleared the
    # moment it's found inside again. Used only to measure how long an
    # exit has lasted, to compare against the grace period.
    outside_since_ms: float | None = None

    # Once True, this touch_id is done for the current stimulus: it has
    # already resolved (fired) and further updates are ignored until the
    # scene explicitly calls reset() for a new phase/stimulus.
    fired: bool = False

    # The most recent position passed to update(), kept so a caller can
    # fire using "wherever this touch currently is" without needing to
    # separately track it.
    last_pos: tuple[int, int] | None = None

    def reset(self) -> None:
        """Clear all dwell progress and the fired latch. Call this when a
        new stimulus appears (fixation -> target transition, or a new
        trial) so a continuously-held input is eligible to dwell-and-fire
        again against the new shape, starting from zero."""
        self.inside_since_ms = None
        self.outside_since_ms = None
        self.fired = False

    def update(
        self,
        pos: tuple[int, int],
        is_inside: bool,
        min_duration_ms: float,
        grace_ms: float,
    ) -> bool:
        """Advance dwell tracking by one sample. Returns True exactly once,
        on the frame the dwell duration is first satisfied while inside
        (the caller should resolve the touch event using self.last_pos,
        which equals pos, on that same call)."""
        self.last_pos = pos

        if self.fired:
            return False

        now_ms = monotonic() * 1000

        if is_inside:
            self.outside_since_ms = None
            if self.inside_since_ms is None:
                self.inside_since_ms = now_ms
        else:
            if self.outside_since_ms is None:
                self.outside_since_ms = now_ms
            elapsed_outside_ms = now_ms - self.outside_since_ms
            if elapsed_outside_ms >= grace_ms:
                # Grace period exhausted: this exit is real, not jitter.
                self.inside_since_ms = None
                self.outside_since_ms = None
            # Within the grace period: leave inside_since_ms exactly as
            # it was, so a brief excursion doesn't cost any accumulated
            # time, but also doesn't let time accumulate *during* the
            # excursion itself (the check below only fires while
            # is_inside is True on the current sample).
            return False

        if self.inside_since_ms is None:
            return False

        dwell_ms = now_ms - self.inside_since_ms
        if dwell_ms < min_duration_ms:
            return False

        self.fired = True
        return True
