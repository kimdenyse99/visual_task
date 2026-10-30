"""Shared driver that feeds mx-eye gaze samples into a scene's existing
touch/dwell machinery as a continuous, synthetic touch.

Each gaze-enabled stage already has its own press/motion/release handlers
(``_on_press`` / ``_on_motion`` / ``_on_release``) built for mouse and finger
input, plus a dwell-to-fire system (``_ActiveTouch`` + ``_evaluate_touch``)
that only counts a touch once it has held still, within a tolerance circle,
for ``min_touch_duration_ms``. Gaze has no physical "down"/"up" of its own -
it is just a stream of (x, y) positions, or ``None`` when tracking is lost -
so this module translates that stream into the same press/motion/release
calls a held, dragging finger would have produced:

- Gaze goes from invalid to valid: synthesize a "press" at the new position.
- Gaze stays valid, position may have moved: synthesize "motion" every frame.
  The scene's own tolerance-circle logic (already written for touch/finger
  drag) decides whether that movement is within normal jitter or a genuine
  re-fixation; this module does not duplicate or second-guess that.
- Gaze is lost: rather than releasing immediately (which would discard
  whatever stability had already accumulated for one blink or a momentary
  tracking dropout), the dwell is held in place, unconfirmed, for up to
  ``lost_grace_ms``. If gaze returns within that window, tracking resumes
  from the same reference point and accumulated dwell clock exactly where
  it left off - and the *grace-period* countdown itself is fully reset:
  even a single intervening valid frame means the next loss gets a brand
  new ``lost_grace_ms`` window, not whatever was left of the previous one.
  If the grace period elapses with gaze still lost, the synthetic touch is
  released and the trial is treated as abandoned: the scene's
  ``on_gaze_abandoned`` callback is invoked so it can mark the trial
  cancelled and move on to inter-trial/quit, the same way any other
  externally-forced end of trial already does. This does not report the
  animal as having left the session (no RFID/detector event is touched) -
  only this one trial is given up on.

A scene using this module still receives ordinary mouse/finger pygame events
through its own ``handle_event`` exactly as before; gaze is an *additional*
input source multiplexed into the same ``_active_touches`` dict under a
reserved id, not a replacement for the event-driven path.
"""

from time import monotonic
from typing import Callable, Protocol

from mxbiflow.utils.logger import logger

# Reserved synthetic touch id for gaze, distinct from mouse (-1) and real
# finger ids (>= 0, assigned by SDL).
GAZE_TOUCH_ID = -2

# Default grace period: how long a lost gaze signal is allowed to pause an
# in-progress dwell before the trial is abandoned.
# Configurable per call/stage; see GazeDwellDriver(..., lost_grace_ms=...).
DEFAULT_GAZE_LOST_GRACE_MS = 5000


class GazeSample(Protocol):
    """Structurally matches anything with readable x/y float coordinates.

    Declared as read-only properties, not plain mutable attributes:
    GazeDwellDriver only ever reads x/y off a sample, never assigns to
    them, and mxbiflow's own EyeSample (the real type eye_sample()
    returns) is an immutable frozen dataclass. A Protocol with plain
    mutable attributes requires the implementing type's attributes to
    also be mutable/invariant to satisfy it; declaring these as
    read-only properties instead makes the protocol covariant here,
    which is what actually matches how this is used and lets a frozen
    dataclass like EyeSample satisfy it.
    """

    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...


class _GazeDwellScene(Protocol):
    """The subset of a scene's API this driver calls into.

    Every gaze-enabled scene already implements ``_on_press``/``_on_motion``/
    ``_on_release`` for mouse/finger input (see e.g.
    ``VisualStabilityTraining._on_press``), and ``on_gaze_abandoned`` is a
    small new hook the scene provides to end its own trial as abandoned.
    This protocol exists only so GazeDwellDriver can be typed against them
    without importing any one concrete scene.
    """

    def _on_press(self, touch_id: int, pos: tuple[int, int]) -> None: ...
    def _on_motion(self, touch_id: int, pos: tuple[int, int]) -> None: ...
    def _on_release(self, touch_id: int, pos: tuple[int, int]) -> None: ...
    def on_gaze_abandoned(self) -> None: ...


class GazeDwellDriver:
    """Feeds mx-eye gaze samples into a scene's existing touch machinery.

    Call :meth:`update` once per frame (e.g. from the scene's own
    ``update(dt_s)``), after the scene has handled its ordinary pygame
    events for that frame.
    """

    def __init__(
        self,
        get_sample: "Callable[[], GazeSample | None]",
        scene: _GazeDwellScene,
        *,
        lost_grace_ms: float = DEFAULT_GAZE_LOST_GRACE_MS,
        already_down: bool = False,
    ) -> None:
        # get_sample is a zero-argument callable returning the latest
        # EyeSample or None (typically ``lambda: mxbiflow.eye_sample``),
        # rather than the mxbiflow object itself, so this driver has no
        # dependency on mxbiflow's shape beyond the one value it actually
        # needs each frame.
        self._get_sample = get_sample
        self._scene = scene
        self._lost_grace_ms = lost_grace_ms

        # already_down=True when the scene has already claimed a gaze
        # _ActiveTouch entry (keyed by GAZE_TOUCH_ID) handed off from the
        # previous trial's _pending_touches, so this driver should
        # synthesize "motion" (continuing an existing press) rather than
        # "press" (starting a new one) on the next valid sample.
        self._is_down = already_down
        self._last_pos: tuple[int, int] | None = None
        self._lost_since_ms: float | None = None
        self._abandon_reported = False

    def update(self) -> None:
        """Advance gaze-to-touch translation by one frame."""
        sample = self._get_sample()

        if sample is None:
            self._handle_lost()
            return

        pos = (round(sample.x), round(sample.y))

        if not self._is_down:
            self._is_down = True
            self._lost_since_ms = None
            self._abandon_reported = False
            self._last_pos = pos
            logger.debug("gaze acquired at ({}, {})", pos[0], pos[1])
            self._scene._on_press(GAZE_TOUCH_ID, pos)
            return

        if self._lost_since_ms is not None:
            logger.debug(
                "gaze recovered after {:.0f} ms, resuming dwell",
                monotonic() * 1000 - self._lost_since_ms,
            )

        # Gaze is back after a momentary loss: resume in place rather than
        # re-pressing, so the dwell clock is not restarted.
        self._lost_since_ms = None
        self._last_pos = pos
        self._scene._on_motion(GAZE_TOUCH_ID, pos)

    def _handle_lost(self) -> None:
        if not self._is_down:
            # Nothing in progress to pause or release.
            return

        now_ms = monotonic() * 1000
        if self._lost_since_ms is None:
            self._lost_since_ms = now_ms
            logger.debug(
                "gaze lost, starting {:.0f} ms grace period", self._lost_grace_ms
            )
            return

        elapsed_ms = now_ms - self._lost_since_ms
        if elapsed_ms < self._lost_grace_ms:
            # Still within the grace period: leave the dwell exactly as it
            # was (no motion/press/release call at all) so the scene's own
            # stability clock neither advances nor resets while waiting.
            return

        # Grace period elapsed with gaze still lost: release the synthetic
        # touch and give up on this trial - it does not resolve as correct,
        # incorrect, or timeout, since none of those actually happened; it
        # is simply abandoned, same as any other externally-forced end of
        # trial. This does not imply the animal has left the session - no
        # detector/RFID event is touched here.
        logger.warning(
            "gaze lost for {:.0f} ms (grace period exceeded); abandoning trial",
            elapsed_ms,
        )
        if self._last_pos is not None:
            self._scene._on_release(GAZE_TOUCH_ID, self._last_pos)

        self._is_down = False
        self._last_pos = None
        self._lost_since_ms = None

        if not self._abandon_reported:
            self._abandon_reported = True
            self._scene.on_gaze_abandoned()

    def discard(self) -> None:
        """Drop any in-progress gaze dwell without emitting a release call.

        Call this from the scene's own teardown (``quit``) so a trial end
        that already resolved normally doesn't also trigger a spurious
        release/abandon call on a future frame that will never come for a
        dead scene instance.
        """
        self._is_down = False
        self._last_pos = None
        self._lost_since_ms = None
        self._abandon_reported = True