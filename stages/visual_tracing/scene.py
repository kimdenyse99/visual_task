import math
from random import randint
from shutil import copy2
from time import monotonic, time
from typing import TYPE_CHECKING, ClassVar

from mxbiflow import get_mxbiflow
from mxbiflow.assets import (
    Corner,
    CornerButton,
    ResultWidget,
    create_background,
)
from mxbiflow.core import ConfigStore
from mxbiflow.infra import DataLogger
from mxbiflow.scene import Scene
from pygame import (
    FINGERDOWN,
    MOUSEBUTTONDOWN,
    MOUSEBUTTONUP,
    MOUSEMOTION,
    QUIT,
    SRCALPHA,
    Color,
    Event,
    Surface,
    event,
)
from pygame.draw import circle as draw_circle
from pygame.draw import line as draw_line
from pygame.sprite import Group, Sprite
from rich.pretty import pprint

from dwell import DwellTracker
from gaze_input import GAZE_TOUCH_ID, GazeDwellDriver, GazeSample
from models import TouchEvent
from trial_events import write_target_shown_event

from .models import (
    LEVEL_TABLE,
    PATH,
    Config,
    Result,
    ResultEnum,
    Screen,
    StoredConfigs,
    TrialResults,
)

# Gaze input: see gaze_input.GazeDwellDriver. It drives this scene's
# _on_press/_on_motion/_on_release directly from mx-eye's per-frame gaze
# samples, as a continuous synthetic touch keyed by GAZE_TOUCH_ID - it does
# not go through pygame's event queue at all.
#
# Dwell tracking (is a position currently inside the fixation point or the
# target, and for how long continuously) is handled by dwell.DwellTracker,
# a direct per-frame shape test - see that module's docstring for why this
# replaced an earlier tolerance-circle design.

if TYPE_CHECKING:
    from mxbiflow.core import MXBIFlow
    from mxbiflow.models import Animal


class CircleTargetSprite(Sprite):
    """A plain filled circle target (no auditory/image stimulus involved)."""

    def __init__(
        self,
        diameter_px: int,
        pos: tuple[int, int],
        color: Color | tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        super().__init__()

        surf = Surface((diameter_px, diameter_px), SRCALPHA)
        radius = diameter_px // 2
        draw_circle(surf, color, (radius, radius), radius)

        self.image = surf
        self.rect = self.image.get_rect(center=pos)


class FixationSprite(Sprite):
    """Green circle with a smaller concentric black circle."""

    def __init__(
        self,
        outer_diameter_px: int,
        inner_to_outer_ratio: float,
        pos: tuple[int, int],
    ) -> None:
        super().__init__()

        surf = Surface((outer_diameter_px, outer_diameter_px), SRCALPHA)
        outer_radius = outer_diameter_px // 2
        inner_radius = max(1, round(outer_radius * inner_to_outer_ratio))

        draw_circle(surf, (0, 180, 0), (outer_radius, outer_radius), outer_radius)
        draw_circle(surf, (0, 0, 0), (outer_radius, outer_radius), inner_radius)

        self.image = surf
        self.rect = self.image.get_rect(center=pos)


# Each trial is a brand-new VisualTracing instance (SceneManager constructs
# a fresh scene() on every switch), so a finger/gaze held across the trial
# boundary would otherwise be invisible to the next trial. To let a
# held-through touch register at the next trial's fixation point without
# needing to be lifted first, the dying scene stashes its still-"down"
# touch ids and their last known position here, and the new scene's
# __init__ claims them as already in progress.
#
# This hand-off is only valid for a normal trial-to-trial transition
# *within this stage*. SceneManager.quit() is called on any scene change —
# including the animal leaving mid-trial (-> IDLE), the session ending, or
# switching to a different stage — and in every one of those cases the
# next trial (whenever/wherever it eventually starts) should begin
# completely fresh. quit() only stashes touches when the trial actually
# reached its own normal conclusion (the result was already resolved to
# something other than PENDING before quit() ran); an externally-forced
# quit() leaves the result at PENDING -> CANCELLED, which is the signal
# used here to suppress the stash.
_pending_touch_positions: dict[int, tuple[int, int]] | None = None


class VisualTracing(Scene):
    level_table: ClassVar[dict[str, list[int]]] = LEVEL_TABLE
    _trials_context_key: ClassVar[str] = "visual_tracing.trials"
    _attention_background_color: ClassVar[tuple[int, int, int]] = (48, 48, 48)
    _stimulus_background_color: ClassVar[tuple[int, int, int]] = (127, 127, 127)
    _trace_color: ClassVar[tuple[int, int, int]] = (220, 0, 0)

    # region initialization
    def __init__(self) -> None:
        super().__init__()

        self._running = False

        self._phase: str = "fixation"  # "fixation" -> "test"
        self._is_reaction_window_open = False
        self._reaction_window_started_at_ms: float | None = None

        self._mxbiflow = get_mxbiflow()
        self._rewarder = self._mxbiflow.mxbi.rewarder
        self._timer = self._mxbiflow.timer

        self._animal = self._mxbiflow.session.require_current_animal()
        self._config = self._init_config(self._animal, self._mxbiflow)
        self._data_logger = self._init_data_logger(self._mxbiflow, self._animal)
        self._control_buttons = self._init_control_buttons()

        self._sprites: Group[Sprite] = Group()
        self._init_backgrounds(self._config)
        self._init_fixation_stimulus(self._config)
        self._target_sprite: CircleTargetSprite | None = None
        self._target_center: tuple[int, int] | None = None

        # One DwellTracker per active input (-1 reserved for mouse,
        # GAZE_TOUCH_ID reserved for gaze - see gaze_input.GazeDwellDriver).
        # Claim any touch positions still held from the previous trial (see
        # _pending_touch_positions) so they're already "down" from frame
        # one, rather than requiring a lift-and-re-press to be noticed.
        self._trackers: dict[int, DwellTracker] = {}
        self._trajectories: dict[int, list[tuple[int, int, float]]] = {}
        self._claim_pending_touches()

        # Completed trace points still being drawn: list of (x, y, ms_fired).
        self._trace_points: list[tuple[int, int, float]] = []

        # Drives this scene's press/motion/release handlers directly from
        # mx-eye gaze samples each frame, as a continuous synthetic touch -
        # in parallel with, not instead of, ordinary mouse/finger events
        # handled in handle_event(). Useful for development/testing without
        # eye-tracking hardware: either input source can complete a trial.
        def _get_gaze_sample() -> "GazeSample | None":
            # EyeSample (mxbiflow.core.context.MXBIFlow.eye_sample's real
            # return type) structurally satisfies GazeSample's Protocol
            # (it has x: float and y: float) and this works correctly at
            # runtime; the explicit return-type annotation here is only
            # to satisfy static type checkers, which check a Callable
            # parameter's return type more strictly than a plain
            # structural/Protocol match on an ordinary value.
            return self._mxbiflow.eye_sample

        self._gaze_driver = GazeDwellDriver(
            _get_gaze_sample,
            self,
            already_down=GAZE_TOUCH_ID in self._trackers,
        )

        self._mxbiflow.session.add_trial()
        self._result = self._init_result(self._animal, self._config)
        self._result_widget = self._init_result_widget()

        self._print_trial_config()

    # region config

    def _init_config(self, animal: "Animal", mxbiflow: "MXBIFlow") -> Config:
        snapshot_path = self._config_snapshot_path(mxbiflow, animal)
        if not snapshot_path.exists():
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            copy2(PATH, snapshot_path)

        stored_configs = ConfigStore(
            snapshot_path,
            StoredConfigs,
            create_default=False,
        ).value
        stored_config = stored_configs.root.get(animal.name) or stored_configs.root.get(
            "default"
        )

        if stored_config is None:
            raise ValueError(f"No config found for animal {animal.name}")

        level = animal.current_stage.level
        level_config = stored_config.levels.get(level)

        if level_config is None:
            raise ValueError(
                f"No config found for animal {animal.name} at level {level}"
            )

        screen_size = mxbiflow.mxbi.screen_size

        return Config(
            animal=animal.name,
            level=level,
            screen_size=Screen(screen_size.width, screen_size.height),
            stored=stored_config,
        )

    def _claim_pending_touches(self) -> None:
        global _pending_touch_positions

        positions = _pending_touch_positions
        _pending_touch_positions = None

        if not positions:
            return

        # The fixation point is shown immediately (see _on_trial_start),
        # so a held-through touch starts its dwell fresh, at its current
        # position, right now. The trajectory (used only for the visible
        # trace dot/trail) deliberately starts EMPTY rather than seeded
        # with this handed-off position: that position is already one
        # frame stale by the time this runs, and gaze in particular may
        # have moved again before its next real sample arrives. Leaving
        # it empty means the trace simply shows nothing until the first
        # genuinely fresh _on_motion call populates it, rather than
        # briefly flashing the old position.
        for touch_id, pos in positions.items():
            tracker = DwellTracker(touch_id=touch_id)
            tracker.last_pos = pos
            self._trackers[touch_id] = tracker
            self._trajectories[touch_id] = []

    def _config_snapshot_path(self, mxbiflow: "MXBIFlow", animal: "Animal"):
        return (
            mxbiflow.session.absolute_animal_data_path(animal.name)
            / self.name()
            / "config.json"
        )

    # endregion config

    # region data

    def _init_result(self, animal: "Animal", config: Config) -> Result:
        now = time()

        return Result(
            animal=animal.base_info,
            result=ResultEnum.PENDING,
            trial_start_time=now,
            trial_end_time=now,
            target_x=0,
            target_y=0,
            target_diameter_px=config.target_diameter_px,
        )

    def _init_data_logger(self, mxbiflow: "MXBIFlow", animal: "Animal") -> DataLogger:
        return DataLogger(
            session=mxbiflow.session,
            animal=animal.name,
            stage=self.name(),
        )

    def _load_trials(self) -> TrialResults:
        trials = self._animal.current_stage.get_context(
            self._trials_context_key,
            TrialResults,
        )
        return trials if trials is not None else TrialResults(root={})

    def _save_trials(self, trials: TrialResults) -> None:
        self._mxbiflow.session.set_stage_context(
            self._trials_context_key,
            trials,
            animal=self._animal.name,
            stage=self._animal.current_stage.stage_name,
        )

    # endregion data

    # region views

    def _init_result_widget(self) -> ResultWidget:
        return ResultWidget(
            self._result_summary(),
            border_width=self._config.border_width,
            color=(0, 0, 0),
        )

    def _init_control_buttons(self) -> Group[CornerButton]:
        border_width = self._config.border_width
        button_size = border_width // 2
        return Group(
            CornerButton(
                Corner.TOP_RIGHT,
                button_size,
                border_width,
                self._give_manual_reward,
                color=(0, 160, 0),
            ),
            CornerButton(
                Corner.BOTTOM_RIGHT,
                button_size,
                border_width,
                self._request_game_quit,
                color=(200, 0, 0),
            ),
        )

    # endregion views

    # region stimulus

    def _init_backgrounds(self, config: Config) -> None:
        screen_size = (config.screen_size.width, config.screen_size.height)
        self._stimulus_background = create_background(
            screen_size,
            border_width=config.border_width,
            background_color=self._stimulus_background_color,
        )
        self._attention_background = create_background(
            screen_size,
            border_width=config.border_width,
            background_color=self._attention_background_color,
        )
        self._background = self._attention_background

    def _init_fixation_stimulus(self, config: Config) -> None:
        # (0, 0) +------------------------------> x = W
        #        |              Fixation (3W/4, H/2)
        #        |
        #        |
        #        v
        #      y = H
        screen_size = config.screen_size
        self._fixation_center = (
            screen_size.width * 4 // 5,  # initially 3 // 4, made it higher
            screen_size.height // 2,
        )
        self._fixation_sprite = FixationSprite(
            config.fixation_outer_diameter_px,
            config.fixation_inner_to_outer_ratio,
            self._fixation_center,
        )

    def _random_target_center(self, config: Config) -> tuple[int, int]:
        """Pick a random center for the target anywhere in the right half of
        the screen, keeping the whole circle inside the border."""

        border = config.border_width
        radius = config.target_diameter_px // 2
        screen_size = config.screen_size

        min_x = max(screen_size.width * 3 // 4, border + radius)
        max_x = screen_size.width - border - radius
        min_y = border + radius
        max_y = screen_size.height - border - radius

        if max_x < min_x:
            max_x = min_x
        if max_y < min_y:
            max_y = min_y

        return (randint(min_x, max_x), randint(min_y, max_y))

    # endregion stimulus

    # region console

    def _print_trial_config(self) -> None:
        print()
        print("=" * 60)
        print(
            f"Beginning level {self._animal.current_stage.level}, "
            f"trial {self._animal.trial_id}, "
            f"for monkey {self._animal.name}, "
            f"in the {self.name()} stage"
        )
        print("=" * 60)
        pprint(self._config)
        print("=" * 60)

    # endregion console

    # endregion initialization

    # region public api
    def start(self) -> None:
        self._result_widget.set_text(self._result_summary())

        self._running = True

        self._on_trial_start()

    def quit(self) -> None:
        # Capture this before _cancel_unfinished_trial() can overwrite it:
        # a trial that reached its own natural conclusion (correct,
        # incorrect, or timeout) will already have a resolved result by
        # the time quit() runs normally, via _on_inter_trial -> self.quit.
        # PENDING here means something else forced this scene to end
        # mid-trial (the animal left, the session ended, the stage
        # changed), which is exactly when the next trial should start
        # completely fresh rather than inheriting any held touch.
        ended_normally = self._result.result != ResultEnum.PENDING

        self._cancel_unfinished_trial()
        self._result.trial_end_time = time()

        is_level_up = self._settlement()
        self._print_result(is_level_up)

        self._data_logger.save(self._result.model_dump())
        self._timer.cancel_all()

        # Hand off any touch still held at the end of this trial so the
        # next trial's scene can claim it as already in progress (see
        # _claim_pending_touches) instead of losing track of it — but
        # only for a normal trial-to-trial transition within this stage.
        if ended_normally:
            self._stash_pending_touches()
        else:
            self._clear_pending_touches()

        # This scene instance is being torn down either way; any gaze
        # dwell's position/timing state now lives in the stashed (or
        # cleared) positions, not in this driver instance, so stop it
        # from acting (e.g. firing a release/abandon call on a future
        # frame that will never come for a dead instance).
        self._gaze_driver.discard()

        self._running = False

    def _stash_pending_touches(self) -> None:
        global _pending_touch_positions
        positions = {
            touch_id: tracker.last_pos
            for touch_id, tracker in self._trackers.items()
            if tracker.last_pos is not None
        }
        _pending_touch_positions = positions or None

    def _clear_pending_touches(self) -> None:
        global _pending_touch_positions
        _pending_touch_positions = None

    def handle_event(self, event_obj: Event) -> None:
        if event_obj.type == MOUSEBUTTONDOWN and event_obj.button == 1:
            if any(
                button.handle_event(event_obj) for button in self._control_buttons
            ):
                return
            self._on_press(-1, event_obj.pos)

        elif event_obj.type == MOUSEBUTTONUP and event_obj.button == 1:
            self._on_release(-1, event_obj.pos)

        elif event_obj.type == MOUSEMOTION and -1 in self._trackers:
            self._on_motion(-1, event_obj.pos)

        elif event_obj.type == FINGERDOWN:
            # Touchscreen input is actively discouraged now that gaze is
            # the intended response channel: any touch abandons the trial
            # immediately, with no reward - it does not register as a
            # dwell at all. Mouse input (MOUSEBUTTONDOWN, above) is left
            # exactly as before for development/testing without a
            # touchscreen or eye tracker attached.
            self._on_touch_penalized()

        # Gaze input does not arrive as a pygame event at all - it is
        # polled directly from mx-eye each frame in update(), via
        # self._gaze_driver (see gaze_input.GazeDwellDriver).

    def update(self, dt_s: float) -> None:
        self._control_buttons.update(dt_s)
        self._gaze_driver.update()
        self._check_all_trackers()
        self._prune_trace()

    def draw(self, screen: Surface) -> None:
        screen.blit(self._background, (0, 0))

        self._sprites.draw(screen)
        self._draw_trace(screen)
        self._result_widget.draw(screen)
        for button in self._control_buttons:
            button.layout(screen)
        self._control_buttons.draw(screen)

    # endregion public api

    # region touch handling helpers

    def _on_press(self, touch_id: int, pos: tuple[int, int]) -> None:
        now_ms = monotonic() * 1000
        self._trackers[touch_id] = DwellTracker(touch_id=touch_id)
        self._trajectories[touch_id] = [(pos[0], pos[1], now_ms)]
        # A dwell could already satisfy the duration rule on the very
        # frame it starts if min_touch_duration_ms is 0; check
        # immediately rather than waiting for the next update() tick.
        self._evaluate_tracker(touch_id, pos)

    def _on_motion(self, touch_id: int, pos: tuple[int, int]) -> None:
        if touch_id not in self._trackers:
            return

        now_ms = monotonic() * 1000
        self._trajectories.setdefault(touch_id, []).append((pos[0], pos[1], now_ms))

        self._evaluate_tracker(touch_id, pos)

    def _on_release(self, touch_id: int, pos: tuple[int, int]) -> None:
        tracker = self._trackers.pop(touch_id, None)
        if tracker is None:
            return

        now_ms = monotonic() * 1000
        trajectory = self._trajectories.pop(touch_id, [])
        trajectory.append((pos[0], pos[1], now_ms))
        self._trace_points.extend(trajectory)

        # Releasing is bookkeeping only (it ends the press and logs it
        # for data completeness) — it does not gate or retroactively
        # change whether this was a valid touch event. That was already
        # decided, continuously, by _evaluate_tracker while the touch
        # was active.
        self._result.touch_event.append(
            TouchEvent(x=pos[0], y=pos[1], timestamp=time())
        )

    def _check_all_trackers(self) -> None:
        """Called every frame so a dwell can fire as soon as it has been
        held stable long enough, without waiting for a motion event or
        release. This matters most for a stationary gaze/mouse: if the
        position never changes, no motion event ever arrives, but the
        dwell duration still elapses in real time."""
        for touch_id, tracker in list(self._trackers.items()):
            if tracker.last_pos is not None:
                self._evaluate_tracker(touch_id, tracker.last_pos)

    def _active_shape(self) -> tuple[tuple[int, int], float] | None:
        """Returns (center, acceptance_radius_px) for whatever shape a
        touch event is currently allowed to register against, or None if
        there is nothing to register against right now (phase is "test"
        but the reaction window has already closed)."""
        if self._phase == "fixation":
            radius_px = (
                self._config.fixation_outer_diameter_px / 2
                + self._config.fixation_touch_reward_radius_px
            )
            return self._fixation_center, radius_px
        if self._phase == "test":
            if not self._is_reaction_window_open or self._target_center is None:
                return None
            radius_px = (
                self._config.target_diameter_px / 2
                + self._config.target_touch_reward_radius_px
            )
            return self._target_center, radius_px
        return None

    def _evaluate_tracker(self, touch_id: int, pos: tuple[int, int]) -> None:
        tracker = self._trackers.get(touch_id)
        if tracker is None or tracker.fired:
            return

        shape = self._active_shape()
        if shape is None:
            # Nothing to register against right now: don't let dwell
            # time accumulate, but don't reset it either - the grace
            # period semantics only apply to brief exits *from* a
            # present shape, not to there being no shape at all (e.g.
            # the instant between the reaction window closing and the
            # next trial's fixation point appearing). Simplest correct
            # behavior here is to leave the tracker exactly as it is.
            return

        center, radius_px = shape
        distance_px = math.hypot(pos[0] - center[0], pos[1] - center[1])
        is_inside = distance_px <= radius_px

        fired_now = tracker.update(
            pos,
            is_inside,
            self._config.min_touch_duration_ms,
            self._config.dwell_grace_ms,
        )
        if fired_now:
            self._handle_touch_event(pos)

    def _handle_touch_event(self, pos: tuple[int, int]) -> None:
        if self._phase == "fixation":
            self._on_fixation_touched()
            return

        if not self._is_reaction_window_open:
            return

        self._is_reaction_window_open = False
        # Check the hit *before* clearing the target: _clear_target()
        # nulls out self._target_center, which _is_target_touch() needs
        # to know what to test against.
        is_correct = self._is_target_touch(pos)
        self._clear_target()

        if is_correct:
            started_at_ms = self._reaction_window_started_at_ms
            if started_at_ms is None:
                raise RuntimeError("Reaction window start time is unavailable")
            latency_ms = max(0.0, monotonic() * 1000 - started_at_ms)
            self._result.response_latency_ms = latency_ms
            self._on_correct()
        else:
            self._on_incorrect()

    def _is_target_touch(self, pos: tuple[int, int]) -> bool:
        if self._target_center is None:
            return False
        radius_px = (
            self._config.target_diameter_px / 2
            + self._config.target_touch_reward_radius_px
        )
        distance_px = math.hypot(
            pos[0] - self._target_center[0], pos[1] - self._target_center[1]
        )
        return distance_px <= radius_px

    def _prune_trace(self) -> None:
        max_age_ms = self._config.touch_trace_duration_ms
        if max_age_ms <= 0:
            self._trace_points.clear()
            return
        now_ms = monotonic() * 1000
        self._trace_points = [
            point for point in self._trace_points if now_ms - point[2] <= max_age_ms
        ]

    def _draw_trace(self, screen: Surface) -> None:
        max_age_ms = self._config.touch_trace_duration_ms

        # Each press's own point history, drawn and faded independently
        # so that different fingers (or an active press and older
        # released history) are never connected to one another. Released
        # history comes from self._trace_points (not tied to any one
        # touch_id, so it's its own group); each still-active touch
        # contributes its live, in-progress trajectory as a separate
        # group — this is what makes the fading trail visible *while* a
        # touch is moving, not only after it's released.
        point_groups: list[list[tuple[int, int, float]]] = []
        if max_age_ms > 0 and self._trace_points:
            point_groups.append(sorted(self._trace_points, key=lambda p: p[2]))
        for trajectory in self._trajectories.values():
            if trajectory:
                point_groups.append(trajectory)

        if not point_groups:
            return

        # The main display surface has no per-pixel alpha channel, so the
        # fading trace is composited on an SRCALPHA overlay first and
        # then blitted onto the screen in one pass.
        overlay = Surface(screen.get_size(), SRCALPHA)

        now_ms = monotonic() * 1000
        active_trajectory_ids = {id(t) for t in self._trajectories.values()}

        for points in point_groups:
            is_live = id(points) in active_trajectory_ids
            last_index = len(points) - 1

            for i, (x, y, fired_ms) in enumerate(points):
                # The current position of a still-active touch is always
                # shown at full opacity, regardless of age — a stationary
                # or moving finger shouldn't have its own dot fade out
                # while it's still touching the screen. Only the trail
                # behind it fades.
                if is_live and i == last_index:
                    alpha = 255
                else:
                    age_ms = now_ms - fired_ms
                    alpha = max(0, 255 - int(255 * age_ms / max_age_ms))
                if alpha <= 0:
                    continue
                draw_circle(overlay, (*self._trace_color, alpha), (x, y), 4)

            for i in range(1, len(points)):
                x0, y0, t0 = points[i - 1]
                x1, y1, t1 = points[i]
                # Only connect points recorded close together in time, to
                # avoid a visible jump if a frame or two was dropped.
                if abs(t1 - t0) > 150:
                    continue
                if is_live and i == last_index:
                    alpha = 255
                else:
                    age_ms = now_ms - t1
                    alpha = max(0, 255 - int(255 * age_ms / max_age_ms))
                if alpha <= 0:
                    continue
                draw_line(overlay, (*self._trace_color, alpha), (x0, y0), (x1, y1), 3)

        screen.blit(overlay, (0, 0))

    # endregion touch handling helpers

    # region life cycle
    def _on_trial_start(self) -> None:
        self._result.trial_start_time = time()
        self._phase = "fixation"
        self._sprites.add(self._fixation_sprite)
        self._fixation_shown_at_ms = monotonic() * 1000

    def _on_fixation_touched(self) -> None:
        self._result.fixation_touch_latency_ms = max(
            0.0, monotonic() * 1000 - self._fixation_shown_at_ms
        )
        self._sprites.remove(self._fixation_sprite)
        # The target appears immediately — there is no blank gap to
        # protect against, so no artificial delay is introduced here.
        self._show_visual_stimulus()

    def _on_inter_trial(self) -> None:
        self._is_reaction_window_open = False
        self._timer.after(
            self._config.inter_trial_duration_ms,
            self.quit,
        )

    def _settlement(self) -> bool:
        trials = self._load_trials()
        trials.root[self._result.animal.trial_id] = self._result
        self._save_trials(trials)

        # This stage is for visual testing / tracing; it does not
        # progress through levels automatically.
        return False

    def _print_result(self, is_level_up: bool) -> None:
        trial_level = self._result.animal.level

        print()
        print("=" * 60)
        print(
            f"Trial result for level {trial_level}, "
            f"trial {self._animal.trial_id}, "
            f"for monkey {self._animal.name}, "
            f"in the {self.name()} stage"
        )
        print("=" * 60)
        print(f"Settlement: remain at level {trial_level}")
        pprint(self._result)
        print("=" * 60)

    # endregion life cycle

    # region views
    def _show_visual_stimulus(self) -> None:
        self._background = self._stimulus_background
        self._phase = "test"

        center = self._random_target_center(self._config)
        self._target_center = center
        self._result.target_x = center[0]
        self._result.target_y = center[1]

        self._target_sprite = CircleTargetSprite(
            self._config.target_diameter_px,
            center,
            color=(0, 0, 0),
        )
        self._sprites.add(self._target_sprite)

        # Written at the exact moment the target becomes visible - not
        # gated on any gaze judgment at all, unlike a "gaze correct"
        # event. The target's position here is randomized per trial
        # (_random_target_center), so unlike visual_stability_training's
        # single fixed center point, each trial's pair should be treated
        # as its own independent known point when later used for
        # calibration, not assumed to be the same location across
        # trials. See trial_events.py's module docstring for the full
        # reasoning on why "target shown" is the right pre-calibration
        # event.
        write_target_shown_event(
            self._mxbiflow,
            self._animal,
            self.name(),
            self._animal.trial_id,
            center,
        )

        self._reaction_window_started_at_ms = monotonic() * 1000
        self._is_reaction_window_open = True
        self._timer.after(
            self._config.reaction_window_duration_ms,
            self._on_timeout,
        )

        # A press that is already down (e.g. held continuously from the
        # fixation touch) is eligible to register on the target too:
        # reset its dwell tracker so it starts accumulating fresh against
        # the new shape, at whatever position it's currently at, rather
        # than carrying over anything from the fixation point.
        for touch_id, tracker in self._trackers.items():
            tracker.reset()
            if tracker.last_pos is not None:
                self._evaluate_tracker(touch_id, tracker.last_pos)

    def _clear_target(self) -> None:
        if self._target_sprite is not None:
            self._sprites.remove(self._target_sprite)
        self._target_center = None

    def _result_summary(self) -> str:
        level = self._result.animal.level
        correct = 0
        incorrect = 0
        timeout = 0
        cancelled = 0
        reward = 0
        for trial in self._load_trials().root.values():
            if trial.result is ResultEnum.CORRECT:
                correct += 1
            elif trial.result is ResultEnum.INCORRECT:
                incorrect += 1
            elif trial.result is ResultEnum.TIMEOUT:
                timeout += 1
            elif trial.result is ResultEnum.CANCELLED:
                cancelled += 1
            reward += trial.trial_reward_count + trial.trial_manual_reward_count

        total = correct + incorrect + timeout
        rate = correct / total if total else 0
        return (
            f"Animal {self._animal.name} | Trial {self._animal.trial_id} | "
            f"Level {level} | Rate {rate:.1%} | "
            f"Correct {correct} | Incorrect {incorrect} | "
            f"Reward {reward} | "
            f"Timeout {timeout} | "
            f"Cancelled {cancelled}"
        )

    # endregion views

    # region events handler
    def _on_correct(self):
        self._result.result = ResultEnum.CORRECT

        self._give_reward()
        self._on_inter_trial()

    def _on_incorrect(self):
        self._result.result = ResultEnum.INCORRECT

        self._on_inter_trial()

    def _on_timeout(self) -> None:
        if not self._is_reaction_window_open:
            return

        self._is_reaction_window_open = False
        self._clear_target()
        self._result.result = ResultEnum.TIMEOUT

        self._on_inter_trial()

    def _cancel_unfinished_trial(self) -> None:
        if self._result.result != ResultEnum.PENDING:
            return

        self._result.result = ResultEnum.CANCELLED

    def _request_game_quit(self) -> None:
        event.post(Event(QUIT))

    def on_gaze_abandoned(self) -> None:
        """Called by GazeDwellDriver when gaze has been lost for longer
        than its grace period. This trial is given up on - not resolved
        as correct, incorrect, or timeout, since none of those actually
        happened - the same way quit() already treats any other
        externally-forced end of trial (_cancel_unfinished_trial marks an
        unresolved PENDING result as CANCELLED). This does not imply the
        animal has left the session; only this one trial ends early."""
        if self._running:
            self.quit()

    def _on_touch_penalized(self) -> None:
        """A touchscreen touch (not mouse) was detected. Touch is
        actively discouraged now that gaze is the intended response
        channel: give no reward and abandon this trial immediately
        (CANCELLED, same as on_gaze_abandoned), restarting fresh at the
        next trial's normal start rather than resolving as
        correct/incorrect/timeout."""
        if self._running:
            self.quit()

    # endregion events handler

    # region stimulus and reward
    def _give_reward(self, duration_ms: int | None = None) -> None:
        self._rewarder.give_reward(duration_ms or self._config.reward_duration_ms)
        self._result.trial_reward_count += 1

    def _give_manual_reward(self) -> None:
        self._give_reward()
        self._result.trial_manual_reward_count += 1

    # endregion stimulus and reward

    # region utils
    # endregion utils