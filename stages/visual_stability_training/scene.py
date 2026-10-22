import math
from pathlib import Path
from random import choice
from shutil import copy2
from time import monotonic, time
from typing import TYPE_CHECKING, ClassVar, Literal

import pygame
from mxbiflow import get_mxbiflow
from mxbiflow.assets import (
    Corner,
    CornerButton,
    ImageSprite,
    ResultWidget,
    create_background,
)
from mxbiflow.core import ConfigStore
from mxbiflow.infra import DataLogger
from mxbiflow.scene import Scene
from pygame import (
    FINGERDOWN,
    FINGERMOTION,
    FINGERUP,
    MOUSEBUTTONDOWN,
    MOUSEBUTTONUP,
    MOUSEMOTION,
    QUIT,
    Event,
    event,
)
from pygame.sprite import Group, Sprite
from rich.pretty import pprint

from assets import M001, M002
from gaze_input import GAZE_TOUCH_ID, GazeDwellDriver
from models import TouchEvent

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

if TYPE_CHECKING:
    from mxbiflow.core import MXBIFlow
    from mxbiflow.models import Animal
    from pygame.surface import Surface

# Gaze input: see gaze_input.GazeDwellDriver. It drives this scene's
# existing press/motion/release handlers (_on_press/_on_motion/_on_release)
# directly from mx-eye's per-frame gaze samples, as a continuous synthetic
# touch keyed by GAZE_TOUCH_ID - it does not go through pygame's event
# queue at all, so there is no custom event type to allocate here.


class FixationPoint(Sprite):
    """A green circle with a small black circle in its center.

    Used as a static/decorative fixation cue during the attention screen.
    Its size is rebuilt per-level (see Visual3ptCalibration-style
    size-reduction schedules) via _init_visual_stimulus, rather than
    being fixed for the whole stage.
    """

    def __init__(
        self,
        center: tuple[int, int],
        radius: int,
        inner_radius_ratio: float,
    ) -> None:
        super().__init__()

        diameter = radius * 2
        self.image = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        outer_center = (radius, radius)
        pygame.draw.circle(self.image, (0, 200, 0), outer_center, radius)
        inner_radius = max(1, round(radius * inner_radius_ratio))
        pygame.draw.circle(self.image, (0, 0, 0), outer_center, inner_radius)

        self.rect = self.image.get_rect(center=center)


class _ActiveTouch:
    """Tracks one in-progress press from down to up.

    A touch event fires at the position where the touch becomes *stable*,
    not necessarily where it first landed. "Stable" means: has stayed
    within the tolerance circle around some reference point for at least
    `min_touch_duration_ms`. The tolerance circle is a sliding window, not
    pinned to the original touch-down point:

    - `reference_pos` / `reference_ms` describe the current tolerance
      circle: its center, and when the touch most recently arrived there
      (or was found to still be within it).
    - On every motion sample, if the new position falls within the
      tolerance circle of `reference_pos`, the touch keeps accumulating
      time toward the duration threshold (reference stays put). If the
      new position falls *outside* that circle, the touch hasn't settled
      yet: the reference point jumps to the new position and the duration
      clock restarts from now.
    - `fired` becomes True the instant accumulated stable time reaches
      `min_touch_duration_ms`. A press held continuously across a phase
      change (fixation point through to the target appearing) is
      eligible to fire again for the new phase: the scene resets `fired`
      and re-anchors the clock on the phase transition, rather than this
      flag being a single lifetime latch for the whole press.
    """

    __slots__ = (
        "start_pos",
        "start_ms",
        "reference_pos",
        "reference_ms",
        "last_pos",
        "fired",
    )

    def __init__(self, pos: tuple[int, int], start_ms: float) -> None:
        self.start_pos = pos
        self.start_ms = start_ms
        self.reference_pos = pos
        self.reference_ms = start_ms
        self.last_pos = pos
        self.fired = False


# Each trial is a brand-new VisualStabilityTraining instance
# (SceneManager constructs a fresh scene() on every switch), so a finger
# held across the trial boundary would otherwise be invisible to the next
# trial: there's no matching _active_touches entry for its later
# FINGERMOTION/FINGERUP events, and pygame has no API to query "what's
# currently touching the screen" at startup. To let a held-through touch
# register at the next trial's fixation point without needing to be
# lifted first, the dying scene stashes its still-active touches here,
# and the new scene's __init__ claims them as already in progress.
#
# This hand-off is only valid for a normal trial-to-trial transition
# *within this stage*. SceneManager.quit() is called on any scene change
# - including the animal leaving mid-trial (-> IDLE), the session ending,
# or switching to a different stage - and in every one of those cases the
# next trial (whenever/wherever it eventually starts) should begin
# completely fresh. _quit() only stashes touches when the trial actually
# reached its own normal conclusion (the result was already resolved to
# something other than PENDING before quit() ran); an externally-forced
# quit() leaves the result at PENDING -> CANCELLED, which is the signal
# used here to suppress the stash.
_pending_touches: dict[int, "_ActiveTouch"] | None = None


class VisualStabilityTraining(Scene):
    level_table: ClassVar[dict[str, list[int]]] = LEVEL_TABLE
    _trials_context_key: ClassVar[str] = "visual_stability_training.trials"

    # region initialization
    def __init__(self) -> None:
        super().__init__()

        self._running = False
        self._is_reaction_window_open = False
        self._reaction_window_started_at_ms: float | None = None

        # Explicit phase model: "blank" (nothing registrable on screen
        # yet, e.g. at construction) -> "fixation" (fixation point
        # visible and touchable) -> "test" (the stimulus/reaction-window
        # phase, once the target is visible).
        self._phase: Literal["blank", "fixation", "test"] = "blank"

        self._mxbiflow = get_mxbiflow()
        self._rewarder = self._mxbiflow.mxbi.rewarder
        self._timer = self._mxbiflow.timer

        self._animal = self._mxbiflow.session.require_current_animal()
        self._config = self._init_config(self._animal, self._mxbiflow)
        self._data_logger = self._init_data_logger(self._mxbiflow, self._animal)
        self._control_buttons = self._init_control_buttons()

        self._init_visual_stimulus(self._config)

        # Active presses, keyed by finger_id (-1 reserved for mouse,
        # GAZE_TOUCH_ID reserved for gaze - see gaze_input.GazeDwellDriver).
        # Claim any touches still held from the previous trial (see
        # _pending_touches) so they're already "down" from frame one,
        # rather than requiring a lift-and-re-press to be noticed.
        self._active_touches: dict[int, _ActiveTouch] = self._claim_pending_touches()

        # Drives this scene's press/motion/release handlers directly from
        # mx-eye gaze samples each frame, as a continuous synthetic touch -
        # in parallel with, not instead of, ordinary mouse/finger events
        # handled in handle_event(). Useful for development/testing without
        # eye-tracking hardware: either input source can complete a trial.
        # already_down=True if a gaze dwell was already claimed above from
        # the previous trial's handed-off _active_touches.
        self._gaze_driver = GazeDwellDriver(
            lambda: self._mxbiflow.eye_sample,
            self,
            already_down=GAZE_TOUCH_ID in self._active_touches,
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
            is_twitter_trial=True,
            screen_size=Screen(screen_size.width, screen_size.height),
            stored=stored_config,
        )

    def _claim_pending_touches(self) -> dict[int, "_ActiveTouch"]:
        global _pending_touches

        touches = _pending_touches
        _pending_touches = None

        if not touches:
            return {}

        # The fixation point is shown immediately (see _on_trial_start),
        # so a held-through touch should start its stability clock
        # fresh, at its current position, right now - not carrying over
        # whatever had already accumulated against the previous trial's
        # target.
        now_ms = monotonic() * 1000
        for active in touches.values():
            active.fired = False
            active.reference_pos = active.last_pos
            active.reference_ms = now_ms

        return touches

    def _config_snapshot_path(self, mxbiflow: "MXBIFlow", animal: "Animal") -> Path:
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
            is_twitter_trial=config.is_twitter_trial,
            result=ResultEnum.PENDING,
            trial_start_time=now,
            trial_end_time=now,
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

    def _init_visual_stimulus(self, config: Config) -> None:
        self._init_backgrounds(config)

        self._sprites: Group[Sprite] = Group()

        target_image = choice((M001, M002))

        screen_size = config.screen_size
        # (0, 0) +------------------------------> x = W
        #        |              Target (3W/4, H/2)
        #        |
        #        |
        #        v
        #      y = H
        target_center = (
            screen_size.width * 3 // 4,
            screen_size.height // 2,
        )

        target_size_px = max(1, round(config.target_size_px))

        self._correct_target = ImageSprite(target_image, target_size_px, target_center)

        fixation_center = (
            screen_size.width * 3 // 4,
            screen_size.height // 2,
        )
        fixation_radius_px = max(1, round(config.fixation_point_radius_px))
        self._fixation_point = FixationPoint(
            fixation_center,
            fixation_radius_px,
            config.fixation_point_inner_radius_ratio,
        )

    def _init_backgrounds(self, config: Config) -> None:
        screen_size = (config.screen_size.width, config.screen_size.height)
        background_color = config.background_color
        self._stimulus_background = create_background(
            screen_size,
            border_width=config.border_width,
            background_color=background_color,
        )
        self._attention_background = create_background(
            screen_size,
            border_width=config.border_width,
            background_color=background_color,
        )
        self._background = self._attention_background

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
        # Capture this before _cancel_unfinished_trial() can overwrite
        # it: a trial that reached its own natural conclusion (correct,
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
        # _claim_pending_touches) instead of losing track of it - but
        # only for a normal trial-to-trial transition within this stage.
        if ended_normally:
            self._stash_pending_touches()
        else:
            self._clear_pending_touches()

        # This scene instance is being torn down either way; any gaze
        # dwell's position/timing state now lives in the stashed (or
        # cleared) _active_touches dict, not in this driver instance, so
        # stop it from acting (e.g. firing a release/abandon call on a
        # future frame that will never come for a dead instance).
        self._gaze_driver.discard()

        self._running = False

    def _stash_pending_touches(self) -> None:
        global _pending_touches
        _pending_touches = self._active_touches or None

    def _clear_pending_touches(self) -> None:
        global _pending_touches
        _pending_touches = None

    def handle_event(self, event: Event) -> None:
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            if any(button.handle_event(event) for button in self._control_buttons):
                return
            self._on_press(-1, event.pos)

        elif event.type == MOUSEBUTTONUP and event.button == 1:
            self._on_release(-1, event.pos)

        elif event.type == MOUSEMOTION and -1 in self._active_touches:
            self._on_motion(-1, event.pos)

        elif event.type == FINGERDOWN:
            # Touchscreen input is actively discouraged now that gaze is
            # the intended response channel: any touch abandons the trial
            # immediately, with no reward - it does not register as a
            # press/dwell at all. Mouse input (MOUSEBUTTONDOWN, above) is
            # left exactly as before for development/testing without a
            # touchscreen or eye tracker attached.
            self._on_touch_penalized()

        # Gaze input does not arrive as a pygame event at all - it is
        # polled directly from mx-eye each frame in update(), via
        # self._gaze_driver (see gaze_input.GazeDwellDriver).

    def update(self, dt_s: float) -> None:
        self._control_buttons.update(dt_s)
        self._gaze_driver.update()
        self._check_pending_touches()

    def draw(self, screen: "Surface") -> None:
        screen.blit(self._background, (0, 0))

        self._sprites.draw(screen)
        self._result_widget.draw(screen)
        for button in self._control_buttons:
            button.layout(screen)
        self._control_buttons.draw(screen)

    # endregion public api

    # region life cycle
    def _on_trial_start(self) -> None:
        self._result.trial_start_time = time()
        self._phase = "fixation"
        self._sprites.add(self._fixation_point)
        self._fixation_shown_at_ms = monotonic() * 1000
        # No timer-based expiry: the fixation phase only ends via a
        # qualifying touch detected anywhere on screen (see
        # _on_fixation_touched), which gives the (short) fixation reward
        # and immediately shows the stimulus. Without such a touch, the
        # trial simply waits here indefinitely - there's no separate
        # "attention_duration_ms" delay any more.

    def _on_fixation_touched(self) -> None:
        self._result.fixation_touch_latency_ms = max(
            0.0, monotonic() * 1000 - self._fixation_shown_at_ms
        )

        reward_duration_ms = self._config.fixation_touch_reward_duration_ms
        if reward_duration_ms > 0:
            self._give_reward(reward_duration_ms)

        self._play_stimulus()

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

        level = self._result.animal.level
        level_correct = 0
        level_incorrect = 0
        level_valid = 0
        for trial in trials.root.values():
            if trial.animal.level != level:
                continue
            if trial.result != ResultEnum.CANCELLED:
                level_valid += 1
            if trial.result is ResultEnum.CORRECT:
                level_correct += 1
            elif trial.result is ResultEnum.INCORRECT:
                level_incorrect += 1

        level_completed = level_correct + level_incorrect
        level_correct_rate = level_correct / level_completed if level_completed else 0

        is_level_up = (
            self._result.result != ResultEnum.CANCELLED
            and self._animal.current_stage.level < self._config.total_levels
            and level_valid > self._config.level_up_trial_count_threshold
            and level_correct_rate >= self._config.level_up_threshold
        )
        if is_level_up:
            self._mxbiflow.session.level_up(
                animal=self._animal.name,
                stage=self._animal.current_stage.stage_name,
            )

        return is_level_up

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
        if is_level_up:
            print(f"Settlement: level up from {trial_level} to {trial_level + 1}")
        else:
            print(f"Settlement: remain at level {trial_level}")
        pprint(self._result)
        print("=" * 60)

    # endregion life cycle

    # region views
    def _show_visual_stimulus(self) -> None:
        self._phase = "test"
        self._reaction_window_started_at_ms = monotonic() * 1000
        self._is_reaction_window_open = True
        self._sprites.remove(self._fixation_point)
        self._sprites.add(
            [
                self._correct_target,
            ]
        )
        # Always schedule a hard close of the trial at
        # reaction_window_duration_ms after stimulus onset. This
        # guarantees the trial eventually proceeds to the next one even
        # if the animal never touches the target at all - there is no
        # per-level enable flag any more; it always applies.
        self._timer.after(
            self._config.reaction_window_duration_ms,
            self._on_response_window_elapsed,
        )

        # A press that is already down (e.g. held continuously from the
        # fixation touch) is eligible to register on the target too:
        # reset its fired state and re-anchor its stability clock to
        # start now, at its current position, rather than carrying over
        # whatever already accumulated against the fixation point.
        now_ms = monotonic() * 1000
        for active in self._active_touches.values():
            active.fired = False
            active.reference_pos = active.last_pos
            active.reference_ms = now_ms

    def _on_response_window_elapsed(self) -> None:
        if not self._is_reaction_window_open:
            return

        self._is_reaction_window_open = False
        self._sprites.remove(
            [
                self._correct_target,
            ]
        )
        self._result.result = ResultEnum.TIMEOUT

        self._on_inter_trial()

    def _result_summary(self) -> str:
        level = self._result.animal.level
        level_correct = 0
        level_incorrect = 0
        correct = 0
        incorrect = 0
        timeout = 0
        cancelled = 0
        reward = 0
        for trial in self._load_trials().root.values():
            if trial.animal.level == level:
                if trial.result is ResultEnum.CORRECT:
                    level_correct += 1
                elif trial.result is ResultEnum.INCORRECT:
                    level_incorrect += 1
            if trial.result is ResultEnum.CORRECT:
                correct += 1
            elif trial.result is ResultEnum.INCORRECT:
                incorrect += 1
            elif trial.result is ResultEnum.TIMEOUT:
                timeout += 1
            elif trial.result is ResultEnum.CANCELLED:
                cancelled += 1
            reward += trial.trial_reward_count + trial.trial_manual_reward_count

        level_completed = level_correct + level_incorrect
        level_correct_rate = level_correct / level_completed if level_completed else 0
        return (
            f"Level {level}/{self._config.total_levels} | "
            f"Animal {self._animal.name} | Trial {self._animal.trial_id} | "
            f"Rate {level_correct_rate:.1%} | "
            f"Correct {correct} | "
            f"Error {incorrect} | "
            f"Reward {reward} | "
            f"Timeout {timeout} | "
            f"Cancelled {cancelled}"
        )

    # endregion views

    # region touch handling helpers
    def _on_press(self, touch_id: int, pos: tuple[int, int]) -> None:
        now_ms = monotonic() * 1000
        self._active_touches[touch_id] = _ActiveTouch(pos, now_ms)
        # A touch could already satisfy the duration rule on the very
        # frame it starts if min_touch_duration_ms is 0; check
        # immediately rather than waiting for the next update() tick.
        self._evaluate_touch(touch_id, self._active_touches[touch_id])

    def _on_motion(self, touch_id: int, pos: tuple[int, int]) -> None:
        active = self._active_touches.get(touch_id)
        if active is None:
            return

        now_ms = monotonic() * 1000
        active.last_pos = pos

        if active.fired:
            # Already confirmed; later motion doesn't unconfirm it (nor
            # does it move the already-recorded fire position).
            return

        if self._exceeds_tolerance(active.reference_pos, pos):
            # Hasn't settled yet: re-anchor the tolerance circle on the
            # new position and restart the stability clock from now.
            active.reference_pos = pos
            active.reference_ms = now_ms
        # else: still within tolerance of the current reference point;
        # keep accumulating time toward the duration threshold.

        self._evaluate_touch(touch_id, active)

    def _on_release(self, touch_id: int, pos: tuple[int, int]) -> None:
        active = self._active_touches.pop(touch_id, None)
        if active is None:
            return

        # Releasing is bookkeeping only (it ends the press and logs it
        # for data completeness) - it does not gate or retroactively
        # change whether this was a valid touch event. That was already
        # decided, continuously, by _evaluate_touch while the touch was
        # active.
        self._result.touch_event.append(
            TouchEvent(x=pos[0], y=pos[1], timestamp=time())
        )

    def _exceeds_tolerance(
        self, reference_pos: tuple[int, int], pos: tuple[int, int]
    ) -> bool:
        tolerance_radius_px = self._config.touch_tolerance_radius_px
        ref_x, ref_y = reference_pos
        dx = pos[0] - ref_x
        dy = pos[1] - ref_y
        return math.hypot(dx, dy) > tolerance_radius_px

    def _check_pending_touches(self) -> None:
        """Called every frame so a touch event can fire as soon as it
        has been held stable long enough, without waiting for a motion
        event or release."""
        for touch_id, active in self._active_touches.items():
            self._evaluate_touch(touch_id, active)

    def _has_active_stimulus(self) -> bool:
        """Whether a touch event is currently allowed to register at
        all - distinct from what's visually on screen. True for
        "fixation" (anywhere on screen counts - see _on_fixation_touched)
        and for "test" while the reaction window is still open (the
        target can be touched; this also covers the period after it has
        already closed - timeout, or an already-resolved trial - so a
        touch resting there can't accumulate stability time or fire
        again). False during "blank" - the moment before the fixation
        point is first placed at trial start - so a touch claimed from a
        previous trial can't accumulate stable time before there's
        actually something to look at."""
        if self._phase == "fixation":
            return True
        if self._phase == "test":
            return self._is_reaction_window_open
        return False

    def _evaluate_touch(self, touch_id: int, active: "_ActiveTouch") -> None:
        if active.fired:
            return

        if not self._has_active_stimulus():
            # Nothing to register a touch event against right now:
            # don't let time accumulate while blank, so the clock only
            # ever reflects stability *in front of* a visible stimulus.
            active.reference_ms = monotonic() * 1000
            return

        stable_duration_ms = monotonic() * 1000 - active.reference_ms
        if stable_duration_ms < self._config.min_touch_duration_ms:
            return

        active.fired = True
        # Fire at the position where the touch actually settled (the
        # current tolerance-circle reference point), not wherever it
        # first landed.
        self._handle_touch_event(active.reference_pos)

    def _handle_touch_event(self, pos: tuple[int, int]) -> None:
        if self._phase == "fixation":
            # Any touch anywhere on screen qualifies during the
            # fixation phase (the goal here is encouraging a stable,
            # engaged posture, not precise fixation-point targeting -
            # see visual_3pt_calibration for that stricter variant).
            self._on_fixation_touched()
            return

        if not self._is_reaction_window_open:
            return

        started_at_ms = self._reaction_window_started_at_ms
        if started_at_ms is None:
            raise RuntimeError("Reaction window start time is unavailable")

        self._is_reaction_window_open = False
        latency_ms = max(0.0, monotonic() * 1000 - started_at_ms)

        # By the time this is called, the touch has already been held
        # stable for at least min_touch_duration_ms (see
        # _evaluate_touch). Unlike visual_3pt_calibration, there is no
        # response-window timing gate here: the goal of this stage is a
        # stable, engaged touch on the target, not precise response
        # timing, so any qualifying touch on the target counts as
        # correct regardless of how long it took.
        if self._is_target_touch(pos):
            self._result.response_latency_ms = latency_ms
            self._on_correct()
        else:
            # An incorrect touch removes the target immediately - only a
            # correct touch keeps it visible, through the reward delay
            # and reward itself (see _on_correct), to reinforce looking
            # at the marmoset specifically as what earns the reward.
            self._sprites.remove(
                [
                    self._correct_target,
                ]
            )
            self._on_incorrect()

    def _is_target_touch(self, pos: tuple[int, int]) -> bool:
        rect = self._correct_target.rect
        if rect is None:
            return False

        buffer_px = round(self._config.touch_buffer_px)
        hit_box = rect.inflate(buffer_px, buffer_px)
        return hit_box.collidepoint(pos)

    # endregion touch handling helpers

    # region events handler
    def _on_correct(self):
        self._result.result = ResultEnum.CORRECT

        # The marmoset stays visible throughout: first for
        # reward_delay_ms on its own (the "look at it before you're
        # rewarded" teaching gap), then for a further reward_duration_ms
        # while the reward itself is administered. Total time the
        # marmoset remains on screen after a correct touch is therefore
        # reward_delay_ms + reward_duration_ms. _give_reward() is a
        # fire-and-forget call to the reward hardware (it doesn't block
        # the game loop), so "keeping the marmoset flashing while the
        # reward is being given" just means not removing its sprite
        # until that second timer elapses.
        reward_delay_ms = self._config.reward_delay_ms
        if reward_delay_ms > 0:
            self._timer.after(reward_delay_ms, self._deliver_reward_and_flash)
        else:
            self._deliver_reward_and_flash()

    def _deliver_reward_and_flash(self) -> None:
        self._give_reward()
        self._timer.after(self._config.reward_duration_ms, self._end_correct_trial)

    def _end_correct_trial(self) -> None:
        self._sprites.remove(
            [
                self._correct_target,
            ]
        )
        self._on_inter_trial()

    def _on_incorrect(self):
        self._result.result = ResultEnum.INCORRECT

        self._on_inter_trial()

    def _cancel_unfinished_trial(self) -> None:
        if self._result.result != ResultEnum.PENDING:
            return

        self._result.result = ResultEnum.CANCELLED

    def _request_game_quit(self) -> None:
        event.post(Event(QUIT))

    def on_gaze_abandoned(self) -> None:
        """Called by GazeDwellDriver when gaze has been lost for longer
        than its grace period. This trial is given up on - not resolved as
        correct, incorrect, or timeout, since none of those actually
        happened - the same way quit() already treats any other
        externally-forced end of trial (_cancel_unfinished_trial marks an
        unresolved PENDING result as CANCELLED). This does not imply the
        animal has left the session; only this one trial ends early.
        """
        if self._running:
            self.quit()

    def _on_touch_penalized(self) -> None:
        """A touchscreen touch (not mouse) was detected. Touch is actively
        discouraged now that gaze is the intended response channel: give
        no reward and abandon this trial immediately (CANCELLED, same as
        on_gaze_abandoned), restarting fresh at the next trial's normal
        start rather than resolving as correct/incorrect/timeout."""
        if self._running:
            self.quit()

    # endregion events handler

    # region stimulus and reward
    def _play_stimulus(self) -> None:
        self._background = self._stimulus_background
        self._show_visual_stimulus()

    def _give_reward(self, duration_ms: int | None = None) -> None:
        self._rewarder.give_reward(duration_ms or self._config.reward_duration_ms)
        self._result.trial_reward_count += 1

    def _give_manual_reward(self) -> None:
        self._give_reward()
        self._result.trial_manual_reward_count += 1

    # endregion stimulus and reward

    # region utils
    # endregion utils