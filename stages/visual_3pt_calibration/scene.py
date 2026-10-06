import math
from pathlib import Path
from random import choice, randint
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

# Gaze-tracking input is not wired up yet. It will be a custom pygame event
# posted by an external gaze process/driver, treated as a pointer-down event
# mirroring the payload shape produced for FINGERDOWN (normalized x/y in
# [0, 1]).
#
# TODO(gaze): once the gaze driver exists, allocate the event type with
# pygame.event.custom_type() (NOT a hardcoded pygame.USEREVENT + N offset).
# mxbiflow itself reserves USEREVENT + 1 for its own EVT_DETECTOR event (see
# mxbiflow.gameloop.detector_bridge); a hardcoded offset here previously
# collided with it, causing this stage to misinterpret detector/shortcut
# events as gaze clicks and crash on a missing x/y attribute.
# GAZEBUTTONDOWN = pygame.event.custom_type()


class FixationPoint(Sprite):
    """A green circle with a small black circle in its center.

    Used as a static/decorative fixation cue during the attention screen.
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
      clock restarts from now. A finger that drifts steadily across the
      screen therefore never accumulates enough time anywhere and never
      fires, even though each individual step is small.
    - `fired` becomes True the instant accumulated stable time reaches
      `min_touch_duration_ms`, which resolves one touch event against
      whatever is currently on screen (the fixation point, or the
      target) at `reference_pos`. A press held continuously across a
      phase change (e.g. from the fixation point through to the target
      appearing) is eligible to fire again for the new phase: the scene
      resets `fired` and re-anchors the clock on the phase transition
      (see `_show_visual_stimulus`), rather than this flag being a single
      lifetime latch for the whole press.
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


# Each trial is a brand-new Visual3ptCalibration instance (SceneManager
# constructs a fresh scene() on every switch), so a finger held across the
# trial boundary would otherwise be invisible to the next trial: there's
# no matching _active_touches entry for its later FINGERMOTION/FINGERUP
# events, and pygame has no API to query "what's currently touching the
# screen" at startup. To let a held-through touch register at the next
# trial's fixation point without needing to be lifted first, the dying
# scene stashes its still-active touches here, and the new scene's
# __init__ claims them as already in progress.
#
# This hand-off is only valid for a normal trial-to-trial transition
# *within this stage*. SceneManager.quit() is called on any scene change -
# including the animal leaving mid-trial (-> IDLE), the session ending, or
# switching to a different stage - and in every one of those cases the
# next trial (whenever/wherever it eventually starts) should begin
# completely fresh. _quit() only stashes touches when the trial actually
# reached its own normal conclusion (the result was already resolved to
# something other than PENDING before quit() ran); an externally-forced
# quit() leaves the result at PENDING -> CANCELLED, which is the signal
# used here to suppress the stash.
_pending_touches: dict[int, "_ActiveTouch"] | None = None


class Visual3ptCalibration(Scene):
    level_table: ClassVar[dict[str, list[int]]] = LEVEL_TABLE
    _trials_context_key: ClassVar[str] = "visual_3pt_calibration.trials"

    # region initialization
    def __init__(self) -> None:
        super().__init__()

        self._running = False
        self._is_reaction_window_open = False
        self._reaction_window_started_at_ms: float | None = None

        # Explicit phase model, following VisualTracing but with an
        # extra "blank" phase: "fixation" (fixation point visible) ->
        # "blank" (nothing on screen - the randomized gap between the
        # fixation touch and the stimulus appearing) -> "test" (the
        # stimulus/reaction-window phase, once the target is actually
        # visible). "blank" also covers the moment between scene
        # construction and the fixation point actually being placed on
        # screen (_on_trial_start), so a touch claimed from a previous
        # trial (see _claim_pending_touches) can't accumulate stable
        # time before there's anything to look at. Replaces the
        # previous _is_fixation_phase_open boolean.
        self._phase: Literal["blank", "fixation", "test"] = "blank"

        self._mxbiflow = get_mxbiflow()
        self._rewarder = self._mxbiflow.mxbi.rewarder
        self._timer = self._mxbiflow.timer

        self._animal = self._mxbiflow.session.require_current_animal()
        self._config = self._init_config(self._animal, self._mxbiflow)
        self._data_logger = self._init_data_logger(self._mxbiflow, self._animal)
        self._control_buttons = self._init_control_buttons()

        self._init_visual_stimulus(self._config)

        # Active presses, keyed by finger_id (-1 reserved for mouse).
        # Claim any touches still held from the previous trial (see
        # _pending_touches) so they're already "down" from frame one,
        # rather than requiring a lift-and-re-press to be noticed.
        self._active_touches: dict[int, _ActiveTouch] = self._claim_pending_touches()

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
        # Fixation point sits where the single target used to be in
        # visual_touchscreen_training (flushed right, at 3/4 screen
        # width), rather than dead-center.
        fixation_center = (
            screen_size.width * 3 // 4,
            screen_size.height // 2,
        )

        # Three candidate positions form a triangle around the fixation
        # point. The physical screen is mounted rotated 90deg
        # counter-clockwise from its normal orientation (so content
        # drawn toward the top of this unrotated coordinate space ends
        # up appearing at the physical LEFT edge once displayed) -
        # these three positions and their labels describe where each
        # one actually appears to the animal on the rotated screen, not
        # where they sit in this method's own (x, y) coordinate space:
        #   - computed as directly above the fixation point in (x, y)
        #     -> appears to the physical LEFT on the rotated screen
        #   - computed as directly below the fixation point in (x, y)
        #     -> appears to the physical RIGHT on the rotated screen
        #   - computed as directly to the right of the fixation point
        #     in (x, y) -> appears ABOVE on the rotated screen (this one
        #     is unaffected by the top<->left swap, since "right"
        #     rotates to "above", matching its own label here)
        # Distance from the fixation point is configurable via
        # config.target_offset_cm (a physical distance converted to
        # pixels via pixels_per_cm, config.target_distances_px). Note:
        # since the fixation point sits at 3/4 screen width, a large
        # target_offset_cm combined with a large target_size_cm could
        # push the (x, y)-rightward position off-screen - keep
        # target_offset_cm modest for your display size if you enlarge
        # the target.
        offset_px = round(config.target_distances_px // 2)
        target_positions = (
            (fixation_center[0] - offset_px, fixation_center[1] - offset_px),  # left
            (fixation_center[0] - offset_px, fixation_center[1] + offset_px),  # right
            (fixation_center[0] + offset_px // 2, fixation_center[1]),  # above
        )
        target_center = choice(target_positions)

        target_size_px = max(1, round(config.target_size_px))

        self._correct_target = ImageSprite(target_image, target_size_px, target_center)

        self._fixation_point = FixationPoint(
            fixation_center,
            config.fixation_point_radius,
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
            self._on_press(event.finger_id, self._finger_pos(event))

        elif event.type == FINGERUP:
            self._on_release(event.finger_id, self._finger_pos(event))

        elif event.type == FINGERMOTION:
            if event.finger_id in self._active_touches:
                self._on_motion(event.finger_id, self._finger_pos(event))

        # TODO(gaze): re-enable once GAZEBUTTONDOWN/UP/MOTION are
        # allocated (see note above the constant) and a gaze driver
        # actually posts them. Gaze dwell start/motion/end can reuse
        # _on_press/_on_motion/_on_release exactly as mouse/finger do
        # above, keyed by a dedicated gaze touch_id - the per-frame
        # stability check (_check_pending_touches) already fires on its
        # own regardless of release, so a sustained gaze dwell (which
        # has no physical "up") naturally progresses the game the same
        # way a held touch does.

    def update(self, dt_s: float) -> None:
        self._control_buttons.update(dt_s)
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
        # The fixation phase has no timer-based expiry of its own: it
        # only ends when a qualifying touch lands within
        # fixation_touch_reward_radius_cm of the fixation point (see
        # _on_fixation_touched), which gives the (short) fixation reward
        # and then schedules the stimulus to appear after a randomized
        # delay (fixation_duration_min_ms/max_ms). Without such a touch,
        # the trial simply waits here indefinitely.

    def _on_fixation_touched(self) -> None:
        self._result.fixation_touch_latency_ms = max(
            0.0, monotonic() * 1000 - self._fixation_shown_at_ms
        )

        reward_duration_ms = self._config.fixation_touch_reward_duration_ms
        if reward_duration_ms > 0:
            self._give_reward(reward_duration_ms)

        # The stimulus doesn't appear immediately on a qualifying
        # fixation touch - it's delayed by a randomized amount
        # (fixation_duration_min_ms/max_ms). The fixation point stays
        # visible on screen for the whole delay (it's only actually
        # removed once the stimulus appears, in _show_visual_stimulus) -
        # but the phase still flips to "blank" so touch registration
        # behaves as if nothing is there: a touch held through this gap
        # doesn't accumulate stable time, and its 100ms clock only truly
        # starts once the target actually appears
        # (_show_visual_stimulus re-anchors every active touch). This
        # means a continued fixation-point touch during the delay is
        # visually reassuring but not itself scoreable - only the
        # target, once shown, can satisfy the duration requirement.
        self._phase = "blank"
        fixation_duration_ms = randint(
            self._config.fixation_duration_min_ms,
            self._config.fixation_duration_max_ms,
        )
        self._timer.after(fixation_duration_ms, self._play_stimulus)

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
        # reaction_window_duration_ms after stimulus onset (unconditional
        # - no enable flag). This guarantees the trial eventually
        # proceeds to the next one even if the animal never touches, or
        # touches outside the scored response window
        # (response_window_min_ms..response_window_max_ms, see
        # _handle_touch_event). A touch landing after
        # response_window_max_ms but before this fires is simply not
        # scored as correct - the window stays open for a possible
        # later touch until this timer closes it.
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
    def _finger_pos(self, event: Event) -> tuple[int, int]:
        screen = self._mxbiflow.mxbi.screen_size
        return (
            int(event.x * screen.width),
            int(event.y * screen.height),
        )

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
        "fixation" (the fixation point can be touched) and for "test"
        while the reaction window is still open (the target can be
        touched; this also covers the period after it has already
        closed - timeout, or an already-resolved trial - so a touch
        resting there can't accumulate stability time or fire again).
        False during "blank": the randomized gap between a qualifying
        fixation touch and the stimulus appearing, and the moment
        before the fixation point is first placed at trial start. The
        fixation point itself stays visible on screen throughout
        "blank" (it's only actually removed once the target appears,
        in _show_visual_stimulus) - but nothing can register against it
        during the gap, so a touch held continuously through it doesn't
        get a head start on its 100ms clock before the target actually
        appears."""
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
            radius_px = self._config.fixation_touch_reward_radius_px
            fixation_center = self._fixation_point.rect.center
            distance_px = math.hypot(
                pos[0] - fixation_center[0], pos[1] - fixation_center[1]
            )
            if distance_px <= radius_px:
                self._on_fixation_touched()
            return

        if not self._is_reaction_window_open:
            return

        started_at_ms = self._reaction_window_started_at_ms
        if started_at_ms is None:
            raise RuntimeError("Reaction window start time is unavailable")

        self._is_reaction_window_open = False
        self._sprites.remove(
            [
                self._correct_target,
            ]
        )

        latency_ms = max(0.0, monotonic() * 1000 - started_at_ms)
        is_on_target = self._is_target_touch(pos)
        is_in_response_window = (
            self._config.response_window_min_ms
            <= latency_ms
            <= self._config.response_window_max_ms
        )

        # By the time this is called, the touch has already been held
        # stable for at least min_touch_duration_ms (see
        # _evaluate_touch) - that part of the gate is implicit here. Any
        # qualifying touch at all - on target or off - ends the trial
        # immediately. Only the complete absence of one waits out
        # reaction_window_duration_ms (_on_response_window_elapsed).
        #
        # Classification:
        #   on target  + within [min,max] response window -> CORRECT
        #   on target  + outside the response window        -> TIMEOUT
        #     (an on-target touch that's mistimed is read as the animal
        #     anticipating/guessing rather than a genuine miss, so it's
        #     excluded from the correct-rate rather than counted as an
        #     error)
        #   off target (any timing)                         -> ERROR
        if not is_on_target:
            self._on_incorrect()
        elif is_in_response_window:
            self._result.response_latency_ms = latency_ms
            self._on_correct()
        else:
            self._on_timeout()

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

        if self._config.target_touch_reward_enabled:
            self._give_reward()
        self._on_inter_trial()

    def _on_incorrect(self):
        self._result.result = ResultEnum.INCORRECT

        self._on_inter_trial()

    def _on_timeout(self):
        self._result.result = ResultEnum.TIMEOUT

        self._on_inter_trial()

    def _cancel_unfinished_trial(self) -> None:
        if self._result.result != ResultEnum.PENDING:
            return

        self._result.result = ResultEnum.CANCELLED

    def _request_game_quit(self) -> None:
        event.post(Event(QUIT))

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