from pathlib import Path
from random import choice, randint
from shutil import copy2
from time import monotonic, time
from typing import TYPE_CHECKING, ClassVar

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
    FINGERUP,
    MOUSEBUTTONDOWN,
    MOUSEBUTTONUP,
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


class Visual3ptCalibration(Scene):
    level_table: ClassVar[dict[str, list[int]]] = LEVEL_TABLE
    _trials_context_key: ClassVar[str] = "visual_3pt_calibration.trials"

    # region initialization
    def __init__(self) -> None:
        super().__init__()

        self._running = False
        self._is_reaction_window_open = False
        self._reaction_window_started_at_ms: float | None = None

        # Tracks the monotonic timestamp (ms) of the most recent unresolved
        # press, so we can measure press duration once the matching "up"
        # event arrives. Pointer-like inputs (mouse/finger/gaze) are all
        # single-touch in this stage, so one pending press at a time
        # suffices.
        self._pending_touch_started_at_ms: float | None = None
        self._pending_touch_pos: tuple[int, int] | None = None

        self._mxbiflow = get_mxbiflow()
        self._rewarder = self._mxbiflow.mxbi.rewarder
        self._timer = self._mxbiflow.timer

        self._animal = self._mxbiflow.session.require_current_animal()
        self._config = self._init_config(self._animal, self._mxbiflow)
        self._data_logger = self._init_data_logger(self._mxbiflow, self._animal)
        self._control_buttons = self._init_control_buttons()

        self._init_visual_stimulus(self._config)

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
        # point: directly above, directly below, and directly to the
        # right. Distance from the fixation point is configurable via
        # config.target_offset_ratio (a fraction of screen width). Note:
        # since the fixation point sits at 3/4 screen width, a large
        # target_offset_ratio combined with a large target_size_cm could
        # push the "right" position off-screen - keep offset_ratio
        # modest (the 0.2 default leaves headroom) if you enlarge the
        # target.
        offset_px = round(screen_size.width * config.target_offset_ratio)
        target_positions = (
            (fixation_center[0], fixation_center[1] - offset_px),  # above
            (fixation_center[0], fixation_center[1] + offset_px),  # below
            (fixation_center[0] + offset_px, fixation_center[1]),  # right
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
        self._cancel_unfinished_trial()
        self._result.trial_end_time = time()

        is_level_up = self._settlement()
        self._print_result(is_level_up)

        self._data_logger.save(self._result.model_dump())
        self._timer.cancel_all()

        self._running = False

    def handle_event(self, event: Event) -> None:

        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            if any(button.handle_event(event) for button in self._control_buttons):
                return
            self._on_touch_down(event.pos)

        if event.type == MOUSEBUTTONUP and event.button == 1:
            self._on_touch_up(event.pos)

        if event.type == FINGERDOWN:
            screen = self._mxbiflow.mxbi.screen_size
            pos = (
                int(event.x * screen.width),
                int(event.y * screen.height),
            )
            self._on_touch_down(pos)

        if event.type == FINGERUP:
            screen = self._mxbiflow.mxbi.screen_size
            pos = (
                int(event.x * screen.width),
                int(event.y * screen.height),
            )
            self._on_touch_up(pos)

        # TODO(gaze): re-enable once GAZEBUTTONDOWN is allocated (see note
        # above the constant) and a gaze driver actually posts it. Gaze
        # input would need its own down/up (dwell start/end) pairing to
        # participate in the same minimum-duration gating as mouse/finger.
        # if event.type == GAZEBUTTONDOWN:
        #     screen = self._mxbiflow.mxbi.screen_size
        #     gaze_event = Event(
        #         MOUSEBUTTONDOWN,
        #         button=1,
        #         pos=(
        #             int(event.x * screen.width),
        #             int(event.y * screen.height),
        #         ),
        #     )
        #     self._on_touched(gaze_event)

    def update(self, dt_s: float) -> None:
        self._control_buttons.update(dt_s)

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
        self._sprites.add(self._fixation_point)
        fixation_duration_ms = randint(
            self._config.fixation_duration_min_ms,
            self._config.fixation_duration_max_ms,
        )
        self._timer.after(
            fixation_duration_ms,
            self._play_stimulus,
        )

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
        # _on_touched). A touch landing after response_window_max_ms but
        # before this fires is simply not scored as correct - the window
        # stays open for a possible later touch until this timer closes
        # it.
        self._timer.after(
            self._config.reaction_window_duration_ms,
            self._on_response_window_elapsed,
        )

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

    # region events handler
    def _on_touch_down(self, pos: tuple[int, int]) -> None:
        self._pending_touch_started_at_ms = monotonic() * 1000
        self._pending_touch_pos = pos

    def _on_touch_up(self, pos: tuple[int, int]) -> None:
        started_at_ms = self._pending_touch_started_at_ms
        down_pos = self._pending_touch_pos
        self._pending_touch_started_at_ms = None
        self._pending_touch_pos = None

        if started_at_ms is None or down_pos is None:
            # An "up" with no matching "down" seen by this stage (e.g. the
            # press began before this scene took over, or was consumed by
            # a control button) - nothing to score.
            return

        duration_ms = max(0.0, monotonic() * 1000 - started_at_ms)

        self._result.touch_event.append(
            TouchEvent(
                x=down_pos[0],
                y=down_pos[1],
                timestamp=time(),
            )
        )

        self._on_touched(down_pos, duration_ms)

    def _on_touched(self, pos: tuple[int, int], duration_ms: float) -> None:
        if not self._is_reaction_window_open:
            return

        started_at_ms = self._reaction_window_started_at_ms
        if started_at_ms is None:
            raise RuntimeError("Reaction window start time is unavailable")

        latency_ms = max(0.0, monotonic() * 1000 - started_at_ms)
        is_on_target = self._classify_touch(pos) is ResultEnum.CORRECT
        has_min_duration = duration_ms >= self._config.min_touch_duration_ms
        is_in_response_window = (
            self._config.response_window_min_ms
            <= latency_ms
            <= self._config.response_window_max_ms
        )

        # Any touch at all - on target or off, long enough or not - ends
        # the trial immediately. Only the complete absence of a touch
        # waits out reaction_window_duration_ms
        # (_on_response_window_elapsed).
        #
        # Classification:
        #   on target  + long enough + within [min,max] window -> CORRECT
        #   on target  + (too short OR outside the window)      -> TIMEOUT
        #     (an on-target touch that's mistimed or too brief is read as
        #     the animal anticipating/guessing rather than a genuine
        #     miss, so it's excluded from the correct-rate rather than
        #     counted as an error)
        #   off target (any duration/timing)                    -> ERROR
        if not is_on_target:
            result = ResultEnum.INCORRECT
        elif has_min_duration and is_in_response_window:
            result = ResultEnum.CORRECT
        else:
            result = ResultEnum.TIMEOUT

        self._is_reaction_window_open = False
        self._sprites.remove(
            [
                self._correct_target,
            ]
        )

        if result is ResultEnum.CORRECT:
            self._result.response_latency_ms = latency_ms
            self._on_correct()
        elif result is ResultEnum.INCORRECT:
            self._on_incorrect()
        else:
            self._on_timeout()

    def _classify_touch(self, pos: tuple[int, int]) -> ResultEnum:
        rect = self._correct_target.rect
        if rect is None:
            return ResultEnum.BACKGROUND_TOUCHED

        buffer_px = round(self._config.touch_buffer_px)
        hit_box = rect.inflate(buffer_px, buffer_px)

        if hit_box.collidepoint(pos):
            return ResultEnum.CORRECT

        return ResultEnum.BACKGROUND_TOUCHED

    def _on_correct(self):
        self._result.result = ResultEnum.CORRECT

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
