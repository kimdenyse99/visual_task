"""Gaze calibration at three fixation positions, with an immediate reward.

Trial flow (one scene instance per trial, like every other stage here):

1. fixation  - a fixation point appears at one of three positions, chosen
               at random. Gaze that dwells on the point or within
               fixation_acceptance_radius_cm of its perimeter for
               min_touch_duration_ms,
               with the usual dwell_grace_ms tolerance, counts as fixation.
               If that doesn't happen within fixation_timeout_ms the trial
               is a TIMEOUT and skips straight to step 3.
2. reward    - on fixation the reward is given immediately, the fixation
               point disappears and a random marmoset GIF plays in the
               same spot for post_reward_stimulus_duration_ms (overlapping
               the reward). The trial is CORRECT.
3. blank     - inter_trial_duration_ms of empty screen, then the next
               trial's fixation point appears.

Compared to visual_3pt_calibration (v1): the fixation point itself moves
between the three positions instead of staying put, gaze must actually be
on/near it (v1 accepted gaze anywhere), there is no randomized delay
between fixation and reward, and the fixation phase can time out.
"""

import math
from pathlib import Path
from random import choice
from shutil import copy2
from time import monotonic, time
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

import pygame
from loguru import logger
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
    Event,
    Surface,
    event,
)
from pygame.draw import circle as draw_circle
from pygame.draw import line as draw_line
from pygame.sprite import Group, Sprite
from rich.pretty import pprint

from animated_gif_sprite import AnimatedGifSprite
from assets import MARMOSET_IMAGE_PATHS
from dwell import DwellTracker
from gaze_input import GAZE_TOUCH_ID, GazeDwellDriver, GazeSample
from models import TouchEvent
from stimulus_events import StimulusEvents
from trial_events import write_target_shown_event

from .models import (
    LEVEL_TABLE,
    PATH,
    Config,
    PositionName,
    Result,
    ResultEnum,
    Screen,
    StoredConfigs,
    TrialResults,
)

if TYPE_CHECKING:
    from mxbiflow.core import MXBIFlow
    from mxbiflow.models import Animal
    from pygame.surface import Surface as SurfaceType

type Phase = Literal["blank", "fixation", "reward"]


class _CancellableTask(Protocol):
    """What FrameTimer.after() returns (its own type is private)."""

    def cancel(self) -> None: ...


class FixationPoint(Sprite):
    """A green circle with a small black circle in its center."""

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


# Touch/gaze hand-off between consecutive trials of this stage - same
# mechanism and rules as visual_3pt_calibration's _pending_touch_positions:
# only stashed when a trial ends normally, so an externally forced quit
# (animal left, session ended, stage changed) starts the next trial fresh.
_pending_touch_positions: dict[int, tuple[int, int]] | None = None


class Visual3ptCalibrationV2(Scene):
    level_table: ClassVar[dict[str, list[int]]] = LEVEL_TABLE
    _trials_context_key: ClassVar[str] = "visual_3pt_calibration_v2.trials"
    _trace_color: ClassVar[tuple[int, int, int]] = (220, 0, 0)

    @classmethod
    def name(cls) -> str:
        # The default (inflection.underscore of the class name) would give
        # "visual3pt_calibration_v2"; use the same spelling as the package.
        return "visual_3pt_calibration_v2"

    # region initialization
    def __init__(self) -> None:
        super().__init__()

        self._running = False
        self._phase: Phase = "blank"

        self._mxbiflow = get_mxbiflow()
        self._stimulus_events = StimulusEvents(self._mxbiflow)
        self._rewarder = self._mxbiflow.mxbi.rewarder
        self._timer = self._mxbiflow.timer
        self._fixation_timeout_task: _CancellableTask | None = None
        self._fixation_shown_at_ms = 0.0

        self._animal = self._mxbiflow.session.require_current_animal()
        self._config = self._init_config(self._animal, self._mxbiflow)
        self._data_logger = self._init_data_logger(self._mxbiflow, self._animal)
        self._control_buttons = self._init_control_buttons()

        self._init_visual_stimulus(self._config)

        # One DwellTracker per active input (-1 mouse, GAZE_TOUCH_ID gaze).
        self._trackers: dict[int, DwellTracker] = {}
        self._trajectories: dict[int, list[tuple[int, int, float]]] = {}
        self._claim_pending_touches()

        self._trace_points: list[tuple[int, int, float]] = []

        def _get_gaze_sample() -> GazeSample | None:
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

    def _init_config(self, animal: Animal, mxbiflow: MXBIFlow) -> Config:
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

    def _claim_pending_touches(self) -> None:
        global _pending_touch_positions

        positions = _pending_touch_positions
        _pending_touch_positions = None

        if not positions:
            return

        # Gaze is polled rather than event-driven and can vanish between
        # trials, so only reinstate it if a fresh sample exists right now
        # (see visual_3pt_calibration._claim_pending_touches).
        current_gaze_sample = self._mxbiflow.eye_sample
        for touch_id, pos in positions.items():
            if touch_id == GAZE_TOUCH_ID and current_gaze_sample is None:
                continue
            tracker = DwellTracker(touch_id=touch_id)
            tracker.last_pos = pos
            self._trackers[touch_id] = tracker
            self._trajectories[touch_id] = []

    def _config_snapshot_path(self, mxbiflow: MXBIFlow, animal: Animal) -> Path:
        return (
            mxbiflow.session.absolute_animal_data_path(animal.name)
            / self.name()
            / "config.json"
        )

    # endregion config

    # region data

    def _init_result(self, animal: Animal, config: Config) -> Result:
        now = time()

        return Result(
            animal=animal.base_info,
            is_twitter_trial=config.is_twitter_trial,
            result=ResultEnum.PENDING,
            trial_start_time=now,
            trial_end_time=now,
            fixation_position=self._position_name,
            fixation_center=self._fixation_center,
        )

    def _init_data_logger(self, mxbiflow: MXBIFlow, animal: Animal) -> DataLogger:
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

    @staticmethod
    def _fixation_positions(config: Config) -> dict[PositionName, tuple[int, int]]:
        """The three candidate fixation positions, in pygame (x, y) pixels.

        Same geometry as visual_3pt_calibration's target positions, so data
        from both stages lines up: an anchor at (3/4 W, 1/2 H) plus two
        points offset by half of position_distance_cm on each axis. The
        screen is mounted rotated 90 degrees counter-clockwise, so the
        labels describe where each point appears to the animal, not its
        direction in (x, y).
        """
        screen_size = config.screen_size
        anchor = (screen_size.width * 3 // 4, screen_size.height // 2)
        offset_px = round(config.position_distance_px // 2)
        return {
            "left": (anchor[0] - offset_px, anchor[1] - offset_px),
            "right": (anchor[0] - offset_px, anchor[1] + offset_px),
            "center": anchor,
        }

    def _init_visual_stimulus(self, config: Config) -> None:
        self._init_background(config)

        self._sprites: Group[Sprite] = Group()

        positions = self._fixation_positions(config)
        self._position_name: PositionName = choice(list(positions))
        self._fixation_center = positions[self._position_name]

        self._fixation_point = FixationPoint(
            self._fixation_center,
            round(config.fixation_point_radius_px),
            config.fixation_point_inner_radius_ratio,
        )

        # The GIF shown after fixation plays in the fixation point's spot.
        target_image = choice(MARMOSET_IMAGE_PATHS)
        self._target_asset_name = target_image.name
        self._target = AnimatedGifSprite(
            target_image,
            max(1, round(config.target_size_px)),
            self._fixation_center,
        )

    def _init_background(self, config: Config) -> None:
        self._background = create_background(
            (config.screen_size.width, config.screen_size.height),
            border_width=config.border_width,
            background_color=config.background_color,
        )

    # endregion stimulus

    # region console

    def _print_trial_config(self) -> None:
        print()
        print("=" * 60)
        print(
            f"Beginning level {self._animal.current_stage.level}, "
            f"trial {self._animal.trial_id}, "
            f"for monkey {self._animal.name}, "
            f"in the {self.name()} stage, "
            f"fixation at {self._position_name} {self._fixation_center}"
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
        self._stimulus_events.clear()
        ended_normally = self._result.result != ResultEnum.PENDING

        self._cancel_unfinished_trial()
        self._result.trial_end_time = time()

        is_level_up = self._settlement()
        self._print_result(is_level_up)

        self._data_logger.save(self._result.model_dump())
        self._timer.cancel_all()

        if ended_normally:
            self._stash_pending_touches()
        else:
            self._clear_pending_touches()

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

    def handle_event(self, event: Event) -> None:
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            if any(button.handle_event(event) for button in self._control_buttons):
                return
            self._on_press(-1, event.pos)

        elif event.type == MOUSEBUTTONUP and event.button == 1:
            self._on_release(-1, event.pos)

        elif event.type == MOUSEMOTION and -1 in self._trackers:
            self._on_motion(-1, event.pos)

        elif event.type == FINGERDOWN:
            # Gaze is the response channel; a touchscreen touch abandons
            # the trial (mouse input stays available for testing).
            self._on_touch_penalized()

    def update(self, dt_s: float) -> None:
        self._control_buttons.update(dt_s)
        self._gaze_driver.update()
        self._check_all_trackers()
        self._prune_trace()
        # Only advance the GIF while it is visible, so it always starts
        # from its first frame.
        if self._phase == "reward":
            self._target.update(dt_s)

    def draw(self, screen: SurfaceType) -> None:
        screen.blit(self._background, (0, 0))

        self._sprites.draw(screen)
        self._stimulus_events.present(
            self._sprites, self._fixation_point, self._target,
            self._target_asset_name,
        )
        if self._config.show_gaze_trace:
            self._draw_trace(screen)
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
        # Dwell progress only counts from the moment the point is visible.
        for tracker in self._trackers.values():
            tracker.reset()
        self._fixation_timeout_task = self._timer.after(
            self._config.fixation_timeout_ms,
            self._on_fixation_timeout,
        )

    def _on_fixation_touched(self) -> None:
        if self._phase != "fixation":
            return

        if self._fixation_timeout_task is not None:
            self._fixation_timeout_task.cancel()
            self._fixation_timeout_task = None

        self._result.fixation_touch_latency_ms = max(
            0.0, monotonic() * 1000 - self._fixation_shown_at_ms
        )
        self._result.result = ResultEnum.CORRECT
        self._result.stimulus_asset = self._target_asset_name

        # Reward and GIF start together, with no delay.
        self._give_reward()
        self._show_target()
        logger.info(
            "Fixation at {} after {:.0f} ms; GIF {} for {} ms.",
            self._position_name,
            self._result.fixation_touch_latency_ms,
            self._target_asset_name,
            self._config.post_reward_stimulus_duration_ms,
        )
        self._timer.after(
            self._config.post_reward_stimulus_duration_ms,
            self._end_reward_phase,
        )

    def _on_fixation_timeout(self) -> None:
        self._fixation_timeout_task = None
        if self._phase != "fixation":
            return

        logger.info(
            "No fixation at {} within {} ms; timeout.",
            self._position_name,
            self._config.fixation_timeout_ms,
        )
        self._result.result = ResultEnum.TIMEOUT
        self._sprites.remove(self._fixation_point)
        self._on_inter_trial()

    def _end_reward_phase(self) -> None:
        self._sprites.remove(self._target)
        self._on_inter_trial()

    def _on_inter_trial(self) -> None:
        self._phase = "blank"
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
        level_timeout = 0
        level_valid = 0
        for trial in trials.root.values():
            if trial.animal.level != level:
                continue
            if trial.result != ResultEnum.CANCELLED:
                level_valid += 1
            if trial.result is ResultEnum.CORRECT:
                level_correct += 1
            elif trial.result is ResultEnum.TIMEOUT:
                level_timeout += 1

        # A timeout is this stage's only failure outcome, so it counts
        # against the level's correct rate.
        level_completed = level_correct + level_timeout
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
    def _show_target(self) -> None:
        self._phase = "reward"
        self._sprites.remove(self._fixation_point)
        self._sprites.add(self._target)

        assert self._target.rect is not None
        target_center_px = self._target.rect.center
        write_target_shown_event(
            self._mxbiflow,
            self._animal,
            self.name(),
            self._animal.trial_id,
            (round(target_center_px[0]), round(target_center_px[1])),
        )

    def _result_summary(self) -> str:
        level = self._result.animal.level
        level_correct = 0
        level_timeout = 0
        correct = 0
        timeout = 0
        cancelled = 0
        reward = 0
        for trial in self._load_trials().root.values():
            if trial.animal.level == level:
                if trial.result is ResultEnum.CORRECT:
                    level_correct += 1
                elif trial.result is ResultEnum.TIMEOUT:
                    level_timeout += 1
            if trial.result is ResultEnum.CORRECT:
                correct += 1
            elif trial.result is ResultEnum.TIMEOUT:
                timeout += 1
            elif trial.result is ResultEnum.CANCELLED:
                cancelled += 1
            reward += trial.trial_reward_count + trial.trial_manual_reward_count

        level_completed = level_correct + level_timeout
        level_correct_rate = level_correct / level_completed if level_completed else 0
        return (
            f"Level {level}/{self._config.total_levels} | "
            f"Animal {self._animal.name} | Trial {self._animal.trial_id} | "
            f"Rate {level_correct_rate:.1%} | "
            f"Correct {correct} | "
            f"Timeout {timeout} | "
            f"Reward {reward} | "
            f"Cancelled {cancelled}"
        )

    # endregion views

    # region touch handling helpers
    def _on_press(self, touch_id: int, pos: tuple[int, int]) -> None:
        now_ms = monotonic() * 1000
        self._trackers[touch_id] = DwellTracker(touch_id=touch_id)
        self._trajectories[touch_id] = [(pos[0], pos[1], now_ms)]
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

        self._result.touch_event.append(
            TouchEvent(x=pos[0], y=pos[1], timestamp=time())
        )

    def _check_all_trackers(self) -> None:
        """Re-evaluate every active input each frame, so a stationary
        gaze/mouse (which produces no motion events) still completes its
        dwell in real time."""
        for touch_id, tracker in list(self._trackers.items()):
            if tracker.last_pos is not None:
                self._evaluate_tracker(touch_id, tracker.last_pos)

    def _is_on_fixation_point(self, pos: tuple[int, int]) -> bool:
        distance = math.dist(pos, self._fixation_center)
        return distance <= self._config.fixation_acceptance_distance_px

    def _active_shape_test(self, pos: tuple[int, int]) -> bool | None:
        """Whether pos is inside the fixation acceptance radius, or None
        outside the fixation phase (nothing to dwell on: dwell time
        neither accumulates nor resets)."""
        if self._phase == "fixation":
            return self._is_on_fixation_point(pos)
        return None

    def _evaluate_tracker(self, touch_id: int, pos: tuple[int, int]) -> None:
        tracker = self._trackers.get(touch_id)
        if tracker is None or tracker.fired:
            return

        is_inside = self._active_shape_test(pos)
        if is_inside is None:
            tracker.last_pos = pos
            return

        fired_now = tracker.update(
            pos,
            is_inside,
            self._config.min_touch_duration_ms,
            self._config.dwell_grace_ms,
        )
        if fired_now:
            self._on_fixation_touched()

    def _prune_trace(self) -> None:
        max_age_ms = self._config.gaze_trace_duration_ms
        if max_age_ms <= 0:
            self._trace_points.clear()
            return
        now_ms = monotonic() * 1000
        self._trace_points = [
            point for point in self._trace_points if now_ms - point[2] <= max_age_ms
        ]

    def _draw_trace(self, screen: SurfaceType) -> None:
        max_age_ms = self._config.gaze_trace_duration_ms

        point_groups: list[list[tuple[int, int, float]]] = []
        if max_age_ms > 0 and self._trace_points:
            point_groups.append(sorted(self._trace_points, key=lambda p: p[2]))
        for trajectory in self._trajectories.values():
            if trajectory:
                point_groups.append(trajectory)

        if not point_groups:
            return

        overlay = Surface(screen.get_size(), SRCALPHA)

        now_ms = monotonic() * 1000
        active_trajectory_ids = {id(t) for t in self._trajectories.values()}

        for points in point_groups:
            is_live = id(points) in active_trajectory_ids
            last_index = len(points) - 1

            for i, (x, y, fired_ms) in enumerate(points):
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

    # region events handler

    def _cancel_unfinished_trial(self) -> None:
        if self._result.result != ResultEnum.PENDING:
            return

        self._result.result = ResultEnum.CANCELLED

    def _request_game_quit(self) -> None:
        event.post(Event(QUIT))

    def on_gaze_abandoned(self) -> None:
        """Gaze lost for longer than GazeDwellDriver's grace period. An
        unresolved trial is cancelled; once the trial has already resolved
        (GIF playing, or inter-trial wait) it is left to finish normally."""
        if self._running and self._result.result == ResultEnum.PENDING:
            self.quit()

    def _on_touch_penalized(self) -> None:
        """A touchscreen touch abandons an unresolved trial with no
        reward (CANCELLED), the same as in visual_3pt_calibration."""
        if self._running and self._result.result == ResultEnum.PENDING:
            self.quit()

    # endregion events handler

    # region reward
    def _give_reward(self, duration_ms: int | None = None) -> None:
        self._rewarder.give_reward(duration_ms or self._config.reward_duration_ms)
        self._result.trial_reward_count += 1

    def _give_manual_reward(self) -> None:
        # Counted only as a manual reward, so the on-screen reward total
        # (automatic + manual) doesn't count it twice.
        self._rewarder.give_reward(self._config.reward_duration_ms)
        self._result.trial_manual_reward_count += 1

    # endregion reward
