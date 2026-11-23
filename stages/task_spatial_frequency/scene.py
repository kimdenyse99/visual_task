"""Spatial frequency task: detect a Gabor patch left or right of fixation.

Trial flow (one scene instance per trial, like every other stage here):

1. fixation  - the fixation point appears at the center position. Gaze must
               dwell (min_touch_duration_ms, with dwell_grace_ms tolerance)
               within the level's fixation_tolerance_cm of it.
2. delay     - a random fixation_delay_min_ms..max_ms gap. The fixation
               point stays or is removed (keep_fixation_point_after_fixation).
3. stimulus  - one of three trial types, chosen by weight:
               * left / right: a Gabor patch at a random spatial frequency
                 from the level's range appears on that side. A gaze dwell
                 on it (within stimulus_tolerance_cm) that completes inside
                 the level's reward time window is CORRECT and rewarded;
                 one that completes before or after the window is a
                 TIMEOUT (early / late). A dwell on the empty side is
                 INCORRECT. No dwell within stimulus_duration_ms is a
                 TIMEOUT (no response).
               * none: nothing appears. Gaze kept on the fixation point
                 until the end of the level's reward time window is
                 CORRECT and rewarded; leaving it for longer than
                 dwell_grace_ms is INCORRECT.
4. blank     - inter_trial_duration_ms of empty screen, then the next trial.
"""

import math
from pathlib import Path
from random import choice, choices, randint
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

from dwell import DwellTracker
from gabor_sprite import GaborSprite
from gaze_input import GAZE_TOUCH_ID, GazeDwellDriver, GazeSample
from models import TouchEvent
from stimulus_events import StimulusEvents
from trial_events import write_target_shown_event

from .models import (
    LEVEL_TABLE,
    PATH,
    Config,
    ResponseTiming,
    Result,
    ResultEnum,
    Screen,
    StimulusSide,
    StoredConfigs,
    TrialResults,
)

if TYPE_CHECKING:
    from mxbiflow.core import MXBIFlow
    from mxbiflow.models import Animal
    from pygame.surface import Surface as SurfaceType

type Phase = Literal["blank", "fixation", "delay", "stimulus"]
type Side = Literal["left", "right"]


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


# Touch/gaze hand-off between consecutive trials of this stage - same rules
# as visual_3pt_calibration_v2: only stashed when a trial ends normally.
_pending_touch_positions: dict[int, tuple[int, int]] | None = None


class TaskSpatialFrequency(Scene):
    level_table: ClassVar[dict[str, list[int]]] = LEVEL_TABLE
    _trials_context_key: ClassVar[str] = "task_spatial_frequency.trials"
    _trace_color: ClassVar[tuple[int, int, int]] = (220, 0, 0)
    _tolerance_area_color: ClassVar[tuple[int, int, int]] = (0, 200, 0)
    _tolerance_area_width_px: ClassVar[int] = 2

    # region initialization
    def __init__(self) -> None:
        super().__init__()

        self._running = False
        self._phase: Phase = "blank"

        self._mxbiflow = get_mxbiflow()
        self._stimulus_events = StimulusEvents(self._mxbiflow)
        self._rewarder = self._mxbiflow.mxbi.rewarder
        self._timer = self._mxbiflow.timer
        self._stimulus_end_task: _CancellableTask | None = None
        self._fixation_shown_at_ms = 0.0
        self._stimulus_onset_ms = 0.0
        self._outside_fixation_since_ms: float | None = None

        self._animal = self._mxbiflow.session.require_current_animal()
        self._config = self._init_config(self._animal, self._mxbiflow)
        self._data_logger = self._init_data_logger(self._mxbiflow, self._animal)
        self._control_buttons = self._init_control_buttons()

        self._init_visual_stimulus(self._config)

        self._trackers: dict[int, DwellTracker] = {}
        self._trajectories: dict[int, list[tuple[int, int, float]]] = {}
        self._claim_pending_touches()

        self._trace_points: list[tuple[int, int, float]] = []

        def _get_gaze_sample() -> GazeSample | None:
            return self._mxbiflow.eye_sample

        self._gaze_driver = GazeDwellDriver(
            _get_gaze_sample,
            self,
            lost_grace_ms=self._config.stored.gaze_lost_grace_ms,
            already_down=GAZE_TOUCH_ID in self._trackers,
        )

        self._mxbiflow.session.add_trial()
        self._result = self._init_result(self._animal)
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
        if level not in stored_config.levels:
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

        # Gaze can vanish between trials, so only reinstate it if a fresh
        # sample exists right now (see visual_3pt_calibration).
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

    def _init_result(self, animal: Animal) -> Result:
        now = time()
        return Result(
            animal=animal.base_info,
            result=ResultEnum.PENDING,
            trial_start_time=now,
            trial_end_time=now,
            stimulus_side=self._stimulus_side,
            stimulus_center=(
                self._positions[self._stimulus_side]
                if self._stimulus_side != "none"
                else None
            ),
            spatial_frequency_cpcm=self._spatial_frequency_cpcm,
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
            border_width=self._config.stored.border_width,
            color=(0, 0, 0),
        )

    def _init_control_buttons(self) -> Group[CornerButton]:
        border_width = self._config.stored.border_width
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
    def _stimulus_positions(config: Config) -> dict[str, tuple[int, int]]:
        """Fixation ("center") and the two stimulus positions, in pygame
        pixels. Same geometry as visual_3pt_calibration_v2's positions."""
        screen_size = config.screen_size
        center = (screen_size.width * 3 // 4, screen_size.height // 2)
        offset_x_px = round(config.cm_to_px(config.stored.position_distance_x_cm))
        offset_y_px = round(config.cm_to_px(config.stored.position_distance_y_cm))
        return {
            "center": center,
            "left": (center[0] - offset_x_px, center[1] - offset_y_px),
            "right": (center[0] - offset_x_px, center[1] + offset_y_px),
        }

    @staticmethod
    def _pick_stimulus_side(config: Config) -> StimulusSide:
        stored = config.stored
        sides: list[StimulusSide] = ["left", "right", "none"]
        weights = [
            stored.stimulus_left_weight,
            stored.stimulus_right_weight,
            stored.stimulus_none_weight,
        ]
        return choices(sides, weights=weights)[0]

    def _init_visual_stimulus(self, config: Config) -> None:
        self._background = create_background(
            (config.screen_size.width, config.screen_size.height),
            border_width=config.stored.border_width,
            background_color=config.stored.background_color,
        )

        self._sprites: Group[Sprite] = Group()
        self._positions = self._stimulus_positions(config)

        self._fixation_point = FixationPoint(
            self._positions["center"],
            round(config.fixation_point_radius_px),
            config.stored.fixation_point_inner_radius_ratio,
        )

        self._stimulus_side: StimulusSide = self._pick_stimulus_side(config)
        self._spatial_frequency_cpcm: float | None = None
        self._gabor: GaborSprite | None = None
        if self._stimulus_side != "none":
            self._spatial_frequency_cpcm = choice(
                config.level_config.spatial_frequencies_cpcm()
            )
            stored = config.stored
            self._gabor = GaborSprite(
                self._positions[self._stimulus_side],
                max(1, round(config.target_size_px)),
                cycles_per_px=self._spatial_frequency_cpcm / stored.pixels_per_cm,
                contrast=stored.gabor_contrast,
                background=stored.background_color,
                orientation_deg=stored.gabor_orientation_deg,
                phase_deg=stored.gabor_phase_deg,
            )

    # endregion stimulus

    # region console

    def _print_trial_config(self) -> None:
        stimulus = (
            "no stimulus"
            if self._stimulus_side == "none"
            else f"{self._spatial_frequency_cpcm} cycles/cm on the "
            f"{self._stimulus_side}"
        )
        print()
        print("=" * 60)
        print(
            f"Beginning level {self._animal.current_stage.level}, "
            f"trial {self._animal.trial_id}, "
            f"for monkey {self._animal.name}, "
            f"in the {self.name()} stage: {stimulus}"
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
        if self._phase == "stimulus" and self._stimulus_side == "none":
            self._check_fixation_held()
        self._prune_trace()

    def draw(self, screen: SurfaceType) -> None:
        screen.blit(self._background, (0, 0))

        self._sprites.draw(screen)
        self._stimulus_events.present(
            self._sprites,
            self._fixation_point,
            self._gabor,
            target_type="gabor",
            target_metadata={
                "spatial_frequency_cpcm": self._spatial_frequency_cpcm,
                "contrast": self._config.stored.gabor_contrast,
                "orientation_deg": self._config.stored.gabor_orientation_deg,
                "side": self._stimulus_side,
            },
        )
        if self._config.stored.show_tolerance_areas:
            self._draw_tolerance_areas(screen)
        if self._config.stored.show_gaze_trace:
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
        self._reset_trackers()

    def _on_fixation_acquired(self) -> None:
        self._result.fixation_latency_ms = max(
            0.0, monotonic() * 1000 - self._fixation_shown_at_ms
        )
        # Gaze isn't scored during the delay: it neither counts toward nor
        # against anything until the stimulus phase starts.
        self._phase = "delay"
        if not self._config.stored.keep_fixation_point_after_fixation:
            self._sprites.remove(self._fixation_point)

        delay_ms = randint(
            self._config.stored.fixation_delay_min_ms,
            self._config.stored.fixation_delay_max_ms,
        )
        self._result.fixation_delay_ms = delay_ms
        self._timer.after(delay_ms, self._start_stimulus_phase)

    def _start_stimulus_phase(self) -> None:
        if self._phase != "delay":
            return
        self._phase = "stimulus"
        self._stimulus_onset_ms = monotonic() * 1000
        level = self._config.level_config

        if self._gabor is None:
            # No-stimulus trial: gaze has to stay on the fixation point
            # until the end of the reward time window.
            self._outside_fixation_since_ms = None
            self._stimulus_end_task = self._timer.after(
                level.response_window_max_ms, self._on_fixation_held
            )
        else:
            self._sprites.add(self._gabor)
            assert self._gabor.rect is not None
            center = self._gabor.rect.center
            write_target_shown_event(
                self._mxbiflow,
                self._animal,
                self.name(),
                self._animal.trial_id,
                (round(center[0]), round(center[1])),
            )
            self._stimulus_end_task = self._timer.after(
                self._config.stored.stimulus_duration_ms, self._on_no_response
            )

        # Dwell progress starts from zero against the new areas; input
        # that is already down can complete a dwell from where it is.
        self._reset_trackers()
        for touch_id, tracker in list(self._trackers.items()):
            if tracker.last_pos is not None:
                self._evaluate_tracker(touch_id, tracker.last_pos)

    def _on_side_dwell(self, pos: tuple[int, int]) -> None:
        """A gaze dwell completed on one of the two side areas (stimulus
        trials only)."""
        latency_ms = max(0.0, monotonic() * 1000 - self._stimulus_onset_ms)
        self._result.response_latency_ms = latency_ms
        level = self._config.level_config

        if not self._is_in_area(pos, self._stimulus_side):
            self._resolve(ResultEnum.INCORRECT, "wrong_side")
        elif latency_ms < level.response_window_min_ms:
            self._resolve(ResultEnum.TIMEOUT, "early")
        elif latency_ms > level.response_window_max_ms:
            self._resolve(ResultEnum.TIMEOUT, "late")
        else:
            self._resolve(ResultEnum.CORRECT, "in_window")

    def _check_fixation_held(self) -> None:
        """No-stimulus trials: fail as soon as gaze has been off the
        fixation point for longer than the dwell grace period."""
        if self._result.result != ResultEnum.PENDING:
            return
        now_ms = monotonic() * 1000
        is_inside = any(
            tracker.last_pos is not None and self._is_in_area(tracker.last_pos, "center")
            for tracker in self._trackers.values()
        )
        if is_inside:
            self._outside_fixation_since_ms = None
            return
        if self._outside_fixation_since_ms is None:
            self._outside_fixation_since_ms = now_ms
        if now_ms - self._outside_fixation_since_ms >= self._config.dwell_grace_ms:
            self._resolve(ResultEnum.INCORRECT, "broke_fixation")

    def _on_fixation_held(self) -> None:
        self._stimulus_end_task = None
        if self._phase != "stimulus" or self._result.result != ResultEnum.PENDING:
            return
        # Check one last time, so a break right at the end still counts.
        self._check_fixation_held()
        if self._result.result == ResultEnum.PENDING:
            self._resolve(ResultEnum.CORRECT, "held_fixation")

    def _on_no_response(self) -> None:
        self._stimulus_end_task = None
        if self._phase == "stimulus" and self._result.result == ResultEnum.PENDING:
            self._resolve(ResultEnum.TIMEOUT, "no_response")

    def _resolve(self, result: ResultEnum, response: ResponseTiming) -> None:
        if self._result.result != ResultEnum.PENDING:
            return
        self._result.result = result
        self._result.response = response
        logger.info(
            "Trial {}: {} ({}), stimulus {}, {} cycles/cm, latency {} ms",
            self._animal.trial_id,
            result,
            response,
            self._stimulus_side,
            self._spatial_frequency_cpcm,
            self._result.response_latency_ms,
        )
        if result is ResultEnum.CORRECT:
            self._give_reward()

        if self._stimulus_end_task is not None:
            self._stimulus_end_task.cancel()
            self._stimulus_end_task = None
        self._sprites.empty()
        self._phase = "blank"
        self._timer.after(self._config.stored.inter_trial_duration_ms, self.quit)

    def _settlement(self) -> bool:
        trials = self._load_trials()
        trials.root[self._result.animal.trial_id] = self._result
        self._save_trials(trials)

        level = self._result.animal.level
        level_correct = 0
        level_valid = 0
        for trial in trials.root.values():
            if trial.animal.level != level or trial.result is ResultEnum.CANCELLED:
                continue
            level_valid += 1
            if trial.result is ResultEnum.CORRECT:
                level_correct += 1

        # Correct rate over every completed trial: incorrect (wrong side,
        # broke fixation) and timeouts (early, late, no response) all count
        # against it.
        level_correct_rate = level_correct / level_valid if level_valid else 0

        is_level_up = (
            self._result.result != ResultEnum.CANCELLED
            and self._animal.current_stage.level < self._config.total_levels
            and level_valid > self._config.stored.level_up_trial_count_threshold
            and level_correct_rate >= self._config.stored.level_up_threshold
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
    def _result_summary(self) -> str:
        level = self._result.animal.level
        level_correct = 0
        level_valid = 0
        counts = {result: 0 for result in ResultEnum}
        reward = 0
        for trial in self._load_trials().root.values():
            counts[trial.result] += 1
            reward += trial.trial_reward_count + trial.trial_manual_reward_count
            if trial.animal.level == level and trial.result is not ResultEnum.CANCELLED:
                level_valid += 1
                if trial.result is ResultEnum.CORRECT:
                    level_correct += 1

        level_correct_rate = level_correct / level_valid if level_valid else 0
        return (
            f"Level {level}/{self._config.total_levels} | "
            f"Animal {self._animal.name} | Trial {self._animal.trial_id} | "
            f"Rate {level_correct_rate:.1%} | "
            f"Correct {counts[ResultEnum.CORRECT]} | "
            f"Error {counts[ResultEnum.INCORRECT]} | "
            f"Timeout {counts[ResultEnum.TIMEOUT]} | "
            f"Reward {reward} | "
            f"Cancelled {counts[ResultEnum.CANCELLED]}"
        )

    def _draw_tolerance_areas(self, screen: SurfaceType) -> None:
        areas: list[str] = []
        if self._phase == "fixation" or (
            self._phase == "stimulus" and self._stimulus_side == "none"
        ):
            areas.append("center")
        if self._phase == "stimulus" and self._stimulus_side != "none":
            areas += ["left", "right"]
        for area in areas:
            draw_circle(
                screen,
                self._tolerance_area_color,
                self._positions[area],
                round(self._area_radius_px(area)),
                self._tolerance_area_width_px,
            )

    # endregion views

    # region touch handling helpers
    def _reset_trackers(self) -> None:
        for tracker in self._trackers.values():
            tracker.reset()

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

    def _area_radius_px(self, area: str) -> float:
        if area == "center":
            return self._config.fixation_tolerance_distance_px
        return self._config.stimulus_tolerance_distance_px

    def _is_in_area(self, pos: tuple[int, int], area: str) -> bool:
        return math.dist(pos, self._positions[area]) <= self._area_radius_px(area)

    def _active_shape_test(self, pos: tuple[int, int]) -> bool | None:
        """Whether pos is inside an area a dwell can complete on right now,
        or None when nothing is scored by dwell (dwell time neither
        accumulates nor resets)."""
        if self._phase == "fixation":
            return self._is_in_area(pos, "center")
        if self._phase == "stimulus" and self._stimulus_side != "none":
            return self._is_in_area(pos, "left") or self._is_in_area(pos, "right")
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
            self._config.stored.min_touch_duration_ms,
            self._config.dwell_grace_ms,
        )
        if not fired_now:
            return
        if self._phase == "fixation":
            self._on_fixation_acquired()
        elif self._phase == "stimulus":
            self._on_side_dwell(pos)

    def _prune_trace(self) -> None:
        max_age_ms = self._config.stored.gaze_trace_duration_ms
        if max_age_ms <= 0:
            self._trace_points.clear()
            return
        now_ms = monotonic() * 1000
        self._trace_points = [
            point for point in self._trace_points if now_ms - point[2] <= max_age_ms
        ]

    def _draw_trace(self, screen: SurfaceType) -> None:
        max_age_ms = self._config.stored.gaze_trace_duration_ms

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
        """Gaze lost for longer than gaze_lost_grace_ms. An unresolved
        trial is cancelled; a resolved one is left to finish its
        inter-trial period."""
        if self._running and self._result.result == ResultEnum.PENDING:
            self.quit()

    def _on_touch_penalized(self) -> None:
        """A touchscreen touch abandons an unresolved trial with no
        reward (CANCELLED)."""
        if self._running and self._result.result == ResultEnum.PENDING:
            self.quit()

    # endregion events handler

    # region reward
    def _give_reward(self) -> None:
        self._rewarder.give_reward(self._config.stored.reward_duration_ms)
        self._result.trial_reward_count += 1

    def _give_manual_reward(self) -> None:
        self._rewarder.give_reward(self._config.stored.reward_duration_ms)
        self._result.trial_manual_reward_count += 1

    # endregion reward
