from pathlib import Path
from random import choice, randint, random
from shutil import copy2
from time import monotonic, time
from typing import TYPE_CHECKING, ClassVar

from loguru import logger
from mxbiflow import get_mxbiflow
from mxbiflow.assets import (
    CLICKER_PATH,
    Corner,
    CornerButton,
    ImageSprite,
    ResultWidget,
    create_background,
)
from mxbiflow.core import ConfigStore
from mxbiflow.driver.audioplayer import (
    LoadedWav,
    PlayStatus,
    PureToneGenerator,
    PureToneUnit,
)
from mxbiflow.infra import DataLogger
from mxbiflow.scene import Scene
from pygame import FINGERDOWN, MOUSEBUTTONDOWN, QUIT, Event, event
from pygame.sprite import Group
from rich.pretty import pprint

from assets import GFG001, M001, M002, NATURAL_TWITTER_PATHS
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
    TrialStimulus,
)

if TYPE_CHECKING:
    from mxbiflow.core import MXBIFlow
    from mxbiflow.driver.audioplayer import PlayResult
    from mxbiflow.models import Animal
    from pygame.sprite import Sprite
    from pygame.surface import Surface


class TargetPositionGeneralizationTraining(Scene):
    level_table: ClassVar[dict[str, list[int]]] = LEVEL_TABLE
    _trials_context_key: ClassVar[str] = (
        "target_position_generalization_training.trials"
    )
    _attention_background_color: ClassVar[tuple[int, int, int]] = (48, 48, 48)

    # region initialization
    def __init__(self) -> None:
        super().__init__()

        self._running = False
        self._is_reaction_window_open = False
        self._reaction_window_started_at_ms: float | None = None

        self._mxbiflow = get_mxbiflow()
        self._rewarder = self._mxbiflow.mxbi.rewarder
        self._timer = self._mxbiflow.timer
        self._aplayer = self._mxbiflow.mxbi.aplayer

        self._animal = self._mxbiflow.session.require_current_animal()
        self._config = self._init_config(self._animal, self._mxbiflow)
        self._data_logger = self._init_data_logger(self._mxbiflow, self._animal)
        self._control_buttons = self._init_control_buttons()

        self._init_visual_stimulus(self._config)
        self._auditory_stimulus = self._init_auditory_stimulus(self._config)

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
            is_twitter_trial=random() < level_config.twitter_trial_probability,
            pure_tone_frequency_hz=randint(*stored_config.pure_tone_frequency_range_hz),
            pure_tone_duration_ms=randint(*stored_config.pure_tone_duration_range_ms),
            screen_size=Screen(screen_size.width, screen_size.height),
            stored=stored_config,
        )

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
            target_side=config.target_side,
            stimulus=TrialStimulus(
                pure_tone_frequency_hz=config.pure_tone_frequency_hz,
                pure_tone_duration_ms=config.pure_tone_duration_ms,
            ),
            result=ResultEnum.PENDING,
            trial_start_time=now,
            trial_end_time=now,
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

    def _init_visual_stimulus(self, config: Config) -> None:
        self._init_backgrounds(config)

        self._sprites: Group[Sprite] = Group()

        target_image = choice((M001, M002)) if config.is_twitter_trial else GFG001

        screen_size = config.screen_size
        # (0, 0) +------------------------------> x = W
        #        |              Target (3W/4, H/2 ± H/6 * ratio)
        #        |
        #        |
        #        v
        #      y = H
        target_center_x = screen_size.width * 3 // 4
        center_y = screen_size.height // 2
        vertical_offset = round(
            (screen_size.height // 6) * config.position_offset_ratio
        )
        target_center_y = (
            center_y + vertical_offset
            if config.target_side == "right"
            else center_y - vertical_offset
        )

        target_size_px = screen_size.width // 5

        self._correct_target = ImageSprite(
            target_image,
            target_size_px,
            (target_center_x, target_center_y),
        )

    def _init_backgrounds(self, config: Config) -> None:
        screen_size = (config.screen_size.width, config.screen_size.height)
        self._stimulus_background = create_background(
            screen_size,
            border_width=config.border_width,
        )
        self._attention_background = create_background(
            screen_size,
            border_width=config.border_width,
            background_color=self._attention_background_color,
        )
        self._background = self._attention_background

    def _init_auditory_stimulus(self, config: Config) -> LoadedWav | list[PureToneUnit]:
        auditory_stimulus: LoadedWav | list[PureToneUnit] | None = None

        if config.is_twitter_trial:
            wav = choice(NATURAL_TWITTER_PATHS)
            auditory_stimulus = self._aplayer.load_wav(wav)
        else:
            generator = PureToneGenerator()
            tone_unit_config = PureToneUnit(
                frequency_hz=config.pure_tone_frequency_hz,
                duration_ms=config.pure_tone_unit_duration_ms,
            )
            interval_unit_config = PureToneUnit(
                frequency_hz=0,
                duration_ms=config.pure_tone_interval_ms,
            )

            tone_unit = generator.gen_stimulus_unit(tone_unit_config)
            interval_unit = generator.gen_stimulus_unit(interval_unit_config)

            auditory_stimulus = generator.generate_stimulus_sequence(
                [tone_unit, interval_unit],
                config.pure_tone_duration_ms,
            )

        return auditory_stimulus

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
        self._aplayer.cancel_all()

        self._running = False

    def handle_event(self, event: Event) -> None:

        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            if any(button.handle_event(event) for button in self._control_buttons):
                return
            self._on_touched(event)

        if event.type == FINGERDOWN:
            screen = self._mxbiflow.mxbi.screen_size
            touch_event = Event(
                MOUSEBUTTONDOWN,
                button=1,
                pos=(
                    int(event.x * screen.width),
                    int(event.y * screen.height),
                ),
            )
            self._on_touched(touch_event)

    def update(self, dt_s: float) -> None:
        self._control_buttons.update(dt_s)

    def draw(self, screen: Surface) -> None:
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
        self._timer.after(
            self._config.attention_duration_ms,
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
        level_background = 0
        level_valid = 0
        for trial in trials.root.values():
            if trial.animal.level != level:
                continue
            if trial.result != ResultEnum.CANCELLED:
                level_valid += 1
            if trial.result is ResultEnum.CORRECT:
                level_correct += 1
            elif trial.result is ResultEnum.BACKGROUND_TOUCHED:
                level_background += 1

        level_completed = level_correct + level_background
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
        self._sprites.add(
            [
                self._correct_target,
            ]
        )
        if self._config.reaction_timeout_enabled:
            self._timer.after(
                self._config.reaction_window_duration_ms,
                self._on_timeout,
            )

    def _result_summary(self) -> str:
        level = self._result.animal.level
        level_correct = 0
        level_background = 0
        correct = 0
        timeout = 0
        cancelled = 0
        reward = 0
        for trial in self._load_trials().root.values():
            if trial.animal.level == level:
                if trial.result is ResultEnum.CORRECT:
                    level_correct += 1
                elif trial.result is ResultEnum.BACKGROUND_TOUCHED:
                    level_background += 1
            if trial.result is ResultEnum.CORRECT:
                correct += 1
            elif trial.result is ResultEnum.TIMEOUT:
                timeout += 1
            elif trial.result is ResultEnum.CANCELLED:
                cancelled += 1
            reward += trial.trial_reward_count + trial.trial_manual_reward_count

        level_completed = level_correct + level_background
        level_correct_rate = level_correct / level_completed if level_completed else 0
        return (
            f"Animal {self._animal.name} | Trial {self._animal.trial_id} | "
            f"Level {level} | Rate {level_correct_rate:.1%} | "
            f"Correct {correct} | "
            f"Reward {reward} | "
            f"Timeout {timeout} | "
            f"Cancelled {cancelled}"
        )

    # endregion views

    # region events handler
    def _on_touched(self, event: Event) -> None:
        self._result.touch_event.append(
            TouchEvent(
                x=event.pos[0],
                y=event.pos[1],
                timestamp=time(),
            )
        )

        if not self._is_reaction_window_open:
            return

        self._is_reaction_window_open = False
        self._sprites.remove(
            [
                self._correct_target,
            ]
        )

        result = self._classify_touch(event.pos)

        if result is ResultEnum.CORRECT:
            started_at_ms = self._reaction_window_started_at_ms
            if started_at_ms is None:
                raise RuntimeError("Reaction window start time is unavailable")
            latency_ms = max(0.0, monotonic() * 1000 - started_at_ms)
            self._result.response_latency_ms = latency_ms
            self._on_correct()
        elif result is ResultEnum.BACKGROUND_TOUCHED:
            self._on_background_touched()

    def _classify_touch(self, pos: tuple[int, int]) -> ResultEnum:
        if (rect := self._correct_target.rect) is not None and rect.collidepoint(pos):
            return ResultEnum.CORRECT

        return ResultEnum.BACKGROUND_TOUCHED

    def _on_correct(self):
        self._result.result = ResultEnum.CORRECT

        self._give_reward()
        self._on_inter_trial()

    def _on_background_touched(self):
        self._result.result = ResultEnum.BACKGROUND_TOUCHED

        self._on_inter_trial()

    def _on_timeout(self) -> None:
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

        click_task = self._aplayer.play_wav(CLICKER_PATH)
        click_task.on_finish(self._on_click_played)

        stimulus = self._auditory_stimulus
        task = None

        if isinstance(stimulus, LoadedWav):
            task = self._aplayer.play_loaded_wav(stimulus)
        else:
            task = self._aplayer.play_puretone_sequence(stimulus)

        visual_task = self._timer.after(
            self._config.visual_delay_ms,
            self._show_visual_stimulus,
        )

        def _on_audio_finished(result: PlayResult) -> None:
            if result.status == PlayStatus.ERROR:
                self._timer.cancel(visual_task)
                raise RuntimeError(f"Stimulus playback failed: {result.error!r}")

        task.on_finish(_on_audio_finished)

    def _on_click_played(self, result: PlayResult) -> None:
        if result.status == PlayStatus.ERROR:
            logger.warning(
                "clicker playback failed: {}",
                result.error,
            )

    def _give_reward(self, duration_ms: int | None = None) -> None:
        self._rewarder.give_reward(duration_ms or self._config.reward_duration_ms)
        self._result.trial_reward_count += 1

    def _give_manual_reward(self) -> None:
        self._give_reward()
        self._result.trial_manual_reward_count += 1

    # endregion stimulus and reward

    # region utils
    # endregion utils
