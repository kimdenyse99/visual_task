from random import choice
from shutil import copy2
from time import monotonic, time
from typing import TYPE_CHECKING, ClassVar

from loguru import logger
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
from pygame import FINGERDOWN, MOUSEBUTTONDOWN, QUIT, Event, event
from pygame.sprite import Group
from rich.pretty import pprint

from assets import MARMOSET_IMAGE_PATHS
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
from .target_sprite import TouchTargetSprite

if TYPE_CHECKING:
    from mxbiflow.core import MXBIFlow
    from mxbiflow.models import Animal
    from pygame.surface import Surface


class SizeReduction3Pt(Scene):
    level_table: ClassVar[dict[str, list[int]]] = LEVEL_TABLE
    _trials_context_key: ClassVar[str] = "size_reduction_3pt.trials"

    # Scene.name() auto-converts the class name to snake_case, but that
    # conversion mishandles digits (e.g. "SizeReduction3Pt" would become
    # "size_reduction3_pt", not "size_reduction_3pt"), so the stage/folder
    # name is pinned explicitly here instead.
    @classmethod
    def name(cls) -> str:
        return "size_reduction_3pt"

    def __init__(self) -> None:
        super().__init__()
        self._mxbiflow = get_mxbiflow()
        self._rewarder = self._mxbiflow.mxbi.rewarder
        self._timer = self._mxbiflow.timer
        self._aplayer = self._mxbiflow.mxbi.aplayer
        self._animal = self._mxbiflow.session.require_current_animal()
        self._config = self._init_config(self._animal, self._mxbiflow)
        self._data_logger = DataLogger(
            session=self._mxbiflow.session,
            animal=self._animal.name,
            stage=self.name(),
        )
        self._mxbiflow.session.add_trial()
        self._result = self._init_result(self._animal, self._config)
        self._is_reaction_window_open = False
        self._reaction_window_started_at_ms: float | None = None
        # Refreshed on every touch/click, hit or miss, whether or not the
        # reaction window is open. Gates when the next trial's target may
        # appear (see _begin_quiet_wait / _check_quiet_period).
        self._last_touch_at_ms: float = monotonic() * 1000
        # Set properly in _finish() for every outcome (as
        # inter_trial_duration_ms), then possibly extended further in
        # _give_automatic_reward_and_finish() if a correct trial's marmoset
        # flash needs more time than that. quit() is never allowed to fire
        # before this, regardless of whether the quiet period is
        # enabled/disabled or how short it's set.
        self._earliest_quit_at_ms: float = 0.0

        target_center = self._target_center(self._config.screen_size)
        self._target = TouchTargetSprite(
            self._config.stimulus.target_size_px,
            target_center,
        )
        self._sprites = Group[TouchTargetSprite | ImageSprite](self._target)
        self._marmoset_sprite: ImageSprite | None = None
        self._marmoset_flash_points = self._init_marmoset_flash_points(target_center)
        self._control_buttons = self._init_control_buttons()
        screen = (self._config.screen_size.width, self._config.screen_size.height)
        self._background = create_background(
            screen,
            border_width=self._config.stored.border_width,
        )
        self._result_widget = ResultWidget(
            self._result_summary(),
            border_width=self._config.stored.border_width,
            color=(0, 0, 0),
        )
        self._print_trial_config()

    @staticmethod
    def _target_center(screen_size: Screen) -> tuple[int, int]:
        # This project's Screen has no target_center property (unlike
        # gongo-frequency-discriminate's), so it's computed the same way the
        # other stages in this project already do it.
        return (screen_size.width * 3 // 4, screen_size.height // 2)

    def _init_config(self, animal: Animal, mxbiflow: MXBIFlow) -> Config:
        snapshot_path = (
            mxbiflow.session.absolute_animal_data_path(animal.name)
            / self.name()
            / "config.json"
        )
        if not snapshot_path.exists():
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            copy2(PATH, snapshot_path)
        configs = ConfigStore(
            snapshot_path,
            StoredConfigs,
            create_default=False,
        ).value
        stored = configs.root.get(animal.name) or configs.root.get("default")
        if stored is None:
            raise ValueError(f"No config found for animal {animal.name}")
        level = animal.current_stage.level
        min_level, max_level = min(stored.levels), max(stored.levels)
        if level < min_level:
            level = min_level
        elif level > max_level:
            level = max_level
        level_config = stored.levels[level]
        screen = mxbiflow.mxbi.screen_size
        return Config(
            animal=animal.name,
            level=level,
            screen_size=Screen(screen.width, screen.height),
            stimulus=TrialStimulus(
                target_size_px=level_config.target_size_px,
                reward_delay_ms=level_config.reward_delay_ms,
            ),
            stored=stored,
        )

    def _init_result(self, animal: Animal, config: Config) -> Result:
        now = time()
        return Result(
            animal=animal.base_info,
            screen_size=config.screen_size,
            stimulus=config.stimulus,
            trial_start_time=now,
            trial_end_time=now,
        )

    def _init_marmoset_flash_points(
        self, target_center: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """The 3 fixed points the marmoset image may flash at: directly
        above, below, and to the right of the target center."""
        center_x, center_y = target_center
        stored = self._config.stored
        offset = int(round(stored.marmoset_offset_cm * stored.pixels_per_cm) // 2)
        return [
            (center_x - offset, center_y - offset),  # left
            (center_x - offset, center_y + offset),  # right
            (center_x + round(offset // 5), center_y),  # above
        ]
    def _init_control_buttons(self) -> Group[CornerButton]:
        border_width = self._config.stored.border_width
        size = border_width // 2
        return Group(
            CornerButton(
                Corner.TOP_RIGHT,
                size,
                border_width,
                self._give_manual_reward,
                color=(0, 160, 0),
            ),
            CornerButton(
                Corner.BOTTOM_RIGHT,
                size,
                border_width,
                self._request_game_quit,
                color=(200, 0, 0),
            ),
        )

    def start(self) -> None:
        self._running = True
        self._result.trial_start_time = time()
        self._is_reaction_window_open = True
        self._reaction_window_started_at_ms = monotonic() * 1000
        self._timer.after(
            self._config.stored.reaction_window_duration_ms,
            self._on_timeout,
        )

    def quit(self) -> None:
        if self._result.result is ResultEnum.PENDING:
            self._result.result = ResultEnum.CANCELLED
        self._result.trial_end_time = time()
        is_level_up = self._settlement()
        logger.info(
            "trial settled: stage={}, animal={}, result={}, level_up={}",
            self.name(),
            self._animal.name,
            self._result.result,
            is_level_up,
        )
        self._data_logger.save(self._result.model_dump())
        self._timer.cancel_all()
        self._aplayer.cancel_all()
        self._running = False

    def handle_event(self, event: Event) -> None:
        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            if any(button.handle_event(event) for button in self._control_buttons):
                return
            self._on_touched(event)
        elif event.type == FINGERDOWN:
            screen = self._mxbiflow.mxbi.screen_size
            converted = Event(
                MOUSEBUTTONDOWN,
                button=1,
                pos=(int(event.x * screen.width), int(event.y * screen.height)),
            )
            self._on_touched(converted)

    def update(self, dt_s: float) -> None:
        self._control_buttons.update(dt_s)

    def draw(self, screen: Surface) -> None:
        screen.blit(self._background, (0, 0))
        self._sprites.draw(screen)
        self._result_widget.draw(screen)
        for button in self._control_buttons:
            button.layout(screen)
        self._control_buttons.draw(screen)

    def _on_touched(self, incoming: Event) -> None:
        self._last_touch_at_ms = monotonic() * 1000
        self._result.touch_event.append(
            TouchEvent(x=incoming.pos[0], y=incoming.pos[1], timestamp=time())
        )
        if not self._is_reaction_window_open:
            return
        self._is_reaction_window_open = False
        self._sprites.remove(self._target)
        assert self._target.rect is not None
        if self._target.rect.collidepoint(incoming.pos):
            started_at_ms = self._reaction_window_started_at_ms
            if started_at_ms is None:
                raise RuntimeError("Reaction window start time is unavailable")
            self._result.response_latency_ms = max(
                0.0, monotonic() * 1000 - started_at_ms
            )
            self._finish(ResultEnum.CORRECT)
        else:
            self._finish(ResultEnum.BACKGROUND_TOUCHED)

    def _on_timeout(self) -> None:
        if self._is_reaction_window_open:
            self._is_reaction_window_open = False
            self._sprites.remove(self._target)
            self._finish(ResultEnum.TIMEOUT)

    def _finish(self, result: ResultEnum) -> None:
        if self._result.result is not ResultEnum.PENDING:
            return
        self._result.result = result
        self._timer.cancel_all()
        # Uniform floor for every outcome - see _begin_quiet_wait /
        # StoredConfig.inter_trial_duration_ms. Extended further below for
        # correct trials if the marmoset flash needs more time than this.
        self._earliest_quit_at_ms = (
            monotonic() * 1000 + self._config.stored.inter_trial_duration_ms
        )
        if result is ResultEnum.CORRECT:
            self._timer.after(
                self._config.stimulus.reward_delay_ms,
                self._give_automatic_reward_and_finish,
            )
        else:
            self._begin_quiet_wait()

    def _give_automatic_reward(self) -> None:
        self._rewarder.give_reward(self._config.stored.reward_duration_ms)
        self._result.trial_reward_count += 1

    def _give_automatic_reward_and_finish(self) -> None:
        self._give_automatic_reward()
        stored = self._config.stored
        flash_floor_ms = (
            monotonic() * 1000
            + stored.marmoset_flash_delay_ms
            + stored.marmoset_flash_duration_ms
        )
        # Extend, don't overwrite: whichever floor is later wins.
        self._earliest_quit_at_ms = max(self._earliest_quit_at_ms, flash_floor_ms)
        self._timer.after(
            stored.marmoset_flash_delay_ms,
            self._flash_marmoset_image,
        )
        self._begin_quiet_wait()

    def _give_manual_reward(self) -> None:
        self._rewarder.give_reward(self._config.stored.reward_duration_ms)
        self._result.trial_manual_reward_count += 1

    def _flash_marmoset_image(self) -> None:
        """A few ms after the reward, show a random marmoset image at a
        random one of the 3 fixed points around the target center, for
        marmoset_flash_duration_ms - independent of reward_duration_ms.
        Unaffected by touches, and independent of whether the quiet-period
        gate is enabled."""
        if not MARMOSET_IMAGE_PATHS:
            return
        image_path = choice(MARMOSET_IMAGE_PATHS)
        center = choice(self._marmoset_flash_points)
        stored = self._config.stored
        size_px = int(round(stored.marmoset_image_size_cm * stored.pixels_per_cm))
        self._marmoset_sprite = ImageSprite(
            image_path,
            size_px,
            center,
        )
        self._sprites.add(self._marmoset_sprite)
        self._timer.after(
            stored.marmoset_flash_duration_ms,
            self._remove_marmoset_image,
        )

    def _remove_marmoset_image(self) -> None:
        if self._marmoset_sprite is not None:
            self._sprites.remove(self._marmoset_sprite)
            self._marmoset_sprite = None

    def _begin_quiet_wait(self) -> None:
        """Start waiting for a touch-free window before ending the trial, if
        touch_quiet_period_enabled is on; otherwise end the trial right away
        (subject to _earliest_quit_at_ms either way - see its docstring).

        The next trial's scene shows its target as soon as it is created, so
        holding off on `quit()` here is what keeps the new target from
        appearing until the animal has left the screen alone for
        `touch_quiet_period_ms`. Any touch (on or off target) during the wait
        pushes the target's reappearance further out.
        """
        if self._config.stored.touch_quiet_period_enabled:
            self._schedule_quiet_check()
        else:
            self._quit_no_earlier_than_floor()

    def _quit_no_earlier_than_floor(self) -> None:
        remaining_ms = self._earliest_quit_at_ms - monotonic() * 1000
        if remaining_ms > 0:
            self._timer.after(remaining_ms, self.quit)
        else:
            self.quit()

    def _schedule_quiet_check(self) -> None:
        self._timer.after(
            self._config.stored.touch_quiet_period_ms,
            self._check_quiet_period,
        )

    def _check_quiet_period(self) -> None:
        now_ms = monotonic() * 1000
        quiet_long_enough = (
            now_ms - self._last_touch_at_ms >= self._config.stored.touch_quiet_period_ms
        )
        floor_not_yet_reached = now_ms < self._earliest_quit_at_ms
        if quiet_long_enough and not floor_not_yet_reached:
            self.quit()
        else:
            self._schedule_quiet_check()

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

    def _settlement(self) -> bool:
        saved = self._load_trials()
        trials = dict(saved.root)
        trials[self._result.animal.trial_id] = self._result
        self._save_trials(TrialResults(root=trials))
        level = self._result.animal.level
        eligible = [
            trial
            for trial in trials.values()
            if trial.animal.level == level
            and trial.result
            in {
                ResultEnum.CORRECT,
                ResultEnum.INCORRECT,
                ResultEnum.BACKGROUND_TOUCHED,
            }
        ]
        correct = sum(trial.result is ResultEnum.CORRECT for trial in eligible)
        rate = correct / len(eligible) if eligible else 0.0
        is_level_up = (
            self._result.result is not ResultEnum.CANCELLED
            and level < self._config.total_levels
            and len(eligible) >= self._config.stored.level_up_trial_count_threshold
            and rate >= self._config.stored.level_up_threshold
        )
        if is_level_up:
            self._mxbiflow.session.level_up(
                animal=self._animal.name,
                stage=self._animal.current_stage.stage_name,
            )
        return is_level_up

    def _result_summary(self) -> str:
        trials = self._load_trials().root.values()
        level = self._animal.current_stage.level
        eligible = [
            trial
            for trial in trials
            if trial.animal.level == level
            and trial.result
            in {
                ResultEnum.CORRECT,
                ResultEnum.INCORRECT,
                ResultEnum.BACKGROUND_TOUCHED,
            }
        ]
        correct = sum(trial.result is ResultEnum.CORRECT for trial in eligible)
        rate = correct / len(eligible) if eligible else 0.0
        return (
            f"Animal {self._animal.name} | Trial {self._animal.trial_id} | "
            f"Level {level}/{self._config.total_levels} | "
            f"Rate {rate:.1%} | Correct {correct}"
        )

    def _print_trial_config(self) -> None:
        print()
        print("=" * 60)
        print(
            f"Beginning level {self._config.level}, "
            f"trial {self._animal.trial_id}, for monkey {self._animal.name}, "
            f"in the {self.name()} stage"
        )
        print("=" * 60)
        pprint(self._config)
        print("=" * 60)

    def _request_game_quit(self) -> None:
        event.post(Event(QUIT))
