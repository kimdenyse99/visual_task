from typing import Any, Literal

from mxbiflow.models.animal import AnimalBaseInfo
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    RootModel,
    field_validator,
)

from models import (
    AnimalName,
    LevelID,
    ResultEnum,
    Screen,
    TouchEvent,
    load_level_table,
    path,
    validate_level_ids,
)

__all__ = [
    "Config",
    "LevelConfig",
    "Result",
    "ResultEnum",
    "Screen",
    "StoredConfig",
    "StoredConfigs",
]

PATH = path(__file__)
LEVEL_TABLE = load_level_table(PATH)

type RGBColor = tuple[int, int, int]


class Result(BaseModel):
    version: Literal["1.0.0"] = "1.0.0"
    animal: AnimalBaseInfo
    is_twitter_trial: bool

    result: ResultEnum = ResultEnum.PENDING

    trial_start_time: float
    trial_end_time: float

    response_latency_ms: float | None = Field(default=None, ge=0)

    trial_reward_count: int = 0
    trial_manual_reward_count: int = 0

    touch_event: list[TouchEvent] = Field(default_factory=list[TouchEvent])


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    animal: AnimalName
    level: LevelID

    is_twitter_trial: bool

    screen_size: Screen

    stored: StoredConfig = Field(repr=False, exclude=True)

    @property
    def inter_trial_duration_ms(self) -> int:
        return self.stored.inter_trial_duration_ms

    @property
    def reward_duration_ms(self) -> int:
        return self.stored.reward_duration_ms

    @property
    def attention_duration_ms(self) -> int:
        return self.stored.attention_duration_ms

    @property
    def fixation_duration_min_ms(self) -> int:
        return self.stored.fixation_duration_min_ms

    @property
    def fixation_duration_max_ms(self) -> int:
        return self.stored.fixation_duration_max_ms

    @property
    def visual_delay_ms(self) -> int:
        return self.stored.visual_delay_ms

    @property
    def reaction_window_duration_ms(self) -> int:
        return self.stored.levels[self.level].reaction_window_duration_ms

    @property
    def target_size_cm(self) -> float:
        return self.stored.levels[self.level].target_size_cm

    @property
    def target_size_px(self) -> float:
        return self.target_size_cm * self.stored.pixels_per_cm

    @property
    def border_width(self) -> int:
        return self.stored.border_width

    @property
    def level_up_threshold(self) -> float:
        return self.stored.level_up_threshold

    @property
    def level_up_trial_count_threshold(self) -> int:
        return self.stored.level_up_trial_count_threshold

    @property
    def total_levels(self) -> int:
        return len(self.stored.levels)

    @property
    def background_color(self) -> RGBColor:
        return self.stored.background_color

    @property
    def fixation_point_radius(self) -> int:
        return self.stored.fixation_point_radius

    @property
    def fixation_point_inner_radius_ratio(self) -> float:
        return self.stored.fixation_point_inner_radius_ratio

    @property
    def target_offset_ratio(self) -> float:
        return self.stored.target_offset_ratio

    @property
    def touch_buffer_px(self) -> float:
        return self.stored.touch_buffer_cm * self.stored.pixels_per_cm

    @property
    def response_window_min_ms(self) -> int:
        return self.stored.response_window_min_ms

    @property
    def response_window_max_ms(self) -> int:
        return self.stored.response_window_max_ms

    @property
    def min_touch_duration_ms(self) -> int:
        return self.stored.min_touch_duration_ms


class LevelConfig(BaseModel):
    reaction_window_duration_ms: PositiveInt
    target_size_cm: float = Field(gt=0)


class StoredConfig(BaseModel):
    inter_trial_duration_ms: PositiveInt
    reward_duration_ms: PositiveInt

    attention_duration_ms: NonNegativeInt
    visual_delay_ms: NonNegativeInt = 500

    # Fixation screen duration is randomized per-trial, uniformly drawn
    # from [fixation_duration_min_ms, fixation_duration_max_ms], rather
    # than using the fixed attention_duration_ms. attention_duration_ms
    # is kept above for schema parity with sibling stages but is not
    # used by this stage's trial-start timing.
    fixation_duration_min_ms: NonNegativeInt = 150
    fixation_duration_max_ms: NonNegativeInt = 300

    border_width: int = Field(ge=2)
    level_up_threshold: float = Field(ge=0, le=1)
    level_up_trial_count_threshold: PositiveInt

    background_color: RGBColor = (127, 127, 127)
    fixation_point_radius: int = Field(default=20, ge=1)
    fixation_point_inner_radius_ratio: float = Field(default=0.35, gt=0, lt=1)
    target_offset_ratio: float = Field(default=0.2, gt=0, lt=0.5)

    # Extra hit-box margin around the target image, as a physical distance
    # (centimeters) converted to pixels via pixels_per_cm. A fixed
    # physical buffer better reflects real touch/finger imprecision than
    # one scaled to the stimulus size, since that imprecision is roughly
    # constant regardless of how large the target is drawn.
    touch_buffer_cm: float = Field(default=1.0, ge=0)

    # Pixels-per-centimeter conversion factor for this display. There is
    # no DPI/physical-size lookup available in this environment, so this
    # must be supplied directly (e.g. from the monitor's spec sheet or a
    # manual ruler-against-screen measurement: pixels_per_cm =
    # screen_width_px / screen_width_cm).
    pixels_per_cm: float = Field(default=37.8, gt=0)

    # Only touches landing within this window after stimulus onset count
    # as a scored ("correct"/"background") response. Touches before
    # response_window_min_ms are treated as anticipatory and ignored;
    # touches after response_window_max_ms are too late to be scored as
    # correct, but the trial doesn't end until reaction_window_duration_ms
    # (per-level, see LevelConfig) elapses - that's the hard cutoff that
    # always ends the trial and moves on, scoring it as TIMEOUT if
    # nothing was scored in time.
    response_window_min_ms: NonNegativeInt = 50
    response_window_max_ms: PositiveInt = 800

    # Minimum press duration (button/finger down -> up) for a touch to be
    # registered at all. Shorter presses (e.g. accidental brushes) are
    # discarded before scoring.
    min_touch_duration_ms: NonNegativeInt = 100

    # Declared after response_window_max_ms so validate_levels (below)
    # can cross-check each level's reaction_window_duration_ms against it
    # via info.data - pydantic only exposes already-validated fields
    # there, in declaration order.
    levels: dict[LevelID, LevelConfig]

    @field_validator("response_window_max_ms")
    @classmethod
    def validate_response_window(cls, value: int, info: Any) -> int:
        min_ms = info.data.get("response_window_min_ms")
        if min_ms is not None and value <= min_ms:
            raise ValueError(
                "response_window_max_ms must be greater than "
                "response_window_min_ms"
            )
        return value

    @field_validator("fixation_duration_max_ms")
    @classmethod
    def validate_fixation_duration(cls, value: int, info: Any) -> int:
        min_ms = info.data.get("fixation_duration_min_ms")
        if min_ms is not None and value < min_ms:
            raise ValueError(
                "fixation_duration_max_ms must be greater than or equal "
                "to fixation_duration_min_ms"
            )
        return value

    @field_validator("levels")
    @classmethod
    def validate_levels(
        cls, levels: dict[LevelID, LevelConfig], info: Any
    ) -> dict[LevelID, LevelConfig]:
        validate_level_ids(levels)

        response_window_max_ms = info.data.get("response_window_max_ms")
        if response_window_max_ms is not None:
            for level_id, level_config in levels.items():
                if level_config.reaction_window_duration_ms <= response_window_max_ms:
                    raise ValueError(
                        f"level {level_id}: reaction_window_duration_ms "
                        f"({level_config.reaction_window_duration_ms}) must "
                        "be greater than response_window_max_ms "
                        f"({response_window_max_ms}), otherwise the trial "
                        "could end before a touch within the scored "
                        "response window has a chance to register"
                    )

        return levels


class StoredConfigs(RootModel[dict[AnimalName, StoredConfig]]):
    model_config = ConfigDict(frozen=True)


class TrialResults(RootModel[dict[int, Result]]):
    model_config = ConfigDict(frozen=True)
