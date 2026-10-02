from typing import Literal, Self

from mxbiflow.models.animal import AnimalBaseInfo
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    RootModel,
    field_validator,
    model_validator,
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
    "TrialStimulus",
]

PATH = path(__file__)
LEVEL_TABLE = load_level_table(PATH)


class TrialStimulus(BaseModel):
    model_config = ConfigDict(frozen=True)

    pure_tone_frequency_hz: int
    pure_tone_duration_ms: int


class Result(BaseModel):
    version: Literal["1.0.0"] = "1.0.0"
    animal: AnimalBaseInfo
    is_twitter_trial: bool
    target_side: Literal["left", "right"]
    stimulus: TrialStimulus

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
    pure_tone_frequency_hz: int
    pure_tone_duration_ms: int

    screen_size: Screen

    stored: StoredConfig = Field(repr=False, exclude=True)

    @property
    def target_side(self) -> Literal["left", "right"]:
        return "left" if self.is_twitter_trial else "right"

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
    def visual_delay_ms(self) -> int:
        return self.stored.visual_delay_ms

    @property
    def reaction_window_duration_ms(self) -> int:
        return self.stored.levels[self.level].reaction_window_duration_ms

    @property
    def position_offset_ratio(self) -> float:
        return self.stored.levels[self.level].position_offset_ratio

    @property
    def reaction_timeout_enabled(self) -> bool:
        return self.stored.levels[self.level].reaction_timeout_enabled

    @property
    def pure_tone_unit_duration_ms(self) -> int:
        return self.stored.pure_tone_unit_duration_ms

    @property
    def pure_tone_interval_ms(self) -> int:
        return self.stored.pure_tone_interval_ms

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


class LevelConfig(BaseModel):
    twitter_trial_probability: float = Field(ge=0, le=1)
    reaction_window_duration_ms: PositiveInt
    position_offset_ratio: float = Field(ge=0, le=1)
    reaction_timeout_enabled: bool


class StoredConfig(BaseModel):
    inter_trial_duration_ms: PositiveInt
    reward_duration_ms: PositiveInt

    attention_duration_ms: NonNegativeInt
    visual_delay_ms: NonNegativeInt = 500

    pure_tone_frequency_range_hz: tuple[PositiveInt, PositiveInt]
    pure_tone_unit_duration_ms: PositiveInt
    pure_tone_interval_ms: NonNegativeInt
    pure_tone_duration_range_ms: tuple[PositiveInt, PositiveInt]

    levels: dict[LevelID, LevelConfig]

    border_width: int = Field(ge=2)
    level_up_threshold: float = Field(ge=0, le=1)
    level_up_trial_count_threshold: PositiveInt

    @field_validator("levels")
    @classmethod
    def validate_levels(
        cls, levels: dict[LevelID, LevelConfig]
    ) -> dict[LevelID, LevelConfig]:
        validate_level_ids(levels)
        return levels

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        for field_name, (lower, upper) in (
            ("pure_tone_frequency_range_hz", self.pure_tone_frequency_range_hz),
            ("pure_tone_duration_range_ms", self.pure_tone_duration_range_ms),
        ):
            if lower > upper:
                raise ValueError(
                    f"{field_name} lower bound must not exceed upper bound"
                )
        return self


class StoredConfigs(RootModel[dict[AnimalName, StoredConfig]]):
    model_config = ConfigDict(frozen=True)


class TrialResults(RootModel[dict[int, Result]]):
    model_config = ConfigDict(frozen=True)
