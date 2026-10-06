from pathlib import Path
from typing import Literal

from mxbiflow.models.animal import AnimalBaseInfo
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
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
    "TrialResults",
]

PATH = path(__file__)
LEVEL_TABLE = load_level_table(PATH)


class Result(BaseModel):
    version: Literal["1.0.0"] = "1.0.0"
    animal: AnimalBaseInfo

    result: ResultEnum = ResultEnum.PENDING

    trial_start_time: float
    trial_end_time: float

    # Center of the target circle for this trial, in pixels.
    target_x: int
    target_y: int
    target_diameter_px: int

    fixation_touch_latency_ms: float | None = Field(default=None, ge=0)
    response_latency_ms: float | None = Field(default=None, ge=0)

    trial_reward_count: int = 0
    trial_manual_reward_count: int = 0

    # Every raw press/release, regardless of whether it qualified as a
    # "touch event" (i.e. met the minimum duration + tolerance-area rule).
    touch_event: list[TouchEvent] = Field(default_factory=list[TouchEvent])


class LevelConfig(BaseModel):
    target_diameter_cm: PositiveFloat
    reaction_window_duration_ms: PositiveInt


class StoredConfig(BaseModel):
    inter_trial_duration_ms: PositiveInt
    border_width: int = Field(ge=2)

    pixels_per_cm: PositiveFloat
    reward_duration_ms: PositiveInt

    # A press shorter than this is not considered a touch event at all.
    min_touch_duration_ms: NonNegativeInt

    # The press must stay within a circular tolerance zone of this radius
    # (in cm) around its current reference point in order to still count
    # as stable (rather than a drag/smear) — see touch_tolerance_radius_px.
    touch_tolerance_radius_cm: PositiveFloat

    fixation_outer_diameter_cm: PositiveFloat
    fixation_inner_to_outer_ratio: float = Field(gt=0, lt=1)

    touch_trace_duration_ms: NonNegativeInt

    levels: dict[LevelID, LevelConfig]

    @field_validator("levels")
    @classmethod
    def validate_levels(
        cls, levels: dict[LevelID, LevelConfig]
    ) -> dict[LevelID, LevelConfig]:
        validate_level_ids(levels)
        return levels


class StoredConfigs(RootModel[dict[AnimalName, StoredConfig]]):
    model_config = ConfigDict(frozen=True)


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    animal: AnimalName
    level: LevelID

    screen_size: Screen

    stored: StoredConfig = Field(repr=False, exclude=True)

    @property
    def inter_trial_duration_ms(self) -> int:
        return self.stored.inter_trial_duration_ms

    @property
    def border_width(self) -> int:
        return self.stored.border_width

    @property
    def pixels_per_cm(self) -> float:
        return self.stored.pixels_per_cm

    @property
    def reward_duration_ms(self) -> int:
        return self.stored.reward_duration_ms

    @property
    def min_touch_duration_ms(self) -> int:
        return self.stored.min_touch_duration_ms

    @property
    def touch_tolerance_radius_cm(self) -> float:
        return self.stored.touch_tolerance_radius_cm

    @property
    def touch_tolerance_radius_px(self) -> float:
        return self.touch_tolerance_radius_cm * self.pixels_per_cm

    @property
    def fixation_outer_diameter_cm(self) -> float:
        return self.stored.fixation_outer_diameter_cm

    @property
    def fixation_inner_to_outer_ratio(self) -> float:
        return self.stored.fixation_inner_to_outer_ratio

    @property
    def touch_trace_duration_ms(self) -> int:
        return self.stored.touch_trace_duration_ms

    @property
    def target_diameter_cm(self) -> float:
        return self.stored.levels[self.level].target_diameter_cm

    @property
    def target_diameter_px(self) -> int:
        return max(1, round(self.target_diameter_cm * self.pixels_per_cm))

    @property
    def fixation_outer_diameter_px(self) -> int:
        return max(1, round(self.fixation_outer_diameter_cm * self.pixels_per_cm))

    @property
    def reaction_window_duration_ms(self) -> int:
        return self.stored.levels[self.level].reaction_window_duration_ms

    @property
    def total_levels(self) -> int:
        return len(self.stored.levels)


class TrialResults(RootModel[dict[int, Result]]):
    model_config = ConfigDict(frozen=True)
