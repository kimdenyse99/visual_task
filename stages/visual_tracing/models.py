from pathlib import Path
from typing import Literal

from mxbiflow.models.animal import AnimalBaseInfo
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
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

    # How long a position must stay continuously inside the active
    # stimulus's shape (the fixation point, or the target) before it
    # resolves as a touch event. See dwell.DwellTracker - this is a
    # direct, continuous-dwell duration, not a settle-then-hold tolerance
    # window.
    min_touch_duration_ms: NonNegativeInt

    # Whether brief exits from the shape are tolerated at all. False
    # means any single frame outside the shape immediately resets
    # accumulated dwell time to zero (dwell_grace_ms is ignored). True
    # enables the grace period below.
    dwell_grace_enabled: bool = True

    # How long (ms) a position may sit outside the shape before an
    # in-progress dwell is reset to zero, when dwell_grace_enabled is
    # True. A brief excursion shorter than this (one noisy sample, a
    # momentary tracking dropout) does not cost any accumulated time;
    # returning inside resumes the dwell exactly where it left off.
    dwell_grace_ms: NonNegativeInt = 150

    fixation_outer_diameter_cm: PositiveFloat
    fixation_inner_to_outer_ratio: float = Field(gt=0, lt=1)

    # Extra margin (in cm), added outward from the fixation sprite's own
    # visible radius, that still qualifies as "on the fixation point" for
    # scoring purposes. 0 means only the circle's own visible radius
    # counts. Useful because gaze naturally lands some pixels off the
    # exact point it is aimed at even when "held still".
    fixation_touch_reward_radius_cm: NonNegativeFloat = 0.0

    # Same idea as fixation_touch_reward_radius_cm, but for the target
    # during the test phase: extra margin (in cm) added outward from the
    # target's own visible radius that still counts as "on the target".
    target_touch_reward_radius_cm: NonNegativeFloat = 0.0

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
    def dwell_grace_enabled(self) -> bool:
        return self.stored.dwell_grace_enabled

    @property
    def dwell_grace_ms(self) -> float:
        return self.stored.dwell_grace_ms if self.stored.dwell_grace_enabled else 0.0

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
    def fixation_touch_reward_radius_cm(self) -> float:
        return self.stored.fixation_touch_reward_radius_cm

    @property
    def fixation_touch_reward_radius_px(self) -> float:
        return self.fixation_touch_reward_radius_cm * self.pixels_per_cm

    @property
    def target_touch_reward_radius_cm(self) -> float:
        return self.stored.target_touch_reward_radius_cm

    @property
    def target_touch_reward_radius_px(self) -> float:
        return self.target_touch_reward_radius_cm * self.pixels_per_cm

    @property
    def reaction_window_duration_ms(self) -> int:
        return self.stored.levels[self.level].reaction_window_duration_ms

    @property
    def total_levels(self) -> int:
        return len(self.stored.levels)


class TrialResults(RootModel[dict[int, Result]]):
    model_config = ConfigDict(frozen=True)