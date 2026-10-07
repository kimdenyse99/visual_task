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
    "TrialStimulus",
]

PATH = path(__file__)
LEVEL_TABLE = load_level_table(PATH)


class TrialStimulus(BaseModel):
    model_config = ConfigDict(frozen=True)

    target_size_px: PositiveInt
    reward_delay_ms: NonNegativeInt


class Result(BaseModel):
    version: Literal["1.0.0"] = "1.0.0"
    animal: AnimalBaseInfo
    screen_size: Screen
    stimulus: TrialStimulus
    result: ResultEnum = ResultEnum.PENDING
    trial_start_time: float
    trial_end_time: float
    response_latency_ms: float | None = Field(default=None, ge=0)
    trial_reward_count: int = 0
    trial_manual_reward_count: int = 0
    touch_event: list[TouchEvent] = Field(default_factory=list[TouchEvent])


class LevelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    target_size_px: PositiveInt
    reward_delay_ms: NonNegativeInt


class StoredConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    reaction_window_duration_ms: PositiveInt
    reward_duration_ms: PositiveInt
    # Minimum stretch of time with zero touch/click events required before the
    # next trial's target is shown. Any touch during the wait resets the
    # clock, so this can take longer than a fixed inter-trial interval.
    # touch_quiet_period_enabled is the actual on/off switch - when False,
    # this gating is skipped entirely, but inter_trial_duration_ms below
    # still applies as a uniform minimum pause regardless of outcome.
    touch_quiet_period_ms: PositiveInt
    touch_quiet_period_enabled: bool = True
    # Minimum pause before the next trial, applied to every outcome (correct,
    # incorrect, background-touched, timeout). This is what actually governs
    # pacing when touch_quiet_period_enabled is False; a correct trial's
    # floor is further extended if the marmoset flash needs more time than
    # this to finish naturally (see _finish / _give_automatic_reward_and_finish).
    inter_trial_duration_ms: PositiveInt = 2000
    border_width: int = Field(default=40, ge=2)

    # Marmoset image flash shown a short delay after a correct trial's
    # reward, at one of three fixed points around the target center (above,
    # below, right), chosen at random, using a randomly chosen image from
    # assets.MARMOSET_IMAGE_PATHS. Stays up for marmoset_flash_duration_ms,
    # independent of reward_duration_ms, touch_quiet_period_enabled, and
    # unaffected by touches, correct or not.
    # pixels_per_cm is a property of the display rig, not the stimulus, and
    # should be measured against the actual monitor in use.
    pixels_per_cm: PositiveFloat
    marmoset_image_size_cm: PositiveFloat
    marmoset_flash_delay_ms: PositiveInt = 200
    marmoset_flash_duration_ms: PositiveInt = 1000
    marmoset_offset_cm: PositiveFloat  # distance from target center to each flash point

    level_up_threshold: float = Field(default=0.8, ge=0, le=1)
    level_up_trial_count_threshold: PositiveInt = 20
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
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    animal: AnimalName
    level: LevelID
    screen_size: Screen
    stimulus: TrialStimulus
    stored: StoredConfig = Field(repr=False, exclude=True)

    @property
    def total_levels(self) -> int:
        return len(self.stored.levels)


class TrialResults(RootModel[dict[int, Result]]):
    model_config = ConfigDict(frozen=True)
