import math
from typing import Literal, Self

from mxbiflow.models.animal import AnimalBaseInfo
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
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
]

PATH = path(__file__)
LEVEL_TABLE = load_level_table(PATH)

type RGBColor = tuple[int, int, int]

# Where the Gabor appeared on a trial; "none" is a no-stimulus (catch) trial.
type StimulusSide = Literal["left", "right", "none"]

# Why a trial ended the way it did, in more detail than ResultEnum:
#   in_window      gaze reached the stimulus inside the reward time window
#   early / late   gaze reached the stimulus before / after the window
#   wrong_side     gaze went to the empty side instead of the stimulus
#   no_response    stimulus shown, gaze never reached either side
#   held_fixation  no-stimulus trial, gaze stayed on the fixation point
#   broke_fixation no-stimulus trial, gaze left the fixation point
type ResponseTiming = Literal[
    "in_window",
    "early",
    "late",
    "wrong_side",
    "no_response",
    "held_fixation",
    "broke_fixation",
]


class Result(BaseModel):
    version: Literal["1.0.0"] = "1.0.0"
    animal: AnimalBaseInfo

    result: ResultEnum = ResultEnum.PENDING
    response: ResponseTiming | None = None

    trial_start_time: float
    trial_end_time: float

    stimulus_side: StimulusSide
    stimulus_center: tuple[int, int] | None = None
    spatial_frequency_cpcm: float | None = None

    # Fixation point onset -> gaze accepted on it.
    fixation_latency_ms: float | None = Field(default=None, ge=0)
    # Randomized gap between fixation and the stimulus phase.
    fixation_delay_ms: int | None = Field(default=None, ge=0)
    # Stimulus onset -> gaze dwell on a side completed (stimulus trials).
    response_latency_ms: float | None = Field(default=None, ge=0)

    trial_reward_count: int = 0
    trial_manual_reward_count: int = 0

    touch_event: list[TouchEvent] = Field(default_factory=list[TouchEvent])


class LevelConfig(BaseModel):
    # Reward time window, measured from stimulus onset to the moment the
    # gaze dwell on the stimulus completes. Gaze on the stimulus before
    # the min (anticipation) or after the max does not count as correct.
    response_window_min_ms: NonNegativeInt
    response_window_max_ms: PositiveInt

    # Spatial frequencies (cycles/cm) shown at this level: every value
    # from min to max in steps of increment, one picked at random per
    # stimulus trial.
    spatial_frequency_min_cpcm: float = Field(gt=0)
    spatial_frequency_max_cpcm: float = Field(gt=0)
    spatial_frequency_increment_cpcm: float = Field(gt=0)

    # Tolerance areas, each measured outward from the edge of the shape:
    # the fixation point's circle, and the Gabor patch (treated as a
    # circle of diameter target_size_cm). 0 accepts only the shape itself.
    fixation_tolerance_cm: float = Field(ge=0)
    stimulus_tolerance_cm: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if self.response_window_max_ms < self.response_window_min_ms:
            raise ValueError(
                "response_window_max_ms must be greater than or equal to "
                "response_window_min_ms"
            )
        if self.spatial_frequency_max_cpcm < self.spatial_frequency_min_cpcm:
            raise ValueError(
                "spatial_frequency_max_cpcm must be greater than or equal to "
                "spatial_frequency_min_cpcm"
            )
        return self

    def spatial_frequencies_cpcm(self) -> list[float]:
        """min, min + increment, ... up to and including max (a max that is
        not an exact step from min is not included)."""
        steps = math.floor(
            (self.spatial_frequency_max_cpcm - self.spatial_frequency_min_cpcm)
            / self.spatial_frequency_increment_cpcm
            + 1e-9
        )
        return [
            round(
                self.spatial_frequency_min_cpcm
                + i * self.spatial_frequency_increment_cpcm,
                6,
            )
            for i in range(steps + 1)
        ]


class StoredConfig(BaseModel):
    inter_trial_duration_ms: PositiveInt = 1500
    reward_duration_ms: PositiveInt = 1000

    # Randomized gap between gaze being accepted on the fixation point
    # and the stimulus phase starting, drawn uniformly per trial.
    fixation_delay_min_ms: NonNegativeInt = 1000
    fixation_delay_max_ms: NonNegativeInt = 2000

    # Whether the fixation point stays on screen once gaze has been
    # accepted on it (through the delay and the stimulus phase). False
    # removes it at that moment. It is always shown before that.
    keep_fixation_point_after_fixation: bool = True

    # How long the Gabor stays up on stimulus trials if gaze never
    # completes a dwell on either side. Must be at least every level's
    # response_window_max_ms. No-stimulus trials end at the level's
    # response_window_max_ms instead.
    stimulus_duration_ms: PositiveInt = 2000

    # Relative weights of the three trial types (they need not sum to 1).
    stimulus_left_weight: NonNegativeFloat = 1.0
    stimulus_right_weight: NonNegativeFloat = 1.0
    stimulus_none_weight: NonNegativeFloat = 1.0

    # Gabor patch: side of its square in cm (the Gaussian window's
    # standard deviation is a sixth of this), Michelson contrast at its
    # center relative to background_color, and grating orientation/phase.
    target_size_cm: float = Field(default=1.5, gt=0)
    gabor_contrast: float = Field(default=0.5, ge=0, le=1)
    gabor_orientation_deg: float = 0.0
    gabor_phase_deg: float = 0.0

    border_width: int = Field(ge=2)
    level_up_threshold: float = Field(ge=0, le=1)
    level_up_trial_count_threshold: PositiveInt

    background_color: RGBColor = (127, 127, 127)
    pixels_per_cm: float = Field(default=46.0, gt=0)

    fixation_point_radius_cm: float = Field(default=0.5, gt=0)
    fixation_point_inner_radius_ratio: float = Field(default=0.40, gt=0, lt=1)

    # Offsets (cm) of the left and right stimulus positions from the
    # fixation point, as in visual_3pt_calibration_v2. The screen is
    # mounted rotated 90 degrees counter-clockwise: x is vertical on the
    # physical screen, y is horizontal.
    position_distance_x_cm: float = Field(default=1.5, ge=0)
    position_distance_y_cm: float = Field(default=3.0, ge=0)

    # Gaze must stay inside a tolerance area this long to count.
    min_touch_duration_ms: NonNegativeInt = 100
    # Brief exits from a tolerance area shorter than this are ignored.
    dwell_grace_enabled: bool = True
    dwell_grace_ms: NonNegativeInt = 150
    # How long the eye tracker may lose the gaze entirely before the trial
    # is cancelled (see gaze_input.GazeDwellDriver).
    gaze_lost_grace_ms: PositiveInt = 5000

    show_gaze_trace: bool = False
    gaze_trace_duration_ms: NonNegativeInt = 500
    # Draw the fixation and stimulus tolerance areas as circle outlines
    # (for setting up; has no effect on scoring).
    show_tolerance_areas: bool = False

    levels: dict[LevelID, LevelConfig]

    @field_validator("levels")
    @classmethod
    def validate_levels(
        cls, levels: dict[LevelID, LevelConfig]
    ) -> dict[LevelID, LevelConfig]:
        validate_level_ids(levels)
        return levels

    @model_validator(mode="after")
    def validate_cross_fields(self) -> Self:
        if self.fixation_delay_max_ms < self.fixation_delay_min_ms:
            raise ValueError(
                "fixation_delay_max_ms must be greater than or equal to "
                "fixation_delay_min_ms"
            )
        weights = (
            self.stimulus_left_weight,
            self.stimulus_right_weight,
            self.stimulus_none_weight,
        )
        if sum(weights) <= 0:
            raise ValueError("at least one stimulus_*_weight must be above 0")
        # Above half a cycle per pixel the grating can't be drawn (aliasing).
        nyquist_cpcm = self.pixels_per_cm / 2
        for level_id, level in self.levels.items():
            if level.response_window_max_ms > self.stimulus_duration_ms:
                raise ValueError(
                    f"level {level_id}: response_window_max_ms is longer than "
                    "stimulus_duration_ms"
                )
            if level.spatial_frequency_max_cpcm > nyquist_cpcm:
                raise ValueError(
                    f"level {level_id}: spatial_frequency_max_cpcm is above "
                    f"{nyquist_cpcm:g} cycles/cm, the most this display "
                    "(pixels_per_cm) can draw"
                )
        return self


class StoredConfigs(RootModel[dict[AnimalName, StoredConfig]]):
    model_config = ConfigDict(frozen=True)


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    animal: AnimalName
    level: LevelID
    screen_size: Screen

    stored: StoredConfig = Field(repr=False, exclude=True)

    @property
    def level_config(self) -> LevelConfig:
        return self.stored.levels[self.level]

    @property
    def total_levels(self) -> int:
        return len(self.stored.levels)

    def cm_to_px(self, cm: float) -> float:
        return cm * self.stored.pixels_per_cm

    @property
    def fixation_point_radius_px(self) -> float:
        return self.cm_to_px(self.stored.fixation_point_radius_cm)

    @property
    def target_size_px(self) -> float:
        return self.cm_to_px(self.stored.target_size_cm)

    @property
    def fixation_tolerance_distance_px(self) -> float:
        """Max distance from the fixation point's center that counts."""
        return self.fixation_point_radius_px + self.cm_to_px(
            self.level_config.fixation_tolerance_cm
        )

    @property
    def stimulus_tolerance_distance_px(self) -> float:
        """Max distance from the stimulus's center that counts."""
        return self.target_size_px / 2 + self.cm_to_px(
            self.level_config.stimulus_tolerance_cm
        )

    @property
    def dwell_grace_ms(self) -> float:
        return self.stored.dwell_grace_ms if self.stored.dwell_grace_enabled else 0.0


class TrialResults(RootModel[dict[int, Result]]):
    model_config = ConfigDict(frozen=True)
