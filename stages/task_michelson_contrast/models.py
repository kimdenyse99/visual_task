from typing import Literal, Self

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
    model_validator,
)

from assets import GIF_STIMULUS_DIR
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
    "ContrastOption",
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
type StimulusSide = Literal["left", "right"]

# How a trial ended, in more detail than ResultEnum:
#   in_window       gaze reached the Gabor inside the reward time window
#   early / late    gaze reached the Gabor before / after the window
#   wrong_side      gaze went to the side without the Gabor
#   no_response     Gabor shown (contrast > 0), gaze never reached a side
#   no_fixation     gaze never settled on the fixation point in time
#   broke_fixation  gaze left the fixation point during the delay
#   withheld        0% contrast, gaze stayed off both sides (rewarded)
#   false_alarm     0% contrast, gaze went to a side
type Response = Literal[
    "in_window",
    "early",
    "late",
    "wrong_side",
    "no_response",
    "no_fixation",
    "broke_fixation",
    "withheld",
    "false_alarm",
]


class Result(BaseModel):
    version: Literal["1.0.0"] = "1.0.0"
    animal: AnimalBaseInfo

    result: ResultEnum = ResultEnum.PENDING
    response: Response | None = None

    trial_start_time: float
    trial_end_time: float

    stimulus_side: StimulusSide
    stimulus_center: tuple[int, int]
    contrast_percent: float
    orientation_deg: float
    spatial_frequency_cpcm: float

    # Fixation point onset -> gaze accepted on it.
    fixation_latency_ms: float | None = Field(default=None, ge=0)
    # Randomized gap between fixation and the Gabor appearing.
    fixation_delay_ms: int | None = Field(default=None, ge=0)
    # Gabor onset -> gaze dwell on a side completed.
    response_latency_ms: float | None = Field(default=None, ge=0)

    # GIF flashed with the reward, if one was given.
    reward_asset: str | None = None

    trial_reward_count: int = 0
    trial_manual_reward_count: int = 0

    touch_event: list[TouchEvent] = Field(default_factory=list[TouchEvent])

    @property
    def is_zero_contrast(self) -> bool:
        return self.contrast_percent == 0

    @property
    def is_fixation_failure(self) -> bool:
        """Ended before the Gabor appeared: never fixated, or left the
        fixation point during the delay."""
        return self.response in ("no_fixation", "broke_fixation")

    @property
    def counts_toward_rate(self) -> bool:
        """Whether this trial is part of the level-up correct rate: the
        Gabor was shown at a visible contrast and the trial wasn't
        cancelled."""
        return not (
            self.result is ResultEnum.CANCELLED
            or self.is_zero_contrast
            or self.is_fixation_failure
        )


class ContrastOption(BaseModel):
    # Michelson contrast of the Gabor's grating, in percent (0 = invisible).
    contrast_percent: float = Field(ge=0, le=100)
    # Relative share of trials at this contrast (need not sum to 1).
    weight: PositiveFloat


class LevelConfig(BaseModel):
    contrasts: list[ContrastOption] = Field(min_length=1)


class StoredConfig(BaseModel):
    # Blank screen after each trial, counted from when the reward GIF
    # disappears (rewarded trials) or from the trial's end (others).
    inter_trial_duration_ms: PositiveInt = 500

    # Reward pump duration; the reward GIF is shown for the same time.
    reward_duration_ms: PositiveInt = 1000
    # Reward GIFs (files in assets/visual_stimulus/gifs_stimulus), one
    # picked at random per reward and shown in the Gabor's place.
    reward_gifs: list[str] = Field(
        default_factory=lambda: ["m001.gif", "m006.gif", "m007.gif", "m009.gif"],
        min_length=1,
    )

    # Gaze must settle on the fixation point within this time, or the
    # trial is a TIMEOUT.
    fixation_timeout_ms: PositiveInt = 5000
    # Random gap between fixation and the Gabor, drawn uniformly per trial.
    # Gaze must stay on the fixation point throughout it.
    fixation_delay_min_ms: NonNegativeInt = 100
    fixation_delay_max_ms: NonNegativeInt = 200
    # Whether the fixation point stays on screen once gaze is accepted on
    # it (through the delay and while the Gabor is shown).
    keep_fixation_point_after_fixation: bool = True

    # Reward time window, from Gabor onset to the moment the gaze dwell on
    # it completes. Earlier (anticipation) or later is not correct.
    response_window_min_ms: NonNegativeInt = 0
    response_window_max_ms: PositiveInt = 500
    # How long the Gabor stays up when gaze reaches neither side; then the
    # trial ends (TIMEOUT, or rewarded for 0% contrast).
    stimulus_duration_ms: PositiveInt = 1000

    # Tolerance areas, measured outward from each shape's edge: the
    # fixation point's circle, and the Gabor (a circle of diameter
    # target_size_cm). 0 accepts only the shape itself.
    fixation_tolerance_cm: float = Field(default=1.0, ge=0)
    stimulus_tolerance_cm: float = Field(default=0.5, ge=0)

    # Gabor patch: side of its square (the Gaussian window's standard
    # deviation is a sixth of this), spatial frequency, and the range a
    # random orientation is drawn from per trial (0 = stripes horizontal
    # on the rotated physical screen, 90 = vertical).
    target_size_cm: float = Field(default=1.5, gt=0)
    spatial_frequency_cpcm: float = Field(default=4.0, gt=0)
    gabor_orientation_min_deg: float = 0.0
    gabor_orientation_max_deg: float = 180.0
    gabor_phase_deg: float = 0.0

    border_width: int = Field(ge=2)
    level_up_threshold: float = Field(ge=0, le=1)
    level_up_trial_count_threshold: PositiveInt

    background_color: RGBColor = (127, 127, 127)
    pixels_per_cm: float = Field(default=46.0, gt=0)

    fixation_point_radius_cm: float = Field(default=0.5, gt=0)
    fixation_point_inner_radius_ratio: float = Field(default=0.40, gt=0, lt=1)

    # Offsets (cm) of the left and right Gabor positions from the fixation
    # point, as in visual_3pt_calibration_v2. The screen is rotated 90
    # degrees counter-clockwise: x is vertical on the physical screen, y
    # is horizontal.
    position_distance_x_cm: float = Field(default=1.5, ge=0)
    position_distance_y_cm: float = Field(default=3.0, ge=0)

    # Gaze must stay inside a tolerance area this long to count. Leaving
    # the area resets it at once (no grace period).
    min_touch_duration_ms: NonNegativeInt = 100
    # How long the eye tracker may lose the gaze signal entirely before
    # the trial is cancelled (see gaze_input.GazeDwellDriver).
    gaze_lost_grace_ms: PositiveInt = 5000

    show_gaze_trace: bool = False
    gaze_trace_duration_ms: NonNegativeInt = 500
    # Outline the tolerance areas on screen (setup aid; no effect on scoring).
    show_tolerance_areas: bool = False

    levels: dict[LevelID, LevelConfig]

    @field_validator("levels")
    @classmethod
    def validate_levels(
        cls, levels: dict[LevelID, LevelConfig]
    ) -> dict[LevelID, LevelConfig]:
        validate_level_ids(levels)
        return levels

    @field_validator("reward_gifs")
    @classmethod
    def validate_reward_gifs(cls, names: list[str]) -> list[str]:
        missing = [name for name in names if not (GIF_STIMULUS_DIR / name).is_file()]
        if missing:
            raise ValueError(f"reward GIFs not found in {GIF_STIMULUS_DIR}: {missing}")
        return names

    @model_validator(mode="after")
    def validate_cross_fields(self) -> Self:
        pairs = (
            ("fixation_delay", self.fixation_delay_min_ms, self.fixation_delay_max_ms),
            (
                "response_window",
                self.response_window_min_ms,
                self.response_window_max_ms,
            ),
            (
                "gabor_orientation",
                self.gabor_orientation_min_deg,
                self.gabor_orientation_max_deg,
            ),
        )
        for name, low, high in pairs:
            if high < low:
                raise ValueError(f"{name}_max must be greater than or equal to the min")
        if self.response_window_max_ms > self.stimulus_duration_ms:
            raise ValueError("response_window_max_ms is longer than stimulus_duration_ms")
        # Above half a cycle per pixel the grating can't be drawn (aliasing).
        if self.spatial_frequency_cpcm > self.pixels_per_cm / 2:
            raise ValueError(
                f"spatial_frequency_cpcm is above {self.pixels_per_cm / 2:g} "
                "cycles/cm, the most this display (pixels_per_cm) can draw"
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
            self.stored.fixation_tolerance_cm
        )

    @property
    def stimulus_tolerance_distance_px(self) -> float:
        """Max distance from the Gabor's center that counts."""
        return self.target_size_px / 2 + self.cm_to_px(self.stored.stimulus_tolerance_cm)


class TrialResults(RootModel[dict[int, Result]]):
    model_config = ConfigDict(frozen=True)
