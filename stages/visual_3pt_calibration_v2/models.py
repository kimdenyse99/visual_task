from typing import Literal

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

# Index into the three candidate fixation positions, labelled by where each
# one appears to the animal on the rotated physical screen (see
# Visual3ptCalibrationV2._fixation_positions).
type PositionName = Literal["left", "right", "center"]


class Result(BaseModel):
    version: Literal["1.0.0"] = "1.0.0"
    animal: AnimalBaseInfo
    is_twitter_trial: bool

    result: ResultEnum = ResultEnum.PENDING

    trial_start_time: float
    trial_end_time: float

    # Which of the three positions the fixation point (and the GIF that
    # replaces it) appeared at on this trial, plus its screen-pixel center.
    fixation_position: PositionName | None = None
    fixation_center: tuple[int, int] | None = None

    # GIF filename played after a successful fixation (None on timeout).
    stimulus_asset: str | None = None

    # Time from the fixation point appearing to the dwell criterion being
    # met. None when the trial timed out or was cancelled.
    fixation_touch_latency_ms: float | None = Field(default=None, ge=0)

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
    def post_reward_stimulus_duration_ms(self) -> int:
        return self.stored.post_reward_stimulus_duration_ms

    @property
    def fixation_timeout_ms(self) -> int:
        return self.stored.fixation_timeout_ms

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
    def fixation_point_radius_cm(self) -> float:
        return self.stored.fixation_point_radius_cm

    @property
    def fixation_point_radius_px(self) -> float:
        return self.fixation_point_radius_cm * self.stored.pixels_per_cm

    @property
    def fixation_point_inner_radius_ratio(self) -> float:
        return self.stored.fixation_point_inner_radius_ratio

    @property
    def fixation_acceptance_radius_cm(self) -> float:
        return self.stored.levels[self.level].fixation_acceptance_radius_cm

    @property
    def fixation_acceptance_radius_px(self) -> float:
        return self.fixation_acceptance_radius_cm * self.stored.pixels_per_cm

    @property
    def fixation_acceptance_distance_px(self) -> float:
        """Max distance (px) from the fixation point's center at which gaze
        still counts: the point's own radius plus the acceptance radius
        measured outward from its perimeter."""
        return self.fixation_point_radius_px + self.fixation_acceptance_radius_px

    @property
    def position_distance_cm(self) -> float:
        return self.stored.position_distance_cm

    @property
    def position_distance_px(self) -> float:
        return self.stored.position_distance_cm * self.stored.pixels_per_cm

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
    def show_gaze_trace(self) -> bool:
        return self.stored.show_gaze_trace

    @property
    def gaze_trace_duration_ms(self) -> int:
        return self.stored.gaze_trace_duration_ms

    @property
    def show_fixation_acceptance_area(self) -> bool:
        return self.stored.show_fixation_acceptance_area


class LevelConfig(BaseModel):
    # Size (width and height) of the GIF that replaces the fixation point.
    target_size_cm: float = Field(gt=0)

    # Extra radius (cm) around the fixation point, measured outward from
    # its perimeter, inside which gaze still counts as "on" the point.
    # Gaze on the point itself always counts; 0 accepts only the point.
    fixation_acceptance_radius_cm: float = Field(default=1.0, ge=0)


class StoredConfig(BaseModel):
    # Blank screen between the end of one trial (GIF removed, or fixation
    # timeout) and the next trial's fixation point appearing.
    inter_trial_duration_ms: PositiveInt = 1500

    # How long (ms) the reward pump runs - passed directly to
    # Rewarder.give_reward(). Given immediately when the dwell criterion
    # on the fixation point is met; there is no delay.
    reward_duration_ms: PositiveInt = 600

    # How long (ms) the GIF plays at the fixation point's position after a
    # successful fixation. Starts at the same moment as the reward, so the
    # two overlap.
    post_reward_stimulus_duration_ms: NonNegativeInt = 1500

    # How long (ms) the fixation point stays up waiting for a qualifying
    # dwell. If it expires first, the trial is a TIMEOUT and goes straight
    # to the inter-trial period.
    fixation_timeout_ms: PositiveInt = 5000

    border_width: int = Field(ge=2)
    level_up_threshold: float = Field(ge=0, le=1)
    level_up_trial_count_threshold: PositiveInt

    background_color: RGBColor = (127, 127, 127)
    fixation_point_radius_cm: float = Field(default=0.5, gt=0)
    fixation_point_inner_radius_ratio: float = Field(default=0.40, gt=0, lt=1)

    # Spacing (cm) of the three candidate fixation positions - the same
    # geometry visual_3pt_calibration uses for its target positions (see
    # Visual3ptCalibrationV2._fixation_positions).
    position_distance_cm: float = Field(default=5.0, gt=0)

    # Pixels-per-centimeter conversion factor for this display (see
    # visual_3pt_calibration's config for how to measure it).
    pixels_per_cm: float = Field(default=56.0, gt=0)

    # How long gaze must dwell inside the acceptance radius before it
    # counts (see dwell.DwellTracker).
    min_touch_duration_ms: NonNegativeInt = 200

    # Whether brief exits from the acceptance radius are tolerated, and
    # for how long, before accumulated dwell resets to zero. See
    # dwell.DwellTracker.
    dwell_grace_enabled: bool = True
    dwell_grace_ms: NonNegativeInt = 150

    # Live fading trace of recent gaze/mouse positions, for debugging.
    show_gaze_trace: bool = False
    gaze_trace_duration_ms: NonNegativeInt = 500

    # Draw the gaze acceptance area (fixation point radius plus
    # fixation_acceptance_radius_cm) as a circle outline around the
    # fixation point while it is on screen. Purely visual; has no effect
    # on dwell/scoring, and is not reported as a stimulus event.
    show_fixation_acceptance_area: bool = False

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


class TrialResults(RootModel[dict[int, Result]]):
    model_config = ConfigDict(frozen=True)
