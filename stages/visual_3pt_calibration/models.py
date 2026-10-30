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
    def fixation_duration_min_ms(self) -> int:
        return self.stored.fixation_duration_min_ms

    @property
    def fixation_duration_max_ms(self) -> int:
        return self.stored.fixation_duration_max_ms

    @property
    def reward_delay_ms(self) -> int:
        return self.stored.levels[self.level].reward_delay_ms

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
    def target_distances_cm(self) -> float:
        return self.stored.target_distances_cm

    @property
    def target_distances_px(self) -> float:
        return self.stored.target_distances_cm * self.stored.pixels_per_cm

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
    def target_touch_reward_enabled(self) -> bool:
        return self.stored.target_touch_reward_enabled


class LevelConfig(BaseModel):
    target_size_cm: float = Field(gt=0)

    # Fixed per-level delay between a correct touch and the reward
    # actually being dispensed. The marmoset image stays visible for
    # this whole delay, then stays visible for a further
    # reward_duration_ms while the reward itself is administered - so
    # the total time the marmoset remains on screen after a correct
    # touch is reward_delay_ms + reward_duration_ms. This teaches the
    # animal to associate looking at the marmoset specifically (not just
    # touching blindly) with the reward that follows. 0 means the
    # reward fires immediately on a correct touch, same as before this
    # field existed.
    reward_delay_ms: NonNegativeInt = 0


class StoredConfig(BaseModel):
    inter_trial_duration_ms: PositiveInt
    reward_duration_ms: PositiveInt

    # Fixation screen duration is randomized per-trial, uniformly drawn
    # from [fixation_duration_min_ms, fixation_duration_max_ms].
    fixation_duration_min_ms: NonNegativeInt = 150
    fixation_duration_max_ms: NonNegativeInt = 300

    border_width: int = Field(ge=2)
    level_up_threshold: float = Field(ge=0, le=1)
    level_up_trial_count_threshold: PositiveInt

    background_color: RGBColor = (127, 127, 127)
    fixation_point_radius_cm: float = Field(default=0.25, gt=0)
    fixation_point_inner_radius_ratio: float = Field(default=0.40, gt=0, lt=1)

    # Distance from the fixation point to each of the 3 target positions,
    # as a physical distance (centimeters) converted to pixels via
    # pixels_per_cm. Previously a ratio of screen width; switched to a
    # fixed physical distance for the same reason as target_size_cm - it
    # stays consistent regardless of screen size/resolution, rather than
    # scaling with screen width.
    target_distances_cm: float = Field(default=3.0, gt=0)

    # Pixels-per-centimeter conversion factor for this display. There is
    # no DPI/physical-size lookup available in this environment, so this
    # must be supplied directly (e.g. from the monitor's spec sheet or a
    # manual ruler-against-screen measurement: pixels_per_cm =
    # screen_width_px / screen_width_cm).
    pixels_per_cm: float = Field(default=56.0, gt=0)

    # Minimum press duration (button/finger down -> up) for a touch to be
    # registered at all. Shorter presses (e.g. accidental brushes) are
    # discarded before scoring.
    min_touch_duration_ms: NonNegativeInt = 100

    # Whether brief exits from the active shape (fixation point or
    # target) are tolerated at all. False means any single frame outside
    # the shape immediately resets accumulated dwell time to zero
    # (dwell_grace_ms is ignored). True enables the grace period below.
    # See dwell.DwellTracker.
    dwell_grace_enabled: bool = True

    # How long (ms) a position may sit outside the active shape before
    # an in-progress dwell is reset to zero, when dwell_grace_enabled is
    # True. A brief excursion shorter than this (one noisy sample, a
    # momentary tracking dropout) does not cost any accumulated time;
    # returning inside resumes the dwell exactly where it left off.
    dwell_grace_ms: NonNegativeInt = 150

    # Whether to draw a live, fading red trace showing recent gaze/mouse
    # positions on screen - useful for visually verifying where input is
    # actually landing. Purely visual; has no effect on dwell/scoring.
    show_gaze_trace: bool = False

    # How long (ms) a point in the trace stays visible before fully
    # fading out. Only meaningful when show_gaze_trace is True.
    gaze_trace_duration_ms: NonNegativeInt = 500

    # Whether the reward is actually dispensed once the GIF stimulus
    # appears (see Visual3ptCalibration._show_visual_stimulus /
    # _on_correct). Every trial that reaches the stimulus phase resolves
    # CORRECT either way - this only gates whether _give_reward() is
    # actually called.
    target_touch_reward_enabled: bool = True

    levels: dict[LevelID, LevelConfig]

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
        cls, levels: dict[LevelID, LevelConfig]
    ) -> dict[LevelID, LevelConfig]:
        validate_level_ids(levels)
        return levels


class StoredConfigs(RootModel[dict[AnimalName, StoredConfig]]):
    model_config = ConfigDict(frozen=True)


class TrialResults(RootModel[dict[int, Result]]):
    model_config = ConfigDict(frozen=True)