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
    "TrialResults",
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
    response_latency_ms: float | None = Field(default=None, ge=0)

    trial_reward_count: int = 0
    trial_manual_reward_count: int = 0

    # Every raw press/release, regardless of whether it qualified as a
    # "touch event" (i.e. met the minimum duration + tolerance-area rule).
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
    def visual_delay_ms(self) -> int:
        return self.stored.visual_delay_ms

    @property
    def reaction_window_duration_ms(self) -> int:
        return self.stored.levels[self.level].reaction_window_duration_ms

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
    def fixation_point_radius_cm(self) -> float:
        return self.stored.levels[self.level].fixation_point_radius_cm

    @property
    def fixation_point_radius_px(self) -> float:
        return self.fixation_point_radius_cm * self.stored.pixels_per_cm

    @property
    def fixation_point_inner_radius_ratio(self) -> float:
        return self.stored.fixation_point_inner_radius_ratio

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
    def pixels_per_cm(self) -> float:
        return self.stored.pixels_per_cm

    @property
    def touch_buffer_cm(self) -> float:
        return self.stored.touch_buffer_cm

    @property
    def touch_buffer_px(self) -> float:
        return self.touch_buffer_cm * self.stored.pixels_per_cm

    @property
    def touch_tolerance_radius_cm(self) -> float:
        return self.stored.touch_tolerance_radius_cm

    @property
    def touch_tolerance_radius_px(self) -> float:
        return self.touch_tolerance_radius_cm * self.stored.pixels_per_cm

    @property
    def min_touch_duration_ms(self) -> int:
        return self.stored.min_touch_duration_ms

    @property
    def fixation_touch_reward_duration_ms(self) -> int:
        return self.stored.fixation_touch_reward_duration_ms


class LevelConfig(BaseModel):
    reaction_window_duration_ms: PositiveInt

    # Independently configurable per level (no lockstep requirement
    # between the two): both are expected to shrink level-over-level in
    # a size-reduction schedule, but neither drives the other.
    target_size_cm: float = Field(gt=0)
    fixation_point_radius_cm: float = Field(gt=0)

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

    visual_delay_ms: NonNegativeInt = 500

    border_width: int = Field(ge=2)
    level_up_threshold: float = Field(ge=0, le=1)
    level_up_trial_count_threshold: PositiveInt

    background_color: RGBColor = (127, 127, 127)
    fixation_point_inner_radius_ratio: float = Field(default=0.35, gt=0, lt=1)

    # Pixels-per-centimeter conversion factor for this display. There is
    # no DPI/physical-size lookup available in this environment, so this
    # must be supplied directly (e.g. from the monitor's spec sheet or a
    # manual ruler-against-screen measurement: pixels_per_cm =
    # screen_width_px / screen_width_cm).
    pixels_per_cm: float = Field(default=56.0, gt=0)

    # Extra hit-box margin around the target image, as a physical
    # distance (centimeters) converted to pixels via pixels_per_cm. A
    # fixed physical buffer better reflects real touch/finger imprecision
    # than one scaled to the stimulus size, since that imprecision is
    # roughly constant regardless of how large the target is drawn.
    touch_buffer_cm: float = Field(default=1.0, ge=0)

    # The press must stay within a circular tolerance zone of this radius
    # (in cm) around its current reference point in order to still count
    # as stable (rather than a drag/smear) - see touch_tolerance_radius_px.
    # A press that drifts beyond this re-anchors its reference point and
    # restarts the stability clock from that new position, rather than
    # being discarded outright.
    touch_tolerance_radius_cm: float = Field(default=0.5, gt=0)

    # Minimum press duration for a touch to be registered at all. 0
    # means a touch counts the instant it's detected - no stability
    # requirement. This is the current default: the goal for now is
    # simply rewarding ANY detected eye signal/touch, on the theory that
    # the animal will learn on its own that a stable, forward-facing
    # posture is what keeps triggering the reward, rather than requiring
    # stability upfront. Raise this later if spurious/noisy signals turn
    # out to need filtering.
    min_touch_duration_ms: NonNegativeInt = 0

    # Reward duration for a qualifying touch detected anywhere on screen
    # during the fixation phase (encourages a stable, engaged posture
    # looking at the screen, distinct from and shorter than the full
    # reward for correctly touching the stimulus). Set to 0 to disable
    # the fixation-phase reward entirely.
    fixation_touch_reward_duration_ms: NonNegativeInt = 500

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