# pyright: reportPrivateUsage=false

import json
import unittest
from pathlib import Path
from typing import Any, Self
from unittest.mock import Mock, patch

import numpy as np
from mxbiflow.models.animal import AnimalBaseInfo
from pydantic import ValidationError
from pygame.sprite import Group

from gabor_sprite import GaborSprite, gabor_pixels
from stages.task_spatial_frequency import TaskSpatialFrequency
from stages.task_spatial_frequency.models import (
    Config,
    LevelConfig,
    Result,
    ResultEnum,
    Screen,
    StimulusSide,
    StoredConfig,
    StoredConfigs,
    TrialResults,
)
from stimulus_events import StimulusEvents

CONFIG_PATH = Path("stages/task_spatial_frequency/config.json")
SCENE = "stages.task_spatial_frequency.scene"

# Writing the target-shown CSV needs a real session; tests that care about
# it patch it again to inspect the call.
_target_shown_patch = patch(f"{SCENE}.write_target_shown_event")


def setUpModule() -> None:
    _target_shown_patch.start()


def tearDownModule() -> None:
    _target_shown_patch.stop()


def make_level(**overrides: Any) -> LevelConfig:
    values: dict[str, Any] = {
        "response_window_min_ms": 300,
        "response_window_max_ms": 800,
        "spatial_frequency_min_cpcm": 1.0,
        "spatial_frequency_max_cpcm": 4.0,
        "spatial_frequency_increment_cpcm": 0.5,
        "fixation_tolerance_cm": 1.0,
        "stimulus_tolerance_cm": 0.5,
    }
    values.update(overrides)
    return LevelConfig(**values)


def make_stored_config(**overrides: Any) -> StoredConfig:
    values: dict[str, Any] = {
        "inter_trial_duration_ms": 1500,
        "reward_duration_ms": 1000,
        "border_width": 40,
        "level_up_threshold": 0.8,
        "level_up_trial_count_threshold": 3,
        "pixels_per_cm": 46.0,
        "min_touch_duration_ms": 100,
        "dwell_grace_ms": 150,
        "stimulus_duration_ms": 2000,
        "levels": {1: make_level()},
    }
    values.update(overrides)
    return StoredConfig(**values)


def make_config(level: int = 1, **overrides: Any) -> Config:
    return Config(
        animal="test-animal",
        level=level,
        screen_size=Screen(width=1920, height=1080),
        stored=make_stored_config(**overrides),
    )


def make_result(level: int = 1, trial_id: int = 1, **fields: Any) -> Result:
    values: dict[str, Any] = {"stimulus_side": "left"}
    values.update(fields)
    return Result(
        animal=AnimalBaseInfo(
            animal="test-animal",
            trial_id=trial_id,
            level=level,
            level_trial_id=trial_id,
            animal_session_id=1,
            animal_session_trial_id=trial_id,
        ),
        trial_start_time=0.0,
        trial_end_time=0.0,
        **values,
    )


def make_scene(side: StimulusSide = "left", **config_overrides: Any) -> Any:
    """A scene with real stimuli and dwell logic but mocked hardware."""
    scene: Any = TaskSpatialFrequency.__new__(TaskSpatialFrequency)
    scene._config = make_config(**config_overrides)
    scene._timer = Mock()
    scene._rewarder = Mock()
    scene._mxbiflow = Mock()
    scene._stimulus_events = StimulusEvents(scene._mxbiflow)
    scene._animal = Mock()
    scene._animal.trial_id = 1
    scene._animal.current_stage.level = 1
    scene._animal.name = "test-animal"
    scene._running = True
    scene._phase = "blank"
    scene._stimulus_end_task = None
    scene._fixation_shown_at_ms = 0.0
    scene._stimulus_onset_ms = 0.0
    scene._outside_fixation_since_ms = None
    scene._trackers = {}
    scene._trajectories = {}
    scene._trace_points = []
    scene._control_buttons = Group()
    scene._gaze_driver = Mock()
    with patch.object(TaskSpatialFrequency, "_pick_stimulus_side", return_value=side):
        scene._init_visual_stimulus(scene._config)
    scene._result = make_result(stimulus_side=side)
    return scene


class Clock:
    """Drives time.monotonic in both the scene and dwell.DwellTracker."""

    def __init__(self) -> None:
        self.now_s = 1000.0
        self._patches = [
            patch("dwell.monotonic", self),
            patch(f"{SCENE}.monotonic", self),
        ]

    def __call__(self) -> float:
        return self.now_s

    def advance_ms(self, ms: float) -> None:
        self.now_s += ms / 1000

    def __enter__(self) -> Self:
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *_: object) -> None:
        for p in self._patches:
            p.stop()


def timer_calls(scene: Any) -> list[tuple[int, Any]]:
    return [(c.args[0], c.args[1]) for c in scene._timer.after.call_args_list]


def start_stimulus_phase(scene: Any, clock: Clock) -> None:
    """Fixate, then run the delay timer's callback immediately."""
    scene._on_trial_start()
    scene._on_press(-1, scene._positions["center"])
    clock.advance_ms(100)
    scene._check_all_trackers()
    assert scene._phase == "delay"
    scene._start_stimulus_phase()


class GaborTests(unittest.TestCase):
    def test_center_and_mean_match_contrast_and_background(self) -> None:
        pixels = gabor_pixels(69, 0.0, 0.5, (127, 127, 127)).astype(float)
        # 0 cycles/px with phase 0 is a pure Gaussian bump: peak = bg * 1.5.
        self.assertEqual(pixels[34, 34, 0], round(127 * 1.5))
        self.assertEqual(pixels.shape, (69, 69, 3))

    def test_grating_amplitude_is_the_contrast(self) -> None:
        bg = 127
        pixels = gabor_pixels(201, 0.05, 0.5, (bg, bg, bg)).astype(float)[:, :, 0]
        center_row = pixels[:, 100]
        # Near the center the envelope is ~1, so peak and trough are close
        # to bg * (1 +/- 0.5): Michelson contrast of the grating = 0.5.
        peak = center_row[100]
        trough = center_row[110]  # half a cycle (10 px) from the center
        self.assertAlmostEqual(peak, bg * 1.5, delta=1)
        self.assertAlmostEqual((peak - trough) / (peak + trough), 0.5, delta=0.03)

    def test_edges_fade_to_background(self) -> None:
        pixels = gabor_pixels(69, 4 / 46, 0.5, (127, 127, 127)).astype(int)
        self.assertTrue(np.all(np.abs(pixels[0, :, :] - 127) <= 1))
        self.assertTrue(np.all(np.abs(pixels[:, 0, :] - 127) <= 1))

    def test_zero_contrast_is_uniform_background(self) -> None:
        pixels = gabor_pixels(31, 0.1, 0.0, (120, 130, 140))
        self.assertTrue(np.all(pixels == np.array([120, 130, 140])))

    def test_orientation_rotates_the_grating(self) -> None:
        along_x = gabor_pixels(61, 0.1, 0.5, (127, 127, 127), orientation_deg=0)
        along_y = gabor_pixels(61, 0.1, 0.5, (127, 127, 127), orientation_deg=90)
        np.testing.assert_allclose(along_x[:, :, 0], along_y[:, :, 0].T, atol=1)

    def test_rejects_bad_parameters(self) -> None:
        with self.assertRaises(ValueError):
            gabor_pixels(0, 0.1, 0.5, (127, 127, 127))
        with self.assertRaises(ValueError):
            gabor_pixels(10, 0.1, 1.5, (127, 127, 127))

    def test_sprite_is_centered(self) -> None:
        sprite = GaborSprite((100, 200), 69, 0.1, 0.5, (127, 127, 127))
        assert sprite.rect is not None
        self.assertEqual(sprite.rect.center, (100, 200))
        self.assertEqual(sprite.rect.size, (69, 69))


class ConfigFileTests(unittest.TestCase):
    def _stored(self) -> StoredConfig:
        data = json.loads(CONFIG_PATH.read_text())
        return StoredConfigs.model_validate(data).root["default"]

    def test_default_config_values(self) -> None:
        stored = self._stored()
        self.assertEqual(list(stored.levels), [1, 2, 3, 4])
        self.assertEqual(stored.target_size_cm, 1.5)
        self.assertEqual(stored.gabor_contrast, 0.5)
        self.assertEqual(stored.dwell_grace_ms, 150)
        self.assertEqual(stored.min_touch_duration_ms, 100)
        self.assertEqual(stored.gaze_lost_grace_ms, 5000)
        self.assertEqual(
            (stored.fixation_delay_min_ms, stored.fixation_delay_max_ms), (1000, 2000)
        )
        self.assertEqual(stored.level_up_threshold, 0.8)
        self.assertEqual(stored.level_up_trial_count_threshold, 200)

    def test_every_level_sets_window_frequencies_and_tolerances(self) -> None:
        data = json.loads(CONFIG_PATH.read_text())["default"]
        for level in data["levels"].values():
            self.assertEqual(
                set(level),
                {
                    "response_window_min_ms",
                    "response_window_max_ms",
                    "spatial_frequency_min_cpcm",
                    "spatial_frequency_max_cpcm",
                    "spatial_frequency_increment_cpcm",
                    "fixation_tolerance_cm",
                    "stimulus_tolerance_cm",
                },
            )

    def test_spatial_frequency_steps(self) -> None:
        level = make_level(
            spatial_frequency_min_cpcm=1.0,
            spatial_frequency_max_cpcm=2.0,
            spatial_frequency_increment_cpcm=0.25,
        )
        self.assertEqual(level.spatial_frequencies_cpcm(), [1.0, 1.25, 1.5, 1.75, 2.0])

    def test_spatial_frequency_max_off_the_step_is_not_exceeded(self) -> None:
        level = make_level(
            spatial_frequency_min_cpcm=1.0,
            spatial_frequency_max_cpcm=2.2,
            spatial_frequency_increment_cpcm=0.5,
        )
        self.assertEqual(level.spatial_frequencies_cpcm(), [1.0, 1.5, 2.0])

    def test_single_spatial_frequency(self) -> None:
        level = make_level(spatial_frequency_min_cpcm=3, spatial_frequency_max_cpcm=3)
        self.assertEqual(level.spatial_frequencies_cpcm(), [3.0])

    def test_invalid_level_ranges_are_rejected(self) -> None:
        for bad in (
            {"response_window_min_ms": 900, "response_window_max_ms": 800},
            {"spatial_frequency_min_cpcm": 5.0, "spatial_frequency_max_cpcm": 4.0},
            {"spatial_frequency_increment_cpcm": 0},
            {"fixation_tolerance_cm": -1},
            {"stimulus_tolerance_cm": -1},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                make_level(**bad)

    def test_invalid_stage_settings_are_rejected(self) -> None:
        for bad in (
            {"fixation_delay_min_ms": 2000, "fixation_delay_max_ms": 1000},
            {
                "stimulus_left_weight": 0,
                "stimulus_right_weight": 0,
                "stimulus_none_weight": 0,
            },
            {"stimulus_duration_ms": 500},  # shorter than the 800 ms window
            {"levels": {1: make_level(spatial_frequency_max_cpcm=30)}},  # > 23
            {"gabor_contrast": 1.5},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                make_stored_config(**bad)

    def test_tolerance_distances_are_measured_from_the_edges(self) -> None:
        config = make_config(
            pixels_per_cm=40.0,
            fixation_point_radius_cm=0.5,
            target_size_cm=1.5,
            levels={1: make_level(fixation_tolerance_cm=1.0, stimulus_tolerance_cm=0.5)},
        )
        self.assertAlmostEqual(config.fixation_tolerance_distance_px, (0.5 + 1.0) * 40)
        self.assertAlmostEqual(config.stimulus_tolerance_distance_px, (0.75 + 0.5) * 40)

    def test_level_settings_follow_the_level(self) -> None:
        levels = {
            1: make_level(fixation_tolerance_cm=1.0),
            2: make_level(fixation_tolerance_cm=0.0),
        }
        self.assertEqual(
            make_config(level=2, levels=levels).level_config.fixation_tolerance_cm, 0.0
        )


class TrialSetupTests(unittest.TestCase):
    def test_name(self) -> None:
        self.assertEqual(TaskSpatialFrequency.name(), "task_spatial_frequency")

    def test_positions_match_v2_geometry(self) -> None:
        config = make_config(
            position_distance_x_cm=1.5, position_distance_y_cm=3.0, pixels_per_cm=46.0
        )
        positions = TaskSpatialFrequency._stimulus_positions(config)
        center = (1920 * 3 // 4, 1080 // 2)
        self.assertEqual(positions["center"], center)
        self.assertEqual(positions["left"], (center[0] - 69, center[1] - 138))
        self.assertEqual(positions["right"], (center[0] - 69, center[1] + 138))

    def test_all_three_trial_types_occur(self) -> None:
        config = make_config()
        seen = {TaskSpatialFrequency._pick_stimulus_side(config) for _ in range(300)}
        self.assertEqual(seen, {"left", "right", "none"})

    def test_weights_control_trial_types(self) -> None:
        config = make_config(stimulus_right_weight=0, stimulus_none_weight=0)
        seen = {TaskSpatialFrequency._pick_stimulus_side(config) for _ in range(50)}
        self.assertEqual(seen, {"left"})

    def test_stimulus_trial_has_gabor_at_its_side(self) -> None:
        for side in ("left", "right"):
            scene = make_scene(side)
            self.assertEqual(scene._gabor.rect.center, scene._positions[side])
            self.assertIn(
                scene._spatial_frequency_cpcm,
                scene._config.level_config.spatial_frequencies_cpcm(),
            )

    def test_no_stimulus_trial_has_no_gabor(self) -> None:
        scene = make_scene("none")
        self.assertIsNone(scene._gabor)
        self.assertIsNone(scene._spatial_frequency_cpcm)


class FixationPhaseTests(unittest.TestCase):
    def test_fixation_point_always_shown_first(self) -> None:
        for keep in (True, False):
            scene = make_scene(keep_fixation_point_after_fixation=keep)
            scene._on_trial_start()
            self.assertEqual(scene._phase, "fixation")
            self.assertIn(scene._fixation_point, scene._sprites)

    def test_dwell_on_fixation_starts_random_delay(self) -> None:
        scene = make_scene()
        with Clock() as clock:
            scene._on_trial_start()
            scene._on_press(-1, scene._positions["center"])
            clock.advance_ms(99)
            scene._check_all_trackers()
            self.assertEqual(scene._phase, "fixation")
            clock.advance_ms(2)
            scene._check_all_trackers()
        self.assertEqual(scene._phase, "delay")
        delay, callback = timer_calls(scene)[-1]
        self.assertTrue(1000 <= delay <= 2000)
        self.assertEqual(callback, scene._start_stimulus_phase)
        self.assertEqual(scene._result.fixation_delay_ms, delay)
        scene._rewarder.give_reward.assert_not_called()

    def test_gaze_outside_fixation_tolerance_does_not_count(self) -> None:
        scene = make_scene()
        x, y = scene._positions["center"]
        outside = round(scene._config.fixation_tolerance_distance_px) + 2
        with Clock() as clock:
            scene._on_trial_start()
            scene._on_press(-1, (x + outside, y))
            clock.advance_ms(1000)
            scene._check_all_trackers()
        self.assertEqual(scene._phase, "fixation")

    def test_fixation_point_kept_or_removed_after_fixation(self) -> None:
        for keep in (True, False):
            scene = make_scene(keep_fixation_point_after_fixation=keep)
            with Clock() as clock:
                start_stimulus_phase(scene, clock)
            self.assertEqual(scene._fixation_point in scene._sprites, keep)


class StimulusTrialTests(unittest.TestCase):
    def _look(self, scene: Any, clock: Clock, area: str, at_ms: float) -> None:
        """Move gaze to an area so the dwell completes at at_ms after onset."""
        clock.advance_ms(at_ms - 100)
        scene._on_motion(-1, scene._positions[area])
        clock.advance_ms(100)
        scene._check_all_trackers()

    def test_stimulus_appears_with_timeout_scheduled(self) -> None:
        scene = make_scene("right")
        with Clock() as clock, patch(f"{SCENE}.write_target_shown_event") as shown:
            start_stimulus_phase(scene, clock)
        self.assertIn(scene._gabor, scene._sprites)
        self.assertEqual(shown.call_args.args[-1], scene._positions["right"])
        self.assertEqual(timer_calls(scene)[-1], (2000, scene._on_no_response))

    def test_gaze_on_stimulus_in_window_is_rewarded(self) -> None:
        scene = make_scene("left")
        with Clock() as clock:
            start_stimulus_phase(scene, clock)
            self._look(scene, clock, "left", at_ms=500)
        self.assertEqual(scene._result.result, ResultEnum.CORRECT)
        self.assertEqual(scene._result.response, "in_window")
        self.assertAlmostEqual(scene._result.response_latency_ms, 500, places=3)
        scene._rewarder.give_reward.assert_called_once_with(1000)
        self.assertEqual(scene._phase, "blank")
        self.assertNotIn(scene._gabor, scene._sprites)
        self.assertEqual(timer_calls(scene)[-1], (1500, scene.quit))

    def test_anticipated_gaze_is_not_correct(self) -> None:
        scene = make_scene("left")  # window 300-800 ms
        with Clock() as clock:
            start_stimulus_phase(scene, clock)
            self._look(scene, clock, "left", at_ms=200)
        self.assertEqual(scene._result.result, ResultEnum.TIMEOUT)
        self.assertEqual(scene._result.response, "early")
        scene._rewarder.give_reward.assert_not_called()

    def test_late_gaze_is_not_correct(self) -> None:
        scene = make_scene("right")
        with Clock() as clock:
            start_stimulus_phase(scene, clock)
            self._look(scene, clock, "right", at_ms=900)
        self.assertEqual(scene._result.result, ResultEnum.TIMEOUT)
        self.assertEqual(scene._result.response, "late")
        scene._rewarder.give_reward.assert_not_called()

    def test_gaze_held_on_fixation_from_before_onset_is_not_a_response(self) -> None:
        scene = make_scene("left")
        with Clock() as clock:
            start_stimulus_phase(scene, clock)
            clock.advance_ms(1500)
            scene._check_all_trackers()
        self.assertEqual(scene._result.result, ResultEnum.PENDING)

    def test_gaze_on_empty_side_is_incorrect(self) -> None:
        scene = make_scene("left")
        with Clock() as clock:
            start_stimulus_phase(scene, clock)
            self._look(scene, clock, "right", at_ms=500)
        self.assertEqual(scene._result.result, ResultEnum.INCORRECT)
        self.assertEqual(scene._result.response, "wrong_side")
        scene._rewarder.give_reward.assert_not_called()

    def test_no_response_times_out(self) -> None:
        scene = make_scene("left")
        with Clock() as clock:
            start_stimulus_phase(scene, clock)
        scene._on_no_response()
        self.assertEqual(scene._result.result, ResultEnum.TIMEOUT)
        self.assertEqual(scene._result.response, "no_response")
        self.assertEqual(scene._sprites.sprites(), [])

    def test_response_cancels_the_stimulus_timeout(self) -> None:
        scene = make_scene("left")
        with Clock() as clock:
            start_stimulus_phase(scene, clock)
            end_task = scene._stimulus_end_task
            self._look(scene, clock, "left", at_ms=500)
        end_task.cancel.assert_called_once_with()


class NoStimulusTrialTests(unittest.TestCase):
    def test_holding_fixation_is_rewarded_at_window_end(self) -> None:
        scene = make_scene("none")
        with Clock() as clock:
            start_stimulus_phase(scene, clock)
            self.assertEqual(timer_calls(scene)[-1], (800, scene._on_fixation_held))
            clock.advance_ms(800)
            scene.update(0.0)
            scene._on_fixation_held()
        self.assertEqual(scene._result.result, ResultEnum.CORRECT)
        self.assertEqual(scene._result.response, "held_fixation")
        scene._rewarder.give_reward.assert_called_once_with(1000)

    def test_holding_fixation_counts_even_if_point_was_removed(self) -> None:
        scene = make_scene("none", keep_fixation_point_after_fixation=False)
        with Clock() as clock:
            start_stimulus_phase(scene, clock)
            scene._on_fixation_held()
        self.assertEqual(scene._result.result, ResultEnum.CORRECT)

    def test_leaving_fixation_beyond_grace_is_incorrect(self) -> None:
        scene = make_scene("none")
        with Clock() as clock:
            start_stimulus_phase(scene, clock)
            scene._on_motion(-1, scene._positions["left"])
            scene.update(0.0)
            clock.advance_ms(151)
            scene.update(0.0)
        self.assertEqual(scene._result.result, ResultEnum.INCORRECT)
        self.assertEqual(scene._result.response, "broke_fixation")
        scene._rewarder.give_reward.assert_not_called()

    def test_brief_exit_within_grace_is_tolerated(self) -> None:
        scene = make_scene("none")
        with Clock() as clock:
            start_stimulus_phase(scene, clock)
            scene._on_motion(-1, scene._positions["left"])
            scene.update(0.0)
            clock.advance_ms(100)
            scene._on_motion(-1, scene._positions["center"])
            scene.update(0.0)
            scene._on_fixation_held()
        self.assertEqual(scene._result.result, ResultEnum.CORRECT)

    def test_no_gaze_at_all_is_incorrect(self) -> None:
        scene = make_scene("none")
        with Clock() as clock:
            start_stimulus_phase(scene, clock)
            scene._on_release(-1, scene._positions["center"])
            scene.update(0.0)
            clock.advance_ms(200)
            scene.update(0.0)
        self.assertEqual(scene._result.result, ResultEnum.INCORRECT)


class EndOfTrialTests(unittest.TestCase):
    def test_gaze_loss_after_resolution_does_not_cut_trial_short(self) -> None:
        scene = make_scene()
        scene.quit = Mock()
        scene._result.result = ResultEnum.CORRECT
        scene.on_gaze_abandoned()
        scene.quit.assert_not_called()

    def test_gaze_loss_before_resolution_cancels(self) -> None:
        scene = make_scene()
        scene.quit = Mock()
        scene.on_gaze_abandoned()
        scene.quit.assert_called_once_with()

    def test_manual_reward_counted_once(self) -> None:
        scene = make_scene()
        scene._give_manual_reward()
        self.assertEqual(scene._result.trial_reward_count, 0)
        self.assertEqual(scene._result.trial_manual_reward_count, 1)


# Level-up needs a level to move to; the fixtures default to a single level.
TWO_LEVELS = {1: make_level(), 2: make_level()}


class SettlementTests(unittest.TestCase):

    def _settle(self, scene: Any, previous: list[ResultEnum]) -> bool:
        trials = TrialResults(
            root={i: make_result(trial_id=i, result=r) for i, r in enumerate(previous, 2)}
        )
        scene._load_trials = Mock(return_value=trials)
        scene._save_trials = Mock()
        return scene._settlement()

    def test_levels_up_at_threshold(self) -> None:
        scene = make_scene(levels=TWO_LEVELS)
        scene._result.result = ResultEnum.CORRECT
        self.assertTrue(self._settle(scene, [ResultEnum.CORRECT] * 3))
        scene._mxbiflow.session.level_up.assert_called_once()

    def test_incorrect_and_timeouts_count_against_the_rate(self) -> None:
        for failure in (ResultEnum.INCORRECT, ResultEnum.TIMEOUT):
            scene = make_scene(levels=TWO_LEVELS)
            scene._result.result = ResultEnum.CORRECT
            # 3 correct out of 5 = 60% < 80%.
            previous = [ResultEnum.CORRECT, ResultEnum.CORRECT, failure, failure]
            self.assertFalse(self._settle(scene, previous))

    def test_cancelled_trials_are_ignored(self) -> None:
        scene = make_scene(levels=TWO_LEVELS)
        scene._result.result = ResultEnum.CORRECT
        previous = [ResultEnum.CORRECT] * 3 + [ResultEnum.CANCELLED] * 5
        self.assertTrue(self._settle(scene, previous))

    def test_last_level_never_levels_up(self) -> None:
        scene = make_scene()
        scene._result.result = ResultEnum.CORRECT
        self.assertFalse(self._settle(scene, [ResultEnum.CORRECT] * 3))


if __name__ == "__main__":
    unittest.main()
