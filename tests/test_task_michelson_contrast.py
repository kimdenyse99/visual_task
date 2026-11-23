# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false

import json
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self
from unittest.mock import Mock, patch

import numpy as np
from mxbiflow.models.animal import AnimalBaseInfo
from pydantic import ValidationError
from pygame.sprite import Group

from assets import GIF_STIMULUS_DIR, MARMOSET_IMAGE_PATHS
from gabor_sprite import GaborSprite, gabor_pixels
from stages.task_michelson_contrast import TaskMichelsonContrast
from stages.task_michelson_contrast.models import (
    Config,
    ContrastOption,
    LevelConfig,
    Result,
    ResultEnum,
    Screen,
    StoredConfig,
    StoredConfigs,
    TrialResults,
)
from stimulus_events import StimulusEvents

CONFIG_PATH = Path("stages/task_michelson_contrast/config.json")
SCENE = "stages.task_michelson_contrast.scene"

# Writing the target-shown CSV needs a real session; tests that care about
# it patch it again to inspect the call.
_target_shown_patch = patch(f"{SCENE}.write_target_shown_event")


def setUpModule() -> None:
    _target_shown_patch.start()


def tearDownModule() -> None:
    _target_shown_patch.stop()


def make_level(*contrasts: tuple[float, float]) -> LevelConfig:
    pairs = contrasts or ((50.0, 1.0),)
    return LevelConfig(
        contrasts=[ContrastOption(contrast_percent=c, weight=w) for c, w in pairs]
    )


def make_stored_config(**overrides: Any) -> StoredConfig:
    values: dict[str, Any] = {
        "border_width": 40,
        "level_up_threshold": 0.8,
        "level_up_trial_count_threshold": 3,
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


def make_result(
    level: int = 1, trial_id: int = 1, contrast: float = 50.0, **fields: Any
) -> Result:
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
        stimulus_side="left",
        stimulus_center=(0, 0),
        contrast_percent=contrast,
        orientation_deg=0.0,
        spatial_frequency_cpcm=4.0,
        **fields,
    )


def make_scene(
    side: str = "left", contrast: float = 50.0, **config_overrides: Any
) -> Any:
    """A scene with real stimuli and dwell logic but mocked hardware."""
    scene: Any = TaskMichelsonContrast.__new__(TaskMichelsonContrast)
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
    scene._phase_task = None
    scene._fixation_shown_at_ms = 0.0
    scene._stimulus_onset_ms = 0.0
    scene._trackers = {}
    scene._trajectories = {}
    scene._trace_points = []
    scene._control_buttons = Group()
    scene._gaze_driver = Mock()
    with (
        patch(f"{SCENE}.choice", side_effect=lambda seq: side if "left" in seq else seq[0]),
        patch.object(
            TaskMichelsonContrast, "_pick_contrast_percent", return_value=contrast
        ),
    ):
        scene._init_visual_stimulus(scene._config)
    scene._result = make_result(contrast=contrast)
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


def last_timer(scene: Any) -> tuple[int, Any]:
    """The most recent timer.after(delay, callback) call."""
    call = scene._timer.after.call_args
    return (call.args[0], call.args[1])


def fixate(scene: Any, clock: Clock) -> None:
    scene._on_trial_start()
    scene._on_press(-1, scene._positions["center"])
    clock.advance_ms(100)
    scene._check_all_trackers()
    assert scene._phase == "delay"


def start_stimulus_phase(scene: Any, clock: Clock) -> None:
    """Fixate, then run the delay timer's callback immediately."""
    fixate(scene, clock)
    scene._start_stimulus_phase()
    assert scene._phase == "stimulus"


def look(scene: Any, clock: Clock, area: str, at_ms: float) -> None:
    """Move gaze to an area so the dwell completes at_ms after onset."""
    clock.advance_ms(at_ms - 100)
    scene._on_motion(-1, scene._positions[area])
    clock.advance_ms(100)
    scene._check_all_trackers()


class GaborTests(unittest.TestCase):
    def test_grating_amplitude_is_the_contrast(self) -> None:
        bg = 127
        pixels = gabor_pixels(201, 0.05, 0.5, (bg, bg, bg)).astype(float)[:, :, 0]
        peak = pixels[100, 100]
        trough = pixels[110, 100]  # half a cycle from the center
        self.assertAlmostEqual(peak, bg * 1.5, delta=1)
        self.assertAlmostEqual((peak - trough) / (peak + trough), 0.5, delta=0.03)

    def test_edges_fade_to_background(self) -> None:
        pixels = gabor_pixels(69, 4 / 46, 0.5, (127, 127, 127)).astype(int)
        self.assertTrue(np.all(np.abs(pixels[0, :, :] - 127) <= 1))

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


class AssetTests(unittest.TestCase):
    def test_gif_folder_was_renamed(self) -> None:
        self.assertEqual(GIF_STIMULUS_DIR.name, "gifs_stimulus")
        self.assertEqual(len(MARMOSET_IMAGE_PATHS), 10)
        self.assertTrue(all(p.parent == GIF_STIMULUS_DIR for p in MARMOSET_IMAGE_PATHS))


class ConfigFileTests(unittest.TestCase):
    def _data(self) -> dict[str, Any]:
        return json.loads(CONFIG_PATH.read_text())["default"]

    def test_default_values(self) -> None:
        stored = StoredConfigs.model_validate({"default": self._data()}).root["default"]
        self.assertEqual(stored.fixation_timeout_ms, 5000)
        self.assertEqual(
            (stored.fixation_delay_min_ms, stored.fixation_delay_max_ms), (100, 200)
        )
        self.assertEqual(
            (stored.response_window_min_ms, stored.response_window_max_ms), (0, 500)
        )
        self.assertEqual(stored.inter_trial_duration_ms, 500)
        self.assertEqual(
            stored.reward_gifs, ["m001.gif", "m006.gif", "m007.gif", "m009.gif"]
        )
        self.assertEqual(stored.target_size_cm, 1.5)
        self.assertEqual(
            (stored.gabor_orientation_min_deg, stored.gabor_orientation_max_deg),
            (0, 180),
        )
        self.assertEqual(list(stored.levels), [1, 2, 3, 4])

    def test_general_settings_are_not_per_level(self) -> None:
        data = self._data()
        for key in (
            "fixation_tolerance_cm",
            "stimulus_tolerance_cm",
            "response_window_min_ms",
            "response_window_max_ms",
            "spatial_frequency_cpcm",
        ):
            self.assertIn(key, data)
        for level in data["levels"].values():
            self.assertEqual(set(level), {"contrasts"})
            for option in level["contrasts"]:
                self.assertEqual(set(option), {"contrast_percent", "weight"})

    def test_no_dwell_grace_settings(self) -> None:
        data = self._data()
        self.assertNotIn("dwell_grace_ms", data)
        self.assertNotIn("dwell_grace_enabled", data)

    def test_invalid_settings_are_rejected(self) -> None:
        bad_settings: list[dict[str, Any]] = [
            {"fixation_delay_min_ms": 300, "fixation_delay_max_ms": 200},
            {"response_window_min_ms": 600, "response_window_max_ms": 500},
            {"gabor_orientation_min_deg": 90, "gabor_orientation_max_deg": 0},
            {"stimulus_duration_ms": 400},  # shorter than the 500 ms window
            {"spatial_frequency_cpcm": 30},  # > 23 cycles/cm at 46 px/cm
            {"reward_gifs": ["missing.gif"]},
            {"reward_gifs": []},
            {"fixation_tolerance_cm": -1},
        ]
        for bad in bad_settings:
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                make_stored_config(**bad)

    def test_invalid_contrasts_are_rejected(self) -> None:
        for bad in ((150.0, 1.0), (-5.0, 1.0), (50.0, 0.0)):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                make_level(bad)
        with self.assertRaises(ValidationError):
            LevelConfig(contrasts=[])

    def test_contrast_weights_set_proportions(self) -> None:
        config = make_config(levels={1: make_level((0, 1), (50, 3))})
        picks = [TaskMichelsonContrast._pick_contrast_percent(config) for _ in range(4000)]
        share_50 = picks.count(50) / len(picks)
        self.assertEqual(set(picks), {0, 50})
        self.assertAlmostEqual(share_50, 0.75, delta=0.04)

    def test_tolerances_are_measured_from_the_edges(self) -> None:
        config = make_config(
            pixels_per_cm=40.0,
            fixation_point_radius_cm=0.5,
            target_size_cm=1.5,
            fixation_tolerance_cm=1.0,
            stimulus_tolerance_cm=0.5,
        )
        self.assertAlmostEqual(config.fixation_tolerance_distance_px, 60)
        self.assertAlmostEqual(config.stimulus_tolerance_distance_px, 50)


class TrialSetupTests(unittest.TestCase):
    def test_name(self) -> None:
        self.assertEqual(TaskMichelsonContrast.name(), "task_michelson_contrast")

    def test_gabor_at_its_side_with_contrast_and_random_orientation(self) -> None:
        orientations: set[float] = set()
        for side in ("left", "right"):
            for _ in range(5):
                scene = make_scene(side, contrast=25)
                self.assertEqual(scene._gabor.rect.center, scene._positions[side])
                self.assertTrue(0 <= scene._orientation_deg <= 180)
                orientations.add(scene._orientation_deg)
        self.assertGreater(len(orientations), 1)

    def test_both_sides_occur(self) -> None:
        sides: set[str] = set()
        for _ in range(50):
            scene: Any = TaskMichelsonContrast.__new__(TaskMichelsonContrast)
            scene._init_visual_stimulus(make_config())
            sides.add(scene._stimulus_side)
        self.assertEqual(sides, {"left", "right"})

    def test_reward_gif_is_one_of_the_configured_ones(self) -> None:
        names: set[str] = set()
        for _ in range(40):
            names.add(make_scene()._reward_asset_name)
        self.assertLessEqual(names, {"m001.gif", "m006.gif", "m007.gif", "m009.gif"})

    def test_reward_gif_replaces_gabor_or_sits_at_fixation_for_zero(self) -> None:
        scene = make_scene("right", contrast=50)
        self.assertEqual(scene._reward_gif.rect.center, scene._positions["right"])
        scene = make_scene("right", contrast=0)
        self.assertEqual(scene._reward_gif.rect.center, scene._positions["center"])


class FixationTests(unittest.TestCase):
    def test_fixation_timeout_is_a_timeout(self) -> None:
        scene = make_scene()
        scene._on_trial_start()
        self.assertEqual(last_timer(scene), (5000, scene._on_fixation_timeout))
        scene._on_fixation_timeout()
        self.assertEqual(scene._result.result, ResultEnum.TIMEOUT)
        self.assertEqual(scene._result.response, "no_fixation")
        self.assertEqual(last_timer(scene), (500, scene.quit))

    def test_fixation_cancels_the_fixation_timeout(self) -> None:
        scene = make_scene()
        tasks: list[Mock] = []
        scene._timer.after.side_effect = lambda *_: tasks.append(Mock()) or tasks[-1]
        with Clock() as clock:
            fixate(scene, clock)
        timeout_task = tasks[0]
        timeout_task.cancel.assert_called_once_with()
        scene._timer.after.side_effect = None
        delay, callback = last_timer(scene)
        self.assertTrue(100 <= delay <= 200)
        self.assertEqual(callback, scene._start_stimulus_phase)

    def test_leaving_the_area_resets_the_dwell_without_grace(self) -> None:
        scene = make_scene()
        x, y = scene._positions["center"]
        outside = round(scene._config.fixation_tolerance_distance_px) + 2
        with Clock() as clock:
            scene._on_trial_start()
            scene._on_press(-1, (x, y))
            clock.advance_ms(90)
            scene._on_motion(-1, (x + outside, y))
            clock.advance_ms(5)
            scene._on_motion(-1, (x, y))
            clock.advance_ms(20)
            scene._check_all_trackers()
            self.assertEqual(scene._phase, "fixation")  # restarted at 95 ms
            clock.advance_ms(80)
            scene._check_all_trackers()
        self.assertEqual(scene._phase, "delay")

    def test_leaving_fixation_during_delay_is_a_timeout(self) -> None:
        scene = make_scene()
        with Clock() as clock:
            fixate(scene, clock)
            scene._on_motion(-1, scene._positions["left"])
            scene.update(0.0)
        self.assertEqual(scene._result.result, ResultEnum.TIMEOUT)
        self.assertEqual(scene._result.response, "broke_fixation")
        self.assertEqual(scene._phase, "blank")
        self.assertEqual(last_timer(scene), (500, scene.quit))

    def test_holding_fixation_through_delay_shows_the_gabor(self) -> None:
        scene = make_scene()
        with Clock() as clock:
            fixate(scene, clock)
            scene.update(0.0)
            scene._start_stimulus_phase()
        self.assertEqual(scene._phase, "stimulus")
        self.assertIn(scene._gabor, scene._sprites)

    def test_fixation_point_kept_or_removed(self) -> None:
        for keep in (True, False):
            scene = make_scene(keep_fixation_point_after_fixation=keep)
            with Clock() as clock:
                start_stimulus_phase(scene, clock)
            self.assertEqual(scene._fixation_point in scene._sprites, keep)


class VisibleContrastTests(unittest.TestCase):
    def test_correct_gaze_flashes_reward_gif_then_inter_trial(self) -> None:
        scene = make_scene("left")
        with Clock() as clock:
            start_stimulus_phase(scene, clock)
            look(scene, clock, "left", at_ms=300)
        self.assertEqual(scene._result.result, ResultEnum.CORRECT)
        self.assertEqual(scene._result.response, "in_window")
        scene._rewarder.give_reward.assert_called_once_with(1000)
        self.assertEqual(scene._phase, "reward")
        self.assertEqual(scene._sprites.sprites(), [scene._reward_gif])
        self.assertEqual(scene._result.reward_asset, scene._reward_asset_name)
        self.assertEqual(last_timer(scene), (1000, scene._on_reward_finished))

        scene._on_reward_finished()
        self.assertEqual(scene._sprites.sprites(), [])
        self.assertEqual(scene._phase, "blank")
        self.assertEqual(last_timer(scene), (500, scene.quit))

    def test_late_gaze_is_a_timeout(self) -> None:
        scene = make_scene("right")
        with Clock() as clock:
            start_stimulus_phase(scene, clock)
            look(scene, clock, "right", at_ms=600)
        self.assertEqual(scene._result.result, ResultEnum.TIMEOUT)
        self.assertEqual(scene._result.response, "late")
        scene._rewarder.give_reward.assert_not_called()

    def test_early_gaze_is_a_timeout_when_window_min_is_set(self) -> None:
        scene = make_scene("left", response_window_min_ms=200)
        with Clock() as clock:
            start_stimulus_phase(scene, clock)
            look(scene, clock, "left", at_ms=150)
        self.assertEqual(scene._result.response, "early")

    def test_wrong_side_is_incorrect(self) -> None:
        scene = make_scene("left")
        with Clock() as clock:
            start_stimulus_phase(scene, clock)
            look(scene, clock, "right", at_ms=300)
        self.assertEqual(scene._result.result, ResultEnum.INCORRECT)
        self.assertEqual(scene._result.response, "wrong_side")
        self.assertEqual(last_timer(scene), (500, scene.quit))

    def test_missed_stimulus_is_a_timeout(self) -> None:
        scene = make_scene("left")
        with Clock() as clock:
            start_stimulus_phase(scene, clock)
            self.assertEqual(last_timer(scene), (1000, scene._on_stimulus_ended))
            clock.advance_ms(1000)
            scene._check_all_trackers()  # still on fixation: no response
        scene._on_stimulus_ended()
        self.assertEqual(scene._result.result, ResultEnum.TIMEOUT)
        self.assertEqual(scene._result.response, "no_response")
        self.assertEqual(scene._sprites.sprites(), [])
        self.assertEqual(last_timer(scene), (500, scene.quit))
        scene._rewarder.give_reward.assert_not_called()


class ZeroContrastTests(unittest.TestCase):
    def test_looking_at_neither_side_is_rewarded(self) -> None:
        scene = make_scene("left", contrast=0)
        with Clock() as clock:
            start_stimulus_phase(scene, clock)
        scene._on_stimulus_ended()
        self.assertEqual(scene._result.result, ResultEnum.CORRECT)
        self.assertEqual(scene._result.response, "withheld")
        scene._rewarder.give_reward.assert_called_once_with(1000)
        self.assertEqual(scene._phase, "reward")

    def test_looking_at_either_side_is_a_false_alarm(self) -> None:
        for area in ("left", "right"):
            scene = make_scene("left", contrast=0)
            with Clock() as clock:
                start_stimulus_phase(scene, clock)
                look(scene, clock, area, at_ms=300)
            self.assertEqual(scene._result.result, ResultEnum.INCORRECT)
            self.assertEqual(scene._result.response, "false_alarm")
            scene._rewarder.give_reward.assert_not_called()


class EndOfTrialTests(unittest.TestCase):
    def test_gaze_loss_after_resolution_does_not_cut_trial_short(self) -> None:
        scene = make_scene()
        scene.quit = Mock()
        scene._result.result = ResultEnum.CORRECT
        scene.on_gaze_abandoned()
        scene.quit.assert_not_called()

    def test_manual_reward_counted_once(self) -> None:
        scene = make_scene()
        scene._give_manual_reward()
        self.assertEqual(scene._result.trial_reward_count, 0)
        self.assertEqual(scene._result.trial_manual_reward_count, 1)


# Level-up needs a level to move to; the fixtures default to a single level.
TWO_LEVELS = {1: make_level(), 2: make_level()}


class SettlementTests(unittest.TestCase):
    def _settle(
        self, scene: Any, previous: Sequence[tuple[ResultEnum, float]]
    ) -> bool:
        trials = TrialResults(
            root={
                i: make_result(trial_id=i, result=r, contrast=c)
                for i, (r, c) in enumerate(previous, 2)
            }
        )
        scene._load_trials = Mock(return_value=trials)
        scene._save_trials = Mock()
        return scene._settlement()

    def test_levels_up_at_threshold(self) -> None:
        scene = make_scene(levels=TWO_LEVELS)
        scene._result.result = ResultEnum.CORRECT
        self.assertTrue(self._settle(scene, [(ResultEnum.CORRECT, 50)] * 3))
        scene._mxbiflow.session.level_up.assert_called_once()

    def test_zero_contrast_trials_are_left_out_of_the_rate(self) -> None:
        scene = make_scene(levels=TWO_LEVELS)
        scene._result.result = ResultEnum.CORRECT
        # Only 1 visible-contrast trial besides this one: not enough to level
        # up, however many 0% trials were rewarded.
        previous = [(ResultEnum.CORRECT, 50)] + [(ResultEnum.CORRECT, 0)] * 10
        self.assertFalse(self._settle(scene, previous))
        # And failed 0% trials don't lower the rate.
        scene = make_scene(levels=TWO_LEVELS)
        scene._result.result = ResultEnum.CORRECT
        previous = [(ResultEnum.CORRECT, 50)] * 3 + [(ResultEnum.INCORRECT, 0)] * 10
        self.assertTrue(self._settle(scene, previous))

    def test_incorrect_and_timeouts_count_against_the_rate(self) -> None:
        scene = make_scene(levels=TWO_LEVELS)
        scene._result.result = ResultEnum.CORRECT
        previous = [
            (ResultEnum.CORRECT, 50),
            (ResultEnum.CORRECT, 50),
            (ResultEnum.INCORRECT, 50),
            (ResultEnum.TIMEOUT, 50),
        ]
        self.assertFalse(self._settle(scene, previous))

    def test_summary_counts_zero_contrast_separately(self) -> None:
        scene = make_scene()
        trials = TrialResults(
            root={
                1: make_result(trial_id=1, result=ResultEnum.CORRECT, contrast=50),
                2: make_result(trial_id=2, result=ResultEnum.CORRECT, contrast=0),
                3: make_result(trial_id=3, result=ResultEnum.INCORRECT, contrast=0),
            }
        )
        scene._load_trials = Mock(return_value=trials)
        summary = scene._result_summary()
        self.assertIn("Correct 1", summary)
        self.assertIn("Error 0", summary)
        self.assertIn("0% 1/2", summary)


if __name__ == "__main__":
    unittest.main()
