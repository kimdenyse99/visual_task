# pyright: reportPrivateUsage=false

import json
import unittest
from pathlib import Path
from typing import Any, Self
from unittest.mock import Mock, patch

from mxbiflow.models.animal import AnimalBaseInfo
from pydantic import ValidationError
from pygame import FINGERDOWN, Event, Surface
from pygame.sprite import Group

from assets import MARMOSET_IMAGE_PATHS
from stages.visual_3pt_calibration_v2 import Visual3ptCalibrationV2
from stages.visual_3pt_calibration_v2.models import (
    Config,
    LevelConfig,
    Result,
    ResultEnum,
    Screen,
    StoredConfig,
    StoredConfigs,
    TrialResults,
)
from stimulus_events import StimulusEvents

CONFIG_PATH = Path("stages/visual_3pt_calibration_v2/config.json")

# Writing the target-shown CSV needs a real session; individual tests that
# care about it patch it again to inspect the call.
_target_shown_patch = patch(
    "stages.visual_3pt_calibration_v2.scene.write_target_shown_event"
)


def setUpModule() -> None:
    _target_shown_patch.start()


def tearDownModule() -> None:
    _target_shown_patch.stop()


def make_stored_config(**overrides: Any) -> StoredConfig:
    """fixation_acceptance_radius_cm is a per-level setting; passing it here
    sets it on the single default level (unless levels is also given)."""
    radius_cm = overrides.pop("fixation_acceptance_radius_cm", 1.0)
    defaults: dict[str, Any] = {
        "inter_trial_duration_ms": 1500,
        "reward_duration_ms": 600,
        "post_reward_stimulus_duration_ms": 1500,
        "fixation_timeout_ms": 5000,
        "border_width": 40,
        "level_up_threshold": 0.8,
        "level_up_trial_count_threshold": 20,
        "fixation_point_radius_cm": 0.5,
        "position_distance_cm": 5.0,
        "pixels_per_cm": 46.0,
        "min_touch_duration_ms": 200,
        "dwell_grace_enabled": True,
        "dwell_grace_ms": 150,
        "levels": {
            1: LevelConfig(
                target_size_cm=1.5, fixation_acceptance_radius_cm=radius_cm
            )
        },
    }
    defaults.update(overrides)
    return StoredConfig(**defaults)


def make_config(level: int = 1, **overrides: Any) -> Config:
    return Config(
        animal="test-animal",
        level=level,
        is_twitter_trial=True,
        screen_size=Screen(width=1920, height=1080),
        stored=make_stored_config(**overrides),
    )


def make_result(level: int = 1, trial_id: int = 1, **fields: Any) -> Result:
    return Result(
        animal=AnimalBaseInfo(
            animal="test-animal",
            trial_id=trial_id,
            level=level,
            level_trial_id=trial_id,
            animal_session_id=1,
            animal_session_trial_id=trial_id,
        ),
        is_twitter_trial=True,
        trial_start_time=0.0,
        trial_end_time=0.0,
        **fields,
    )


def make_scene(**config_overrides: Any) -> Any:
    """A scene with real stimuli and dwell logic but mocked hardware."""
    scene: Any = Visual3ptCalibrationV2.__new__(Visual3ptCalibrationV2)
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
    scene._fixation_timeout_task = None
    scene._fixation_shown_at_ms = 0.0
    scene._trackers = {}
    scene._trajectories = {}
    scene._trace_points = []
    scene._init_visual_stimulus(scene._config)
    scene._result = make_result()
    return scene


class Clock:
    """Drives time.monotonic in both the scene and dwell.DwellTracker."""

    def __init__(self) -> None:
        self.now_s = 1000.0
        self._patches = [
            patch("dwell.monotonic", self),
            patch("stages.visual_3pt_calibration_v2.scene.monotonic", self),
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


def timer_delays(scene: Any) -> list[tuple[int, Any]]:
    return [(c.args[0], c.args[1]) for c in scene._timer.after.call_args_list]


class StageNameTests(unittest.TestCase):
    def test_name_matches_package(self) -> None:
        self.assertEqual(Visual3ptCalibrationV2.name(), "visual_3pt_calibration_v2")

    def test_trial_context_key_is_separate_from_v1(self) -> None:
        self.assertEqual(
            Visual3ptCalibrationV2._trials_context_key,
            "visual_3pt_calibration_v2.trials",
        )


class ConfigFileTests(unittest.TestCase):
    def test_default_config_has_requested_values(self) -> None:
        data = json.loads(CONFIG_PATH.read_text())
        stored = StoredConfigs.model_validate(data).root["default"]
        self.assertEqual(stored.dwell_grace_ms, 150)
        self.assertTrue(stored.dwell_grace_enabled)
        self.assertEqual(stored.reward_duration_ms, 600)
        self.assertEqual(stored.post_reward_stimulus_duration_ms, 1500)
        self.assertEqual(stored.inter_trial_duration_ms, 1500)
        self.assertEqual(stored.fixation_timeout_ms, 5000)

    def test_default_config_has_four_levels_with_per_level_radius(self) -> None:
        data = json.loads(CONFIG_PATH.read_text())["default"]
        self.assertNotIn("fixation_acceptance_radius_cm", data)
        self.assertEqual(list(data["levels"]), ["1", "2", "3", "4"])
        for level in data["levels"].values():
            self.assertIn("target_size_cm", level)
            self.assertIn("fixation_acceptance_radius_cm", level)
        stored = StoredConfigs.model_validate({"default": data}).root["default"]
        self.assertEqual(stored.level_up_threshold, 0.8)
        self.assertEqual(stored.level_up_trial_count_threshold, 200)

    def test_config_has_no_fixation_delay_range(self) -> None:
        data = json.loads(CONFIG_PATH.read_text())["default"]
        self.assertNotIn("fixation_duration_min_ms", data)
        self.assertNotIn("fixation_duration_max_ms", data)

    def test_negative_acceptance_radius_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            LevelConfig(target_size_cm=1.5, fixation_acceptance_radius_cm=-0.1)

    def test_acceptance_radius_and_target_size_follow_the_level(self) -> None:
        levels = {
            1: LevelConfig(target_size_cm=2.0, fixation_acceptance_radius_cm=1.5),
            2: LevelConfig(target_size_cm=1.0, fixation_acceptance_radius_cm=0.5),
        }
        for level, size_cm, radius_cm in ((1, 2.0, 1.5), (2, 1.0, 0.5)):
            config = make_config(level=level, levels=levels, pixels_per_cm=40.0)
            self.assertEqual(config.target_size_cm, size_cm)
            self.assertEqual(config.fixation_acceptance_radius_cm, radius_cm)
            self.assertAlmostEqual(
                config.fixation_acceptance_distance_px, (0.5 + radius_cm) * 40.0
            )

    def test_scene_uses_the_current_level_radius(self) -> None:
        levels = {
            1: LevelConfig(target_size_cm=1.5, fixation_acceptance_radius_cm=2.0),
            2: LevelConfig(target_size_cm=1.5, fixation_acceptance_radius_cm=0.0),
        }
        near_px = round(1.5 * 46.0)  # 1 cm beyond the 0.5 cm point's edge
        for level, expected in ((1, True), (2, False)):
            scene = make_scene()
            scene._config = make_config(level=level, levels=levels)
            x, y = scene._fixation_center
            self.assertEqual(scene._is_on_fixation_point((x + near_px, y)), expected)

    def test_acceptance_radius_is_measured_from_the_perimeter(self) -> None:
        config = make_config(
            fixation_point_radius_cm=0.5,
            fixation_acceptance_radius_cm=2.0,
            pixels_per_cm=40.0,
        )
        self.assertAlmostEqual(config.fixation_acceptance_radius_px, 80.0)
        # 0.5 cm point radius + 2.0 cm beyond its edge = 2.5 cm from center.
        self.assertAlmostEqual(config.fixation_acceptance_distance_px, 100.0)

    def test_zero_acceptance_radius_accepts_only_the_point(self) -> None:
        config = make_config(fixation_acceptance_radius_cm=0)
        self.assertAlmostEqual(
            config.fixation_acceptance_distance_px, config.fixation_point_radius_px
        )

    def test_dwell_grace_is_zero_when_disabled(self) -> None:
        config = make_config(dwell_grace_enabled=False, dwell_grace_ms=150)
        self.assertEqual(config.dwell_grace_ms, 0.0)


class PositionTests(unittest.TestCase):
    def test_three_positions_match_v1_geometry(self) -> None:
        config = make_config()
        positions = Visual3ptCalibrationV2._fixation_positions(config)
        anchor = (1920 * 3 // 4, 1080 // 2)
        offset = round(config.position_distance_px // 2)
        self.assertEqual(
            positions,
            {
                "left": (anchor[0] - offset, anchor[1] - offset),
                "right": (anchor[0] - offset, anchor[1] + offset),
                "center": anchor,
            },
        )

    def test_every_position_is_used(self) -> None:
        seen: set[str] = set()
        for _ in range(200):
            seen.add(make_scene()._position_name)
            if len(seen) == 3:
                break
        self.assertEqual(seen, {"left", "right", "center"})

    def test_gif_plays_in_the_fixation_point_spot(self) -> None:
        scene = make_scene()
        self.assertEqual(scene._fixation_point.rect.center, scene._fixation_center)
        self.assertEqual(scene._target.rect.center, scene._fixation_center)

    def test_gif_is_one_of_the_marmoset_gifs(self) -> None:
        names = {p.name for p in MARMOSET_IMAGE_PATHS}
        self.assertIn("m001.gif", names)
        self.assertIn("m010.gif", names)
        self.assertIn(make_scene()._target_asset_name, names)


class FixationTests(unittest.TestCase):
    def _on_point(self, scene: Any, dx: int = 0) -> tuple[int, int]:
        x, y = scene._fixation_center
        return (x + dx, y)

    def test_trial_start_shows_point_and_schedules_timeout(self) -> None:
        scene = make_scene()
        scene._sprites = Mock()
        scene._on_trial_start()
        self.assertEqual(scene._phase, "fixation")
        scene._sprites.add.assert_called_once_with(scene._fixation_point)
        self.assertEqual(
            timer_delays(scene), [(5000, scene._on_fixation_timeout)]
        )

    def test_dwell_on_point_gives_reward_and_shows_gif_immediately(self) -> None:
        scene = make_scene()
        with Clock() as clock, patch(
            "stages.visual_3pt_calibration_v2.scene.write_target_shown_event"
        ) as target_shown:
            scene._on_trial_start()
            timeout_task = scene._timer.after.return_value
            scene._on_press(-1, self._on_point(scene))
            clock.advance_ms(199)
            scene._check_all_trackers()
            scene._rewarder.give_reward.assert_not_called()
            clock.advance_ms(2)
            scene._check_all_trackers()

        scene._rewarder.give_reward.assert_called_once_with(600)
        timeout_task.cancel.assert_called_once_with()
        self.assertEqual(scene._result.result, ResultEnum.CORRECT)
        self.assertEqual(scene._phase, "reward")
        self.assertNotIn(scene._fixation_point, scene._sprites)
        self.assertIn(scene._target, scene._sprites)
        self.assertEqual(scene._result.stimulus_asset, scene._target_asset_name)
        self.assertAlmostEqual(scene._result.fixation_touch_latency_ms, 201.0, places=3)
        target_shown.assert_called_once()
        self.assertEqual(target_shown.call_args.args[-1], scene._fixation_center)
        self.assertEqual(
            timer_delays(scene)[-1], (1500, scene._end_reward_phase)
        )

    def test_gaze_in_vicinity_counts(self) -> None:
        scene = make_scene()
        inside_px = int(scene._config.fixation_acceptance_distance_px) - 1
        with Clock() as clock:
            scene._on_trial_start()
            scene._on_press(-1, self._on_point(scene, dx=inside_px))
            clock.advance_ms(200)
            scene._check_all_trackers()
        self.assertEqual(scene._result.result, ResultEnum.CORRECT)

    def test_gaze_outside_radius_does_not_count(self) -> None:
        scene = make_scene()
        outside_px = int(scene._config.fixation_acceptance_distance_px) + 2
        with Clock() as clock:
            scene._on_trial_start()
            scene._on_press(-1, self._on_point(scene, dx=outside_px))
            clock.advance_ms(4000)
            scene._check_all_trackers()
        self.assertEqual(scene._result.result, ResultEnum.PENDING)
        scene._rewarder.give_reward.assert_not_called()

    def test_brief_exit_within_grace_keeps_dwell(self) -> None:
        scene = make_scene()
        outside_px = int(scene._config.fixation_acceptance_distance_px) + 2
        with Clock() as clock:
            scene._on_trial_start()
            scene._on_press(-1, self._on_point(scene))
            clock.advance_ms(150)
            scene._on_motion(-1, self._on_point(scene, dx=outside_px))
            clock.advance_ms(100)
            scene._on_motion(-1, self._on_point(scene))
            clock.advance_ms(50)
            scene._check_all_trackers()
        self.assertEqual(scene._result.result, ResultEnum.CORRECT)

    def test_exit_longer_than_grace_resets_dwell(self) -> None:
        scene = make_scene()
        outside_px = int(scene._config.fixation_acceptance_distance_px) + 2
        with Clock() as clock:
            scene._on_trial_start()
            scene._on_press(-1, self._on_point(scene))
            clock.advance_ms(150)
            scene._on_motion(-1, self._on_point(scene, dx=outside_px))
            clock.advance_ms(160)
            scene._check_all_trackers()
            scene._on_motion(-1, self._on_point(scene))
            clock.advance_ms(50)
            scene._check_all_trackers()
        self.assertEqual(scene._result.result, ResultEnum.PENDING)

    def test_no_dwell_outside_fixation_phase(self) -> None:
        scene = make_scene()
        with Clock() as clock:
            scene._on_press(-1, self._on_point(scene))
            clock.advance_ms(1000)
            scene._check_all_trackers()
        self.assertEqual(scene._result.result, ResultEnum.PENDING)


class AcceptanceAreaOutlineTests(unittest.TestCase):
    def _draw(self, scene: Any) -> list[Any]:
        scene._result_widget = Mock()
        scene._control_buttons = Group()
        screen = Surface((1920, 1080))
        with patch("stages.visual_3pt_calibration_v2.scene.draw_circle") as circle:
            scene.draw(screen)
        return circle.call_args_list

    def test_hidden_by_default(self) -> None:
        self.assertFalse(make_config().show_fixation_acceptance_area)
        data = json.loads(CONFIG_PATH.read_text())["default"]
        self.assertFalse(data["show_fixation_acceptance_area"])
        scene = make_scene()
        scene._on_trial_start()
        self.assertEqual(self._draw(scene), [])

    def test_outline_matches_acceptance_distance(self) -> None:
        scene = make_scene(show_fixation_acceptance_area=True)
        scene._on_trial_start()
        calls = self._draw(scene)
        self.assertEqual(len(calls), 1)
        _, color, center, radius, width = calls[0].args
        self.assertEqual(center, scene._fixation_center)
        self.assertEqual(
            radius, round(scene._config.fixation_acceptance_distance_px)
        )
        self.assertGreater(width, 0)  # outline, not a filled circle
        self.assertEqual(color, scene._acceptance_area_color)

    def test_not_drawn_once_fixation_is_over(self) -> None:
        scene = make_scene(show_fixation_acceptance_area=True)
        scene._on_trial_start()
        scene._on_fixation_timeout()
        self.assertEqual(self._draw(scene), [])


class TimeoutTests(unittest.TestCase):
    def test_timeout_hides_point_and_goes_to_inter_trial(self) -> None:
        scene = make_scene()
        scene._on_trial_start()
        scene._on_fixation_timeout()
        self.assertEqual(scene._result.result, ResultEnum.TIMEOUT)
        self.assertEqual(scene._phase, "blank")
        self.assertNotIn(scene._fixation_point, scene._sprites)
        self.assertEqual(timer_delays(scene)[-1], (1500, scene.quit))
        scene._rewarder.give_reward.assert_not_called()

    def test_timeout_after_fixation_is_ignored(self) -> None:
        scene = make_scene()
        with patch("stages.visual_3pt_calibration_v2.scene.write_target_shown_event"):
            scene._on_trial_start()
            scene._on_fixation_touched()
        scene._on_fixation_timeout()
        self.assertEqual(scene._result.result, ResultEnum.CORRECT)
        self.assertEqual(scene._phase, "reward")


class RewardPhaseTests(unittest.TestCase):
    def test_end_of_gif_starts_inter_trial(self) -> None:
        scene = make_scene()
        with patch("stages.visual_3pt_calibration_v2.scene.write_target_shown_event"):
            scene._on_trial_start()
            scene._on_fixation_touched()
        scene._end_reward_phase()
        self.assertNotIn(scene._target, scene._sprites)
        self.assertEqual(scene._phase, "blank")
        self.assertEqual(timer_delays(scene)[-1], (1500, scene.quit))

    def test_gaze_loss_after_correct_does_not_cut_trial_short(self) -> None:
        scene = make_scene()
        scene.quit = Mock()
        scene._result.result = ResultEnum.CORRECT
        scene.on_gaze_abandoned()
        scene.quit.assert_not_called()

    def test_gaze_loss_during_fixation_cancels(self) -> None:
        scene = make_scene()
        scene.quit = Mock()
        scene.on_gaze_abandoned()
        scene.quit.assert_called_once_with()

    def test_touchscreen_touch_abandons_pending_trial(self) -> None:
        scene = make_scene()
        scene.quit = Mock()
        scene.handle_event(Event(FINGERDOWN, x=0.5, y=0.5, finger_id=0, touch_id=0))
        scene.quit.assert_called_once_with()

    def test_manual_reward_is_not_double_counted(self) -> None:
        scene = make_scene()
        scene._give_manual_reward()
        scene._rewarder.give_reward.assert_called_once_with(600)
        self.assertEqual(scene._result.trial_reward_count, 0)
        self.assertEqual(scene._result.trial_manual_reward_count, 1)


# Level-up logic needs a second level to move to; the shipped config has one.
TWO_LEVELS = {1: LevelConfig(target_size_cm=1.5), 2: LevelConfig(target_size_cm=1.5)}


class SettlementTests(unittest.TestCase):
    def _settle(
        self, scene: Any, previous: list[ResultEnum]
    ) -> bool:
        trials = TrialResults(
            root={i: make_result(trial_id=i, result=r) for i, r in enumerate(previous, 2)}
        )
        scene._load_trials = Mock(return_value=trials)
        scene._save_trials = Mock()
        return scene._settlement()

    def test_last_level_never_levels_up(self) -> None:
        scene = make_scene(level_up_trial_count_threshold=3)
        scene._result.result = ResultEnum.CORRECT
        self.assertFalse(self._settle(scene, [ResultEnum.CORRECT] * 3))
        scene._mxbiflow.session.level_up.assert_not_called()

    def test_timeouts_count_against_correct_rate(self) -> None:
        scene = make_scene(level_up_trial_count_threshold=3, levels=TWO_LEVELS)
        scene._result.result = ResultEnum.CORRECT
        is_level_up = self._settle(scene, [ResultEnum.TIMEOUT] * 3)
        self.assertFalse(is_level_up)

    def test_levels_up_when_rate_and_count_are_met(self) -> None:
        scene = make_scene(level_up_trial_count_threshold=3, levels=TWO_LEVELS)
        scene._result.result = ResultEnum.CORRECT
        is_level_up = self._settle(scene, [ResultEnum.CORRECT] * 3)
        self.assertTrue(is_level_up)
        scene._mxbiflow.session.level_up.assert_called_once()


if __name__ == "__main__":
    unittest.main()
