# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from mxbiflow.models.animal import AnimalBaseInfo
from pydantic import ValidationError
from pygame import FINGERDOWN, FINGERUP, MOUSEBUTTONDOWN, MOUSEBUTTONUP, QUIT, Event
from pygame.sprite import Sprite

from assets import M001
from stages.visual_3pt_calibration.models import (
    Config,
    LevelConfig,
    Result,
    ResultEnum,
    Screen,
    StoredConfig,
    TrialResults,
)
from stages.visual_3pt_calibration.scene import (
    FixationPoint,
    Visual3ptCalibration,
)

# NOTE: gaze-tracking input (GAZEBUTTONDOWN) is not wired up yet; the
# corresponding tests below are commented out until the gaze driver and
# its event type exist. See visual_3pt_calibration.py for details.


def make_stored_config() -> StoredConfig:
    return StoredConfig(
        inter_trial_duration_ms=5000,
        reward_duration_ms=1000,
        attention_duration_ms=1000,
        fixation_duration_min_ms=150,
        fixation_duration_max_ms=300,
        levels={
            1: LevelConfig(
                reaction_window_duration_ms=5000,
                target_size_cm=1.5,
            ),
            2: LevelConfig(
                reaction_window_duration_ms=4000,
                target_size_cm=1.5,
            ),
            3: LevelConfig(
                reaction_window_duration_ms=4000,
                target_size_cm=1.5,
            ),
            4: LevelConfig(
                reaction_window_duration_ms=4000,
                target_size_cm=1.5,
            ),
        },
        border_width=40,
        level_up_threshold=0.8,
        level_up_trial_count_threshold=20,
        background_color=(127, 127, 127),
        fixation_point_radius=20,
        fixation_point_inner_radius_ratio=0.35,
        target_offset_ratio=0.2,
        touch_buffer_cm=1.0,
        pixels_per_cm=37.8,
        response_window_min_ms=50,
        response_window_max_ms=800,
        min_touch_duration_ms=100,
    )


def make_config(level: int = 1) -> Config:
    return Config(
        animal="test-animal",
        level=level,
        is_twitter_trial=True,
        screen_size=Screen(width=1920, height=1080),
        stored=make_stored_config(),
    )


def make_scene() -> Any:
    scene: Any = Visual3ptCalibration.__new__(Visual3ptCalibration)
    scene._config = make_config()
    scene._timer = Mock()
    scene._sprites = Mock()
    scene._correct_target = Mock()
    scene._fixation_point = Mock()
    scene._is_reaction_window_open = False
    scene._reaction_window_started_at_ms = None
    scene._pending_touch_started_at_ms = None
    scene._pending_touch_pos = None
    scene._result = SimpleNamespace(
        animal=SimpleNamespace(level=1, trial_id=1),
        result=ResultEnum.PENDING,
        trial_end_time=0,
        response_latency_ms=None,
        trial_reward_count=0,
        trial_manual_reward_count=0,
        touch_event=[],
        model_dump=Mock(return_value={}),
    )
    scene._on_inter_trial = Mock()
    return scene


def make_trial_result(
    *,
    trial_id: int,
    level: int = 1,
    result: ResultEnum = ResultEnum.CORRECT,
    reward_count: int = 0,
    manual_reward_count: int = 0,
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
        is_twitter_trial=True,
        result=result,
        trial_start_time=float(trial_id),
        trial_end_time=float(trial_id) + 1,
        trial_reward_count=reward_count,
        trial_manual_reward_count=manual_reward_count,
    )


def make_settlement_scene(
    *,
    past_trials: dict[int, Result] | None = None,
    current_result: ResultEnum = ResultEnum.CORRECT,
    trial_id: int = 21,
    level: int = 1,
) -> Any:
    scene = make_scene()
    scene._result = SimpleNamespace(
        animal=SimpleNamespace(level=level, trial_id=trial_id),
        result=current_result,
        trial_reward_count=0,
        trial_manual_reward_count=0,
    )
    scene._animal = SimpleNamespace(
        name="test-animal",
        current_stage=SimpleNamespace(
            level=level,
            stage_name="test-stage",
            get_context=Mock(
                return_value=TrialResults(
                    root=past_trials if past_trials is not None else {}
                )
            ),
        ),
    )
    scene._config = SimpleNamespace(
        total_levels=4,
        level_up_threshold=0.8,
        level_up_trial_count_threshold=20,
    )
    scene._mxbiflow = SimpleNamespace(session=Mock())
    scene._print_result = Mock()
    return scene


def make_past_correct(count: int, start: int = 1, level: int = 1) -> dict[int, Result]:
    return {
        trial_id: make_trial_result(trial_id=trial_id, level=level)
        for trial_id in range(start, start + count)
    }


class TrialInitializationTests(unittest.TestCase):
    @patch(
        "stages.visual_3pt_calibration.scene.get_mxbiflow"
    )
    def test_resources_are_prepared_before_trial_result(
        self, get_mxbiflow: Mock
    ) -> None:
        events: list[str] = []
        animal = Mock()
        session = Mock()
        session.require_current_animal.return_value = animal
        session.add_trial.side_effect = lambda: events.append("add_trial")
        get_mxbiflow.return_value = SimpleNamespace(
            session=session,
            mxbi=SimpleNamespace(
                rewarder=Mock(),
                timer=Mock(),
            ),
            timer=Mock(),
        )

        result = Mock()
        result_widget = Mock()
        with (
            patch.object(
                Visual3ptCalibration,
                "_init_config",
                side_effect=lambda *_: events.append("config") or make_config(),
            ),
            patch.object(
                Visual3ptCalibration,
                "_init_data_logger",
                side_effect=lambda *_: events.append("data_logger") or Mock(),
            ),
            patch.object(
                Visual3ptCalibration,
                "_init_control_buttons",
                side_effect=lambda: events.append("controls") or Mock(),
            ),
            patch.object(
                Visual3ptCalibration,
                "_init_visual_stimulus",
                side_effect=lambda *_: events.append("visual_stimulus"),
            ),
            patch.object(
                Visual3ptCalibration,
                "_init_result",
                side_effect=lambda *_: events.append("result") or result,
            ),
            patch.object(
                Visual3ptCalibration,
                "_init_result_widget",
                side_effect=lambda: events.append("result_widget") or result_widget,
            ),
            patch.object(Visual3ptCalibration, "_print_trial_config"),
        ):
            scene = Visual3ptCalibration()

        self.assertEqual(
            events,
            [
                "config",
                "data_logger",
                "controls",
                "visual_stimulus",
                "add_trial",
                "result",
                "result_widget",
            ],
        )
        self.assertIs(scene._result, result)
        self.assertIs(scene._result_widget, result_widget)

    @patch(
        "stages.visual_3pt_calibration.scene.time",
        return_value=123.0,
    )
    def test_result_uses_current_trial_metadata_and_time(self, _time: Mock) -> None:
        scene = make_scene()
        animal = SimpleNamespace(
            base_info=AnimalBaseInfo(
                animal="test-animal",
                trial_id=7,
                level=2,
                level_trial_id=3,
                animal_session_id=1,
                animal_session_trial_id=5,
            ),
            current_stage=SimpleNamespace(get_context=Mock()),
        )

        result = scene._init_result(animal, make_config(level=2))

        self.assertEqual(result.animal, animal.base_info)
        self.assertTrue(result.is_twitter_trial)
        self.assertNotIn("screen_size", result.model_dump())
        self.assertNotIn("stimulus", result.model_dump())
        self.assertEqual(result.trial_start_time, 123.0)
        self.assertEqual(result.trial_end_time, 123.0)

    def test_start_does_not_add_another_trial(self) -> None:
        scene = make_scene()
        scene._mxbiflow = SimpleNamespace(session=Mock())
        scene._result_widget = Mock()
        scene._result_summary = Mock(return_value="summary")
        scene._on_trial_start = Mock()

        scene.start()

        scene._mxbiflow.session.add_trial.assert_not_called()
        scene._result_widget.set_text.assert_called_once_with("summary")
        scene._on_trial_start.assert_called_once_with()

    @patch(
        "stages.visual_3pt_calibration.scene.time",
        return_value=456.0,
    )
    def test_trial_start_waits_for_attention_before_stimulus(self, _time: Mock) -> None:
        scene = make_scene()
        scene._play_stimulus = Mock()

        scene._on_trial_start()

        self.assertEqual(scene._result.trial_start_time, 456.0)
        scene._sprites.add.assert_called_once_with(scene._fixation_point)
        scene._timer.after.assert_called_once_with(1000, scene._play_stimulus)

    def test_result_has_no_cumulative_state(self) -> None:
        scene = make_scene()
        animal = SimpleNamespace(
            base_info=AnimalBaseInfo(
                animal="test-animal",
                trial_id=8,
                level=2,
                level_trial_id=1,
                animal_session_id=1,
                animal_session_trial_id=6,
            ),
            current_stage=SimpleNamespace(get_context=Mock()),
        )

        result = scene._init_result(animal, make_config(level=2))

        animal.current_stage.get_context.assert_not_called()
        self.assertEqual(result.trial_reward_count, 0)
        self.assertEqual(result.trial_manual_reward_count, 0)
        dumped = result.model_dump()
        for field in (
            "correct_trial_count",
            "background_touch_count",
            "timeout_trial_count",
            "cancelled_trial_count",
            "reward_count",
            "manual_reward_count",
            "level_correct_trial_count",
            "level_background_touch_count",
            "level_valid_trial_count",
            "correct_rate",
            "level_correct_rate",
        ):
            self.assertNotIn(field, dumped)


class TargetSizeConfigTests(unittest.TestCase):
    def test_reaction_window_duration_is_level_specific(self) -> None:
        self.assertEqual(make_config(level=1).reaction_window_duration_ms, 5000)
        self.assertEqual(make_config(level=2).reaction_window_duration_ms, 4000)

    def test_visual_delay_defaults_to_500(self) -> None:
        self.assertEqual(make_config().visual_delay_ms, 500)
        self.assertEqual(make_stored_config().visual_delay_ms, 500)

    def test_fixation_duration_range_defaults(self) -> None:
        self.assertEqual(make_config().fixation_duration_min_ms, 150)
        self.assertEqual(make_config().fixation_duration_max_ms, 300)

    def test_fixation_duration_max_must_be_at_least_min(self) -> None:
        with self.assertRaises(ValidationError):
            StoredConfig.model_validate(
                {
                    **make_stored_config().model_dump(),
                    "fixation_duration_min_ms": 300,
                    "fixation_duration_max_ms": 150,
                }
            )

    def test_fixation_duration_max_equal_to_min_is_allowed(self) -> None:
        config = StoredConfig.model_validate(
            {
                **make_stored_config().model_dump(),
                "fixation_duration_min_ms": 200,
                "fixation_duration_max_ms": 200,
            }
        )
        self.assertEqual(config.fixation_duration_min_ms, 200)
        self.assertEqual(config.fixation_duration_max_ms, 200)

    def test_target_size_cm_is_fixed_at_1_5_for_all_levels(self) -> None:
        self.assertEqual(make_config(level=1).target_size_cm, 1.5)
        self.assertEqual(make_config(level=2).target_size_cm, 1.5)
        self.assertEqual(make_config(level=3).target_size_cm, 1.5)
        self.assertEqual(make_config(level=4).target_size_cm, 1.5)

    def test_target_size_px_converts_cm_to_pixels(self) -> None:
        # target_size_cm=1.5 * pixels_per_cm=37.8 = 56.7
        self.assertAlmostEqual(make_config().target_size_px, 56.7)

    def test_reaction_window_duration_must_exceed_response_window_max(self) -> None:
        with self.assertRaises(ValidationError):
            StoredConfig.model_validate(
                {
                    **make_stored_config().model_dump(),
                    "response_window_max_ms": 800,
                    "levels": {
                        1: {
                            "reaction_window_duration_ms": 500,
                            "target_size_cm": 1.5,
                        },
                    },
                }
            )

    def test_target_size_cm_is_validated(self) -> None:
        with self.assertRaises(ValidationError):
            LevelConfig.model_validate(
                {
                    "reaction_window_duration_ms": 5000,
                    "target_size_cm": 0,
                }
            )

    def test_runtime_properties_are_not_serialized(self) -> None:
        dumped = make_config().model_dump()

        self.assertNotIn("stored", dumped)
        self.assertNotIn("reward_duration_ms", dumped)
        self.assertEqual(make_config().reward_duration_ms, 1000)

    def test_background_color_defaults_to_rgb_127(self) -> None:
        self.assertEqual(make_config().background_color, (127, 127, 127))
        self.assertEqual(StoredConfig.model_fields["background_color"].default, (127, 127, 127))

    def test_fixation_point_config_defaults(self) -> None:
        self.assertEqual(make_config().fixation_point_radius, 20)
        self.assertEqual(make_config().fixation_point_inner_radius_ratio, 0.35)

    def test_fixation_point_inner_radius_ratio_is_validated(self) -> None:
        with self.assertRaises(ValidationError):
            StoredConfig.model_validate(
                {
                    **make_stored_config().model_dump(),
                    "fixation_point_inner_radius_ratio": 1.0,
                }
            )

    def test_target_offset_ratio_defaults_to_0_2(self) -> None:
        self.assertEqual(make_config().target_offset_ratio, 0.2)
        self.assertEqual(StoredConfig.model_fields["target_offset_ratio"].default, 0.2)

    def test_target_offset_ratio_is_validated(self) -> None:
        for invalid_value in (0, 0.5, -0.1, 1.0):
            with self.subTest(value=invalid_value):
                with self.assertRaises(ValidationError):
                    StoredConfig.model_validate(
                        {
                            **make_stored_config().model_dump(),
                            "target_offset_ratio": invalid_value,
                        }
                    )

    def test_animal_config_snapshot_is_created_once_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "config.json"
            source_text = '{"default": ' + make_stored_config().model_dump_json() + "}"
            source_path.write_text(source_text, encoding="utf-8")
            animal = SimpleNamespace(
                name="test-animal",
                current_stage=SimpleNamespace(level=1),
            )
            mxbiflow = SimpleNamespace(
                session=SimpleNamespace(
                    absolute_animal_data_path=lambda _animal: root / "animal-session"
                ),
                mxbi=SimpleNamespace(screen_size=Screen(width=1920, height=1080)),
            )
            scene = make_scene()

            with patch(
                "stages.visual_3pt_calibration.scene.PATH",
                source_path,
            ):
                first = scene._init_config(animal, mxbiflow)
                snapshot = (
                    root
                    / "animal-session"
                    / "visual_3pt_calibration"
                    / "config.json"
                )
                self.assertTrue(snapshot.exists())
                self.assertEqual(
                    snapshot.read_text(encoding="utf-8"),
                    source_text,
                )
                second = scene._init_config(animal, mxbiflow)

        self.assertEqual(first.animal, "test-animal")
        self.assertEqual(first.level, 1)
        self.assertEqual(first.target_size_cm, 1.5)
        self.assertTrue(first.is_twitter_trial)
        self.assertEqual(second.animal, "test-animal")


class InputModalityTests(unittest.TestCase):
    """Covers the pointer-like input modalities (mouse, finger touch) and
    their down/up pairing used to measure press duration. Every
    completed press is now forwarded to _on_touched (with its measured
    duration) regardless of length - duration only affects
    classification, it no longer gates whether the touch is forwarded at
    all."""

    def test_mouse_down_then_up_is_forwarded_to_on_touched(self) -> None:
        scene = make_scene()
        scene._control_buttons = []
        scene._on_touched = Mock()

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            side_effect=[1.0, 1.2],
        ):
            scene.handle_event(Event(MOUSEBUTTONDOWN, button=1, pos=(10, 20)))
            scene.handle_event(Event(MOUSEBUTTONUP, button=1, pos=(10, 20)))

        scene._on_touched.assert_called_once_with((10, 20), 200.0)

    def test_mouse_click_consumed_by_control_button_is_not_forwarded(self) -> None:
        scene = make_scene()
        button = Mock()
        button.handle_event.return_value = True
        scene._control_buttons = [button]
        scene._on_touched = Mock()

        mouse_event = Event(MOUSEBUTTONDOWN, button=1, pos=(10, 20))
        scene.handle_event(mouse_event)

        scene._on_touched.assert_not_called()

    def test_finger_down_then_up_is_converted_to_screen_pixels(self) -> None:
        scene = make_scene()
        scene._control_buttons = []
        scene._on_touched = Mock()
        scene._mxbiflow = SimpleNamespace(
            mxbi=SimpleNamespace(screen_size=Screen(width=1000, height=500))
        )

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            side_effect=[1.0, 1.2],
        ):
            scene.handle_event(Event(FINGERDOWN, x=0.5, y=0.25))
            scene.handle_event(Event(FINGERUP, x=0.5, y=0.25))

        scene._on_touched.assert_called_once_with((500, 125), 200.0)

    def test_touch_shorter_than_min_duration_is_still_forwarded(self) -> None:
        # Unlike the previous behavior, a too-short press is no longer
        # silently discarded - it's forwarded with its (short) duration,
        # and _on_touched decides the classification (an on-target short
        # press reads as TIMEOUT/anticipation, an off-target one as
        # INCORRECT).
        scene = make_scene()
        scene._control_buttons = []
        scene._on_touched = Mock()

        # min_touch_duration_ms is 100 in the test config; 50ms press
        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            side_effect=[1.0, 1.05],
        ):
            scene.handle_event(Event(MOUSEBUTTONDOWN, button=1, pos=(10, 20)))
            scene.handle_event(Event(MOUSEBUTTONUP, button=1, pos=(10, 20)))

        scene._on_touched.assert_called_once_with((10, 20), 50.0)
        self.assertEqual(len(scene._result.touch_event), 1)

    def test_up_without_matching_down_is_ignored(self) -> None:
        scene = make_scene()
        scene._control_buttons = []
        scene._on_touched = Mock()

        scene.handle_event(Event(MOUSEBUTTONUP, button=1, pos=(10, 20)))

        scene._on_touched.assert_not_called()
        self.assertEqual(len(scene._result.touch_event), 0)

    # TODO(gaze): re-enable once GAZEBUTTONDOWN is allocated and a gaze
    # driver actually posts it (see visual_3pt_calibration.py).
    #
    # def test_gaze_input_is_converted_to_screen_pixels(self) -> None:
    #     scene = make_scene()
    #     scene._control_buttons = []
    #     scene._on_touched = Mock()
    #     scene._mxbiflow = SimpleNamespace(
    #         mxbi=SimpleNamespace(screen_size=Screen(width=1000, height=500))
    #     )
    #
    #     gaze_event = Event(GAZEBUTTONDOWN, x=0.75, y=0.5)
    #     scene.handle_event(gaze_event)
    #
    #     scene._on_touched.assert_called_once()
    #     forwarded = scene._on_touched.call_args.args[0]
    #     self.assertEqual(forwarded.type, MOUSEBUTTONDOWN)
    #     self.assertEqual(forwarded.button, 1)
    #     self.assertEqual(forwarded.pos, (750, 250))
    #
    # def test_gaze_input_reuses_same_touch_classification_path(self) -> None:
    #     scene = make_scene()
    #     scene._control_buttons = []
    #     scene._classify_touch = Mock(return_value=ResultEnum.CORRECT)
    #     scene._on_correct = Mock()
    #     scene._is_reaction_window_open = True
    #     scene._reaction_window_started_at_ms = 1000
    #     scene._mxbiflow = SimpleNamespace(
    #         mxbi=SimpleNamespace(screen_size=Screen(width=1000, height=500))
    #     )
    #
    #     with patch(
    #         "stages.visual_3pt_calibration.scene.monotonic",
    #         return_value=2.0,
    #     ):
    #         scene.handle_event(Event(GAZEBUTTONDOWN, x=0.75, y=0.5))
    #
    #     scene._classify_touch.assert_called_once_with((750, 250))
    #     scene._on_correct.assert_called_once_with()


class FixationPointTests(unittest.TestCase):
    def test_fixation_point_is_a_sprite_centered_at_given_point(self) -> None:
        point = FixationPoint((100, 200), radius=20, inner_radius_ratio=0.35)

        self.assertIsInstance(point, Sprite)
        self.assertEqual(point.rect.center, (100, 200))
        self.assertEqual(point.image.get_width(), 40)
        self.assertEqual(point.image.get_height(), 40)

    def test_fixation_point_draws_green_outer_and_black_inner_circle(self) -> None:
        point = FixationPoint((50, 50), radius=20, inner_radius_ratio=0.5)

        center_pixel = point.image.get_at((20, 20))
        self.assertEqual(center_pixel[:3], (0, 0, 0))

        edge_pixel = point.image.get_at((20, 2))
        self.assertEqual(edge_pixel[:3], (0, 200, 0))

    def test_fixation_point_shown_at_trial_start_and_removed_before_stimulus(
        self,
    ) -> None:
        scene = make_scene()
        scene._reaction_window_started_at_ms = None

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=1.0,
        ):
            scene._on_trial_start()
            scene._show_visual_stimulus()

        scene._sprites.add.assert_any_call(scene._fixation_point)
        scene._sprites.remove.assert_any_call(scene._fixation_point)

    def test_fixation_duration_is_randomized_within_configured_range(self) -> None:
        scene = make_scene()
        scene._timer = Mock()

        with patch(
            "stages.visual_3pt_calibration.scene.randint",
            return_value=217,
        ) as randint_mock:
            scene._on_trial_start()

        # test config: fixation_duration_min_ms=150, max_ms=300
        randint_mock.assert_called_once_with(150, 300)
        scene._timer.after.assert_called_once_with(217, scene._play_stimulus)

    def test_fixation_duration_draws_a_new_value_each_trial(self) -> None:
        scene = make_scene()
        scene._timer = Mock()

        with patch(
            "stages.visual_3pt_calibration.scene.randint",
            side_effect=[180, 260],
        ):
            scene._on_trial_start()
            scene._on_trial_start()

        first_call, second_call = scene._timer.after.call_args_list
        self.assertEqual(first_call.args[0], 180)
        self.assertEqual(second_call.args[0], 260)


class BackgroundColorTests(unittest.TestCase):
    def test_backgrounds_use_configured_rgb_color(self) -> None:
        scene = make_scene()

        with patch(
            "stages.visual_3pt_calibration.scene"
            ".create_background"
        ) as create_background:
            scene._init_backgrounds(make_config())

        self.assertEqual(create_background.call_count, 2)
        for call_args in create_background.call_args_list:
            self.assertEqual(
                call_args.kwargs["background_color"],
                (127, 127, 127),
            )

    def test_attention_and_stimulus_backgrounds_share_the_same_color(self) -> None:
        scene = make_scene()
        scene._init_backgrounds(make_config())

        attention_pixel = scene._attention_background.get_at((100, 100))
        stimulus_pixel = scene._stimulus_background.get_at((100, 100))

        self.assertEqual(attention_pixel[:3], (127, 127, 127))
        self.assertEqual(stimulus_pixel[:3], (127, 127, 127))


class TouchFlowTests(unittest.TestCase):
    def test_touch_outside_reaction_window_is_ignored(self) -> None:
        scene = make_scene()
        scene._is_reaction_window_open = False
        scene._classify_touch = Mock()

        scene._on_touched((10, 20), 150.0)

        scene._classify_touch.assert_not_called()
        scene._sprites.remove.assert_not_called()
        scene._on_inter_trial.assert_not_called()

    # --- Row 1: on target, >=100ms hold, latency in [50, 800] -> CORRECT

    def test_correct_on_target_long_enough_within_window(self) -> None:
        scene = make_scene()
        scene._is_reaction_window_open = True
        scene._reaction_window_started_at_ms = 1000
        scene._classify_touch = Mock(return_value=ResultEnum.CORRECT)
        scene._on_correct = Mock()

        # latency = 1200 - 1000 = 200ms, within [50, 800]; duration 150ms
        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=1.2,
        ):
            scene._on_touched((10, 20), 150.0)

        self.assertFalse(scene._is_reaction_window_open)
        scene._sprites.remove.assert_called_once_with([scene._correct_target])
        scene._classify_touch.assert_called_once_with((10, 20))
        scene._on_correct.assert_called_once_with()
        self.assertEqual(scene._result.response_latency_ms, 200)

    def test_correct_at_response_window_boundaries(self) -> None:
        for latency_s, label in ((1.05, "min boundary"), (1.8, "max boundary")):
            with self.subTest(label):
                scene = make_scene()
                scene._is_reaction_window_open = True
                scene._reaction_window_started_at_ms = 1000
                scene._classify_touch = Mock(return_value=ResultEnum.CORRECT)
                scene._on_correct = Mock()

                with patch(
                    "stages.visual_3pt_calibration.scene"
                    ".monotonic",
                    return_value=latency_s,
                ):
                    scene._on_touched((10, 20), 150.0)

                scene._on_correct.assert_called_once_with()

    # --- Rows 2, 3, 7a: on target but mistimed or too short -> TIMEOUT

    def test_timeout_on_target_but_latency_after_max(self) -> None:
        scene = make_scene()
        scene._is_reaction_window_open = True
        scene._reaction_window_started_at_ms = 1000
        scene._classify_touch = Mock(return_value=ResultEnum.CORRECT)
        scene._on_timeout = Mock()
        scene._on_correct = Mock()

        # latency = 1900 - 1000 = 900ms, above the 800ms maximum
        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=1.9,
        ):
            scene._on_touched((10, 20), 150.0)

        self.assertFalse(scene._is_reaction_window_open)
        scene._on_timeout.assert_called_once_with()
        scene._on_correct.assert_not_called()

    def test_timeout_on_target_but_latency_before_min(self) -> None:
        scene = make_scene()
        scene._is_reaction_window_open = True
        scene._reaction_window_started_at_ms = 1000
        scene._classify_touch = Mock(return_value=ResultEnum.CORRECT)
        scene._on_timeout = Mock()
        scene._on_correct = Mock()

        # latency = 1020 - 1000 = 20ms, below the 50ms minimum
        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=1.02,
        ):
            scene._on_touched((10, 20), 150.0)

        self.assertFalse(scene._is_reaction_window_open)
        scene._on_timeout.assert_called_once_with()
        scene._on_correct.assert_not_called()

    def test_timeout_on_target_but_too_short_duration(self) -> None:
        scene = make_scene()
        scene._is_reaction_window_open = True
        scene._reaction_window_started_at_ms = 1000
        scene._classify_touch = Mock(return_value=ResultEnum.CORRECT)
        scene._on_timeout = Mock()
        scene._on_correct = Mock()

        # within the response window, but duration (50ms) < min (100ms)
        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=1.2,
        ):
            scene._on_touched((10, 20), 50.0)

        scene._on_timeout.assert_called_once_with()
        scene._on_correct.assert_not_called()

    def test_timeout_does_not_record_response_latency(self) -> None:
        scene = make_scene()
        scene._is_reaction_window_open = True
        scene._reaction_window_started_at_ms = 1000
        scene._classify_touch = Mock(return_value=ResultEnum.CORRECT)

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=1.9,
        ):
            scene._on_touched((10, 20), 150.0)

        self.assertIsNone(scene._result.response_latency_ms)

    # --- Rows 4, 5, 6, 7b: off target, any timing/duration -> INCORRECT

    def test_incorrect_off_target_within_window(self) -> None:
        scene = make_scene()
        scene._is_reaction_window_open = True
        scene._reaction_window_started_at_ms = 1000
        scene._classify_touch = Mock(return_value=ResultEnum.BACKGROUND_TOUCHED)
        scene._on_incorrect = Mock()

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=1.2,
        ):
            scene._on_touched((10, 20), 150.0)

        self.assertIsNone(scene._result.response_latency_ms)
        scene._on_incorrect.assert_called_once_with()

    def test_incorrect_off_target_after_max_window(self) -> None:
        scene = make_scene()
        scene._is_reaction_window_open = True
        scene._reaction_window_started_at_ms = 1000
        scene._classify_touch = Mock(return_value=ResultEnum.BACKGROUND_TOUCHED)
        scene._on_incorrect = Mock()

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=1.9,
        ):
            scene._on_touched((10, 20), 150.0)

        scene._on_incorrect.assert_called_once_with()

    def test_incorrect_off_target_before_min_window(self) -> None:
        scene = make_scene()
        scene._is_reaction_window_open = True
        scene._reaction_window_started_at_ms = 1000
        scene._classify_touch = Mock(return_value=ResultEnum.BACKGROUND_TOUCHED)
        scene._on_incorrect = Mock()

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=1.02,
        ):
            scene._on_touched((10, 20), 150.0)

        scene._on_incorrect.assert_called_once_with()

    def test_incorrect_off_target_with_too_short_duration(self) -> None:
        scene = make_scene()
        scene._is_reaction_window_open = True
        scene._reaction_window_started_at_ms = 1000
        scene._classify_touch = Mock(return_value=ResultEnum.BACKGROUND_TOUCHED)
        scene._on_incorrect = Mock()

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=1.2,
        ):
            scene._on_touched((10, 20), 50.0)

        scene._on_incorrect.assert_called_once_with()

    # --- _classify_touch unit tests (on/off-target probe, unchanged) ---

    def test_target_touch_is_classified_as_correct(self) -> None:
        scene = make_scene()
        rect = Mock()
        rect.width = 100
        rect.height = 100
        rect.inflate.return_value = SimpleNamespace(
            collidepoint=Mock(return_value=True)
        )
        scene._correct_target = SimpleNamespace(rect=rect)

        self.assertEqual(
            scene._classify_touch((10, 20)),
            ResultEnum.CORRECT,
        )

    def test_non_target_touch_is_classified_as_background(self) -> None:
        scene = make_scene()
        rect = Mock()
        rect.width = 100
        rect.height = 100
        rect.inflate.return_value = SimpleNamespace(
            collidepoint=Mock(return_value=False)
        )
        scene._correct_target = SimpleNamespace(rect=rect)

        self.assertEqual(
            scene._classify_touch((10, 20)),
            ResultEnum.BACKGROUND_TOUCHED,
        )

    def test_touch_buffer_inflates_hit_box_by_configured_physical_distance(
        self,
    ) -> None:
        scene = make_scene()
        rect = Mock()
        rect.width = 100
        rect.height = 80
        inflated = SimpleNamespace(collidepoint=Mock(return_value=True))
        rect.inflate.return_value = inflated
        scene._correct_target = SimpleNamespace(rect=rect)

        # touch_buffer_cm=1.0, pixels_per_cm=37.8 in the test config
        # => buffer_px = round(1.0 * 37.8) = 38
        scene._classify_touch((10, 20))

        rect.inflate.assert_called_once_with(38, 38)
        inflated.collidepoint.assert_called_once_with((10, 20))

    def test_missing_target_rect_is_classified_as_background(self) -> None:
        scene = make_scene()
        scene._correct_target = SimpleNamespace(rect=None)

        self.assertEqual(
            scene._classify_touch((10, 20)),
            ResultEnum.BACKGROUND_TOUCHED,
        )


class TouchDurationGatingTests(unittest.TestCase):
    """Covers _on_touch_down/_on_touch_up pairing and duration
    measurement. Duration no longer gates whether a touch is forwarded -
    every completed press reaches _on_touched with its measured
    duration, which only affects classification (see TouchFlowTests)."""

    def test_touch_down_records_start_time_and_position(self) -> None:
        scene = make_scene()

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=1.0,
        ):
            scene._on_touch_down((10, 20))

        self.assertEqual(scene._pending_touch_started_at_ms, 1000)
        self.assertEqual(scene._pending_touch_pos, (10, 20))

    def test_touch_up_clears_pending_state(self) -> None:
        scene = make_scene()
        scene._on_touched = Mock()

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            side_effect=[1.0, 1.2],
        ):
            scene._on_touch_down((10, 20))
            scene._on_touch_up((10, 20))

        self.assertIsNone(scene._pending_touch_started_at_ms)
        self.assertIsNone(scene._pending_touch_pos)

    def test_short_press_is_logged_and_forwarded_with_its_duration(self) -> None:
        scene = make_scene()
        scene._on_touched = Mock()

        # 99ms press, under the 100ms minimum - still forwarded, but
        # with a short duration that _on_touched will classify
        # accordingly (TIMEOUT if on target, INCORRECT if off target).
        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            side_effect=[1.0, 1.099],
        ):
            scene._on_touch_down((10, 20))
            scene._on_touch_up((10, 20))

        scene._on_touched.assert_called_once()
        args = scene._on_touched.call_args.args
        self.assertEqual(args[0], (10, 20))
        self.assertAlmostEqual(args[1], 99.0, places=0)
        self.assertEqual(len(scene._result.touch_event), 1)
        self.assertEqual(scene._result.touch_event[0].x, 10)
        self.assertEqual(scene._result.touch_event[0].y, 20)

    def test_long_enough_press_is_forwarded_with_its_duration(self) -> None:
        scene = make_scene()
        scene._on_touched = Mock()

        # 150ms press, over the 100ms minimum
        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            side_effect=[1.0, 1.15],
        ):
            scene._on_touch_down((10, 20))
            scene._on_touch_up((10, 20))

        scene._on_touched.assert_called_once()
        args = scene._on_touched.call_args.args
        self.assertEqual(args[0], (10, 20))
        self.assertAlmostEqual(args[1], 150.0, places=0)


class ResponseWindowConfigTests(unittest.TestCase):
    def test_response_window_defaults(self) -> None:
        config = make_config()
        self.assertEqual(config.response_window_min_ms, 50)
        self.assertEqual(config.response_window_max_ms, 800)

    def test_min_touch_duration_default(self) -> None:
        self.assertEqual(make_config().min_touch_duration_ms, 100)

    def test_touch_buffer_cm_and_pixels_per_cm_defaults(self) -> None:
        config = make_config()
        self.assertEqual(config.stored.touch_buffer_cm, 1.0)
        self.assertEqual(config.stored.pixels_per_cm, 37.8)

    def test_touch_buffer_px_converts_cm_to_pixels(self) -> None:
        config = make_config()
        # touch_buffer_cm=1.0 * pixels_per_cm=37.8
        self.assertAlmostEqual(config.touch_buffer_px, 37.8)

    def test_response_window_max_must_exceed_min(self) -> None:
        with self.assertRaises(ValidationError):
            StoredConfig.model_validate(
                {
                    **make_stored_config().model_dump(),
                    "response_window_min_ms": 800,
                    "response_window_max_ms": 800,
                }
            )

    def test_touch_buffer_cm_rejects_negative(self) -> None:
        with self.assertRaises(ValidationError):
            StoredConfig.model_validate(
                {
                    **make_stored_config().model_dump(),
                    "touch_buffer_cm": -0.1,
                }
            )

    def test_pixels_per_cm_rejects_zero_and_negative(self) -> None:
        for invalid_value in (0, -37.8):
            with self.subTest(value=invalid_value):
                with self.assertRaises(ValidationError):
                    StoredConfig.model_validate(
                        {
                            **make_stored_config().model_dump(),
                            "pixels_per_cm": invalid_value,
                        }
                    )


class RewardCountingTests(unittest.TestCase):
    def test_automatic_reward_updates_trial_count(self) -> None:
        scene = make_scene()
        scene._rewarder = Mock()

        scene._give_reward()

        scene._rewarder.give_reward.assert_called_once_with(1000)
        self.assertEqual(scene._result.trial_reward_count, 1)
        self.assertEqual(scene._result.trial_manual_reward_count, 0)

    def test_manual_rewards_update_trial_counts(self) -> None:
        scene = make_scene()
        scene._rewarder = Mock()

        scene._give_manual_reward()
        scene._give_manual_reward()

        self.assertEqual(scene._rewarder.give_reward.call_count, 2)
        self.assertEqual(scene._result.trial_reward_count, 2)
        self.assertEqual(scene._result.trial_manual_reward_count, 2)

    def test_failed_reward_does_not_update_counts(self) -> None:
        scene = make_scene()
        scene._rewarder = Mock()
        scene._rewarder.give_reward.side_effect = RuntimeError("reward failure")

        with self.assertRaisesRegex(RuntimeError, "reward failure"):
            scene._give_manual_reward()

        self.assertEqual(scene._result.trial_reward_count, 0)
        self.assertEqual(scene._result.trial_manual_reward_count, 0)

    def test_result_summary_uses_trial_rewards(self) -> None:
        scene = make_scene()
        past = {
            1: make_trial_result(trial_id=1, reward_count=3, manual_reward_count=2),
            2: make_trial_result(
                trial_id=2,
                result=ResultEnum.INCORRECT,
            ),
        }
        scene._animal = SimpleNamespace(
            name="test-animal",
            trial_id=3,
            current_stage=SimpleNamespace(
                level=1,
                stage_name="test-stage",
                get_context=Mock(return_value=TrialResults(root=past)),
            ),
        )
        scene._result.animal = SimpleNamespace(level=1, trial_id=3)

        summary = scene._result_summary()

        self.assertIn("Correct 1 | Error 1 | Reward 5", summary)
        self.assertIn("Rate 50.0%", summary)


class BehavioralOutcomeTests(unittest.TestCase):
    def test_correct_sets_outcome_and_gives_reward(self) -> None:
        scene = make_scene()
        scene._give_reward = Mock()

        scene._on_correct()

        self.assertEqual(scene._result.result, ResultEnum.CORRECT)
        scene._give_reward.assert_called_once_with()
        scene._on_inter_trial.assert_called_once_with()

    def test_incorrect_sets_outcome(self) -> None:
        scene = make_scene()

        scene._on_incorrect()

        self.assertEqual(scene._result.result, ResultEnum.INCORRECT)
        scene._on_inter_trial.assert_called_once_with()

    def test_timeout_sets_outcome(self) -> None:
        scene = make_scene()

        scene._on_timeout()

        self.assertEqual(scene._result.result, ResultEnum.TIMEOUT)
        scene._on_inter_trial.assert_called_once_with()


class SettlementTests(unittest.TestCase):
    def test_settlement_saves_current_trial_in_context(self) -> None:
        scene = make_settlement_scene(trial_id=7)

        scene._settlement()

        session = scene._mxbiflow.session
        self.assertEqual(session.set_stage_context.call_count, 1)
        trials = session.set_stage_context.call_args.args[1]
        self.assertIs(trials.root[7], scene._result)
        self.assertEqual(
            session.set_stage_context.call_args.kwargs["animal"],
            "test-animal",
        )
        self.assertEqual(
            session.set_stage_context.call_args.kwargs["stage"],
            "test-stage",
        )

    def test_repeated_settlement_overwrites_same_trial_id(self) -> None:
        scene = make_settlement_scene(
            past_trials={5: make_trial_result(trial_id=5, result=ResultEnum.TIMEOUT)},
            trial_id=5,
        )

        scene._settlement()
        scene._settlement()

        session = scene._mxbiflow.session
        self.assertEqual(session.set_stage_context.call_count, 2)
        for call_args in session.set_stage_context.call_args_list:
            trials = call_args.args[1]
            self.assertEqual(list(trials.root), [5])
            self.assertIs(trials.root[5], scene._result)

    def test_timeout_is_excluded_from_correct_rate(self) -> None:
        past = make_past_correct(20)
        past.update(
            {
                trial_id: make_trial_result(
                    trial_id=trial_id, result=ResultEnum.TIMEOUT
                )
                for trial_id in range(21, 24)
            }
        )
        scene = make_settlement_scene(
            past_trials=past,
            current_result=ResultEnum.CORRECT,
            trial_id=24,
        )

        is_level_up = scene._settlement()

        self.assertTrue(is_level_up)
        scene._mxbiflow.session.level_up.assert_called_once()

    def test_only_timeouts_produce_zero_correct_rate(self) -> None:
        past = {
            trial_id: make_trial_result(trial_id=trial_id, result=ResultEnum.TIMEOUT)
            for trial_id in range(1, 21)
        }
        scene = make_settlement_scene(
            past_trials=past,
            current_result=ResultEnum.TIMEOUT,
            trial_id=21,
        )

        is_level_up = scene._settlement()

        self.assertFalse(is_level_up)
        scene._mxbiflow.session.level_up.assert_not_called()

    def test_cancelled_trial_does_not_trigger_level_up(self) -> None:
        scene = make_settlement_scene(
            past_trials=make_past_correct(20),
            current_result=ResultEnum.CANCELLED,
            trial_id=21,
        )

        scene._settlement()

        scene._mxbiflow.session.level_up.assert_not_called()
        scene._mxbiflow.session.set_current_stage.assert_not_called()
        self.assertEqual(scene._mxbiflow.session.set_stage_context.call_count, 1)

    def test_level_up_requires_more_than_configured_trial_count(self) -> None:
        for count, expected_level_up in ((19, False), (20, True)):
            with self.subTest(count=count):
                scene = make_settlement_scene(
                    past_trials=make_past_correct(count),
                    trial_id=count + 1,
                )

                is_level_up = scene._settlement()

                self.assertEqual(is_level_up, expected_level_up)
                self.assertEqual(
                    scene._mxbiflow.session.level_up.called,
                    expected_level_up,
                )

    def test_level_up_uses_level_correct_rate(self) -> None:
        for incorrect_count, expected_level_up in ((4, True), (5, False)):
            with self.subTest(incorrect_count=incorrect_count):
                past = make_past_correct(16)
                past.update(
                    {
                        trial_id: make_trial_result(
                            trial_id=trial_id,
                            result=ResultEnum.INCORRECT,
                        )
                        for trial_id in range(17, 17 + incorrect_count)
                    }
                )
                scene = make_settlement_scene(
                    past_trials=past,
                    trial_id=17 + incorrect_count,
                )

                is_level_up = scene._settlement()

                self.assertEqual(is_level_up, expected_level_up)

    def test_non_final_level_calls_level_up(self) -> None:
        scene = make_settlement_scene(
            past_trials=make_past_correct(20),
            trial_id=21,
        )

        is_level_up = scene._settlement()

        self.assertTrue(is_level_up)
        scene._mxbiflow.session.level_up.assert_called_once_with(
            animal="test-animal",
            stage="test-stage",
        )
        scene._mxbiflow.session.set_current_stage.assert_not_called()

    def test_highest_level_does_not_level_up(self) -> None:
        scene = make_settlement_scene(
            past_trials=make_past_correct(20, level=4),
            trial_id=21,
            level=4,
        )

        is_level_up = scene._settlement()

        self.assertFalse(is_level_up)
        scene._mxbiflow.session.level_up.assert_not_called()
        scene._mxbiflow.session.set_current_stage.assert_not_called()

    def test_final_level_without_criteria_stays_in_stage(self) -> None:
        scene = make_settlement_scene(
            past_trials=make_past_correct(5, level=4),
            trial_id=6,
            level=4,
        )

        is_level_up = scene._settlement()

        self.assertFalse(is_level_up)
        scene._mxbiflow.session.level_up.assert_not_called()

    def test_timeout_counts_as_level_trial(self) -> None:
        scene = make_settlement_scene(
            past_trials=make_past_correct(20),
            current_result=ResultEnum.TIMEOUT,
            trial_id=21,
        )

        is_level_up = scene._settlement()

        self.assertTrue(is_level_up)


class QuitFlowTests(unittest.TestCase):
    def test_quit_without_reason_falls_back_to_cancelled(self) -> None:
        scene = make_scene()
        scene._settlement = Mock(return_value=False)
        scene._print_result = Mock()
        scene._data_logger = Mock()
        scene._timer = Mock()
        scene._running = True

        scene.quit()

        self.assertEqual(scene._result.result, ResultEnum.CANCELLED)
        scene._settlement.assert_called_once_with()
        scene._print_result.assert_called_once_with(False)
        scene._data_logger.save.assert_called_once_with({})
        scene._timer.cancel_all.assert_called_once_with()
        self.assertFalse(scene._running)

    def test_request_game_quit_posts_quit_event(self) -> None:
        scene = make_scene()

        with patch(
            "stages.visual_3pt_calibration.scene"
            ".event.post"
        ) as post:
            scene._request_game_quit()

        post.assert_called_once()
        posted = post.call_args.args[0]
        self.assertEqual(posted.type, QUIT)


class StructureConsistencyTests(unittest.TestCase):
    def test_scene_name_is_canonical(self) -> None:
        self.assertEqual(
            Visual3ptCalibration.name(), "visual_3pt_calibration"
        )

    def test_no_audio_dependencies_are_imported(self) -> None:
        module_path = Path("stages/visual_3pt_calibration/scene.py")
        source = module_path.read_text(encoding="utf-8")

        for forbidden in ("aplayer", "audioplayer", "LoadedWav", "play_wav", "CLICKER_PATH", "wav_path"):
            self.assertNotIn(
                forbidden,
                source,
                msg=f"'{forbidden}' should not appear in the visual stage",
            )

    def test_response_window_always_schedules_a_hard_close(self) -> None:
        # Unlike the old reaction_timeout_enabled flag, this timer is now
        # unconditional - it's what guarantees the trial always proceeds
        # to the next one, even with no touch at all.
        scene = make_scene()
        scene._sprites = Mock()
        scene._timer = Mock()
        scene._config = SimpleNamespace(
            reaction_window_duration_ms=5000,
        )

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=3.0,
        ):
            scene._show_visual_stimulus()

        self.assertTrue(scene._is_reaction_window_open)
        self.assertEqual(scene._reaction_window_started_at_ms, 3000)
        scene._sprites.add.assert_called_once_with([scene._correct_target])
        scene._timer.after.assert_called_once_with(
            5000, scene._on_response_window_elapsed
        )

    def test_response_window_elapsed_marks_timeout_and_proceeds(self) -> None:
        scene = make_scene()
        scene._is_reaction_window_open = True

        scene._on_response_window_elapsed()

        self.assertFalse(scene._is_reaction_window_open)
        self.assertEqual(scene._result.result, ResultEnum.TIMEOUT)
        scene._sprites.remove.assert_called_once_with([scene._correct_target])
        scene._on_inter_trial.assert_called_once_with()

    def test_response_window_elapsed_is_a_noop_if_already_closed(self) -> None:
        scene = make_scene()
        scene._is_reaction_window_open = False

        scene._on_response_window_elapsed()

        scene._sprites.remove.assert_not_called()
        scene._on_inter_trial.assert_not_called()

    def test_response_window_elapsed_does_not_override_an_existing_result(
        self,
    ) -> None:
        # If a touch was already scored (correct/background) just before
        # the window-elapsed timer fires, the window is already closed
        # and this must not clobber that result with TIMEOUT.
        scene = make_scene()
        scene._is_reaction_window_open = False
        scene._result.result = ResultEnum.CORRECT

        scene._on_response_window_elapsed()

        self.assertEqual(scene._result.result, ResultEnum.CORRECT)

    def test_play_stimulus_switches_background_and_shows_stimulus_immediately(
        self,
    ) -> None:
        scene = make_scene()
        scene._background = None
        scene._stimulus_background = "stimulus"
        scene._show_visual_stimulus = Mock()

        scene._play_stimulus()

        self.assertEqual(scene._background, "stimulus")
        scene._show_visual_stimulus.assert_called_once_with()


class RealStimulusTests(unittest.TestCase):
    def test_visual_stimulus_uses_real_marmoset_image(self) -> None:
        scene = make_scene()
        scene._init_backgrounds = Mock()
        with (
            patch(
                "stages.visual_3pt_calibration.scene"
                ".choice",
                side_effect=[M001, (1440, 540)],
            ),
            patch(
                "stages.visual_3pt_calibration.scene"
                ".ImageSprite"
            ) as image_sprite,
        ):
            scene._init_visual_stimulus(make_config())

        # target_size_cm=1.5 * pixels_per_cm=37.8 = 56.7, rounded = 57
        image_sprite.assert_called_once_with(M001, 57, (1440, 540))

    def test_fixation_point_is_at_3_quarters_screen_width(self) -> None:
        # Matches where the single target used to sit in
        # visual_touchscreen_training (flushed right), rather than
        # dead-center.
        scene = make_scene()
        scene._init_backgrounds = Mock()

        scene._init_visual_stimulus(make_config())

        self.assertEqual(scene._fixation_point.rect.center, (1440, 540))

    def test_target_size_uses_cm_to_pixel_conversion(self) -> None:
        scene = make_scene()
        scene._init_backgrounds = Mock()

        scene._init_visual_stimulus(make_config())

        # target_size_cm=1.5 * pixels_per_cm=37.8 = 56.7, rounded = 57
        self.assertEqual(scene._correct_target.rect.width, 57)
        self.assertEqual(scene._correct_target.rect.height, 57)

    def test_target_is_placed_at_one_of_the_three_triangle_positions(self) -> None:
        # fixation at (1440, 540), screen width 1920, offset_ratio 0.2
        # => offset_px = 384
        expected_positions = {
            (1440, 540 - 384),  # above
            (1440, 540 + 384),  # below
            (1440 + 384, 540),  # right
        }

        for _ in range(20):
            scene = make_scene()
            scene._init_backgrounds = Mock()

            scene._init_visual_stimulus(make_config())

            self.assertIn(scene._correct_target.rect.center, expected_positions)

    def test_all_three_triangle_positions_are_reachable(self) -> None:
        expected_positions = {
            (1440, 540 - 384),
            (1440, 540 + 384),
            (1440 + 384, 540),
        }
        observed_positions: set[tuple[int, int]] = set()

        for _ in range(200):
            scene = make_scene()
            scene._init_backgrounds = Mock()

            scene._init_visual_stimulus(make_config())

            observed_positions.add(scene._correct_target.rect.center)
            if observed_positions == expected_positions:
                break

        self.assertEqual(observed_positions, expected_positions)

    def test_target_offset_ratio_changes_triangle_size(self) -> None:
        config = Config(
            animal="test-animal",
            level=1,
            is_twitter_trial=True,
            screen_size=Screen(width=1920, height=1080),
            stored=StoredConfig(
                **{
                    **make_stored_config().model_dump(),
                    "target_offset_ratio": 0.1,
                }
            ),
        )
        expected_positions = {
            (1440, 540 - 192),
            (1440, 540 + 192),
            (1440 + 192, 540),
        }

        scene = make_scene()
        scene._init_backgrounds = Mock()

        scene._init_visual_stimulus(config)

        self.assertIn(scene._correct_target.rect.center, expected_positions)


if __name__ == "__main__":
    unittest.main()
