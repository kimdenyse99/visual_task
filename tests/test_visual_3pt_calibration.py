# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from mxbiflow.models.animal import AnimalBaseInfo
from pydantic import ValidationError
from pygame import (
    FINGERDOWN,
    FINGERMOTION,
    FINGERUP,
    MOUSEBUTTONDOWN,
    MOUSEBUTTONUP,
    MOUSEMOTION,
    QUIT,
    Event,
)

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
    _ActiveTouch,
)


def make_stored_config() -> StoredConfig:
    return StoredConfig(
        inter_trial_duration_ms=1500,
        reward_duration_ms=1000,
        attention_duration_ms=1000,
        fixation_duration_min_ms=1000,
        fixation_duration_max_ms=2000,
        visual_delay_ms=500,
        border_width=40,
        level_up_threshold=0.8,
        level_up_trial_count_threshold=20,
        background_color=(127, 127, 127),
        fixation_point_radius_cm=0.25,
        fixation_point_inner_radius_ratio=0.40,
        target_distances_cm=2.6,
        touch_buffer_cm=1.0,
        pixels_per_cm=46.0,
        response_window_min_ms=50,
        response_window_max_ms=800,
        min_touch_duration_ms=100,
        touch_tolerance_radius_cm=0.5,
        target_touch_reward_enabled=True,
        fixation_touch_reward_duration_ms=500,
        levels={
            1: LevelConfig(
                reaction_window_duration_ms=10000,
                fixation_touch_reward_radius_cm=1000.0,
                target_size_cm=1.0,
                reward_delay_ms=500,
            ),
            2: LevelConfig(
                reaction_window_duration_ms=5000,
                fixation_touch_reward_radius_cm=10.0,
                target_size_cm=1.0,
                reward_delay_ms=500,
            ),
            3: LevelConfig(
                reaction_window_duration_ms=5000,
                fixation_touch_reward_radius_cm=3.0,
                target_size_cm=1.0,
                reward_delay_ms=300,
            ),
            4: LevelConfig(
                reaction_window_duration_ms=5000,
                fixation_touch_reward_radius_cm=0.0,
                target_size_cm=1.0,
                reward_delay_ms=300,
            ),
        },
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
    scene._fixation_point.rect.center = (1440, 540)
    scene._is_reaction_window_open = False
    scene._reaction_window_started_at_ms = None
    scene._fixation_shown_at_ms = None
    scene._phase = "blank"
    scene._active_touches = {}
    scene._control_buttons = []
    scene._mxbiflow = SimpleNamespace(
        mxbi=SimpleNamespace(screen_size=Screen(width=1920, height=1080))
    )
    scene._result = SimpleNamespace(
        animal=SimpleNamespace(level=1, trial_id=1),
        result=ResultEnum.PENDING,
        trial_start_time=0,
        trial_end_time=0,
        fixation_touch_latency_ms=None,
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
    @patch("stages.visual_3pt_calibration.scene.get_mxbiflow")
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
            mxbi=SimpleNamespace(rewarder=Mock(), timer=Mock()),
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
        self.assertEqual(scene._phase, "blank")
        self.assertEqual(scene._active_touches, {})

    def test_start_runs_trial_start(self) -> None:
        scene = make_scene()
        scene._mxbiflow = SimpleNamespace(session=Mock())
        scene._result_widget = Mock()
        scene._result_summary = Mock(return_value="summary")
        scene._on_trial_start = Mock()

        scene.start()

        scene._result_widget.set_text.assert_called_once_with("summary")
        scene._on_trial_start.assert_called_once_with()

    @patch("stages.visual_3pt_calibration.scene.time", return_value=123.0)
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
        )

        result = scene._init_result(animal, make_config(level=2))

        self.assertEqual(result.animal, animal.base_info)
        self.assertTrue(result.is_twitter_trial)
        self.assertEqual(result.trial_start_time, 123.0)
        self.assertEqual(result.trial_end_time, 123.0)
        self.assertIsNone(result.fixation_touch_latency_ms)
        self.assertIsNone(result.response_latency_ms)


class ConfigTests(unittest.TestCase):
    def test_target_size_cm_is_level_specific(self) -> None:
        self.assertEqual(make_config(level=1).target_size_cm, 1.0)
        self.assertEqual(make_config(level=3).target_size_cm, 1.0)

    def test_target_size_px_converts_cm_to_pixels(self) -> None:
        # target_size_cm=1.0 * pixels_per_cm=46.0 = 46.0
        self.assertAlmostEqual(make_config().target_size_px, 46.0)

    def test_fixation_touch_reward_radius_cm_is_level_specific(self) -> None:
        self.assertEqual(make_config(level=1).fixation_touch_reward_radius_cm, 1000.0)
        self.assertEqual(make_config(level=2).fixation_touch_reward_radius_cm, 10.0)
        self.assertEqual(make_config(level=3).fixation_touch_reward_radius_cm, 3.0)
        self.assertEqual(make_config(level=4).fixation_touch_reward_radius_cm, 0.0)

    def test_fixation_touch_reward_radius_px_converts_cm_to_pixels(self) -> None:
        # level 3: 3.0cm * 46.0px/cm = 138.0
        self.assertAlmostEqual(make_config(level=3).fixation_touch_reward_radius_px, 138.0)

    def test_reward_delay_ms_is_level_specific(self) -> None:
        self.assertEqual(make_config(level=1).reward_delay_ms, 500)
        self.assertEqual(make_config(level=3).reward_delay_ms, 300)

    def test_reward_delay_ms_field_default_is_0(self) -> None:
        self.assertEqual(LevelConfig.model_fields["reward_delay_ms"].default, 0)

    def test_reward_delay_ms_rejects_negative(self) -> None:
        with self.assertRaises(ValidationError):
            LevelConfig.model_validate(
                {
                    "reaction_window_duration_ms": 10000,
                    "target_size_cm": 1.0,
                    "fixation_touch_reward_radius_cm": 0.0,
                    "reward_delay_ms": -1,
                }
            )

    def test_target_touch_reward_enabled_defaults_to_true(self) -> None:
        self.assertTrue(make_config().target_touch_reward_enabled)
        self.assertTrue(
            StoredConfig.model_fields["target_touch_reward_enabled"].default
        )

    def test_fixation_touch_reward_duration_defaults_to_500(self) -> None:
        self.assertEqual(make_config().fixation_touch_reward_duration_ms, 500)

    def test_fixation_duration_range_is_configured(self) -> None:
        self.assertEqual(make_config().fixation_duration_min_ms, 1000)
        self.assertEqual(make_config().fixation_duration_max_ms, 2000)

    def test_fixation_duration_max_must_be_at_least_min(self) -> None:
        with self.assertRaises(ValidationError):
            StoredConfig.model_validate(
                {
                    **make_stored_config().model_dump(),
                    "fixation_duration_min_ms": 2000,
                    "fixation_duration_max_ms": 1000,
                }
            )

    def test_response_window_max_must_exceed_min(self) -> None:
        with self.assertRaises(ValidationError):
            StoredConfig.model_validate(
                {
                    **make_stored_config().model_dump(),
                    "response_window_min_ms": 800,
                    "response_window_max_ms": 800,
                }
            )

    def test_reaction_window_duration_must_exceed_response_window_max(self) -> None:
        with self.assertRaises(ValidationError):
            StoredConfig.model_validate(
                {
                    **make_stored_config().model_dump(),
                    "response_window_max_ms": 800,
                    "levels": {
                        1: {
                            "reaction_window_duration_ms": 500,
                            "target_size_cm": 1.0,
                            "fixation_touch_reward_radius_cm": 1000.0,
                        },
                    },
                }
            )

    def test_touch_buffer_px_converts_cm_to_pixels(self) -> None:
        # touch_buffer_cm=1.0 * pixels_per_cm=46.0 = 46.0
        self.assertAlmostEqual(make_config().touch_buffer_px, 46.0)

    def test_touch_tolerance_radius_px_converts_cm_to_pixels(self) -> None:
        # touch_tolerance_radius_cm=0.5 * pixels_per_cm=46.0 = 23.0
        self.assertAlmostEqual(make_config().touch_tolerance_radius_px, 23.0)

    def test_target_distances_px_converts_cm_to_pixels(self) -> None:
        # target_distances_cm=2.6 * pixels_per_cm=46.0 = 119.6
        self.assertAlmostEqual(make_config().target_distances_px, 119.6)

    def test_fixation_point_radius_px_converts_cm_to_pixels(self) -> None:
        # fixation_point_radius_cm=0.25 * pixels_per_cm=46.0 = 11.5
        self.assertAlmostEqual(make_config().fixation_point_radius_px, 11.5)

    def test_runtime_properties_are_not_serialized(self) -> None:
        dumped = make_config().model_dump()

        self.assertNotIn("stored", dumped)
        self.assertNotIn("reward_duration_ms", dumped)
        self.assertEqual(make_config().reward_duration_ms, 1000)

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
                    root / "animal-session" / "visual_3pt_calibration" / "config.json"
                )
                self.assertTrue(snapshot.exists())
                self.assertEqual(snapshot.read_text(encoding="utf-8"), source_text)
                second = scene._init_config(animal, mxbiflow)

        self.assertEqual(first.animal, "test-animal")
        self.assertEqual(first.level, 1)
        self.assertEqual(first.target_size_cm, 1.0)
        self.assertTrue(first.is_twitter_trial)
        self.assertEqual(second.animal, "test-animal")


class ActiveTouchTests(unittest.TestCase):
    def test_initializes_at_the_given_position_and_time(self) -> None:
        touch = _ActiveTouch((10, 20), 1000.0)

        self.assertEqual(touch.start_pos, (10, 20))
        self.assertEqual(touch.start_ms, 1000.0)
        self.assertEqual(touch.reference_pos, (10, 20))
        self.assertEqual(touch.reference_ms, 1000.0)
        self.assertEqual(touch.last_pos, (10, 20))
        self.assertFalse(touch.fired)


class FixationPointTests(unittest.TestCase):
    def test_fixation_point_is_a_sprite_centered_at_given_point(self) -> None:
        point = FixationPoint((100, 200), radius=20, inner_radius_ratio=0.35)

        self.assertEqual(point.rect.center, (100, 200))
        self.assertEqual(point.image.get_width(), 40)
        self.assertEqual(point.image.get_height(), 40)

    def test_fixation_point_draws_green_outer_and_black_inner_circle(self) -> None:
        point = FixationPoint((50, 50), radius=20, inner_radius_ratio=0.5)

        center_pixel = point.image.get_at((20, 20))
        self.assertEqual(center_pixel[:3], (0, 0, 0))

        edge_pixel = point.image.get_at((20, 2))
        self.assertEqual(edge_pixel[:3], (0, 200, 0))


class TrialStartTests(unittest.TestCase):
    def test_trial_start_opens_fixation_phase_with_no_scheduled_timer(self) -> None:
        # The fixation phase has no timer-based expiry of its own - only
        # a qualifying touch ends it (see _on_fixation_touched).
        scene = make_scene()

        scene._on_trial_start()

        self.assertEqual(scene._phase, "fixation")
        scene._sprites.add.assert_called_once_with(scene._fixation_point)
        scene._timer.after.assert_not_called()

    def test_trial_start_records_fixation_shown_time(self) -> None:
        scene = make_scene()

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=2.0,
        ):
            scene._on_trial_start()

        self.assertEqual(scene._fixation_shown_at_ms, 2000)


class InputModalityTests(unittest.TestCase):
    """Covers press/motion/release across mouse and finger, and
    multi-touch tracking."""

    def test_mouse_press_registers_an_active_touch(self) -> None:
        scene = make_scene()
        scene._handle_touch_event = Mock()

        scene.handle_event(Event(MOUSEBUTTONDOWN, button=1, pos=(10, 20)))

        self.assertIn(-1, scene._active_touches)
        self.assertEqual(scene._active_touches[-1].start_pos, (10, 20))

    def test_mouse_click_consumed_by_control_button_does_not_register(self) -> None:
        scene = make_scene()
        button = Mock()
        button.handle_event.return_value = True
        scene._control_buttons = [button]

        scene.handle_event(Event(MOUSEBUTTONDOWN, button=1, pos=(10, 20)))

        self.assertEqual(scene._active_touches, {})

    def test_mouse_release_removes_the_active_touch_and_logs_it(self) -> None:
        scene = make_scene()
        scene._handle_touch_event = Mock()

        scene.handle_event(Event(MOUSEBUTTONDOWN, button=1, pos=(10, 20)))
        scene.handle_event(Event(MOUSEBUTTONUP, button=1, pos=(10, 20)))

        self.assertEqual(scene._active_touches, {})
        self.assertEqual(len(scene._result.touch_event), 1)
        self.assertEqual(scene._result.touch_event[0].x, 10)
        self.assertEqual(scene._result.touch_event[0].y, 20)

    def test_release_without_matching_press_is_a_no_op(self) -> None:
        scene = make_scene()

        scene.handle_event(Event(MOUSEBUTTONUP, button=1, pos=(10, 20)))

        self.assertEqual(len(scene._result.touch_event), 0)

    def test_mouse_motion_only_tracked_while_a_touch_is_active(self) -> None:
        scene = make_scene()
        scene._handle_touch_event = Mock()
        scene._phase = "fixation"

        scene.handle_event(Event(MOUSEMOTION, pos=(50, 50)))
        self.assertEqual(scene._active_touches, {})

        scene.handle_event(Event(MOUSEBUTTONDOWN, button=1, pos=(10, 20)))
        scene.handle_event(Event(MOUSEMOTION, pos=(50, 50)))
        self.assertEqual(scene._active_touches[-1].last_pos, (50, 50))

    def test_finger_press_is_converted_to_screen_pixels(self) -> None:
        scene = make_scene()
        scene._handle_touch_event = Mock()
        scene._mxbiflow = SimpleNamespace(
            mxbi=SimpleNamespace(screen_size=Screen(width=1000, height=500))
        )

        scene.handle_event(Event(FINGERDOWN, finger_id=7, x=0.5, y=0.25))

        self.assertIn(7, scene._active_touches)
        self.assertEqual(scene._active_touches[7].start_pos, (500, 125))

    def test_finger_release_is_converted_to_screen_pixels(self) -> None:
        scene = make_scene()
        scene._handle_touch_event = Mock()
        scene._mxbiflow = SimpleNamespace(
            mxbi=SimpleNamespace(screen_size=Screen(width=1000, height=500))
        )

        scene.handle_event(Event(FINGERDOWN, finger_id=7, x=0.5, y=0.25))
        scene.handle_event(Event(FINGERUP, finger_id=7, x=0.5, y=0.25))

        self.assertEqual(scene._active_touches, {})
        self.assertEqual(scene._result.touch_event[0].x, 500)
        self.assertEqual(scene._result.touch_event[0].y, 125)

    def test_finger_motion_only_tracked_for_a_matching_active_finger(self) -> None:
        scene = make_scene()
        scene._handle_touch_event = Mock()
        scene._phase = "fixation"
        scene._mxbiflow = SimpleNamespace(
            mxbi=SimpleNamespace(screen_size=Screen(width=1000, height=500))
        )

        scene.handle_event(Event(FINGERMOTION, finger_id=9, x=0.1, y=0.1))
        self.assertEqual(scene._active_touches, {})

        scene.handle_event(Event(FINGERDOWN, finger_id=7, x=0.5, y=0.25))
        scene.handle_event(Event(FINGERMOTION, finger_id=7, x=0.6, y=0.3))
        self.assertEqual(scene._active_touches[7].last_pos, (600, 150))

    def test_multiple_simultaneous_touches_are_tracked_independently(self) -> None:
        scene = make_scene()
        scene._handle_touch_event = Mock()
        scene._mxbiflow = SimpleNamespace(
            mxbi=SimpleNamespace(screen_size=Screen(width=1000, height=500))
        )

        scene.handle_event(Event(MOUSEBUTTONDOWN, button=1, pos=(10, 20)))
        scene.handle_event(Event(FINGERDOWN, finger_id=3, x=0.2, y=0.2))
        scene.handle_event(Event(FINGERDOWN, finger_id=4, x=0.8, y=0.8))

        self.assertEqual(set(scene._active_touches.keys()), {-1, 3, 4})

        scene.handle_event(Event(MOUSEBUTTONUP, button=1, pos=(10, 20)))
        self.assertEqual(set(scene._active_touches.keys()), {3, 4})

    # TODO(gaze): re-enable once GAZEBUTTONDOWN/UP/MOTION are allocated
    # and a gaze driver actually posts them (see scene.py).


class TouchStabilityTests(unittest.TestCase):
    """Covers _exceeds_tolerance, _evaluate_touch, and
    _check_pending_touches - the per-frame re-evaluation mechanism."""

    def test_touch_within_tolerance_radius_does_not_exceed(self) -> None:
        scene = make_scene()

        # touch_tolerance_radius_px = 0.5cm * 46.0px/cm = 23px
        self.assertFalse(scene._exceeds_tolerance((100, 100), (110, 100)))

    def test_touch_beyond_tolerance_radius_exceeds(self) -> None:
        scene = make_scene()

        self.assertTrue(scene._exceeds_tolerance((100, 100), (200, 100)))

    def test_touch_at_exactly_the_tolerance_boundary_does_not_exceed(self) -> None:
        scene = make_scene()

        # Exactly 23px away (0.5cm * 46.0px/cm)
        self.assertFalse(scene._exceeds_tolerance((100, 100), (123, 100)))

    def test_evaluate_touch_does_not_refire_once_fired(self) -> None:
        scene = make_scene()
        scene._phase = "fixation"
        scene._handle_touch_event = Mock()
        touch = _ActiveTouch((10, 20), 1000.0)
        touch.fired = True

        scene._evaluate_touch(-1, touch)

        scene._handle_touch_event.assert_not_called()

    def test_evaluate_touch_does_not_accumulate_time_while_blank(self) -> None:
        scene = make_scene()
        scene._phase = "blank"
        scene._handle_touch_event = Mock()
        touch = _ActiveTouch((10, 20), 1000.0)

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=5.0,
        ):
            scene._evaluate_touch(-1, touch)

        self.assertFalse(touch.fired)
        scene._handle_touch_event.assert_not_called()
        self.assertEqual(touch.reference_ms, 5000)

    def test_evaluate_touch_requires_min_duration(self) -> None:
        scene = make_scene()
        scene._phase = "fixation"
        scene._handle_touch_event = Mock()
        touch = _ActiveTouch((10, 20), 1000.0)

        # min_touch_duration_ms=100 in the test config; only 50ms elapsed
        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=1.05,
        ):
            scene._evaluate_touch(-1, touch)

        self.assertFalse(touch.fired)
        scene._handle_touch_event.assert_not_called()

    def test_evaluate_touch_fires_once_min_duration_elapses(self) -> None:
        scene = make_scene()
        scene._phase = "fixation"
        scene._handle_touch_event = Mock()
        touch = _ActiveTouch((10, 20), 1000.0)

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=1.1,
        ):
            scene._evaluate_touch(-1, touch)

        self.assertTrue(touch.fired)
        scene._handle_touch_event.assert_called_once_with((10, 20))

    def test_motion_within_tolerance_keeps_the_same_reference(self) -> None:
        scene = make_scene()
        scene._phase = "fixation"
        scene._handle_touch_event = Mock()
        scene._active_touches[-1] = _ActiveTouch((100, 100), 1000.0)

        scene._on_motion(-1, (110, 100))

        self.assertEqual(scene._active_touches[-1].reference_pos, (100, 100))

    def test_motion_beyond_tolerance_re_anchors_the_reference(self) -> None:
        scene = make_scene()
        scene._phase = "fixation"
        scene._handle_touch_event = Mock()
        scene._active_touches[-1] = _ActiveTouch((100, 100), 1000.0)

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=2.0,
        ):
            scene._on_motion(-1, (300, 100))

        active = scene._active_touches[-1]
        self.assertEqual(active.reference_pos, (300, 100))
        self.assertEqual(active.reference_ms, 2000)

    def test_motion_on_already_fired_touch_does_not_refire_or_move_reference(
        self,
    ) -> None:
        scene = make_scene()
        scene._phase = "fixation"
        scene._handle_touch_event = Mock()
        touch = _ActiveTouch((100, 100), 1000.0)
        touch.fired = True
        scene._active_touches[-1] = touch

        scene._on_motion(-1, (300, 100))

        self.assertEqual(touch.reference_pos, (100, 100))
        scene._handle_touch_event.assert_not_called()

    def test_check_pending_touches_evaluates_every_active_touch(self) -> None:
        scene = make_scene()
        scene._phase = "fixation"
        scene._handle_touch_event = Mock()
        scene._active_touches[-1] = _ActiveTouch((10, 20), 0.0)
        scene._active_touches[3] = _ActiveTouch((30, 40), 0.0)

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=1.0,
        ):
            scene._check_pending_touches()

        self.assertTrue(scene._active_touches[-1].fired)
        self.assertTrue(scene._active_touches[3].fired)
        self.assertEqual(scene._handle_touch_event.call_count, 2)


class HasActiveStimulusTests(unittest.TestCase):
    def test_fixation_phase_always_has_an_active_stimulus(self) -> None:
        scene = make_scene()
        scene._phase = "fixation"

        self.assertTrue(scene._has_active_stimulus())

    def test_test_phase_depends_on_reaction_window_open(self) -> None:
        scene = make_scene()
        scene._phase = "test"

        scene._is_reaction_window_open = True
        self.assertTrue(scene._has_active_stimulus())

        scene._is_reaction_window_open = False
        self.assertFalse(scene._has_active_stimulus())

    def test_blank_phase_never_has_an_active_stimulus(self) -> None:
        scene = make_scene()
        scene._phase = "blank"

        self.assertFalse(scene._has_active_stimulus())


class FixationTouchTests(unittest.TestCase):
    """Covers _handle_touch_event's fixation-phase branch: a radius
    check around the fixation point (unlike visual_touchscreen_training,
    where any screen position qualifies). The qualifying radius is
    measured outward from the edge of the visible green circle
    (fixation_point_radius_px), not from its center point - so the
    actual cutoff distance from center is
    fixation_point_radius_px + fixation_touch_reward_radius_px."""

    def test_touch_within_radius_triggers_on_fixation_touched(self) -> None:
        scene = make_scene()
        scene._phase = "fixation"
        scene._on_fixation_touched = Mock()
        scene._config = SimpleNamespace(
            fixation_point_radius_px=10.0,
            fixation_touch_reward_radius_px=100.0,
        )

        # qualifying radius = 10 + 100 = 110px; 50px away qualifies
        scene._handle_touch_event((1440 + 50, 540))

        scene._on_fixation_touched.assert_called_once_with()

    def test_touch_outside_radius_does_not_trigger(self) -> None:
        scene = make_scene()
        scene._phase = "fixation"
        scene._on_fixation_touched = Mock()
        scene._config = SimpleNamespace(
            fixation_point_radius_px=10.0,
            fixation_touch_reward_radius_px=100.0,
        )

        # qualifying radius = 10 + 100 = 110px; 500px away does not
        scene._handle_touch_event((1440 + 500, 540))

        scene._on_fixation_touched.assert_not_called()

    def test_touch_within_the_visible_circle_qualifies_at_tolerance_0(self) -> None:
        # With fixation_touch_reward_radius_px=0, the qualifying radius
        # collapses to just fixation_point_radius_px - i.e. "anywhere
        # strictly within the visible green circle", which is still a
        # real, achievable target for real input (unlike a single exact
        # pixel at the center).
        scene = make_scene()
        scene._phase = "fixation"
        scene._on_fixation_touched = Mock()
        scene._config = SimpleNamespace(
            fixation_point_radius_px=12.0,
            fixation_touch_reward_radius_px=0.0,
        )

        scene._handle_touch_event((1440 + 8, 540))

        scene._on_fixation_touched.assert_called_once_with()

    def test_touch_exactly_at_fixation_point_center_qualifies_at_tolerance_0(
        self,
    ) -> None:
        scene = make_scene()
        scene._phase = "fixation"
        scene._on_fixation_touched = Mock()
        scene._config = SimpleNamespace(
            fixation_point_radius_px=12.0,
            fixation_touch_reward_radius_px=0.0,
        )

        scene._handle_touch_event((1440, 540))

        scene._on_fixation_touched.assert_called_once_with()

    def test_touch_beyond_the_visible_circle_does_not_qualify_at_tolerance_0(
        self,
    ) -> None:
        scene = make_scene()
        scene._phase = "fixation"
        scene._on_fixation_touched = Mock()
        scene._config = SimpleNamespace(
            fixation_point_radius_px=12.0,
            fixation_touch_reward_radius_px=0.0,
        )

        scene._handle_touch_event((1440 + 50, 540))

        scene._on_fixation_touched.assert_not_called()

    def test_fixation_touched_records_latency_and_gives_reward(self) -> None:
        scene = make_scene()
        scene._fixation_shown_at_ms = 1000.0
        scene._give_reward = Mock()

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=1.3,
        ):
            scene._on_fixation_touched()

        self.assertEqual(scene._result.fixation_touch_latency_ms, 300)
        scene._give_reward.assert_called_once_with(500)

    def test_fixation_touched_skips_reward_when_duration_is_0(self) -> None:
        scene = make_scene()
        scene._fixation_shown_at_ms = 1000.0
        scene._config = SimpleNamespace(
            fixation_touch_reward_duration_ms=0,
            fixation_duration_min_ms=1000,
            fixation_duration_max_ms=2000,
        )
        scene._give_reward = Mock()

        with patch(
            "stages.visual_3pt_calibration.scene.randint",
            return_value=1500,
        ):
            scene._on_fixation_touched()

        scene._give_reward.assert_not_called()

    def test_fixation_touched_sets_phase_to_blank_and_schedules_stimulus(
        self,
    ) -> None:
        scene = make_scene()
        scene._fixation_shown_at_ms = 1000.0
        scene._give_reward = Mock()

        with patch(
            "stages.visual_3pt_calibration.scene.randint",
            return_value=1500,
        ) as randint_mock:
            scene._on_fixation_touched()

        self.assertEqual(scene._phase, "blank")
        randint_mock.assert_called_once_with(1000, 2000)
        scene._timer.after.assert_called_once_with(1500, scene._play_stimulus)


class StimulusTouchTests(unittest.TestCase):
    """Covers _handle_touch_event's "test" phase branch: the
    response-window timing classification (Correct / mistimed-Timeout /
    off-target-Incorrect)."""

    def test_touch_outside_reaction_window_is_ignored(self) -> None:
        scene = make_scene()
        scene._phase = "test"
        scene._is_reaction_window_open = False
        scene._is_target_touch = Mock()

        scene._handle_touch_event((10, 20))

        scene._is_target_touch.assert_not_called()
        scene._sprites.remove.assert_not_called()
        scene._on_inter_trial.assert_not_called()

    def test_correct_on_target_within_response_window(self) -> None:
        scene = make_scene()
        scene._phase = "test"
        scene._is_reaction_window_open = True
        scene._reaction_window_started_at_ms = 1000
        scene._is_target_touch = Mock(return_value=True)
        scene._on_correct = Mock()

        # latency = 1200 - 1000 = 200ms, within [50, 800]
        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=1.2,
        ):
            scene._handle_touch_event((10, 20))

        self.assertFalse(scene._is_reaction_window_open)
        scene._sprites.remove.assert_not_called()
        scene._on_correct.assert_called_once_with()
        self.assertEqual(scene._result.response_latency_ms, 200)

    def test_correct_at_response_window_boundaries(self) -> None:
        for latency_s, label in ((1.05, "min boundary"), (1.8, "max boundary")):
            with self.subTest(label):
                scene = make_scene()
                scene._phase = "test"
                scene._is_reaction_window_open = True
                scene._reaction_window_started_at_ms = 1000
                scene._is_target_touch = Mock(return_value=True)
                scene._on_correct = Mock()

                with patch(
                    "stages.visual_3pt_calibration.scene.monotonic",
                    return_value=latency_s,
                ):
                    scene._handle_touch_event((10, 20))

                scene._on_correct.assert_called_once_with()

    def test_timeout_on_target_but_latency_after_max(self) -> None:
        scene = make_scene()
        scene._phase = "test"
        scene._is_reaction_window_open = True
        scene._reaction_window_started_at_ms = 1000
        scene._is_target_touch = Mock(return_value=True)
        scene._on_timeout = Mock()
        scene._on_correct = Mock()

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=1.9,
        ):
            scene._handle_touch_event((10, 20))

        scene._sprites.remove.assert_called_once_with([scene._correct_target])
        scene._on_timeout.assert_called_once_with()
        scene._on_correct.assert_not_called()

    def test_timeout_on_target_but_latency_before_min(self) -> None:
        scene = make_scene()
        scene._phase = "test"
        scene._is_reaction_window_open = True
        scene._reaction_window_started_at_ms = 1000
        scene._is_target_touch = Mock(return_value=True)
        scene._on_timeout = Mock()

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=1.02,
        ):
            scene._handle_touch_event((10, 20))

        scene._on_timeout.assert_called_once_with()

    def test_timeout_does_not_record_response_latency(self) -> None:
        scene = make_scene()
        scene._phase = "test"
        scene._is_reaction_window_open = True
        scene._reaction_window_started_at_ms = 1000
        scene._is_target_touch = Mock(return_value=True)

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=1.9,
        ):
            scene._handle_touch_event((10, 20))

        self.assertIsNone(scene._result.response_latency_ms)

    def test_incorrect_off_target_removes_target_immediately(self) -> None:
        scene = make_scene()
        scene._phase = "test"
        scene._is_reaction_window_open = True
        scene._reaction_window_started_at_ms = 1000
        scene._is_target_touch = Mock(return_value=False)
        scene._on_incorrect = Mock()

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=1.2,
        ):
            scene._handle_touch_event((10, 20))

        scene._sprites.remove.assert_called_once_with([scene._correct_target])
        scene._on_incorrect.assert_called_once_with()

    def test_correct_touch_does_not_remove_target_immediately(self) -> None:
        # Unlike INCORRECT/TIMEOUT, a CORRECT touch keeps the marmoset
        # visible - see _on_correct/_deliver_reward_and_flash/
        # _end_correct_trial for when it's actually removed.
        scene = make_scene()
        scene._phase = "test"
        scene._is_reaction_window_open = True
        scene._reaction_window_started_at_ms = 1000
        scene._is_target_touch = Mock(return_value=True)
        scene._on_correct = Mock()

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=1.2,
        ):
            scene._handle_touch_event((10, 20))

        scene._sprites.remove.assert_not_called()

    def test_target_touch_is_classified_using_touch_buffer(self) -> None:
        scene = make_scene()
        rect = Mock()
        inflated = SimpleNamespace(collidepoint=Mock(return_value=True))
        rect.inflate.return_value = inflated
        scene._correct_target = SimpleNamespace(rect=rect)

        self.assertTrue(scene._is_target_touch((10, 20)))

        # touch_buffer_cm=1.0 * pixels_per_cm=46.0 = 46px
        rect.inflate.assert_called_once_with(46, 46)

    def test_non_target_touch_is_classified_as_false(self) -> None:
        scene = make_scene()
        rect = Mock()
        rect.inflate.return_value = SimpleNamespace(
            collidepoint=Mock(return_value=False)
        )
        scene._correct_target = SimpleNamespace(rect=rect)

        self.assertFalse(scene._is_target_touch((10, 20)))

    def test_missing_target_rect_is_classified_as_false(self) -> None:
        scene = make_scene()
        scene._correct_target = SimpleNamespace(rect=None)

        self.assertFalse(scene._is_target_touch((10, 20)))


class CorrectTouchSequenceTests(unittest.TestCase):
    """Covers the full post-correct-touch sequence: the marmoset stays
    visible through reward_delay_ms, then through reward_duration_ms
    while the reward is given, then is removed and inter-trial begins."""

    def test_on_correct_sets_result_and_schedules_the_delay(self) -> None:
        scene = make_scene()
        scene._config = SimpleNamespace(reward_delay_ms=500)
        scene._give_reward = Mock()

        scene._on_correct()

        self.assertEqual(scene._result.result, ResultEnum.CORRECT)
        scene._give_reward.assert_not_called()
        scene._on_inter_trial.assert_not_called()
        scene._timer.after.assert_called_once_with(
            500, scene._deliver_reward_and_flash
        )

    def test_zero_reward_delay_delivers_reward_immediately(self) -> None:
        scene = make_scene()
        scene._config = SimpleNamespace(
            reward_delay_ms=0,
            reward_duration_ms=1000,
            target_touch_reward_enabled=True,
        )
        scene._give_reward = Mock()

        scene._on_correct()

        scene._give_reward.assert_called_once_with()
        scene._timer.after.assert_called_once_with(1000, scene._end_correct_trial)

    def test_deliver_reward_and_flash_gives_reward_when_enabled(self) -> None:
        scene = make_scene()
        scene._config = SimpleNamespace(
            reward_duration_ms=1000, target_touch_reward_enabled=True
        )
        scene._give_reward = Mock()

        scene._deliver_reward_and_flash()

        scene._give_reward.assert_called_once_with()
        scene._timer.after.assert_called_once_with(1000, scene._end_correct_trial)

    def test_deliver_reward_and_flash_still_flashes_when_reward_disabled(self) -> None:
        # Even with the reward toggle off, the marmoset still flashes
        # for reward_duration_ms - only the actual reward is withheld.
        scene = make_scene()
        scene._config = SimpleNamespace(
            reward_duration_ms=1000, target_touch_reward_enabled=False
        )
        scene._give_reward = Mock()

        scene._deliver_reward_and_flash()

        scene._give_reward.assert_not_called()
        scene._timer.after.assert_called_once_with(1000, scene._end_correct_trial)

    def test_end_correct_trial_removes_target_and_proceeds(self) -> None:
        scene = make_scene()

        scene._end_correct_trial()

        scene._sprites.remove.assert_called_once_with([scene._correct_target])
        scene._on_inter_trial.assert_called_once_with()

    def test_marmoset_stays_visible_for_delay_plus_reward_duration(self) -> None:
        scene = make_scene()
        scene._config = SimpleNamespace(
            reward_delay_ms=500,
            reward_duration_ms=1000,
            target_touch_reward_enabled=True,
        )
        scene._give_reward = Mock()
        scheduled: list[tuple[int, Any]] = []
        scene._timer.after.side_effect = lambda ms, cb: scheduled.append((ms, cb))

        scene._on_correct()
        self.assertEqual(scheduled, [(500, scene._deliver_reward_and_flash)])
        scene._sprites.remove.assert_not_called()

        scheduled.pop()[1]()
        scene._give_reward.assert_called_once_with()
        self.assertEqual(scheduled, [(1000, scene._end_correct_trial)])
        scene._sprites.remove.assert_not_called()

        scheduled.pop()[1]()
        scene._sprites.remove.assert_called_once_with([scene._correct_target])
        scene._on_inter_trial.assert_called_once_with()


class ReactionWindowElapsedTests(unittest.TestCase):
    def test_show_visual_stimulus_schedules_unconditional_hard_cutoff(self) -> None:
        scene = make_scene()
        scene._config = SimpleNamespace(reaction_window_duration_ms=5000)

        scene._show_visual_stimulus()

        self.assertEqual(scene._phase, "test")
        self.assertTrue(scene._is_reaction_window_open)
        scene._sprites.remove.assert_called_once_with(scene._fixation_point)
        scene._sprites.add.assert_called_once_with([scene._correct_target])
        scene._timer.after.assert_called_once_with(
            5000, scene._on_response_window_elapsed
        )

    def test_show_visual_stimulus_re_anchors_active_touches(self) -> None:
        scene = make_scene()
        scene._config = SimpleNamespace(reaction_window_duration_ms=5000)
        touch = _ActiveTouch((10, 20), 0.0)
        touch.fired = True
        touch.last_pos = (99, 99)
        scene._active_touches[-1] = touch

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=3.0,
        ):
            scene._show_visual_stimulus()

        self.assertFalse(touch.fired)
        self.assertEqual(touch.reference_pos, (99, 99))
        self.assertEqual(touch.reference_ms, 3000)

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


class CrossTrialHandoffTests(unittest.TestCase):
    """Covers the module-level _pending_touches stash/claim mechanism."""

    def test_stash_pending_touches_stores_active_touches(self) -> None:
        import stages.visual_3pt_calibration.scene as scene_module

        scene = make_scene()
        scene._active_touches = {-1: _ActiveTouch((10, 20), 0.0)}

        scene_module._pending_touches = None
        scene._stash_pending_touches()

        self.assertEqual(scene_module._pending_touches, scene._active_touches)
        scene_module._pending_touches = None

    def test_stash_pending_touches_stores_none_when_nothing_active(self) -> None:
        import stages.visual_3pt_calibration.scene as scene_module

        scene = make_scene()
        scene._active_touches = {}

        scene_module._pending_touches = None
        scene._stash_pending_touches()

        self.assertIsNone(scene_module._pending_touches)

    def test_clear_pending_touches_always_sets_none(self) -> None:
        import stages.visual_3pt_calibration.scene as scene_module

        scene = make_scene()
        scene_module._pending_touches = {-1: _ActiveTouch((10, 20), 0.0)}

        scene._clear_pending_touches()

        self.assertIsNone(scene_module._pending_touches)

    def test_quit_stashes_touches_when_trial_ended_normally(self) -> None:
        import stages.visual_3pt_calibration.scene as scene_module

        scene = make_scene()
        scene._settlement = Mock(return_value=False)
        scene._print_result = Mock()
        scene._data_logger = Mock()
        scene._timer = Mock()
        scene._running = True
        scene._result.result = ResultEnum.CORRECT
        scene._active_touches = {-1: _ActiveTouch((10, 20), 0.0)}

        scene_module._pending_touches = None
        scene.quit()

        self.assertEqual(scene_module._pending_touches, scene._active_touches)
        scene_module._pending_touches = None

    def test_quit_clears_touches_when_trial_was_force_quit(self) -> None:
        import stages.visual_3pt_calibration.scene as scene_module

        scene = make_scene()
        scene._settlement = Mock(return_value=False)
        scene._print_result = Mock()
        scene._data_logger = Mock()
        scene._timer = Mock()
        scene._running = True
        scene._result.result = ResultEnum.PENDING
        scene._active_touches = {-1: _ActiveTouch((10, 20), 0.0)}

        scene_module._pending_touches = {-1: _ActiveTouch((99, 99), 0.0)}
        scene.quit()

        self.assertIsNone(scene_module._pending_touches)
        self.assertEqual(scene._result.result, ResultEnum.CANCELLED)

    def test_claim_pending_touches_returns_empty_dict_when_none_stashed(self) -> None:
        import stages.visual_3pt_calibration.scene as scene_module

        scene = make_scene()
        scene_module._pending_touches = None

        claimed = scene._claim_pending_touches()

        self.assertEqual(claimed, {})

    def test_claim_pending_touches_resets_fired_and_re_anchors(self) -> None:
        import stages.visual_3pt_calibration.scene as scene_module

        scene = make_scene()
        touch = _ActiveTouch((10, 20), 0.0)
        touch.fired = True
        touch.last_pos = (77, 88)
        scene_module._pending_touches = {-1: touch}

        with patch(
            "stages.visual_3pt_calibration.scene.monotonic",
            return_value=4.0,
        ):
            claimed = scene._claim_pending_touches()

        self.assertIn(-1, claimed)
        self.assertFalse(claimed[-1].fired)
        self.assertEqual(claimed[-1].reference_pos, (77, 88))
        self.assertEqual(claimed[-1].reference_ms, 4000)
        self.assertIsNone(scene_module._pending_touches)


class RewardTests(unittest.TestCase):
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

    def test_give_reward_updates_trial_count(self) -> None:
        scene = make_scene()
        scene._rewarder = Mock()

        scene._give_reward()

        scene._rewarder.give_reward.assert_called_once_with(1000)
        self.assertEqual(scene._result.trial_reward_count, 1)

    def test_give_reward_with_explicit_duration(self) -> None:
        scene = make_scene()
        scene._rewarder = Mock()

        scene._give_reward(500)

        scene._rewarder.give_reward.assert_called_once_with(500)

    def test_manual_reward_updates_both_counts(self) -> None:
        scene = make_scene()
        scene._rewarder = Mock()

        scene._give_manual_reward()

        self.assertEqual(scene._result.trial_reward_count, 1)
        self.assertEqual(scene._result.trial_manual_reward_count, 1)


class SettlementTests(unittest.TestCase):
    def test_settlement_saves_current_trial_in_context(self) -> None:
        scene = make_settlement_scene(trial_id=7)

        scene._settlement()

        session = scene._mxbiflow.session
        trials = session.set_stage_context.call_args.args[1]
        self.assertIs(trials.root[7], scene._result)

    def test_timeout_is_excluded_from_correct_rate_but_counts_as_trial(self) -> None:
        past = make_past_correct(20)
        past.update(
            {
                trial_id: make_trial_result(trial_id=trial_id, result=ResultEnum.TIMEOUT)
                for trial_id in range(21, 24)
            }
        )
        scene = make_settlement_scene(
            past_trials=past, current_result=ResultEnum.CORRECT, trial_id=24
        )

        is_level_up = scene._settlement()

        self.assertTrue(is_level_up)

    def test_cancelled_trial_does_not_trigger_level_up(self) -> None:
        scene = make_settlement_scene(
            past_trials=make_past_correct(20),
            current_result=ResultEnum.CANCELLED,
            trial_id=21,
        )

        scene._settlement()

        scene._mxbiflow.session.level_up.assert_not_called()

    def test_highest_level_does_not_level_up(self) -> None:
        scene = make_settlement_scene(
            past_trials=make_past_correct(20, level=4),
            trial_id=21,
            level=4,
        )

        is_level_up = scene._settlement()

        self.assertFalse(is_level_up)
        scene._mxbiflow.session.level_up.assert_not_called()

    def test_level_up_requires_more_than_configured_trial_count(self) -> None:
        for count, expected_level_up in ((19, False), (20, True)):
            with self.subTest(count=count):
                scene = make_settlement_scene(
                    past_trials=make_past_correct(count), trial_id=count + 1
                )

                is_level_up = scene._settlement()

                self.assertEqual(is_level_up, expected_level_up)

    def test_level_up_uses_level_correct_rate(self) -> None:
        for incorrect_count, expected_level_up in ((4, True), (5, False)):
            with self.subTest(incorrect_count=incorrect_count):
                past = make_past_correct(16)
                past.update(
                    {
                        trial_id: make_trial_result(
                            trial_id=trial_id, result=ResultEnum.INCORRECT
                        )
                        for trial_id in range(17, 17 + incorrect_count)
                    }
                )
                scene = make_settlement_scene(
                    past_trials=past, trial_id=17 + incorrect_count
                )

                is_level_up = scene._settlement()

                self.assertEqual(is_level_up, expected_level_up)


class QuitFlowTests(unittest.TestCase):
    def test_quit_without_resolved_result_falls_back_to_cancelled(self) -> None:
        scene = make_scene()
        scene._settlement = Mock(return_value=False)
        scene._print_result = Mock()
        scene._data_logger = Mock()
        scene._timer = Mock()
        scene._running = True

        scene.quit()

        self.assertEqual(scene._result.result, ResultEnum.CANCELLED)
        scene._data_logger.save.assert_called_once_with({})
        scene._timer.cancel_all.assert_called_once_with()
        self.assertFalse(scene._running)

    def test_request_game_quit_posts_quit_event(self) -> None:
        scene = make_scene()

        with patch("stages.visual_3pt_calibration.scene.event.post") as post:
            scene._request_game_quit()

        post.assert_called_once()
        posted = post.call_args.args[0]
        self.assertEqual(posted.type, QUIT)


class StructureConsistencyTests(unittest.TestCase):
    def test_scene_name_is_canonical(self) -> None:
        self.assertEqual(Visual3ptCalibration.name(), "visual_3pt_calibration")

    def test_no_audio_dependencies_are_imported(self) -> None:
        module_path = Path("stages/visual_3pt_calibration/scene.py")
        source = module_path.read_text(encoding="utf-8")

        for forbidden in (
            "aplayer",
            "audioplayer",
            "LoadedWav",
            "play_wav",
            "CLICKER_PATH",
            "wav_path",
        ):
            self.assertNotIn(
                forbidden,
                source,
                msg=f"'{forbidden}' should not appear in the visual stage",
            )


class RealStimulusTests(unittest.TestCase):
    def test_visual_stimulus_uses_real_marmoset_image(self) -> None:
        scene = make_scene()
        scene._init_backgrounds = Mock()
        with (
            patch(
                "stages.visual_3pt_calibration.scene.choice",
                side_effect=[M001, (1440, 540)],
            ),
            patch(
                "stages.visual_3pt_calibration.scene.ImageSprite"
            ) as image_sprite,
        ):
            scene._init_visual_stimulus(make_config())

        # target_size_cm=1.0 * pixels_per_cm=46.0 = 46.0
        image_sprite.assert_called_once_with(M001, 46, (1440, 540))

    def test_fixation_point_is_at_3_quarters_screen_width(self) -> None:
        scene = make_scene()
        scene._init_backgrounds = Mock()

        scene._init_visual_stimulus(make_config())

        self.assertEqual(scene._fixation_point.rect.center, (1440, 540))

    def test_fixation_point_size_uses_cm_to_pixel_conversion(self) -> None:
        scene = make_scene()
        scene._init_backgrounds = Mock()

        scene._init_visual_stimulus(make_config())

        # fixation_point_radius_cm=0.25 * pixels_per_cm=46.0 = 11.5,
        # rounded to 12 -> diameter 24
        self.assertEqual(scene._fixation_point.rect.width, 24)

    def test_target_is_placed_at_one_of_the_three_triangle_positions(self) -> None:
        # fixation at (1440, 540); target_distances_cm=2.6 *
        # pixels_per_cm=46.0 = 119.6 -> offset_px = round(119.6 // 2)
        # = round(59.0) = 59 (Python's // floors before round() runs)
        # half_target = round(46.0 // 2) = round(23.0) = 23
        # fixation_radius = round(11.5) = 12
        expected_positions = {
            (1440 - 59, 540 - 59),  # left
            (1440 - 59, 540 + 59),  # right
            (1440 + 23 + 12, 540),  # above
        }

        for _ in range(20):
            scene = make_scene()
            scene._init_backgrounds = Mock()

            scene._init_visual_stimulus(make_config())

            self.assertIn(scene._correct_target.rect.center, expected_positions)

    def test_all_three_triangle_positions_are_reachable(self) -> None:
        expected_positions = {
            (1440 - 59, 540 - 59),
            (1440 - 59, 540 + 59),
            (1440 + 23 + 12, 540),
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


if __name__ == "__main__":
    unittest.main()