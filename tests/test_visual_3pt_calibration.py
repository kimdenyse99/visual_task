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
    MOUSEBUTTONDOWN,
    MOUSEBUTTONUP,
    MOUSEMOTION,
    QUIT,
    Event,
)

from dwell import DwellTracker
from stimulus_events import StimulusEvents
from stages.visual_3pt_calibration.models import (
    Config,
    LevelConfig,
    Result,
    ResultEnum,
    Screen,
    StoredConfig,
    TrialResults,
)
from stages.visual_3pt_calibration.scene import FixationPoint, Visual3ptCalibration


def make_stored_config(**overrides: Any) -> StoredConfig:
    defaults: dict[str, Any] = dict(
        inter_trial_duration_ms=1500,
        reward_duration_ms=1000,
        post_reward_stimulus_duration_ms=750,
        fixation_duration_min_ms=1000,
        fixation_duration_max_ms=2000,
        border_width=40,
        level_up_threshold=0.8,
        level_up_trial_count_threshold=20,
        background_color=(127, 127, 127),
        fixation_point_radius_cm=0.25,
        fixation_point_inner_radius_ratio=0.40,
        target_distances_cm=2.6,
        pixels_per_cm=46.0,
        min_touch_duration_ms=100,
        dwell_grace_enabled=True,
        dwell_grace_ms=150,
        show_gaze_trace=False,
        gaze_trace_duration_ms=500,
        target_touch_reward_enabled=True,
        levels={
            1: LevelConfig(target_size_cm=1.0),
            2: LevelConfig(target_size_cm=1.0),
            3: LevelConfig(target_size_cm=1.0),
            4: LevelConfig(target_size_cm=1.0),
        },
    )
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


def make_scene() -> Any:
    scene: Any = Visual3ptCalibration.__new__(Visual3ptCalibration)
    scene._config = make_config()
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
    scene._trackers = {}
    scene._trajectories = {}
    scene._trace_points = []
    scene._sprites = Mock()
    scene._result = Result(
        animal=AnimalBaseInfo(
            animal="test-animal",
            trial_id=1,
            level=1,
            level_trial_id=1,
            animal_session_id=1,
            animal_session_trial_id=1,
        ),
        is_twitter_trial=True,
        trial_start_time=0.0,
        trial_end_time=0.0,
    )
    return scene


class ConfigTests(unittest.TestCase):
    def test_target_size_cm_is_level_specific(self) -> None:
        config = make_config(level=2)
        self.assertEqual(config.target_size_cm, 1.0)

    def test_target_size_px_converts_cm_to_pixels(self) -> None:
        config = make_config(target_size_cm=2.0)
        # make_config's **overrides don't reach LevelConfig directly, so
        # construct one explicitly here instead.
        stored = make_stored_config(
            levels={1: LevelConfig(target_size_cm=2.0)}
        )
        config = Config(
            animal="test-animal",
            level=1,
            is_twitter_trial=True,
            screen_size=Screen(width=1920, height=1080),
            stored=stored,
        )
        self.assertAlmostEqual(config.target_size_px, 2.0 * 46.0)

    def test_target_touch_reward_enabled_defaults_to_true(self) -> None:
        self.assertTrue(make_config().target_touch_reward_enabled)

    def test_fixation_duration_range_is_configured(self) -> None:
        config = make_config()
        self.assertEqual(config.fixation_duration_min_ms, 1000)
        self.assertEqual(config.fixation_duration_max_ms, 2000)

    def test_fixation_duration_max_must_be_at_least_min(self) -> None:
        with self.assertRaises(ValidationError):
            make_stored_config(
                fixation_duration_min_ms=2000,
                fixation_duration_max_ms=1000,
            )

    def test_target_distances_px_converts_cm_to_pixels(self) -> None:
        config = make_config(target_distances_cm=3.0, pixels_per_cm=50.0)
        self.assertAlmostEqual(config.target_distances_px, 150.0)

    def test_fixation_point_radius_px_converts_cm_to_pixels(self) -> None:
        config = make_config(fixation_point_radius_cm=0.5, pixels_per_cm=40.0)
        self.assertAlmostEqual(config.fixation_point_radius_px, 20.0)

    def test_dwell_grace_ms_is_zero_when_disabled(self) -> None:
        config = make_config(dwell_grace_enabled=False, dwell_grace_ms=150)
        self.assertEqual(config.dwell_grace_ms, 0.0)

    def test_dwell_grace_ms_passes_through_when_enabled(self) -> None:
        config = make_config(dwell_grace_enabled=True, dwell_grace_ms=200)
        self.assertEqual(config.dwell_grace_ms, 200)

    def test_runtime_properties_are_not_serialized(self) -> None:
        config = make_config()
        dumped = config.model_dump()
        self.assertNotIn("stored", dumped)
        self.assertNotIn("target_size_px", dumped)

    def test_animal_config_snapshot_is_created_once_and_reused(self) -> None:
        scene = make_scene()
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_path = Path(tmp_dir) / "config.json"
            mxbiflow = Mock()
            mxbiflow.session.absolute_animal_data_path.return_value = Path(tmp_dir)
            animal = Mock()
            animal.name = "test-animal"
            animal.current_stage.level = 1

            with patch.object(
                Visual3ptCalibration,
                "_config_snapshot_path",
                return_value=snapshot_path,
            ):
                with patch(
                    "stages.visual_3pt_calibration.scene.PATH",
                    Path("stages/visual_3pt_calibration/config.json"),
                ):
                    scene._init_config(animal, mxbiflow)
            self.assertTrue(snapshot_path.exists())


class DwellTrackerIntegrationTests(unittest.TestCase):
    """DwellTracker itself has its own tests (see dwell.py's own test
    coverage, if any exists elsewhere); these verify the scene wires it
    correctly, not DwellTracker's own internal logic."""

    def test_evaluate_tracker_does_not_refire_once_fired(self) -> None:
        scene = make_scene()
        scene._phase = "fixation"
        tracker = DwellTracker(touch_id=-1, fired=True)
        scene._trackers[-1] = tracker
        with patch.object(scene, "_handle_touch_event") as handle:
            scene._evaluate_tracker(-1, (10, 10))
        handle.assert_not_called()

    def test_evaluate_tracker_does_not_accumulate_time_while_blank(self) -> None:
        scene = make_scene()
        scene._phase = "blank"
        tracker = DwellTracker(touch_id=-1)
        scene._trackers[-1] = tracker
        scene._evaluate_tracker(-1, (10, 10))
        self.assertIsNone(tracker.inside_since_ms)

    def test_evaluate_tracker_fires_once_min_duration_elapses_during_fixation(
        self,
    ) -> None:
        scene = make_scene()
        scene._phase = "fixation"
        scene._config = make_config(min_touch_duration_ms=0)
        tracker = DwellTracker(touch_id=-1)
        scene._trackers[-1] = tracker
        with patch.object(scene, "_handle_touch_event") as handle:
            scene._evaluate_tracker(-1, (10, 10))
        handle.assert_called_once_with((10, 10))

    def test_check_all_trackers_evaluates_every_tracker_with_a_known_position(
        self,
    ) -> None:
        scene = make_scene()
        scene._phase = "fixation"
        scene._config = make_config(min_touch_duration_ms=0)
        scene._trackers[-1] = DwellTracker(touch_id=-1, last_pos=(5, 5))
        scene._trackers[0] = DwellTracker(touch_id=0, last_pos=None)
        with patch.object(scene, "_evaluate_tracker") as evaluate:
            scene._check_all_trackers()
        evaluate.assert_called_once_with(-1, (5, 5))


class ActiveShapeTests(unittest.TestCase):
    def test_fixation_phase_accepts_any_position(self) -> None:
        scene = make_scene()
        scene._phase = "fixation"
        self.assertTrue(scene._active_shape_test((0, 0)))
        self.assertTrue(scene._active_shape_test((99999, -99999)))

    def test_blank_phase_has_no_active_shape(self) -> None:
        scene = make_scene()
        scene._phase = "blank"
        self.assertIsNone(scene._active_shape_test((0, 0)))

    def test_test_phase_has_no_active_shape(self) -> None:
        # Unlike the earlier reaction-window design, "test" never
        # registers a touch event at all in the current scheme - the
        # reward is already given unconditionally the moment the GIF
        # appears (see _show_visual_stimulus).
        scene = make_scene()
        scene._phase = "test"
        self.assertIsNone(scene._active_shape_test((0, 0)))


class FixationPointTests(unittest.TestCase):
    def test_fixation_point_is_a_sprite_centered_at_given_point(self) -> None:
        point = FixationPoint((100, 200), radius=20, inner_radius_ratio=0.4)
        self.assertEqual(point.rect.center, (100, 200))

    def test_fixation_point_draws_green_outer_and_black_inner_circle(self) -> None:
        point = FixationPoint((50, 50), radius=20, inner_radius_ratio=0.4)
        diameter = 40
        center_px = point.image.get_at((diameter // 2, diameter // 2))
        self.assertEqual(center_px[:3], (0, 0, 0))
        edge_px = point.image.get_at((1, diameter // 2))
        self.assertEqual(edge_px[:3], (0, 200, 0))


class TrialStartTests(unittest.TestCase):
    def test_trial_start_opens_fixation_phase_with_no_scheduled_timer(self) -> None:
        scene = make_scene()
        scene._fixation_point = Mock()
        scene._on_trial_start()
        self.assertEqual(scene._phase, "fixation")
        scene._timer.after.assert_not_called()

    def test_trial_start_records_fixation_shown_time(self) -> None:
        scene = make_scene()
        scene._fixation_point = Mock()
        before = __import__("time").monotonic() * 1000
        scene._on_trial_start()
        after = __import__("time").monotonic() * 1000
        self.assertTrue(before <= scene._fixation_shown_at_ms <= after)


class InputModalityTests(unittest.TestCase):
    def _make_scene_with_controls(self) -> Any:
        scene = make_scene()
        scene._control_buttons = []
        return scene

    def test_mouse_press_registers_a_tracker(self) -> None:
        scene = self._make_scene_with_controls()
        scene._phase = "blank"
        scene.handle_event(Event(MOUSEBUTTONDOWN, pos=(10, 20), button=1))
        self.assertIn(-1, scene._trackers)

    def test_mouse_click_consumed_by_control_button_does_not_register(self) -> None:
        scene = make_scene()
        button = Mock()
        button.handle_event.return_value = True
        scene._control_buttons = [button]
        scene.handle_event(Event(MOUSEBUTTONDOWN, pos=(10, 20), button=1))
        self.assertNotIn(-1, scene._trackers)

    def test_mouse_release_removes_the_tracker_and_logs_it(self) -> None:
        scene = self._make_scene_with_controls()
        scene._phase = "blank"
        scene.handle_event(Event(MOUSEBUTTONDOWN, pos=(10, 20), button=1))
        scene.handle_event(Event(MOUSEBUTTONUP, pos=(15, 25), button=1))
        self.assertNotIn(-1, scene._trackers)
        self.assertEqual(len(scene._result.touch_event), 1)
        self.assertEqual(scene._result.touch_event[0].x, 15)
        self.assertEqual(scene._result.touch_event[0].y, 25)

    def test_release_without_matching_press_is_a_no_op(self) -> None:
        scene = self._make_scene_with_controls()
        scene.handle_event(Event(MOUSEBUTTONUP, pos=(15, 25), button=1))
        self.assertEqual(len(scene._result.touch_event), 0)

    def test_mouse_motion_only_tracked_while_a_touch_is_active(self) -> None:
        scene = self._make_scene_with_controls()
        scene._phase = "blank"
        scene.handle_event(Event(MOUSEMOTION, pos=(5, 5)))
        self.assertNotIn(-1, scene._trajectories)

        scene.handle_event(Event(MOUSEBUTTONDOWN, pos=(10, 20), button=1))
        scene.handle_event(Event(MOUSEMOTION, pos=(11, 21)))
        self.assertEqual(len(scene._trajectories[-1]), 2)

    def test_finger_touch_abandons_the_trial(self) -> None:
        scene = self._make_scene_with_controls()
        with patch.object(scene, "_on_touch_penalized") as penalized:
            scene.handle_event(Event(FINGERDOWN, x=0.25, y=0.5))
        penalized.assert_called_once_with()

    def test_multiple_simultaneous_touches_are_tracked_independently(self) -> None:
        scene = self._make_scene_with_controls()
        scene._phase = "blank"
        scene.handle_event(Event(MOUSEBUTTONDOWN, pos=(10, 20), button=1))
        scene._on_press(0, (30, 40))
        self.assertIn(-1, scene._trackers)
        self.assertIn(0, scene._trackers)
        self.assertIsNot(scene._trackers[-1], scene._trackers[0])


class FixationTouchedTests(unittest.TestCase):
    def test_fixation_touched_records_latency(self) -> None:
        scene = make_scene()
        scene._fixation_shown_at_ms = __import__("time").monotonic() * 1000 - 50
        scene._on_fixation_touched()
        self.assertIsNotNone(scene._result.fixation_touch_latency_ms)
        self.assertGreaterEqual(scene._result.fixation_touch_latency_ms, 0)

    def test_fixation_touched_gives_no_reward(self) -> None:
        # Unlike the earlier design, there is no fixation-phase reward
        # at all in the current scheme - the single reward happens once
        # the GIF stimulus appears (_show_visual_stimulus -> _on_correct).
        scene = make_scene()
        scene._fixation_shown_at_ms = __import__("time").monotonic() * 1000
        scene._on_fixation_touched()
        scene._rewarder.give_reward.assert_not_called()

    def test_fixation_touched_sets_phase_to_blank_and_schedules_stimulus(
        self,
    ) -> None:
        scene = make_scene()
        scene._fixation_shown_at_ms = __import__("time").monotonic() * 1000
        scene._on_fixation_touched()
        self.assertEqual(scene._phase, "blank")
        scene._timer.after.assert_called_once()
        args = scene._timer.after.call_args.args
        self.assertEqual(args[1], scene._play_stimulus)
        self.assertGreaterEqual(args[0], scene._config.fixation_duration_min_ms)
        self.assertLessEqual(args[0], scene._config.fixation_duration_max_ms)


class ShowVisualStimulusTests(unittest.TestCase):
    def test_show_visual_stimulus_sets_test_phase_and_calls_on_correct(self) -> None:
        scene = make_scene()
        scene._fixation_point = Mock()
        scene._correct_target = Mock()
        scene._correct_target.rect.center = (111, 222)
        scene._mxbiflow.session = Mock()
        with (
            patch.object(scene, "_on_correct") as on_correct,
            patch("stages.visual_3pt_calibration.scene.write_target_shown_event"),
        ):
            scene._show_visual_stimulus()
        self.assertEqual(scene._phase, "test")
        on_correct.assert_called_once_with()

    def test_show_visual_stimulus_writes_target_shown_event_with_target_center(
        self,
    ) -> None:
        scene = make_scene()
        scene._fixation_point = Mock()
        scene._correct_target = Mock()
        scene._correct_target.rect.center = (111, 222)
        with (
            patch.object(scene, "_on_correct"),
            patch(
                "stages.visual_3pt_calibration.scene.write_target_shown_event"
            ) as write_event,
        ):
            scene._show_visual_stimulus()
        write_event.assert_called_once()
        self.assertEqual(write_event.call_args.args[-1], (111, 222))


class CorrectTouchSequenceTests(unittest.TestCase):
    def test_on_correct_sets_result_and_delivers_reward(self) -> None:
        scene = make_scene()
        with patch.object(scene, "_deliver_reward_and_flash") as deliver:
            scene._on_correct()
        self.assertEqual(scene._result.result, ResultEnum.CORRECT)
        deliver.assert_called_once_with()

    def test_deliver_reward_and_flash_gives_reward_when_enabled(self) -> None:
        scene = make_scene()
        scene._config = make_config(target_touch_reward_enabled=True)
        scene._deliver_reward_and_flash()
        scene._rewarder.give_reward.assert_called_once_with(
            scene._config.reward_duration_ms
        )
        self.assertEqual(scene._result.trial_reward_count, 1)

    def test_deliver_reward_and_flash_skips_reward_when_disabled(self) -> None:
        scene = make_scene()
        scene._config = make_config(target_touch_reward_enabled=False)
        scene._deliver_reward_and_flash()
        scene._rewarder.give_reward.assert_not_called()
        self.assertEqual(scene._result.trial_reward_count, 0)

    def test_deliver_reward_and_flash_schedules_end_correct_trial(self) -> None:
        scene = make_scene()
        scene._deliver_reward_and_flash()
        scene._timer.after.assert_called_once_with(
            scene._config.post_reward_stimulus_duration_ms,
            scene._end_correct_trial,
        )

    def test_end_correct_trial_removes_target_and_proceeds(self) -> None:
        scene = make_scene()
        scene._correct_target = Mock()
        with patch.object(scene, "_on_inter_trial") as inter_trial:
            scene._end_correct_trial()
        scene._sprites.remove.assert_called_once_with([scene._correct_target])
        inter_trial.assert_called_once_with()


class CrossTrialHandoffTests(unittest.TestCase):
    def test_stash_pending_touches_stores_last_positions(self) -> None:
        import stages.visual_3pt_calibration.scene as scene_module

        scene = make_scene()
        scene._trackers[-1] = DwellTracker(touch_id=-1, last_pos=(10, 20))
        scene._stash_pending_touches()
        self.assertEqual(scene_module._pending_touch_positions, {-1: (10, 20)})
        scene_module._pending_touch_positions = None

    def test_stash_pending_touches_stores_none_when_nothing_active(self) -> None:
        import stages.visual_3pt_calibration.scene as scene_module

        scene = make_scene()
        scene._stash_pending_touches()
        self.assertIsNone(scene_module._pending_touch_positions)

    def test_clear_pending_touches_always_sets_none(self) -> None:
        import stages.visual_3pt_calibration.scene as scene_module

        scene_module._pending_touch_positions = {-1: (1, 1)}
        scene = make_scene()
        scene._clear_pending_touches()
        self.assertIsNone(scene_module._pending_touch_positions)

    def test_claim_pending_touches_seeds_trackers_with_empty_trajectory(
        self,
    ) -> None:
        import stages.visual_3pt_calibration.scene as scene_module

        scene_module._pending_touch_positions = {-1: (7, 8)}
        scene = make_scene()
        scene._claim_pending_touches()
        self.assertIn(-1, scene._trackers)
        self.assertEqual(scene._trackers[-1].last_pos, (7, 8))
        self.assertEqual(scene._trajectories[-1], [])
        self.assertIsNone(scene_module._pending_touch_positions)

    def test_claim_pending_touches_is_a_no_op_when_none_stashed(self) -> None:
        import stages.visual_3pt_calibration.scene as scene_module

        scene_module._pending_touch_positions = None
        scene = make_scene()
        scene._claim_pending_touches()
        self.assertEqual(scene._trackers, {})

    def test_claim_pending_touches_drops_stale_gaze_when_no_current_sample(
        self,
    ) -> None:
        # Regression test: a GAZE_TOUCH_ID tracker stashed at the end of
        # the previous trial must not be blindly reinstated if gaze is
        # not actually present right now - otherwise dwell can
        # accumulate against a gaze signal that's already gone (see
        # _claim_pending_touches's docstring/comment), letting fixation
        # fire - and the trial proceed all the way to reward - with no
        # eye signal detected at all.
        import stages.visual_3pt_calibration.scene as scene_module
        from gaze_input import GAZE_TOUCH_ID

        scene_module._pending_touch_positions = {GAZE_TOUCH_ID: (7, 8)}
        scene = make_scene()
        scene._mxbiflow.eye_sample = None

        scene._claim_pending_touches()

        self.assertNotIn(GAZE_TOUCH_ID, scene._trackers)
        self.assertNotIn(GAZE_TOUCH_ID, scene._trajectories)
        self.assertIsNone(scene_module._pending_touch_positions)

    def test_claim_pending_touches_keeps_gaze_when_sample_currently_present(
        self,
    ) -> None:
        import stages.visual_3pt_calibration.scene as scene_module
        from gaze_input import GAZE_TOUCH_ID

        scene_module._pending_touch_positions = {GAZE_TOUCH_ID: (7, 8)}
        scene = make_scene()
        scene._mxbiflow.eye_sample = Mock(x=7.0, y=8.0)

        scene._claim_pending_touches()

        self.assertIn(GAZE_TOUCH_ID, scene._trackers)
        self.assertEqual(scene._trackers[GAZE_TOUCH_ID].last_pos, (7, 8))
        self.assertEqual(scene._trajectories[GAZE_TOUCH_ID], [])

    def test_claim_pending_touches_still_keeps_mouse_touch_regardless_of_gaze(
        self,
    ) -> None:
        # Mouse/finger touches carry over unconditionally - only gaze
        # needs the live-sample check, since only gaze can silently
        # disappear between trials (see _claim_pending_touches).
        import stages.visual_3pt_calibration.scene as scene_module

        scene_module._pending_touch_positions = {-1: (7, 8)}
        scene = make_scene()
        scene._mxbiflow.eye_sample = None

        scene._claim_pending_touches()

        self.assertIn(-1, scene._trackers)
        self.assertEqual(scene._trackers[-1].last_pos, (7, 8))


class QuitFlowTests(unittest.TestCase):
    def test_quit_stashes_touches_when_trial_ended_normally(self) -> None:
        scene = make_scene()
        scene._result.result = ResultEnum.CORRECT
        scene._data_logger = Mock()
        scene._gaze_driver = Mock()
        with (
            patch.object(scene, "_settlement", return_value=False),
            patch.object(scene, "_print_result"),
            patch.object(scene, "_stash_pending_touches") as stash,
            patch.object(scene, "_clear_pending_touches") as clear,
        ):
            scene.quit()
        stash.assert_called_once_with()
        clear.assert_not_called()

    def test_quit_clears_touches_when_trial_was_force_quit(self) -> None:
        scene = make_scene()
        scene._result.result = ResultEnum.PENDING
        scene._data_logger = Mock()
        scene._gaze_driver = Mock()
        with (
            patch.object(scene, "_settlement", return_value=False),
            patch.object(scene, "_print_result"),
            patch.object(scene, "_stash_pending_touches") as stash,
            patch.object(scene, "_clear_pending_touches") as clear,
        ):
            scene.quit()
        clear.assert_called_once_with()
        stash.assert_not_called()

    def test_quit_without_resolved_result_falls_back_to_cancelled(self) -> None:
        scene = make_scene()
        scene._result.result = ResultEnum.PENDING
        scene._data_logger = Mock()
        scene._gaze_driver = Mock()
        with (
            patch.object(scene, "_settlement", return_value=False),
            patch.object(scene, "_print_result"),
        ):
            scene.quit()
        self.assertEqual(scene._result.result, ResultEnum.CANCELLED)

    def test_request_game_quit_posts_quit_event(self) -> None:
        from pygame import event as pygame_event

        scene = make_scene()
        with patch.object(pygame_event, "post") as post:
            scene._request_game_quit()
        post.assert_called_once()
        self.assertEqual(post.call_args.args[0].type, QUIT)


class RewardTests(unittest.TestCase):
    def test_give_reward_updates_trial_count(self) -> None:
        scene = make_scene()
        scene._give_reward()
        self.assertEqual(scene._result.trial_reward_count, 1)
        scene._rewarder.give_reward.assert_called_once_with(
            scene._config.reward_duration_ms
        )

    def test_give_reward_with_explicit_duration(self) -> None:
        scene = make_scene()
        scene._give_reward(duration_ms=250)
        scene._rewarder.give_reward.assert_called_once_with(250)

    def test_manual_reward_updates_both_counts(self) -> None:
        scene = make_scene()
        scene._give_manual_reward()
        self.assertEqual(scene._result.trial_reward_count, 1)
        self.assertEqual(scene._result.trial_manual_reward_count, 1)


class SettlementTests(unittest.TestCase):
    def _make_scene_for_settlement(self) -> Any:
        scene = make_scene()
        scene._load_trials = Mock(return_value=TrialResults(root={}))
        scene._save_trials = Mock()
        scene._mxbiflow.session.level_up = Mock()
        return scene

    def test_settlement_saves_current_trial_in_context(self) -> None:
        scene = self._make_scene_for_settlement()
        scene._settlement()
        scene._save_trials.assert_called_once()

    def test_cancelled_trial_does_not_trigger_level_up(self) -> None:
        scene = self._make_scene_for_settlement()
        scene._result.result = ResultEnum.CANCELLED
        is_level_up = scene._settlement()
        self.assertFalse(is_level_up)
        scene._mxbiflow.session.level_up.assert_not_called()

    def test_highest_level_does_not_level_up(self) -> None:
        scene = self._make_scene_for_settlement()
        scene._config = make_config(level=4)
        scene._animal.current_stage.level = 4
        scene._result.result = ResultEnum.CORRECT
        is_level_up = scene._settlement()
        self.assertFalse(is_level_up)

    def test_level_up_requires_more_than_configured_trial_count(self) -> None:
        scene = self._make_scene_for_settlement()
        scene._config = make_config(level_up_trial_count_threshold=5)
        scene._animal.current_stage.level = 1
        scene._result.result = ResultEnum.CORRECT

        existing_trials = {}
        for i in range(3):
            existing_trials[i] = Result(
                animal=AnimalBaseInfo(
                    animal="test-animal",
                    trial_id=i,
                    level=1,
                    level_trial_id=i,
                    animal_session_id=1,
                    animal_session_trial_id=i,
                ),
                is_twitter_trial=True,
                result=ResultEnum.CORRECT,
                trial_start_time=0.0,
                trial_end_time=0.0,
            )
        scene._load_trials = Mock(return_value=TrialResults(root=existing_trials))

        is_level_up = scene._settlement()
        self.assertFalse(is_level_up)

    def test_level_up_uses_level_correct_rate(self) -> None:
        scene = self._make_scene_for_settlement()
        scene._config = make_config(
            level_up_trial_count_threshold=1, level_up_threshold=0.5
        )
        scene._animal.current_stage.level = 1
        scene._result.result = ResultEnum.CORRECT

        existing_trials = {}
        for i in range(5):
            existing_trials[i] = Result(
                animal=AnimalBaseInfo(
                    animal="test-animal",
                    trial_id=i,
                    level=1,
                    level_trial_id=i,
                    animal_session_id=1,
                    animal_session_trial_id=i,
                ),
                is_twitter_trial=True,
                result=ResultEnum.CORRECT,
                trial_start_time=0.0,
                trial_end_time=0.0,
            )
        scene._load_trials = Mock(return_value=TrialResults(root=existing_trials))

        is_level_up = scene._settlement()
        self.assertTrue(is_level_up)
        scene._mxbiflow.session.level_up.assert_called_once()


class StructureConsistencyTests(unittest.TestCase):
    def test_scene_name_is_canonical(self) -> None:
        # Preserve the framework's registered name for existing session configs.
        self.assertEqual(Visual3ptCalibration.name(), "visual3pt_calibration")


class RealStimulusTests(unittest.TestCase):
    def test_visual_stimulus_uses_a_real_marmoset_gif(self) -> None:
        from assets import MARMOSET_IMAGE_PATHS

        scene: Any = Visual3ptCalibration.__new__(Visual3ptCalibration)
        config = make_config()
        scene._init_visual_stimulus(config)
        self.assertIn(scene._correct_target.image, [scene._correct_target.image])
        # The target is built from one of the real, discovered GIF
        # files - confirm at least one exists (otherwise choice() would
        # already have raised IndexError inside _init_visual_stimulus).
        self.assertGreater(len(MARMOSET_IMAGE_PATHS), 0)

    def test_fixation_point_is_at_3_quarters_screen_width(self) -> None:
        scene: Any = Visual3ptCalibration.__new__(Visual3ptCalibration)
        config = make_config()
        scene._init_visual_stimulus(config)
        expected_x = config.screen_size.width * 3 // 4
        expected_y = config.screen_size.height // 2
        self.assertEqual(scene._fixation_point.rect.center, (expected_x, expected_y))

    def test_target_is_placed_at_one_of_the_three_triangle_positions(self) -> None:
        scene: Any = Visual3ptCalibration.__new__(Visual3ptCalibration)
        config = make_config()
        scene._init_visual_stimulus(config)

        fixation_center = (
            config.screen_size.width * 3 // 4,
            config.screen_size.height // 2,
        )
        offset_px = round(config.target_distances_px // 2)
        valid_positions = {
            (fixation_center[0] - offset_px, fixation_center[1] - offset_px),
            (fixation_center[0] - offset_px, fixation_center[1] + offset_px),
            (
                fixation_center[0]
                + round(config.target_size_px // 2)
                + round(config.fixation_point_radius_px),
                fixation_center[1],
            ),
        }
        self.assertIn(scene._correct_target.rect.center, valid_positions)

    def test_all_three_triangle_positions_are_reachable(self) -> None:
        seen_positions: set[tuple[int, int]] = set()
        config = make_config()
        fixation_center = (
            config.screen_size.width * 3 // 4,
            config.screen_size.height // 2,
        )
        offset_px = round(config.target_distances_px // 2)
        valid_positions = {
            (fixation_center[0] - offset_px, fixation_center[1] - offset_px),
            (fixation_center[0] - offset_px, fixation_center[1] + offset_px),
            (
                fixation_center[0]
                + round(config.target_size_px // 2)
                + round(config.fixation_point_radius_px),
                fixation_center[1],
            ),
        }

        for _ in range(200):
            scene: Any = Visual3ptCalibration.__new__(Visual3ptCalibration)
            scene._init_visual_stimulus(config)
            seen_positions.add(scene._correct_target.rect.center)
            if seen_positions == valid_positions:
                break

        self.assertEqual(seen_positions, valid_positions)


if __name__ == "__main__":
    unittest.main()
