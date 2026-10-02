# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false

import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from mxbiflow.assets import CLICKER_PATH
from mxbiflow.driver.audioplayer import PlayStatus
from mxbiflow.models.animal import AnimalBaseInfo
from pydantic import ValidationError
from pygame import QUIT

from assets import M001, NATURAL_TWITTER_PATHS
from stages.touchscreen_training.models import (
    Config,
    LevelConfig,
    Result,
    ResultEnum,
    Screen,
    StoredConfig,
    TrialResults,
    TrialStimulus,
)
from stages.touchscreen_training.touchscreen_training import TouchscreenTraining


def make_stored_config() -> StoredConfig:
    return StoredConfig(
        inter_trial_duration_ms=5000,
        reward_duration_ms=1000,
        attention_duration_ms=1000,
        levels={
            1: LevelConfig(
                reaction_window_duration_ms=5000,
                target_size_ratio=0.4,
                reaction_timeout_enabled=False,
            ),
            2: LevelConfig(
                reaction_window_duration_ms=4000,
                target_size_ratio=0.3,
                reaction_timeout_enabled=False,
            ),
            3: LevelConfig(
                reaction_window_duration_ms=4000,
                target_size_ratio=0.25,
                reaction_timeout_enabled=False,
            ),
            4: LevelConfig(
                reaction_window_duration_ms=4000,
                target_size_ratio=0.2,
                reaction_timeout_enabled=True,
            ),
        },
        border_width=40,
        level_up_threshold=0.8,
        level_up_trial_count_threshold=20,
    )


def make_config(level: int = 1) -> Config:
    return Config(
        animal="test-animal",
        level=level,
        is_twitter_trial=True,
        wav_path=NATURAL_TWITTER_PATHS[0],
        screen_size=Screen(width=1920, height=1080),
        stored=make_stored_config(),
    )


def make_scene() -> Any:
    scene: Any = TouchscreenTraining.__new__(TouchscreenTraining)
    scene._config = make_config()
    scene._timer = Mock()
    scene._sprites = Mock()
    scene._correct_target = Mock()
    scene._is_reaction_window_open = False
    scene._reaction_window_started_at_ms = None
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
        stimulus=TrialStimulus(wav_filename="natural-twitter-1.wav"),
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
    @patch("stages.touchscreen_training.touchscreen_training.get_mxbiflow")
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
                aplayer=Mock(),
            ),
            timer=Mock(),
        )

        result = Mock()
        result_widget = Mock()
        with (
            patch.object(
                TouchscreenTraining,
                "_init_config",
                side_effect=lambda *_: events.append("config") or make_config(),
            ),
            patch.object(
                TouchscreenTraining,
                "_init_data_logger",
                side_effect=lambda *_: events.append("data_logger") or Mock(),
            ),
            patch.object(
                TouchscreenTraining,
                "_init_control_buttons",
                side_effect=lambda: events.append("controls") or Mock(),
            ),
            patch.object(
                TouchscreenTraining,
                "_init_visual_stimulus",
                side_effect=lambda *_: events.append("visual_stimulus"),
            ),
            patch.object(
                TouchscreenTraining,
                "_init_auditory_stimulus",
                side_effect=lambda *_: events.append("auditory_stimulus") or Mock(),
            ),
            patch.object(
                TouchscreenTraining,
                "_init_result",
                side_effect=lambda *_: events.append("result") or result,
            ),
            patch.object(
                TouchscreenTraining,
                "_init_result_widget",
                side_effect=lambda: events.append("result_widget") or result_widget,
            ),
            patch.object(TouchscreenTraining, "_print_trial_config"),
        ):
            scene = TouchscreenTraining()

        self.assertEqual(
            events,
            [
                "config",
                "data_logger",
                "controls",
                "visual_stimulus",
                "auditory_stimulus",
                "add_trial",
                "result",
                "result_widget",
            ],
        )
        self.assertIs(scene._result, result)
        self.assertIs(scene._result_widget, result_widget)

    @patch(
        "stages.touchscreen_training.touchscreen_training.time",
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
        self.assertEqual(
            result.stimulus,
            TrialStimulus(wav_filename="natural-twitter-1.wav"),
        )
        self.assertNotIn("screen_size", result.model_dump())
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
        "stages.touchscreen_training.touchscreen_training.time",
        return_value=456.0,
    )
    def test_trial_start_waits_for_attention_before_stimulus(self, _time: Mock) -> None:
        scene = make_scene()
        scene._play_stimulus = Mock()

        scene._on_trial_start()

        self.assertEqual(scene._result.trial_start_time, 456.0)
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

    def test_target_size_ratio_is_level_specific(self) -> None:
        self.assertEqual(make_config(level=1).target_size_ratio, 0.4)
        self.assertEqual(make_config(level=2).target_size_ratio, 0.3)
        self.assertEqual(make_config(level=3).target_size_ratio, 0.25)
        self.assertEqual(make_config(level=4).target_size_ratio, 0.2)

    def test_reaction_timeout_is_enabled_only_on_final_level(self) -> None:
        self.assertFalse(make_config(level=1).reaction_timeout_enabled)
        self.assertFalse(make_config(level=2).reaction_timeout_enabled)
        self.assertFalse(make_config(level=3).reaction_timeout_enabled)
        self.assertTrue(make_config(level=4).reaction_timeout_enabled)

    def test_reaction_timeout_missing_raises(self) -> None:
        with self.assertRaises(ValidationError):
            LevelConfig.model_validate(
                {
                    "reaction_window_duration_ms": 5000,
                    "target_size_ratio": 0.4,
                }
            )

    def test_target_size_ratio_is_validated(self) -> None:
        with self.assertRaises(ValidationError):
            LevelConfig.model_validate(
                {
                    "reaction_window_duration_ms": 5000,
                    "target_size_ratio": 0,
                    "reaction_timeout_enabled": True,
                }
            )

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
                "stages.touchscreen_training.touchscreen_training.PATH",
                source_path,
            ):
                first = scene._init_config(animal, mxbiflow)
                snapshot = (
                    root / "animal-session" / "touchscreen_training" / "config.json"
                )
                self.assertTrue(snapshot.exists())
                self.assertEqual(
                    snapshot.read_text(encoding="utf-8"),
                    source_text,
                )
                second = scene._init_config(animal, mxbiflow)

        self.assertEqual(first.animal, "test-animal")
        self.assertEqual(first.level, 1)
        self.assertEqual(first.target_size_ratio, 0.4)
        self.assertTrue(first.is_twitter_trial)
        self.assertIn(first.wav_path, NATURAL_TWITTER_PATHS)
        self.assertEqual(second.animal, "test-animal")


class TouchFlowTests(unittest.TestCase):
    def test_touch_outside_reaction_window_is_recorded_and_ignored(self) -> None:
        scene = make_scene()
        scene._classify_touch = Mock()

        scene._on_touched(SimpleNamespace(pos=(10, 20)))

        self.assertEqual(len(scene._result.touch_event), 1)
        self.assertEqual(scene._result.touch_event[0].x, 10)
        self.assertEqual(scene._result.touch_event[0].y, 20)
        scene._classify_touch.assert_not_called()
        scene._sprites.remove.assert_not_called()
        scene._on_inter_trial.assert_not_called()

    def test_first_touch_in_reaction_window_closes_window(self) -> None:
        scene = make_scene()
        scene._is_reaction_window_open = True
        scene._reaction_window_started_at_ms = 1000
        scene._classify_touch = Mock(return_value=ResultEnum.CORRECT)
        scene._on_correct = Mock()

        with patch(
            "stages.touchscreen_training.touchscreen_training.monotonic",
            return_value=2.0,
        ):
            scene._on_touched(SimpleNamespace(pos=(10, 20)))
            scene._on_touched(SimpleNamespace(pos=(30, 40)))

        self.assertFalse(scene._is_reaction_window_open)
        self.assertEqual(len(scene._result.touch_event), 2)
        scene._sprites.remove.assert_called_once_with([scene._correct_target])
        scene._classify_touch.assert_called_once_with((10, 20))
        scene._on_correct.assert_called_once_with()
        self.assertEqual(scene._result.response_latency_ms, 1000)

    def test_background_touch_does_not_record_latency(self) -> None:
        scene = make_scene()
        scene._is_reaction_window_open = True
        scene._reaction_window_started_at_ms = 1000
        scene._classify_touch = Mock(return_value=ResultEnum.BACKGROUND_TOUCHED)
        scene._on_background_touched = Mock()

        scene._on_touched(SimpleNamespace(pos=(10, 20)))

        self.assertIsNone(scene._result.response_latency_ms)
        scene._on_background_touched.assert_called_once_with()

    def test_target_touch_is_classified_as_correct(self) -> None:
        scene = make_scene()
        scene._correct_target = SimpleNamespace(
            rect=SimpleNamespace(collidepoint=Mock(return_value=True))
        )

        self.assertEqual(
            scene._classify_touch((10, 20)),
            ResultEnum.CORRECT,
        )

    def test_non_target_touch_is_classified_as_background(self) -> None:
        scene = make_scene()
        scene._correct_target = SimpleNamespace(
            rect=SimpleNamespace(collidepoint=Mock(return_value=False))
        )

        self.assertEqual(
            scene._classify_touch((10, 20)),
            ResultEnum.BACKGROUND_TOUCHED,
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
                result=ResultEnum.BACKGROUND_TOUCHED,
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

        self.assertIn("Correct 1 | Reward 5", summary)
        self.assertIn("Rate 50.0%", summary)


class BehavioralOutcomeTests(unittest.TestCase):
    def test_correct_sets_outcome_and_gives_reward(self) -> None:
        scene = make_scene()
        scene._give_reward = Mock()

        scene._on_correct()

        self.assertEqual(scene._result.result, ResultEnum.CORRECT)
        scene._give_reward.assert_called_once_with()
        scene._on_inter_trial.assert_called_once_with()

    def test_background_touch_sets_outcome(self) -> None:
        scene = make_scene()

        scene._on_background_touched()

        self.assertEqual(scene._result.result, ResultEnum.BACKGROUND_TOUCHED)
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
        for background, expected_level_up in ((4, True), (5, False)):
            with self.subTest(background=background):
                past = make_past_correct(16)
                past.update(
                    {
                        trial_id: make_trial_result(
                            trial_id=trial_id,
                            result=ResultEnum.BACKGROUND_TOUCHED,
                        )
                        for trial_id in range(17, 17 + background)
                    }
                )
                scene = make_settlement_scene(
                    past_trials=past,
                    trial_id=17 + background,
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
        scene._aplayer = Mock()
        scene._running = True

        scene.quit()

        self.assertEqual(scene._result.result, ResultEnum.CANCELLED)
        scene._settlement.assert_called_once_with()
        scene._print_result.assert_called_once_with(False)
        scene._data_logger.save.assert_called_once_with({})
        scene._timer.cancel_all.assert_called_once_with()
        scene._aplayer.cancel_all.assert_called_once_with()
        self.assertFalse(scene._running)

    def test_request_game_quit_posts_quit_event(self) -> None:
        scene = make_scene()

        with patch(
            "stages.touchscreen_training.touchscreen_training.event.post"
        ) as post:
            scene._request_game_quit()

        post.assert_called_once()
        posted = post.call_args.args[0]
        self.assertEqual(posted.type, QUIT)


class StructureConsistencyTests(unittest.TestCase):
    REFERENCE_PATH = Path(
        "stages/visual_two_choice_training/visual_two_choice_training.py"
    )
    TARGET_PATH = Path("stages/touchscreen_training/touchscreen_training.py")

    @staticmethod
    def _region_markers(path: Path) -> list[str]:
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith("# region")
            or line.strip().startswith("# endregion")
        ]

    @staticmethod
    def _class_member_names(path: Path) -> list[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                for member in node.body:
                    if isinstance(member, ast.FunctionDef):
                        names.append(member.name)
        return names

    def test_region_markers_match_reference_stage(self) -> None:
        self.assertEqual(
            self._region_markers(self.TARGET_PATH),
            self._region_markers(self.REFERENCE_PATH),
        )

    def test_top_level_function_order_matches_reference_stage(self) -> None:
        target_names = self._class_member_names(self.TARGET_PATH)
        reference_names = self._class_member_names(self.REFERENCE_PATH)

        # _on_incorrect is unreachable in the single-target stage and is
        # deliberately omitted; every other member must match in order.
        expected_omitted = ("_on_incorrect",)
        expected_target_names = [
            name for name in reference_names if name not in expected_omitted
        ]

        self.assertEqual(target_names, expected_target_names)
        self.assertEqual(len(target_names), len(reference_names) - 1)

    def test_scene_name_is_canonical(self) -> None:
        self.assertEqual(TouchscreenTraining.name(), "touchscreen_training")

    def test_play_status_finished_without_timeout_keeps_window_open(self) -> None:
        scene = make_scene()
        scene._sprites = Mock()
        scene._timer = Mock()
        scene._config = SimpleNamespace(
            reaction_window_duration_ms=5000,
            reaction_timeout_enabled=False,
        )

        with patch(
            "stages.touchscreen_training.touchscreen_training.monotonic",
            return_value=3.0,
        ):
            scene._show_visual_stimulus()

        self.assertTrue(scene._is_reaction_window_open)
        self.assertEqual(scene._reaction_window_started_at_ms, 3000)
        scene._sprites.add.assert_called_once_with([scene._correct_target])
        scene._timer.after.assert_not_called()

    def test_play_status_finished_with_timeout_schedules_timeout(self) -> None:
        scene = make_scene()
        scene._sprites = Mock()
        scene._timer = Mock()
        scene._config = SimpleNamespace(
            reaction_window_duration_ms=5000,
            reaction_timeout_enabled=True,
        )

        with patch(
            "stages.touchscreen_training.touchscreen_training.monotonic",
            return_value=3.0,
        ):
            scene._show_visual_stimulus()

        self.assertTrue(scene._is_reaction_window_open)
        self.assertEqual(scene._reaction_window_started_at_ms, 3000)
        scene._sprites.add.assert_called_once_with([scene._correct_target])
        scene._timer.after.assert_called_once_with(5000, scene._on_timeout)


class RealStimulusTests(unittest.TestCase):
    def test_visual_stimulus_uses_real_marmoset_image(self) -> None:
        scene = make_scene()
        scene._init_backgrounds = Mock()
        with (
            patch(
                "stages.touchscreen_training.touchscreen_training.choice",
                return_value=M001,
            ),
            patch(
                "stages.touchscreen_training.touchscreen_training.ImageSprite"
            ) as image_sprite,
        ):
            scene._init_visual_stimulus(make_config())

        image_sprite.assert_called_once_with(M001, 768, (1440, 540))

    def test_auditory_stimulus_always_loads_natural_twitter(self) -> None:
        scene = make_scene()
        scene._aplayer = Mock()

        stimulus = scene._init_auditory_stimulus()

        self.assertIn(scene._config.wav_path, NATURAL_TWITTER_PATHS)
        scene._aplayer.load_wav.assert_called_once_with(scene._config.wav_path)
        self.assertIs(stimulus, scene._aplayer.load_wav.return_value)

    def test_play_stimulus_plays_loaded_wav_only(self) -> None:
        scene = make_scene()
        scene._background = None
        scene._stimulus_background = "stimulus"
        scene._auditory_stimulus = "loaded-wav"
        scene._aplayer = Mock()
        scene._show_visual_stimulus = Mock()
        click_task = Mock()
        scene._aplayer.play_wav.return_value = click_task

        scene._play_stimulus()

        self.assertEqual(scene._background, "stimulus")
        scene._aplayer.play_wav.assert_called_once_with(CLICKER_PATH)
        click_task.on_finish.assert_called_once_with(scene._on_click_played)
        scene._aplayer.play_loaded_wav.assert_called_once_with("loaded-wav")
        scene._timer.after.assert_called_once_with(500, scene._show_visual_stimulus)
        task = scene._aplayer.play_loaded_wav.return_value
        task.on_finish.assert_called_once()
        on_audio_finished = task.on_finish.call_args.args[0]
        on_audio_finished(SimpleNamespace(status=PlayStatus.FINISHED, error=None))
        scene._timer.cancel.assert_not_called()
        scene._aplayer.play_puretone_sequence.assert_not_called()

    def test_audio_error_cancels_visual_and_raises(self) -> None:
        scene = make_scene()
        scene._background = None
        scene._stimulus_background = "stimulus"
        scene._auditory_stimulus = "loaded-wav"
        scene._aplayer = Mock()
        click_task = Mock()
        scene._aplayer.play_wav.return_value = click_task

        scene._play_stimulus()

        visual_task = scene._timer.after.return_value
        task = scene._aplayer.play_loaded_wav.return_value
        on_audio_finished = task.on_finish.call_args.args[0]
        with self.assertRaisesRegex(RuntimeError, "playback failed"):
            on_audio_finished(
                SimpleNamespace(status=PlayStatus.ERROR, error=RuntimeError("boom"))
            )
        scene._timer.cancel.assert_called_once_with(visual_task)

    def test_click_success_is_silent(self) -> None:
        scene = make_scene()

        with patch("stages.touchscreen_training.touchscreen_training.logger") as logger:
            scene._on_click_played(
                SimpleNamespace(status=PlayStatus.FINISHED, error=None)
            )

        logger.warning.assert_not_called()

    def test_click_failure_only_warns(self) -> None:
        scene = make_scene()

        with patch("stages.touchscreen_training.touchscreen_training.logger") as logger:
            scene._on_click_played(
                SimpleNamespace(
                    status=PlayStatus.ERROR,
                    error=RuntimeError("boom"),
                )
            )

        logger.warning.assert_called_once()
        self.assertIn("clicker", logger.warning.call_args.args[0])

    def test_click_canceled_is_silent(self) -> None:
        scene = make_scene()

        with patch("stages.touchscreen_training.touchscreen_training.logger") as logger:
            scene._on_click_played(
                SimpleNamespace(status=PlayStatus.CANCELED, error=None)
            )

        logger.warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
