# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false

import json
import unittest
from pathlib import Path
from typing import Any

import models
from stages.target_position_generalization_training import models as position_models
from stages.touchscreen_training import models as size_models
from stages.visual_category_position_decoupling_training import (
    models as single_target_position_models,
)
from stages.visual_two_choice_training import models as visual_cue_models


class SharedModelTests(unittest.TestCase):
    def test_screen_is_shared(self) -> None:
        self.assertIs(models.Screen, size_models.Screen)
        self.assertIs(models.Screen, position_models.Screen)
        self.assertIs(models.Screen, visual_cue_models.Screen)
        self.assertIs(models.Screen, single_target_position_models.Screen)

    def test_result_enum_is_shared(self) -> None:
        self.assertIs(models.ResultEnum, size_models.ResultEnum)
        self.assertIs(models.ResultEnum, position_models.ResultEnum)
        self.assertIs(models.ResultEnum, visual_cue_models.ResultEnum)
        self.assertIs(models.ResultEnum, single_target_position_models.ResultEnum)

    def test_each_stage_config_file_parses_with_its_stored_configs(self) -> None:
        for stage_models, stage in (
            (size_models, "touchscreen_training"),
            (position_models, "target_position_generalization_training"),
            (visual_cue_models, "visual_two_choice_training"),
            (
                single_target_position_models,
                "visual_category_position_decoupling_training",
            ),
        ):
            with self.subTest(stage=stage):
                config_path = Path("stages") / stage / "config.json"
                data: dict[str, Any] = json.loads(
                    config_path.read_text(encoding="utf-8")
                )
                stored = stage_models.StoredConfigs.model_validate(data).root["default"]
                self.assertEqual(
                    list(stored.levels),
                    list(range(1, len(stored.levels) + 1)),
                )
                self.assertEqual(stored.visual_delay_ms, 500)

    def test_trial_results_round_trip_per_stage(self) -> None:
        for stage_models, stimulus in (
            (size_models, {"wav_filename": "natural-twitter-1.wav"}),
            (
                position_models,
                {
                    "pure_tone_frequency_hz": 2000,
                    "pure_tone_duration_ms": 1000,
                },
            ),
            (
                visual_cue_models,
                {
                    "pure_tone_frequency_hz": 2000,
                    "pure_tone_duration_ms": 1000,
                },
            ),
            (
                single_target_position_models,
                {
                    "pure_tone_frequency_hz": 2000,
                    "pure_tone_duration_ms": 1000,
                },
            ),
        ):
            with self.subTest(stage=stage_models.__name__):
                raw: dict[str, Any] = {
                    "1": {
                        "version": "1.0.0",
                        "animal": {
                            "animal": "test-animal",
                            "trial_id": 1,
                            "level": 1,
                            "level_trial_id": 1,
                            "animal_session_id": 1,
                            "animal_session_trial_id": 1,
                        },
                        "is_twitter_trial": True,
                        "stimulus": stimulus,
                        "result": "correct",
                        "trial_start_time": 1.0,
                        "trial_end_time": 2.0,
                    }
                }
                if stage_models in (position_models, single_target_position_models):
                    raw["1"]["target_side"] = "left"

                trials = stage_models.TrialResults.model_validate_json(json.dumps(raw))
                restored = stage_models.TrialResults.model_validate_json(
                    trials.model_dump_json()
                )

                self.assertEqual(restored, trials)
                self.assertEqual(list(restored.root), [1])


if __name__ == "__main__":
    unittest.main()
