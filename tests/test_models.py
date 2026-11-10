# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false

import json
import unittest
from pathlib import Path
from typing import Any

import models
from stages.visual_3pt_calibration import models as visual_3pt_calibration_models
from stages.visual_stability_training import models as visual_stability_training_models
from stages.visual_tracing import models as visual_tracing_models


class SharedModelTests(unittest.TestCase):
    def test_screen_is_shared(self) -> None:
        self.assertIs(models.Screen, visual_3pt_calibration_models.Screen)
        self.assertIs(models.Screen, visual_stability_training_models.Screen)
        self.assertIs(models.Screen, visual_tracing_models.Screen)

    def test_result_enum_is_shared(self) -> None:
        self.assertIs(models.ResultEnum, visual_3pt_calibration_models.ResultEnum)
        self.assertIs(models.ResultEnum, visual_stability_training_models.ResultEnum)
        self.assertIs(models.ResultEnum, visual_tracing_models.ResultEnum)

    def test_each_stage_config_file_parses_with_its_stored_configs(self) -> None:
        for stage_models, stage in (
            (visual_3pt_calibration_models, "visual_3pt_calibration"),
            (visual_stability_training_models, "visual_stability_training"),
            (visual_tracing_models, "visual_tracing"),
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

    def test_visual_tracing_trial_results_round_trip(self) -> None:
        # visual_tracing's Result shape is used here as a representative
        # example; each stage's Result shape now genuinely differs (e.g.
        # visual_3pt_calibration no longer has response_latency_ms at
        # all), so a single shared round-trip shape across every stage,
        # like the old auditory-stimulus project's stimulus/target_side
        # fields, would no longer be accurate.
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
                "result": "correct",
                "trial_start_time": 1.0,
                "trial_end_time": 2.0,
                "target_x": 100,
                "target_y": 200,
                "target_diameter_px": 28,
            }
        }

        trials = visual_tracing_models.TrialResults.model_validate_json(
            json.dumps(raw)
        )
        restored = visual_tracing_models.TrialResults.model_validate_json(
            trials.model_dump_json()
        )

        self.assertEqual(restored, trials)
        self.assertEqual(list(restored.root), [1])


if __name__ == "__main__":
    unittest.main()