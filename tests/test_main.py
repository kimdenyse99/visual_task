import unittest
from unittest.mock import Mock, patch

import main
from stages.size_reduction_3pt import SizeReduction3Pt
from stages.target_position_generalization_training import (
    TargetPositionGeneralizationTraining,
)
from stages.touchscreen_training import TouchscreenTraining
from stages.visual_3pt_calibration import Visual3ptCalibration
from stages.visual_stability_training import VisualStabilityTraining
from stages.visual_tracing import VisualTracing


class StartupFlowTests(unittest.TestCase):
    def test_main_delegates_the_run_to_mxbiflow(self):
        scene_manager = Mock()
        with (
            patch("main.SceneManager", return_value=scene_manager),
            patch("main.set_base_path"),
            patch("main.setup_logging"),
            patch("main.run_experiment") as run_experiment,
        ):
            main.main()

        scene_manager.register.assert_called_once_with(
            [
                SizeReduction3Pt,
                VisualStabilityTraining,
                Visual3ptCalibration,
                VisualTracing,
                TouchscreenTraining,
                TargetPositionGeneralizationTraining,
            ]
        )
        run_experiment.assert_called_once_with(scene_manager, max_fps=30)


if __name__ == "__main__":
    unittest.main()
