from pathlib import Path

from loguru import logger
from mxbiflow import set_base_path
from mxbiflow.run import run_experiment
from mxbiflow.scene import SceneManager
from mxbiflow.utils.logger import setup_logging

from stages.size_reduction_3pt import SizeReduction3Pt
from stages.target_position_generalization_training import (
    TargetPositionGeneralizationTraining,
)
from stages.touchscreen_training import TouchscreenTraining
from stages.visual_3pt_calibration import Visual3ptCalibration
from stages.visual_stability_training import VisualStabilityTraining
from stages.visual_tracing import VisualTracing


def main() -> None:
    set_base_path(Path.cwd())
    setup_logging()

    logger.info("starting visual_tasks")

    scene_manager = SceneManager()
    scene_manager.register(
        [
            SizeReduction3Pt,
            VisualStabilityTraining,
            Visual3ptCalibration,
            VisualTracing,
            TouchscreenTraining,
            TargetPositionGeneralizationTraining,
        ]
    )

    # The wizard by default; `--from-cogmote` runs headless from the cogmote configuration.
    # Either way the run is recorded in CogMatrix and reported when it ends.
    run_experiment(scene_manager, max_fps=30)


if __name__ == "__main__":
    main()
