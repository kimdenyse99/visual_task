#For object-oriented programming
from pathlib import Path

#Importing tools
from loguru import logger #logging for debugging

#Importing from mxbiflow library
from mxbiflow import get_mxbiflow, set_base_path
from mxbiflow.bootstrap import init_gameloop
from mxbiflow.infra import send_crash_report
from mxbiflow.infra.post_processing import StagePostProcessor
from mxbiflow.models.session import Session
from mxbiflow.scene import SceneManager
from mxbiflow.ui import run_session_post_processing, run_session_summary, run_wizard
from mxbiflow.utils.logger import setup_logging

#Import different modules for actual game
from stages.target_position_generalization_training import (
    TargetPositionGeneralizationTraining,
)
from stages.touchscreen_training import TouchscreenTraining
from stages.visual_3pt_calibration import Visual3ptCalibration
from stages.visual_stability_training import VisualStabilityTraining
from stages.size_reduction_3pt import SizeReduction3Pt
from stages.visual_tracing import VisualTracing

#Game code
STAGE_POST_PROCESSORS: dict[str, StagePostProcessor] = {}


def main() -> None:
    set_base_path(Path.cwd())
    setup_logging()

    logger.info("starting 2ac-vocalization-discriminate")

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

    session: Session | None = None

    try:
        if not run_wizard(scene_manager):
            return

        init_gameloop(scene_manager, 30).play()

        session = get_mxbiflow().session

        if not run_session_summary():
            return

        run_session_post_processing(session, STAGE_POST_PROCESSORS)

    except Exception as exc:
        logger.exception("experiment crashed")
        send_crash_report(exc, session)
        raise


if __name__ == "__main__":
    main()
