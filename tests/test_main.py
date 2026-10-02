# pyright: reportUnknownLambdaType=false

import unittest
from unittest.mock import Mock, patch

import main
from stages.target_position_generalization_training import (
    TargetPositionGeneralizationTraining,
)
from stages.touchscreen_training import TouchscreenTraining
from stages.twitter_category_boundary_mapping import TwitterCategoryBoundaryMapping
from stages.visual_category_position_decoupling_training import (
    VisualCategoryPositionDecouplingTraining,
)
from stages.visual_two_choice_training import VisualTwoChoiceTraining


class StartupFlowTests(unittest.TestCase):
    def test_stage_names_describe_training_purposes(self) -> None:
        self.assertEqual(
            {
                scene.name()
                for scene in (
                    TouchscreenTraining,
                    TargetPositionGeneralizationTraining,
                    VisualTwoChoiceTraining,
                    VisualCategoryPositionDecouplingTraining,
                    TwitterCategoryBoundaryMapping,
                )
            },
            {
                "touchscreen_training",
                "target_position_generalization_training",
                "visual_two_choice_training",
                "visual_category_position_decoupling_training",
                "twitter_category_boundary_mapping",
            },
        )

    def test_stage_post_processor_uses_canonical_scene_name(self) -> None:
        self.assertEqual(
            set(main.STAGE_POST_PROCESSORS),
            {VisualTwoChoiceTraining.name()},
        )
        self.assertEqual(
            VisualTwoChoiceTraining.name(),
            "visual_two_choice_training",
        )

    def test_wizard_rejection_stops_the_flow(self) -> None:
        scene_manager = Mock()
        with (
            patch("main.SceneManager", return_value=scene_manager),
            patch("main.set_base_path"),
            patch("main.setup_logging") as setup_logging,
            patch("main.run_wizard", return_value=False) as run_wizard,
            patch("main.init_gameloop") as init_gameloop,
            patch("main.get_mxbiflow") as get_mxbiflow,
        ):
            main.main()

        setup_logging.assert_called_once_with()
        scene_manager.register.assert_called_once_with(
            [
                TouchscreenTraining,
                TargetPositionGeneralizationTraining,
                VisualTwoChoiceTraining,
                VisualCategoryPositionDecouplingTraining,
                TwitterCategoryBoundaryMapping,
            ]
        )
        run_wizard.assert_called_once_with(scene_manager)
        init_gameloop.assert_not_called()
        get_mxbiflow.assert_not_called()

    def test_accepted_summary_runs_post_processing(self) -> None:
        events: list[str] = []
        scene_manager = Mock()
        session = Mock()
        with (
            patch("main.SceneManager", return_value=scene_manager),
            patch("main.set_base_path"),
            patch("main.setup_logging"),
            patch("main.run_wizard", return_value=True),
            patch("main.init_gameloop") as init_gameloop,
            patch("main.get_mxbiflow") as get_mxbiflow,
            patch("main.run_session_summary") as run_session_summary,
            patch("main.run_session_post_processing") as run_session_post_processing,
        ):
            game = init_gameloop.return_value
            game.play.side_effect = lambda: events.append("pygame")
            get_mxbiflow.return_value.session = session
            run_session_summary.side_effect = lambda: events.append("summary") or True
            run_session_post_processing.side_effect = lambda _session, _processors: (
                events.append("post-processing")
            )

            main.main()

        init_gameloop.assert_called_once_with(scene_manager, 30)
        game.play.assert_called_once_with()
        run_session_summary.assert_called_once_with()
        self.assertEqual(events, ["pygame", "summary", "post-processing"])
        run_session_post_processing.assert_called_once_with(
            session,
            main.STAGE_POST_PROCESSORS,
        )

    def test_rejected_summary_skips_post_processing(self) -> None:
        scene_manager = Mock()
        with (
            patch("main.SceneManager", return_value=scene_manager),
            patch("main.set_base_path"),
            patch("main.setup_logging"),
            patch("main.run_wizard", return_value=True),
            patch("main.init_gameloop") as init_gameloop,
            patch("main.get_mxbiflow") as get_mxbiflow,
            patch(
                "main.run_session_summary", return_value=False
            ) as run_session_summary,
            patch("main.run_session_post_processing") as run_session_post_processing,
        ):
            main.main()

        init_gameloop.return_value.play.assert_called_once_with()
        get_mxbiflow.assert_called_once_with()
        run_session_summary.assert_called_once_with()
        run_session_post_processing.assert_not_called()

    def test_crash_sends_report_and_reraises(self) -> None:
        error = RuntimeError("boom")
        scene_manager = Mock()
        session = Mock()
        with (
            patch("main.SceneManager", return_value=scene_manager),
            patch("main.set_base_path"),
            patch("main.setup_logging"),
            patch("main.run_wizard", return_value=True),
            patch("main.init_gameloop"),
            patch("main.get_mxbiflow") as get_mxbiflow,
            patch("main.run_session_summary", side_effect=error),
            patch("main.send_crash_report") as send_crash_report,
            patch("main.logger"),
        ):
            get_mxbiflow.return_value.session = session

            with self.assertRaises(RuntimeError):
                main.main()

        send_crash_report.assert_called_once_with(error, session)


if __name__ == "__main__":
    unittest.main()
