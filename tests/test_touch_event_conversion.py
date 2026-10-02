# pyright: reportPrivateUsage=false

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

from pygame import FINGERDOWN, MOUSEBUTTONDOWN, Event

from stages.target_position_generalization_training.target_position_generalization_training import (
    TargetPositionGeneralizationTraining,
)
from stages.touchscreen_training.touchscreen_training import TouchscreenTraining
from stages.twitter_category_boundary_mapping.twitter_category_boundary_mapping import (
    TwitterCategoryBoundaryMapping,
)
from stages.visual_category_position_decoupling_training.visual_category_position_decoupling_training import (
    VisualCategoryPositionDecouplingTraining,
)
from stages.visual_two_choice_training.visual_two_choice_training import (
    VisualTwoChoiceTraining,
)


class FingerEventConversionTests(unittest.TestCase):
    def test_finger_event_is_converted_for_every_touchscreen_stage(self) -> None:
        stage_classes: tuple[type[Any], ...] = (
            TouchscreenTraining,
            TargetPositionGeneralizationTraining,
            VisualTwoChoiceTraining,
            VisualCategoryPositionDecouplingTraining,
            TwitterCategoryBoundaryMapping,
        )

        for stage_class in stage_classes:
            with self.subTest(stage=stage_class.__name__):
                scene: Any = stage_class.__new__(stage_class)
                scene._mxbiflow = SimpleNamespace(
                    mxbi=SimpleNamespace(
                        screen_size=SimpleNamespace(width=1024, height=600)
                    )
                )
                scene._on_touched = Mock()

                scene.handle_event(Event(FINGERDOWN, x=0.25, y=0.5))

                scene._on_touched.assert_called_once()
                touch_event = scene._on_touched.call_args.args[0]
                self.assertEqual(touch_event.type, MOUSEBUTTONDOWN)
                self.assertEqual(touch_event.button, 1)
                self.assertEqual(touch_event.pos, (256, 300))


if __name__ == "__main__":
    unittest.main()
