import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call

from pygame import Surface
from pygame.sprite import Group, Sprite

from stages.visual_3pt_calibration.scene import Visual3ptCalibration
from stages.visual_stability_training.scene import VisualStabilityTraining
from stages.visual_tracing.scene import VisualTracing
from stimulus_events import StimulusEvents


def sprite(center):
    result = Sprite()
    result.image = Surface((40, 40))
    result.rect = result.image.get_rect(center=center)
    return result


class StimulusEventTests(unittest.TestCase):
    def test_visibility_changes_and_interruptions(self):
        flow = Mock()
        events = StimulusEvents(flow)
        fixation, target = sprite((100, 200)), sprite((400, 300))
        group = Group(fixation)
        events.present(group, fixation, target, "m001.gif")
        details = {
            "stimulus_type": "circle",
            "position": (100, 200),
            "size": (40, 40),
            "metadata": {},
        }
        flow.stimulus_event.assert_called_once_with(
            "stimulus_shown", "fixation", **details
        )
        # A delay/phase change with unchanged sprites must not hide fixation.
        events.present(group, fixation, target, "m001.gif")
        self.assertEqual(flow.stimulus_event.call_count, 1)
        group.remove(fixation)
        group.add(target)
        events.present(group, fixation, target, "m001.gif")
        self.assertEqual(
            flow.stimulus_event.call_args_list[1],
            call("stimulus_hidden", "fixation", **details),
        )
        shown = flow.stimulus_event.call_args
        self.assertEqual(shown.args, ("stimulus_shown", "target"))
        self.assertEqual(shown.kwargs["metadata"], {"asset": "m001.gif"})
        # GIF image changes alone do not emit per-animation-frame events.
        target.image = Surface((40, 40))
        events.present(group, fixation, target, "m001.gif")
        self.assertEqual(flow.stimulus_event.call_count, 3)
        events.clear()
        events.clear()
        self.assertEqual(flow.stimulus_event.call_count, 4)
        self.assertEqual(
            flow.stimulus_event.call_args.args, ("stimulus_hidden", "target")
        )

    def test_normal_removal_and_unpresented_stimulus(self):
        flow = Mock()
        events = StimulusEvents(flow)
        fixation, target = sprite((100, 200)), sprite((400, 300))
        group = Group()
        group.add(target)
        group.remove(target)
        events.present(group, fixation, target)
        flow.stimulus_event.assert_not_called()
        group.add(target)
        events.present(group, fixation, target)
        group.remove(target)
        events.present(group, fixation, target)
        events.clear()
        self.assertEqual(
            [c.args for c in flow.stimulus_event.call_args_list],
            [("stimulus_shown", "target"), ("stimulus_hidden", "target")],
        )

    def test_all_stage_draw_hooks_report_actual_visible_sprites(self):
        for stage in (Visual3ptCalibration, VisualStabilityTraining, VisualTracing):
            with self.subTest(stage=stage.__name__):
                scene = stage.__new__(stage)
                flow = Mock()
                scene._stimulus_events = StimulusEvents(flow)
                fixation, target = sprite((100, 200)), sprite((400, 300))
                scene._fixation_point = scene._fixation_sprite = fixation
                scene._correct_target = scene._target_sprite = target
                scene._target_asset_name = "m001.gif"
                scene._sprites = Group(fixation)
                scene._background = Surface((640, 480))
                scene._config = SimpleNamespace(show_gaze_trace=False)
                scene._draw_trace = Mock()
                scene._result_widget = Mock()
                scene._control_buttons = Group()
                screen = Surface((640, 480))
                scene.draw(screen)
                self.assertEqual(
                    flow.stimulus_event.call_args.args, ("stimulus_shown", "fixation")
                )
                scene._sprites.remove(fixation)
                scene._sprites.add(target)
                scene.draw(screen)
                self.assertEqual(
                    flow.stimulus_event.call_args.args, ("stimulus_shown", "target")
                )
                scene._sprites.remove(target)
                scene.draw(screen)
                self.assertEqual(
                    flow.stimulus_event.call_args.args, ("stimulus_hidden", "target")
                )
