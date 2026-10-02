# pyright: reportPrivateUsage=false
"""Regression test for the marimo explorer form (notebook.py cell 3).

A `.batch()` only substitutes its cloned controls into *templated* HTML.
Building the form from a `mo.vstack` of the original controls leaves the
originals in the DOM while the batch tracks the clones, so submissions
silently use the default values and UI edits never take effect. The explorer
therefore builds the form from an `mo.md` template; this test pins that
structure and the submit roundtrip.
"""

import unittest
from typing import Any

import marimo as mo


def _build_form() -> tuple[Any, dict[str, Any]]:
    """Mirror the explorer's template form construction (cell 3)."""
    center_frequency = mo.ui.number(
        label="center frequency",
        start=1,
        stop=24000,
        value=9000.0,
        full_width=True,
    )
    mean_phrase_sweep_time = mo.ui.number(
        label="mean phrase sweep time",
        start=0.001,
        step=0.001,
        value=0.05,
        full_width=True,
    )
    phrase_count = mo.ui.number(
        label="phrase count", start=0, step=1, value=3, full_width=True
    )
    controls = {
        "center_frequency": center_frequency,
        "mean_phrase_sweep_time": mean_phrase_sweep_time,
        "phrase_count": phrase_count,
    }
    form = (
        mo.md(
            "**head**\n\n{center_frequency}\n\n{mean_phrase_sweep_time}\n\n{phrase_count}"
        )
        .batch(**controls)
        .form(submit_button_label="Synthesize", show_clear_button=True)
    )
    return form, controls


class TestExplorerForm(unittest.TestCase):
    def test_renders_batched_clones_not_originals(self) -> None:
        form, controls = _build_form()
        element = form.element
        for key, original in controls.items():
            clone_id = element.elements[key]._id
            with self.subTest(key=key):
                self.assertIn(clone_id, element.text)
                self.assertNotIn(original._id, element.text)

    def test_submission_delivers_edited_values(self) -> None:
        form, _ = _build_form()
        edits = {
            "center_frequency": 8000.0,
            "mean_phrase_sweep_time": 0.123,
            "phrase_count": 7.0,
        }
        # Frontend equivalent: edits reach the batch (via the rendered clones),
        # then the submit button sends the batch value to the form.
        element = form.element
        element._update(edits)
        form._update(element.value)
        self.assertEqual(form.value, edits)


if __name__ == "__main__":
    unittest.main()
