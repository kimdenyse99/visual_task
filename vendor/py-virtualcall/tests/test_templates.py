# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Tests for the call parameter template loader."""

import unittest
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError
from scipy.io import loadmat

from py_virtualcall.template import CallOptions, Chop, Phrase, TimeReversal, TwitStruct
from py_virtualcall.template.assets import TWITTER

ASSETS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "py_virtualcall"
    / "template"
    / "assets"
)

MAT_PATH = ASSETS / "raw_mat_data" / "twitter.mat"

# C_cell column names in the MATLAB cell array, mapped to the Python Phrase
# field names, in column order (0-15).
_MAT_TO_MODEL_PHRASE = (
    "start_frequency",  # col 0: fst
    "end_frequency",  # col 1: fsp
    "knee_frequency_fraction",  # col 2: fkn
    "knee_time_fraction",  # col 3: tkn
    "sweep_time",  # col 4: tsw
    "relative_amplitude",  # col 5: ram
    "time_before_knee",  # col 6: t1bk
    "time_after_knee",  # col 7: t1ak
    "frequency_before_knee",  # col 8: f1bk
    "frequency_after_knee",  # col 9: f1ak
    "am1_before_knee",  # col 10: a1bk
    "am1_after_knee",  # col 11: a1ak
    "am2_before_knee",  # col 12: a2bk
    "am2_after_knee",  # col 13: a2ak
    "am_time_before_knee",  # col 14: t1bkam
    "am_time_after_knee",  # col 15: t1akam
)

# MATLAB twitstruct scalar/vector field names -> TwitStruct attribute names.
_MAT_TO_MODEL = {
    "fc": "center_frequency",
    "IPI": "inter_phrase_interval",
    "f2f1": "harmonic_ratio",
    "A2A1": "harmonic_amplitude_ratio",
    "bwv": "phrase_bandwidths",
    "bw": "mean_phrase_bandwidth",
    "nphr": "phrase_count",
    "tsw": "phrase_sweep_times",
    "tkn": "knee_time_fractions",
    "fkn": "knee_frequency_fractions",
    "tphr": "mean_phrase_sweep_time",
    "tknee": "mean_knee_time_fraction",
    "fknee": "mean_knee_frequency_fraction",
    "dur": "duration",
    "snr": "signal_to_noise_ratio",
    "bpbw": "noise_bandwidth",
    "seed": "seed",
    "rAttn": "harmonic_attenuation",
    "cont": "multi_phrase_source",
    "phr1": "single_phrase_source",
    "mnk": "structure_source",
    "C_cell": "phrase",
    "opt": "options",
    "rev": "time_reversal",
}


class TestAssetConstants(unittest.TestCase):
    def test_all_assets_exist(self) -> None:
        for path in (TWITTER,):
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file(), f"{path} is not a file")
                self.assertTrue(path.parent.samefile(ASSETS))


class TestLoadTemplate(unittest.TestCase):
    def test_loads_twit_template(self) -> None:
        template = TwitStruct.load()
        self.assertIsInstance(template, TwitStruct)

    def test_missing_asset_raises(self) -> None:
        with self.assertRaises(OSError):
            TwitStruct.load(Path("/nonexistent"))

    def test_twit_key_values(self) -> None:
        twit = TwitStruct.load()
        self.assertEqual(len(twit.phrase), 9)
        self.assertAlmostEqual(twit.center_frequency, 9023.106189474018)
        self.assertAlmostEqual(twit.inter_phrase_interval, 0.12832346580668622)
        self.assertAlmostEqual(twit.mean_phrase_bandwidth, 6942.8098692955255)
        self.assertEqual(twit.multi_phrase_source, 2)
        self.assertEqual(twit.single_phrase_source, 1)
        self.assertEqual(twit.structure_source, "ALL")
        self.assertAlmostEqual(twit.phrase[0].start_frequency, 8451.515655222438)
        self.assertIsInstance(twit.phrase[0].time_before_knee, tuple)
        self.assertIsInstance(twit.phrase[0].time_after_knee, tuple)
        self.assertIsInstance(twit.phrase_sweep_times, tuple)
        self.assertIsInstance(twit.phrase_bandwidths, tuple)
        self.assertEqual(len(twit.phrase[0].time_before_knee), 25)
        self.assertEqual(len(twit.phrase[0].time_after_knee), 10)
        self.assertEqual(len(twit.phrase_sweep_times), 9)
        self.assertEqual(len(twit.phrase_bandwidths), 9)
        self.assertEqual(twit.time_reversal, TimeReversal())

    def test_vectors_are_immutable(self) -> None:
        twit = TwitStruct.load()
        self.assertFalse(hasattr(twit.phrase_sweep_times, "__setitem__"))
        self.assertFalse(hasattr(twit.phrase[0].time_before_knee, "__setitem__"))

    def test_twit_options(self) -> None:
        twit = TwitStruct.load()
        self.assertEqual(
            twit.options,
            CallOptions(
                reverse_order=False, flat_am=False, flat_fm=0, chop=Chop.NORMAL
            ),
        )


class TestTwitStructValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.valid = TwitStruct.load().model_dump()

    def assert_invalid(self, **changes: object) -> None:
        with self.assertRaises(ValidationError):
            TwitStruct.model_validate({**self.valid, **changes})

    def test_scalar_bounds(self) -> None:
        cases: dict[str, object] = {
            "center_frequency": 0,
            "inter_phrase_interval": -1,
            "harmonic_ratio": 0,
            "harmonic_amplitude_ratio": 0,
            "mean_phrase_bandwidth": -1,
            "phrase_count": -1,
            "mean_phrase_sweep_time": 0,
            "mean_knee_time_fraction": -0.1,
            "mean_knee_frequency_fraction": -0.1,
            "duration": -1,
            "signal_to_noise_ratio": -1,
            "noise_bandwidth": -1,
            "seed": 0,
            "harmonic_attenuation": 0.1,
            "multi_phrase_source": 3,
            "single_phrase_source": 4,
            "structure_source": "OTHER",
        }
        for field, value in cases.items():
            with self.subTest(field=field, value=value):
                self.assert_invalid(**{field: value})
        self.assert_invalid(harmonic_amplitude_ratio=1.1)
        self.assert_invalid(mean_knee_time_fraction=1.1)

    def test_legal_boundaries(self) -> None:
        for mean_knee_time_fraction in (0, 1):
            with self.subTest(mean_knee_time_fraction=mean_knee_time_fraction):
                model = TwitStruct.model_validate(
                    {
                        **self.valid,
                        "phrase_count": 0,
                        "mean_phrase_bandwidth": 0,
                        "mean_knee_frequency_fraction": 0,
                        "mean_knee_time_fraction": mean_knee_time_fraction,
                        "harmonic_amplitude_ratio": 1,
                    }
                )
                self.assertEqual(model.phrase_count, 0)
                self.assertEqual(model.mean_knee_time_fraction, mean_knee_time_fraction)

    def test_vector_element_bounds(self) -> None:
        for field, bad_value in (
            ("phrase_bandwidths", 0),
            ("phrase_sweep_times", 0),
            ("knee_time_fractions", -0.1),
            ("knee_time_fractions", 1.1),
            ("knee_frequency_fractions", -0.1),
            ("knee_frequency_fractions", 1.1),
        ):
            with self.subTest(field=field, value=bad_value):
                values = list(self.valid[field])
                values[0] = bad_value
                self.assert_invalid(**{field: values})

    def test_structure(self) -> None:
        self.assert_invalid(phrase={})
        self.assert_invalid(phrase_bandwidths=())
        self.assert_invalid(phrase_sweep_times=self.valid["phrase_sweep_times"][:-1])
        phrases = dict(self.valid["phrase"])
        phrases[10] = phrases.pop(8)
        self.assert_invalid(phrase=phrases)


class TestPhraseValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.valid = TwitStruct.load().phrase[0].model_dump()

    def assert_invalid(self, **changes: object) -> None:
        with self.assertRaises(ValidationError):
            Phrase.model_validate({**self.valid, **changes})

    def test_scalar_bounds(self) -> None:
        for field, value in {
            "start_frequency": 0,
            "end_frequency": 0,
            "knee_frequency_fraction": -0.1,
            "knee_time_fraction": -0.1,
            "sweep_time": 0,
            "relative_amplitude": -0.1,
        }.items():
            with self.subTest(field=field, value=value):
                self.assert_invalid(**{field: value})

    def test_legal_scalar_boundaries(self) -> None:
        for field in (
            "knee_frequency_fraction",
            "knee_time_fraction",
            "relative_amplitude",
        ):
            values = (0, 1)
            for value in values:
                with self.subTest(field=field, value=value):
                    phrase = Phrase.model_validate({**self.valid, field: value})
                    self.assertEqual(getattr(phrase, field), value)

    def test_contour_lengths(self) -> None:
        before = (
            "time_before_knee",
            "frequency_before_knee",
            "am1_before_knee",
            "am2_before_knee",
            "am_time_before_knee",
        )
        after = (
            "time_after_knee",
            "frequency_after_knee",
            "am1_after_knee",
            "am2_after_knee",
            "am_time_after_knee",
        )
        for field, expected_length in (
            *((field, 25) for field in before),
            *((field, 10) for field in after),
        ):
            values = self.valid[field]
            self.assertEqual(len(values), expected_length)
            for bad_values in (values[:-1], (*values, values[-1])):
                with self.subTest(field=field, length=len(bad_values)):
                    self.assert_invalid(**{field: bad_values})

    def test_normalized_contour_bounds(self) -> None:
        fields = (
            "time_before_knee",
            "time_after_knee",
            "am1_before_knee",
            "am1_after_knee",
            "am2_before_knee",
            "am2_after_knee",
            "am_time_before_knee",
            "am_time_after_knee",
        )
        for field in fields:
            for bad_value in (-0.1, 1.1):
                values = list(self.valid[field])
                values[0] = bad_value
                with self.subTest(field=field, value=bad_value):
                    self.assert_invalid(**{field: values})

    def test_non_finite_values(self) -> None:
        for field in ("start_frequency", "frequency_before_knee", "am1_before_knee"):
            for bad_value in (float("nan"), float("inf"), float("-inf")):
                value: object = bad_value
                if isinstance(self.valid[field], tuple):
                    values = list(self.valid[field])
                    values[0] = bad_value
                    value = values
                with self.subTest(field=field, value=bad_value):
                    self.assert_invalid(**{field: value})

    def test_non_monotonic_fm_contours_are_allowed(self) -> None:
        frequency_before_knee = tuple(reversed(self.valid["frequency_before_knee"]))
        phrase = Phrase.model_validate(
            {**self.valid, "frequency_before_knee": frequency_before_knee}
        )
        self.assertEqual(phrase.frequency_before_knee, frequency_before_knee)


class TestFidelity(unittest.TestCase):
    """Loaded template must match the raw .mat content field by field."""

    def _load_raw(self) -> Any:
        data = loadmat(MAT_PATH, squeeze_me=True)
        return data["twitstruct"]

    def test_twitstruct_fidelity(self) -> None:
        raw = self._load_raw()
        model = TwitStruct.load()
        self.assertIsInstance(model, TwitStruct)
        for field in raw.dtype.names:
            raw_value = raw[field]
            model_value = getattr(model, _MAT_TO_MODEL[field])
            if field == "opt":
                opt_struct = np.asarray(raw_value).item()
                self.assertEqual(
                    model_value.reverse_order,
                    int(np.asarray(opt_struct["ord"]).ravel()[0]),
                )
                self.assertEqual(
                    model_value.flat_am, int(np.asarray(opt_struct["noam"]).ravel()[0])
                )
                self.assertEqual(
                    model_value.flat_fm, int(np.asarray(opt_struct["nofm"]).ravel()[0])
                )
                self.assertEqual(
                    model_value.chop, int(np.asarray(opt_struct["chop"]).ravel()[0])
                )
            elif field == "rev":
                expected = np.asarray(raw_value).item().ravel()
                self.assertEqual(model_value.fm, bool(expected[0]))
                self.assertEqual(model_value.am, bool(expected[1]))
            elif field == "C_cell":
                raw_cell = np.asarray(raw_value).item()
                self.assertEqual(len(model_value), raw_cell.shape[0])
                for i, phrase in model_value.items():
                    self.assertIsInstance(phrase, Phrase)
                    for col, fname in enumerate(_MAT_TO_MODEL_PHRASE):
                        expected = np.asarray(raw_cell[i, col])
                        actual = getattr(phrase, fname)
                        if isinstance(actual, tuple):
                            self.assertTrue(
                                np.array_equal(
                                    np.asarray(actual), expected.astype(np.float64)
                                ),
                                f"TWIT.C_cell[{i}].{fname}",
                            )
                        else:
                            self.assertEqual(
                                actual, expected.item(), f"TWIT.C_cell[{i}].{fname}"
                            )
            elif isinstance(model_value, tuple):
                expected = np.asarray(raw_value).item()
                self.assertEqual(
                    (len(model_value),),
                    expected.shape,
                    f"TWIT.{field}: shape mismatch",
                )
                self.assertTrue(
                    np.array_equal(
                        np.asarray(model_value), expected.astype(np.float64)
                    ),
                    f"TWIT.{field}",
                )
            else:
                expected = np.asarray(raw_value).item()
                self.assertEqual(model_value, expected, f"TWIT.{field}")


if __name__ == "__main__":
    unittest.main()
