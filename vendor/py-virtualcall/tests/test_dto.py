# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportArgumentType=false
"""Tests for the TwitterDTO user-adjustable parameters model."""

import unittest

from pydantic import ValidationError

from py_virtualcall.template import Chop, TimeReversal, TwitterDTO

# Valid parameter set, values near the bundled TWIT.mat template defaults.
VALID = {
    "center_frequency": 9023.1,
    "inter_phrase_interval": 0.1283,
    "mean_phrase_sweep_time": 0.0443,
    "phrase_count": 9,
    "harmonic_attenuation": -22.09,
    "signal_to_noise_ratio": 100.0,
    "noise_bandwidth": 100.0,
    "seed": 5006,
    "mean_phrase_bandwidth": 6942.8,
    "mean_knee_frequency_fraction": 0.3658,
    "mean_knee_time_fraction": 0.7644,
}


class TestTwitterDTO(unittest.TestCase):
    def test_valid_construction(self) -> None:
        dto = TwitterDTO(**VALID)
        self.assertEqual(dto.center_frequency, 9023.1)
        self.assertEqual(dto.phrase_count, 9)
        self.assertEqual(dto.time_reversal, TimeReversal())
        self.assertEqual(dto.reverse_order, False)
        self.assertEqual(dto.flat_am, False)
        self.assertEqual(dto.chop, Chop.NORMAL)

    def test_modulation_options(self) -> None:
        for time_reversal in (
            TimeReversal(fm=True),
            TimeReversal(am=True),
        ):
            with self.subTest(time_reversal=time_reversal):
                dto = TwitterDTO(
                    **VALID,
                    time_reversal=time_reversal,
                    reverse_order=True,
                    flat_am=True,
                    chop=Chop.AFTER_KNEE,
                )
                self.assertEqual(dto.time_reversal, time_reversal)
                self.assertEqual(dto.reverse_order, True)
                self.assertEqual(dto.flat_am, True)
                self.assertEqual(dto.chop, Chop.AFTER_KNEE)

    def test_missing_required_field_raises(self) -> None:
        required = (
            "center_frequency",
            "inter_phrase_interval",
            "mean_phrase_sweep_time",
            "phrase_count",
            "harmonic_attenuation",
            "signal_to_noise_ratio",
            "noise_bandwidth",
            "seed",
            "mean_phrase_bandwidth",
            "mean_knee_frequency_fraction",
            "mean_knee_time_fraction",
        )
        for field in required:
            kwargs = {k: v for k, v in VALID.items() if k != field}
            with self.subTest(field=field), self.assertRaises(ValidationError):
                TwitterDTO(**kwargs)

    def test_field_bounds_raise(self) -> None:
        cases: dict[str, object] = {
            "center_frequency": 0.0,
            "inter_phrase_interval": -0.1,
            "mean_phrase_sweep_time": 0.0,
            "phrase_count": -1,
            "harmonic_attenuation": 0.1,
            "signal_to_noise_ratio": -1.0,
            "noise_bandwidth": -1.0,
            "seed": 0,
            "mean_phrase_bandwidth": -1.0,
            "mean_knee_frequency_fraction": -0.1,
            "mean_knee_time_fraction": -0.1,
        }
        for field, bad in cases.items():
            with (
                self.subTest(field=field, value=bad),
                self.assertRaises(ValidationError),
            ):
                TwitterDTO(**{**VALID, field: bad})
        with self.assertRaises(ValidationError):
            TwitterDTO(**{**VALID, "mean_knee_time_fraction": 1.1})

    def test_chop_accepts_int(self) -> None:
        self.assertIs(TwitterDTO(**VALID, chop=1).chop, Chop.BEFORE_KNEE)
        self.assertIs(TwitterDTO(**VALID, chop=2).chop, Chop.AFTER_KNEE)

    def test_option_values_raise(self) -> None:
        with self.assertRaises(ValidationError):
            TwitterDTO(**VALID, time_reversal={"fm": False, "am": 2})
        with self.assertRaises(ValidationError):
            TwitterDTO(**VALID, time_reversal=(False, False))
        with self.assertRaises(ValidationError):
            TwitterDTO(**VALID, reverse_order=2)
        with self.assertRaises(ValidationError):
            TwitterDTO(**VALID, flat_am=2)
        with self.assertRaises(ValidationError):
            TwitterDTO(**VALID, chop=3)

    def test_frozen(self) -> None:
        dto = TwitterDTO(**VALID)
        with self.assertRaises(ValidationError):
            dto.center_frequency = 10000.0
        with self.assertRaises(ValidationError):
            dto.time_reversal.fm = True

    def test_sample_rate_defaults_to_48000(self) -> None:
        self.assertEqual(TwitterDTO(**VALID).sample_rate, 48_000)
        self.assertEqual(TwitterDTO(**VALID, sample_rate=44_100).sample_rate, 44_100)

    def test_sample_rate_must_be_positive(self) -> None:
        with self.assertRaises(ValidationError):
            TwitterDTO(**VALID, sample_rate=0)
        with self.assertRaises(ValidationError):
            TwitterDTO(**VALID, sample_rate=-1)


if __name__ == "__main__":
    unittest.main()
