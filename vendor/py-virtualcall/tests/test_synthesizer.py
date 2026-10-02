# pyright: reportPrivateUsage=false
"""Tests for mutable, copy-on-stage twitter synthesis state."""

import unittest
from math import floor
from typing import Literal
from unittest import mock

import numpy as np
from pydantic import ValidationError

from py_virtualcall.synthesizer import (
    PhraseState,
    TwitterSignal,
    TwitterState,
    TwitterSynthesizer,
)
from py_virtualcall.synthesizer.twitter_synthesizer import (
    _am_envelopes,
    _band_limit,
    _make_am,
    _make_fm,
    _oscillate,
)
from py_virtualcall.template import (
    CallOptions,
    Chop,
    TimeReversal,
    TwitStruct,
    TwitterDTO,
)


def _default_dto(template: TwitStruct) -> TwitterDTO:
    return TwitterDTO(
        center_frequency=template.center_frequency,
        inter_phrase_interval=template.inter_phrase_interval,
        mean_phrase_sweep_time=template.mean_phrase_sweep_time,
        phrase_count=template.phrase_count,
        harmonic_attenuation=template.harmonic_attenuation,
        signal_to_noise_ratio=template.signal_to_noise_ratio,
        noise_bandwidth=template.noise_bandwidth,
        seed=template.seed,
        mean_phrase_bandwidth=template.mean_phrase_bandwidth,
        mean_knee_frequency_fraction=template.mean_knee_frequency_fraction,
        mean_knee_time_fraction=template.mean_knee_time_fraction,
        time_reversal=template.time_reversal,
        reverse_order=template.options.reverse_order,
        flat_am=template.options.flat_am,
        chop=template.options.chop,
    )


# --- Step 5 fixtures: tiny synthetic state and an independent reference --------


def _tiny_phrase(
    start_frequency: float = 1000.0,
    end_frequency: float = 2000.0,
    relative_amplitude: float = 1.0,
) -> PhraseState:
    """One phrase with small, well-behaved contours for hand-checked synthesis."""
    return PhraseState(
        start_frequency=start_frequency,
        end_frequency=end_frequency,
        knee_frequency_fraction=0.5,
        knee_time_fraction=0.5,
        sweep_time=0.01,
        relative_amplitude=relative_amplitude,
        time_before_knee=np.linspace(0.05, 0.45, 25),
        time_after_knee=np.linspace(0.05, 0.45, 10),
        frequency_before_knee=np.linspace(0.3, 0.6, 25),
        frequency_after_knee=np.linspace(0.4, 0.7, 10),
        am1_before_knee=np.full(25, 0.8),
        am1_after_knee=np.full(10, 0.2),
        am2_before_knee=np.full(25, 0.8),
        am2_after_knee=np.full(10, 0.2),
        am_time_before_knee=np.linspace(0.05, 0.45, 25),
        am_time_after_knee=np.linspace(0.05, 0.45, 10),
    )


def _tiny_state(
    nphr: int = 1,
    ipi: float = 0.02,
    sr: int = 8000,
    phrases: list[PhraseState] | None = None,
) -> TwitterState:
    if phrases is None:
        phrases = [_tiny_phrase() for _ in range(nphr)]
    return TwitterState(
        center_frequency=5000.0,
        sample_rate=sr,
        inter_phrase_interval=ipi,
        harmonic_ratio=2,
        harmonic_amplitude_ratio=0.5,
        phrase_bandwidths=np.full(nphr, 1000.0),
        mean_phrase_bandwidth=1000.0,
        phrase={index: phrase for index, phrase in enumerate(phrases)},
        phrase_count=nphr,
        phrase_sweep_times=np.array([phrase.sweep_time for phrase in phrases]),
        knee_time_fractions=np.array([phrase.knee_time_fraction for phrase in phrases]),
        knee_frequency_fractions=np.array(
            [phrase.knee_frequency_fraction for phrase in phrases]
        ),
        mean_phrase_sweep_time=0.01,
        mean_knee_time_fraction=0.5,
        mean_knee_frequency_fraction=0.5,
        duration=0.0,
        signal_to_noise_ratio=100.0,
        noise_bandwidth=100.0,
        seed=1,
        harmonic_attenuation=-6.0,
        multi_phrase_source=2,
        single_phrase_source=1,
        options=CallOptions(
            reverse_order=False, flat_am=False, flat_fm=0, chop=Chop.NORMAL
        ),
        time_reversal=TimeReversal(),
        structure_source="ALL",
    )


def _reference_contour(length: int, t1: np.ndarray, values: np.ndarray) -> np.ndarray:
    """MATLAB A(i:j)/F(i:j) inclusive-slice piecewise linear interpolation."""
    x = np.linspace(0, 1, length)
    result = np.zeros(length)
    for k in range(len(t1) - 1):
        i = int(np.argmin(np.abs(x - t1[k])))
        j = int(np.argmin(np.abs(x - t1[k + 1])))
        result[i : j + 1] = values[k] + (
            (values[k + 1] - values[k]) / (t1[k + 1] - t1[k])
        ) * (x[i : j + 1] - x[i])
    return result


def _reference_make_am(
    length: int,
    knee_time_fraction: float,
    time_before_knee: np.ndarray,
    time_after_knee: np.ndarray,
    amplitude_before_knee: np.ndarray,
    amplitude_after_knee: np.ndarray,
) -> np.ndarray:
    knee_times_before = time_before_knee * knee_time_fraction
    knee_times_after = knee_time_fraction + time_after_knee * (1 - knee_time_fraction)
    return _reference_contour(
        length,
        np.concatenate(([0.0], knee_times_before, knee_times_after, [1.0])),
        np.concatenate(([0.0], amplitude_before_knee, amplitude_after_knee, [0.0])),
    )


def _reference_make_fm(
    length: int,
    knee_time_fraction: float,
    knee_frequency_fraction: float,
    time_before_knee: np.ndarray,
    time_after_knee: np.ndarray,
    frequency_before_knee: np.ndarray,
    frequency_after_knee: np.ndarray,
    start_frequency: float,
    end_frequency: float,
) -> np.ndarray:
    knee_times_before = time_before_knee * knee_time_fraction
    knee_times_after = knee_time_fraction + time_after_knee * (1 - knee_time_fraction)
    t1 = np.concatenate((knee_times_before, knee_times_after))
    knee_frequency = start_frequency + (end_frequency - start_frequency) * (
        knee_frequency_fraction
    )
    f1 = np.concatenate(
        (
            start_frequency
            + (knee_frequency - start_frequency) * frequency_before_knee,
            knee_frequency + (end_frequency - knee_frequency) * frequency_after_knee,
        )
    )
    keep = np.ones(len(f1), dtype=bool)
    keep[1:] = ~(np.diff(f1) < 0)
    t1, f1 = t1[keep], f1[keep]
    start_slope = (f1[1] - f1[0]) / (t1[1] - t1[0])
    end_slope = (f1[-1] - f1[-2]) / (t1[-1] - t1[-2])
    frequency_at_start = f1[0] - start_slope * t1[0]
    frequency_at_end = f1[-1] + end_slope * (1 - t1[-1])
    return _reference_contour(
        length,
        np.concatenate(([0.0], t1, [1.0])),
        np.concatenate(([frequency_at_start], f1, [frequency_at_end])),
    )


def _reference_phrase_signal(
    state: TwitterState, sample_rate: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent translation of the MATLAB per-phrase synthesis loop."""
    sample_count = floor(sample_rate * state.duration + 0.5)
    combined = np.zeros(sample_count)
    fundamental = np.zeros(sample_count)
    harmonic = np.zeros(sample_count)
    max_relative_amplitude = max(
        state.phrase[i].relative_amplitude for i in range(state.phrase_count)
    )
    phrase_order = (
        range(state.phrase_count - 1, -1, -1)
        if state.options.reverse_order
        else range(state.phrase_count)
    )
    onset_samples = floor(sample_rate / 1000)
    offset_samples = floor(sample_rate / 1000)
    first_sweep_time = state.phrase_sweep_times[0]
    for index in range(state.phrase_count):
        phrase = state.phrase[phrase_order[index]]
        scaled_amplitude = phrase.relative_amplitude / max_relative_amplitude
        phrase_offset = index * state.inter_phrase_interval
        start_index = (
            floor(
                sample_rate
                * (0.5 * first_sweep_time + phrase_offset - 0.5 * phrase.sweep_time)
            )
            - 1
        )
        stop_index = floor(sample_rate * (phrase_offset + phrase.sweep_time)) - 1
        start_index = max(start_index, 0)
        stop_index = min(stop_index, sample_count - 1)
        length = stop_index - start_index + 1
        frequency_before_normalized = (
            phrase.frequency_before_knee - phrase.frequency_before_knee.min()
        ) / (phrase.frequency_before_knee.max() - phrase.frequency_before_knee.min())
        frequency_after_normalized = (
            phrase.frequency_after_knee - phrase.frequency_after_knee.min()
        ) / (phrase.frequency_after_knee.max() - phrase.frequency_after_knee.min())
        am1_before, am1_after = (
            phrase.am1_before_knee.copy(),
            phrase.am1_after_knee.copy(),
        )
        am2_before, am2_after = (
            phrase.am2_before_knee.copy(),
            phrase.am2_after_knee.copy(),
        )
        if state.options.chop is Chop.BEFORE_KNEE:
            am1_after[:] = 0.0
            am2_after[:] = 0.0
        elif state.options.chop is Chop.AFTER_KNEE:
            am1_before[:] = 0.0
            am2_before[:] = 0.0
        if state.options.flat_am:
            am1 = np.ones(length)
            am2 = np.ones(length)
        else:
            am1 = _reference_make_am(
                length,
                phrase.knee_time_fraction,
                phrase.am_time_before_knee,
                phrase.am_time_after_knee,
                am1_before,
                am1_after,
            )
            am2 = _reference_make_am(
                length,
                phrase.knee_time_fraction,
                phrase.am_time_before_knee,
                phrase.am_time_after_knee,
                am2_before,
                am2_after,
            )
        am1 = scaled_amplitude * (am1 / np.max(np.abs(am1)))
        am2 = scaled_amplitude * (am2 / np.max(np.abs(am2)))
        am1_anchor, am2_anchor = am1[onset_samples], am2[onset_samples]
        am1[: onset_samples + 1] = (
            am1_anchor * np.arange(onset_samples + 1) / onset_samples
        )
        am2[: onset_samples + 1] = (
            am2_anchor * np.arange(onset_samples + 1) / onset_samples
        )
        am1_anchor, am2_anchor = am1[-offset_samples - 1], am2[-offset_samples - 1]
        am1[-offset_samples - 1 :] = (
            am1_anchor * np.arange(offset_samples, -1, -1) / offset_samples
        )
        am2[-offset_samples - 1 :] = (
            am2_anchor * np.arange(offset_samples, -1, -1) / offset_samples
        )
        fundamental_frequency = _reference_make_fm(
            length,
            phrase.knee_time_fraction,
            phrase.knee_frequency_fraction,
            phrase.time_before_knee,
            phrase.time_after_knee,
            frequency_before_normalized,
            frequency_after_normalized,
            phrase.start_frequency,
            phrase.end_frequency,
        )
        harmonic_frequency = state.harmonic_ratio * fundamental_frequency
        am2[harmonic_frequency > sample_rate / 2] = 0
        if state.time_reversal.fm:
            fundamental_frequency = fundamental_frequency[::-1].copy()
            harmonic_frequency = harmonic_frequency[::-1].copy()
        if state.time_reversal.am:
            am1 = am1[::-1].copy()
            am2 = am2[::-1].copy()
        fundamental_frequency = np.cumsum(fundamental_frequency / sample_rate)
        harmonic_frequency = np.cumsum(harmonic_frequency / sample_rate)
        phrase_fundamental = np.cos(2 * np.pi * fundamental_frequency - np.pi / 2) * am1
        phrase_harmonic = np.cos(2 * np.pi * harmonic_frequency - np.pi / 2) * am2
        combined[start_index : stop_index + 1] = (
            phrase_fundamental + state.harmonic_amplitude_ratio * phrase_harmonic
        ) / (1 + state.harmonic_amplitude_ratio)
        fundamental[start_index : stop_index + 1] = phrase_fundamental
        harmonic[start_index : stop_index + 1] = phrase_harmonic
    return combined, fundamental, harmonic


class TestTwitterState(unittest.TestCase):
    def setUp(self) -> None:
        self.template = TwitStruct.load()
        self.dto = _default_dto(self.template)

    def test_from_template_builds_numpy_state(self) -> None:
        state = TwitterState.from_template(self.template, self.dto)
        self.assertIsInstance(state, TwitterState)
        self.assertIsInstance(state.phrase[0], PhraseState)
        for array in (
            state.phrase_bandwidths,
            state.knee_frequency_fractions,
            state.knee_time_fractions,
            state.phrase_sweep_times,
        ):
            self.assertEqual(array.dtype, np.float64)
            self.assertEqual(array.shape, (9,))
            self.assertTrue(array.flags.c_contiguous)
        self.assertEqual(state.phrase[0].time_before_knee.shape, (25,))
        self.assertEqual(state.phrase[0].time_after_knee.shape, (10,))

    def test_from_template_applies_dto(self) -> None:
        dto = self.dto.model_copy(
            update={
                "center_frequency": 10_000.0,
                "harmonic_attenuation": -20.0,
                "reverse_order": True,
                "flat_am": True,
                "chop": Chop.AFTER_KNEE,
            }
        )
        state = TwitterState.from_template(self.template, dto)
        self.assertEqual(state.center_frequency, 10_000.0)
        self.assertAlmostEqual(state.harmonic_amplitude_ratio, 0.1)
        self.assertTrue(state.options.reverse_order)
        self.assertTrue(state.options.flat_am)
        self.assertIs(state.options.chop, Chop.AFTER_KNEE)

    def test_state_is_mutable(self) -> None:
        state = TwitterState.from_template(self.template, self.dto)
        state.center_frequency = 10_000.0
        state.phrase_bandwidths[0] = 0.0
        self.assertEqual(state.center_frequency, 10_000.0)
        self.assertEqual(state.phrase_bandwidths[0], 0.0)

    def test_clone_deep_copies_all_arrays(self) -> None:
        state = TwitterState.from_template(self.template, self.dto)
        clone = state.clone()
        self.assertIsNot(clone, state)
        for name in (
            "phrase_bandwidths",
            "knee_frequency_fractions",
            "knee_time_fractions",
            "phrase_sweep_times",
        ):
            self.assertFalse(
                np.shares_memory(getattr(clone, name), getattr(state, name))
            )
        for index in state.phrase:
            self.assertIsNot(clone.phrase[index], state.phrase[index])
            for name in (
                "time_before_knee",
                "time_after_knee",
                "frequency_before_knee",
                "frequency_after_knee",
                "am1_before_knee",
                "am1_after_knee",
                "am2_before_knee",
                "am2_after_knee",
                "am_time_before_knee",
                "am_time_after_knee",
            ):
                self.assertFalse(
                    np.shares_memory(
                        getattr(clone.phrase[index], name),
                        getattr(state.phrase[index], name),
                    )
                )

    def test_rejects_invalid_array_shape_and_values(self) -> None:
        data = {
            **TwitterState.from_template(self.template, self.dto).model_dump(),
            "phrase_bandwidths": np.ones((3, 3)),
        }
        with self.assertRaises(ValidationError):
            TwitterState.model_validate(data)
        data["phrase_bandwidths"] = np.array(
            [*self.template.phrase_bandwidths[:-1], np.nan]
        )
        with self.assertRaises(ValidationError):
            TwitterState.model_validate(data)


class TestTwitterSynthesizer(unittest.TestCase):
    def setUp(self) -> None:
        self.template = TwitStruct.load()
        self.synthesizer = TwitterSynthesizer(self.template)
        self.dto = _default_dto(self.template)

    def test_prepare_returns_state(self) -> None:
        self.assertIsInstance(self.synthesizer.prepare(self.dto), TwitterState)

    def test_default_values_preserve_template_vectors(self) -> None:
        state = self.synthesizer.scale_mean_gain(self.synthesizer.prepare(self.dto))
        np.testing.assert_allclose(
            state.phrase_bandwidths, self.template.phrase_bandwidths
        )
        np.testing.assert_allclose(
            state.knee_frequency_fractions, self.template.knee_frequency_fractions
        )
        np.testing.assert_allclose(
            state.knee_time_fractions, self.template.knee_time_fractions
        )
        np.testing.assert_allclose(
            state.phrase_sweep_times, self.template.phrase_sweep_times
        )

    def test_scales_each_vector_with_matlab_formula(self) -> None:
        dto = self.dto.model_copy(
            update={
                "mean_phrase_bandwidth": 4_000.0,
                "mean_knee_frequency_fraction": 0.5,
                "mean_knee_time_fraction": 0.6,
                "mean_phrase_sweep_time": 0.05,
            }
        )
        state = self.synthesizer.scale_mean_gain(self.synthesizer.prepare(dto))
        np.testing.assert_allclose(
            state.phrase_bandwidths,
            np.asarray(self.template.phrase_bandwidths)
            * dto.mean_phrase_bandwidth
            / self.template.phrase_bandwidths[4],
        )
        np.testing.assert_allclose(
            state.knee_frequency_fractions,
            np.asarray(self.template.knee_frequency_fractions)
            * dto.mean_knee_frequency_fraction
            / np.mean(self.template.knee_frequency_fractions),
        )
        np.testing.assert_allclose(
            state.knee_time_fractions,
            np.asarray(self.template.knee_time_fractions)
            * dto.mean_knee_time_fraction
            / np.mean(self.template.knee_time_fractions),
        )
        np.testing.assert_allclose(
            state.phrase_sweep_times,
            np.asarray(self.template.phrase_sweep_times)
            * dto.mean_phrase_sweep_time
            / np.mean(self.template.phrase_sweep_times),
        )

    def test_bandwidth_uses_middle_phrase_not_mean(self) -> None:
        state = self.synthesizer.scale_mean_gain(
            self.synthesizer.prepare(
                self.dto.model_copy(update={"mean_phrase_bandwidth": 4_000.0})
            )
        )
        self.assertAlmostEqual(state.phrase_bandwidths[4], 4_000.0)
        self.assertNotAlmostEqual(float(np.mean(state.phrase_bandwidths)), 4_000.0)

    def test_stage_does_not_modify_input(self) -> None:
        source = self.synthesizer.prepare(
            self.dto.model_copy(
                update={
                    "mean_phrase_bandwidth": 4_000.0,
                    "mean_knee_frequency_fraction": 0.5,
                }
            )
        )
        original_bandwidths = source.phrase_bandwidths.copy()
        original_knee_fractions = source.knee_frequency_fractions.copy()
        original_phrase_knee = source.phrase[0].knee_frequency_fraction
        result = self.synthesizer.scale_mean_gain(source)
        self.assertIsNot(result, source)
        np.testing.assert_array_equal(source.phrase_bandwidths, original_bandwidths)
        np.testing.assert_array_equal(
            source.knee_frequency_fractions, original_knee_fractions
        )
        self.assertEqual(source.phrase[0].knee_frequency_fraction, original_phrase_knee)
        self.assertFalse(
            np.shares_memory(result.phrase_bandwidths, source.phrase_bandwidths)
        )

    def test_phrase_values_stay_in_sync(self) -> None:
        state = self.synthesizer.scale_mean_gain(
            self.synthesizer.prepare(
                self.dto.model_copy(
                    update={
                        "mean_knee_frequency_fraction": 0.5,
                        "mean_knee_time_fraction": 0.6,
                        "mean_phrase_sweep_time": 0.05,
                    }
                )
            )
        )
        for index, phrase in state.phrase.items():
            self.assertEqual(
                (
                    phrase.knee_frequency_fraction,
                    phrase.knee_time_fraction,
                    phrase.sweep_time,
                ),
                (
                    state.knee_frequency_fractions[index],
                    state.knee_time_fractions[index],
                    state.phrase_sweep_times[index],
                ),
            )

    def test_zero_targets_produce_zero_vectors(self) -> None:
        state = self.synthesizer.scale_mean_gain(
            self.synthesizer.prepare(
                self.dto.model_copy(
                    update={
                        "mean_phrase_bandwidth": 0.0,
                        "mean_knee_frequency_fraction": 0.0,
                        "mean_knee_time_fraction": 0.0,
                    }
                )
            )
        )
        np.testing.assert_array_equal(state.phrase_bandwidths, np.zeros(9))
        np.testing.assert_array_equal(state.knee_frequency_fractions, np.zeros(9))
        np.testing.assert_array_equal(state.knee_time_fractions, np.zeros(9))

    def test_rejects_non_nine_phrase_template(self) -> None:
        data = self.template.model_dump()
        data["phrase"] = {index: data["phrase"][index] for index in range(3)}
        for field in (
            "phrase_bandwidths",
            "knee_frequency_fractions",
            "knee_time_fractions",
            "phrase_sweep_times",
        ):
            data[field] = data[field][:3]
        short_template = TwitStruct.model_validate(data)
        with self.assertRaisesRegex(ValueError, "exactly 9"):
            TwitterSynthesizer(short_template)

    def test_rejects_zero_scaling_denominators(self) -> None:
        cases = (
            ("phrase_bandwidths", 4, "bandwidth"),
            ("knee_frequency_fractions", None, "frequency"),
            ("knee_time_fractions", None, "time"),
        )
        for field, index, message in cases:
            with self.subTest(field=field):
                state = self.synthesizer.prepare(self.dto)
                values = getattr(state, field)
                if index is None:
                    values[:] = 0.0
                else:
                    values[index] = 0.0
                with self.assertRaisesRegex(ValueError, message):
                    self.synthesizer.scale_mean_gain(state)

    def test_resets_phrase_count_with_matlab_indices(self) -> None:
        expected = {
            2: (0, 8),
            3: (0, 4, 8),
            4: (0, 3, 5, 8),
            5: (0, 3, 4, 5, 8),
            6: (0, 2, 3, 5, 6, 8),
            7: (0, 2, 3, 4, 5, 6, 8),
            8: (0, 1, 2, 3, 5, 6, 7, 8),
            9: tuple(range(9)),
        }
        for phrase_count, indices in expected.items():
            with self.subTest(phrase_count=phrase_count):
                source = self.synthesizer.prepare(
                    self.dto.model_copy(update={"phrase_count": phrase_count})
                )
                result = self.synthesizer.reset_phrase_count(source)
                np.testing.assert_array_equal(
                    result.phrase_bandwidths,
                    np.asarray(source.phrase_bandwidths)[list(indices)],
                )
                self.assertEqual(
                    [phrase.start_frequency for phrase in result.phrase.values()],
                    [source.phrase[index].start_frequency for index in indices],
                )
                self.assertEqual(tuple(result.phrase), tuple(range(phrase_count)))
                for name in (
                    "phrase_bandwidths",
                    "knee_frequency_fractions",
                    "knee_time_fractions",
                    "phrase_sweep_times",
                ):
                    self.assertEqual(getattr(result, name).shape, (phrase_count,))

    def test_single_phrase_uses_phr1_selection(self) -> None:
        cases: tuple[tuple[Literal[1, 2, 3], int], ...] = ((1, 0), (2, 4), (3, 8))
        for single_phrase_source, source_index in cases:
            with self.subTest(single_phrase_source=single_phrase_source):
                source = self.synthesizer.prepare(
                    self.dto.model_copy(update={"phrase_count": 1})
                )
                source.single_phrase_source = single_phrase_source
                result = self.synthesizer.reset_phrase_count(source)
                self.assertEqual(
                    result.phrase_bandwidths[0],
                    source.phrase_bandwidths[source_index],
                )
                self.assertEqual(
                    result.phrase[0].start_frequency,
                    source.phrase[source_index].start_frequency,
                )

    def test_expands_with_independent_middle_phrases(self) -> None:
        source = self.synthesizer.prepare(
            self.dto.model_copy(update={"phrase_count": 12})
        )
        result = self.synthesizer.reset_phrase_count(source)
        expected_indices = (0, 1, 2, 3, 4, 4, 4, 4, 5, 6, 7, 8)
        np.testing.assert_array_equal(
            result.phrase_bandwidths,
            np.asarray(source.phrase_bandwidths)[list(expected_indices)],
        )
        for left, right in ((4, 5), (5, 6), (6, 7)):
            self.assertIsNot(result.phrase[left], result.phrase[right])
            self.assertFalse(
                np.shares_memory(
                    result.phrase[left].time_before_knee,
                    result.phrase[right].time_before_knee,
                )
            )

    def test_phrase_count_reset_does_not_modify_input(self) -> None:
        source = self.synthesizer.prepare(
            self.dto.model_copy(update={"phrase_count": 3})
        )
        original_bandwidths = source.phrase_bandwidths.copy()
        result = self.synthesizer.reset_phrase_count(source)
        self.assertEqual(len(source.phrase), 9)
        np.testing.assert_array_equal(source.phrase_bandwidths, original_bandwidths)
        self.assertFalse(
            np.shares_memory(result.phrase_bandwidths, source.phrase_bandwidths)
        )
        for phrase in result.phrase.values():
            self.assertTrue(
                all(
                    phrase is not source_phrase
                    for source_phrase in source.phrase.values()
                )
            )

    def test_cont_one_preserves_unreset_copy(self) -> None:
        source = self.synthesizer.prepare(
            self.dto.model_copy(update={"phrase_count": 3})
        )
        source.multi_phrase_source = 1
        result = self.synthesizer.reset_phrase_count(source)
        self.assertEqual(len(result.phrase), 9)
        np.testing.assert_array_equal(
            result.phrase_bandwidths, source.phrase_bandwidths
        )
        self.assertFalse(
            np.shares_memory(result.phrase_bandwidths, source.phrase_bandwidths)
        )
        self.assertIsNot(result.phrase[0], source.phrase[0])

    def test_rejects_zero_phrase_count(self) -> None:
        source = self.synthesizer.prepare(
            self.dto.model_copy(update={"phrase_count": 0})
        )
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            self.synthesizer.reset_phrase_count(source)

    def test_updates_start_end_frequency_with_matlab_formula(self) -> None:
        fc = 10_000.0
        source = self.synthesizer.prepare(
            self.dto.model_copy(update={"center_frequency": fc})
        )
        result = self.synthesizer.update_phrase_frequencies(source)
        middle = (source.phrase_count + 1) // 2 - 1  # MATLAB fst(ceil(nphr/2)), 0-based
        center_frequency_shift = fc - 0.5 * (
            source.phrase[middle].start_frequency + source.phrase[middle].end_frequency
        )
        for index, phrase in result.phrase.items():
            phrase_center = 0.5 * (
                source.phrase[index].start_frequency
                + source.phrase[index].end_frequency
            )
            self.assertAlmostEqual(
                phrase.start_frequency,
                phrase_center
                - 0.5 * result.phrase_bandwidths[index]
                + center_frequency_shift,
            )
            self.assertAlmostEqual(
                phrase.end_frequency,
                phrase_center
                + 0.5 * result.phrase_bandwidths[index]
                + center_frequency_shift,
            )

    def test_middle_phrase_centered_at_fc(self) -> None:
        fc = 10_000.0
        result = self.synthesizer.update_phrase_frequencies(
            self.synthesizer.prepare(
                self.dto.model_copy(update={"center_frequency": fc})
            )
        )
        middle = (result.phrase_count + 1) // 2 - 1
        self.assertAlmostEqual(
            0.5
            * (
                result.phrase[middle].start_frequency
                + result.phrase[middle].end_frequency
            ),
            fc,
        )

    def test_bandwidth_preserved_per_phrase(self) -> None:
        result = self.synthesizer.update_phrase_frequencies(
            self.synthesizer.prepare(self.dto)
        )
        for index, phrase in result.phrase.items():
            self.assertAlmostEqual(
                phrase.end_frequency - phrase.start_frequency,
                result.phrase_bandwidths[index],
            )

    def test_uses_ceil_middle_phrase_for_even_count(self) -> None:
        fc = 10_000.0
        source = self.synthesizer.reset_phrase_count(
            self.synthesizer.prepare(
                self.dto.model_copy(update={"phrase_count": 4, "center_frequency": fc})
            )
        )
        result = self.synthesizer.update_phrase_frequencies(source)
        # MATLAB fst(ceil(4/2)) = fst(2) → 0-based index 1 (not floor → 2).
        self.assertAlmostEqual(
            0.5 * (result.phrase[1].start_frequency + result.phrase[1].end_frequency),
            fc,
        )

    def test_phrase_centers_shift_by_common_delta(self) -> None:
        source = self.synthesizer.prepare(
            self.dto.model_copy(update={"center_frequency": 9_500.0})
        )
        result = self.synthesizer.update_phrase_frequencies(source)
        middle = (source.phrase_count + 1) // 2 - 1
        center_frequency_shift = 9_500.0 - 0.5 * (
            source.phrase[middle].start_frequency + source.phrase[middle].end_frequency
        )
        for index, phrase in result.phrase.items():
            original_center = 0.5 * (
                source.phrase[index].start_frequency
                + source.phrase[index].end_frequency
            )
            self.assertAlmostEqual(
                0.5 * (phrase.start_frequency + phrase.end_frequency),
                original_center + center_frequency_shift,
            )

    def test_frequency_update_does_not_modify_input(self) -> None:
        source = self.synthesizer.prepare(
            self.dto.model_copy(update={"center_frequency": 10_000.0})
        )
        original = {
            index: (phrase.start_frequency, phrase.end_frequency)
            for index, phrase in source.phrase.items()
        }
        result = self.synthesizer.update_phrase_frequencies(source)
        self.assertIsNot(result, source)
        for index, phrase in source.phrase.items():
            self.assertEqual(
                (phrase.start_frequency, phrase.end_frequency), original[index]
            )
            self.assertIsNot(result.phrase[index], phrase)

    def test_cont_one_updates_only_first_nphr(self) -> None:
        source = self.synthesizer.prepare(
            self.dto.model_copy(
                update={"phrase_count": 3, "center_frequency": 10_000.0}
            )
        )
        source.multi_phrase_source = 1
        result = self.synthesizer.update_phrase_frequencies(source)
        self.assertEqual(len(result.phrase), 9)
        for index, phrase in result.phrase.items():
            if index < 3:
                self.assertNotEqual(
                    (phrase.start_frequency, phrase.end_frequency),
                    (
                        source.phrase[index].start_frequency,
                        source.phrase[index].end_frequency,
                    ),
                )
            else:
                self.assertEqual(
                    (phrase.start_frequency, phrase.end_frequency),
                    (
                        source.phrase[index].start_frequency,
                        source.phrase[index].end_frequency,
                    ),
                )
        # MATLAB fst(ceil(3/2)) = fst(2) → 0-based index 1.
        self.assertAlmostEqual(
            0.5 * (result.phrase[1].start_frequency + result.phrase[1].end_frequency),
            10_000.0,
        )

    def test_update_rejects_zero_phrase_count(self) -> None:
        source = self.synthesizer.prepare(
            self.dto.model_copy(update={"phrase_count": 0})
        )
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            self.synthesizer.update_phrase_frequencies(source)

    def test_update_rejects_phrase_count_exceeding_available(self) -> None:
        source = self.synthesizer.prepare(
            self.dto.model_copy(update={"phrase_count": 10})
        )
        source.multi_phrase_source = 1
        with self.assertRaisesRegex(ValueError, "exceeds available"):
            self.synthesizer.update_phrase_frequencies(source)

    def test_computes_duration_with_matlab_formula(self) -> None:
        for phrase_count in (9, 5, 1):
            with self.subTest(phrase_count=phrase_count):
                source = self.synthesizer.prepare(
                    self.dto.model_copy(update={"phrase_count": phrase_count})
                )
                if phrase_count != 9:
                    source = self.synthesizer.reset_phrase_count(source)
                result = self.synthesizer.set_duration(source)
                expected = (
                    0.5 * source.phrase_sweep_times[0]
                    + (phrase_count - 1) * source.inter_phrase_interval
                    + 0.5 * source.phrase_sweep_times[phrase_count - 1]
                )
                self.assertAlmostEqual(result.duration, expected)

    def test_duration_overwrites_template_placeholder(self) -> None:
        source = self.synthesizer.prepare(self.dto)
        result = self.synthesizer.set_duration(source)
        self.assertEqual(source.duration, self.template.duration)  # placeholder 1.0
        self.assertNotAlmostEqual(result.duration, source.duration)

    def test_uses_nth_phrase_sweep_time_not_last_template_phrase(self) -> None:
        source = self.synthesizer.prepare(
            self.dto.model_copy(update={"phrase_count": 3})
        )
        source.multi_phrase_source = 1
        result = self.synthesizer.set_duration(source)
        expected = (
            0.5 * source.phrase_sweep_times[0]
            + 2 * source.inter_phrase_interval
            + 0.5 * source.phrase_sweep_times[2]
        )
        self.assertAlmostEqual(result.duration, expected)
        # tsw(3) is the third template phrase, not tsw(9); the two differ.
        self.assertNotAlmostEqual(
            result.duration,
            0.5 * source.phrase_sweep_times[0]
            + 2 * source.inter_phrase_interval
            + 0.5 * source.phrase_sweep_times[8],
        )

    def test_single_phrase_duration_equals_sweep_time(self) -> None:
        source = self.synthesizer.reset_phrase_count(
            self.synthesizer.prepare(self.dto.model_copy(update={"phrase_count": 1}))
        )
        result = self.synthesizer.set_duration(source)
        self.assertAlmostEqual(result.duration, source.phrase_sweep_times[0])

    def test_zero_ipi_halves_only_sweep_times(self) -> None:
        source = self.synthesizer.prepare(
            self.dto.model_copy(update={"inter_phrase_interval": 0.0})
        )
        result = self.synthesizer.set_duration(source)
        self.assertAlmostEqual(
            result.duration,
            0.5 * (source.phrase_sweep_times[0] + source.phrase_sweep_times[8]),
        )

    def test_duration_stage_does_not_modify_input(self) -> None:
        source = self.synthesizer.prepare(
            self.dto.model_copy(update={"inter_phrase_interval": 0.2})
        )
        original_duration = source.duration
        result = self.synthesizer.set_duration(source)
        self.assertIsNot(result, source)
        self.assertEqual(source.duration, original_duration)
        self.assertFalse(
            np.shares_memory(result.phrase_sweep_times, source.phrase_sweep_times)
        )
        self.assertIsNot(result.phrase[0], source.phrase[0])

    def test_duration_rejects_zero_phrase_count(self) -> None:
        source = self.synthesizer.prepare(
            self.dto.model_copy(update={"phrase_count": 0})
        )
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            self.synthesizer.set_duration(source)

    def test_duration_rejects_phrase_count_exceeding_available(self) -> None:
        source = self.synthesizer.prepare(
            self.dto.model_copy(update={"phrase_count": 10})
        )
        source.multi_phrase_source = 1
        with self.assertRaisesRegex(ValueError, "exceeds available"):
            self.synthesizer.set_duration(source)


class TestContourHelpers(unittest.TestCase):
    def test_make_am_piecewise_linear(self) -> None:
        result = _make_am(
            9,
            0.5,
            np.array([0.25, 0.5]),
            np.array([0.25, 0.5]),
            np.array([0.5, 1.0]),
            np.array([0.25, 0.0]),
        )
        expected = np.array([0.0, 0.5, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected)

    def test_make_fm_removes_decreasing_points_and_extrapolates(self) -> None:
        result = _make_fm(
            9,
            0.5,
            0.5,
            np.array([0.25, 0.5]),
            np.array([0.25, 0.5]),
            np.array([0.25, 0.5]),
            np.array([0.5, 0.25]),  # 9250 < 9500: last point must be removed
            8000.0,
            10000.0,
        )
        expected = np.array(
            [
                8000.0,
                8250.0,
                8500.0,
                8500.0 + 1000.0 / 3,
                8500.0 + 2000.0 / 3,
                9500.0,
                9500.0 + 1000.0 / 3,
                9500.0 + 2000.0 / 3,
                10500.0,
            ]
        )
        np.testing.assert_allclose(result, expected, rtol=1e-6)


class TestPhraseSynthesis(unittest.TestCase):
    def setUp(self) -> None:
        self.template = TwitStruct.load()
        self.synthesizer = TwitterSynthesizer(self.template)
        self.dto = _default_dto(self.template)

    def _pipeline_state(self) -> TwitterState:
        """Default template through steps 1-4."""
        state = self.synthesizer.prepare(self.dto)
        state = self.synthesizer.scale_mean_gain(state)
        state = self.synthesizer.reset_phrase_count(state)
        state = self.synthesizer.update_phrase_frequencies(state)
        return self.synthesizer.set_duration(state)

    def test_single_phrase_matches_reference(self) -> None:
        state = self.synthesizer.set_duration(_tiny_state())
        signal = self.synthesizer.synthesize_phrases(state)
        ref_combined, ref_fund, ref_harm = _reference_phrase_signal(
            state, state.sample_rate
        )
        self.assertIsInstance(signal, TwitterSignal)
        np.testing.assert_allclose(signal.combined, ref_combined, atol=1e-12)
        np.testing.assert_allclose(signal.fundamental, ref_fund, atol=1e-12)
        np.testing.assert_allclose(signal.harmonic, ref_harm, atol=1e-12)

    def test_options_match_reference(self) -> None:
        cases: list[tuple[str, TwitterState]] = []
        flat_am = _tiny_state()
        flat_am.options = CallOptions(
            reverse_order=False, flat_am=True, flat_fm=0, chop=Chop.NORMAL
        )
        cases.append(("flat_am", flat_am))
        chop_before = _tiny_state()
        chop_before.options = CallOptions(
            reverse_order=False, flat_am=False, flat_fm=0, chop=Chop.BEFORE_KNEE
        )
        cases.append(("chop_before", chop_before))
        chop_after = _tiny_state()
        chop_after.options = CallOptions(
            reverse_order=False, flat_am=False, flat_fm=0, chop=Chop.AFTER_KNEE
        )
        cases.append(("chop_after", chop_after))
        reverse_fm = _tiny_state()
        reverse_fm.time_reversal = TimeReversal(fm=True)
        cases.append(("rev_fm", reverse_fm))
        reverse_am = _tiny_state()
        reverse_am.time_reversal = TimeReversal(am=True)
        cases.append(("rev_am", reverse_am))
        ordered = _tiny_state(
            nphr=2,
            phrases=[_tiny_phrase(1000.0, 2000.0), _tiny_phrase(3000.0, 4000.0)],
        )
        ordered.options = CallOptions(
            reverse_order=True, flat_am=False, flat_fm=0, chop=Chop.NORMAL
        )
        cases.append(("reverse_order", ordered))
        for name, state in cases:
            with self.subTest(name=name):
                state = self.synthesizer.set_duration(state)
                signal = self.synthesizer.synthesize_phrases(state)
                ref_combined, ref_fund, ref_harm = _reference_phrase_signal(
                    state, state.sample_rate
                )
                np.testing.assert_allclose(signal.combined, ref_combined, atol=1e-12)
                np.testing.assert_allclose(signal.fundamental, ref_fund, atol=1e-12)
                np.testing.assert_allclose(signal.harmonic, ref_harm, atol=1e-12)

    def test_signal_lengths_and_combination_identity(self) -> None:
        state = self._pipeline_state()
        signal = self.synthesizer.synthesize_phrases(state)
        self.assertEqual(len(signal.combined), floor(48_000 * state.duration + 0.5))
        self.assertEqual(len(signal.fundamental), len(signal.combined))
        self.assertEqual(len(signal.harmonic), len(signal.combined))
        np.testing.assert_allclose(
            signal.combined,
            (signal.fundamental + state.harmonic_amplitude_ratio * signal.harmonic)
            / (1 + state.harmonic_amplitude_ratio),
        )

    def test_ipi_gap_samples_are_zero(self) -> None:
        state = self._pipeline_state()
        signal = self.synthesizer.synthesize_phrases(state)
        stop_first = floor(48_000 * state.phrase_sweep_times[0]) - 1
        start_second = (
            floor(
                48_000
                * (
                    0.5 * state.phrase_sweep_times[0]
                    + state.inter_phrase_interval
                    - 0.5 * state.phrase_sweep_times[1]
                )
            )
            - 1
        )
        midpoint = (stop_first + start_second) // 2
        self.assertGreater(midpoint, stop_first)
        self.assertLess(midpoint, start_second)
        self.assertEqual(signal.combined[midpoint], 0.0)
        self.assertEqual(signal.fundamental[midpoint], 0.0)
        self.assertEqual(signal.harmonic[midpoint], 0.0)

    def test_chop_before_knee_silences_tail(self) -> None:
        state = _tiny_state()
        state.options = CallOptions(
            reverse_order=False, flat_am=False, flat_fm=0, chop=Chop.BEFORE_KNEE
        )
        state = self.synthesizer.set_duration(state)
        signal = self.synthesizer.synthesize_phrases(state)
        # The knee sits at contour x=0.5 (sample ~40 of 80); x > 0.525 is silent.
        np.testing.assert_array_equal(signal.combined[50:], np.zeros(30))

    def test_does_not_modify_input_state(self) -> None:
        state = self.synthesizer.set_duration(_tiny_state())
        before = {
            index: (phrase.frequency_before_knee.copy(), phrase.am1_before_knee.copy())
            for index, phrase in state.phrase.items()
        }
        self.synthesizer.synthesize_phrases(state)
        for index, phrase in state.phrase.items():
            np.testing.assert_array_equal(
                phrase.frequency_before_knee, before[index][0]
            )
            np.testing.assert_array_equal(phrase.am1_before_knee, before[index][1])

    def test_rejects_invalid_inputs(self) -> None:
        bad_sr = _tiny_state()
        bad_sr.sample_rate = 0
        with self.assertRaisesRegex(ValueError, "sampling rate"):
            self.synthesizer.synthesize_phrases(bad_sr)
        zero = _tiny_state()
        zero.phrase_count = 0
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            self.synthesizer.synthesize_phrases(zero)
        excess = self.synthesizer.prepare(
            self.dto.model_copy(update={"phrase_count": 10})
        )
        excess.multi_phrase_source = 1
        with self.assertRaisesRegex(ValueError, "exceeds available"):
            self.synthesizer.synthesize_phrases(excess)
        silent = self.synthesizer.set_duration(
            _tiny_state(phrases=[_tiny_phrase(relative_amplitude=0.0)])
        )
        with self.assertRaisesRegex(ValueError, "not be all zero"):
            self.synthesizer.synthesize_phrases(silent)


class TestEnvelopeHelpers(unittest.TestCase):
    def test_flat_am_envelope_ramps(self) -> None:
        am1, am2 = _am_envelopes(
            _tiny_phrase(), 80, 1.0, 8, 8, Chop.NORMAL, flat_am=True
        )
        expected_onset = np.arange(9) / 8
        expected_offset = np.arange(8, -1, -1) / 8
        for envelope in (am1, am2):
            np.testing.assert_allclose(envelope[:9], expected_onset)
            np.testing.assert_allclose(envelope[9:71], 1.0)
            np.testing.assert_allclose(envelope[71:], expected_offset)

    def test_am_envelope_scaled_by_relative_amplitude(self) -> None:
        am1, am2 = _am_envelopes(
            _tiny_phrase(), 80, 0.5, 8, 8, Chop.NORMAL, flat_am=True
        )
        for envelope in (am1, am2):
            np.testing.assert_allclose(envelope[9:71], 0.5)
            self.assertAlmostEqual(envelope[8], 0.5)

    def test_chop_before_knee_zeroes_envelope_tail(self) -> None:
        am1, _ = _am_envelopes(
            _tiny_phrase(), 80, 1.0, 8, 8, Chop.BEFORE_KNEE, flat_am=False
        )
        # The knee sits at contour x=0.5 (sample ~40 of 80); x > 0.525 is silent.
        np.testing.assert_array_equal(am1[50:], np.zeros(30))

    def test_oscillate_hand_computed(self) -> None:
        fundamental_frequency = np.full(4, 1000.0)
        harmonic_frequency = np.full(4, 2000.0)
        fundamental_amplitude = np.ones(4)
        harmonic_amplitude = np.zeros(4)
        combined, fundamental, harmonic = _oscillate(
            fundamental_frequency,
            harmonic_frequency,
            fundamental_amplitude,
            harmonic_amplitude,
            8000,
            0.5,
        )
        # Phase = cumsum(f/sr) = [1/8, 1/4, 3/8, 1/2] cycles → sin(2*pi*phase).
        expected = np.sin(2 * np.pi * np.array([0.125, 0.25, 0.375, 0.5]))
        np.testing.assert_allclose(fundamental, expected, atol=1e-12)
        np.testing.assert_array_equal(harmonic, np.zeros(4))
        np.testing.assert_allclose(combined, expected / 1.5, atol=1e-12)


class TestSignalNormalization(unittest.TestCase):
    def setUp(self) -> None:
        self.synthesizer = TwitterSynthesizer(TwitStruct.load())
        self.signal = TwitterSignal(
            combined=np.array([0.0, 0.25, -0.5, 0.0]),
            fundamental=np.array([1.0, 2.0, 3.0, 4.0]),
            harmonic=np.array([5.0, 6.0, 7.0, 8.0]),
        )

    def test_normalizes_combined_to_unit_peak(self) -> None:
        result = self.synthesizer.normalize_signal(self.signal)
        np.testing.assert_allclose(result.combined, [0.0, 0.5, -1.0, 0.0])
        self.assertAlmostEqual(float(np.max(np.abs(result.combined))), 1.0)

    def test_preserves_fundamental_and_harmonic(self) -> None:
        result = self.synthesizer.normalize_signal(self.signal)
        np.testing.assert_array_equal(result.fundamental, self.signal.fundamental)
        np.testing.assert_array_equal(result.harmonic, self.signal.harmonic)

    def test_normalize_does_not_modify_input(self) -> None:
        original_combined = self.signal.combined.copy()
        result = self.synthesizer.normalize_signal(self.signal)
        self.assertIsNot(result, self.signal)
        np.testing.assert_array_equal(self.signal.combined, original_combined)
        self.assertFalse(np.shares_memory(result.combined, self.signal.combined))
        self.assertFalse(np.shares_memory(result.fundamental, self.signal.fundamental))

    def test_rejects_all_zero_signal(self) -> None:
        silent = TwitterSignal(
            combined=np.zeros(4), fundamental=np.zeros(4), harmonic=np.zeros(4)
        )
        with self.assertRaisesRegex(ValueError, "not be all zero"):
            self.synthesizer.normalize_signal(silent)

    def test_rejects_empty_signal(self) -> None:
        with self.assertRaisesRegex(ValidationError, "must not be empty"):
            TwitterSignal(
                combined=np.zeros(0), fundamental=np.zeros(0), harmonic=np.zeros(0)
            )

    def test_pipeline_normalizes_to_unit_peak(self) -> None:
        state = self.synthesizer.set_duration(_tiny_state())
        signal = self.synthesizer.synthesize_phrases(state)
        normalized = self.synthesizer.normalize_signal(signal)
        self.assertAlmostEqual(float(np.max(np.abs(normalized.combined))), 1.0)


class TestBackgroundNoise(unittest.TestCase):
    def setUp(self) -> None:
        self.template = TwitStruct.load()
        self.synthesizer = TwitterSynthesizer(self.template)
        self.dto = _default_dto(self.template)
        self.signal = TwitterSignal(
            combined=np.full(80, 2.0),
            fundamental=np.zeros(80),
            harmonic=np.zeros(80),
        )

    def _state(self) -> TwitterState:
        return self.synthesizer.set_duration(_tiny_state())

    def test_pure_tone_when_bandwidth_zero(self) -> None:
        state = self._state()
        state.noise_bandwidth = 0.0
        result = self.synthesizer.add_background_noise(self.signal, state)
        np.testing.assert_array_equal(result.combined, self.signal.combined)

    def test_pure_noise_when_snr_zero(self) -> None:
        state = self._state()
        state.signal_to_noise_ratio = 0.0
        noise = np.arange(1.0, 81.0)  # wide-band branch passes it through
        noise_rms = float(np.linalg.norm(noise) / np.sqrt(len(noise)))
        with mock.patch(
            "py_virtualcall.synthesizer.twitter_synthesizer._noise_samples",
            return_value=noise,
        ):
            result = self.synthesizer.add_background_noise(self.signal, state)
        # Signal part (rms=2) is dropped; noise is scaled by 2/rms_noise.
        np.testing.assert_allclose(
            result.combined, noise * (2.0 / noise_rms), atol=1e-12
        )

    def test_snr_scaling_formula(self) -> None:
        state = self._state()
        state.signal_to_noise_ratio = 20.0
        noise = np.arange(1.0, 81.0)
        noise_rms = float(np.linalg.norm(noise) / np.sqrt(len(noise)))
        with mock.patch(
            "py_virtualcall.synthesizer.twitter_synthesizer._noise_samples",
            return_value=noise,
        ):
            result = self.synthesizer.add_background_noise(self.signal, state)
        # rms(y)=2, so rms_desired = 2 * 10^(-20/20) = 0.2.
        expected = 2.0 + noise * (0.2 / noise_rms)
        np.testing.assert_allclose(result.combined, expected, atol=1e-12)

    def test_noise_is_deterministic_for_seed(self) -> None:
        state = self._state()  # default bpbw=100 → wide-band branch
        first = self.synthesizer.add_background_noise(self.signal, state)
        second = self.synthesizer.add_background_noise(self.signal, state)
        np.testing.assert_array_equal(first.combined, second.combined)
        other_state = self._state()
        other_state.seed = state.seed + 1
        other = self.synthesizer.add_background_noise(self.signal, other_state)
        self.assertFalse(np.array_equal(first.combined, other.combined))

    def test_band_limit_branches(self) -> None:
        dc = np.ones(1000)
        wide = _band_limit(dc, 100.0, 60000.0, 8000)  # start<200, stop>49400
        np.testing.assert_array_equal(wide, dc)
        band = _band_limit(dc, 1000.0, 2000.0, 8000)  # band-pass within nyquist
        self.assertLess(float(np.max(np.abs(band))), 0.01)
        high = _band_limit(dc, 1000.0, 60000.0, 8000)  # high-pass
        self.assertLess(float(np.max(np.abs(high))), 0.01)
        low = _band_limit(dc, 100.0, 3000.0, 8000)  # low-pass
        np.testing.assert_allclose(low, 1.0, atol=0.05)

    def test_does_not_modify_input(self) -> None:
        state = self._state()
        original_combined = self.signal.combined.copy()
        result = self.synthesizer.add_background_noise(self.signal, state)
        self.assertIsNot(result, self.signal)
        np.testing.assert_array_equal(self.signal.combined, original_combined)
        self.assertFalse(np.shares_memory(result.combined, self.signal.combined))

    def test_rejects_invalid_sr(self) -> None:
        state = self._state()
        state.sample_rate = 0
        with self.assertRaisesRegex(ValueError, "sampling rate"):
            self.synthesizer.add_background_noise(self.signal, state)

    def test_pipeline_end_to_end(self) -> None:
        state = self.synthesizer.prepare(self.dto)
        state = self.synthesizer.scale_mean_gain(state)
        state = self.synthesizer.reset_phrase_count(state)
        state = self.synthesizer.update_phrase_frequencies(state)
        state = self.synthesizer.set_duration(state)
        signal = self.synthesizer.synthesize_phrases(state)
        normalized = self.synthesizer.normalize_signal(signal)
        result = self.synthesizer.add_background_noise(normalized, state)
        self.assertTrue(np.isfinite(result.combined).all())
        np.testing.assert_array_equal(result.fundamental, normalized.fundamental)
        np.testing.assert_array_equal(result.harmonic, normalized.harmonic)
        # Wide-band branch (template bpbw=100) adds tiny but nonzero noise.
        self.assertFalse(np.array_equal(result.combined, normalized.combined))


class TestEdgeRamps(unittest.TestCase):
    def setUp(self) -> None:
        self.template = TwitStruct.load()
        self.synthesizer = TwitterSynthesizer(self.template)
        self.dto = _default_dto(self.template)
        self.signal = TwitterSignal(
            combined=np.ones(500),
            fundamental=np.full(500, 2.0),
            harmonic=np.full(500, 3.0),
        )
        self.state = self.synthesizer.set_duration(_tiny_state(sr=48_000))

    def test_onset_ramp_matches_matlab_formula(self) -> None:
        sr = 48_000
        ramp_length = floor(sr * 0.005) + 1  # 241 samples
        time = np.arange(ramp_length) / sr
        expected_onset = (1 - np.cos(2 * np.pi * time / 0.01)) / 2
        result = self.synthesizer.apply_edge_ramps(self.signal, self.state)
        np.testing.assert_allclose(result.combined[:ramp_length], expected_onset)
        np.testing.assert_allclose(result.combined[ramp_length:-ramp_length], 1.0)
        np.testing.assert_allclose(result.combined[-ramp_length:], expected_onset[::-1])

    def test_off_ramp_is_time_reversed_onset(self) -> None:
        sr = 48_000
        ramp_length = floor(sr * 0.005) + 1
        result = self.synthesizer.apply_edge_ramps(self.signal, self.state)
        np.testing.assert_array_equal(
            result.combined[-ramp_length:], result.combined[:ramp_length][::-1]
        )

    def test_edges_start_and_end_at_zero(self) -> None:
        result = self.synthesizer.apply_edge_ramps(self.signal, self.state)
        self.assertEqual(result.combined[0], 0.0)
        self.assertEqual(result.combined[-1], 0.0)

    def test_fund_and_harm_unchanged(self) -> None:
        result = self.synthesizer.apply_edge_ramps(self.signal, self.state)
        np.testing.assert_array_equal(result.fundamental, self.signal.fundamental)
        np.testing.assert_array_equal(result.harmonic, self.signal.harmonic)

    def test_does_not_modify_input(self) -> None:
        original_combined = self.signal.combined.copy()
        result = self.synthesizer.apply_edge_ramps(self.signal, self.state)
        self.assertIsNot(result, self.signal)
        np.testing.assert_array_equal(self.signal.combined, original_combined)
        self.assertFalse(np.shares_memory(result.combined, self.signal.combined))

    def test_rejects_invalid_sr(self) -> None:
        bad_state = self.state.model_copy(deep=True)
        bad_state.sample_rate = 0
        with self.assertRaisesRegex(ValueError, "sampling rate"):
            self.synthesizer.apply_edge_ramps(self.signal, bad_state)

    def test_pipeline_end_to_end(self) -> None:
        state = self.synthesizer.prepare(self.dto)
        state = self.synthesizer.scale_mean_gain(state)
        state = self.synthesizer.reset_phrase_count(state)
        state = self.synthesizer.update_phrase_frequencies(state)
        state = self.synthesizer.set_duration(state)
        signal = self.synthesizer.synthesize_phrases(state)
        signal = self.synthesizer.normalize_signal(signal)
        signal = self.synthesizer.add_background_noise(signal, state)
        result = self.synthesizer.apply_edge_ramps(signal, state)
        self.assertTrue(np.isfinite(result.combined).all())
        self.assertEqual(result.combined[0], 0.0)
        self.assertEqual(result.combined[-1], 0.0)
        np.testing.assert_array_equal(result.fundamental, signal.fundamental)
        np.testing.assert_array_equal(result.harmonic, signal.harmonic)


if __name__ == "__main__":
    unittest.main()
