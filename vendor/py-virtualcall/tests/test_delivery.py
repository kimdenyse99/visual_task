"""Tests for the final delivered virtual call model."""

import tempfile
import unittest
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile  # pyright: ignore[reportUnknownVariableType]

from py_virtualcall.delivery import VirtualCall, build_virtual_call, write_wav
from py_virtualcall.synthesizer import TwitterSignal, TwitterSynthesizer
from py_virtualcall.template import TwitStruct, TwitterDTO


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


class TestVirtualCall(unittest.TestCase):
    def setUp(self) -> None:
        self.template = TwitStruct.load()
        self.synthesizer = TwitterSynthesizer(self.template)
        self.dto = _default_dto(self.template)

    def _full_signal(self) -> TwitterSignal:
        state = self.synthesizer.prepare(self.dto)
        state = self.synthesizer.scale_mean_gain(state)
        state = self.synthesizer.reset_phrase_count(state)
        state = self.synthesizer.update_phrase_frequencies(state)
        state = self.synthesizer.set_duration(state)
        signal = self.synthesizer.synthesize_phrases(state)
        signal = self.synthesizer.normalize_signal(signal)
        signal = self.synthesizer.add_background_noise(signal, state)
        return self.synthesizer.apply_edge_ramps(signal, state)

    def test_fields_match_inputs(self) -> None:
        signal = self._full_signal()
        stimulus = build_virtual_call(signal, self.dto)
        self.assertIsInstance(stimulus, VirtualCall)
        self.assertEqual(stimulus.metadata, self.dto)
        self.assertEqual(stimulus.sample_rate, 48_000)
        self.assertEqual(stimulus.metadata.sample_rate, 48_000)
        self.assertIsInstance(stimulus.waveform, np.ndarray)
        self.assertEqual(stimulus.waveform.dtype, np.float64)
        np.testing.assert_array_equal(stimulus.waveform, signal.combined)

    def test_is_frozen_dataclass(self) -> None:
        stimulus = build_virtual_call(self._full_signal(), self.dto)
        self.assertTrue(is_dataclass(stimulus))
        with self.assertRaises(FrozenInstanceError):
            stimulus.sample_rate = 44_100  # type: ignore[misc]

    def test_waveform_is_an_owned_copy(self) -> None:
        signal = self._full_signal()
        stimulus = build_virtual_call(signal, self.dto)
        self.assertFalse(np.shares_memory(stimulus.waveform, signal.combined))
        signal.combined[0] = 123.0
        self.assertNotEqual(stimulus.waveform[0], 123.0)

    def test_metadata_reflects_custom_dto(self) -> None:
        signal = self._full_signal()
        dto = self.dto.model_copy(
            update={
                "center_frequency": 10_000.0,
                "phrase_count": 5,
                "sample_rate": 44_100,
            }
        )
        stimulus = build_virtual_call(signal, dto)
        self.assertEqual(stimulus.metadata.center_frequency, 10_000.0)
        self.assertEqual(stimulus.metadata.phrase_count, 5)
        self.assertEqual(stimulus.sample_rate, 44_100)

    def test_pipeline_end_to_end(self) -> None:
        signal = self._full_signal()
        stimulus = build_virtual_call(signal, self.dto)
        self.assertTrue(np.isfinite(stimulus.waveform).all())
        self.assertEqual(stimulus.waveform[0], 0.0)
        self.assertEqual(stimulus.waveform[-1], 0.0)
        self.assertGreater(float(np.max(np.abs(stimulus.waveform))), 0.0)
        self.assertEqual(stimulus.metadata, self.dto)

    def test_write_wav_round_trip(self) -> None:
        time = np.arange(1000) / 8000
        waveform = 0.5 * np.sin(2 * np.pi * 440 * time)
        call = VirtualCall(
            metadata=self.dto.model_copy(update={"sample_rate": 8000}),
            sample_rate=8000,
            waveform=waveform,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.wav"
            write_wav(call, path)
            rate, data = cast(
                tuple[int, NDArray[np.int16]],
                wavfile.read(path),  # pyright: ignore[reportUnknownMemberType]
            )
        self.assertEqual(rate, 8000)
        expected = np.round(waveform * np.iinfo(np.int16).max).astype(np.int16)
        np.testing.assert_array_equal(data, expected)

    def test_write_wav_normalizes_overfull_waveform(self) -> None:
        call = VirtualCall(
            metadata=self.dto.model_copy(update={"sample_rate": 8000}),
            sample_rate=8000,
            waveform=np.full(1000, 2.0),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "loud.wav"
            write_wav(call, path)
            _, data = cast(
                tuple[int, NDArray[np.int16]],
                wavfile.read(path),  # pyright: ignore[reportUnknownMemberType]
            )
        self.assertEqual(int(np.max(data)), np.iinfo(np.int16).max)


if __name__ == "__main__":
    unittest.main()
