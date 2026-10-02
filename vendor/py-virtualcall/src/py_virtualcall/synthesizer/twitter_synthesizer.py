"""Copy-on-stage transformations of mutable twitter synthesis state."""

from math import floor
from typing import cast

import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray
from scipy.signal import filtfilt, firwin2  # pyright: ignore[reportUnknownVariableType]

from py_virtualcall.synthesizer.twitter_state import (
    PhraseState,
    TwitterSignal,
    TwitterState,
)
from py_virtualcall.template import Chop, TwitStruct, TwitterDTO

_TEMPLATE_PHRASE_COUNT = 9
_MIDDLE_PHRASE_INDEX = 4
_NOISE_HIGH_LIMIT = 49400  # MATLAB hardcoded branch thresholds (TDT-era)
_NOISE_LOW_LIMIT = 200
_EDGE_RAMP_DURATION = 0.005  # MATLAB ramp = 5/1000 (VirtualCall.m)


class TwitterSynthesizer:
    """Build and transform isolated copies of twitter synthesis state."""

    def __init__(self, template: TwitStruct) -> None:
        if len(template.phrase) != _TEMPLATE_PHRASE_COUNT:
            raise ValueError("twitter template must contain exactly 9 phrases")
        self._template = template

    def prepare(self, dto: TwitterDTO) -> TwitterState:
        """Apply user-adjustable parameters to a fresh mutable state."""
        return TwitterState.from_template(self._template, dto)

    # region Step 1: Mean gain scaling

    def scale_mean_gain(self, state: TwitterState) -> TwitterState:
        """Scale a deep copy using the MATLAB mean-gain formulas."""
        bandwidth_reference = state.phrase_bandwidths[_MIDDLE_PHRASE_INDEX]
        mean_knee_frequency_fraction = np.mean(state.knee_frequency_fractions)
        mean_knee_time_fraction = np.mean(state.knee_time_fractions)
        if bandwidth_reference == 0:
            raise ValueError("middle phrase bandwidth must be non-zero")
        if mean_knee_frequency_fraction == 0:
            raise ValueError("mean frequency knee fraction must be non-zero")
        if mean_knee_time_fraction == 0:
            raise ValueError("mean time knee fraction must be non-zero")

        result = state.clone()
        result.phrase_bandwidths *= result.mean_phrase_bandwidth / bandwidth_reference
        result.knee_frequency_fractions *= (
            result.mean_knee_frequency_fraction / mean_knee_frequency_fraction
        )
        result.knee_time_fractions *= (
            result.mean_knee_time_fraction / mean_knee_time_fraction
        )
        result.phrase_sweep_times *= result.mean_phrase_sweep_time / np.mean(
            result.phrase_sweep_times
        )
        for index, phrase in result.phrase.items():
            phrase.knee_frequency_fraction = result.knee_frequency_fractions[index]
            phrase.knee_time_fraction = result.knee_time_fractions[index]
            phrase.sweep_time = result.phrase_sweep_times[index]
        return result

    # endregion

    # region Step 2: Phrase count reset

    def reset_phrase_count(self, state: TwitterState) -> TwitterState:
        """Reset phrases using the MATLAB twitter phrase-selection rules."""
        if state.phrase_count == 0:
            raise ValueError("phrase count must be greater than zero")

        result = state.clone()
        if result.multi_phrase_source == 1:
            return result
        if len(result.phrase) != _TEMPLATE_PHRASE_COUNT:
            raise ValueError("phrase count reset requires exactly 9 source phrases")

        indices = _phrase_indices(result.phrase_count, result.single_phrase_source)
        result.phrase = {
            target: result.phrase[source].model_copy(deep=True)
            for target, source in enumerate(indices)
        }
        for name in (
            "phrase_bandwidths",
            "knee_frequency_fractions",
            "knee_time_fractions",
            "phrase_sweep_times",
        ):
            values = getattr(result, name)
            setattr(result, name, values[np.asarray(indices)])
        return result

    # endregion

    # region Step 3: fst/fsp update

    def update_phrase_frequencies(self, state: TwitterState) -> TwitterState:
        """Recenter phrase bandwidths and shift the middle phrase to fc.

        Mirrors twitter_synth.m:156-175: keep each phrase's center frequency,
        set its bandwidth from the phrase bandwidth vector, then translate all
        phrases so that the middle phrase (MATLAB fst(ceil(nphr/2)), 1-based)
        centers on the call center frequency.
        """
        if state.phrase_count == 0:
            raise ValueError("phrase count must be greater than zero")
        if state.phrase_count > len(state.phrase):
            raise ValueError("phrase count exceeds available phrases")

        result = state.clone()
        middle_index = (
            state.phrase_count + 1
        ) // 2 - 1  # MATLAB ceil(nphr/2) in 0-based keys
        middle_phrase = result.phrase[middle_index]
        center_frequency_shift = result.center_frequency - 0.5 * (
            middle_phrase.start_frequency + middle_phrase.end_frequency
        )
        for index in range(result.phrase_count):
            phrase = result.phrase[index]
            phrase_center = 0.5 * (
                phrase.start_frequency + phrase.end_frequency
            )  # read before overwrite
            phrase.start_frequency = (
                phrase_center
                - 0.5 * result.phrase_bandwidths[index]
                + center_frequency_shift
            )
            phrase.end_frequency = (
                phrase_center
                + 0.5 * result.phrase_bandwidths[index]
                + center_frequency_shift
            )
        return result

    # endregion

    # region Step 4: Set IPI and duration

    def set_duration(self, state: TwitterState) -> TwitterState:
        """Compute the call duration from phrase timing (twitter_synth.m:178-186).

        duration = 0.5*tsw(1) + sum(IPI) + 0.5*tsw(nphr) with MATLAB 1-based
        indexing, so the last term uses the sweep time of the nphr-th phrase.
        """
        if state.phrase_count == 0:
            raise ValueError("phrase count must be greater than zero")
        if state.phrase_count > len(state.phrase):
            raise ValueError("phrase count exceeds available phrases")

        result = state.clone()
        interval_sum = (
            state.phrase_count - 1
        ) * state.inter_phrase_interval  # sum of the IPI vector
        result.duration = (
            0.5 * state.phrase_sweep_times[0]
            + interval_sum
            + 0.5 * state.phrase_sweep_times[state.phrase_count - 1]
        )
        return result

    # endregion

    # region Step 5: Phrase synthesis

    def synthesize_phrases(self, state: TwitterState) -> TwitterSignal:
        """Sum phrase_count synthesized phrases into signal buffers (twitter_synth.m:188-314).

        Requires the state prepared by steps 1-4 (duration computed in step 4);
        the sample rate comes from state.sample_rate.
        """
        if state.sample_rate <= 0:
            raise ValueError("sampling rate must be positive")
        if state.phrase_count == 0:
            raise ValueError("phrase count must be greater than zero")
        if state.phrase_count > len(state.phrase):
            raise ValueError("phrase count exceeds available phrases")
        max_relative_amplitude = max(
            state.phrase[i].relative_amplitude for i in range(state.phrase_count)
        )
        if max_relative_amplitude == 0:
            raise ValueError("phrase amplitudes must not be all zero")

        sample_rate = state.sample_rate
        sample_count = floor(
            sample_rate * state.duration + 0.5
        )  # MATLAB round (half away from zero)
        combined = np.zeros(sample_count)
        fundamental = np.zeros(sample_count)
        harmonic = np.zeros(sample_count)
        phrase_order = (
            range(state.phrase_count - 1, -1, -1)
            if state.options.reverse_order
            else range(state.phrase_count)
        )
        onset_samples = floor(sample_rate / 1000)  # 1 ms linear ramp per phrase
        offset_samples = floor(sample_rate / 1000)
        first_sweep_time = state.phrase_sweep_times[0]  # MATLAB tsw(1): lead-in

        for index in range(state.phrase_count):
            phrase = state.phrase[phrase_order[index]]
            scaled_amplitude = phrase.relative_amplitude / max_relative_amplitude
            phrase_offset = (
                index * state.inter_phrase_interval
            )  # sum(IPI(1:idx-1)), idx 1-based
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

            # Normalize frequency contours to [0,1] (new arrays, never mutate state)
            frequency_before_normalized = (
                phrase.frequency_before_knee - phrase.frequency_before_knee.min()
            ) / (
                phrase.frequency_before_knee.max() - phrase.frequency_before_knee.min()
            )
            frequency_after_normalized = (
                phrase.frequency_after_knee - phrase.frequency_after_knee.min()
            ) / (phrase.frequency_after_knee.max() - phrase.frequency_after_knee.min())

            am1, am2 = _am_envelopes(
                phrase,
                length,
                scaled_amplitude,
                onset_samples,
                offset_samples,
                state.options.chop,
                state.options.flat_am,
            )

            # Make FM sweeps and silence the harmonic above Nyquist
            fundamental_frequency = _make_fm(
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

            # Time-reversal options
            if state.time_reversal.fm:
                fundamental_frequency = fundamental_frequency[::-1].copy()
                harmonic_frequency = harmonic_frequency[::-1].copy()
            if state.time_reversal.am:
                am1 = am1[::-1].copy()
                am2 = am2[::-1].copy()

            phrase_combined, phrase_fundamental, phrase_harmonic = _oscillate(
                fundamental_frequency,
                harmonic_frequency,
                am1,
                am2,
                sample_rate,
                state.harmonic_amplitude_ratio,
            )
            combined[start_index : stop_index + 1] = phrase_combined
            fundamental[start_index : stop_index + 1] = phrase_fundamental
            harmonic[start_index : stop_index + 1] = phrase_harmonic

        return TwitterSignal(
            combined=combined, fundamental=fundamental, harmonic=harmonic
        )

    # endregion

    # region Step 6: Normalize signal

    def normalize_signal(self, signal: TwitterSignal) -> TwitterSignal:
        """Normalize the combined signal to unit peak (twitter_synth.m:316-317).

        fundamental and harmonic keep their unnormalized values, like in MATLAB.
        """
        peak = float(np.max(np.abs(signal.combined)))
        if peak == 0:
            raise ValueError("signal must not be all zero")
        result = signal.model_copy(deep=True)
        result.combined = result.combined / peak
        return result

    # endregion

    # region Step 7: Background noise

    def add_background_noise(
        self, signal: TwitterSignal, state: TwitterState
    ) -> TwitterSignal:
        """Add band-limited noise at the requested SNR (twitter_synth.m:319-363).

        Requires the step-6 normalized signal: the SNR is defined against the
        pre-noise rms of that normalized combined signal, like in MATLAB.

        Uses NumPy's default_rng instead of the MATLAB mcg16807 stream, so
        noise sequences are reproducible per seed but not bit-identical to
        MATLAB. fundamental and harmonic stay unmodified.
        """
        if state.sample_rate <= 0:
            raise ValueError("sampling rate must be positive")

        sample_rate = state.sample_rate
        sample_count = len(signal.combined)
        signal_rms = float(np.linalg.norm(signal.combined) / np.sqrt(sample_count))
        noise_start = state.center_frequency * 2 ** (-state.noise_bandwidth / 2)
        noise_stop = state.center_frequency * 2 ** (state.noise_bandwidth / 2)

        result = signal.model_copy(deep=True)
        if state.noise_bandwidth == 0:
            noise = np.zeros(sample_count)  # pure tone
        else:
            noise = _band_limit(
                _noise_samples(state.seed, sample_count),
                noise_start,
                noise_stop,
                sample_rate,
            )
            noise_rms = float(np.linalg.norm(noise) / np.sqrt(sample_count))
            noise = noise * (
                signal_rms * 10 ** (-state.signal_to_noise_ratio / 20) / noise_rms
            )
        if state.signal_to_noise_ratio == 0:
            result.combined = np.zeros(sample_count)  # pure noise: drop the signal
        result.combined = result.combined + noise
        return result

    # endregion

    # region Step 8: Edge ramps (GUI post-processing)

    def apply_edge_ramps(
        self, signal: TwitterSignal, state: TwitterState
    ) -> TwitterSignal:
        """Fade the whole call with 5 ms sin² onset/offset ramps.

        Mirrors the VirtualCall GUI post-processing (VirtualCall.m:1318-1322
        and :1709-1713), applied after background noise for every call type.
        Only the combined signal is ramped; fundamental and harmonic stay
        unmodified.
        """
        if state.sample_rate <= 0:
            raise ValueError("sampling rate must be positive")

        sample_rate = state.sample_rate
        ramp_sample_count = floor(sample_rate * _EDGE_RAMP_DURATION) + 1  # 0:1/SR:ramp
        ramp_time = np.arange(ramp_sample_count) / sample_rate
        on_ramp = (1 - np.cos(2 * np.pi * ramp_time / (2 * _EDGE_RAMP_DURATION))) / 2
        off_ramp = on_ramp[::-1].copy()  # MATLAB fliplr

        result = signal.model_copy(deep=True)
        result.combined[:ramp_sample_count] *= on_ramp
        result.combined[-ramp_sample_count:] *= off_ramp
        return result

    # endregion


# region Step 2 helpers


def _phrase_indices(phrase_count: int, single_phrase_source: int) -> tuple[int, ...]:
    match phrase_count:
        case 1:
            match single_phrase_source:
                case 1:
                    return (0,)
                case 2:
                    return (_MIDDLE_PHRASE_INDEX,)
                case _:
                    return (8,)
        case 2:
            return (0, 8)
        case 3:
            return (0, 4, 8)
        case 4:
            return (0, 3, 5, 8)
        case 5:
            return (0, 3, 4, 5, 8)
        case 6:
            return (0, 2, 3, 5, 6, 8)
        case 7:
            return (0, 2, 3, 4, 5, 6, 8)
        case 8:
            return (0, 1, 2, 3, 5, 6, 7, 8)
        case 9:
            return tuple(range(_TEMPLATE_PHRASE_COUNT))
        case _:
            repeated_middle = (phrase_count - _TEMPLATE_PHRASE_COUNT) * (
                _MIDDLE_PHRASE_INDEX,
            )
            return (*range(5), *repeated_middle, *range(5, _TEMPLATE_PHRASE_COUNT))


# endregion

# region Step 5 helpers


def _make_am(
    length: int,
    knee_time_fraction: float,
    time_before_knee: NDArray[np.float64],
    time_after_knee: NDArray[np.float64],
    amplitude_before_knee: NDArray[np.float64],
    amplitude_after_knee: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build a phrase amplitude envelope via linear interpolation (makeam.m, Eq. 23)."""
    knee_times_before = time_before_knee * knee_time_fraction
    knee_times_after = knee_time_fraction + time_after_knee * (1 - knee_time_fraction)
    knee_times = np.concatenate(([0.0], knee_times_before, knee_times_after, [1.0]))
    amplitude_contour = np.concatenate(
        ([0.0], amplitude_before_knee, amplitude_after_knee, [0.0])
    )
    time_grid = np.linspace(0, 1, length)  # MATLAB (0:len-1)/(len-1)
    result = np.zeros(length)
    for k in range(len(knee_times) - 1):
        i = int(np.argmin(np.abs(time_grid - knee_times[k])))
        j = int(np.argmin(np.abs(time_grid - knee_times[k + 1])))
        # MATLAB A(i:j) is inclusive of j, hence the i:j+1 slice
        result[i : j + 1] = amplitude_contour[k] + (
            (amplitude_contour[k + 1] - amplitude_contour[k])
            / (knee_times[k + 1] - knee_times[k])
        ) * (time_grid[i : j + 1] - time_grid[i])
    return result


def _make_fm(
    length: int,
    knee_time_fraction: float,
    knee_frequency_fraction: float,
    time_before_knee: NDArray[np.float64],
    time_after_knee: NDArray[np.float64],
    frequency_before_knee: NDArray[np.float64],
    frequency_after_knee: NDArray[np.float64],
    start_frequency: float,
    end_frequency: float,
) -> NDArray[np.float64]:
    """Build a phrase frequency contour via linear interpolation (makefm.m, Eq. 25)."""
    knee_times_before = time_before_knee * knee_time_fraction
    knee_times_after = knee_time_fraction + time_after_knee * (1 - knee_time_fraction)
    knee_times = np.concatenate((knee_times_before, knee_times_after))
    knee_frequency = start_frequency + (end_frequency - start_frequency) * (
        knee_frequency_fraction
    )
    freq_before = start_frequency + (knee_frequency - start_frequency) * (
        frequency_before_knee
    )
    freq_after = knee_frequency + (end_frequency - knee_frequency) * (
        frequency_after_knee
    )
    frequency_contour = np.concatenate((freq_before, freq_after))
    # Remove points where the contour strictly decreases (MATLAB find(d<0),
    # so NaN points survive and keep propagating like in MATLAB)
    keep = np.ones(len(frequency_contour), dtype=bool)
    keep[1:] = ~(np.diff(frequency_contour) < 0)
    knee_times, frequency_contour = (
        knee_times[keep],
        frequency_contour[keep],
    )
    # Extrapolate first and last freq points to t=0 and t=1
    start_slope = (frequency_contour[1] - frequency_contour[0]) / (
        knee_times[1] - knee_times[0]
    )
    end_slope = (frequency_contour[-1] - frequency_contour[-2]) / (
        knee_times[-1] - knee_times[-2]
    )
    frequency_at_start = frequency_contour[0] - start_slope * knee_times[0]
    frequency_at_end = frequency_contour[-1] + end_slope * (1 - knee_times[-1])
    knee_times = np.concatenate(([0.0], knee_times, [1.0]))
    frequency_contour = np.concatenate(
        ([frequency_at_start], frequency_contour, [frequency_at_end])
    )
    time_grid = np.linspace(0, 1, length)
    result = np.zeros(length)
    for k in range(len(knee_times) - 1):
        i = int(np.argmin(np.abs(time_grid - knee_times[k])))
        j = int(np.argmin(np.abs(time_grid - knee_times[k + 1])))
        # MATLAB F(i:j) is inclusive of j, hence the i:j+1 slice
        result[i : j + 1] = frequency_contour[k] + (
            (frequency_contour[k + 1] - frequency_contour[k])
            / (knee_times[k + 1] - knee_times[k])
        ) * (time_grid[i : j + 1] - time_grid[i])
    return result


def _am_envelopes(
    phrase: PhraseState,
    length: int,
    scaled_amplitude: float,
    onset_samples: int,
    offset_samples: int,
    chop: Chop,
    flat_am: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build amplitude-scaled AM envelopes with chop/flat_am options and 1 ms ramps."""
    am1_before, am1_after = phrase.am1_before_knee.copy(), phrase.am1_after_knee.copy()
    am2_before, am2_after = phrase.am2_before_knee.copy(), phrase.am2_after_knee.copy()
    # Chopping contour by silencing AM
    if chop is Chop.BEFORE_KNEE:
        am1_after[:] = 0.0
        am2_after[:] = 0.0
    elif chop is Chop.AFTER_KNEE:
        am1_before[:] = 0.0
        am2_before[:] = 0.0
    # Make AM envelopes
    if flat_am:
        am1 = np.ones(length)
        am2 = np.ones(length)
    else:
        am1 = _make_am(
            length,
            phrase.knee_time_fraction,
            phrase.am_time_before_knee,
            phrase.am_time_after_knee,
            am1_before,
            am1_after,
        )
        am2 = _make_am(
            length,
            phrase.knee_time_fraction,
            phrase.am_time_before_knee,
            phrase.am_time_after_knee,
            am2_before,
            am2_after,
        )
    am1 = scaled_amplitude * (am1 / np.max(np.abs(am1)))
    am2 = scaled_amplitude * (am2 / np.max(np.abs(am2)))
    # Add amplitude onset and offset ramps (anchors read pre-ramp)
    am1_anchor, am2_anchor = am1[onset_samples], am2[onset_samples]
    am1[: onset_samples + 1] = am1_anchor * np.arange(onset_samples + 1) / onset_samples
    am2[: onset_samples + 1] = am2_anchor * np.arange(onset_samples + 1) / onset_samples
    am1_anchor, am2_anchor = am1[-offset_samples - 1], am2[-offset_samples - 1]
    am1[-offset_samples - 1 :] = (
        am1_anchor * np.arange(offset_samples, -1, -1) / offset_samples
    )
    am2[-offset_samples - 1 :] = (
        am2_anchor * np.arange(offset_samples, -1, -1) / offset_samples
    )
    return am1, am2


def _oscillate(
    fundamental_frequency: NDArray[np.float64],
    harmonic_frequency: NDArray[np.float64],
    fundamental_amplitude: NDArray[np.float64],
    harmonic_amplitude: NDArray[np.float64],
    sample_rate: int,
    harmonic_amplitude_ratio: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Phase-integrate FM contours and synthesize combined/fund/harm buffers."""
    fundamental_frequency = np.cumsum(fundamental_frequency / sample_rate)
    harmonic_frequency = np.cumsum(harmonic_frequency / sample_rate)
    fundamental_carrier = np.cos(2 * np.pi * fundamental_frequency - np.pi / 2)
    harmonic_carrier = np.cos(2 * np.pi * harmonic_frequency - np.pi / 2)
    phrase_fundamental = fundamental_carrier * fundamental_amplitude
    phrase_harmonic = harmonic_carrier * harmonic_amplitude
    combined = (phrase_fundamental + harmonic_amplitude_ratio * phrase_harmonic) / (
        1 + harmonic_amplitude_ratio
    )
    return combined, phrase_fundamental, phrase_harmonic


# endregion

# region Step 7 helpers


def _noise_samples(seed: int, sample_count: int) -> NDArray[np.float64]:
    """White noise from a seeded NumPy stream (MATLAB uses mcg16807)."""
    return default_rng(seed).standard_normal(sample_count)


def _band_limit(
    noise_samples: NDArray[np.float64],
    start_frequency: float,
    stop_frequency: float,
    sample_rate: int,
) -> NDArray[np.float64]:
    """Filter noise through the MATLAB fir2/filtfilt band selection."""
    nyquist = sample_rate / 2
    if (
        stop_frequency > _NOISE_HIGH_LIMIT and start_frequency >= _NOISE_LOW_LIMIT
    ):  # high-pass
        fir = cast(
            NDArray[np.float64],
            firwin2(
                201,
                np.asarray([0, start_frequency * 0.99, start_frequency, nyquist])
                / nyquist,
                [0, 0, 1, 1],
            ),
        )
    elif (
        stop_frequency <= _NOISE_HIGH_LIMIT and start_frequency < _NOISE_LOW_LIMIT
    ):  # low-pass
        fir = cast(
            NDArray[np.float64],
            firwin2(
                201,
                np.asarray([0, stop_frequency, stop_frequency * 1.01, nyquist])
                / nyquist,
                [1, 1, 0, 0],
            ),
        )
    elif (
        stop_frequency > _NOISE_HIGH_LIMIT and start_frequency < _NOISE_LOW_LIMIT
    ):  # wide-band
        return noise_samples
    else:  # band-pass
        fir = cast(
            NDArray[np.float64],
            firwin2(
                201,
                np.asarray(
                    [
                        0,
                        start_frequency * 0.99,
                        start_frequency,
                        stop_frequency,
                        stop_frequency * 1.01,
                        nyquist,
                    ]
                )
                / nyquist,
                [0, 0, 1, 1, 0, 0],
            ),
        )
    return cast(NDArray[np.float64], filtfilt(fir, [1.0], noise_samples))


# endregion
