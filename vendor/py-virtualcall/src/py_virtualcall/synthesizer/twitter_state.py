"""Mutable NumPy-backed state for the twitter synthesis pipeline."""

from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from py_virtualcall.template import (
    CallOptions,
    Float64Array,
    Phrase,
    TimeReversal,
    TwitStruct,
    TwitterDTO,
)


class PhraseState(BaseModel):
    """Mutable NumPy-backed state for one twitter phrase."""

    model_config = ConfigDict(arbitrary_types_allowed=True, allow_inf_nan=False)

    start_frequency: float = Field(gt=0)
    end_frequency: float = Field(gt=0)
    knee_frequency_fraction: float = Field(ge=0)
    knee_time_fraction: float = Field(ge=0)
    sweep_time: float = Field(gt=0)
    relative_amplitude: float = Field(ge=0, le=1)
    time_before_knee: Float64Array
    time_after_knee: Float64Array
    frequency_before_knee: Float64Array
    frequency_after_knee: Float64Array
    am1_before_knee: Float64Array
    am1_after_knee: Float64Array
    am2_before_knee: Float64Array
    am2_after_knee: Float64Array
    am_time_before_knee: Float64Array
    am_time_after_knee: Float64Array

    @model_validator(mode="after")
    def validate_arrays(self) -> Self:
        for name in (
            "time_before_knee",
            "frequency_before_knee",
            "am1_before_knee",
            "am2_before_knee",
            "am_time_before_knee",
        ):
            _validate_array(getattr(self, name), name=name, length=25)
        for name in (
            "time_after_knee",
            "frequency_after_knee",
            "am1_after_knee",
            "am2_after_knee",
            "am_time_after_knee",
        ):
            _validate_array(getattr(self, name), name=name, length=10)
        return self

    @classmethod
    def from_template(cls, phrase: Phrase) -> Self:
        return cls(
            start_frequency=phrase.start_frequency,
            end_frequency=phrase.end_frequency,
            knee_frequency_fraction=phrase.knee_frequency_fraction,
            knee_time_fraction=phrase.knee_time_fraction,
            sweep_time=phrase.sweep_time,
            relative_amplitude=phrase.relative_amplitude,
            time_before_knee=_array(phrase.time_before_knee),
            time_after_knee=_array(phrase.time_after_knee),
            frequency_before_knee=_array(phrase.frequency_before_knee),
            frequency_after_knee=_array(phrase.frequency_after_knee),
            am1_before_knee=_array(phrase.am1_before_knee),
            am1_after_knee=_array(phrase.am1_after_knee),
            am2_before_knee=_array(phrase.am2_before_knee),
            am2_after_knee=_array(phrase.am2_after_knee),
            am_time_before_knee=_array(phrase.am_time_before_knee),
            am_time_after_knee=_array(phrase.am_time_after_knee),
        )


class TwitterState(BaseModel):
    """Mutable NumPy-backed state transformed by synthesis stages."""

    model_config = ConfigDict(arbitrary_types_allowed=True, allow_inf_nan=False)

    center_frequency: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    inter_phrase_interval: float = Field(ge=0)
    harmonic_ratio: int = Field(gt=0)
    harmonic_amplitude_ratio: float = Field(gt=0, le=1)
    phrase_bandwidths: Float64Array
    mean_phrase_bandwidth: float = Field(ge=0)
    phrase: dict[int, PhraseState]
    phrase_count: int = Field(ge=0)
    phrase_sweep_times: Float64Array
    knee_time_fractions: Float64Array
    knee_frequency_fractions: Float64Array
    mean_phrase_sweep_time: float = Field(gt=0)
    mean_knee_time_fraction: float = Field(ge=0, le=1)
    mean_knee_frequency_fraction: float = Field(ge=0)
    duration: float = Field(ge=0)
    signal_to_noise_ratio: float = Field(ge=0)
    noise_bandwidth: float = Field(ge=0)
    seed: int = Field(gt=0)
    harmonic_attenuation: float = Field(le=0)
    multi_phrase_source: Literal[1, 2]
    single_phrase_source: Literal[1, 2, 3]
    options: CallOptions
    time_reversal: TimeReversal
    structure_source: Literal["ALL"]

    @model_validator(mode="after")
    def validate_structure(self) -> Self:
        phrase_count = len(self.phrase)
        if phrase_count == 0:
            raise ValueError("phrase must not be empty")
        if tuple(self.phrase) != tuple(range(phrase_count)):
            raise ValueError("phrase keys must be consecutive integers starting at 0")
        for name in (
            "phrase_bandwidths",
            "phrase_sweep_times",
            "knee_time_fractions",
            "knee_frequency_fractions",
        ):
            _validate_array(getattr(self, name), name=name, length=phrase_count)
        return self

    @classmethod
    def from_template(cls, template: TwitStruct, dto: TwitterDTO) -> Self:
        return cls(
            center_frequency=dto.center_frequency,
            sample_rate=dto.sample_rate,
            inter_phrase_interval=dto.inter_phrase_interval,
            harmonic_ratio=template.harmonic_ratio,
            harmonic_amplitude_ratio=10 ** (dto.harmonic_attenuation / 20),
            phrase_bandwidths=_array(template.phrase_bandwidths),
            mean_phrase_bandwidth=dto.mean_phrase_bandwidth,
            phrase={
                index: PhraseState.from_template(phrase)
                for index, phrase in template.phrase.items()
            },
            phrase_count=dto.phrase_count,
            phrase_sweep_times=_array(template.phrase_sweep_times),
            knee_time_fractions=_array(template.knee_time_fractions),
            knee_frequency_fractions=_array(template.knee_frequency_fractions),
            mean_phrase_sweep_time=dto.mean_phrase_sweep_time,
            mean_knee_time_fraction=dto.mean_knee_time_fraction,
            mean_knee_frequency_fraction=dto.mean_knee_frequency_fraction,
            duration=template.duration,
            signal_to_noise_ratio=dto.signal_to_noise_ratio,
            noise_bandwidth=dto.noise_bandwidth,
            seed=dto.seed,
            harmonic_attenuation=dto.harmonic_attenuation,
            multi_phrase_source=template.multi_phrase_source,
            single_phrase_source=template.single_phrase_source,
            options=CallOptions(
                reverse_order=dto.reverse_order,
                flat_am=dto.flat_am,
                flat_fm=template.options.flat_fm,
                chop=dto.chop,
            ),
            time_reversal=dto.time_reversal,
            structure_source=template.structure_source,
        )

    def clone(self) -> Self:
        """Return a fully independent synthesis state."""
        return self.model_copy(deep=True)


class TwitterSignal(BaseModel):
    """Output buffers of the twitter synthesis pipeline (MATLAB y/fund/harm)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, allow_inf_nan=False)

    combined: Float64Array  # (y) fund + harmonic_amplitude_ratio*harmonic, /(1+ratio)
    fundamental: Float64Array  # (fund) fundamental component
    harmonic: Float64Array  # (harm) harmonic component

    @model_validator(mode="after")
    def validate_signal(self) -> Self:
        if len(self.combined) == 0:
            raise ValueError("signal must not be empty")
        length = len(self.combined)
        for name in ("combined", "fundamental", "harmonic"):
            _validate_array(getattr(self, name), name=name, length=length)
        return self


def _validate_array(array: Float64Array, *, name: str, length: int) -> None:
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(array) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")


def _array(values: tuple[float, ...]) -> Float64Array:
    return np.array(values, dtype=np.float64, copy=True, order="C")
