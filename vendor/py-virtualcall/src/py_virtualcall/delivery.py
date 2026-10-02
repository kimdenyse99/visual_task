"""Final delivery models: virtual call parameters plus waveform."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile  # pyright: ignore[reportUnknownVariableType]

from py_virtualcall.synthesizer import TwitterSignal
from py_virtualcall.template import TwitterDTO


@dataclass(frozen=True, slots=True)
class VirtualCall:
    """One delivered virtual call.

    Top level exposes the Python-ecosystem conventions: sample_rate plus a
    plain NumPy waveform; metadata carries the defining parameter template
    (its sr always equals sample_rate). Invariants are guaranteed by the
    upstream models and by build_virtual_call, so this record carries no
    runtime validation of its own.
    """

    metadata: TwitterDTO  # the existing parameter template defining this call
    sample_rate: int  # sample rate (Hz), always equal to metadata.sample_rate
    waveform: NDArray[np.float64]  # plain NumPy array, no custom wrapper type


def build_virtual_call(signal: TwitterSignal, dto: TwitterDTO) -> VirtualCall:
    """Wrap the final pipeline signal into a delivered virtual call.

    The waveform is the pipeline's final combined signal as-is (no xblaster
    9.999 scaling). The builder is the single construction path and hands the
    model an owned copy of the waveform array.
    """
    return VirtualCall(
        metadata=dto,
        sample_rate=dto.sample_rate,
        waveform=signal.combined.copy(),
    )


def write_wav(call: VirtualCall, path: Path) -> None:
    """Write the delivered waveform as a 16-bit mono wav file.

    Mirrors the legacy waveWriter semantics: a waveform exceeding unit peak
    is re-normalized before the int16 scaling.
    """
    waveform = call.waveform
    peak = float(np.max(np.abs(waveform)))
    if peak > 1:
        waveform = waveform / peak
    scale = np.iinfo(np.int16).max  # 32767
    samples = np.clip(
        np.round(waveform * scale), np.iinfo(np.int16).min, np.iinfo(np.int16).max
    )
    wavfile.write(  # pyright: ignore[reportUnknownMemberType]
        path, call.sample_rate, samples.astype(np.int16)
    )
