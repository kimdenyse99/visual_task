"""Pydantic models of the ``twitstruct`` template (TWIT), plus its loader.

Field names use a full descriptive style; the mapping to the MATLAB
``twitstruct`` fields is documented below. Types are assigned by semantic
meaning (continuous physical quantities are ``float``, semantically integral
fields are ``int``) rather than by the incidental storage type of the .mat
file. Loading reads the bundled ``twitter.json`` asset (``assets.TWITTER``)
and builds a frozen :class:`TwitStruct` via :meth:`TwitStruct.load`.
"""

import json
from enum import IntEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from py_virtualcall.template.assets import TWITTER

# ==============================================================================
# Parameters of ``twitstruct`` -- copied from the VirtualCall V2.0 MATLAB
# sources (VirtualCall.m header; twitter_synth.m), with the original grouping.
# MATLAB short names are kept here as the port reference; the Python models
# use the full names listed after each "->".
#
# ------ (1) variable time-varying parm ---------------------------------------
# ------  Not directly controlled by users ------------------------------------
#
# - twitstruct.dur          - call duration                 (sec) -> duration
# - twitstruct.tsw          - phrase sweep time vector      (sec) -> phrase_sweep_times
# - twitstruct.tkn          - time fraction of knee vector  ([0,1]) -> knee_time_fractions
# - twitstruct.fkn          - freq fraction of knee vector  ([0,1]) -> knee_frequency_fractions
# - twitstruct.bwv          - phrase bandwidth vector (fsp-fst) (Hz) -> phrase_bandwidths
# - twitstruct.C_cell{:,1}  - starting frequency vector (fst)   (Hz) -> Phrase.start_frequency
# - twitstruct.C_cell{:,2}  - ending frequency vector   (fsp)   (Hz) -> Phrase.end_frequency
# - twitstruct.C_cell{:,3}  - same as twitstruct.fkn    (fkn)   ([0,1]) -> Phrase.knee_frequency_fraction
# - twitstruct.C_cell{:,4}  - same as twitstruct.tkn    (tkn)   ([0,1]) -> Phrase.knee_time_fraction
# - twitstruct.C_cell{:,5}  - same as twitstruct.tsw    (tsw)   (sec) -> Phrase.sweep_time
# - twitstruct.C_cell{:,9}  - freq contour before knee  (f1bk)  (Hz) -> Phrase.frequency_before_knee
# - twitstruct.C_cell{:,10} - freq contour after knee   (f1ak)  (Hz) -> Phrase.frequency_after_knee
# - twitstruct.rev          - time reversed option [FM, AM] (default [0 0]) -> time_reversal
#
# ------ (2) constant scalar parm ---------------------------------------------
# ------  Unable to change in this program ------------------------------------
#
# - trillstruct.mnk         - structure source [sic; actually twitstruct] -> structure_source
# - trillstruct.f2f1        - harmonic ratio  [sic; actually twitstruct] -> harmonic_ratio
# - twitstruct.C_cell{:,6}  - relative phrase amplitude vector (ram) -> Phrase.relative_amplitude
# - twitstruct.C_cell{:,7}  - before knee time 25D vector (t1bk) ([0,1]) -> Phrase.time_before_knee
# - twitstruct.C_cell{:,8}  - after knee time 10D vector  (t1ak) ([0,1]) -> Phrase.time_after_knee
# - twitstruct.C_cell{:,11} - AM1 contour before knee (a1bk) ([0,1]) -> Phrase.am1_before_knee
# - twitstruct.C_cell{:,12} - AM1 contour after knee  (a1ak) ([0,1]) -> Phrase.am1_after_knee
# - twitstruct.C_cell{:,13} - AM2 contour before knee (a2bk) ([0,1]) -> Phrase.am2_before_knee
# - twitstruct.C_cell{:,14} - AM2 contour after knee  (a2ak) ([0,1]) -> Phrase.am2_after_knee
# - twitstruct.C_cell{:,15} - same as twitstruct.C_cell{:,7}  (t1bk) ([0,1]) -> Phrase.am_time_before_knee
# - twitstruct.C_cell{:,16} - same as twitstruct.C_cell{:,8}  (t1ak) ([0,1]) -> Phrase.am_time_after_knee
#
# ------ (3) variable scalar parm ---------------------------------------------
# ------  Directly modified by users ------------------------------------------
#
# - twitstruct.nphr         - phrase number -> phrase_count
# - twitstruct.fc           - center frequency               (Hz) -> center_frequency
# - twitstruct.bw           - mean mid-phrase bandwidth      (Hz) -> mean_phrase_bandwidth
# - twitstruct.IPI          - mean inter-phrase interval     (sec) -> inter_phrase_interval
# - twitstruct.tphr         - mean phrase sweep time         (sec) -> mean_phrase_sweep_time
# - twitstruct.tknee        - mean time fraction of knee     ([0,1]) -> mean_knee_time_fraction
# - twitstruct.fknee        - mean freq fraction of knee     ([0,1]) -> mean_knee_frequency_fraction
# - twitstruct.rAttn        - harmonic attenuation           (dB) -> harmonic_attenuation
# - twitstruct.snr          - Signal-to-Noise Ratio          (dB) -> signal_to_noise_ratio
# - twitstruct.bpbw         - noise bandwidth                (oct) -> noise_bandwidth
# - twitstruct.seed         - psuedo random seed -> seed
# - twitstruct.opt.ord      - phrase order   (0: normal, 1: reversed) -> options.reverse_order
# - twitstruct.opt.noam     - AM mod         (0: normal, 1: flat AM) -> options.flat_am
# - twitstruct.opt.nofm     - (unused, always 0) -> options.flat_fm
# - twitstruct.opt.chop     - phrase chop    (0: normal, 1: before knee only,
#                                            2: after knee only) -> options.chop
# - twitstruct.cont         - multi-phrase source (1: count from beginning,
#                                                 2: replicate mid phrase) -> multi_phrase_source
# - twitstruct.phr1         - one-phrase source   (1: 1st, 2: mid, 3: last) -> single_phrase_source
# ==============================================================================


class Chop(IntEnum):
    """Phrase chop mode (``twitstruct.opt.chop``)."""

    NORMAL = 0  # full phrase
    BEFORE_KNEE = 1  # before knee only
    AFTER_KNEE = 2  # after knee only


class CallOptions(BaseModel):
    """Modulation options of a twitter template (``twitstruct.opt``)."""

    model_config = ConfigDict(frozen=True)

    reverse_order: bool  # phrase order (False: normal, True: reversed)
    flat_am: bool  # AM mod (False: normal, True: flat AM)
    flat_fm: int  # (unused, always 0)
    chop: Chop  # phrase chop (0: normal, 1: before knee, 2: after knee)


class TimeReversal(BaseModel):
    """Independent time-reversal options for FM and AM contours."""

    model_config = ConfigDict(frozen=True)

    fm: bool = False
    am: bool = False


class Phrase(BaseModel):
    """One row of ``C_cell``: parameters of a single twitter phrase.

    Mirrors one row of the MATLAB ``C_cell`` cell array (16 columns):
    0 fst, 1 fsp, 2 fkn, 3 tkn, 4 tsw, 5 ram (scalars); 6-15 contour
    vectors (25 pts before knee, 10 pts after knee).
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    start_frequency: float = Field(gt=0)  # starting frequency (fst)  (Hz)
    end_frequency: float = Field(gt=0)  # ending frequency   (fsp)   (Hz)
    knee_frequency_fraction: float = Field(ge=0, le=1)  # (fkn)
    knee_time_fraction: float = Field(ge=0, le=1)  # (tkn)
    sweep_time: float = Field(gt=0)  # phrase sweep time (tsw)        (sec)
    relative_amplitude: float = Field(ge=0, le=1)  # (ram) ([0,1])
    time_before_knee: tuple[Annotated[float, Field(ge=0, le=1)], ...] = Field(
        min_length=25, max_length=25
    )  # (t1bk)
    time_after_knee: tuple[Annotated[float, Field(ge=0, le=1)], ...] = Field(
        min_length=10, max_length=10
    )  # (t1ak)
    frequency_before_knee: tuple[float, ...] = Field(
        min_length=25, max_length=25
    )  # (f1bk)
    frequency_after_knee: tuple[float, ...] = Field(
        min_length=10, max_length=10
    )  # (f1ak)
    am1_before_knee: tuple[Annotated[float, Field(ge=0, le=1)], ...] = Field(
        min_length=25, max_length=25
    )  # (a1bk)
    am1_after_knee: tuple[Annotated[float, Field(ge=0, le=1)], ...] = Field(
        min_length=10, max_length=10
    )  # (a1ak)
    am2_before_knee: tuple[Annotated[float, Field(ge=0, le=1)], ...] = Field(
        min_length=25, max_length=25
    )  # (a2bk)
    am2_after_knee: tuple[Annotated[float, Field(ge=0, le=1)], ...] = Field(
        min_length=10, max_length=10
    )  # (a2ak)
    am_time_before_knee: tuple[Annotated[float, Field(ge=0, le=1)], ...] = Field(
        min_length=25, max_length=25
    )  # (t1bkam)
    am_time_after_knee: tuple[Annotated[float, Field(ge=0, le=1)], ...] = Field(
        min_length=10, max_length=10
    )  # (t1akam)


class TwitStruct(BaseModel):
    """Parameters of a twitter call (TWIT), mirroring the MATLAB
    ``twitstruct`` fields (Python field names in full style)."""

    model_config = ConfigDict(frozen=True)

    center_frequency: float = Field(gt=0)  # (fc) Hz
    inter_phrase_interval: float = Field(ge=0)  # (IPI) sec
    harmonic_ratio: int = Field(gt=0)  # (f2f1)

    harmonic_amplitude_ratio: float = Field(gt=0, le=1)  # (A2A1)
    phrase_bandwidths: tuple[Annotated[float, Field(gt=0)], ...]  # (bwv) Hz
    mean_phrase_bandwidth: float = Field(ge=0)  # (bw) Hz

    # Phrase parameters, one per phrase (max 9 naturally occurring), each
    # a :class:`Phrase` mirroring one row of the MATLAB ``C_cell`` cell array
    # (16 columns: 0 fst, 1 fsp, 2 fkn, 3 tkn, 4 tsw, 5 ram, 6 t1bk (25 pts),
    # 7 t1ak (10 pts), 8 f1bk (25), 9 f1ak (10), 10 a1bk (25), 11 a1ak (10),
    # 12 a2bk (25), 13 a2ak (10), 14 t1bkam (25), 15 t1akam (10)).
    phrase: dict[int, Phrase]

    phrase_count: int = Field(ge=0)  # (nphr)
    phrase_sweep_times: tuple[Annotated[float, Field(gt=0)], ...]  # (tsw) sec
    knee_time_fractions: tuple[Annotated[float, Field(ge=0, le=1)], ...]  # (tkn)
    knee_frequency_fractions: tuple[Annotated[float, Field(ge=0, le=1)], ...]  # (fkn)

    mean_phrase_sweep_time: float = Field(gt=0)  # (tphr) sec
    mean_knee_time_fraction: float = Field(ge=0, le=1)  # (tknee) [0,1]
    mean_knee_frequency_fraction: float = Field(ge=0)  # (fknee) [0,1]
    duration: float = Field(ge=0)  # (dur) sec

    signal_to_noise_ratio: float = Field(ge=0)  # (snr) dB
    noise_bandwidth: float = Field(ge=0)  # (bpbw) oct
    seed: int = Field(gt=0)  # pseudo random seed
    harmonic_attenuation: float = Field(le=0)  # (rAttn) dB

    multi_phrase_source: Literal[1, 2]  # (cont) count or replicate middle
    single_phrase_source: Literal[1, 2, 3]  # (phr1) first, middle, or last

    options: CallOptions  # (opt) modulation options
    time_reversal: TimeReversal  # (rev) independent FM/AM time reversal

    structure_source: Literal["ALL"]  # (mnk)

    @model_validator(mode="after")
    def validate_structure(self) -> Self:
        """Validate the alignment of per-phrase template data."""
        phrase_count = len(self.phrase)
        if phrase_count == 0:
            raise ValueError("phrase must not be empty")
        if tuple(self.phrase) != tuple(range(phrase_count)):
            raise ValueError("phrase keys must be consecutive integers starting at 0")

        vector_lengths = {
            len(self.phrase_bandwidths),
            len(self.phrase_sweep_times),
            len(self.knee_time_fractions),
            len(self.knee_frequency_fractions),
        }
        if 0 in vector_lengths:
            raise ValueError("per-phrase vectors must not be empty")
        if vector_lengths != {phrase_count}:
            raise ValueError("per-phrase vectors must match the phrase count")
        return self

    @classmethod
    def load(cls, path: Path = TWITTER) -> Self:
        """Load the twitstruct template from a JSON file (default: bundled)."""
        with path.open(encoding="utf-8") as f:
            return cls.model_validate(json.load(f))


class TwitterDTO(BaseModel):
    """User-adjustable parameters of a twitter call (DTO).

    Only the parameters directly editable in the VirtualCall GUI TWIT panel
    (VirtualCall.m callbacks; see the ``twitstruct`` parameter list above).
    Everything else (``phrase``/``C_cell`` contours, ``phrase_sweep_times``/
    ``knee_time_fractions``/``knee_frequency_fractions``/``phrase_bandwidths``,
    ``harmonic_ratio``, ``harmonic_amplitude_ratio``, ``multi_phrase_source``,
    ``single_phrase_source``, ``structure_source``, computed ``duration``) is
    template-inherent data and stays in :class:`TwitStruct`.

    Field constraints mirror the MATLAB GUI callback bounds checks
    (VirtualCall.m ``CF_Callback`` ... ``tTrans_Callback``). The sample rate
    is a GUI global in MATLAB (``handles.SR``, operative default 48000, not a
    twitstruct field); it is folded into the DTO so the parameter model fully
    defines one stimulus.
    """

    model_config = ConfigDict(frozen=True)

    # ---- scalar parameters (GUI TWIT panel, all required) --------------
    sample_rate: int = Field(default=48_000, gt=0)  # (sr) Hz; MATLAB handles.SR
    center_frequency: float = Field(gt=0)  # (fc) Hz; GUI: (0, SR/2), SR-dependent
    inter_phrase_interval: float = Field(ge=0)  # (IPI) sec
    mean_phrase_sweep_time: float = Field(gt=0)  # (tphr) sec
    phrase_count: int = Field(ge=0)  # (nphr) GUI floors to int
    harmonic_attenuation: float = Field(le=0)  # (rAttn) dB; A2A1 = 10^(rAttn/20)
    signal_to_noise_ratio: float = Field(ge=0)  # (snr) dB
    noise_bandwidth: float = Field(ge=0)  # (bpbw) oct
    seed: int = Field(gt=0)  # pseudo random seed (GUI floors to int)
    mean_phrase_bandwidth: float = Field(ge=0)  # (bw) Hz
    mean_knee_frequency_fraction: float = Field(ge=0)  # (fknee) [0,1]; GUI: >=0
    mean_knee_time_fraction: float = Field(ge=0, le=1)  # (tknee) [0,1]

    # ---- modulation options (GUI buttons, default = normal) -------------
    time_reversal: TimeReversal = TimeReversal()  # (rev)
    reverse_order: bool = False  # (ord) phrase order (False: normal, True: reversed)
    flat_am: bool = False  # (noam) AM mod (False: normal, True: flat AM)
    chop: Chop = Chop.NORMAL  # phrase chop (0: normal, 1: before knee, 2: after knee)
