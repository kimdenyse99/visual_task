import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from pydantic import ValidationError

    from py_virtualcall.delivery import build_virtual_call, write_wav
    from py_virtualcall.synthesizer import TwitterSynthesizer
    from py_virtualcall.template import Chop, TimeReversal, TwitStruct, TwitterDTO

    return (
        Chop,
        Path,
        TimeReversal,
        TwitStruct,
        TwitterDTO,
        TwitterSynthesizer,
        ValidationError,
        build_virtual_call,
        json,
        mo,
        np,
        plt,
        write_wav,
    )


@app.cell
def _(TwitStruct, TwitterDTO, TwitterSynthesizer):
    twit = TwitStruct.load()
    synthesizer = TwitterSynthesizer(twit)

    # Mirrors how the MATLAB GUI backfills its panels from twitstruct on startup.
    default_dto = TwitterDTO(
        center_frequency=twit.center_frequency,
        inter_phrase_interval=twit.inter_phrase_interval,
        mean_phrase_sweep_time=twit.mean_phrase_sweep_time,
        phrase_count=twit.phrase_count,
        harmonic_attenuation=twit.harmonic_attenuation,
        signal_to_noise_ratio=twit.signal_to_noise_ratio,
        noise_bandwidth=twit.noise_bandwidth,
        seed=twit.seed,
        mean_phrase_bandwidth=twit.mean_phrase_bandwidth,
        mean_knee_frequency_fraction=twit.mean_knee_frequency_fraction,
        mean_knee_time_fraction=twit.mean_knee_time_fraction,
        time_reversal=twit.time_reversal,
        reverse_order=twit.options.reverse_order,
        flat_am=twit.options.flat_am,
        chop=twit.options.chop,
    )
    return default_dto, synthesizer


@app.cell
def _(Chop, TimeReversal, TwitterDTO, ValidationError, default_dto, mo):
    def dto_from_form(values) -> TwitterDTO:
        return TwitterDTO(
            center_frequency=values["center_frequency"],
            inter_phrase_interval=values["inter_phrase_interval"],
            mean_phrase_sweep_time=values["mean_phrase_sweep_time"],
            phrase_count=int(values["phrase_count"]),
            harmonic_attenuation=values["harmonic_attenuation"],
            signal_to_noise_ratio=values["signal_to_noise_ratio"],
            noise_bandwidth=values["noise_bandwidth"],
            seed=int(values["seed"]),
            mean_phrase_bandwidth=values["mean_phrase_bandwidth"],
            mean_knee_frequency_fraction=values["mean_knee_frequency_fraction"],
            mean_knee_time_fraction=values["mean_knee_time_fraction"],
            time_reversal=TimeReversal(
                fm=values["reverse_fm"], am=values["reverse_am"]
            ),
            reverse_order=values["reverse_order"],
            flat_am=values["flat_am"],
            chop=chop_by_label[values["chop"]],
        )

    def validate_form(values):
        try:
            dto_from_form(values)
        except ValidationError as e:
            return "; ".join(f"{err['loc'][0]}: {err['msg']}" for err in e.errors())
        return None

    # 1. controls
    center_frequency = mo.ui.number(
        label="center frequency (Hz)",
        start=1,
        stop=24000,
        value=default_dto.center_frequency,
        full_width=True,
    )
    mean_phrase_sweep_time = mo.ui.number(
        label="mean phrase sweep time (s)",
        start=0.001,
        step=0.001,
        value=default_dto.mean_phrase_sweep_time,
        full_width=True,
    )
    inter_phrase_interval = mo.ui.number(
        label="mean inter-phrase interval (s)",
        start=0,
        step=0.001,
        value=default_dto.inter_phrase_interval,
        full_width=True,
    )
    phrase_count = mo.ui.number(
        label="phrase count",
        start=0,
        step=1,
        value=default_dto.phrase_count,
        full_width=True,
    )
    mean_phrase_bandwidth = mo.ui.number(
        label="mean mid-phrase bandwidth (Hz)",
        start=0,
        value=default_dto.mean_phrase_bandwidth,
        full_width=True,
    )
    mean_knee_frequency_fraction = mo.ui.number(
        start=0.001,
        step=0.001,
        label="mean knee frequency fraction",
        value=default_dto.mean_knee_frequency_fraction,
        full_width=True,
    )
    mean_knee_time_fraction = mo.ui.number(
        start=0,
        stop=1,
        step=0.001,
        label="mean knee time fraction",
        value=default_dto.mean_knee_time_fraction,
        full_width=True,
    )
    harmonic_attenuation = mo.ui.number(
        label="harmonic attenuation (dB)",
        stop=0,
        value=default_dto.harmonic_attenuation,
        full_width=True,
    )
    signal_to_noise_ratio = mo.ui.number(
        label="signal-to-noise ratio (dB)",
        start=0,
        value=default_dto.signal_to_noise_ratio,
        full_width=True,
    )
    noise_bandwidth = mo.ui.number(
        label="noise bandwidth (oct)",
        start=0,
        value=default_dto.noise_bandwidth,
        full_width=True,
    )
    seed = mo.ui.number(
        label="random seed",
        start=1,
        step=1,
        value=default_dto.seed,
        full_width=True,
    )
    reverse_fm = mo.ui.checkbox(
        value=default_dto.time_reversal.fm, label="reverse FM in time"
    )
    reverse_am = mo.ui.checkbox(
        value=default_dto.time_reversal.am, label="reverse AM in time"
    )
    reverse_order = mo.ui.checkbox(
        value=default_dto.reverse_order, label="reverse phrase order"
    )
    flat_am = mo.ui.checkbox(value=default_dto.flat_am, label="flat AM")
    chop_labels = {
        Chop.NORMAL: "full",
        Chop.BEFORE_KNEE: "before knee only",
        Chop.AFTER_KNEE: "after knee only",
    }
    chop_by_label = {label: chop for chop, label in chop_labels.items()}
    chop = mo.ui.radio(
        options=list(chop_by_label),
        value=chop_labels[default_dto.chop],
        label="phrase chop",
        inline=True,
    )

    # 2. layout + form
    # NOTE: `.batch()` only substitutes its cloned controls into *templated*
    # HTML. The template must therefore list every control as a placeholder;
    # building a `mo.vstack` of the original controls instead would render the
    # originals in the DOM while the batch tracks the clones, so submissions
    # would silently keep using the default values.
    form = (
        mo.md(
            """
            **Frequency & time**

            {center_frequency}

            {mean_phrase_sweep_time}

            {inter_phrase_interval}

            **Phrases**

            {phrase_count}

            {mean_phrase_bandwidth}

            {mean_knee_frequency_fraction}

            {mean_knee_time_fraction}

            **Acoustics**

            {harmonic_attenuation}

            {signal_to_noise_ratio}

            {noise_bandwidth}

            {seed}

            **Modes**

            {reverse_fm}

            {reverse_am}

            {reverse_order}

            {flat_am}

            {chop}
            """
        )
        .batch(
            center_frequency=center_frequency,
            mean_phrase_sweep_time=mean_phrase_sweep_time,
            inter_phrase_interval=inter_phrase_interval,
            phrase_count=phrase_count,
            mean_phrase_bandwidth=mean_phrase_bandwidth,
            mean_knee_frequency_fraction=mean_knee_frequency_fraction,
            mean_knee_time_fraction=mean_knee_time_fraction,
            harmonic_attenuation=harmonic_attenuation,
            signal_to_noise_ratio=signal_to_noise_ratio,
            noise_bandwidth=noise_bandwidth,
            seed=seed,
            reverse_fm=reverse_fm,
            reverse_am=reverse_am,
            reverse_order=reverse_order,
            flat_am=flat_am,
            chop=chop,
        )
        .form(
            submit_button_label="Synthesize",
            show_clear_button=True,
            validate=validate_form,
        )
    )
    form
    return dto_from_form, form


@app.cell
def _(dto_from_form, form):
    values = form.value
    dto = dto_from_form(values) if values is not None else None
    return (dto,)


@app.cell
def _(build_virtual_call, dto, synthesizer):
    def synthesize_full(dto):
        if dto is None:
            return None
        state = synthesizer.prepare(dto)
        state = synthesizer.scale_mean_gain(state)
        state = synthesizer.reset_phrase_count(state)
        state = synthesizer.update_phrase_frequencies(state)
        state = synthesizer.set_duration(state)
        signal = synthesizer.synthesize_phrases(state)
        signal = synthesizer.normalize_signal(signal)
        signal = synthesizer.add_background_noise(signal, state)
        signal = synthesizer.apply_edge_ramps(signal, state)
        return build_virtual_call(signal, dto)

    call = synthesize_full(dto)
    return (call,)


@app.cell
def _(call, dto, json, mo, np, plt):
    if dto is None or call is None:
        view = mo.md("**Submit the form to synthesize**")
    else:
        npts = len(call.waveform)
        duration = npts / call.sample_rate
        peak = float(np.max(np.abs(call.waveform)))
        rms = float(np.linalg.norm(call.waveform) / np.sqrt(npts))
        stats = mo.md(
            f"""
    **Synthesis complete** · `npts = {npts}` · `duration = {duration:.3f} s`
    · `peak = {peak:.4f}` · `rms = {rms:.4f}`

    **Parameters (TwitterDTO)**

    ```json
    {json.dumps(json.loads(dto.model_dump_json()), indent=2, ensure_ascii=False)}
    ```
    """
        )
        figure, axis = plt.subplots(figsize=(12, 3))
        axis.plot(np.arange(npts) / call.sample_rate * 1000, call.waveform, lw=0.4)
        axis.set_xlabel("time (ms)")
        axis.set_ylabel("amplitude")
        axis.set_title(f"virtual twitter · {npts} samples @ {call.sample_rate} Hz")
        axis.set_xlim(0, duration * 1000)
        view = mo.vstack([stats, figure])
    view
    return


@app.cell
def _(mo):
    wav_name = mo.ui.text(
        value="virtualcall_twitter.wav", label="wav filename", full_width=True
    )

    # marimo buttons keep their value unless on_click updates it, so latch
    # the value to True on the first click to trigger the dependent cell.
    write_button = mo.ui.button(label="Write wav", value=False, on_click=lambda _: True)
    mo.vstack([wav_name, write_button])
    return wav_name, write_button


@app.cell
def _(Path, call, mo, wav_name, write_button, write_wav):
    if write_button.value and call is not None:
        path = Path(wav_name.value)
        write_wav(call, path)
        result = mo.md(f"**Wrote `{path.resolve()}`**")
    elif write_button.value:
        result = mo.md("**Submit the form and synthesize first**")
    else:
        result = mo.md("")
    result
    return


if __name__ == "__main__":
    app.run()
