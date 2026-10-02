# pyright: reportUnknownMemberType=false

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from stages.visual_two_choice_training.models import Result

type AnimalName = str


def plot_trial_activity(data: Mapping[AnimalName, Sequence[Result]]) -> Figure:
    rows = [
        (trial.trial_start_time, animal)
        for animal, trials in data.items()
        for trial in trials
    ]

    bin_seconds = 10 * 60
    bin_starts = [
        int(start_time // bin_seconds) * bin_seconds for start_time, _ in rows
    ]
    has_multiple_dates = (
        len(
            {
                datetime.fromtimestamp(value, UTC).astimezone().date()
                for value in bin_starts
            }
        )
        > 1
    )
    label_format = "%m-%d %H:%M" if has_multiple_dates else "%H:%M"
    labels = [
        datetime.fromtimestamp(value, UTC).astimezone().strftime(label_format)
        for value in bin_starts
    ]

    figure, axes = plt.subplots(figsize=(8, 4))
    sns.countplot(
        x=labels,
        hue=[animal for _, animal in rows],
        order=list(dict.fromkeys(labels)),
        ax=axes,
    )
    axes.set(
        title="Trial Activity (10-minute bins)",
        xlabel="Time Bin",
        ylabel="Trial Count",
    )
    axes.tick_params(axis="x", rotation=45)
    axes.legend(title="Animal", fontsize=8)
    return figure
