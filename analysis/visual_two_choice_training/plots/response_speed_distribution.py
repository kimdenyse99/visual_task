# pyright: reportUnknownMemberType=false

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from stages.visual_two_choice_training.models import (
    ResponseSpeed,
    Result,
    ResultEnum,
)

type AnimalName = str

RESPONSE_SPEED_ORDER = list(ResponseSpeed)
RESPONSE_SPEED_PALETTE = {
    ResponseSpeed.FAST.value: "#2ca02c",
    ResponseSpeed.MEDIUM.value: "#ff7f0e",
    ResponseSpeed.SLOW.value: "#d62728",
}


def plot_response_speed_distribution(
    data: Mapping[AnimalName, Sequence[Result]],
) -> Figure:
    rows = [
        (animal, trial.response_speed.value)
        for animal, records in data.items()
        for trial in records
        if trial.result is ResultEnum.CORRECT and trial.response_speed is not None
    ]

    figure, axes = plt.subplots(figsize=(8, 4))
    if rows:
        sns.countplot(
            x=[row[0] for row in rows],
            hue=[row[1] for row in rows],
            hue_order=[speed.value for speed in RESPONSE_SPEED_ORDER],
            palette=RESPONSE_SPEED_PALETTE,
            ax=axes,
        )
        axes.legend(title="Response Speed", fontsize=8, loc="upper right")
    axes.set(
        title="Correct Response Speed Distribution",
        xlabel="Animal",
        ylabel="Correct Trial Count",
    )
    return figure
