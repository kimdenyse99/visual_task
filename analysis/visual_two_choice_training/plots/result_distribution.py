# pyright: reportUnknownMemberType=false

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from stages.visual_two_choice_training.models import Result

from ._common import RESULT_ORDER, RESULT_PALETTE

type AnimalName = str


def plot_result_distribution(
    data: Mapping[AnimalName, Sequence[Result]],
) -> Figure:
    rows = [
        (animal, trial.result.value)
        for animal, records in data.items()
        for trial in records
    ]

    figure, axes = plt.subplots(figsize=(8, 4))
    sns.countplot(
        x=[row[0] for row in rows],
        hue=[row[1] for row in rows],
        hue_order=[result.value for result in RESULT_ORDER],
        palette=RESULT_PALETTE,
        ax=axes,
    )
    axes.set(title="Trial Result Distribution", xlabel="Animal", ylabel="Trial Count")
    axes.legend(title="Result", fontsize=8, loc="upper right")
    return figure
