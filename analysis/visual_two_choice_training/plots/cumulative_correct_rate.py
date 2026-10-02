# pyright: reportUnknownMemberType=false

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from stages.visual_two_choice_training.models import Result, ResultEnum

type AnimalName = str


def plot_cumulative_correct_rate(
    data: Mapping[AnimalName, Sequence[Result]],
) -> Figure:
    rows: list[tuple[int, float, str]] = []
    for animal, records in data.items():
        correct = 0
        completed = 0
        for trial in sorted(records, key=lambda record: record.trial_start_time):
            if trial.result in (
                ResultEnum.CORRECT,
                ResultEnum.INCORRECT,
                ResultEnum.BACKGROUND_TOUCHED,
            ):
                completed += 1
                if trial.result is ResultEnum.CORRECT:
                    correct += 1
            rows.append(
                (
                    trial.animal.trial_id,
                    correct / completed if completed else 0,
                    animal,
                )
            )

    figure, axes = plt.subplots(figsize=(8, 4))
    sns.lineplot(
        x=[row[0] for row in rows],
        y=[row[1] for row in rows],
        hue=[row[2] for row in rows],
        marker="o",
        estimator=None,
        sort=False,
        ax=axes,
    )
    axes.set(
        title="Cumulative Correct Rate",
        xlabel="Global Trial ID",
        ylabel="Correct Rate",
        ylim=(-0.05, 1.05),
    )
    axes.grid(alpha=0.25)
    return figure
