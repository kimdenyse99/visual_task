# pyright: reportUnknownMemberType=false

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from stages.visual_two_choice_training.models import Result, ResultEnum

type AnimalName = str

VALID_RESULTS = {
    ResultEnum.CORRECT,
    ResultEnum.INCORRECT,
    ResultEnum.BACKGROUND_TOUCHED,
}


@dataclass(frozen=True)
class StimulusPerformance:
    animal: str
    stimulus: str
    correct_trials: int
    valid_trials: int

    @property
    def correct_rate(self) -> float:
        return self.correct_trials / self.valid_trials


def calculate_stimulus_performance(
    data: Mapping[AnimalName, Sequence[Result]],
) -> list[StimulusPerformance]:
    """Calculate accuracy using the experiment's valid-outcome denominator."""
    counts: dict[tuple[str, str], list[int]] = {}
    for animal, records in data.items():
        for trial in records:
            if trial.result not in VALID_RESULTS:
                continue
            stimulus = "Natural vocalization" if trial.is_twitter_trial else "Pure tone"
            correct_and_total = counts.setdefault((animal, stimulus), [0, 0])
            correct_and_total[1] += 1
            if trial.result is ResultEnum.CORRECT:
                correct_and_total[0] += 1

    return [
        StimulusPerformance(
            animal=animal,
            stimulus=stimulus,
            correct_trials=correct,
            valid_trials=total,
        )
        for (animal, stimulus), (correct, total) in sorted(counts.items())
    ]


def plot_stimulus_performance(
    data: Mapping[AnimalName, Sequence[Result]],
) -> Figure:
    performance = calculate_stimulus_performance(data)
    figure, axes = plt.subplots(figsize=(8, 4))
    sns.barplot(
        x=[item.animal for item in performance],
        y=[item.correct_rate for item in performance],
        hue=[item.stimulus for item in performance],
        hue_order=["Natural vocalization", "Pure tone"],
        errorbar=None,
        ax=axes,
    )
    axes.set(
        title="Performance by Auditory Stimulus",
        xlabel="Animal",
        ylabel="Correct Rate",
        ylim=(0, 1),
    )
    axes.legend(title="Stimulus", fontsize=8)
    return figure
