"""Plot generators for the visual two-choice training report."""

import matplotlib

matplotlib.use("Agg")

from .cumulative_correct_rate import plot_cumulative_correct_rate
from .response_speed_distribution import plot_response_speed_distribution
from .result_distribution import plot_result_distribution
from .stimulus_performance import (
    StimulusPerformance,
    calculate_stimulus_performance,
    plot_stimulus_performance,
)
from .touch_positions import plot_touch_positions
from .trial_activity import plot_trial_activity

__all__ = [
    "StimulusPerformance",
    "calculate_stimulus_performance",
    "plot_cumulative_correct_rate",
    "plot_response_speed_distribution",
    "plot_result_distribution",
    "plot_stimulus_performance",
    "plot_touch_positions",
    "plot_trial_activity",
]
