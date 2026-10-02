# pyright: reportUnknownMemberType=false

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mxbiflow.models.session import Session

from stages.visual_two_choice_training.models import Result, ResultEnum

type AnimalName = str


_RESULT_MARKERS = {
    ResultEnum.CORRECT.value: "o",
    ResultEnum.INCORRECT.value: "X",
    ResultEnum.BACKGROUND_TOUCHED.value: "s",
    ResultEnum.TIMEOUT.value: "^",
    ResultEnum.CANCELLED.value: "D",
    ResultEnum.PENDING.value: "P",
}


def plot_touch_positions(
    session: Session,
    data: Mapping[AnimalName, Sequence[Result]],
) -> Figure:
    title = "Touch Positions"
    touch_rows = [
        (
            touch.x,
            touch.y,
            animal,
            trial.result.value,
        )
        for animal, records in data.items()
        for trial in records
        for touch in trial.touch_event
    ]
    width, height = session.mxbi_config.screen_size

    figure, axes = plt.subplots(figsize=(8, 5))
    if touch_rows:
        sns.scatterplot(
            x=[row[0] for row in touch_rows],
            y=[row[1] for row in touch_rows],
            hue=[row[2] for row in touch_rows],
            hue_order=sorted({row[2] for row in touch_rows}),
            style=[row[3] for row in touch_rows],
            style_order=list(_RESULT_MARKERS),
            markers=_RESULT_MARKERS,
            alpha=0.65,
            s=55,
            ax=axes,
        )

    axes.add_patch(
        Rectangle(
            (0, 0),
            width,
            height,
            fill=False,
            linestyle="-",
            linewidth=1.5,
            edgecolor="black",
            label=f"Screen {width}×{height}",
        )
    )

    target_size = width // 5
    target_center_x = width * 3 // 4
    target_centers_y = (height // 3, height // 3 * 2)
    for index, center_y in enumerate(target_centers_y):
        axes.add_patch(
            Rectangle(
                (
                    target_center_x - target_size / 2,
                    center_y - target_size / 2,
                ),
                target_size,
                target_size,
                fill=False,
                linestyle="--",
                linewidth=1.5,
                edgecolor="black",
                label=(
                    f"Visual targets {width}×{height}" if index == 0 else "_nolegend_"
                ),
            )
        )

    x_values = [0, width]
    y_values = [0, height]
    x_values.extend(row[0] for row in touch_rows)
    y_values.extend(row[1] for row in touch_rows)
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    padding = max(x_max - x_min, y_max - y_min, 1) * 0.03

    axes.set(
        title=title,
        xlabel="X (px)",
        ylabel="Y (px)",
        xlim=(x_min - padding, x_max + padding),
        ylim=(y_max + padding, y_min - padding),
    )
    axes.set_aspect("equal")
    axes.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    return figure
