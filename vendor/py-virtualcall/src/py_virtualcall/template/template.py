"""Shared types for call parameter templates."""

from collections.abc import Sequence
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import BeforeValidator


def _to_numpy_float64_array(
    arr: Sequence[float] | NDArray[np.generic],
) -> NDArray[np.float64]:
    return np.array(arr, dtype=np.float64, copy=True, order="C")


type Float64Array = Annotated[
    NDArray[np.float64], BeforeValidator(_to_numpy_float64_array)
]
