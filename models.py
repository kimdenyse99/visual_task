import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel

type AnimalName = str
type LevelID = int


class _AnimalConfig(TypedDict):
    levels: dict[str, object]


class TouchEvent(BaseModel):
    x: int
    y: int
    timestamp: float


def path(base_path: str):
    return Path(base_path).parent / "config.json"


def validate_level_ids(level_ids: Iterable[int]) -> list[int]:
    sorted_level_ids = sorted(level_ids)
    expected_level_ids = list(range(1, len(sorted_level_ids) + 1))
    if sorted_level_ids != expected_level_ids:
        raise ValueError("level IDs must start at 1 and be consecutive")
    return sorted_level_ids


def load_level_table(config_path: Path) -> dict[str, list[int]]:
    with open(config_path) as f:
        data: dict[str, _AnimalConfig] = json.load(f)
    return {
        name: validate_level_ids(int(k) for k in config.get("levels", {}))
        for name, config in data.items()
    }


@dataclass
class Screen:
    width: int
    height: int


class ResultEnum(StrEnum):
    PENDING = auto()
    CORRECT = auto()
    INCORRECT = auto()
    BACKGROUND_TOUCHED = auto()
    TIMEOUT = auto()
    CANCELLED = auto()
