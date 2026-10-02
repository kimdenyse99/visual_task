import json
from pathlib import Path
from typing import cast

from loguru import logger

from stages.visual_two_choice_training.models import Result

type AnimalName = str

DATA_FILENAME = "result.jsonl"
RESULT_VERSIONS = {"1.0.0", "2.0.0"}


def _load_trial_data(stage_path: Path) -> list[Result] | None:
    """Load trial records from one exact stage data directory."""
    data_file = stage_path / DATA_FILENAME
    if not data_file.is_file():
        return None

    try:
        lines = data_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        logger.exception("failed to read trial data: {}", data_file)
        return None

    trials: list[Result] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw_payload = json.loads(line)
            if not isinstance(raw_payload, dict):
                raise TypeError("trial record must be a JSON object")
            payload = cast(dict[str, object], raw_payload)
            if payload.get("version") not in RESULT_VERSIONS:
                continue
            trials.append(Result.model_validate(payload))
        except (TypeError, ValueError) as exc:
            logger.warning(
                "skipping invalid trial record at {}:{}: {}",
                data_file,
                line_number,
                exc,
            )

    if not trials:
        return None
    return sorted(trials, key=lambda trial: trial.trial_start_time)


def load_stage_trial_data_for_animal(
    animal_stage_path: tuple[AnimalName, Path],
) -> tuple[AnimalName, list[Result]] | None:
    """Load one animal's trial records from its session stage directory."""
    animal, stage_path = animal_stage_path
    trials = _load_trial_data(stage_path)
    return (animal, trials) if trials else None
