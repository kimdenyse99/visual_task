"""Describe visible task stimuli; the framework timestamps after display flip."""

from typing import Any

from mxbiflow.core import MXBIFlow
from pygame.sprite import Group, Sprite


class StimulusEvents:
    def __init__(self, mxbiflow: MXBIFlow) -> None:
        self._mxbiflow = mxbiflow
        self._visible: dict[str, dict[str, Any]] = {}

    def present(
        self,
        sprites: Group,
        fixation: Sprite,
        target: Sprite | None,
        asset: str | None = None,
        *,
        target_type: str | None = None,
        target_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Call from draw(), once the final visible sprite group is known.

        GIF animation frames are deliberately excluded from the description.
        A sprite added and removed before drawing produces no onset event.

        target_type / target_metadata describe a target that is neither a
        GIF nor a plain circle (e.g. a Gabor patch and its parameters);
        they take precedence over asset when given.
        """
        visible: dict[str, dict[str, Any]] = {}
        for stimulus_id, sprite in (("fixation", fixation), ("target", target)):
            if sprite is None or sprite not in sprites:
                continue
            assert sprite.rect is not None
            stimulus_type = "circle"
            metadata: dict[str, Any] = {}
            if stimulus_id == "target":
                if target_type is not None:
                    stimulus_type = target_type
                    metadata = dict(target_metadata or {})
                elif asset:
                    stimulus_type = "gif"
                    metadata = {"asset": asset}
            visible[stimulus_id] = {
                "stimulus_type": stimulus_type,
                "position": sprite.rect.center,
                "size": sprite.rect.size,
                "metadata": metadata,
            }
        for stimulus_id, details in self._visible.items():
            if stimulus_id not in visible:
                self._mxbiflow.stimulus_event("stimulus_hidden", stimulus_id, **details)
        for stimulus_id, details in visible.items():
            previous = self._visible.get(stimulus_id)
            if previous != details:
                kind = "stimulus_shown" if previous is None else "stimulus_changed"
                self._mxbiflow.stimulus_event(kind, stimulus_id, **details)
        self._visible = visible

    def clear(self) -> None:
        """Hide previously presented stimuli when a scene ends or is interrupted."""
        for stimulus_id, details in self._visible.items():
            self._mxbiflow.stimulus_event("stimulus_hidden", stimulus_id, **details)
        self._visible.clear()
