"""A looping animated-GIF sprite, as a drop-in alternative to
mxbiflow.assets.ImageSprite for stages that want a moving target instead
of a static image (e.g. a marmoset GIF instead of a still marmoset
photo).

pygame itself has no native multi-frame GIF support: pygame.image.load()
only ever decodes a GIF's first frame, never the animation (confirmed
directly against this environment's pygame-ce 2.5.8 build - there is no
pygame.movie or equivalent here). This class uses Pillow (PIL) instead,
purely for decoding: it opens the GIF, walks every frame via
PIL.Image.seek(), reads each frame's own per-frame duration from the
GIF's own metadata (GIFs do not have to use a single fixed frame rate -
each frame can specify its own duration), and converts each decoded
frame into a pygame.Surface once, up front, at construction time via
pygame.image.frombytes (the current, non-deprecated conversion function
in this pygame-ce version - pygame.image.fromstring also works but is
flagged deprecated as of 2.3.0). From then on this is pure pygame: no PIL
object is kept around, and no decoding happens again during playback -
update() just advances a timer and swaps self.image to a pre-converted
Surface already sitting in memory, which is cheap enough to call every
frame like any other sprite.

Requires Pillow (PIL) - see pyproject.toml.
"""

from pathlib import Path

from PIL import Image
from pygame import Surface
from pygame.image import frombytes as image_frombytes
from pygame.sprite import Sprite
from pygame.transform import scale


class AnimatedGifSprite(Sprite):
    """Decodes every frame of a GIF up front, then loops them
    continuously once update(dt_s) starts being called each frame.

    Mirrors mxbiflow.assets.ImageSprite's constructor shape (path, size,
    pos) and its self.image/self.rect attributes, so it can generally
    replace an ImageSprite(...) call directly in a stage's
    _init_visual_stimulus - the one addition a caller must make is
    calling sprite.update(dt_s) once per frame (e.g. alongside the
    scene's existing self._sprites.update(dt_s) call, if the sprite is
    added to a pygame.sprite.Group that already calls update() on its
    members, or directly from the scene's own update() otherwise), since
    a plain ImageSprite never needed per-frame advancement at all.
    """

    def __init__(
        self,
        path: Path,
        size: int | None = None,
        pos: tuple[int, int] = (0, 0),
    ) -> None:
        super().__init__()

        self._frames: list[Surface] = []
        self._durations_ms: list[float] = []

        with Image.open(path) as gif:
            frame_index = 0
            while True:
                try:
                    gif.seek(frame_index)
                except EOFError:
                    break

                frame = gif.convert("RGBA")
                duration_ms = frame.info.get("duration", 100)
                # Some GIFs declare a 0ms (or missing) duration for one
                # or more frames, which would otherwise make this frame
                # flash by in a single update() call, effectively
                # invisible - clamp to a sane minimum instead.
                duration_ms = max(20, duration_ms)

                surface = image_frombytes(frame.tobytes(), frame.size, frame.mode)
                if size is not None:
                    surface = scale(surface, (size, size))

                self._frames.append(surface)
                self._durations_ms.append(duration_ms)
                frame_index += 1

        if not self._frames:
            raise ValueError(f"No frames decoded from GIF: {path}")

        self._current_index = 0
        self._elapsed_in_frame_ms = 0.0

        self.image = self._frames[0]
        self.rect = self.image.get_rect(center=pos)

    def update(self, dt_s: float) -> None:
        """Advance playback by dt_s seconds, looping continuously. Call
        this once per frame for as long as this sprite is visible -
        unlike a static ImageSprite, this sprite's self.image will not
        change on its own otherwise."""
        if len(self._frames) <= 1:
            return

        self._elapsed_in_frame_ms += dt_s * 1000

        while self._elapsed_in_frame_ms >= self._durations_ms[self._current_index]:
            self._elapsed_in_frame_ms -= self._durations_ms[self._current_index]
            self._current_index = (self._current_index + 1) % len(self._frames)

        center = self.rect.center
        self.image = self._frames[self._current_index]
        self.rect = self.image.get_rect(center=center)
