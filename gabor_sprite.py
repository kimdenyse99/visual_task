"""A static Gabor patch sprite: a sinusoidal grating under a Gaussian window.

Pixel values are computed as

    value = background * (1 + contrast * envelope(x, y) * cos(2*pi*f*x' + phase))

where x' is the position along the grating's direction (rotated by
orientation_deg), f is the spatial frequency in cycles per pixel and the
envelope is a Gaussian that is 1 at the center. ``contrast`` is the
Michelson contrast of the underlying grating, (max - min) / (max + min),
before the Gaussian window fades it toward the background - the usual way
a Gabor's contrast is specified. Because the window also dims the troughs
next to the center, the contrast measured on the finished patch is a little
lower, more so at low spatial frequencies.

Contrast is defined on pixel values, not measured luminance: the display's
gamma is not corrected for, so the physical luminance contrast will differ
unless the monitor is linearized.

The patch is drawn on an opaque square of side ``size_px``. The Gaussian's
standard deviation is size_px / 6, so the window has fallen to about 1% at
the square's edge and the square blends into a background of the same
color.
"""

import math

import numpy as np
from pygame import Surface
from pygame.sprite import Sprite
from pygame.surfarray import make_surface

# Envelope standard deviation as a fraction of the patch size (size / 6
# puts the edge at 3 standard deviations from the center).
SIGMA_FRACTION = 1 / 6


def gabor_pixels(
    size_px: int,
    cycles_per_px: float,
    contrast: float,
    background: tuple[int, int, int],
    orientation_deg: float = 0.0,
    phase_deg: float = 0.0,
) -> np.ndarray:
    """Return a (size_px, size_px, 3) uint8 array indexed [x, y, channel].

    orientation_deg 0 gives a grating whose stripes run along pygame's y
    axis (the luminance varies along x).
    """
    if size_px < 1:
        raise ValueError("size_px must be at least 1")
    if not 0 <= contrast <= 1:
        raise ValueError("contrast must be between 0 and 1")

    coords = np.arange(size_px, dtype=np.float64) - (size_px - 1) / 2
    x, y = np.meshgrid(coords, coords, indexing="ij")

    theta = math.radians(orientation_deg)
    along = x * math.cos(theta) + y * math.sin(theta)
    grating = np.cos(2 * math.pi * cycles_per_px * along + math.radians(phase_deg))

    sigma = size_px * SIGMA_FRACTION
    envelope = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    modulation = 1 + contrast * envelope * grating
    channels = [np.clip(np.rint(c * modulation), 0, 255) for c in background]
    return np.stack(channels, axis=-1).astype(np.uint8)


class GaborSprite(Sprite):
    def __init__(
        self,
        center: tuple[int, int],
        size_px: int,
        cycles_per_px: float,
        contrast: float,
        background: tuple[int, int, int],
        orientation_deg: float = 0.0,
        phase_deg: float = 0.0,
    ) -> None:
        super().__init__()
        pixels = gabor_pixels(
            size_px,
            cycles_per_px,
            contrast,
            background,
            orientation_deg,
            phase_deg,
        )
        self.image: Surface = make_surface(pixels)
        self.rect = self.image.get_rect(center=center)
