from pathlib import Path

_REAL_MARMOSET_IMAGE_DIR = (
    Path(__file__).parent / "visual_stimulus" / "real_marmoset_images"
)
M001 = _REAL_MARMOSET_IMAGE_DIR / "m001.png"
M002 = _REAL_MARMOSET_IMAGE_DIR / "m002.png"
MARMOSET_IMAGE_PATHS = sorted(_REAL_MARMOSET_IMAGE_DIR.glob("*.png"))

_SYNTHETIC_IMAGE_DIR = Path(__file__).parent / "visual_stimulus" / "synthetic_images"
GFG001 = _SYNTHETIC_IMAGE_DIR / "gfG001.png"

_AUDITORY_STIMULUS_DIR = Path(__file__).parent / "auditory_stimulus"
NATURAL_TWITTER_PATHS = sorted(_AUDITORY_STIMULUS_DIR.glob("natural-twitter-*.wav"))


__all__ = [
    "GFG001",
    "M001",
    "M002",
    "MARMOSET_IMAGE_PATHS",
    "NATURAL_TWITTER_PATHS",
]
