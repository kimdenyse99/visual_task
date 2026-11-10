from pathlib import Path

_REAL_MARMOSET_IMAGE_DIR = (
    Path(__file__).parent / "visual_stimulus" / "real_marmoset_images"
)
M001 = _REAL_MARMOSET_IMAGE_DIR / "m001.gif"
M002 = _REAL_MARMOSET_IMAGE_DIR / "m002.gif"
M003 = _REAL_MARMOSET_IMAGE_DIR / "m003.gif"
M004 = _REAL_MARMOSET_IMAGE_DIR / "m004.gif"
M005 = _REAL_MARMOSET_IMAGE_DIR / "m005.gif"
M006 = _REAL_MARMOSET_IMAGE_DIR / "m006.gif"
M007 = _REAL_MARMOSET_IMAGE_DIR / "m007.gif"
M008 = _REAL_MARMOSET_IMAGE_DIR / "m008.gif"
M009 = _REAL_MARMOSET_IMAGE_DIR / "m009.gif"
M010 = _REAL_MARMOSET_IMAGE_DIR / "m010.gif"
# The actual source of truth stages should use when picking a random
# target image: every .gif file present in the folder, discovered
# automatically rather than hardcoded, so adding/removing a GIF here
# never requires a code change anywhere else. M001-M010 above are kept as
# named individual paths only for convenience/clarity elsewhere, not used
# directly by the random-selection logic.
MARMOSET_IMAGE_PATHS = sorted(_REAL_MARMOSET_IMAGE_DIR.glob("*.gif"))

_SYNTHETIC_IMAGE_DIR = Path(__file__).parent / "visual_stimulus" / "synthetic_images"
GFG001 = _SYNTHETIC_IMAGE_DIR / "gfG001.png"

_AUDITORY_STIMULUS_DIR = Path(__file__).parent / "auditory_stimulus"
NATURAL_TWITTER_PATHS = sorted(_AUDITORY_STIMULUS_DIR.glob("natural-twitter-*.wav"))


__all__ = [
    "GFG001",
    "M001",
    "M002",
    "M003",
    "M004",
    "M005",
    "M006",
    "M007",
    "M008",
    "M009",
    "M010",
    "MARMOSET_IMAGE_PATHS",
    "NATURAL_TWITTER_PATHS",
]