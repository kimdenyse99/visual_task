"""Bundled call parameter template assets (.mat), one constant per file.

Follows the pattern of ``2ac-vocalization-discriminate/assets``: paths are
exported as module-level constants, resolved from ``Path(__file__)``.

Only ``twitter.mat`` is referenced by code; the other .mat files (tone, phee,
trill, trill-phee) are kept in this directory as raw template data for future
narrowband call support.
"""

from pathlib import Path

TWITTER = Path(__file__).parent / "twitter.json"

__all__ = ["TWITTER"]
