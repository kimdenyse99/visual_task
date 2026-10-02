"""CLI launcher for the marimo explorer: ``python -m py_virtualcall``."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import cyclopts

_EXPLORER = Path(__file__).resolve().parent / "marimo" / "notebook.py"

app = cyclopts.App(
    name="py-virtualcall",
    help="Launch the py-virtualcall marimo explorer.",
)


@app.default
def main(edit: bool = False) -> int:
    """Launch the marimo explorer, or open it for editing with --edit."""
    if importlib.util.find_spec("marimo") is None:
        print(
            "marimo is not installed; run `uv sync` to install it.",
            file=sys.stderr,
        )
        return 1
    mode = "edit" if edit else "run"
    return subprocess.run(
        [sys.executable, "-m", "marimo", mode, str(_EXPLORER)], check=False
    ).returncode


if __name__ == "__main__":
    app()
