# py-virtualcall

Marmoset virtual vocalization synthesizer, ported from Wang Lab VirtualCall
V2.0 (MATLAB) based on DiMattina & Wang (2006, J. Neurophysiol.).

## Access and installation

This private repository is provided only for authorized internal use. Do not
publish or redistribute its source code or derived copies outside the existing
authorized group.

Clone or include the repository through the team's private GitHub access, then
install its checked-out source with `uv`:

```bash
uv sync  # includes the marimo explorer app (marimo + matplotlib)
```

## Visual Explorer

A marimo notebook app (`src/py_virtualcall/marimo/notebook.py`) is bundled
with the package: form-based twitter synthesis through the full pipeline,
waveform preview, and wav export.

```bash
python -m py_virtualcall         # run the app: opens in the browser,
                                 # all cells run automatically
python -m py_virtualcall --edit  # edit the notebook
```

**Important**: in *edit* mode cells do **not** run automatically — press
"Run all cells" (top-right ▶▶ button, or `Cmd+Shift+Enter`) after opening.
Use plain `python -m py_virtualcall` (run mode) for a read-only app that
executes everything on load.

Without the launcher, run the notebook directly with the marimo CLI:

```bash
marimo run src/py_virtualcall/marimo/notebook.py
```

## Tests

```bash
uv run python -m unittest discover -s tests
```
