# pyright: reportPrivateUsage=false
"""Tests for the ``python -m py_virtualcall`` launcher."""

import unittest
from pathlib import Path
from unittest import mock

from py_virtualcall.__main__ import _EXPLORER, app, main

_EXPLORER_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "py_virtualcall"
    / "marimo"
    / "notebook.py"
)


class TestMain(unittest.TestCase):
    def test_default_run_mode(self) -> None:
        with mock.patch("subprocess.run", return_value=mock.Mock(returncode=0)) as run:
            self.assertEqual(main(), 0)
        command = run.call_args.args[0]
        self.assertEqual(command[1:4], ["-m", "marimo", "run"])
        self.assertEqual(Path(command[4]), _EXPLORER_PATH)

    def test_edit_mode(self) -> None:
        with mock.patch("subprocess.run", return_value=mock.Mock(returncode=0)) as run:
            self.assertEqual(main(edit=True), 0)
        command = run.call_args.args[0]
        self.assertEqual(command[3], "edit")

    def test_propagates_returncode(self) -> None:
        with mock.patch("subprocess.run", return_value=mock.Mock(returncode=3)) as run:
            self.assertEqual(main(), 3)
        run.assert_called_once()

    def test_missing_marimo(self) -> None:
        with (
            mock.patch(
                "py_virtualcall.__main__.importlib.util.find_spec", return_value=None
            ),
            mock.patch("subprocess.run") as run,
        ):
            self.assertEqual(main(), 1)
        run.assert_not_called()

    def test_explorer_path(self) -> None:
        self.assertEqual(_EXPLORER, _EXPLORER_PATH)


class TestApp(unittest.TestCase):
    """End-to-end CLI dispatch: cyclopts parses argv and runs the real main."""

    def _launch(self, argv: list[str]) -> tuple[int, mock.Mock]:
        with (
            mock.patch(
                "py_virtualcall.__main__.importlib.util.find_spec",
                return_value=mock.MagicMock(),
            ),
            mock.patch(
                "py_virtualcall.__main__.subprocess.run",
                return_value=mock.Mock(returncode=0),
            ) as run,
            self.assertRaises(SystemExit) as ctx,
        ):
            app(argv)
        code = ctx.exception.code
        assert isinstance(code, int)
        return code, run

    def test_default_dispatch(self) -> None:
        code, run = self._launch([])
        self.assertEqual(code, 0)
        command = run.call_args.args[0]
        self.assertEqual(command[1:4], ["-m", "marimo", "run"])
        self.assertEqual(Path(command[4]), _EXPLORER_PATH)

    def test_edit_dispatch(self) -> None:
        code, run = self._launch(["--edit"])
        self.assertEqual(code, 0)
        command = run.call_args.args[0]
        self.assertEqual(command[3], "edit")

    def test_returncode_exit(self) -> None:
        with (
            mock.patch(
                "py_virtualcall.__main__.importlib.util.find_spec",
                return_value=mock.MagicMock(),
            ),
            mock.patch(
                "py_virtualcall.__main__.subprocess.run",
                return_value=mock.Mock(returncode=7),
            ),
            self.assertRaises(SystemExit) as ctx,
        ):
            app([])
        self.assertEqual(ctx.exception.code, 7)


if __name__ == "__main__":
    unittest.main()
