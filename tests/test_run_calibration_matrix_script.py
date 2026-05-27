from __future__ import annotations

import subprocess
from pathlib import Path
import unittest


class CalibrationMatrixScriptTests(unittest.TestCase):
    def test_run_calibration_matrix_script_exists_and_is_shell_valid(self) -> None:
        script_path = Path("run_calibration_matrix.sh")
        script_text = script_path.read_text(encoding="utf-8")

        self.assertIn("--no-interval-calibration --no-pit-calibration", script_text)
        self.assertIn("--interval-calibration --no-pit-calibration", script_text)
        self.assertIn("--no-interval-calibration --pit-calibration", script_text)
        self.assertIn("--interval-calibration --pit-calibration", script_text)

        result = subprocess.run(
            ["bash", "-n", str(script_path)],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
