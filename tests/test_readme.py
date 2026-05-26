from __future__ import annotations

import unittest
from pathlib import Path


class ReadmeTests(unittest.TestCase):
    def test_readme_mentions_distribution_calibration_workflow(self) -> None:
        source = Path("README.md").read_text(encoding="utf-8").lower()

        self.assertIn("--distribution-calibration", source)
        self.assertIn("--artifact-dir", source)
        self.assertIn("raw vs calibrated pit", source)

