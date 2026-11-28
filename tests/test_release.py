"""Checks of the release metadata: the licence text and the record's address."""
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# SHA-256 of the PolyForm Strict License 1.0.0 as published by the PolyForm
# Project (tag 1.0.0 of its licence repository), with LF line ends.
LICENCE_SHA256 = "9eb48619fbc193ab7bb327b090cfcc703000265b83e670f81f231d0b1c43c56e"


class Release(unittest.TestCase):
    def test_licence_text_is_unchanged(self):
        text = (ROOT / "LICENSE").read_text(encoding="utf-8").replace("\r\n", "\n")
        self.assertTrue(text.startswith("# PolyForm Strict License 1.0.0"))
        self.assertEqual(hashlib.sha256(text.encode("utf-8")).hexdigest(), LICENCE_SHA256)

    def test_record_names_the_repository_and_release(self):
        rec = json.loads((ROOT / "data" / "measurement_record.json").read_text(encoding="utf-8"))
        self.assertEqual(rec["sources"]["repo"], "github.com/DumpAnalysis/Metrology")
        self.assertIn("Version 72", rec["sources"]["release"])
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("https://github.com/DumpAnalysis/Metrology", readme)
        self.assertIn("PolyForm Strict License 1.0.0", readme)


if __name__ == "__main__":
    unittest.main()
