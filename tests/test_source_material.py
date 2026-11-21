import json
import unittest
from pathlib import Path

DATA = Path(__file__).parents[1] / "data" / 'models.json'
REQUIRED = {"id", "text"}


class SourceMaterialTests(unittest.TestCase):
    def test_synthetic_source_records_are_readable(self):
        records = json.loads(DATA.read_text(encoding="utf-8"))
        self.assertIsInstance(records, list)
        self.assertGreaterEqual(len(records), 4)
        self.assertTrue(all(REQUIRED <= record.keys() for record in records))
        self.assertTrue(all(isinstance(record["text"], str) and record["text"].strip() for record in records))


if __name__ == "__main__":
    unittest.main()
