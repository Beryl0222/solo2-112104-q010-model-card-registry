"""原始材料的读取、版本快照与校验。"""

import json
import tempfile
import unittest
from pathlib import Path

from model_catalog.records import (
    Corpus,
    SourceMaterialError,
    SourceRecord,
    load_corpus,
    load_records,
)

DATA = Path(__file__).parents[1] / "data" / "models.json"


def record(**overrides):
    base = dict(id="r1", kind="model_card", version=1, text="Base-Cedar 支持中文。")
    base.update(overrides)
    return SourceRecord(**base)


class LoadTests(unittest.TestCase):
    def test_loads_repository_materials(self):
        corpus = load_corpus(DATA)
        self.assertGreaterEqual(len(corpus), 5)
        self.assertTrue(all(r.version >= 1 for r in corpus))

    def test_rejects_missing_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "materials.json"
            path.write_text(json.dumps([{"id": "x", "text": "内容"}]), encoding="utf-8")
            with self.assertRaises(SourceMaterialError):
                load_records(path)

    def test_rejects_unknown_kind(self):
        with self.assertRaises(SourceMaterialError):
            record(kind="press_release")

    def test_rejects_blank_text_and_bad_version(self):
        with self.assertRaises(SourceMaterialError):
            record(text="   ")
        with self.assertRaises(SourceMaterialError):
            record(version=0)
        with self.assertRaises(SourceMaterialError):
            record(version=True)

    def test_optional_models_declaration(self):
        rec = record(models=("Base-Cedar",))
        self.assertEqual(rec.models, ("Base-Cedar",))
        with self.assertRaises(SourceMaterialError):
            record(models=("",))


class CorpusTests(unittest.TestCase):
    def test_latest_takes_highest_version_per_id(self):
        corpus = Corpus(
            [
                record(id="a", version=1),
                record(id="a", version=3),
                record(id="a", version=2),
                record(id="b", version=1, kind="release_note"),
            ]
        )
        latest = corpus.latest()
        self.assertEqual([r.label for r in latest], ["a@v3", "b@v1"])
        self.assertEqual([r.version for r in corpus.history("a")], [1, 2, 3])
        self.assertEqual(corpus.latest_version("a"), 3)

    def test_rejects_duplicate_versions(self):
        with self.assertRaises(SourceMaterialError):
            Corpus([record(id="a", version=2), record(id="a", version=2)])

    def test_extend_returns_new_corpus(self):
        corpus = Corpus([record(id="a", version=1)])
        grown = corpus.extend([record(id="a", version=2)])
        self.assertEqual(len(corpus), 1)
        self.assertEqual(len(grown), 2)


if __name__ == "__main__":
    unittest.main()
