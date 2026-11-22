"""目录构建：归属、继承、谱系、待核点与引用一致性。"""

import unittest
from pathlib import Path

from model_catalog.catalog import Catalog, CatalogError
from model_catalog.cli import audit
from model_catalog.records import Corpus, SourceRecord

DATA = Path(__file__).parents[1] / "data" / "models.json"


def rec(id, kind, version, text, models=()):
    return SourceRecord(id=id, kind=kind, version=version, text=text, models=models)


class SampleCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = Catalog.from_path(DATA)

    def test_entries_include_all_versions(self):
        names = [entry.name for entry in self.catalog.entries()]
        self.assertEqual(names, ["Base-Cedar", "Cedar-zh", "Cedar-zh-Q4"])

    def test_lineage_is_transitive(self):
        q4 = self.catalog.entry("Cedar-zh-Q4")
        self.assertEqual(q4.lineage, ("Cedar-zh", "Base-Cedar"))

    def test_base_entry_has_no_lineage(self):
        self.assertEqual(self.catalog.entry("Base-Cedar").lineage, ())

    def test_child_does_not_absorb_parent_only_facts(self):
        base = self.catalog.entry("Base-Cedar")
        self.assertEqual([ref.value for ref in base.fields("evaluation")], [])
        child = self.catalog.entry("Cedar-zh")
        self.assertTrue(child.fields("evaluation"))

    def test_license_not_inherited(self):
        # Cedar-zh 没写自己的许可；不应自动套用 Base-Cedar 的商业授权。
        self.assertEqual(self.catalog.entry("Cedar-zh").fields("license"), ())

    def test_language_is_inherited_with_provenance(self):
        refs = self.catalog.entry("Cedar-zh").fields("language")
        self.assertTrue(refs)
        for ref in refs:
            self.assertEqual(ref.basis, "inherited")
            self.assertEqual(ref.inherited_from, "Base-Cedar")

    def test_quantized_variant_does_not_inherit_evaluation(self):
        # “未附带独立评测”意味着量化版不能拿基础评测当自己的证据。
        q4 = self.catalog.entry("Cedar-zh-Q4")
        self.assertEqual(q4.fields("evaluation"), ())
        self.assertTrue(any(risk.code == "evaluation_gap" for risk in q4.risks))

    def test_exclusions_are_traceable_to_benchmark_note(self):
        refs = self.catalog.entry("Cedar-zh").fields("evaluation_exclusion")
        tasks = {ref.value["task"] for ref in refs}
        self.assertEqual(tasks, {"开放问答", "法律摘要任务"})
        self.assertTrue(all(ref.source_label == "model-4@v1" for ref in refs))

    def test_every_reference_points_back_to_source_text(self):
        self.assertEqual(audit(self.catalog), [])

    def test_every_entry_lists_source_versions(self):
        for entry in self.catalog.entries():
            self.assertTrue(entry.updated_at)
            for label in entry.updated_at:
                self.assertRegex(label, r"model-\d+@v\d+")


class RiskTests(unittest.TestCase):
    def test_missing_license_is_a_pending_check(self):
        catalog = Catalog.from_path(DATA)
        codes = {risk.code for risk in catalog.entry("Cedar-zh").risks}
        self.assertIn("license_not_stated", codes)

    def test_unresolved_terms_in_addendum(self):
        catalog = Catalog.from_path(DATA)
        codes = {risk.code for risk in catalog.entry("Base-Cedar").risks}
        self.assertIn("license_terms_unresolved", codes)

    def test_missing_parent_material_flagged(self):
        corpus = Corpus(
            [
                rec("m1", "release_note", 1, "Nova-mini 衍生自 Nova-base，使用合成语料。"),
            ]
        )
        catalog = Catalog(corpus)
        codes = {risk.code for risk in catalog.entry("Nova-mini").risks}
        self.assertIn("parent_material_missing", codes)

    def test_no_eval_is_not_reported_as_incapable(self):
        corpus = Corpus([rec("m1", "release_note", 1, "Nova-mini 未报告法律文本评测。")])
        catalog = Catalog(corpus)
        risk = next(r for r in catalog.entry("Nova-mini").risks if r.code == "evaluation_gap")
        self.assertIn("证据空缺", risk.message)
        self.assertNotIn("不支持", risk.message)


class AttributionTests(unittest.TestCase):
    def test_sibling_versions_are_separate_entries(self):
        corpus = Corpus(
            [
                rec("m1", "model_card", 1, "Nova-base 支持中文。许可允许商业使用。"),
                rec("m2", "release_note", 1, "Nova-base-Q4 衍生自 Nova-base，为量化版本。"),
            ]
        )
        catalog = Catalog(corpus)
        self.assertEqual(catalog.entry("Nova-base-Q4").lineage, ("Nova-base",))
        self.assertEqual(catalog.entry("Nova-base-Q4").fields("license"), ())

    def test_explicit_models_material_attributes_facts(self):
        corpus = Corpus(
            [
                rec("m1", "model_card", 1, "支持中文。", models=("Nova-base",)),
            ]
        )
        catalog = Catalog(corpus)
        self.assertTrue(catalog.entry("Nova-base").fields("language"))

    def test_unassigned_material_is_listed_not_spread(self):
        corpus = Corpus(
            [
                rec("m1", "model_card", 1, "Nova-base 支持中文。"),
                rec("m2", "benchmark_note", 1, "此外，某些测试集的表现尚未公开。"),
            ]
        )
        catalog = Catalog(corpus)
        unassigned = catalog.unassigned()
        self.assertTrue(unassigned)
        self.assertIn("某些测试集", " ".join(item["quote"] for item in unassigned))
        self.assertEqual(catalog.entry("Nova-base").fields("evaluation"), ())


class BuildErrorTests(unittest.TestCase):
    def test_cycle_is_rejected(self):
        corpus = Corpus(
            [
                rec("m1", "release_note", 1, "A-one 衍生自 B-two。"),
                rec("m2", "release_note", 1, "B-two 衍生自 A-one。"),
            ]
        )
        with self.assertRaises(CatalogError):
            Catalog(corpus).entries()

    def test_unknown_model_lookup(self):
        with self.assertRaises(KeyError):
            Catalog.from_path(DATA).entry("Nope")


if __name__ == "__main__":
    unittest.main()
