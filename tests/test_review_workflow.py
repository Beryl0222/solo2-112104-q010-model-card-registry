"""评审工作流：开始评审、上游修订检测、批准冻结、派生版本代入。"""

import unittest
from pathlib import Path

from model_catalog.catalog import Catalog, CatalogError
from model_catalog.records import SourceRecord

DATA = Path(__file__).parents[1] / "data" / "models.json"

REVISED_BASE_V2 = SourceRecord(
    id="model-1",
    kind="model_card",
    version=2,
    text="Base-Cedar 支持中文和英语文本生成；许可仅限研究使用。",
)


class StartReviewTests(unittest.TestCase):
    def test_start_review_records_source_versions(self):
        catalog = Catalog.from_path(DATA)
        state = catalog.start_review("Cedar-zh", reviewer="alice")
        self.assertEqual(state.status, "in_progress")
        self.assertIn("evaluation", state.seen_sources)
        self.assertEqual(state.seen_sources["evaluation"], "model-4@v1")

    def test_review_unknown_model(self):
        with self.assertRaises(KeyError):
            Catalog.from_path(DATA).start_review("Nope")

    def test_drift_without_review_raises(self):
        with self.assertRaises(CatalogError):
            Catalog.from_path(DATA).check_drift("Cedar-zh")


class DriftTests(unittest.TestCase):
    def test_no_drift_on_unchanged_materials(self):
        catalog = Catalog.from_path(DATA)
        catalog.start_review("Base-Cedar")
        self.assertEqual(catalog.check_drift("Base-Cedar"), ())

    def test_upstream_revision_marks_affected_fields(self):
        catalog = Catalog.from_path(DATA)
        catalog.start_review("Base-Cedar", reviewer="alice")
        updated = catalog.with_updated_materials([REVISED_BASE_V2])
        self.assertIn("license", updated.reviews["Base-Cedar"].stale_fields)
        self.assertEqual(updated.reviews["Base-Cedar"].status, "in_progress")

    def test_inherited_child_reacts_to_parent_revision(self):
        catalog = Catalog.from_path(DATA)
        catalog.start_review("Cedar-zh", reviewer="alice")
        # 基础卡片去掉英语支持，继承语言的子模型应被标为受影响。
        trimmed = SourceRecord(
            id="model-1",
            kind="model_card",
            version=2,
            text="Base-Cedar 支持中文文本生成；许可允许研究与商业使用。",
        )
        updated = catalog.with_updated_materials([trimmed])
        self.assertIn("language", updated.reviews["Cedar-zh"].stale_fields)


class ApprovalTests(unittest.TestCase):
    def test_approving_freezes_snapshot(self):
        catalog = Catalog.from_path(DATA)
        catalog.start_review("Base-Cedar", reviewer="alice")
        state = catalog.approve("Base-Cedar", reviewer="bob")
        self.assertEqual(state.status, "approved")
        self.assertIsNotNone(state.frozen_snapshot)
        quotes = [field["quote"] for field in state.frozen_snapshot["fields"]]
        self.assertTrue(any("商业使用" in quote for quote in quotes))

    def test_approved_review_keeps_original_text_after_revision(self):
        catalog = Catalog.from_path(DATA)
        catalog.start_review("Base-Cedar", reviewer="alice")
        approved = catalog.approve("Base-Cedar", reviewer="bob")
        before = [f["quote"] for f in approved.frozen_snapshot["fields"]]

        updated = catalog.with_updated_materials([REVISED_BASE_V2])
        state = updated.reviews["Base-Cedar"]

        # 结论仍是已批准，快照里的文本没有被上游修订覆盖。
        self.assertEqual(state.status, "approved")
        after = [f["quote"] for f in state.frozen_snapshot["fields"]]
        self.assertEqual(before, after)
        # 但审计信息记录了批准后发生的漂移。
        self.assertIn("license", state.drift_since_approval)

    def test_check_drift_does_not_reopen_approved_review(self):
        catalog = Catalog.from_path(DATA)
        catalog.start_review("Base-Cedar")
        catalog.approve("Base-Cedar", "bob")
        self.assertEqual(catalog.check_drift("Base-Cedar"), ())

    def test_approve_requires_started_review(self):
        with self.assertRaises(CatalogError):
            Catalog.from_path(DATA).approve("Cedar-zh", "bob")

    def test_frozen_snapshot_survives_catalog_rebuild_independently(self):
        catalog = Catalog.from_path(DATA)
        catalog.start_review("Base-Cedar")
        state = catalog.approve("Base-Cedar", "bob")
        original_fields = list(state.frozen_snapshot["fields"])

        updated = catalog.with_updated_materials([REVISED_BASE_V2])
        rebuilt = updated.reviews["Base-Cedar"].frozen_snapshot
        # 新目录的冻结快照是副本，内容与批准时一致，且不是同一对象。
        self.assertEqual(rebuilt["fields"], original_fields)
        self.assertIsNot(rebuilt, state.frozen_snapshot)


class AttachDerivedTests(unittest.TestCase):
    def test_attach_derived_inherits_language_not_license(self):
        catalog = Catalog.from_path(DATA)
        material = SourceRecord(
            id="model-9",
            kind="release_note",
            version=1,
            text="Cedar-lite 衍生自 Cedar-zh，使用工业问答语料。",
        )
        derived = catalog.attach_derived("Cedar-lite", "Cedar-zh", material, "finetuned")
        entry = derived.entry("Cedar-lite")
        self.assertEqual(entry.lineage, ("Cedar-zh", "Base-Cedar"))
        self.assertTrue(entry.fields("language"))
        self.assertTrue(all(ref.basis == "inherited" for ref in entry.fields("language")))
        self.assertEqual(entry.fields("license"), ())

    def test_attach_derived_does_not_mutate_original(self):
        catalog = Catalog.from_path(DATA)
        material = SourceRecord(id="model-9", kind="release_note", version=1, text="Cedar-lite 衍生自 Cedar-zh。")
        catalog.attach_derived("Cedar-lite", "Cedar-zh", material)
        self.assertNotIn("Cedar-lite", [entry.name for entry in catalog.entries()])

    def test_attach_derived_rejects_stale_version(self):
        catalog = Catalog.from_path(DATA)
        stale = SourceRecord(id="model-1", kind="model_card", version=1, text="旧版本重复。")
        with self.assertRaises(CatalogError):
            catalog.attach_derived("Cedar-zh", "Base-Cedar", stale)

    def test_attach_derived_rejects_wrong_type(self):
        with self.assertRaises(TypeError):
            Catalog.from_path(DATA).attach_derived("X", "Base-Cedar", "not a record")

    def test_attach_derived_requires_material_to_name_child(self):
        catalog = Catalog.from_path(DATA)
        material = SourceRecord(
            id="model-9",
            kind="release_note",
            version=1,
            text="这是一份没有点名任何模型的适配说明，使用工业问答语料。",
        )
        with self.assertRaises(CatalogError):
            catalog.attach_derived("Cedar-lite", "Cedar-zh", material)

    def test_forced_link_is_anchored_to_material(self):
        from model_catalog.cli import audit

        catalog = Catalog.from_path(DATA)
        material = SourceRecord(
            id="model-9",
            kind="release_note",
            version=1,
            text="Cedar-lite 是适配版本，使用工业问答语料。",
        )
        derived = catalog.attach_derived("Cedar-lite", "Cedar-zh", material)
        refs = derived.entry("Cedar-lite").fields("weights_source")
        self.assertTrue(refs)
        self.assertTrue(all(ref.quote for ref in refs))
        self.assertEqual(audit(derived), [])


class ReportTests(unittest.TestCase):
    def test_report_groups_fields_by_source(self):
        catalog = Catalog.from_path(DATA)
        report = catalog.review_report("Cedar-zh")
        self.assertEqual(report["lineage"], ["Base-Cedar"])
        self.assertIn("model-4@v1", report["fields_by_source"])
        self.assertIn("model-1@v1", report["fields_by_source"])
        self.assertEqual(report["review"]["status"], "not_started")

    def test_export_round_trips_as_json(self):
        import json

        catalog = Catalog.from_path(DATA)
        catalog.start_review("Cedar-zh", reviewer="alice")
        payload = json.loads(catalog.to_json())
        self.assertEqual({item["name"] for item in payload["entries"]}, {"Base-Cedar", "Cedar-zh", "Cedar-zh-Q4"})
        self.assertEqual(payload["reviews"]["Cedar-zh"]["status"], "in_progress")


if __name__ == "__main__":
    unittest.main()
