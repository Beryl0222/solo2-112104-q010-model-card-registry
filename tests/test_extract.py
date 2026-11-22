"""原文抽取：字段、引用区间与语义区分。"""

import unittest

from model_catalog.extract import Span, extract
from model_catalog.records import SourceRecord


def run(text, **kw):
    return extract(SourceRecord(id="s", kind=kw.pop("kind", "model_card"), version=1, text=text, **kw))


def quotes(spans):
    return [span.quote for span in spans]


def labelled(pairs):
    return [label for label, _span in pairs]


class SpanInvariantTests(unittest.TestCase):
    def assert_spans_valid(self, extraction):
        text = extraction.source.text
        all_spans = list(extraction.models)
        for field in ("restrictions", "restrictions_unknown", "evals", "eval_data", "eval_gaps", "training_data"):
            all_spans.extend(getattr(extraction, field))
        all_spans.extend(span for _, span in extraction.licenses)
        all_spans.extend(span for _, span in extraction.languages)
        all_spans.extend(span for _, span in extraction.language_exclusions)
        all_spans.extend(link.span for link in extraction.derivation_links if link.span.quote)
        all_spans.extend(item.span for item in extraction.eval_exclusions)
        all_spans.extend(item.cue for item in extraction.eval_exclusions)
        for span in all_spans:
            self.assertIsInstance(span, Span)
            self.assertEqual(span.quote, text[span.start : span.end])
            self.assertTrue(span.quote)

    def test_sample_materials_keep_exact_offsets(self):
        texts = [
            "Base-Cedar 支持中文和英语文本生成；许可允许研究与商业使用，具体限制见附录。",
            "Cedar-zh 是基于 Base-Cedar 的领域微调版本，使用合成客服语料，未报告法律文本评测。",
            "附录更正：Base-Cedar 再分发衍生权重须保留本条款；上游模型许可其余部分不变。",
            "Cedar-zh 在客服意图集合上评估；该结果不覆盖开放问答或法律摘要任务。",
        ]
        for text in texts:
            with self.subTest(text=text[:12]):
                self.assert_spans_valid(run(text))


class ModelNameTests(unittest.TestCase):
    def test_extracts_hyphenated_names(self):
        self.assertEqual(quotes(run("Cedar-zh 是基于 Base-Cedar 的微调版本。").models), ["Cedar-zh", "Base-Cedar"])

    def test_ignores_plain_words(self):
        self.assertEqual(quotes(run("benchmark 显示良好。").models), [])

    def test_version_suffix_is_part_of_name(self):
        self.assertEqual(quotes(run("Cedar-zh 2.1 已发布。").models), ["Cedar-zh 2.1"])


class DerivationTests(unittest.TestCase):
    def test_finetuned_link(self):
        link = run("Cedar-zh 是基于 Base-Cedar 的领域微调版本。").derivation_links[0]
        self.assertEqual((link.child, link.parent, link.relation), ("Cedar-zh", "Base-Cedar", "finetuned"))

    def test_quantized_link(self):
        link = run("Cedar-zh-Q4 衍生自 Cedar-zh，为 4 位量化。").derivation_links[0]
        self.assertEqual((link.child, link.parent, link.relation), ("Cedar-zh-Q4", "Cedar-zh", "quantized"))


class LicenseTests(unittest.TestCase):
    def test_commercial_and_research_allowed(self):
        self.assertEqual(labelled(run("许可允许研究与商业使用。").licenses), ["commercial_use"])

    def test_research_only(self):
        self.assertEqual(labelled(run("许可仅限研究使用。").licenses), ["research_only"])

    def test_non_commercial_is_not_commercial(self):
        labels = labelled(run("本模型非商业使用。").licenses)
        self.assertIn("non_commercial", labels)
        self.assertNotIn("commercial_use", labels)

    def test_unresolved_terms_flagged(self):
        ex = run("许可允许商业使用，具体限制见附录。")
        self.assertEqual(quotes(ex.restrictions_unknown), ["具体限制见附录"])

    def test_license_awaiting_confirmation(self):
        ex = run("许可条款需另行确认。")
        self.assertEqual(quotes(ex.restrictions_unknown), ["许可条款需另行确认"])

    def test_redistribution_obligation(self):
        ex = run("再分发衍生权重须保留本条款。")
        self.assertEqual(quotes(ex.restrictions), ["再分发衍生权重须保留本条款"])

    def test_upstream_license_reference(self):
        labels = labelled(run("上游模型许可其余部分不变。").licenses)
        self.assertIn("inherits_upstream", labels)


class LanguageTests(unittest.TestCase):
    def test_supported_languages(self):
        self.assertEqual(labelled(run("支持中文和英语文本生成。").languages), ["zh", "en"])

    def test_excluded_language(self):
        ex = run("模型支持中文，不支持英语。")
        self.assertIn("en", labelled(ex.language_exclusions))
        self.assertNotIn("en", labelled(ex.languages))

    def test_multilingual_marker(self):
        self.assertIn("multi", labelled(run("这是一个多语言模型。").languages))


class EvaluationTests(unittest.TestCase):
    def test_evaluation_and_dataset(self):
        ex = run("Cedar-zh 在客服意图集合上评估。")
        self.assertEqual(quotes(ex.evals), ["Cedar-zh 在客服意图集合上评估"])
        self.assertEqual(quotes(ex.eval_data), ["客服意图集合"])

    def test_training_corpus_is_not_eval_evidence(self):
        ex = run("Cedar-zh 基于 Base-Cedar 微调，使用合成客服语料。", kind="release_note")
        self.assertEqual(ex.evals, ())
        self.assertEqual(quotes(ex.training_data), ["使用合成客服语料"])

    def test_excluded_tasks_are_listed(self):
        ex = run("该结果不覆盖开放问答或法律摘要任务。", kind="benchmark_note")
        self.assertEqual([item.task for item in ex.eval_exclusions], ["开放问答", "法律摘要任务"])

    def test_missing_eval_is_a_gap_not_a_denial(self):
        ex = run("未报告法律文本评测。", kind="release_note")
        self.assertEqual(quotes(ex.eval_gaps), ["未报告法律文本评测"])
        self.assertEqual(ex.evals, ())


class AttributionTests(unittest.TestCase):
    def test_derivation_sentence_belongs_to_child(self):
        ex = run("Cedar-zh 是基于 Base-Cedar 的领域微调版本。", kind="release_note")
        attributed = {name for _start, names in ex.attribution for name in names}
        self.assertEqual(attributed, {"Cedar-zh"})
        self.assertEqual(ex.ambiguous_sentences, ())

    def test_omitted_subject_falls_back_to_document_model(self):
        ex = run("Base-Cedar 支持中文。许可允许商业使用。")
        owners = [names for _start, names in ex.attribution]
        self.assertEqual(owners, [("Base-Cedar",), ("Base-Cedar",)])

    def test_explicit_models_override(self):
        ex = run("再分发须保留本条款。", kind="license_addendum", models=("Base-Cedar",))
        self.assertEqual(ex.attribution, ((0, ("Base-Cedar",)),))

    def test_two_models_in_one_sentence_is_ambiguous(self):
        ex = run("Cedar-zh 与 Cedar-zh-Q4 共享同一许可。", kind="release_note")
        self.assertEqual(len(ex.ambiguous_sentences), 1)


if __name__ == "__main__":
    unittest.main()
