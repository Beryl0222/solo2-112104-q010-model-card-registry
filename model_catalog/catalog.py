"""目录、版本谱系、字段继承与评审状态。

关键规则：

* 目录只收录材料里出现过的模型标识（以及显式派生关系里点名的模型）；
  不做“名称相近即同一模型”的匹配。
* 同一份材料里，每个事实句按主语归属到具体模型。“Cedar-zh 未报告法律
  文本评测”不会被算到 Base-Cedar 头上。
* 派生版本的字段有两种来源：本材料明写的，或从基础版本继承的。每条
  :class:`FieldRef` 记 ``basis``（explicit / inherited）；继承值永远附上
  基础材料的引用，评审可判断是否仍适用。
* 许可与许可限制不自动继承——派生版本的条款可能改变适用边界，因此只在
  材料明写时出现，缺失即列为待核点。
* 基础卡片修订后，继承字段随之变化；受影响的未完成评审标记 ``stale``，
  已批准的评审冻结批准时的目录快照与文本版本，不静默漂移。
* 缺说明、明说未评测、只提及未展开的限制，都进入待核点，不解释成
  “无限制”或“无此能力”。
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .extract import DerivationLink, Extraction, Span, extract
from .records import Corpus, SourceRecord, load_corpus


class CatalogError(ValueError):
    """目录构建或评审操作不合法。"""


def _fingerprint(value: object) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class FieldRef:
    """目录中一个字段值及其出处。"""

    field: str
    value: object
    basis: str  # "explicit" | "inherited"
    source_id: str
    source_kind: str
    source_version: int
    quote: str
    start: int
    end: int
    inherited_from: str | None = None

    @property
    def source_label(self) -> str:
        return f"{self.source_id}@v{self.source_version}"

    def to_dict(self) -> dict:
        data = {
            "field": self.field,
            "value": self.value,
            "basis": self.basis,
            "source": self.source_label,
            "source_kind": self.source_kind,
            "source_version": self.source_version,
            "quote": self.quote,
            "start": self.start,
            "end": self.end,
        }
        if self.inherited_from:
            data["inherited_from"] = self.inherited_from
        return data


@dataclass(frozen=True)
class Risk:
    """待核点：材料没有给出、或明确说未验证的事项。"""

    code: str
    field: str
    message: str
    source_label: str
    quote: str

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "field": self.field,
            "message": self.message,
            "source": self.source_label,
            "quote": self.quote,
        }


@dataclass(frozen=True)
class CatalogEntry:
    """目录中的一个模型版本视图。"""

    name: str
    refs: tuple[FieldRef, ...]
    lineage: tuple[str, ...]
    risks: tuple[Risk, ...]
    updated_at: tuple[str, ...]

    def fields(self, name: str) -> tuple[FieldRef, ...]:
        return tuple(ref for ref in self.refs if ref.field == name)

    def values(self, name: str) -> tuple[object, ...]:
        return tuple(ref.value for ref in self.refs if ref.field == name)

    def by_source(self) -> dict[str, tuple[FieldRef, ...]]:
        grouped: dict[str, list[FieldRef]] = {}
        for ref in self.refs:
            grouped.setdefault(ref.source_label, []).append(ref)
        return {key: tuple(value) for key, value in sorted(grouped.items())}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "lineage": list(self.lineage),
            "updated_at": list(self.updated_at),
            "risks": [risk.to_dict() for risk in self.risks],
            "fields": [ref.to_dict() for ref in self.refs],
        }


@dataclass
class ReviewState:
    """某个候选模型的一次评审。"""

    candidate: str
    status: str = "not_started"  # not_started | in_progress | approved
    reviewer: str | None = None
    note: str = ""
    #: 开始评审时每个事实引用的来源标签，用于检测上游修订。
    seen_sources: dict[str, str] = field(default_factory=dict)
    #: 已批准时冻结的目录快照，保留当时引用的文本与版本。
    frozen_snapshot: dict | None = None
    #: 批准后上游材料又发生修订、影响到的字段（不改变已批准的结论）。
    drift_since_approval: tuple[str, ...] = ()
    #: 最近一次检查后发生漂移的字段。
    stale_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate,
            "status": self.status,
            "reviewer": self.reviewer,
            "note": self.note,
            "seen_sources": dict(sorted(self.seen_sources.items())),
            "stale_fields": list(self.stale_fields),
            "drift_since_approval": list(self.drift_since_approval),
            "frozen_snapshot_fingerprint": (
                _fingerprint(self.frozen_snapshot) if self.frozen_snapshot else None
            ),
        }


# (目录字段, Extraction 属性)；是否可继承由 _INHERITABLE 单独决定。
_FIELD_MAP = (
    ("license", "licenses"),
    ("license_restriction", "restrictions"),
    ("language", "languages"),
    ("language_exclusion", "language_exclusions"),
    ("evaluation", "evals"),
    ("evaluation_data", "eval_data"),
    ("training_data", "training_data"),
)

#: 派生版本可以沿用基础版本、但必须复核适用边界的字段。
#: 许可以及许可限制不在其列：新政或许可附录可能改变派生版本的适用边界。
_INHERITABLE = frozenset({"language", "language_exclusion", "evaluation", "evaluation_data"})
#: 继承来的字段若涉及这些边界，需要额外标注待核。
_BOUNDARY_FIELDS = frozenset({"language", "language_exclusion"})


class Catalog:
    """由多份材料构建的模型目录。"""

    def __init__(self, corpus: Corpus) -> None:
        self._corpus = corpus
        self._extractions: dict[str, Extraction] = {
            record.id: extract(record) for record in corpus.latest()
        }
        self._entries: dict[str, CatalogEntry] | None = None
        self.reviews: dict[str, ReviewState] = {}

    @classmethod
    def from_path(cls, path) -> "Catalog":
        return cls(load_corpus(path))

    @property
    def corpus(self) -> Corpus:
        return self._corpus

    # ------------------------------------------------------------------ 构建
    def entries(self) -> tuple[CatalogEntry, ...]:
        built = self._built()
        return tuple(built[key] for key in sorted(built))

    def entry(self, name: str) -> CatalogEntry:
        built = self._built()
        if name not in built:
            raise KeyError(f"目录中没有模型 {name!r}")
        return built[name]

    def _built(self) -> dict[str, CatalogEntry]:
        if self._entries is None:
            self._entries = self._build()
        return self._entries

    def _links(self) -> tuple[DerivationLink, ...]:
        result: list[DerivationLink] = []
        for extraction in self._extractions.values():
            result.extend(extraction.derivation_links)
        return tuple(result)

    def _names(self, links: Iterable[DerivationLink]) -> set[str]:
        names: set[str] = set()
        for extraction in self._extractions.values():
            names.update(span.quote for span in extraction.models)
            # 材料级归属声明也应出现在目录里。
            names.update(extraction.source.models)
        for link in links:
            names.update((link.child, link.parent))
        return names

    def _build(self) -> dict[str, CatalogEntry]:
        links = self._links()
        parents: dict[str, set[str]] = {}
        for link in links:
            if link.child != link.parent:
                parents.setdefault(link.child, set()).add(link.parent)

        entries: dict[str, CatalogEntry] = {}
        in_progress: set[str] = set()

        def resolve(name: str) -> CatalogEntry:
            if name in entries:
                return entries[name]
            if name in in_progress:
                raise CatalogError(f"派生关系存在环：{name}")
            in_progress.add(name)
            refs = self._explicit_refs(name)
            inherited = self._inherit(name, refs, parents, resolve)
            merged = _dedupe_refs((*refs, *inherited))
            lineage = self._lineage(name, parents)
            risks = _dedupe_risks(
                self._risks(name, merged, lineage) + self._ambiguity_risks(name)
            )
            entries[name] = CatalogEntry(
                name=name,
                refs=merged,
                lineage=lineage,
                risks=risks,
                updated_at=tuple(sorted({ref.source_label for ref in merged})),
            )
            in_progress.discard(name)
            return entries[name]

        for name in sorted(self._names(links)):
            resolve(name)
        return entries

    def _owns(self, extraction: Extraction, model: str, start: int) -> bool:
        """材料里的这个位置是否归属于 model。

        归属以句子为准：字段的引用区间若落在某句内，就看该句归属哪些模型。
        ``weights_source`` 这类关系事实由调用方单独加上。
        """

        if extraction.source.models:
            return model in extraction.source.models
        sentence_start = _sentence_start(extraction, start)
        for attributed_start, attributed_models in extraction.attribution:
            if attributed_start == sentence_start:
                return model in attributed_models
        return False

    def _explicit_refs(self, name: str) -> tuple[FieldRef, ...]:
        refs: list[FieldRef] = []
        for extraction in self._extractions.values():
            for link in extraction.derivation_links:
                if link.child == name or link.parent == name:
                    refs.append(
                        self._ref(
                            extraction,
                            "weights_source",
                            {"child": link.child, "parent": link.parent, "relation": link.relation},
                            link.span,
                        )
                    )
            for field_name, attr in _FIELD_MAP:
                for value in getattr(extraction, attr):
                    if isinstance(value, tuple):
                        label, span = value
                        payload: object = {"label": label, "text": span.quote}
                    else:
                        span = value
                        payload = span.quote
                    if self._owns(extraction, name, span.start):
                        refs.append(self._ref(extraction, field_name, payload, span))
            for gap in extraction.eval_gaps:
                if self._owns(extraction, name, gap.start):
                    refs.append(self._ref(extraction, "evaluation_gap", gap.quote, gap))
            for exclusion in extraction.eval_exclusions:
                if self._owns(extraction, name, exclusion.span.start):
                    refs.append(
                        self._ref(
                            extraction,
                            "evaluation_exclusion",
                            {"task": exclusion.task, "cue": exclusion.cue.quote},
                            exclusion.span,
                        )
                    )
        return tuple(refs)

    def _ambiguity_risks(self, name: str) -> tuple[Risk, ...]:
        """同一句提到多个模型时，目录无法确定该说明属于谁，列为待核。"""

        risks: list[Risk] = []
        for extraction in self._extractions.values():
            for span in extraction.ambiguous_sentences:
                names = {item.quote for item in extraction.models if span.start <= item.start < span.end}
                if name not in names:
                    continue
                risks.append(
                    Risk(
                        "attribution_ambiguous",
                        "weights_source",
                        "该句同时提到多个模型，目录按句首主语归属，需人工确认另一版本是否也受约束。",
                        extraction.source.label,
                        span.quote,
                    )
                )
        return tuple(risks)

    def _has_materials(self, name: str) -> bool:
        """目录里是否有材料真正描述这个模型（而不只是被派生关系提到）。

        判定依据是句子归属：派生句“A 基于 B”归 A，因此 B 不会仅因为被
        提到就算“有材料”。材料级 ``models`` 声明也算描述。
        """

        for extraction in self._extractions.values():
            if name in extraction.source.models:
                return True
            if any(name in names for _start, names in extraction.attribution):
                return True
        return False

    def _inherit(
        self,
        name: str,
        own: tuple[FieldRef, ...],
        parents: Mapping[str, set[str]],
        resolve,
    ) -> tuple[FieldRef, ...]:
        inherited: list[FieldRef] = []
        have = {ref.field for ref in own}
        # 派生版本明说“未做独立评测”时，不把基础版本的评测结果当成自己的证据。
        if "evaluation_gap" in have:
            have.update({"evaluation", "evaluation_data"})
        for parent in sorted(parents.get(name, ())):
            if not self._has_materials(parent):
                # 基础模型没有材料，留给 parent_material_missing 待核点。
                continue
            parent_entry = resolve(parent)
            for field_name in sorted(_INHERITABLE):
                if field_name in have:
                    continue
                parent_refs = parent_entry.fields(field_name)
                if not parent_refs:
                    continue
                for ref in parent_refs:
                    inherited.append(
                        FieldRef(
                            field=field_name,
                            value=ref.value,
                            basis="inherited",
                            source_id=ref.source_id,
                            source_kind=ref.source_kind,
                            source_version=ref.source_version,
                            quote=ref.quote,
                            start=ref.start,
                            end=ref.end,
                            inherited_from=parent,
                        )
                    )
                have.add(field_name)
        return tuple(inherited)

    def _ref(self, extraction: Extraction, field_name: str, value: object, span: Span) -> FieldRef:
        return FieldRef(
            field=field_name,
            value=value,
            basis="explicit",
            source_id=extraction.source.id,
            source_kind=extraction.source.kind,
            source_version=extraction.source.version,
            quote=span.quote,
            start=span.start,
            end=span.end,
        )

    @staticmethod
    def _lineage(name: str, parents: Mapping[str, set[str]]) -> tuple[str, ...]:
        ordered: list[str] = []
        seen = {name}
        frontier = sorted(parents.get(name, ()))
        while frontier:
            current = frontier.pop(0)
            if current in seen:
                continue
            seen.add(current)
            ordered.append(current)
            frontier.extend(sorted(parents.get(current, ())))
        return tuple(ordered)

    def _risks(
        self, name: str, refs: tuple[FieldRef, ...], lineage: tuple[str, ...]
    ) -> tuple[Risk, ...]:
        risks: list[Risk] = []
        fields = {ref.field for ref in refs}
        anchor = _anchor(refs)
        has_license = "license" in fields or "license_restriction" in fields

        if not has_license:
            risks.append(
                Risk(
                    "license_not_stated",
                    "license",
                    "材料未写明该模型的许可条件，需向上游原件确认，不能沿用其他模型的结论。",
                    anchor[0],
                    "",
                )
            )
        if "evaluation" not in fields and "evaluation_gap" not in fields:
            risks.append(
                Risk(
                    "evaluation_not_stated",
                    "evaluation",
                    "材料没有该模型的任务评测说明，不能据此推断能力。",
                    anchor[0],
                    "",
                )
            )
        if "language" not in fields:
            risks.append(
                Risk(
                    "language_not_stated",
                    "language",
                    "材料未写明支持语言，不能默认沿用其他版本。",
                    anchor[0],
                    "",
                )
            )

        for ref in refs:
            if ref.field == "evaluation_gap":
                risks.append(
                    Risk(
                        "evaluation_gap",
                        "evaluation",
                        "材料明确表示该项评测未完成或未报告，属证据空缺而非否定能力。",
                        ref.source_label,
                        ref.quote,
                    )
                )
            if ref.basis == "inherited" and ref.field in _BOUNDARY_FIELDS:
                risks.append(
                    Risk(
                        "inherited_boundary",
                        ref.field,
                        "该字段继承自基础版本，派生版本的微调或条款可能改变适用边界，须复核。",
                        ref.source_label,
                        ref.quote,
                    )
                )

        # 明说“许可条款另行确认/具体限制见附录”但未展开的，逐条列为待查。
        for extraction in self._extractions.values():
            for span in extraction.restrictions_unknown:
                if not self._owns(extraction, name, span.start):
                    continue
                risks.append(
                    Risk(
                        "license_terms_unresolved",
                        "license",
                        "许可条款引用附录、另见文件或尚待确认，具体限制需查原文。",
                        extraction.source.label,
                        span.quote,
                    )
                )

        for parent in lineage:
            if not self._has_materials(parent):
                risks.append(
                    Risk(
                        "parent_material_missing",
                        "weights_source",
                        f"派生关系指向 {parent}，但目录没有其材料，继承字段无法核验。",
                        *anchor,
                    )
                )

        risks.sort(key=lambda risk: (risk.field, risk.code, risk.source_label, risk.quote))
        return tuple(risks)

    # ------------------------------------------------------------------ 派生
    def attach_derived(
        self,
        child: str,
        parent: str,
        material: SourceRecord,
        relation: str = "finetuned",
    ) -> "Catalog":
        """用一份新适配材料把派生版本代入目录，返回新目录（不修改自身）。

        派生版本明写的字段覆盖继承字段；未明写的可继承字段沿用基础版本，
        并标注 ``inherited_from``。许可与许可限制一律要求明写。
        """

        if not isinstance(material, SourceRecord):
            raise TypeError("material 必须是 SourceRecord")
        existing = {r.id for r in self._corpus.latest()}
        if material.id in existing and material.version <= self._corpus.latest_version(material.id):
            raise CatalogError("新材料的版本必须高于同 id 的现有版本")
        corpus = self._corpus.extend([material])
        catalog = Catalog(corpus)
        # 材料原文若没有逐字写出派生关系，用调用方给出的链接补上，
        # 引用锚定到材料中描述该子模型的句子；材料压根没提子模型则拒绝。
        if not any(link.child == child and link.parent == parent for link in catalog._links()):
            catalog._extractions[material.id] = _with_forced_link(
                catalog._extractions[material.id], child, parent, relation
            )
        return catalog

    # ------------------------------------------------------------------ 评审
    def start_review(self, candidate: str, reviewer: str | None = None) -> ReviewState:
        entry = self.entry(candidate)
        state = ReviewState(
            candidate=candidate,
            status="in_progress",
            reviewer=reviewer,
            seen_sources=_field_versions(entry),
        )
        self.reviews[candidate] = state
        return state

    def check_drift(self, candidate: str) -> tuple[str, ...]:
        """对照评审开始时的来源版本，返回被上游修订影响的字段。

        已批准的评审不重开，返回空元组；它的冻结快照保持原样。
        """

        state = self.reviews.get(candidate)
        if state is None:
            raise CatalogError(f"{candidate} 尚未开始评审")
        if state.status == "approved":
            return ()
        current = _field_versions(self.entry(candidate))
        changed = tuple(
            sorted(
                field_name
                for field_name, label in state.seen_sources.items()
                if current.get(field_name) != label
            )
        )
        state.stale_fields = changed
        return changed

    def approve(self, candidate: str, reviewer: str) -> ReviewState:
        """批准评审并冻结当前目录快照，保留当时引用的文本版本。"""

        state = self.reviews.get(candidate)
        if state is None:
            raise CatalogError(f"{candidate} 尚未开始评审")
        state.status = "approved"
        state.reviewer = reviewer
        state.frozen_snapshot = copy.deepcopy(self.entry(candidate).to_dict())
        state.stale_fields = ()
        return state

    def unassigned(self) -> tuple[dict, ...]:
        """无法归属到任何模型的材料分句。

        这类文本不会被套到任何候选模型上，但会在报告里显式列出，避免评审
        人员误以为“没提到就没有”。
        """

        items: list[dict] = []
        for extraction in self._extractions.values():
            attributed_starts = {start for start, _ in extraction.attribution}
            for start, end in _sentence_ranges(extraction.source.text):
                if start in attributed_starts:
                    continue
                quote = extraction.source.text[start:end].strip()
                if quote:
                    items.append(
                        {
                            "source": extraction.source.label,
                            "kind": extraction.source.kind,
                            "quote": quote,
                        }
                    )
        return tuple(items)

    def with_updated_materials(self, records: Iterable[SourceRecord]) -> "Catalog":
        """用修订后的材料重建目录，并延续已有评审状态。

        * 未完成的评审：对照开始时的来源版本，标出受影响字段（``stale``）。
        * 已批准的评审：不重开，冻结快照原样保留，同时记录新增漂移字段，
          以便审计“批准后上游又改了什么”。
        """

        corpus = self._corpus.extend(records)
        catalog = Catalog(corpus)
        for candidate, state in self.reviews.items():
            if candidate not in {entry.name for entry in catalog.entries()}:
                continue
            current = _field_versions(catalog.entry(candidate))
            changed = tuple(
                sorted(
                    field_name
                    for field_name, label in state.seen_sources.items()
                    if current.get(field_name) != label
                )
            )
            catalog.reviews[candidate] = copy.deepcopy(state)
            # 已批准的评审不重开：冻结快照原样保留，改记批准后的上游漂移。
            if state.status == "approved":
                catalog.reviews[candidate].drift_since_approval = changed
            else:
                catalog.reviews[candidate].stale_fields = changed
        return catalog

    def review_report(self, candidate: str) -> dict:
        state = self.reviews.get(candidate)
        entry = self.entry(candidate)
        return {
            "candidate": candidate,
            "lineage": list(entry.lineage),
            "review": state.to_dict() if state else {"status": "not_started"},
            "risks": [risk.to_dict() for risk in entry.risks],
            "fields_by_source": {
                label: [ref.to_dict() for ref in refs]
                for label, refs in entry.by_source().items()
            },
        }

    # ------------------------------------------------------------------ 导出
    def to_dict(self) -> dict:
        return {
            "entries": [entry.to_dict() for entry in self.entries()],
            "reviews": {
                key: state.to_dict() for key, state in sorted(self.reviews.items())
            },
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def _sentence_ranges(text: str) -> list[tuple[int, int]]:
    """按中文/英文分句符切成 (start, end)，保留原文偏移。"""

    ranges: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[；;。！？\n]", text):
        if match.start() > start:
            ranges.append((start, match.start()))
        start = match.end()
    if start < len(text):
        ranges.append((start, len(text)))
    return ranges


def _sentence_start(extraction: Extraction, position: int) -> int:
    best = 0
    for start, _models in extraction.attribution:
        if start <= position:
            best = max(best, start)
    return best


def _dedupe_refs(refs: Iterable[FieldRef]) -> tuple[FieldRef, ...]:
    seen: set[tuple] = set()
    result: list[FieldRef] = []
    for ref in refs:
        key = (
            ref.field,
            ref.source_id,
            ref.source_version,
            ref.start,
            ref.end,
            json.dumps(ref.value, ensure_ascii=False, sort_keys=True, default=str),
            ref.basis,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(ref)
    result.sort(key=lambda r: (r.field, r.source_id, r.source_version, r.start, r.end))
    return tuple(result)


def _dedupe_risks(risks: Iterable[Risk]) -> tuple[Risk, ...]:
    seen: set[tuple] = set()
    result: list[Risk] = []
    for risk in risks:
        key = (risk.code, risk.field, risk.source_label, risk.quote)
        if key in seen:
            continue
        seen.add(key)
        result.append(risk)
    result.sort(key=lambda r: (r.field, r.code, r.source_label))
    return tuple(result)


def _field_versions(entry: CatalogEntry) -> dict[str, str]:
    """每个字段由哪些来源标签支撑；用于检测上游修订导致的漂移。"""

    versions: dict[str, str] = {}
    for ref in entry.refs:
        versions.setdefault(ref.field, ref.source_label)
        if versions[ref.field] != ref.source_label:
            versions[ref.field] = f"{versions[ref.field]}|{ref.source_label}"
    return versions


def _anchor(refs: tuple[FieldRef, ...]) -> tuple[str, str]:
    return (refs[0].source_label, refs[0].quote) if refs else ("n/a", "")


def _with_forced_link(
    extraction: Extraction, child: str, parent: str, relation: str
) -> Extraction:
    """补一条材料未逐字写出的派生关系，引用锚定到描述子模型的句子。

    材料原文完全没提子模型时无法锚定，拒绝补链——派生关系必须有出处。
    """

    from dataclasses import replace

    text = extraction.source.text
    anchor = next(
        (span for span in extraction.models if span.quote == child),
        None,
    )
    if anchor is None:
        raise CatalogError(
            f"材料 {extraction.source.label} 没有提到 {child}，无法建立可核对的派生关系"
        )
    span = _sentence_span_at(text, anchor.start, anchor.end)
    link = DerivationLink(child=child, parent=parent, relation=relation, span=span)
    return replace(extraction, derivation_links=(*extraction.derivation_links, link))


def _sentence_span_at(text: str, start: int, end: int) -> Span:
    boundaries = "；;。！？\n"
    left = max((text.rfind(b, 0, start) for b in boundaries), default=-1) + 1
    right = min(
        (pos for pos in (text.find(b, end) for b in boundaries) if pos >= 0),
        default=len(text),
    )
    while left < right and text[left].isspace():
        left += 1
    while right > left and text[right - 1].isspace():
        right -= 1
    return Span(text[left:right], left, right)


def derive_entry(
    catalog: Catalog,
    child: str,
    parent: str,
    material: SourceRecord,
    relation: str = "finetuned",
) -> CatalogEntry:
    """便捷函数：返回代入派生材料后 child 的目录条目。"""

    return catalog.attach_derived(child, parent, material, relation).entry(child)
