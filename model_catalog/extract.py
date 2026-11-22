"""从材料原文抽取结构化字段，并保留原文区间作为引用。

设计原则：

* 只抽取原文中显式出现的表述，不从名称、下载量或参数量推断能力。
* 每个抽取结果都是 :class:`Span`，满足 ``span.quote == text[span.start:span.end]``，
  评审人员可以逐字回到材料核对。
* 训练/适配语料不是评测证据，单独抽取为 ``training_data``，避免把
  “使用合成客服语料”误读成“在客服任务上评测过”。
* 材料明确排除的任务、明确未做的评测、以及只提及而未写明的限制，
  分别落在 ``eval_exclusions`` / ``eval_gaps`` / ``restrictions_unknown``，
  供目录标为待核点，而不是当作“没有能力”或“没有限制”。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .records import SourceRecord

#: 表示“该说明是推断/传闻、而非材料确认”的措辞。
HEDGE_MARKERS = ("未证实", "传闻", "待确认", "据说", "未经证实", "疑似")
#: 表示“来源方自己明确否定/更正”的措辞。
DENIAL_MARKERS = ("并非", "不是", "否认")

_SENTENCE_BREAKS = ("；", ";", "。", "！", "？", "\n")
_CONTEXT_BREAKS = ("，", ",", "；", ";", "。", "！", "？", "\n")

_MODEL_NAME = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)*)"
    r"(?P<version>\s+\d+(?:\.\d+)*)?"
)

#: 出现这些词的分句谈论的是训练/适配语料，不是评测数据。
_TRAINING_LEAD = re.compile(r"微调|训练|适配|语料|蒸馏|继续预训练")
#: 据此判断一个分句是否在描述评测，而不是训练/适配。
_EVAL_CUE_STRONG = re.compile(r"评估|测评|评测|测试|基准|benchmark", re.IGNORECASE)

_DERIVATION_PATTERNS = (
    re.compile(
        r"(?P<child>[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)*)"
        r"\s*(?:是|为)?\s*基于\s*"
        r"(?P<parent>[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)*)"
    ),
    re.compile(
        r"(?P<child>[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)*)"
        r"\s*(?:衍生自|派生自|微调自|改编自)\s*"
        r"(?P<parent>[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)*)"
    ),
)
_RELATION_WORDS = (
    ("quantized", ("量化",)),
    ("distilled", ("蒸馏",)),
    ("merged", ("合并", "融合")),
    ("finetuned", ("微调", "继续训练", "继续预训练")),
    ("derived", ("衍生", "派生", "改编")),
)

_LICENSE_CUES = (
    ("commercial_use", re.compile(r"商业使用|商用|商业用途")),
    ("research_only", re.compile(r"仅限研究|研究用途|限研究|仅用于研究")),
    ("non_commercial", re.compile(r"非商业|不得用于商业|不允许商业")),
)
_REDISTRIBUTE = re.compile(r"再分发|重新分发|分发衍生|转发衍生权重")
_OBLIGATION = re.compile(r"须|必须|需|应|不得|禁止|允许")
_LIMITS_EXIST = re.compile(r"具体限制|限制见|限制详见|详见附录|限制待|限制以")
#: 明说许可条款尚未确定、需要另行确认。
_LICENSE_UNRESOLVED = re.compile(
    r"许可(?:条款|条件)?[^；;。，,\n]{0,12}(?:另行确认|待确认|尚不明确|需另行|需确认|未明确)"
)
_LICENSE_REFERENCE = re.compile(r"许可(?:条款)?(?:其余部分)?不变|沿用(?:上游|原)?许可|适用(?:上游|原)许可")
_INTERNAL_ONLY = re.compile(r"(?:仅限|仅用于|只限)内部|内部评估专用")

_LANGUAGE_TOKENS = (
    ("中文", "zh"),
    ("汉语", "zh"),
    ("普通话", "zh"),
    ("英语", "en"),
    ("英文", "en"),
)
_UNIVERSAL_LANGUAGE = re.compile(r"多语言|multilingual", re.IGNORECASE)
_LANG_EXCLUDE_CUE = re.compile(r"不支持|不含|不包括|未支持|不覆盖")

_EVAL_CUE = re.compile(r"评估|测评|评测|测试|基准|benchmark", re.IGNORECASE)
_EVAL_DATASET = re.compile(
    r"(?:评估|测评|评测|测试)(?:集|集合|数据集)?\s*[：:]?\s*([^；;。\n]{1,40}?)"
    r"(?=[上中里]?的?(?:评估|测评|评测|测试|表现|结果|基准)|[；;。，,\n]|$)"
    r"|在([^；;。\n]{1,40}?)(?:集合|数据集|测试集|集)?上(?:的)?(?:评估|测评|评测|测试|表现|结果)",
    re.IGNORECASE,
)
_SUBJECT_CUE = re.compile(r"(?:使用|采用|基于|借助|用)([^；;。，,\n]{1,30}?)(?:语料|数据)")

_EXCLUSION_CUE = re.compile(r"不覆盖|不适用于|不包括|未覆盖|不涉及|不支持|无法|不保证|未验证|未评测|没有覆盖")
_EXCLUSION_SEP = re.compile(r"或|、|,|，|以及|和")
_EXCLUSION_TRIM = re.compile(r"^(?:在|对|于|关于|上|中|里|的|其|该|此|结果|模型|系统)+")
_NO_EVAL = re.compile(r"未(?:报告|报告过|发现)?([^；;。，,\n]{1,20}?)(?:评测|评估|测评|测试)")


@dataclass(frozen=True)
class Span:
    """原文中的一段可核对文本。"""

    quote: str
    start: int
    end: int

    def to_dict(self) -> dict:
        return {"quote": self.quote, "start": self.start, "end": self.end}


@dataclass(frozen=True)
class DerivationLink:
    """材料中显式写出的派生关系。"""

    child: str
    parent: str
    relation: str
    span: Span

    def to_dict(self) -> dict:
        return {
            "child": self.child,
            "parent": self.parent,
            "relation": self.relation,
            "source": self.span.to_dict(),
        }


@dataclass(frozen=True)
class Exclusion:
    """材料明确排除的任务/范围，附排除项自身与整句引用。"""

    task: str
    span: Span
    cue: Span

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "task_span": self.span.to_dict(),
            "cue_span": self.cue.to_dict(),
        }


@dataclass(frozen=True)
class Extraction:
    """一份材料上抽出的全部字段，值都指向本材料原文。"""

    source: SourceRecord
    models: tuple[Span, ...] = ()
    #: 每个句子起点归属到哪些模型。派生关系句归子模型，其余按句首主语。
    attribution: tuple[tuple[int, tuple[str, ...]], ...] = ()
    #: 句中同时点名多个模型、原文未说明各条款分别约束谁的句子。
    ambiguous_sentences: tuple[Span, ...] = ()
    derivation_links: tuple[DerivationLink, ...] = ()
    licenses: tuple[tuple[str, Span], ...] = ()
    restrictions: tuple[Span, ...] = ()
    restrictions_unknown: tuple[Span, ...] = ()
    languages: tuple[tuple[str, Span], ...] = ()
    language_exclusions: tuple[tuple[str, Span], ...] = ()
    evals: tuple[Span, ...] = ()
    eval_data: tuple[Span, ...] = ()
    eval_exclusions: tuple[Exclusion, ...] = ()
    eval_gaps: tuple[Span, ...] = ()
    training_data: tuple[Span, ...] = ()
    flags: tuple[tuple[str, Span], ...] = ()


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = max((text.rfind(b, 0, start) for b in _SENTENCE_BREAKS), default=-1)
    left = left + 1 if left >= 0 else 0
    right = min(
        (pos for pos in (text.find(b, end) for b in _SENTENCE_BREAKS) if pos >= 0),
        default=len(text),
    )
    return left, right


def _context_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = max((text.rfind(b, 0, start) for b in _CONTEXT_BREAKS), default=-1)
    left = left + 1 if left >= 0 else 0
    right = min(
        (pos for pos in (text.find(b, end) for b in _CONTEXT_BREAKS) if pos >= 0),
        default=len(text),
    )
    return left, right


def _span(text: str, start: int, end: int) -> Span:
    """构造 Span，并把两端空白收进区间，保证 quote == text[start:end]。"""

    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return Span(text[start:end], start, end)


def _sentence_span(text: str, start: int, end: int) -> Span:
    left, right = _sentence_bounds(text, start, end)
    return _span(text, left, right)


def _spans_from_matches(text: str, matches: Iterable[re.Match]) -> tuple[Span, ...]:
    spans: list[Span] = []
    seen: set[tuple[int, int]] = set()
    for match in matches:
        span = _sentence_span(text, *match.span())
        if (span.start, span.end) in seen:
            continue
        seen.add((span.start, span.end))
        spans.append(span)
    spans.sort(key=lambda s: s.start)
    return tuple(spans)


def _models(text: str) -> tuple[Span, ...]:
    spans: list[Span] = []
    seen: set[tuple[int, int]] = set()
    for match in _MODEL_NAME.finditer(text):
        name = match.group("name")
        # 单个英文词不是模型标识；要求连字符/下划线/点分写法。
        if not re.search(r"[-_.]", name):
            continue
        start, end = match.span("name")
        if match.group("version"):
            end = match.end("version")
        if (start, end) in seen:
            continue
        seen.add((start, end))
        spans.append(_span(text, start, end))
    spans.sort(key=lambda s: s.start)
    return tuple(spans)


def _relation(sentence: str) -> str:
    for name, words in _RELATION_WORDS:
        if any(word in sentence for word in words):
            return name
    return "derived"


def _derivation(text: str) -> tuple[DerivationLink, ...]:
    links: list[DerivationLink] = []
    for pattern in _DERIVATION_PATTERNS:
        for match in pattern.finditer(text):
            # 关系类型（量化/蒸馏/微调…）常在主句之外的补充分句里，取整句判断。
            sentence = _sentence_span(text, *match.span())
            links.append(
                DerivationLink(
                    child=match.group("child"),
                    parent=match.group("parent"),
                    relation=_relation(sentence.quote),
                    span=sentence,
                )
            )
    links.sort(key=lambda link: link.span.start)
    return tuple(links)


def _licenses(
    text: str,
) -> tuple[tuple[tuple[str, Span], ...], tuple[Span, ...], tuple[Span, ...]]:
    found: dict[tuple[str, int, int], tuple[str, Span]] = {}
    for name, pattern in _LICENSE_CUES:
        for match in pattern.finditer(text):
            if re.search(r"(?:仅|只|非)$", text[max(0, match.start() - 2) : match.start()]):
                # “非商业”里的“商业”不是商业授权。
                continue
            span = _sentence_span(text, *match.span())
            found.setdefault((name, span.start, span.end), (name, span))
    for match in _INTERNAL_ONLY.finditer(text):
        span = _sentence_span(text, *match.span())
        found.setdefault(("internal_only", span.start, span.end), ("internal_only", span))
    for match in _LICENSE_REFERENCE.finditer(text):
        span = _sentence_span(text, *match.span())
        found.setdefault(("inherits_upstream", span.start, span.end), ("inherits_upstream", span))

    restrictions: list[Span] = []
    for match in _REDISTRIBUTE.finditer(text):
        left, right = _context_bounds(text, *match.span())
        clause = text[left:right]
        if _OBLIGATION.search(clause):
            restrictions.append(_span(text, left, right))

    unknown = [
        _span(text, *_context_bounds(text, *match.span()))
        for match in _LIMITS_EXIST.finditer(text)
    ]
    unknown.extend(
        _span(text, *_context_bounds(text, *match.span()))
        for match in _LICENSE_UNRESOLVED.finditer(text)
    )
    licenses = tuple(value for _key, value in sorted(found.items()))
    return licenses, _dedupe(restrictions), _dedupe(unknown)


def _dedupe(spans: Iterable[Span]) -> tuple[Span, ...]:
    seen: set[tuple[int, int]] = set()
    result: list[Span] = []
    for span in spans:
        if (span.start, span.end) in seen:
            continue
        seen.add((span.start, span.end))
        result.append(span)
    result.sort(key=lambda s: s.start)
    return tuple(result)


def _languages(
    text: str,
) -> tuple[tuple[tuple[str, Span], ...], tuple[tuple[str, Span], ...]]:
    supported: dict[tuple[int, int], tuple[str, Span]] = {}
    excluded: dict[tuple[int, int], tuple[str, Span]] = {}
    for token, code in _LANGUAGE_TOKENS:
        for match in re.finditer(re.escape(token), text):
            left, right = _context_bounds(text, *match.span())
            prefix = text[left : match.start()]
            span = _span(text, *match.span())
            if _LANG_EXCLUDE_CUE.search(prefix):
                excluded.setdefault((span.start, span.end), (code, span))
            else:
                supported.setdefault((span.start, span.end), (code, span))
    for match in _UNIVERSAL_LANGUAGE.finditer(text):
        span = _span(text, *match.span())
        supported.setdefault((span.start, span.end), ("multi", span))
    return (
        tuple(value for _k, value in sorted(supported.items())),
        tuple(value for _k, value in sorted(excluded.items())),
    )


def _eval_matches(text: str) -> list[re.Match]:
    return [
        match
        for match in _EVAL_CUE.finditer(text)
        if not _in_training_context(text, match) and not _is_negated_eval(text, match)
    ]


def _is_negated_eval(text: str, match: re.Match) -> bool:
    """“未报告法律文本评测”说的是没做评测，不该计入已评测。"""

    left, _right = _context_bounds(text, *match.span())
    return bool(re.search(r"未(?:报告|报告过|发现|进行|开展)?[^；;。，,\n]{0,20}$", text[left : match.start()]))


def _in_training_context(text: str, match: re.Match) -> bool:
    left, _right = _context_bounds(text, *match.span())
    return bool(_TRAINING_LEAD.search(text[left : match.start()]))


def _eval_data(text: str) -> tuple[Span, ...]:
    """评测所依据的数据集/任务集合，只取数据集本身，不扩到整句。"""

    spans: list[Span] = []
    seen: set[tuple[int, int]] = set()
    for match in _EVAL_DATASET.finditer(text):
        if _in_training_context(text, match) or _is_negated_eval(text, match):
            continue
        left, _right = _context_bounds(text, *match.span())
        if not _EVAL_CUE.search(text[left : match.end()]):
            continue
        for group in (1, 2):
            if not match.group(group):
                continue
            start, end = match.span(group)
            # “客服意图集合”里的后缀也被 lazy 分组排除在捕获外，补回来。
            suffix = re.match(r"(?:集合|数据集|测试集|集)", text[end:])
            if suffix:
                end += suffix.end()
            span = _span(text, start, end)
            if (span.start, span.end) not in seen:
                seen.add((span.start, span.end))
                spans.append(span)
            break
    spans.sort(key=lambda s: s.start)
    return tuple(spans)


def _eval_exclusions(text: str) -> tuple[Exclusion, ...]:
    results: list[Exclusion] = []
    seen: set[tuple[int, int, str]] = set()
    for cue in _EXCLUSION_CUE.finditer(text):
        left, right = _context_bounds(text, *cue.span())
        rest = text[cue.end() : right]
        for raw in _EXCLUSION_SEP.split(rest):
            task = _EXCLUSION_TRIM.sub("", raw).strip("：: ，,。 ")
            if not task:
                continue
            offset = text.find(task, cue.end(), right)
            if offset < 0:
                continue
            key = (offset, offset + len(task), task)
            if key in seen:
                continue
            seen.add(key)
            results.append(
                Exclusion(
                    task=task,
                    span=_span(text, offset, offset + len(task)),
                    cue=_sentence_span(text, *cue.span()),
                )
            )
    results.sort(key=lambda item: item.span.start)
    return tuple(results)


def _eval_gaps(text: str) -> tuple[Span, ...]:
    """材料明说“未评测/未报告评测”的句子，作为能力证据的空缺，而非否定能力。"""

    return _dedupe(_span(text, *match.span()) for match in _NO_EVAL.finditer(text))


def _training_data(text: str) -> tuple[Span, ...]:
    """适配/训练数据只取包含语料的那个分句，且排除纯评测分句。

    同一句里既有微调描述又有“未报告…评测”时，训练语料仍应取到。
    """

    spans: list[Span] = []
    seen: set[tuple[int, int]] = set()
    for match in _SUBJECT_CUE.finditer(text):
        left, right = _context_bounds(text, *match.span())
        clause = text[left:right]
        if not _TRAINING_LEAD.search(clause):
            continue
        if _EVAL_CUE_STRONG.search(clause) and not re.search(
            r"微调|训练|适配|蒸馏|预训练", clause
        ):
            continue
        span = _span(text, left, right)
        if (span.start, span.end) in seen:
            continue
        seen.add((span.start, span.end))
        spans.append(span)
    spans.sort(key=lambda s: s.start)
    return tuple(spans)


def _flags(text: str) -> tuple[tuple[str, Span], ...]:
    flags: list[tuple[str, Span]] = []
    for label, markers in (("hedged", HEDGE_MARKERS), ("denied", DENIAL_MARKERS)):
        for marker in markers:
            for match in re.finditer(re.escape(marker), text):
                flags.append((label, _span(text, *match.span())))
    flags.sort(key=lambda item: item[1].start)
    return tuple(flags)


def _attribution(record: SourceRecord, text: str) -> tuple[tuple[int, tuple[str, ...]], ...]:
    """把每个句子归属到一组模型。

    * 材料自带 ``models`` 声明时，声明即该材料所有句子的归属。
    * 否则使用“文档主语”：全文最先出现的模型名，作为未点名句子的默认。
    * 句中点名多个模型时，取句首主语；派生关系句归子模型。
    * 全文没有任何模型名时不归属任何模型，由目录列为未指派材料。
    """

    declared = tuple(record.models)
    spans = _models(text)
    if declared:
        return tuple((start, declared) for start in _sentence_starts(text))
    if not spans:
        return ()
    document_models: tuple[str, ...] = (spans[0].quote,)
    result: list[tuple[int, tuple[str, ...]], ...] = []
    for start in _sentence_starts(text):
        end = _sentence_end(text, start)
        present = [span.quote for span in spans if start <= span.start < end]
        # 句首主语优先；省略主语的句子沿用文档主语。
        result.append((start, (present[0],) if present else document_models))
    return tuple(result)


def _ambiguous(record: SourceRecord, text: str) -> tuple[Span, ...]:
    """同一句点名多个模型、又不是显式派生句时，适用对象需人工确认。"""

    if record.models:
        return ()
    spans = _models(text)
    links = {(link.child, link.parent) for link in _derivation(text)}
    known_children = {child for child, _parent in links}
    ambiguous: list[Span] = []
    for start in _sentence_starts(text):
        end = _sentence_end(text, start)
        names = [span.quote for span in spans if start <= span.start < end]
        unique = set(names)
        if len(unique) <= 1:
            continue
        if names and names[0] in known_children and len(unique) == 2:
            # 显式的“子模型基于父模型”不是歧义。
            continue
        ambiguous.append(_span(text, start, end))
    return tuple(ambiguous)


def _sentence_starts(text: str) -> list[int]:
    starts = [0]
    for match in re.finditer(r"[；;。！？\n]", text):
        starts.append(match.end())
    return [start for start in starts if start < len(text)]


def _sentence_end(text: str, start: int) -> int:
    ends = [text.find(brk, start) for brk in _SENTENCE_BREAKS]
    ends = [pos for pos in ends if pos >= 0]
    return min(ends) if ends else len(text)


def extract(record: SourceRecord) -> Extraction:
    """抽取单份材料的字段，所有结果都指回本材料原文。"""

    text = record.text
    licenses, restrictions, restrictions_unknown = _licenses(text)
    languages, language_exclusions = _languages(text)
    return Extraction(
        source=record,
        models=_models(text),
        attribution=_attribution(record, text),
        ambiguous_sentences=_ambiguous(record, text),
        derivation_links=_derivation(text),
        licenses=licenses,
        restrictions=restrictions,
        restrictions_unknown=restrictions_unknown,
        languages=languages,
        language_exclusions=language_exclusions,
        evals=_spans_from_matches(text, _eval_matches(text)),
        eval_data=_eval_data(text),
        eval_exclusions=_eval_exclusions(text),
        eval_gaps=_eval_gaps(text),
        training_data=_training_data(text),
        flags=_flags(text),
    )
