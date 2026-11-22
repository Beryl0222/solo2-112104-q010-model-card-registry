"""原始材料的读取、校验与修订版本管理。

目录只处理显式写在材料里的信息。同一 ``id`` 可以有多条 ``version``，
构建目录时默认使用每个 ``id`` 的最新版本，但历史版本仍保留，供审批回看。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SOURCE_KINDS = frozenset(
    {"model_card", "release_note", "license_addendum", "benchmark_note"}
)


class SourceMaterialError(ValueError):
    """原始材料不满足目录构建要求。"""


@dataclass(frozen=True)
class SourceRecord:
    """一份带版本的原始材料。

    ``models`` 是可选的材料级归属声明：当原文没有在每个分句里点明模型名时，
    用它说明该材料适用于哪些模型。留空则按原文句首主语或全文首个模型名归属；
    全文没有任何模型名的材料进入目录的“未指派材料”，而不是被自动套到所有模型上。
    """

    id: str
    kind: str
    version: int
    text: str
    models: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise SourceMaterialError("记录 id 不能为空")
        if self.kind not in SOURCE_KINDS:
            raise SourceMaterialError(f"未知材料类型 {self.kind!r}：{self.id}")
        if (
            not isinstance(self.version, int)
            or isinstance(self.version, bool)
            or self.version < 1
        ):
            raise SourceMaterialError(f"{self.id} 的 version 必须是正整数")
        if not isinstance(self.text, str) or not self.text.strip():
            raise SourceMaterialError(f"{self.id} 的 text 不能为空")
        if not all(isinstance(name, str) and name.strip() for name in self.models):
            raise SourceMaterialError(f"{self.id} 的 models 必须是非空字符串列表")

    @property
    def label(self) -> str:
        return f"{self.id}@v{self.version}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "version": self.version,
            "label": self.label,
        }


def load_records(path: str | Path) -> list[SourceRecord]:
    """从 JSON 文件读取原始材料，并校验必需字段。"""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SourceMaterialError("材料文件顶层必须是记录数组")
    records: list[SourceRecord] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise SourceMaterialError(f"第 {index} 条记录不是对象")
        missing = {"id", "kind", "version", "text"} - item.keys()
        if missing:
            raise SourceMaterialError(
                f"第 {index} 条记录缺少字段：{', '.join(sorted(missing))}"
            )
        records.append(
            SourceRecord(
                id=item["id"],
                kind=item["kind"],
                version=item["version"],
                text=item["text"],
                models=tuple(item.get("models", ())),
            )
        )
    if not records:
        raise SourceMaterialError("材料文件至少需要一条记录")
    return records


class Corpus:
    """一组不可变的材料快照。

    ``latest`` 按 ``id`` 取最高版本；同一 ``corpus`` 内不允许同一
    ``(id, version)`` 出现两次，以免同一引用指向不同原文。
    """

    def __init__(self, records: Iterable[SourceRecord]) -> None:
        rows = tuple(records)
        seen: dict[tuple[str, int], SourceRecord] = {}
        for record in rows:
            if not isinstance(record, SourceRecord):
                raise TypeError("Corpus 只接受 SourceRecord")
            key = (record.id, record.version)
            if key in seen:
                raise SourceMaterialError(f"重复材料版本：{record.label}")
            seen[key] = record
        self._records = tuple(sorted(rows, key=lambda r: (r.id, r.version)))

    @property
    def records(self) -> tuple[SourceRecord, ...]:
        return self._records

    def latest(self) -> tuple[SourceRecord, ...]:
        """每个材料 id 只返回最新版本，按 id 排序。"""

        best: dict[str, SourceRecord] = {}
        for record in self._records:
            current = best.get(record.id)
            if current is None or record.version > current.version:
                best[record.id] = record
        return tuple(best[key] for key in sorted(best))

    def history(self, record_id: str) -> tuple[SourceRecord, ...]:
        return tuple(r for r in self._records if r.id == record_id)

    def latest_version(self, record_id: str) -> int:
        history = self.history(record_id)
        if not history:
            raise KeyError(record_id)
        return history[-1].version

    def extend(self, records: Iterable[SourceRecord]) -> "Corpus":
        return Corpus(self._records + tuple(records))

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)


def load_corpus(path: str | Path) -> Corpus:
    return Corpus(load_records(path))
