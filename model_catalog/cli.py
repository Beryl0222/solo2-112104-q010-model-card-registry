"""命令行入口：构建目录、查看候选模型、导出评审报告。

用法示例::

    python3 -m model_catalog.cli build  data/models.json
    python3 -m model_catalog.cli show   data/models.json Cedar-zh
    python3 -m model_catalog.cli export data/models.json -o catalog.json
    python3 -m model_catalog.cli audit  data/models.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .catalog import Catalog, CatalogError
from .records import Corpus, load_corpus


def _load(path: str) -> Catalog:
    return Catalog(load_corpus(path))


def _source_text(corpus: Corpus, source_id: str, version: int) -> str:
    for record in corpus.records:
        if record.id == source_id and record.version == version:
            return record.text
    raise CatalogError(f"目录引用了不存在的材料 {source_id}@v{version}")


def audit(catalog: Catalog) -> list[str]:
    """核对每个目录字段的引用区间确实指向自称的来源原文。"""

    problems: list[str] = []
    for entry in catalog.entries():
        for ref in entry.refs:
            if not ref.quote:
                problems.append(f"{entry.name}: {ref.field} 缺少原文引用（{ref.source_label}）")
                continue
            text = _source_text(catalog.corpus, ref.source_id, ref.source_version)
            actual = text[ref.start : ref.end]
            if actual != ref.quote:
                problems.append(
                    f"{entry.name}: {ref.field} 的引用与 {ref.source_label} 原文不一致"
                    f"（{ref.quote!r} != {actual!r}）"
                )
            if ref.basis == "inherited" and not ref.inherited_from:
                problems.append(f"{entry.name}: {ref.field} 标为继承但未记录来源模型")
    return problems


def _print_entry(entry) -> None:
    print(f"模型：{entry.name}")
    print(f"  谱系：{' <- '.join(entry.lineage) if entry.lineage else '（无派生来源）'}")
    print(f"  材料版本：{', '.join(entry.updated_at) or '（无）'}")
    if entry.risks:
        print("  待核点：")
        for risk in entry.risks:
            quote = f"「{risk.quote}」" if risk.quote else ""
            print(f"    - [{risk.code}] {risk.message} {quote}".rstrip())
    print("  字段与出处：")
    for ref in entry.refs:
        basis = "继承" if ref.basis == "inherited" else "明写"
        inherited = f"，源自 {ref.inherited_from}" if ref.inherited_from else ""
        print(f"    - {ref.field}（{basis}{inherited}）{ref.source_label}：「{ref.quote}」")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="开放文本模型版本目录")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="列出目录中的模型版本")
    build.add_argument("materials")

    show = sub.add_parser("show", help="查看单个候选模型")
    show.add_argument("materials")
    show.add_argument("name")

    export = sub.add_parser("export", help="导出完整目录与评审状态")
    export.add_argument("materials")
    export.add_argument("-o", "--output")

    check = sub.add_parser("audit", help="核对引用是否与原文一致")
    check.add_argument("materials")

    report = sub.add_parser("report", help="输出候选模型的评审报告")
    report.add_argument("materials")
    report.add_argument("name")

    args = parser.parse_args(argv)
    try:
        catalog = _load(args.materials)
        if args.command == "build":
            for entry in catalog.entries():
                print(entry.name)
            unassigned = catalog.unassigned()
            if unassigned:
                print("\n未归属材料（未套用到任何模型）：")
                for item in unassigned:
                    print(f"  {item['source']}：「{item['quote']}」")
            return 0
        if args.command == "show":
            _print_entry(catalog.entry(args.name))
            return 0
        if args.command == "report":
            print(json.dumps(catalog.review_report(args.name), ensure_ascii=False, indent=2))
            return 0
        if args.command == "export":
            blob = catalog.to_json()
            if args.output:
                Path(args.output).write_text(blob + "\n", encoding="utf-8")
            else:
                print(blob)
            return 0
        if args.command == "audit":
            problems = audit(catalog)
            if problems:
                print("发现引用问题：")
                for problem in problems:
                    print(f"  - {problem}")
                return 1
            print(f"引用校验通过：{len(catalog.entries())} 个模型，所有字段均指回原文。")
            return 0
    except (CatalogError, KeyError, FileNotFoundError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
