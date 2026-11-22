# 开放模型说明中的能力范围与派生版本沿袭核对

本项目整理企业模型治理团队从模型卡、发布公告和内部适配说明中整理的开放文本模型。上游开放权重并不自动代表所有派生版本沿用相同许可、语言范围或任务能力说明；名称相近的量化版和微调版若被合并，应用团队会把不适用的能力说明带入采购或上线评审。

演练资料位于 `data/models.json`，内容为合成记录，保留不同来源的原始表述和修订关系。项目只用 Python 3.11 标准库。

## 能做什么

`model_catalog` 把一组带版本的文本材料整理成可检索的模型目录，并保证每一条结论都能回到原文：

| 能力 | 说明 |
| --- | --- |
| 原文引用 | 每个字段都是 `Span`，满足 `quote == text[start:end]`，可逐字核对 |
| 归属 | 材料里的每条说明按句子主语归到具体模型；“A 基于 B”归 A，不串到 B |
| 谱系 | 从派生关系递归得到 `lineage`，例如 `Cedar-zh-Q4 -> Cedar-zh -> Base-Cedar` |
| 继承 | 语言、评测可继承并标注 `inherited_from`；许可与许可限制**不自动继承** |
| 待核点 | 缺许可、缺评测、明说未评测、条款未展开、继承边界、基础材料缺失、归属歧义，逐条列出 |
| 修订检测 | 上游材料改版后，未完成的评审标出受影响字段 `stale_fields` |
| 审批冻结 | 已批准的评审保留批准时的目录快照；批准后的上游改动只记入 `drift_since_approval` |
| 未归属材料 | 无法归属到任何模型的分句单独列出，不会被悄悄扩散到所有模型 |

目录整理只服务于内部筛选，不替代法律审查或模型能力验证。

## 快速开始

```bash
# 列出目录中的模型版本，以及未归属的材料
python3 -m model_catalog.cli build data/models.json

# 查看单个候选模型：谱系、字段出处、待核点
python3 -m model_catalog.cli show data/models.json Cedar-zh-Q4

# 输出评审报告（JSON）
python3 -m model_catalog.cli report data/models.json Cedar-zh

# 导出完整目录与评审状态
python3 -m model_catalog.cli export data/models.json -o catalog.json

# 核对所有引用区间是否与原材料一致
python3 -m model_catalog.cli audit data/models.json
```

## 在代码中使用

```python
from model_catalog import Catalog, SourceRecord

catalog = Catalog.from_path("data/models.json")

entry = catalog.entry("Cedar-zh")
entry.lineage                 # ('Base-Cedar',)
entry.fields("language")      # 继承自 Base-Cedar，basis="inherited"
entry.fields("license")       # 空：派生版本未写许可，不能套用上游结论
[c.code for c in entry.risks] # 待核点代码

# 评审：记录当时的来源版本
catalog.start_review("Cedar-zh", reviewer="alice")

# 上游修订后重建目录；未完成评审会被标出受影响字段
revised = SourceRecord(
    id="model-1", kind="model_card", version=2,
    text="Base-Cedar 支持英语文本生成；许可仅限研究使用。",
)
updated = catalog.with_updated_materials([revised])
updated.reviews["Cedar-zh"].stale_fields   # ('language',)：继承的语言受上游修订影响

# 审批后冻结，修订不会覆盖当时引用的文本
updated.approve("Cedar-zh", reviewer="bob")
```

### 派生版本代入

用一份新的适配材料把变体代入目录。明写的字段覆盖继承字段，未明写的语言/评测沿用基础版本并标注来源；许可始终要求明写：

```python
material = SourceRecord(
    id="model-9", kind="release_note", version=1,
    text="Cedar-lite 衍生自 Cedar-zh，使用工业问答语料。",
)
derived = catalog.attach_derived("Cedar-lite", "Cedar-zh", material, relation="finetuned")
derived.entry("Cedar-lite").lineage   # ('Cedar-zh', 'Base-Cedar')
```

## 材料格式

```json
{
  "id": "model-2",
  "kind": "release_note",
  "version": 1,
  "text": "Cedar-zh 是基于 Base-Cedar 的领域微调版本，使用合成客服语料，未报告法律文本评测。",
  "models": ["Cedar-zh"]
}
```

- `id` + `version` 唯一标识一份材料；同名 `id` 的多个版本中默认取最新。
- `kind` 取 `model_card`、`release_note`、`license_addendum`、`benchmark_note`。
- `models` 可选。原文没有逐句点名模型时，用它声明该材料适用于哪些模型；留空则按句首主语归属。

## 设计取舍

- **不按名字合并模型。** `Cedar-zh` 与 `Cedar-zh-Q4` 是两个目录条目，各自有独立的许可与评测状态。
- **许可不自动继承。** 上游开放权重不代表派生版本沿用同样条款，缺失即列为 `license_not_stated`。
- **“未评测”不等于“不支持”。** 材料说没做评测时，记入 `evaluation_gap`，不推断能力缺失。
- **下载量、参数量、名称都不进入证据。** 目录只记录材料明写的内容。
- **句子主语决定归属。** 同句出现多个模型且非显式派生时，标为 `attribution_ambiguous` 待人工确认。

## 测试

```bash
python3 -m unittest discover -s tests -v
```
