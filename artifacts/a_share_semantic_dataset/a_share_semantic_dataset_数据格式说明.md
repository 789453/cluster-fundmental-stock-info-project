# A 股语义数据集（a_share_semantic_dataset）数据格式说明

本文档说明目录 `d:\Trading\stock_semantic_pipeline_project\artifacts\a_share_semantic_dataset` 下的数据结构、字段含义与使用方式，便于后续将数据信息喂给 AI（LLM/RAG/向量检索）。

## 1. 总览

该数据集采用“三层存储”：

- **Parquet（结构化文本层）**：`parquet/records-all.parquet`，保存每只股票的抽取结果（文本/JSON 字符串/标签等），是“事实源”（source of truth）。
- **NPY（向量层）**：`npy/<view>/<view>-all.npy`，按字段（view）对每条记录做 embedding，方便向量检索/聚类/相似度计算。
- **Metadata（索引层）**：`metadata/*.jsonl/*.json`，保存运行记录、索引、失败信息、合并报告等。

当前数据集状态（以本次产物为准）：

- 记录行数：`5502`
- 股票数（unique stock_code）：`5502`
- embedding 维度：`1024`
- view 数：`23`

## 2. 目录结构

```
artifacts/
  a_share_semantic_dataset/
    parquet/
      records-all.parquet
    npy/
      <view>/
        <view>-all.npy
        <view>-all.meta.json
    metadata/
      records.jsonl
      manifest.json
      npy-merge-all.json
      checkpoints/
        processed_codes.txt
```

其中 `<view>` 是“被向量化的字段名”，例如：`profile_text`、`product_text`、`business_model_json`、`raw_json` 等。

## 3. Parquet：结构化文本数据（records-all.parquet）

### 3.1 核心概念

- **一行 = 一条股票记录**（通常是一只股票在某个 `asof_date` 的语义快照）。
- **record_id**：该行记录的唯一 ID（字符串），同时也是向量层的 `row_id`（用于对齐）。
- **view 列**：用于向量化的字段列，既可能是自然语言文本，也可能是 JSON 字符串（数组/对象）。

### 3.2 Parquet 列（schema 摘要）

当前 `records-all.parquet` 共 35 列（示例）：

- 标识与时间：
  - `record_id`：记录 ID（字符串）
  - `stock_code`：股票代码（如 `000001.SZ`）
  - `stock_name`：股票名称
  - `asof_date`：数据日期（字符串，形如 `YYYY-MM-DD`）
  - `language`：语言（如 `zh-CN`）
  - `created_at`：生成时间（ISO 字符串）
- 文本类 view（直接可喂给 LLM）：
  - `profile_text` / `product_text` / `model_text` / `chain_text` / `theme_text` / `full_text`
  - `main_business_one_liner` / `main_business_detail` / `source_quality_note`
- 标签类 view（值通常是短标签字符串）：
  - `biz_form`（如 `service`）
  - `customer_side`（如 `mixed`）
  - `chain_level`（如 `midstream`）
- JSON 字符串类 view（值是 JSON 的字符串形式）：
  - `business_model_json` / `business_logic_json` / `industry_chain_position_json`
  - `core_products_services_json` / `end_markets_json` / `customer_types_json`
  - `application_scenarios_json` / `business_scope_keywords_json` / `structural_themes_json`
  - `uncertainties_json`
  - `raw_json`（包含上述信息的更完整 JSON 串）
- 版本信息：
  - `schema_version` / `prompt_version` / `embedding_version`
  - `extraction_model` / `embedding_model`

### 3.3 文本字段示例（来自 Parquet 第一行，截取）

```json
{
  "stock_code": "000001.SZ",
  "stock_name": "平安银行",
  "asof_date": "2024-06-30",
  "profile_text": "National joint-stock commercial bank licensed by CBIRC; core business: deposit-taking, lending (retail & corporate), payment settlement, wealth management, trade finance, and custodial services; operates through physical branches, mobile app, and API-based open banking.",
  "product_text": "Personal loans (mortgage, business, credit card), corporate loans (working capital, project, trade finance), structured deposits, agency-sold wealth products, FX services, cash management, supply chain financing, custody, and insurance-linked credit solutions.",
  "main_business_one_liner": "持牌全国性股份制商业银行，以零售与对公双轮驱动， 提供全牌照银行服务。",
  "biz_form": "service",
  "customer_side": "mixed",
  "chain_level": "midstream"
}
```

### 3.4 JSON 字符串字段示例（来自 Parquet 第一行，截取）

注意：以下字段在 Parquet 中是“字符串”，要先 `json.loads(...)` 才能得到 Python dict/list。

```json
{
  "record_id": "f28b37e410e91cba67ee44b2",
  "business_model_json": "{\"r_and_d_mode\": \"...\", \"production_or_delivery_mode\": \"...\", \"sales_mode\": \"...\"}",
  "business_scope_keywords_json": "[\"商业银行\", \"零售银行\", \"公司银行\", \"资金同业\", \"资产管理\", \"...\"]",
  "application_scenarios_json": "[\"小微企业日常经营资金周转\", \"居民购房与消费升级融资\", \"...\"]",
  "uncertainties_json": "[]",
  "raw_json": "{\"stock_code\": \"000001.SZ\", \"stock_name\": \"平安银行\", \"asof_date\": \"2024-06-30\", \"language\": \"zh-CN\", ...}"
}
```

## 4. NPY：向量化结果（npy/<view>/*-all.npy）

### 4.1 view 的含义

view = “要做 embedding 的某一列字段名”。每个 view 会生成一份独立的向量矩阵，便于：

- 只检索某类语义（如只用 `product_text` 做相似检索）
- 做多视图融合（multi-view）或加权组合
- 对不同数据类型（文本/标签/JSON）分别建索引

当前已生成并合并的 view 共 23 个，对应 `npy/` 下的 23 个子目录（与 Parquet 列名一致）。

### 4.2 NPY 文件格式

以 `product_text` 为例：

- 向量文件：`npy/product_text/product_text-all.npy`
- 元信息：`npy/product_text/product_text-all.meta.json`

`*-all.npy`：

- 类型：NumPy `.npy` 文件
- dtype：`float32`
- shape：`(rows, dim)`，本数据集为 `(5502, 1024)`

`*-all.meta.json`（关键字段）：

- `view`：view 名称
- `path`：对应 npy 文件路径（相对 artifacts）
- `rows`：行数（与 Parquet 记录数一致）
- `dim`：向量维度
- `row_ids`：长度为 `rows` 的列表，每个元素是一个 `record_id`
- `non_finite`：该 view 合并时统计的非有限值计数（NaN/Inf）；本数据集为 0
- `zero_norm`：合并时统计的“向量范数接近 0”的行数；本数据集为 0
- `l2_min/l2_mean/l2_max`：向量 L2 范数统计（用于快速检查 embedding 是否归一化/是否异常）

### 4.3 通过 record_id 读取向量（关键用法）

NPY 向量矩阵的第 `i` 行，对应 `meta["row_ids"][i]` 的那条记录。

推荐流程：

1) 从 Parquet 里定位目标记录（拿到 `record_id`）。  
2) 从 `view-all.meta.json` 找到该 `record_id` 的行号 `i`。  
3) `np.load(view-all.npy)[i]` 取出向量。

示例代码（按 record_id 取某个 view 的向量）：

```python
import json
from pathlib import Path
import numpy as np

view = "product_text"
meta_path = Path("artifacts/a_share_semantic_dataset/npy") / view / f"{view}-all.meta.json"
npy_path = Path("artifacts/a_share_semantic_dataset/npy") / view / f"{view}-all.npy"

meta = json.loads(meta_path.read_text(encoding="utf-8"))
row_ids = meta["row_ids"]
index = {rid: i for i, rid in enumerate(row_ids)}  # 建议缓存到磁盘或内存，避免每次 O(n) 查找

record_id = "f28b37e410e91cba67ee44b2"
i = index[record_id]

mat = np.load(npy_path, mmap_mode="r")  # mmap 适合大文件
vec = mat[i]  # shape = (1024,)
```

### 4.4 重要提示：records.jsonl 的 vector_paths 可能过期

`metadata/records.jsonl` 是向量化阶段生成的索引文件，其中 `vector_paths` 记录了“当时分片版”的路径（例如 `view-000000.npy`）。

当你执行过 “合并 NPY 分片并删除分片”（`merge-npy --delete-shards true`）之后：

- 原分片 `view-000xxx.npy` 会被删除
- 这会导致 `records.jsonl` 的 `vector_paths` 指向的路径不存在

因此，合并后推荐以 `view-all.meta.json` 的 `row_ids` 为准做对齐与索引。

## 5. Metadata：索引/运行记录

### 5.1 records.jsonl（记录级索引）

路径：`metadata/records.jsonl`，每行一个 JSON object，字段示例：

- `record_id`：记录 ID（与 Parquet 的 record_id 一致）
- `stock_code` / `stock_name` / `asof_date`
- `parquet_path`：对应 Parquet 文件路径
- `vector_paths`：view -> 向量文件路径（可能是分片路径，合并后可能失效，见 4.4）

用途：

- 快速按股票定位记录（不必读 Parquet）
- 保存生成时的向量落盘路径（在未合并前很有用）

### 5.2 npy-merge-all.json（向量合并报告）

路径：`metadata/npy-merge-all.json`

用途：

- 记录合并时的强校验：`expected_rows`、`expected_dim`、`validate_row_ids`
- 记录每个 view 合并统计：`shards_merged`、`non_finite`、`zero_norm`、`l2_*`

你要喂给 AI 时，可以把这份报告当作“数据健康证明”。

### 5.3 checkpoints/processed_codes.txt（抽取断点）

路径：`metadata/checkpoints/processed_codes.txt`

用途：

- 抽取阶段/流水线断点续跑时记录已处理股票代码
- 仅用于工程运行控制，不是建模数据的必要输入

### 5.4 manifest.json

路径：`metadata/manifest.json`

用途：

- 预留的运行汇总信息（本工程里有写入逻辑，但不保证该文件一定反映最终统计）

## 6. 如何“喂给 AI”

### 6.1 直接把结构化信息喂给 LLM（推荐）

从 Parquet 中读取 `raw_json`（或组合多个 text 字段）喂给 LLM 最稳健：

- `raw_json`：信息更全，但需要 `json.loads`
- `full_text`：更适合做摘要/问答
- `profile_text/product_text/...`：更适合做“定向”提问（如只问产品/客户/产业链）

示例：拼一个“给 LLM 的输入”：

```python
import json
import pyarrow.dataset as ds
from pathlib import Path

p = Path("artifacts/a_share_semantic_dataset/parquet/records-all.parquet")
dataset = ds.dataset(str(p), format="parquet")

stock_code = "000001.SZ"
table = dataset.to_table(filter=(ds.field("stock_code") == stock_code), columns=["stock_code","stock_name","asof_date","raw_json","full_text"])
row = table.to_pydict()

raw = json.loads(row["raw_json"][0])
prompt_payload = {
  "stock_code": raw["stock_code"],
  "stock_name": raw["stock_name"],
  "asof_date": raw["asof_date"],
  "business": raw.get("main_business_one_liner"),
  "detail": raw.get("main_business_detail"),
  "notes": raw.get("source_quality_note"),
}
```

### 6.2 用向量做检索（RAG / 相似股票 / 聚类）

典型两步：

1) **候选召回**：对 query 生成同维度向量（同一 embedding 模型），与某个 view 的 `*-all.npy` 做相似度计算（通常 cosine）。  
2) **再排序/生成**：拿 TopK 的记录，从 Parquet 取出 `raw_json/full_text` 等字段喂给 LLM 做回答。

因为本数据集的向量 L2 范数约为 1（`l2_mean≈1.0`），所以 cosine 相似度可以用点积近似：

```python
import numpy as np

# mat: (N, 1024) 已归一化
# q: (1024,) 已归一化
scores = mat @ q
topk = np.argsort(-scores)[:20]
```

## 7. 常见问题

### 7.1 为什么既有 *_text 又有 *_json？

- `*_text`：适合直接给 LLM、也适合做自然语言向量检索。
- `*_json`：更结构化，适合后续做字段级分析、聚类、规则抽取，也可以向量化用于“结构语义检索”（本工程是把 JSON 串作为文本去做 embedding）。

### 7.2 如何保证 Parquet 与 NPY 对齐？

对齐锚点是 `record_id`：

- Parquet 每行都有 `record_id`
- 每个 `view-all.meta.json` 的 `row_ids` 列表存的就是同一批 `record_id`，并且顺序一致

只要用 `record_id -> 行号` 映射，就可以在任意 view 中取到同一条记录的向量。

