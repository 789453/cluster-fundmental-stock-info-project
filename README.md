# A-share Semantic Engine

A production-oriented Python 3.11 project for building a low-frequency semantic relation layer for A-share research.
It follows the earlier engineering document and emphasizes:

- CSV-first execution when `.npy` semantic embeddings are unavailable
- Optional `NPY + meta.json` multi-view loading when those artifacts are provided later
- Multi-view semantic compact features
- Anchor / quality features
- Lightweight kNN graph construction and graph statistics
- Clustering, prototype exposure, graph smoothing
- Export for daily quant pipelines
- Logging, figures, checkpoints, smoke tests, unit tests

## Supported execution modes

### Mode A: CSV-only fallback
When you only have `records-all.csv`, the engine builds per-view semantic vectors with:
- `TfidfVectorizer`
- `TruncatedSVD`
- row-wise L2 normalization

This is the default and is immediately runnable with the bundled demo dataset.

### Mode B: NPY-backed execution
When you later provide:
- `npy/<view>/<view>-all.npy`
- `npy/<view>/<view>-all.meta.json`

the engine will automatically prefer NPY view embeddings, validate `row_ids` against `record_id`, and reduce them to compact dimensions.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m a_share_semantic_engine.cli smoke --config configs/base.yaml
```

## Main outputs

The pipeline writes a timestamped run folder under `outputs/`:

- `semantic_feature_table.parquet` / `csv`
- `cluster_result.parquet` / `csv`
- `prototype_exposure.parquet` / `csv`
- `graph_stats.parquet` / `csv`
- `graphs/fused_graph.npz`
- `reports/run_report.json`
- `figures/cluster_sizes.png`
- `figures/degree_hist.png`
- `logs/run.log`

## Main package structure

- `a_share_semantic_engine/core`: config, exceptions, logging
- `a_share_semantic_engine/data`: schema, dataset, loaders, validation
- `a_share_semantic_engine/features`: text / npy embeddings, reduction, anchors, quality, clustering, prototypes, fusion
- `a_share_semantic_engine/graph`: graph build, fusion, stats, smoothing
- `a_share_semantic_engine/pipeline`: orchestration and export
- `a_share_semantic_engine/tests`: smoke and unit tests

## Notes

1. The project is intentionally light-weight and stable before heavy GNN.
2. It is designed to integrate with an existing daily factor / risk / trading system.
3. The bundled demo dataset is your uploaded 20-stock `records-all.csv`.
