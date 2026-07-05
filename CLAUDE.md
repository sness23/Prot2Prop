# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Prot2Prop fine-tunes a **frozen ProstT5 encoder** (`Rostlab/ProstT5`) for joint prediction of
multiple protein developability properties. The architecture is: frozen encoder → one shared
trainable adapter → per-task lightweight residual adapters → attention pooling → per-task heads.
Only the adapters, pooling layer, and heads are trained; the encoder backbone is never updated.

Six tasks are currently active (defined in `aggregate_data.py`, `TASKS`):
- **Classification (bool)**: `material_production`, `solubility`, `temperature_stability`
- **Regression (float)**: `aggregation_propensity`, `expression_yield`, `folding_stability`

Other ProteinGym tasks (binding_affinity, enzymatic_activity, membrane_topology, etc.) are
commented out in `TASKS` because they need interaction-partner/substrate context.

## Pipeline (run in order)

The full pipeline is a linear sequence of scripts that write artifacts into the git-ignored
`data/` directory:

```sh
python aggregate_data.py    # HF datasets + ProteinGym CSVs -> data/aggregated/aggregated.duckdb
python tokenize_data.py     # DuckDB -> data/tokenized/multitask_prostt5_tokens.pt (splits + masks + norm stats)
python train.py             # -> ./checkpoints/prostt5_multitask_adapter_best_{date}_seed_{seed}.pt
```

- `aggregate_data.py` requires ProteinGym data downloaded locally (see README "Download ProteinGym").
  Pass its location with `--proteingym <dir>` (default `DMS_ProteinGym_substitutions`). It reads
  `proteingym_manifests/<task>.csv` to know which DMS CSVs map to each task. HF datasets
  (`AI4Protein/*`) download automatically. Tables: `samples(sequence, source, task_name, label)`
  and `tasks(task_name, dtype, head_type, num_classes, loss, label_semantics)`.
  Inspect with `duckdb -ui data/aggregated/aggregated.duckdb`.
- `tokenize_data.py` does the **global sequence-level split** (80/10/10 by `SPLIT_SEED`), so the
  same sequence never crosses splits. It computes regression normalization stats from the **train
  split only** and caches both raw and normalized labels with per-task presence masks.
- `train.py` is run as a top-level script (not a `main()`); it requires CUDA in practice (enables
  bfloat16, `torch.compile`, fused AdamW). It fits post-hoc calibration on the best epoch's
  validation predictions and embeds it in the checkpoint `config`.

## Evaluation & inference

```sh
# Single checkpoint on a split (validation default; choices: train/validation/test)
python validate.py --checkpoint checkpoints/<file>.pt --split test

# Average predictions across checkpoints (defaults to glob 'prostt5_multitask_adapter_best_*.pt' in cwd)
python ensemble_validate.py --checkpoints checkpoints/*.pt --split test

# Inference from a sequence or FASTA (writes a CSV)
python inference.py --sequence MKT... 
python inference.py --fasta examples/seqs.fasta --output-csv out.csv
python inference.py --download-weights   # prefetch ProstT5 assets, then exit (no prediction)

# One-hot linear sanity baseline (logistic for bool, ridge for float)
python scripts/run_onehot_linear_baseline.py
```

Note: `inference.py` auto-resolves the newest checkpoint by globbing the **current directory**
(not `checkpoints/`), so pass `--checkpoint checkpoints/<file>.pt` explicitly to use a committed
weight. `scripts/plot_results.py` builds Plotly figures from numbers transcribed into `README.md`
(it does not re-run validation).

## Setup, lint

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e .            # core: torch, transformers, sentencepiece, duckdb, neurosnap
pip install -e ".[dev]"     # training/eval: datasets, scikit-learn, plotly, tqdm, ruff, tiktoken, protobuf
ruff check .                # ruff is the only configured linter; there is no test suite
```

## Conventions that matter

- **2-space indentation** for all Python (`ruff` is configured with `indent-width = 2`,
  `line-length = 150`). This is non-standard for Python — match it or `ruff` will complain.
- **Single source of truth for hyperparameters is `config.py`.** `GLOBAL_SEED` drives the split,
  training, and batch-sampler seeds together. To train a new seed, change `GLOBAL_SEED`
  (the checkpoint filename encodes the seed and date). Paths (`AGGREGATED_DB_PATH`,
  `TRAIN_CACHE_PATH`, `TOKENIZED_DATA_DIR`) also live here.
- **Regression is trained on normalized labels** (train-split mean/std), so all tasks contribute
  comparable gradients. Predictions are de-normalized back to raw units for reported metrics and
  inference. Early stopping uses **F1 for classification and −normalized_MAE for regression**,
  averaged across tasks.
- **Checkpoint format** stores separate state dicts: `adapter_state_dict`,
  `task_adapter_state_dicts`, `pool_state_dict`, `head_state_dicts`, plus a `config` block
  (dims, `task_metas`, `regression_mean/std`, embedded `calibration`).
- **Backward compatibility is load-bearing**: `TaskHead` with `hidden_dim=0` reproduces the older
  `LayerNorm + Linear` heads, and `task_adapter_dim=0` makes task adapters `nn.Identity`.
  `validate.py`/`inference.py` reconstruct old or new architectures from the checkpoint `config`
  defaults — preserve these fallbacks when changing the model.
- **Long-sequence OOM is handled by token-budget batching**, not example count. The
  `MultiTaskBatchSampler` caps batches by padded-token budget (`TRAIN_/EVAL_MAX_TOKENS_PER_BATCH`)
  and sorts by length within weighted sample pools. Rare tasks are upweighted via inverse
  label-frequency sample weights so dense tasks don't dominate batches.

## Shared module

`model.py` is imported by all training/eval scripts and holds the dataset, batch sampler, the
`Adapter`/`AttnPool`/`TaskHead`/`MultiTaskAdapterModel` modules, and `collate_multitask_batch`.
`calibration.py` holds the post-hoc affine regression calibrator and F1-tuned binary thresholds,
reused by both `train.py` (fit-and-save) and `validate.py` (report).
