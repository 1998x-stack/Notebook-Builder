---
name: notebook-builder
description: >
  Create, edit, execute, validate and publish reproducible Jupyter notebooks in
  /mnt. Use for .ipynb authoring, code/Markdown cells, versioned editing,
  experimental analysis, charts, outputs, HTML previews and MCP tool workflows.
---

# Notebook Builder — Local Agent Skill v2.1

## When to use

Any request to create or improve a Jupyter `.ipynb`, run trusted Python with
reproducible outputs, generate a notebook report, or expose notebooks through
an MCP tool workflow. This local file is not automatically a registered ChatGPT
platform Skill: run commands explicitly or import the Python API.

## Entrypoints

- CLI: `/mnt/notebook-builder/nbuild`
- MCP: `/mnt/notebook-builder/nbuild-mcp` (stdio; **read-only default**)
- API: `notebook_builder.py`, `notebook_ops.py`, `run_jobs.py`
- Outputs: `/mnt/data/notebooks/`

## Required workflow

1. **Inspect**: Run `nbuild doctor`, `nbuild list`, `nbuild inspect` for any existing notebook. Do not overwrite a file without the user's authorization.
2. **Plan**: Structure cells as title/context, imports and environment, data provenance, methods, execution, charts/tables, interpretation and limitations. Fixed random seeds and explicit data assumptions are preferred.
3. **Author**: `nbuild create --spec` or `nbuild add`, `update`, `move`, `delete`. Favor Cell IDs over indices and pass `--expected-version`. For retrying a mutation, set a unique stable `--operation-id`.
4. **Validate**: `nbuild validate`; check source for unintended side effects, secrets, unbounded loops, network/external code execution.
5. **Execute**: Only run **trusted** code when task requires it, via `nbuild execute` or `run-cells`; failure should leave the original Notebook unchanged. `run-submit` is a separately launched local subprocess and can outlive its client connection, but not necessarily the host container.
6. **Review**: inspect outputs, figures, test assertions and unexplained errors; if necessary, repair selected cells and rerun. Use `nbuild diff`, `versions`, `merge-preview` for conflicting edits.
7. **Quality gate**: Run `nbuild quality` on executed tutorial notebooks; treat `passed=false` as failure, inspect broken PNG and missing stdout, and avoid skipping missing built artifacts. For long-running code, use versioned snapshot/CAS so concurrent human edits remain intact.
8. **Deliver**: `nbuild html` / `markdown` as appropriate; provide sandbox links to verified files under `/mnt/data` and state test/validation status.

## Agent operation rules

- Never assert arbitrary code execution is isolated. Runtime is the user's ordinary Python kernel, **no Docker sandbox**.
- MCP write is disabled unless `NB_MCP_READONLY=0`; code execution also requires `NB_MCP_ALLOW_EXEC=1`. Never silently relax these settings.
- MCP `workspace_*` is scoped to a per-notebook hidden directory, but code execution itself is NOT scoped to that directory.
- `runtime_start` retains variables only during the same MCP server process. `cell_run` uses a new kernel with bootstrap, not the live kernel.
- `run_submit` uses SQLite plus local subprocess, with explicit idempotency keys and no automatic retry on unknown results. Poll with `run_wait`/`run_status`; cancel only when requested.
- Maintain version history and snapshots, do not fabricate data, benchmarks, citations or experiments.
- Do not claim upstream's full 54-tool server, web UI, RBAC, Docker, GPU or CRDT is installed.

## Quick commands

```bash
nbuild doctor
nbuild create /mnt/data/notebooks/example.ipynb --title 'Example'
nbuild add /mnt/data/notebooks/example.ipynb --type code --source 'print(2 + 2)' --expected-version 1 --operation-id add-001
nbuild execute /mnt/data/notebooks/example.ipynb
nbuild validate /mnt/data/notebooks/example.ipynb
nbuild html /mnt/data/notebooks/example.ipynb --output /mnt/data/notebooks/example.html
```

Further docs: `README.md`, `docs/MCP_TOOL_MAP.md`, `docs/ARCHITECTURE.md`, `docs/TEST_REPORT.md`.

## Experimental evaluation (optional)

For a bounded research experiment, create a frozen evaluator Cell and compute
real metrics; use `research_create` with caller-asserted data/environment hashes,
register successful candidate Run IDs, execute the evaluator as a separate
successful Run, and finalize a Trial only after parsing `NB_METRIC_V1` output.
Treat reported results as evidence-linked local runs, **not** as independent
or tamper-proof third-party validation. Respect trial budgets.

## Tutorial collections / 图解课程

若用户提出“用 tutorials（print 和 figures）教学某个代码仓库”，优先制作一个可执行的分节系列：**明确锁定上游提交、列出原始函数映射、讲原理、print 中间形状/数值、至少一个真实 matplotlib figure、断言、练习、独立执行及 HTML 导出**。严禁把随机或合成的训练曲线标成上游基准。

已实施的 nanoGPT 教学集合：`/mnt/notebook-builder/tutorials/nanogpt/`。

```bash
nbuild tutorial nanogpt --list
nbuild tutorial nanogpt --build --execute --html
# 显式更新（保留历史版本）
nbuild tutorial nanogpt --build --execute --html --force
```

参考 `tutorials/nanogpt/README.md`、`tutorials/nanogpt/index.html` 及可修改的 `specs/*.json`。此教学代码保留 GPT 核心概念，不包含生产模型训练、任意外部代码下载执行或 GPU 性能保证。

## Reproducibility policy

When a user asks for "paper reproduction" or "复现论文" instead of an explanatory tutorial, refer to [`docs/PAPER_REPRODUCTION.md`](docs/PAPER_REPRODUCTION.md), and use `nbuild paper-resnet` / `nbuild reproduce-quality` for real-dataset protocol programs. Do not replace unavailable real datasets with synthetic data or treat `nbuild quality` success as scientific evidence. Generated scientific Notebooks are `protocol_ready` until genuine real-data experiments finish and are externally verified.
