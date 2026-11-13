# Agent Engineering Guide

This repository is a **local, trusted-host Notebook builder**, not the upstream Nebula platform.

## Architecture and ownership

- `notebook_builder.py`: `nbformat` document lifecycle, CAS, immutable prior versions, atomic writes, `operation_id` receipt replay.
- `notebook_quality.py`: read-only QA for execution records, stdout, PNG header/dimensions and Cell error outputs.
- `notebook_ops.py`: import/export, Cell inspection, version/diff/restore/fork/merge, fresh-kernel execution.
- `workspace_ops.py`: MCP path guard and per-Notebook scoped file API.
- `runtime_manager.py`: session-lifetime stateful Jupyter Kernel (max 4).
- `research_ledger.py`: 本地实验/Trial 记录，SHA-256 评估源码和数据环境声明、真实 Run 结果与工件
- `run_jobs.py` / `run_worker.py`: SQLite and asynchronous local process execution with interrupted recovery.
- `mcp_tools.py`: typed 57-tool registry and **centralized** permission guards; new tools must be added here.
- `mcp_protocol.py`: small standard-input/output JSON-RPC MCP subset; must keep stdout protocol-pure.
- `mcp_server.py`: native/optional official-SDK adapter (both must call guarded `invoke`).
- `nbuild.py`: CLI; should mirror relevant capabilities while keeping local commands explicit.

## Correctness invariants

1. Mutations are lock+CAS atomic. For execution, acquire the lock to take a snapshot, RELEASE it while the kernel runs, then re-lock and compare both source bytes hash and version before commit. On conflict never overwrite a newer file; retain monotonic `metadata.notebook_builder.version`.
2. Do not run a Cell before validating an incoming Notebook and source size caps. No automatic retry of code whose effects are uncertain.
3. Use stable Cell IDs for diff/merge. Never silently resolve competing changes to the same Cell; require a choice.
4. `operation_id` must represent the same input, action and expected version on every retry. Colliding content is an error; receipts are capped to recent 128 entries.
5. No file/API tool may accept an absolute path, traversal or hidden path escaping `NB_ROOT` (CLI direct paths are intentionally trusted-local and different).
6. MCP write and exec gates are independent; defaults are read-only/exec-denied. The official SDK wrapper must **not bypass `invoke`**.
7. MCP stdout is JSON-RPC only; diagnostic output goes to stderr. Avoid future claims of certified protocol conformance without real client checks.
8. SQLite `run_submit` state is evidence, not transactional coupling to Notebook file output. `interrupted` is not retried automatically.

## Required tests

```bash
cd /mnt/notebook-builder
python -m pytest -q tests
python -m compileall -q notebook_quality.py notebook_builder.py notebook_ops.py mcp_tools.py mcp_protocol.py mcp_server.py runtime_manager.py run_jobs.py run_worker.py workspace_ops.py research_ledger.py
nbuild doctor
nbuild mcp-tools
```

Test normal, conflict, traversal, permission, timeout/cancel, and on-disk output paths. Security improvements should be validated with adversarial filenames, unexpected JSON arguments and symlink escapes. Do not merge changes that break existing CLI API names. Avoid networking or root-only dependencies as install requirements.

## Fresh-session deployment contract (must follow)

Canonical repo: `https://github.com/1998x-stack/Notebook-Builder`.

1. Read this file, `README.md`, and [`docs/NEW_SESSION.md`](docs/NEW_SESSION.md) before modifying or installing anything.
2. Establish actual `/mnt` contents. Do not assume earlier ChatGPT sessions share a filesystem.
3. If repo source is available, use `scripts/install-to-mnt.sh --quick` from the checkout or unpacked ZIP; it is conservative and will not overwrite a nonempty target. Do not hardcode credentials or require Docker.
4. Run `python3 /mnt/notebook-builder/nbuild.py doctor`; run smoke tests. State clearly whether GitHub clone, installation, kernel execution, and MCP client registration actually succeeded.
5. For tutorial authoring, use spec → notebook → execute → quality → export; keep first-class `.ipynb` and source + tests in git, generated HTML/figures/checkpoints outside git unless intentionally designated release artifacts.
6. ResNet Colab guide: [`docs/COLAB_RESNET.md`](docs/COLAB_RESNET.md). Notebook must select CUDA if available, otherwise CPU, avoid automatic network downloads, and remain standalone (all model definitions embedded).
7. New sessions may use a connected GitHub plugin to fetch/publish repository content when shell internet access is unavailable. **Do not claim the repository or remote CI were modified without readback evidence.**

### Minimal bootstrap

```bash
git clone https://github.com/1998x-stack/Notebook-Builder.git /mnt/notebook-builder
bash /mnt/notebook-builder/scripts/install-to-mnt.sh --quick
python3 /mnt/notebook-builder/nbuild.py doctor
```

If git networking is blocked, use a verified source ZIP then execute `scripts/install-to-mnt.sh --quick` from the unpacked directory.
