# Notebook Builder 2.1 — P0 Reliability Optimization

## Scope implemented

1. **Optimistic execution commit:** `notebook_builder.execute`, `notebook_ops.execute_cells`, and `notebook_ops.code_execute` release the per-file advisory lock while an unprivileged, trusted local Jupyter kernel runs. They capture a validated snapshot and compare both the committed notebook version and canonical content SHA-256 before writing new output. Concurrent edits remain intact, and the stale runner receives `ConflictError`. Executed code may still have irrecoverable side effects outside the notebook.
2. **Source-spec drift gate:** `build_tutorials.run_series` uses canonical SHA-256 and compares saved cell source/order/type against generated specs, rejecting stale notebooks without explicit `--force`. New notebooks persist a source spec digest.
3. **Output QA:** `notebook_quality.inspect` checks valid notebook schema, complete execution counts, stdout evidence, PNG signature/dimensions, and error outputs. New read-only MCP tool `notebook_quality`; CLI `nbuild quality` returns exit 2 if contract not met. This is structural evidence only, not scientific validation.
4. **Worker logging:** job subprocesses emit per-run error/diagnostic logs to `.notebook-runs/logs/<run_id>.log` instead of discarding stderr; names are generated UUIDs, filesystem entries use exclusive creation and no-follow, and file permissions default to `0600`.
5. **Strict tests:** regressions for version conflicts, byte mutation without version increments, no long-held locks, tutorial spec drift and explicit force rebuilding, invalid figures, and CLI nonzero status. Missing tutorial files now fail instead of skipping.

## Remaining risks and next priorities

- **Not a sandbox:** user-supplied Python has the full privileges of the local Python process. Public multi-tenant use would require a separate container/microVM worker control plane, resource quotas, and network restrictions.
- **Not a total-process execution timeout:** nbclient's timeout governs cells. Host/kernel failure can still leave unknown side effects; add process-level watchdog and controlled cancellation as a separate change.
- **Artifacts:** Google Drive archival of `.ipynb` does not automatically upload a sibling `nanogpt_edu.py` or set Colab's cwd. Keep a companion module and clear bootstrapping instructions; a later portable export mode should remove this runtime dependency.
- **No automatic LLM repair loop:** Reflection Loop is a separately scoped feature requiring immutable attempts, capped retries, error taxonomy, and review gates. No such automatic source editing is claimed in this release.
- **Provenance:** New specs store authoring source SHA-256; reproducibility across differing dependency versions still needs environment snapshots/lockfiles, dataset provenance, and independent evaluations.

## Tests

Run `pytest -q tests/test_optimizations.py`, then the full suite in isolated groups or one process, recording both exact commands and outcomes. CLI smoke: `nbuild quality <executed-file.ipynb>`, `nbuild mcp-tools` (expect 57 tools), and `nbuild tutorial nanogpt --list`. Do not count previously timed-out whole-suite runs as passing.
