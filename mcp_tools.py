"""Shared, dependency-light Notebook MCP tool registry.

Registration is independent of a specific MCP SDK. The optional official-SDK
adapter and the tested minimal stdio JSON-RPC transport expose the same tools.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, get_type_hints

from pydantic import create_model

import notebook_builder as core
import notebook_ops as ops
import workspace_ops as workspace
import run_jobs as jobs
import research_ledger as research
from runtime_manager import runtime_manager

ROOT = Path(os.getenv('NB_ROOT', '/mnt/data/notebooks')).resolve()
READ_ONLY = os.getenv('NB_MCP_READONLY', '1') != '0'
ALLOW_EXEC = os.getenv('NB_MCP_ALLOW_EXEC', '0') == '1'

@dataclass
class Tool:
    name: str
    fn: Callable
    permission: str
    description: str
    schema: dict
    validator: Any

REGISTRY: dict[str, Tool] = {}


def tool(*, permission: str = 'read'):
    assert permission in ('read', 'write', 'execute', 'destructive')
    def decorate(fn: Callable):
        hints = get_type_hints(fn)
        sig = inspect.signature(fn)
        fields = {}
        for name, param in sig.parameters.items():
            fields[name] = (hints.get(name, Any), ... if param.default is inspect.Parameter.empty else param.default)
        model = create_model(f'{fn.__name__}Args', **fields)
        schema = model.model_json_schema()
        schema['additionalProperties'] = False
        REGISTRY[fn.__name__] = Tool(fn.__name__, fn, permission,
                                    inspect.getdoc(fn) or '', schema, model)
        return fn
    return decorate


def require(permission: str) -> None:
    if permission in ('write', 'destructive') and READ_ONLY:
        raise PermissionError('MCP write operations disabled: set NB_MCP_READONLY=0')
    if permission == 'execute' and (READ_ONLY or not ALLOW_EXEC):
        raise PermissionError('MCP execution disabled: set NB_MCP_READONLY=0 and NB_MCP_ALLOW_EXEC=1')


def invoke(name: str, arguments: dict | None = None) -> Any:
    if name not in REGISTRY:
        raise KeyError(f'No tool named {name}')
    entry = REGISTRY[name]
    require(entry.permission)
    validated = entry.validator.model_validate(arguments or {}, extra='forbid')
    return entry.fn(**validated.model_dump())


def _nb_path(relative: str) -> Path:
    return workspace.checked_notebook(ROOT, relative)


def dest(relative: str, suffix: str) -> Path:
    if not isinstance(relative, str) or not relative or '\x00' in relative:
        raise ValueError('Invalid output path')
    raw = Path(relative)
    if raw.is_absolute() or '..' in raw.parts or any(p.startswith('.') for p in raw.parts):
        raise ValueError('Output must be a relative path without dot-directories')
    target = (ROOT / raw).resolve()
    if not target.is_relative_to(ROOT) or target.suffix.lower() != suffix:
        raise ValueError(f'Output must be a path inside NB_ROOT ending in {suffix}')
    return target


@tool()
def nebula_health() -> dict:
    """Check local notebook-builder and MCP environment availability."""
    return {'status': 'ok', 'root': str(ROOT), 'read_only': READ_ONLY,
            'execute_enabled': not READ_ONLY and ALLOW_EXEC, 'service': 'notebook-builder-local'}


@tool()
def nebula_capabilities() -> dict:
    """List verified local capabilities; not a hosted Nebula server."""
    return ops.execution_diagnostics() | {'mcp_tool_count': len(REGISTRY)}


@tool()
def notebook_list(recursive: bool = True) -> list[dict]:
    """List Notebook files and their document versions under NB_ROOT."""
    return ops.list_notebooks(ROOT, recursive=recursive)


@tool()
def notebook_get(path: str, include_outputs: bool = False) -> dict:
    """Get Notebook metadata and complete sources with bounded output previews."""
    return ops.notebook_get(_nb_path(path), include_outputs=include_outputs)


@tool(permission='read')
def notebook_quality(path: str, require_executed: bool = True,
                     min_print_cells: int = 2, min_figure_cells: int = 1) -> dict:
    """Validate real saved stdout, PNG figure and execution evidence without running code."""
    from notebook_quality import inspect
    return inspect(_nb_path(path), require_executed=require_executed,
                   min_print_cells=min_print_cells, min_figure_cells=min_figure_cells)


@tool(permission='read')
def paper_reproduction_quality(path: str, scope: str = 'cifar10_paper') -> dict:
    """Check declared real-dataset evidence and full paper-coverage gates; synthetic/partial runs fail."""
    from paper_reproduction import inspect
    from workspace_ops import checked_notebook
    # Guarded workspace API: evidence must be within NB_ROOT, never arbitrary host paths.
    raw = Path(path)
    if raw.is_absolute() or '..' in raw.parts or any(part.startswith('.') for part in raw.parts) or not path.endswith('.json'):
        raise ValueError('Evidence path must be a relative JSON path under NB_ROOT')
    full = (ROOT / raw).resolve()
    if not full.is_relative_to(ROOT): raise ValueError('Evidence path escapes NB_ROOT')
    if full.stat().st_size > 2_000_000: raise ValueError('Evidence manifest exceeds 2MB')
    return inspect(full, scope=scope)

@tool(permission='write')
def notebook_create(path: str, title: str = '', cells: list[dict] | None = None) -> dict:
    """Create a standard .ipynb from a title and optional cell specifications."""
    return core.create(_nb_path(path), title=title, cells=cells)


@tool(permission='write')
def notebook_create_file(path: str, title: str = '') -> dict:
    """Alias for creating a real on-disk .ipynb document."""
    return core.create(_nb_path(path), title=title)


@tool(permission='write')
def notebook_import(source_path: str, destination_path: str) -> dict:
    """Import a validated notebook from a file already contained inside NB_ROOT."""
    return ops.import_notebook(_nb_path(source_path), _nb_path(destination_path))


@tool(permission='write')
def notebook_export(path: str, destination_path: str, format: str = 'ipynb') -> dict:
    """Export a notebook to a separate .ipynb, .html or .md file under NB_ROOT."""
    p = _nb_path(path)
    if format == 'ipynb':
        return ops.export_ipynb(p, dest(destination_path, '.ipynb'))
    if format == 'html':
        return core.export_html(p, dest(destination_path, '.html'))
    if format == 'markdown':
        return ops.export_markdown(p, dest(destination_path, '.md'))
    raise ValueError('format must be ipynb, html or markdown')


@tool(permission='write')
def notebook_rename(path: str, title: str, expected_version: int | None = None) -> dict:
    """Update the Notebook title metadata without renaming its stable file path."""
    return ops.title(_nb_path(path), title, expected_version=expected_version)


@tool(permission='destructive')
def notebook_delete(path: str, expected_version: int | None = None) -> dict:
    """Soft-delete a notebook into the hidden recoverable trash folder."""
    p = _nb_path(path)
    if runtime_manager.status(p)['status'] == 'ready':
        raise RuntimeError('Stop the running kernel before deleting its Notebook')
    return ops.delete_notebook(p, expected_version=expected_version)


@tool()
def notebook_versions(path: str) -> dict:
    """List historical immutable versions and the current document version."""
    p = _nb_path(path)
    history = core.history(p)
    return {'path': str(p), 'current_version': core.describe(p)['version'],
            'history': [{'version': int(Path(entry).stem[1:7]), 'path': entry}
                        for entry in history['snapshots']]}


@tool()
def notebook_version(path: str, version: int, include_outputs: bool = False) -> dict:
    """Read a particular retained Notebook revision by integer version."""
    return ops.get_version(_nb_path(path), version, include_outputs=include_outputs)


@tool(permission='write')
def notebook_fork(path: str, version: int, destination_path: str) -> dict:
    """Fork a retained document version into an independent Notebook file."""
    return ops.fork_version(_nb_path(path), version, globals()['path'](destination_path))


@tool(permission='write')
def notebook_restore(path: str, version: int, expected_version: int) -> dict:
    """Restore a prior snapshot as a new version; preserve history."""
    return ops.restore_version(_nb_path(path), version, expected_version=expected_version)


@tool()
def notebook_diff(path: str, from_version: int, to_version: int | None = None) -> dict:
    """Compare cells by stable ID across two Notebook revisions."""
    return ops.diff_versions(_nb_path(path), from_version, to_version)


@tool()
def notebook_merge_preview(path: str, base_version: int, draft_document: dict,
                           choices: dict[str, str] | None = None,
                           include_document: bool = False) -> dict:
    """Preview cell-level three-way merge; conflicting cells need explicit choices."""
    result = ops.merge_preview(_nb_path(path), base_version, draft_document, choices)
    if not include_document:
        result.pop('merged_document', None)
    return result


@tool(permission='write')
def notebook_merge_commit(path: str, base_version: int, draft_document: dict,
                          expected_version: int, choices: dict[str, str] | None = None,
                          operation_id: str | None = None) -> dict:
    """Commit a conflict-resolved draft against an exact latest Notebook version."""
    return ops.merge_commit(_nb_path(path), base_version, draft_document,
                            expected_version=expected_version, choices=choices,
                            operation_id=operation_id)


@tool()
def folder_list() -> list[str]:
    """List actual nonhidden directories containing local Notebook documents."""
    return workspace.folder_list(ROOT)


@tool(permission='write')
def folder_ensure(name: str) -> dict:
    """Idempotently create a local Notebook folder under NB_ROOT."""
    return workspace.folder_ensure(ROOT, name)


@tool()
def cell_list(path: str) -> list[dict]:
    """List stable Cell IDs, types and short source summaries."""
    return core.describe(_nb_path(path))['cells']


@tool()
def cell_get(path: str, cell_id: str, include_outputs: bool = False) -> dict:
    """Read the full Cell source by stable ID or zero-based index."""
    return ops.cell_get(_nb_path(path), cell_id, include_outputs=include_outputs)


@tool(permission='write')
def cell_add(path: str, source: str, cell_type: str = 'code', at: int | None = None,
             after_cell_id: str | None = None, expected_version: int | None = None,
             operation_id: str | None = None) -> dict:
    """Insert a Cell with optional CAS version and idempotent operation identifier."""
    return core.add(_nb_path(path), cell_type, source, at=at, after=after_cell_id,
                    expected_version=expected_version, operation_id=operation_id)


@tool(permission='write')
def cell_insert(path: str, source: str, index: int, cell_type: str = 'code',
                expected_version: int | None = None, operation_id: str | None = None) -> dict:
    """Insert a Cell at a zero-based index (alias for cell_add)."""
    return core.add(_nb_path(path), cell_type, source, at=index,
                    expected_version=expected_version, operation_id=operation_id)


@tool(permission='write')
def cell_update(path: str, cell_id: str, source: str | None = None,
                cell_type: str | None = None, expected_version: int | None = None,
                operation_id: str | None = None) -> dict:
    """Replace Cell source/type and invalidate stale execution outputs."""
    return core.update(_nb_path(path), cell_id, source=source, cell_type=cell_type,
                       expected_version=expected_version, operation_id=operation_id)


@tool(permission='destructive')
def cell_delete(path: str, cell_id: str, expected_version: int | None = None,
                operation_id: str | None = None) -> dict:
    """Delete a selected Cell while retaining immutable previous Notebook version."""
    return core.delete(_nb_path(path), cell_id, expected_version=expected_version,
                       operation_id=operation_id)


@tool(permission='write')
def cell_move(path: str, cell_id: str, new_index: int,
              expected_version: int | None = None, operation_id: str | None = None) -> dict:
    """Move an existing Cell to a zero-based index."""
    return core.move(_nb_path(path), cell_id, new_index,
                     expected_version=expected_version, operation_id=operation_id)


@tool(permission='execute')
def cell_run(path: str, cell_id: str, timeout: int = 120,
             expected_version: int | None = None) -> dict:
    """Run one Cell with fresh kernel and prior code cells as bootstrap dependencies."""
    p = _nb_path(path)
    idx = core._index(core._read(p), cell_id)
    return ops.execute_cells(p, start=idx, count=1, bootstrap=True,
                             timeout=timeout, expected_version=expected_version)


@tool(permission='execute')
def cell_run_all(path: str, timeout: int = 120,
                 expected_version: int | None = None) -> dict:
    """Run every code Cell in a fresh Python Kernel and persist outputs on success."""
    return core.execute(_nb_path(path), timeout=timeout,
                        expected_version=expected_version)


@tool(permission='execute')
def notebook_run_cells(path: str, start: int = 0, count: int = 10,
                       bootstrap: bool = True, timeout: int = 120,
                       expected_version: int | None = None) -> dict:
    """Execute a bounded window with optionally executed prerequisite Cells."""
    if not 1 <= count <= 100:
        raise ValueError('count must be 1..100')
    return ops.execute_cells(_nb_path(path), start=start, count=count,
                             bootstrap=bootstrap, timeout=timeout,
                             expected_version=expected_version)


@tool(permission='execute')
def code_execute(path: str, code: str, timeout: int = 60,
                 persist: bool = False, expected_version: int | None = None,
                 use_live_kernel: bool = False) -> dict:
    """Execute trusted Python. Live mode shares memory; fresh mode can persist Cell."""
    p = _nb_path(path)
    if use_live_kernel:
        if persist:
            raise ValueError('Persisting live-kernel execution is not supported; use fresh mode')
        return runtime_manager.execute(p, code, timeout=timeout)
    return ops.code_execute(p, code, timeout=timeout, persist=persist,
                            expected_version=expected_version)


@tool(permission='execute')
def runtime_start(path: str) -> dict:
    """Start a local stateful Python Kernel for the lifetime of this MCP process."""
    p = _nb_path(path)
    core._read(p)
    return runtime_manager.start(p)


@tool()
def runtime_status(path: str) -> dict:
    """Get state of the local in-process MCP Python Kernel."""
    return runtime_manager.status(_nb_path(path))


@tool(permission='execute')
def runtime_stop(path: str) -> dict:
    """Stop the in-process Kernel and discard its in-memory Python variables."""
    return runtime_manager.stop(_nb_path(path))


@tool(permission='execute')
def kernel_variables(path: str, names: list[str] | None = None,
                     limit: int = 50) -> dict:
    """Read bounded variable metadata from an already-running Kernel."""
    return runtime_manager.variables(_nb_path(path), names=names, limit=limit)


@tool()
def runtime_diagnose() -> dict:
    """Inspect dependency and runtime capabilities without executing Notebook code."""
    return ops.execution_diagnostics()


@tool()
def workspace_list(path: str, recursive: bool = False) -> list[dict]:
    """List stored files in the isolated per-notebook logical workspace directory."""
    return workspace.list_files(ROOT, path, recursive=recursive)


@tool()
def workspace_read(path: str, relative: str, max_chars: int = 20000) -> dict:
    """Read bounded UTF-8 content of a per-notebook workspace file."""
    return workspace.read(ROOT, path, relative, max_chars=max_chars)


@tool(permission='write')
def workspace_write(path: str, relative: str, content: str, overwrite: bool = False) -> dict:
    """Write UTF-8 file within that Notebook's guarded workspace (2 MB cap)."""
    return workspace.write(ROOT, path, relative, content, overwrite=overwrite)


@tool(permission='destructive')
def workspace_delete(path: str, relative: str) -> dict:
    """Delete a per-notebook workspace file, not the Notebook document."""
    return workspace.delete(ROOT, path, relative)


@tool()
def notebook_validate(path: str) -> dict:
    """Validate nbformat schema and report document metadata."""
    return core.validate(_nb_path(path))


@tool(permission='write')
def notebook_render_html(path: str, destination_path: str) -> dict:
    """Render a Notebook as local standalone HTML. Outputs may contain untrusted HTML."""
    return core.export_html(_nb_path(path), dest(destination_path, '.html'))


@tool()
def notebook_audit(path: str) -> dict:
    """Report recent embedded idempotent operations and preserved version snapshots."""
    p = _nb_path(path)
    nb = core._read(p)
    receipts = nb.metadata.get(core.META_KEY, {}).get('operation_receipts', [])
    return {'version': core._version(nb), 'snapshot_count': len(core.history(p)['snapshots']),
            'recent_operations': [{'id': r['id'], 'version': r['description']['version']}
                                  for r in receipts[-30:]]}


@tool()
def notebook_fingerprint(path: str) -> dict:
    """Produce reproducibility hash of full Notebook bytes and each Cell source."""
    p = _nb_path(path)
    nb = core._read(p)
    return {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
            'version': core._version(nb),
            'cells': [{'id': c.id, 'source_sha256': hashlib.sha256(c.source.encode()).hexdigest()}
                      for c in nb.cells]}


@tool(permission='execute')
def run_submit(path: str, cell_id: str, idempotency_key: str,
               timeout: int = 120, expected_version: int | None = None) -> dict:
    """Submit one trusted Cell as an idempotent local subprocess job, not a cloud task."""
    _nb_path(path)
    return jobs.submit(ROOT, path, cell_id, idempotency_key=idempotency_key,
                       timeout=timeout, expected_version=expected_version)


@tool()
def run_status(run_id: str) -> dict:
    """Read a durable SQLite run record; dead processes become interrupted/cancelled."""
    return jobs.status(ROOT, run_id)


@tool()
def run_wait(run_id: str, wait_seconds: int = 10) -> dict:
    """Wait at most 30 seconds for a local job to reach a terminal state."""
    return jobs.wait(ROOT, run_id, wait_seconds=wait_seconds)


@tool(permission='execute')
def run_cancel(run_id: str) -> dict:
    """Signal a local job's isolated process group to cancel running code."""
    return jobs.cancel(ROOT, run_id)


# Local bounded experiment ledger. This is not an independent evaluation sandbox.
@tool()
def research_list(path: str) -> list[dict]:
    """List evidence-linked research experiments for one Notebook."""
    return research.list_experiments(ROOT, path)


@tool(permission='write')
def research_create(path: str, title: str, evaluator_cell_id: str,
                    metric_name: str, direction: str, dataset_sha256: str,
                    environment_sha256: str, budget: int = 10) -> dict:
    """Freeze evaluator Cell SHA and user-asserted dataset/environment SHA in a bounded protocol."""
    return research.create(ROOT, path, title=title, evaluator_cell_id=evaluator_cell_id,
                           metric_name=metric_name, direction=direction,
                           dataset_sha256=dataset_sha256,
                           environment_sha256=environment_sha256, budget=budget)


@tool()
def research_get(path: str, experiment_id: str) -> dict:
    """Read trial history, remaining budget and verified best metric."""
    return research.get(ROOT, path, experiment_id)


@tool(permission='write')
def research_trial_register(path: str, experiment_id: str, candidate_run_id: str) -> dict:
    """Register a Trial backed by a successful non-evaluator local candidate Run ID."""
    return research.register_trial(ROOT, path, experiment_id, candidate_run_id)


@tool(permission='write')
def research_trial_finalize(path: str, experiment_id: str, trial_id: str,
                            evaluator_run_id: str) -> dict:
    """Score an actual evaluator Run with frozen source and NB_METRIC_V1 JSON output."""
    return research.finalize_trial(ROOT, path, experiment_id, trial_id, evaluator_run_id)


@tool(permission='write')
def research_close(path: str, experiment_id: str) -> dict:
    """Close experiment enrollment while retaining its immutable trial records."""
    return research.close(ROOT, path, experiment_id)


@tool()
def research_artifacts(path: str, experiment_id: str) -> list[dict]:
    """List experiment-associated content-addressed SHA-256 artifacts."""
    return research.artifacts(ROOT, path, experiment_id)


@tool(permission='write')
def research_artifact_upload(path: str, experiment_id: str, workspace_path: str) -> dict:
    """Snapshot a per-notebook workspace file into immutable 2MB SHA-256 storage."""
    return research.artifact_upload(ROOT, path, experiment_id, workspace_path)


@tool(permission='write')
def research_artifact_download(path: str, experiment_id: str, sha256: str,
                               workspace_path: str, overwrite: bool = False) -> dict:
    """Verify and copy an experiment artifact back into a per-notebook workspace."""
    return research.artifact_download(ROOT, path, experiment_id, sha256, workspace_path,
                                      overwrite=overwrite)
