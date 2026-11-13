"""Advanced local notebook operations inspired by Nebula Notebook's MCP interfaces.

No network services, no multi-user authentication, and no sandbox are supplied.
All writes share notebook_builder's file lock and atomic commit conventions.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbconvert import MarkdownExporter

import notebook_builder as core


def list_notebooks(root: str | Path, *, recursive: bool = True) -> list[dict]:
    base = Path(root).resolve()
    if not base.is_dir():
        return []
    entries = base.rglob('*.ipynb') if recursive else base.glob('*.ipynb')
    results = []
    for p in sorted(entries):
        if any(part.startswith('.') for part in p.relative_to(base).parts) or not p.is_file():
            continue
        # Skip symlink targets outside the scope and overlarge files.
        if not p.resolve().is_relative_to(base) or p.stat().st_size > core.MAX_NOTEBOOK_BYTES:
            continue
        try:
            nb = core._read(p)
            results.append({'path': str(p.relative_to(base)), 'version': core._version(nb),
                            'cells': len(nb.cells), 'bytes': p.stat().st_size,
                            'title': nb.metadata.get(core.META_KEY, {}).get('title', p.stem)})
        except (ValueError, OSError, nbformat.ValidationError):
            continue
    return results


def notebook_get(path: str | Path, *, include_outputs: bool = False,
                 max_output_chars: int = 10000) -> dict:
    doc = core._read(core._path(path))
    value = core.describe(path)
    value['metadata'] = dict(doc.metadata)
    value['cells'] = [cell_get(path, i, include_outputs=include_outputs,
                               max_output_chars=max_output_chars, doc=doc)
                      for i in range(len(doc.cells))]
    return value


def cell_get(path: str | Path, selector: str | int, *, include_outputs: bool = False,
             max_output_chars: int = 10000, doc: Any = None) -> dict:
    if not 0 <= max_output_chars <= 100000:
        raise ValueError('max_output_chars must be 0..100000')
    nb = doc if doc is not None else core._read(core._path(path))
    i = core._index(nb, selector)
    c = nb.cells[i]
    item = {'id': c.id, 'index': i, 'cell_type': c.cell_type,
            'source': c.source, 'execution_count': c.get('execution_count'),
            'output_count': len(c.get('outputs', []))}
    if include_outputs:
        bounded = []
        for out in c.get('outputs', [])[:20]:
            v = dict(out)
            if 'text' in v:
                v['text'] = ''.join(v['text'])[:max_output_chars]
            if 'data' in v:
                # Images and other rich output can be huge; only previews and sizes.
                v['data'] = {mime: (str(payload)[:max_output_chars] if mime.startswith('text/')
                                    else f'<{mime} content omitted; {len(str(payload))} characters>')
                             for mime, payload in v['data'].items()}
            bounded.append(v)
        item['outputs'] = bounded
    return item


def title(path: str | Path, new_title: str, *, expected_version: int | None = None) -> dict:
    if not new_title.strip() or len(new_title) > 256:
        raise ValueError('title must be 1..256 non-blank characters')
    def op(nb):
        nb.metadata.setdefault(core.META_KEY, {})['title'] = new_title.strip()
        return {'title': new_title.strip()}
    desc, r = core._modify(path, 'rename-title', op, expected_version=expected_version)
    return {**desc, **r}


def import_notebook(source: str | Path, destination: str | Path, *, overwrite: bool = False) -> dict:
    src = core._path(source)
    nb = core._read(src)
    dst = core._path(destination)
    with core._lock(dst):
        if dst.exists() and not overwrite:
            raise FileExistsError(dst)
        previous = core._read(dst) if dst.exists() else None
        nb = copy.deepcopy(nb)
        nb.metadata.pop(core.META_KEY, None)  # Never trust third-party operation receipts.
        core._set_version(nb, (core._version(previous)+1 if previous is not None else 1), 'import')
        nb.metadata[core.META_KEY]['created_at'] = core.timestamp()
        core._save(dst, nb, old=previous)
    return core.describe(dst)


def export_ipynb(path: str | Path, destination: str | Path, *, overwrite: bool = False) -> dict:
    src = core._path(path)
    dst = core._path(destination)
    if src.resolve() == dst.resolve():
        raise ValueError('export destination must differ from source')
    nb = core._read(src)
    with core._lock(dst):
        if dst.exists() and not overwrite:
            raise FileExistsError(dst)
        core._atomic_bytes(dst, core._serialized(nb))
    return {'source': str(src), 'exported': str(dst), 'bytes': dst.stat().st_size}


def export_markdown(path: str | Path, destination: str | Path) -> dict:
    doc = core._read(core._path(path))
    body, resources = MarkdownExporter().from_notebook_node(doc)
    # Refuse silently discarding extracted image assets; materialize under _files.
    dest = Path(destination).expanduser().absolute()
    if dest.suffix.lower() != '.md':
        raise ValueError('output must end in .md')
    outputs = resources.get('outputs', {})
    if outputs:
        folder = dest.with_name(dest.stem + '_files')
        for name, data in outputs.items():
            fname = Path(name).name
            if not fname or fname != name:
                raise ValueError('Unsafe exported asset filename')
            core._atomic_bytes(folder / fname, data if isinstance(data, bytes) else data.encode('utf-8'))
    if outputs:
        for name in outputs:
            # MarkdownExporter emits links relative to the .md, move assets into
            # a sibling directory and rewrite those links accordingly.
            body = body.replace(f']({name})', f']({folder.name}/{name})')
    core._atomic_bytes(dest, body.encode('utf-8'))
    return {'markdown': str(dest), 'assets': len(outputs), 'bytes': dest.stat().st_size}


def _find_snapshot(path: Path, version: int) -> Path | None:
    folder = path.parent / '.notebook-history' / path.stem
    matches = sorted(folder.glob(f'v{version:06d}-*.ipynb')) if folder.exists() else []
    return matches[0] if matches else None


def get_version(path: str | Path, version: int, *, include_outputs: bool = False) -> dict:
    p = core._path(path)
    doc = core._read(p)
    if version == core._version(doc):
        selected = p
    else:
        selected = _find_snapshot(p, version)
        if selected is None:
            raise FileNotFoundError(f'Snapshot v{version} not found')
    r = notebook_get(selected, include_outputs=include_outputs)
    r['source_notebook'] = str(p)
    r['snapshot'] = str(selected)
    return r


def fork_version(path: str | Path, version: int, destination: str | Path) -> dict:
    p = core._path(path)
    selected = p if core._version(core._read(p)) == version else _find_snapshot(p, version)
    if selected is None:
        raise FileNotFoundError(f'Snapshot v{version} not found')
    result = import_notebook(selected, destination)
    # Detached document, no kernel state, workspace, or inherited history.
    return {**result, 'forked_from': str(p), 'source_version': version}


def restore_version(path: str | Path, version: int, *, expected_version: int | None = None) -> dict:
    p = core._path(path)
    def op(nb):
        selected = _find_snapshot(p, version)
        if selected is None:
            raise FileNotFoundError(f'Snapshot v{version} not found')
        prev = core._read(selected)
        nb.cells = copy.deepcopy(prev.cells)
        for key in ('kernelspec', 'language_info'):
            if key in prev.metadata:
                nb.metadata[key] = copy.deepcopy(prev.metadata[key])
        return {'restored_from_version': version}
    desc, r = core._modify(p, 'restore', op, expected_version=expected_version)
    return {**desc, **r}


def diff_versions(path: str | Path, from_version: int, to_version: int | None = None) -> dict:
    p = core._path(path)
    current = core._version(core._read(p))
    target = to_version if to_version is not None else current
    a = get_version(p, from_version)
    b = get_version(p, target)
    cells_a = {c['id']: c for c in a['cells']}
    cells_b = {c['id']: c for c in b['cells']}
    changes = []
    for id in cells_a.keys() | cells_b.keys():
        ca, cb = cells_a.get(id), cells_b.get(id)
        if ca is None: kind = 'added'
        elif cb is None: kind = 'deleted'
        elif (ca['source'], ca['cell_type']) != (cb['source'], cb['cell_type']): kind = 'modified'
        elif ca['index'] != cb['index']: kind = 'moved'
        else: continue
        changes.append({'cell_id': id, 'change': kind,
                        'from_index': ca['index'] if ca else None,
                        'to_index': cb['index'] if cb else None,
                        'source_diff': ''.join(difflib.unified_diff(
                            ca['source'].splitlines(True) if ca else [],
                            cb['source'].splitlines(True) if cb else [],
                            fromfile=f'v{from_version}', tofile=f'v{target}'))[:16000]})
    return {'path': str(p), 'from_version': from_version, 'to_version': target,
            'changes': sorted(changes, key=lambda x: (x['to_index'] if x['to_index'] is not None else 10**9,
                                                       x['from_index'] if x['from_index'] is not None else 10**9))}


def execute_cells(path: str | Path, *, start: int = 0, count: int | None = None,
                  bootstrap: bool = True, timeout: int = 120,
                  expected_version: int | None = None, cwd: str | Path | None = None) -> dict:
    """Execute a cell window on a snapshot and commit with optimistic CAS.

    No lock is held while Python runs, so other edits remain possible. A stale
    output commit fails instead of overwriting a concurrent edit. Side effects
    of notebook code cannot be rolled back on conflict.
    """
    if timeout < 1:
        raise ValueError('timeout must be positive')
    p = core._path(path)
    with core._lock(p):
        original = core._read(p)
        old_v = core._version(original)
        if expected_version is not None and old_v != expected_version:
            raise core.ConflictError(f'Stale notebook: expected v{expected_version}, current v{old_v}')
        total = len(original.cells)
        if not 0 <= start <= total or (count is not None and count < 0):
            raise ValueError('Invalid cell window')
        end = total if count is None else min(start + count, total)
        target = [i for i in range(start, end) if original.cells[i].cell_type == 'code']
        old_hash = hashlib.sha256(core._serialized(original)).hexdigest()
    if not target:
        return {**core.describe(p), 'executed_indices': [], 'no_change': True, 'next_start': end}
    work = Path(cwd).expanduser().resolve() if cwd else p.parent.resolve()
    if not work.is_dir():
        raise NotADirectoryError(work)
    working = copy.deepcopy(original)
    client = NotebookClient(working, kernel_name='python3', timeout=timeout,
                            allow_errors=False, resources={'metadata': {'path': str(work)}})
    with client.setup_kernel():
        indices = list(range(start)) + target if bootstrap else target
        for i in indices:
            if working.cells[i].cell_type == 'code':
                client.execute_cell(working.cells[i], i)
    if bootstrap:
        for i in range(start):
            working.cells[i] = copy.deepcopy(original.cells[i])
    with core._lock(p):
        latest = core._read(p)
        if core._version(latest) != old_v or hashlib.sha256(core._serialized(latest)).hexdigest() != old_hash:
            raise core.ConflictError('Notebook changed while executing; output was not committed')
        core._set_version(working, old_v + 1, 'execute-cells')
        working.metadata[core.META_KEY]['executed_at'] = core.timestamp()
        core._save(p, working, old=latest)
    return {**core.describe(p), 'executed_indices': target, 'next_start': end}


def code_execute(path: str | Path, source: str, *, timeout: int = 60,
                 persist: bool = False, expected_version: int | None = None) -> dict:
    """Run a snippet in a fresh trusted kernel; optionally commit a new cell."""
    if not isinstance(source, str) or timeout < 1:
        raise ValueError('Code must be text and timeout must be positive')
    p = core._path(path)
    with core._lock(p):
        original = core._read(p)
        v = core._version(original)
        if expected_version is not None and v != expected_version:
            raise core.ConflictError(f'Stale notebook: expected v{expected_version}, current v{v}')
        old_hash = hashlib.sha256(core._serialized(original)).hexdigest()
    cell = core.new_cell('code', source)
    temp = nbformat.v4.new_notebook(cells=[cell], metadata={'kernelspec': {
        'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
    core._serialized(temp)
    client = NotebookClient(temp, kernel_name='python3', timeout=timeout,
                            allow_errors=False, resources={'metadata': {'path': str(p.parent.resolve())}})
    client.execute()
    result = {'status': 'ok', 'cell_id': cell.id, 'outputs': _output_preview(cell.outputs),
              'persisted': False, 'version': v}
    if persist:
        with core._lock(p):
            latest = core._read(p)
            if core._version(latest) != v or hashlib.sha256(core._serialized(latest)).hexdigest() != old_hash:
                raise core.ConflictError('Notebook changed while executing; cell was not committed')
            updated = copy.deepcopy(latest)
            updated.cells.append(cell)
            core._set_version(updated, v + 1, 'code-execute')
            core._save(p, updated, old=latest)
            result.update(persisted=True, version=v + 1)
    return result


def _output_preview(outputs, *, limit=16000) -> list[dict]:
    result = []
    for item in outputs[:20]:
        r = {'output_type': item.output_type}
        if 'text' in item:
            r['text'] = str(item.text)[:limit]
        if 'data' in item:
            r['data'] = {key: (str(value)[:limit] if key.startswith('text/')
                               else f'<omitted {len(str(value))} characters>')
                         for key, value in item.data.items()}
        if 'ename' in item:
            r['ename'] = item.ename
            r['evalue'] = item.evalue
        result.append(r)
    return result


def execution_diagnostics() -> dict:
    import importlib.util
    import platform
    import sys
    packages = ('nbformat', 'nbclient', 'nbconvert', 'ipykernel', 'jupyter_client',
                'matplotlib', 'pandas', 'numpy', 'mcp', 'docker')
    return {'python': sys.version.split()[0], 'platform': platform.platform(),
            'modules': {p: bool(importlib.util.find_spec(p)) for p in packages},
            'kernel': 'fresh Python3 ipykernel (not an isolation boundary)',
            'stateful_kernel_in_mcp_process': True,
            'kernel_persistent_across_process_restarts': False,
            'docker_isolation': False, 'gpu_managed': False}


def _validated_draft(document: dict) -> Any:
    if not isinstance(document, dict) or not isinstance(document.get('cells'), list):
        raise ValueError('draft_document must be a full nbformat v4 notebook JSON object')
    nb = nbformat.from_dict(copy.deepcopy(document))
    core._serialized(nb)  # checks schema and configured size limits
    ids = [cell.id for cell in nb.cells]
    if len(ids) != len(set(ids)):
        raise ValueError('draft contains duplicate cell IDs')
    return nb


def _merge_nodes(base: Any, mine: Any, latest: Any,
                 choices: dict[str, str] | None = None) -> tuple[Any, list[dict]]:
    """Conservative cell-atomic three-way merge; no text auto-merging.

    Returns conflicts as explicit cell IDs / __order__. Unresolved conflicts
    leave the merged document incomplete and must not be committed.
    """
    choices = choices or {}
    maps = [{c.id: c for c in doc.cells} for doc in (base, mine, latest)]
    b, m, l = maps
    union = list(dict.fromkeys([*b, *m, *l]))
    sentinel = object()
    def canonical(cell):
        if cell is sentinel:
            return None
        return json.dumps(dict(cell), sort_keys=True, ensure_ascii=False, default=str)

    merged = copy.deepcopy(latest)
    decided = {}
    conflicts = []
    for cid in union:
        old, local, remote = (d.get(cid, sentinel) for d in maps)
        co, cm, cr = canonical(old), canonical(local), canonical(remote)
        mine_changed, latest_changed = cm != co, cr != co
        if mine_changed and latest_changed and cm != cr:
            choice = choices.get(cid)
            if choice not in ('mine', 'theirs'):
                conflicts.append({'cell_id': cid, 'type': 'cell',
                                  'mine': 'deleted' if local is sentinel else 'present',
                                  'theirs': 'deleted' if remote is sentinel else 'present'})
                continue
            chosen = local if choice == 'mine' else remote
        else:
            chosen = local if mine_changed else remote
        if chosen is not sentinel:
            decided[cid] = copy.deepcopy(chosen)

    base_ids = list(b)
    mine_base = [cid for cid in m if cid in b and cid in l]
    latest_base = [cid for cid in l if cid in b and cid in m]
    common_base = [cid for cid in base_ids if cid in m and cid in l]
    mine_reordered = mine_base != common_base
    latest_reordered = latest_base != common_base
    mine_order = [cid for cid in m if cid in decided]
    latest_order = [cid for cid in l if cid in decided]
    if mine_reordered and latest_reordered and mine_base != latest_base:
        selection = choices.get('__order__')
        if selection not in ('mine', 'theirs'):
            conflicts.append({'cell_id': '__order__', 'type': 'order'})
            selection = 'theirs'
    else:
        selection = 'mine' if mine_reordered and not latest_reordered else 'theirs'
    preferred = mine_order if selection == 'mine' else latest_order
    alternative = latest_order if selection == 'mine' else mine_order
    final_ids = list(dict.fromkeys([*preferred, *alternative, *decided.keys()]))
    merged.cells = [decided[cid] for cid in final_ids if cid in decided]
    if not conflicts:
        core._serialized(merged)
    return merged, conflicts


def merge_preview(path: str | Path, base_version: int, draft_document: dict,
                  choices: dict[str, str] | None = None) -> dict:
    p = core._path(path)
    current = core._read(p)
    mine = _validated_draft(draft_document)
    base_ref = _find_snapshot(p, base_version)
    if base_ref is None:
        if core._version(current) != base_version:
            raise FileNotFoundError(f'Base snapshot v{base_version} not found')
        base_ref = p
    base = core._read(base_ref)
    merged, conflicts = _merge_nodes(base, mine, current, choices)
    return {'path': str(p), 'base_version': base_version,
            'current_version': core._version(current), 'conflicts': conflicts,
            'can_commit': not conflicts,
            'merged_document': dict(merged) if not conflicts else None}


def merge_commit(path: str | Path, base_version: int, draft_document: dict,
                 *, expected_version: int, choices: dict[str, str] | None = None,
                 operation_id: str | None = None) -> dict:
    p = core._path(path)
    mine = _validated_draft(draft_document)
    def op(latest):
        base_ref = _find_snapshot(p, base_version)
        if base_ref is None:
            if core._version(latest) != base_version:
                raise FileNotFoundError(f'Base snapshot v{base_version} not found')
            base_ref = p
        base = core._read(base_ref)
        merged, conflicts = _merge_nodes(base, mine, latest, choices)
        if conflicts:
            raise core.ConflictError(f'Unresolved merge conflicts: {conflicts}')
        latest.cells = merged.cells
        return {'merged_cells': len(merged.cells), 'base_version': base_version}
    desc, r = core._modify(p, 'merge', op, expected_version=expected_version,
                           operation_id=operation_id,
                           request={'base_version': base_version, 'draft': draft_document, 'choices': choices})
    return {**desc, **r}


def delete_notebook(path: str | Path, *, expected_version: int | None = None) -> dict:
    """Soft-delete to a local hidden trash directory; preserves history separately."""
    p = core._path(path)
    with core._lock(p):
        nb = core._read(p)
        version = core._version(nb)
        if expected_version is not None and version != expected_version:
            raise core.ConflictError(f'Stale notebook: expected v{expected_version}, current v{version}')
        trash = p.parent / '.notebook-trash'
        trash.mkdir(exist_ok=True)
        suffix = hashlib.sha256((str(p) + core.timestamp()).encode()).hexdigest()[:12]
        target = trash / f'{p.stem}-v{version}-{suffix}.ipynb'
        p.replace(target)
    return {'deleted': str(p), 'recoverable_at': str(target), 'version': version}
