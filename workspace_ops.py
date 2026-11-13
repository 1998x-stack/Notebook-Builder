"""Least-privilege per-notebook workspace files for the local MCP adapter.

Protects against accidental path traversal and ordinary symlink escapes. This
is NOT a race-free or hostile-multi-user file sandbox; use a dedicated OS user.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import notebook_builder as core

MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_READ_CHARS = 20000


def checked_notebook(root: str | Path, relative: str) -> Path:
    base = Path(root).resolve()
    if not isinstance(relative, str) or not relative or '\x00' in relative:
        raise ValueError('non-empty relative notebook path required')
    target = Path(relative)
    if target.is_absolute() or '..' in target.parts or any(p.startswith('.') for p in target.parts):
        raise ValueError('Notebook path must be relative and may not contain dot-directories')
    p = (base / target).resolve()
    if p == base or not p.is_relative_to(base) or p.suffix.lower() != '.ipynb':
        raise ValueError('Notebook path escapes workspace or is not .ipynb')
    return p


def checked_workspace(root: str | Path, notebook: str, relative: str) -> Path:
    base = Path(root).resolve()
    notebook_path = checked_notebook(base, notebook)
    core._read(notebook_path)  # Workspace cannot be created for unknown Notebook.
    if not isinstance(relative, str) or not relative or '\x00' in relative:
        raise ValueError('non-empty relative workspace path required')
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts or any(p.startswith('.') for p in rel.parts):
        raise ValueError('Invalid workspace relative path')
    key = hashlib.sha256(str(notebook_path.relative_to(base)).encode()).hexdigest()[:20]
    folder = (base / '.notebook-workspaces' / key).resolve()
    if not folder.is_relative_to(base):
        raise ValueError('Workspace root escaped')
    target = (folder / rel).resolve()
    if target == folder or not target.is_relative_to(folder):
        raise ValueError('Workspace path escapes notebook workspace')
    return target


def write(root: str | Path, notebook: str, relative: str, text: str,
          *, overwrite: bool = False) -> dict:
    if not isinstance(text, str) or len(text.encode('utf-8')) > MAX_FILE_BYTES:
        raise ValueError('Workspace text exceeds configured size cap')
    p = checked_workspace(root, notebook, relative)
    with core._lock(p):
        if p.exists() and not overwrite:
            raise FileExistsError(p)
        core._atomic_bytes(p, text.encode('utf-8'))
    return {'path': relative, 'bytes': p.stat().st_size}


def read(root: str | Path, notebook: str, relative: str,
         *, max_chars: int = MAX_READ_CHARS) -> dict:
    if not 0 <= max_chars <= MAX_READ_CHARS:
        raise ValueError('max_chars outside allowed limit')
    p = checked_workspace(root, notebook, relative)
    if p.stat().st_size > MAX_FILE_BYTES:
        raise ValueError('Workspace file exceeds configured size cap')
    content = p.read_text(encoding='utf-8')
    return {'path': relative, 'text': content[:max_chars],
            'truncated': len(content) > max_chars, 'bytes': p.stat().st_size}


def list_files(root: str | Path, notebook: str, *, recursive: bool = False) -> list[dict]:
    anchor = checked_workspace(root, notebook, 'workspace-marker').parent
    if not anchor.exists():
        return []
    files = anchor.rglob('*') if recursive else anchor.glob('*')
    result = []
    for p in sorted(files):
        if p.name.startswith('.') or p.name.endswith('.lock') or not p.is_file():
            continue
        resolved = p.resolve()
        if not resolved.is_relative_to(anchor) or len(result) >= 1000:
            continue
        result.append({'path': str(p.relative_to(anchor)), 'bytes': p.stat().st_size})
    return result


def delete(root: str | Path, notebook: str, relative: str) -> dict:
    p = checked_workspace(root, notebook, relative)
    with core._lock(p):
        if not p.is_file():
            raise FileNotFoundError(p)
        p.unlink()
    return {'deleted': relative}


def folder_ensure(root: str | Path, relative: str) -> dict:
    base = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or '..' in rel.parts or any(p.startswith('.') for p in rel.parts):
        raise ValueError('Invalid folder name')
    target = checked_notebook(base, str(rel / 'placeholder.ipynb')).parent
    target.mkdir(parents=True, exist_ok=True)
    return {'folder': str(target.relative_to(base))}


def folder_list(root: str | Path) -> list[str]:
    base = Path(root).resolve()
    if not base.exists():
        return []
    return sorted(str(p.relative_to(base)) for p in base.rglob('*')
                  if p.is_dir() and not any(segment.startswith('.') for segment in p.relative_to(base).parts))
