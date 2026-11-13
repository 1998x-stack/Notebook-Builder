"""Local notebook authoring and execution, adapted from Nebula-Notebook concepts.

Single-host tool for *trusted* callers. No Docker isolation, auth, RBAC, or API.
Design ideas: nbformat validation, stable Cell IDs, expected-version CAS,
immutable previous-version snapshots, and separate executable kernel lifecycle.
Upstream: https://github.com/1998x-stack/Nebula-Notebook
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, TypeVar
import re

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

UPSTREAM_COMMIT = "e8e7e8e38afa5c414b70f3323fca98475287c2fb"
META_KEY = "notebook_builder"
T = TypeVar("T")
MAX_NOTEBOOK_BYTES = int(os.getenv("NB_MAX_NOTEBOOK_BYTES", 20 * 1024 * 1024))
MAX_CELL_SOURCE_BYTES = int(os.getenv("NB_MAX_CELL_SOURCE_BYTES", 2 * 1024 * 1024))
MAX_OPERATION_RECEIPTS = 128


class ConflictError(RuntimeError):
    """An expected notebook version did not match the persisted version."""


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _path(path: str | Path) -> Path:
    p = Path(path).expanduser().absolute()
    if p.suffix.lower() != ".ipynb":
        raise ValueError("Notebook file must end with .ipynb")
    return p


@contextmanager
def _lock(path: Path):
    """Linux advisory per-document lock, including create and execute."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = path.with_name("." + path.name + ".lock")
    fd = os.open(lock_file, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
    try:
        os.chmod(temp, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        # Flush containing directory to make the rename durable on local Linux filesystems.
        fd_dir = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd_dir)
        finally:
            os.close(fd_dir)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _serialized(nb: Any) -> bytes:
    nbformat.validate(nb)
    for cell in nb.cells:
        if len(cell.source.encode("utf-8")) > MAX_CELL_SOURCE_BYTES:
            raise ValueError("Cell source exceeds configured maximum")
    data = nbformat.writes(nb, version=4).encode("utf-8")
    if len(data) > MAX_NOTEBOOK_BYTES:
        raise ValueError("Notebook exceeds configured maximum size")
    return data


def _version(nb: Any) -> int:
    value = nb.metadata.get(META_KEY, {}).get("version", 0)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _set_version(nb: Any, version: int, action: str) -> None:
    previous = dict(nb.metadata.get(META_KEY, {}))
    previous.update({"version": version, "modified_at": timestamp(), "last_action": action,
                     "upstream_repository": "1998x-stack/Nebula-Notebook",
                     "upstream_commit": UPSTREAM_COMMIT})
    nb.metadata[META_KEY] = previous


def _read(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > MAX_NOTEBOOK_BYTES:
        raise ValueError("Notebook exceeds configured maximum size")
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    return nb


def _save(path: Path, nb: Any, *, old: Any = None) -> None:
    new_data = _serialized(nb)
    if old is not None:
        old_data = _serialized(old)
        original_version = _version(old)
        folder = path.parent / ".notebook-history" / path.stem
        if not folder.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError('History directory symlink escapes Notebook folder')
        stem = f"v{original_version:06d}-{hashlib.sha256(old_data).hexdigest()[:12]}"
        snapshot = folder / f"{stem}.ipynb"
        if not snapshot.exists():
            _atomic_bytes(snapshot, old_data)
    _atomic_bytes(path, new_data)


def new_cell(cell_type: str, source: str):
    if cell_type == "code":
        return nbformat.v4.new_code_cell(source=source)
    if cell_type == "markdown":
        return nbformat.v4.new_markdown_cell(source=source)
    if cell_type == "raw":
        return nbformat.v4.new_raw_cell(source=source)
    raise ValueError("Cell type must be code, markdown or raw")


def create(path: str | Path, *, title: str | None = None, cells: list[dict] | None = None,
           overwrite: bool = False, spec_sha256: str | None = None) -> dict:
    p = _path(path)
    with _lock(p):
        if p.exists() and not overwrite:
            raise FileExistsError(f"Notebook already exists: {p} (use --overwrite explicitly)")
        previous = _read(p) if p.exists() else None
        entries = []
        if title:
            entries.append(nbformat.v4.new_markdown_cell(f"# {title}"))
        for entry in cells or []:
            if not isinstance(entry, dict):
                raise ValueError("Each cell in the spec must be an object")
            source = entry.get("source", "")
            if not isinstance(source, str):
                raise ValueError("Cell source must be a string")
            entries.append(new_cell(entry.get("cell_type", "code"), source))
        nb = nbformat.v4.new_notebook(cells=entries,
                 metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                           "language_info": {"name": "python"}})
        _set_version(nb, _version(previous) + 1 if previous is not None else 1, "create")
        if title:
            nb.metadata[META_KEY]["title"] = title
        nb.metadata[META_KEY]["created_at"] = timestamp()
        if spec_sha256 is not None:
            if not re.fullmatch(r"[a-f0-9]{64}", spec_sha256):
                raise ValueError('spec_sha256 must be a lowercase SHA-256 hex digest')
            nb.metadata[META_KEY]['spec_sha256'] = spec_sha256
        _save(p, nb, old=previous)
    return describe(p)


def from_spec(path: str | Path, spec: str | Path, *, overwrite: bool = False):
    data = json.loads(Path(spec).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cells", []), list):
        raise ValueError("Spec requires an object containing a cells array")
    fingerprint = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                           separators=(',', ':')).encode('utf-8')).hexdigest()
    return create(path, title=data.get("title"), cells=data.get("cells", []),
                  overwrite=overwrite, spec_sha256=fingerprint)


def _modify(path: str | Path, action: str, callback: Callable[[Any], T],
            *, expected_version: int | None = None, operation_id: str | None = None,
            request: dict | None = None) -> tuple[dict, T]:
    p = _path(path)
    with _lock(p):
        old = _read(p)
        actual = _version(old)
        fingerprint = None
        if operation_id is not None:
            if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", operation_id):
                raise ValueError("operation_id must be 1-128 safe characters")
            fingerprint = hashlib.sha256(json.dumps(
                {"action": action, "request": request, "expected_version": expected_version},
                sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
            for receipt in old.metadata.get(META_KEY, {}).get("operation_receipts", []):
                if receipt.get("id") == operation_id:
                    if receipt.get("hash") != fingerprint:
                        raise ConflictError("operation_id already used with different arguments")
                    return {**receipt["description"], "replayed": True}, receipt["result"]
        if expected_version is not None and actual != expected_version:
            raise ConflictError(f"Stale notebook: expected v{expected_version}, current v{actual}")
        # Independent in-memory copy so immutable history captures the original.
        nb = copy.deepcopy(old)
        result = callback(nb)
        _set_version(nb, actual + 1, action)
        if operation_id is not None:
            receipts = list(nb.metadata[META_KEY].get("operation_receipts", []))
            # Store the replay receipt inside the same atomic Notebook commit.
            desc = _describe_node(p, nb)
            receipts.append({"id": operation_id, "hash": fingerprint, "result": result,
                             "description": desc})
            nb.metadata[META_KEY]["operation_receipts"] = receipts[-MAX_OPERATION_RECEIPTS:]
        _save(p, nb, old=old)
    return describe(p), result


def _index(nb: Any, selector: str | int) -> int:
    # nbformat v4.5 IDs may consist entirely of digits; a string matching a
    # stable Cell ID MUST take priority over a numeric index interpretation.
    if isinstance(selector, str):
        match = next((i for i, c in enumerate(nb.cells) if c.id == selector), None)
        if match is not None:
            return match
    if isinstance(selector, int) or (isinstance(selector, str) and selector.isdecimal()):
        i = int(selector)
    else:
        i = -1
    if not 0 <= i < len(nb.cells):
        raise IndexError(f"No cell matches {selector!r}")
    return i


def add(path: str | Path, cell_type: str, source: str, *, at: int | None = None,
        after: str | None = None, expected_version: int | None = None,
        operation_id: str | None = None) -> dict:
    if at is not None and after is not None:
        raise ValueError("Use only one of at or after")
    def op(nb):
        index = (_index(nb, after) + 1) if after is not None else (len(nb.cells) if at is None else at)
        if not 0 <= index <= len(nb.cells):
            raise IndexError(f"Insert index out of bounds: {index}")
        cell = new_cell(cell_type, source)
        nb.cells.insert(index, cell)
        return {"cell_id": cell.id, "index": index}
    desc, result = _modify(path, "add", op, expected_version=expected_version,
                           operation_id=operation_id,
                           request={"type": cell_type, "source": source, "at": at, "after": after})
    return {**desc, **result}


def update(path: str | Path, selector: str | int, *, source: str | None = None,
           cell_type: str | None = None, expected_version: int | None = None,
           operation_id: str | None = None) -> dict:
    if source is None and cell_type is None:
        raise ValueError("Provide source and/or cell_type")
    def op(nb):
        i = _index(nb, selector)
        old = nb.cells[i]
        replacement = new_cell(cell_type or old.cell_type, old.source if source is None else source)
        replacement.id = old.id
        replacement.metadata = old.metadata
        nb.cells[i] = replacement
        return {"cell_id": replacement.id, "index": i}
    desc, result = _modify(path, "update", op, expected_version=expected_version,
                           operation_id=operation_id,
                           request={"selector": selector, "source": source, "type": cell_type})
    return {**desc, **result}


def delete(path: str | Path, selector: str | int, *, expected_version: int | None = None,
           operation_id: str | None = None) -> dict:
    def op(nb):
        i = _index(nb, selector)
        return {"deleted_cell_id": nb.cells.pop(i).id, "index": i}
    desc, result = _modify(path, "delete", op, expected_version=expected_version,
                           operation_id=operation_id, request={"selector": selector})
    return {**desc, **result}


def move(path: str | Path, selector: str | int, new_index: int,
         *, expected_version: int | None = None, operation_id: str | None = None) -> dict:
    def op(nb):
        i = _index(nb, selector)
        if not 0 <= new_index < len(nb.cells):
            raise IndexError(f"Move index out of bounds: {new_index}")
        cell = nb.cells.pop(i)
        nb.cells.insert(new_index, cell)
        return {"cell_id": cell.id, "from": i, "to": new_index}
    desc, result = _modify(path, "move", op, expected_version=expected_version,
                           operation_id=operation_id,
                           request={"selector": selector, "new_index": new_index})
    return {**desc, **result}


def _describe_node(p: Path, nb: Any) -> dict:
    return {"path": str(p), "version": _version(nb), "valid": True,
            "cell_count": len(nb.cells), "cells": [
                {"index": i, "id": cell.id, "type": cell.cell_type,
                 "first_line": (cell.source.splitlines() or [""])[0][:120],
                 "outputs": len(cell.get("outputs", []))}
                for i, cell in enumerate(nb.cells)]}


def describe(path: str | Path) -> dict:
    p = _path(path)
    return _describe_node(p, _read(p))


def validate(path: str | Path) -> dict:
    return describe(path)


def execute(path: str | Path, *, timeout: int = 120, kernel: str = "python3",
            cwd: str | Path | None = None, expected_version: int | None = None) -> dict:
    """Execute an immutable snapshot without holding the document write lock.

    The output commit is optimistic CAS on BOTH document version and source bytes,
    so a manual edit that forgot to bump the version cannot be lost. A conflict
    discards the uncommitted execution output; Python side effects cannot be
    rolled back. Execute only trusted notebook code.
    """
    if timeout <= 0:
        raise ValueError("timeout must be a positive number of seconds")
    p = _path(path)
    working = Path(cwd).expanduser().resolve() if cwd else p.parent.resolve()
    if not working.is_dir():
        raise NotADirectoryError(working)
    # Hold the lock only long enough to establish the consistent source snapshot.
    with _lock(p):
        old = _read(p)
        actual = _version(old)
        if expected_version is not None and expected_version != actual:
            raise ConflictError(f"Stale notebook: expected v{expected_version}, current v{actual}")
        old_digest = hashlib.sha256(_serialized(old)).hexdigest()
    nb = copy.deepcopy(old)
    client = NotebookClient(nb, timeout=timeout, kernel_name=kernel,
                            resources={"metadata": {"path": str(working)}},
                            allow_errors=False)
    # This executes arbitrary code without sandboxing. The lock is released here.
    client.execute()
    with _lock(p):
        latest = _read(p)
        if (_version(latest) != actual or
                hashlib.sha256(_serialized(latest)).hexdigest() != old_digest):
            raise ConflictError("Notebook changed while executing; output was not committed")
        _set_version(nb, actual + 1, "execute")
        nb.metadata[META_KEY]["executed_at"] = timestamp()
        _save(p, nb, old=latest)
    return describe(p)


def export_html(path: str | Path, output: str | Path) -> dict:
    nb = _read(_path(path))
    html, _ = HTMLExporter().from_notebook_node(nb)
    out = Path(output).expanduser().absolute()
    if out.suffix.lower() != ".html":
        raise ValueError("Output must have an .html extension")
    _atomic_bytes(out, html.encode("utf-8"))
    return {"html": str(out), "bytes": out.stat().st_size}


def history(path: str | Path) -> dict:
    p = _path(path)
    hist = p.parent / ".notebook-history" / p.stem
    return {"path": str(p), "snapshots": [str(f) for f in sorted(hist.glob("*.ipynb"))] if hist.exists() else []}
