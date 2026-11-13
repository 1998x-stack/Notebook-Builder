"""Crash-aware, idempotent local Notebook execution jobs.

Jobs run in a separate trusted OS process and are recorded in SQLite. This is
not a remote queue, sandbox, durable cloud scheduler, or exactly-once executor.
Unknown post-crash execution outcomes are marked interrupted and never replayed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import notebook_builder as core
import notebook_ops as ops
import workspace_ops as workspace

STATES = ('queued', 'running', 'succeeded', 'failed', 'cancelling', 'cancelled', 'interrupted')
ACTIVE = ('queued', 'running', 'cancelling')
_ID = re.compile(r'^[a-zA-Z0-9._:-]{1,128}$')


def _dbfile(root):
    base = Path(root).resolve()
    folder = base / '.notebook-runs'
    folder.mkdir(parents=True,exist_ok=True,mode=0o700)
    if not folder.resolve().is_relative_to(base):
        raise ValueError('Run ledger symlink escapes notebook root')
    return folder / 'runs.sqlite3'


def _connect(root):
    db = sqlite3.connect(_dbfile(root), timeout=20)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA busy_timeout=20000')
    db.execute('''CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY, path TEXT NOT NULL, cell_id TEXT NOT NULL,
        idempotency_key TEXT NOT NULL, request_hash TEXT NOT NULL,
        expected_version INTEGER NOT NULL, timeout INTEGER NOT NULL,
        status TEXT NOT NULL, pid INTEGER, result TEXT, error TEXT,
        created_at REAL NOT NULL, updated_at REAL NOT NULL,
        UNIQUE(path, idempotency_key))''')
    db.commit()
    return db


def _pid_alive(pid, run_id):
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        cmd = Path(f'/proc/{pid}/cmdline')
        if cmd.exists():
            command = cmd.read_bytes().decode('utf-8','replace')
            if 'run_worker.py' not in command or run_id not in command:
                return False
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _refresh(db, row, *, commit: bool = True):
    stale_without_pid = (row['status'] == 'queued' and not row['pid']
                         and time.time() - row['updated_at'] > 10)
    dead_pid = row['status'] in ACTIVE and row['pid'] and not _pid_alive(row['pid'], row['id'])
    if stale_without_pid or dead_pid:
        status = 'cancelled' if row['status'] == 'cancelling' else 'interrupted'
        db.execute('UPDATE runs SET status=?,updated_at=? WHERE id=? AND status IN (?,?,?)',
                   (status,time.time(),row['id'],*ACTIVE))
        if commit:
            db.commit()
        return db.execute('SELECT * FROM runs WHERE id=?',(row['id'],)).fetchone()
    return row


def _as_dict(row):
    if row is None:
        raise FileNotFoundError('Run ID not found')
    return {'log_relative_path':f".notebook-runs/logs/{row['id']}.log",
            'run_id':row['id'],'path':row['path'],'cell_id':row['cell_id'],
            'status':row['status'],'expected_version':row['expected_version'],
            'created_at':row['created_at'],'updated_at':row['updated_at'],
            'result':json.loads(row['result']) if row['result'] else None,
            'error':row['error'][:1000] if row['error'] else None}


def submit(root, relative: str, cell_id: str, *, idempotency_key: str,
           timeout: int = 120, expected_version: int | None = None) -> dict:
    if not _ID.fullmatch(idempotency_key):
        raise ValueError('idempotency_key must be 1..128 safe characters')
    if not 1 <= timeout <= 600:
        raise ValueError('timeout must be 1..600')
    p = workspace.checked_notebook(root,relative)
    doc = core._read(p)
    idx = core._index(doc,cell_id)
    if doc.cells[idx].cell_type != 'code':
        raise ValueError('Only code Cells may be submitted')
    v = core._version(doc)
    req_version = v if expected_version is None else expected_version
    if req_version != v:
        raise core.ConflictError(f'Stale notebook: expected v{req_version}, current v{v}')
    req_hash = hashlib.sha256(json.dumps([relative,cell_id,timeout,req_version]).encode()).hexdigest()
    db = _connect(root)
    try:
        db.execute('BEGIN IMMEDIATE')
        old = db.execute('SELECT * FROM runs WHERE path=? AND idempotency_key=?',
                         (relative,idempotency_key)).fetchone()
        if old:
            if old['request_hash'] != req_hash:
                raise core.ConflictError('Run idempotency key reused with different inputs')
            db.commit()
            return {**_as_dict(_refresh(db,old)), 'replayed':True}
        active = db.execute('SELECT * FROM runs WHERE path=? AND status IN (?,?,?)',
                            (relative,*ACTIVE)).fetchall()
        for row in active:
            _refresh(db,row,commit=False)
        if db.execute('SELECT 1 FROM runs WHERE path=? AND status IN (?,?,?)',
                      (relative,*ACTIVE)).fetchone():
            raise core.ConflictError('A run is already active for this notebook')
        rid=str(uuid4())
        now=time.time()
        db.execute('INSERT INTO runs(id,path,cell_id,idempotency_key,request_hash,expected_version,timeout,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)',
                   (rid,relative,cell_id,idempotency_key,req_hash,req_version,timeout,'queued',now,now))
        db.commit()
        args=[sys.executable,str(Path(__file__).with_name('run_worker.py')),
              str(Path(root).resolve()),relative,cell_id,rid]
        try:
            log_dir = _dbfile(root).parent / 'logs'
            if log_dir.exists() and log_dir.is_symlink():
                raise ValueError('Run log directory must not be a symlink')
            log_dir.mkdir(mode=0o700, exist_ok=True)
            log_file = log_dir / f'{rid}.log'
            log_fd = os.open(log_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            with os.fdopen(log_fd, 'wb') as log:
                proc=subprocess.Popen(args,cwd=str(p.parent),stdin=subprocess.DEVNULL,
                                      stdout=log,stderr=subprocess.STDOUT,
                                      close_fds=True,start_new_session=True)
        except Exception as exc:
            db.execute('UPDATE runs SET status=?,error=?,updated_at=? WHERE id=?',
                       ('failed',str(exc)[:1000],time.time(),rid))
            db.commit()
            raise
        db.execute('UPDATE runs SET pid=?,updated_at=? WHERE id=?',(proc.pid,time.time(),rid))
        db.commit()
        return {**_as_dict(db.execute('SELECT * FROM runs WHERE id=?',(rid,)).fetchone()),'replayed':False}
    finally:
        db.close()


def status(root, run_id: str) -> dict:
    db=_connect(root)
    try:
        row=db.execute('SELECT * FROM runs WHERE id=?',(run_id,)).fetchone()
        if row is None:
            raise FileNotFoundError(run_id)
        return _as_dict(_refresh(db,row))
    finally:
        db.close()


def wait(root, run_id: str, *, wait_seconds: int = 10) -> dict:
    if not 0 <= wait_seconds <= 30:
        raise ValueError('wait_seconds must be 0..30')
    end=time.monotonic()+wait_seconds
    while True:
        result=status(root,run_id)
        if result['status'] not in ACTIVE or time.monotonic()>=end:
            return result
        time.sleep(0.2)


def cancel(root, run_id: str) -> dict:
    db=_connect(root)
    try:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT * FROM runs WHERE id=?',(run_id,)).fetchone()
        if row is None:
            raise FileNotFoundError(run_id)
        if row['status'] not in ACTIVE:
            db.commit()
            return _as_dict(row)
        db.execute('UPDATE runs SET status=?,updated_at=? WHERE id=?',
                   ('cancelling',time.time(),run_id))
        db.commit()
        pid=row['pid']
        try:
            if _pid_alive(pid,run_id) and os.getpgid(pid)==pid:
                os.killpg(pid,signal.SIGTERM)
        except ProcessLookupError:
            pass  # The worker may have just finished.
        return status(root,run_id)
    finally:
        db.close()


def finish(root, run_id: str, state: str, *, result=None, error=None):
    if state not in ('succeeded','failed'):
        raise ValueError('Invalid terminal status')
    db=_connect(root)
    try:
        previous=db.execute('SELECT status FROM runs WHERE id=?',(run_id,)).fetchone()
        if previous and previous['status'] == 'cancelling':
            state='cancelled'
        db.execute('UPDATE runs SET status=?,result=?,error=?,updated_at=? WHERE id=?',
                   (state,json.dumps(result,ensure_ascii=False,default=str) if result else None,
                    str(error)[:1000] if error else None,time.time(),run_id))
        db.commit()
    finally:
        db.close()


def run_worker(root, relative, cell_id, run_id):
    db=_connect(root)
    try:
        row=db.execute('SELECT * FROM runs WHERE id=?',(run_id,)).fetchone()
        if row is None or row['path']!=relative or row['cell_id']!=cell_id:
            return 2
        if row['status']=='cancelling':
            finish(root,run_id,'failed',error='cancelled before start')
            return 1
        db.execute('UPDATE runs SET status=?,updated_at=? WHERE id=? AND status=?',
                   ('running',time.time(),run_id,'queued'))
        db.commit()
        expected=row['expected_version'];timeout=row['timeout']
    finally:
        db.close()
    try:
        p=workspace.checked_notebook(root,relative)
        index=core._index(core._read(p),cell_id)
        r=ops.execute_cells(p,start=index,count=1,bootstrap=True,
                            timeout=timeout,expected_version=expected)
        finish(root,run_id,'succeeded',result={'version':r['version'],
                                               'executed_indices':r['executed_indices']})
        return 0
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        finish(root,run_id,'failed',error=f'{type(exc).__name__}: {exc}')
        return 1
