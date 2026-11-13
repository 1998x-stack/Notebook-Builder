"""Evidence-linked local research experiments for Notebook Builder.

Candidate/evaluator results must originate from successful local Run IDs.
The evaluator Cell source is frozen by SHA-256, and a metric is parsed from
its saved Notebook output. Dataset/environment hashes are caller assertions;
evaluation is NOT hermetic, adversary-resistant, or independently attested.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from uuid import uuid4

import notebook_builder as core
import notebook_ops as ops
import run_jobs as jobs
import workspace_ops as workspace

_SHA = re.compile(r'^[0-9a-fA-F]{64}$')
MARKER = 'NB_METRIC_V1:'


def _connect(root):
    base=Path(root).resolve()
    directory=base/'.notebook-research'
    directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    if not directory.resolve().is_relative_to(base):
        raise ValueError('Research ledger escapes NB_ROOT')
    db=sqlite3.connect(directory/'research.sqlite3',timeout=20)
    db.row_factory=sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('''CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY, path TEXT NOT NULL, title TEXT NOT NULL,
        evaluator_cell_id TEXT NOT NULL, evaluator_source_sha256 TEXT NOT NULL,
        metric_name TEXT NOT NULL, direction TEXT NOT NULL,
        dataset_sha256 TEXT NOT NULL, environment_sha256 TEXT NOT NULL,
        budget INTEGER NOT NULL, state TEXT NOT NULL, created_version INTEGER NOT NULL,
        created_at REAL NOT NULL)''')
    db.execute('''CREATE TABLE IF NOT EXISTS trials (
        id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL,
        candidate_run_id TEXT NOT NULL, evaluator_run_id TEXT,
        candidate_version INTEGER NOT NULL, evaluator_version INTEGER,
        score REAL, status TEXT NOT NULL, created_at REAL NOT NULL,
        UNIQUE(experiment_id,evaluator_run_id),
        UNIQUE(experiment_id,candidate_run_id))''')
    db.execute('''CREATE TABLE IF NOT EXISTS artifacts (
        experiment_id TEXT NOT NULL, sha256 TEXT NOT NULL,
        origin TEXT NOT NULL, size INTEGER NOT NULL, created_at REAL NOT NULL,
        PRIMARY KEY(experiment_id, sha256))''')
    db.commit()
    return db


def _experiment(db, relative, exp_id):
    row=db.execute('SELECT * FROM experiments WHERE id=? AND path=?',(exp_id,relative)).fetchone()
    if row is None:
        raise FileNotFoundError('Experiment not found in the requested Notebook')
    return row


def _require_hash(value,label):
    if not isinstance(value,str) or not _SHA.fullmatch(value):
        raise ValueError(f'{label} must be a SHA-256 hex digest (assertion by caller)')


def create(root, relative, *, title, evaluator_cell_id, metric_name,
           direction, dataset_sha256, environment_sha256, budget=10):
    p=workspace.checked_notebook(root,relative)
    nb=core._read(p)
    evaluator=nb.cells[core._index(nb,evaluator_cell_id)]
    if evaluator.cell_type!='code':
        raise ValueError('Evaluator must be a code Cell')
    if not title.strip() or len(title)>200 or not metric_name.strip() or len(metric_name)>80:
        raise ValueError('Invalid experiment title or metric name')
    if direction not in ('maximize','minimize'):
        raise ValueError('direction must be maximize or minimize')
    if not 1 <= budget <= 100:
        raise ValueError('budget must be 1..100')
    _require_hash(dataset_sha256,'dataset_sha256')
    _require_hash(environment_sha256,'environment_sha256')
    exp_id=str(uuid4())
    now=time.time()
    db=_connect(root)
    try:
        db.execute('''INSERT INTO experiments VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                   (exp_id,relative,title.strip(),evaluator.id,
                    hashlib.sha256(evaluator.source.encode()).hexdigest(),
                    metric_name.strip(),direction,dataset_sha256.lower(),
                    environment_sha256.lower(),budget,'open',core._version(nb),now))
        db.commit()
        return get(root,relative,exp_id)
    finally:
        db.close()


def list_experiments(root,relative):
    workspace.checked_notebook(root,relative)
    db=_connect(root)
    try:
        return [{'id':r['id'],'title':r['title'],'state':r['state'],
                 'metric_name':r['metric_name'],'direction':r['direction'],
                 'budget':r['budget']} for r in db.execute(
                     'SELECT * FROM experiments WHERE path=? ORDER BY created_at',(relative,)).fetchall()]
    finally:
        db.close()


def get(root,relative,experiment_id):
    workspace.checked_notebook(root,relative)
    db=_connect(root)
    try:
        exp=dict(_experiment(db,relative,experiment_id))
        trials=[dict(row) for row in db.execute('SELECT * FROM trials WHERE experiment_id=? ORDER BY created_at',
                                            (experiment_id,)).fetchall()]
        scored=[r for r in trials if r['status']=='scored']
        ranked=sorted(scored,key=lambda x:x['score'],reverse=exp['direction']=='maximize')
        used=len(trials)
        return {'experiment':exp,'trials':trials,'best_trial':ranked[0] if ranked else None,
                'remaining_budget':max(0,exp['budget']-used)}
    finally:
        db.close()


def register_trial(root,relative,experiment_id,candidate_run_id):
    workspace.checked_notebook(root,relative)
    run=jobs.status(root,candidate_run_id)
    if run['path']!=relative or run['status']!='succeeded' or not run['result']:
        raise ValueError('Candidate must be a successful local Notebook Run')
    db=_connect(root)
    try:
        db.execute('BEGIN IMMEDIATE')
        exp=_experiment(db,relative,experiment_id)
        existing=db.execute('SELECT * FROM trials WHERE experiment_id=? AND candidate_run_id=?',
                            (experiment_id,candidate_run_id)).fetchone()
        if existing:
            db.commit()
            return dict(existing) | {'replayed':True}
        if exp['state']!='open':
            raise ValueError('Experiment is closed')
        if run['cell_id']==exp['evaluator_cell_id']:
            raise ValueError('Candidate must not be the evaluator Cell')
        used=db.execute('SELECT count(*) FROM trials WHERE experiment_id=?',(experiment_id,)).fetchone()[0]
        if used>=exp['budget']:
            raise ValueError('Experiment trial budget exhausted')
        tid=str(uuid4())
        db.execute('''INSERT INTO trials(id,experiment_id,candidate_run_id,candidate_version,status,created_at)
                      VALUES(?,?,?,?,?,?)''',
                   (tid,experiment_id,candidate_run_id,run['result']['version'],'pending',time.time()))
        db.commit()
        return dict(db.execute('SELECT * FROM trials WHERE id=?',(tid,)).fetchone()) | {'replayed':False}
    finally:
        db.close()


def _extract_metric(saved_cell, expected_name):
    chunks=[]
    for output in saved_cell.get('outputs',[]):
        if output.get('output_type')=='stream' and output.get('name')=='stdout':
            raw=output.get('text','')
            chunks.append(''.join(raw) if isinstance(raw,list) else str(raw))
    lines=''.join(chunks).splitlines()
    metrics=[]
    for line in lines:
        if line.startswith(MARKER):
            try:
                data=json.loads(line[len(MARKER):])
            except json.JSONDecodeError as exc:
                raise ValueError('Malformed evaluator metric JSON') from exc
            if not isinstance(data,dict) or data.get('name')!=expected_name:
                continue
            value=data.get('value')
            if type(value) not in (int,float) or not math.isfinite(value):
                raise ValueError('Evaluator metric must be a finite numeric value')
            metrics.append(float(value))
    if len(metrics)!=1:
        raise ValueError(f'Expected exactly one {MARKER} metric named {expected_name}')
    return metrics[0]


def finalize_trial(root,relative,experiment_id,trial_id,evaluator_run_id):
    workspace.checked_notebook(root,relative)
    run=jobs.status(root,evaluator_run_id)
    if run['status']!='succeeded' or run['path']!=relative or not run['result']:
        raise ValueError('Evaluator must be a successful local Notebook Run')
    db=_connect(root)
    try:
        db.execute('BEGIN IMMEDIATE')
        exp=_experiment(db,relative,experiment_id)
        row=db.execute('SELECT * FROM trials WHERE id=? AND experiment_id=?',
                       (trial_id,experiment_id)).fetchone()
        if row is None:
            raise FileNotFoundError('Trial not found')
        if row['status']=='scored':
            if row['evaluator_run_id']!=evaluator_run_id:
                raise core.ConflictError('Trial already finalized with another evaluator Run')
            db.commit()
            return dict(row)|{'replayed':True}
        if exp['state']!='open' or row['candidate_run_id']==evaluator_run_id:
            raise ValueError('Experiment closed or evaluator is not an independent Run')
        if run['cell_id']!=exp['evaluator_cell_id']:
            raise ValueError('Run does not execute the frozen evaluator Cell')
        version=run['result']['version']
        if version<=row['candidate_version']:
            raise ValueError('Evaluator must run after candidate')
        snapshot=ops.get_version(workspace.checked_notebook(root,relative),version)
        cells=snapshot['cells']
        matches=[c for c in cells if c['id']==exp['evaluator_cell_id']]
        if len(matches)!=1:
            raise ValueError('Evaluator Cell missing from executed document')
        if hashlib.sha256(matches[0]['source'].encode()).hexdigest()!=exp['evaluator_source_sha256']:
            raise ValueError('Evaluator source changed since experiment was created')
        # Read exact retained version, not mutable live Notebook output.
        saved_path=Path(snapshot['snapshot'])
        saved_nb=core._read(saved_path)
        evaluator=next(c for c in saved_nb.cells if c.id==exp['evaluator_cell_id'])
        score=_extract_metric(evaluator,exp['metric_name'])
        db.execute('''UPDATE trials SET evaluator_run_id=?,evaluator_version=?,score=?,status=?
                      WHERE id=? AND status=?''',
                   (evaluator_run_id,version,score,'scored',trial_id,'pending'))
        db.commit()
        return dict(db.execute('SELECT * FROM trials WHERE id=?',(trial_id,)).fetchone())|{'replayed':False}
    finally:
        db.close()


def close(root,relative,experiment_id):
    workspace.checked_notebook(root,relative)
    db=_connect(root)
    try:
        _experiment(db,relative,experiment_id)
        db.execute('UPDATE experiments SET state=? WHERE id=?',('closed',experiment_id))
        db.commit()
        return {'experiment_id':experiment_id,'state':'closed'}
    finally:
        db.close()


def artifacts(root,relative,experiment_id):
    workspace.checked_notebook(root,relative)
    db=_connect(root)
    try:
        _experiment(db,relative,experiment_id)
        return [dict(r) for r in db.execute('SELECT * FROM artifacts WHERE experiment_id=?',
                                             (experiment_id,)).fetchall()]
    finally:
        db.close()


def _artifact_root(root):
    base=Path(root).resolve()
    folder=base/'.notebook-artifacts'
    folder.mkdir(parents=True,exist_ok=True,mode=0o700)
    if not folder.resolve().is_relative_to(base):
        raise ValueError('Artifact path escapes NB_ROOT')
    return folder


def artifact_upload(root,relative,experiment_id,workspace_relative):
    workspace.checked_notebook(root,relative)
    p=workspace.checked_workspace(root,relative,workspace_relative)
    if p.stat().st_size>workspace.MAX_FILE_BYTES:
        raise ValueError('Research artifacts are limited to 2MB')
    raw=p.read_bytes()
    sha=hashlib.sha256(raw).hexdigest()
    db=_connect(root)
    try:
        _experiment(db,relative,experiment_id)
        dest=_artifact_root(root)/sha
        if not dest.exists():
            core._atomic_bytes(dest,raw)
        db.execute('INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?,?)',
                   (experiment_id,sha,workspace_relative,len(raw),time.time()))
        db.commit()
        return {'sha256':sha,'bytes':len(raw),'origin':workspace_relative}
    finally:
        db.close()


def artifact_download(root,relative,experiment_id,sha256,workspace_relative,
                      *,overwrite=False):
    _require_hash(sha256,'sha256')
    sha256=sha256.lower()
    db=_connect(root)
    try:
        _experiment(db,relative,experiment_id)
        if not db.execute('SELECT 1 FROM artifacts WHERE experiment_id=? AND sha256=?',
                          (experiment_id,sha256)).fetchone():
            raise FileNotFoundError('Artifact not associated with this experiment')
    finally:
        db.close()
    data=(_artifact_root(root)/sha256).read_bytes()
    if hashlib.sha256(data).hexdigest()!=sha256.lower():
        raise IOError('Artifact content hash mismatch')
    out=workspace.checked_workspace(root,relative,workspace_relative)
    with core._lock(out):
        if out.exists() and not overwrite:
            raise FileExistsError(out)
        core._atomic_bytes(out,data)
    return {'path':workspace_relative,'sha256':sha256,'bytes':len(data)}
