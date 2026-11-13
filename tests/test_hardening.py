import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import nbformat
import pytest

import notebook_builder as b
import notebook_ops as ops
import run_jobs as jobs


def test_overwrite_monotonic_and_snapshot(tmp_path):
    p=tmp_path/'same.ipynb'
    assert b.create(p,title='First')['version']==1
    assert b.create(p,title='Second',overwrite=True)['version']==2
    assert len(b.history(p)['snapshots'])==1
    assert ops.get_version(p,1)['cells'][0]['source'].startswith('# First')


def test_symlink_lock_and_history_escape(tmp_path):
    p=tmp_path/'test.ipynb'
    lock=tmp_path/'.test.ipynb.lock'
    elsewhere=tmp_path/'external.txt';elsewhere.write_text('DO NOT CHANGE')
    lock.symlink_to(elsewhere)
    with pytest.raises(OSError):
        b.create(p)
    assert elsewhere.read_text()=='DO NOT CHANGE'
    lock.unlink()
    b.create(p)
    (tmp_path/'.notebook-history').symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        b.add(p,'code','print(1)')
    assert b.describe(p)['version']==1


def test_receipt_is_durable_after_more_edits(tmp_path):
    p=tmp_path/'retry.ipynb'
    b.create(p)
    r=b.add(p,'code','a=1',expected_version=1,operation_id='first')
    b.add(p,'markdown','note',expected_version=2)
    replay=b.add(p,'code','a=1',expected_version=1,operation_id='first')
    assert replay['version']==2 and replay['replayed']
    assert replay['cell_id']==r['cell_id']
    assert b.describe(p)['version']==3


def test_markdown_export_figures(tmp_path):
    p=tmp_path/'plot.ipynb'
    b.create(p,cells=[{'source':'import matplotlib.pyplot as plt\nplt.plot([1,2],[2,3])\nplt.show()'}])
    b.execute(p,timeout=60)
    result=ops.export_markdown(p,tmp_path/'plot.md')
    assert result['assets']>=1
    body=(tmp_path/'plot.md').read_text()
    assets=list((tmp_path/'plot_files').iterdir())
    assert assets
    assert any(asset.name in body and 'plot_files/' in body for asset in assets)


def test_interrupted_jobs_are_not_replayed(tmp_path):
    db=jobs._connect(tmp_path)
    try:
        now=time.time()
        db.execute('''INSERT INTO runs(id,path,cell_id,idempotency_key,request_hash,expected_version,timeout,status,pid,created_at,updated_at)
                     VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
                   ('missing-worker','dead.ipynb','cell-1','job1','hash',1,30,'running',99999999,now,now))
        db.commit()
    finally:
        db.close()
    assert jobs.status(tmp_path,'missing-worker')['status']=='interrupted'


def test_concurrent_cas_workers(tmp_path):
    p=tmp_path/'concurrent.ipynb'
    b.create(p)
    cli=Path(__file__).resolve().parents[1]/'nbuild.py'
    cmd=[sys.executable,str(cli),'add',str(p),'--type','code',
         '--source','print("once")','--expected-version','1']
    a=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    c=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    _, ae=a.communicate(timeout=30)
    _, ce=c.communicate(timeout=30)
    assert sorted([a.returncode,c.returncode])==[0,1],(ae,ce)
    assert b.describe(p)['version']==2
    assert b.describe(p)['cell_count']==1


def test_orphan_queued_no_pid_becomes_interrupted(tmp_path):
    db=jobs._connect(tmp_path)
    try:
        past=time.time()-60
        db.execute('''INSERT INTO runs(id,path,cell_id,idempotency_key,request_hash,expected_version,timeout,status,pid,created_at,updated_at)
                     VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
                   ('orphan-queued','dead.ipynb','cell-1','job2','hash2',1,30,'queued',None,past,past))
        db.commit()
    finally:
        db.close()
    assert jobs.status(tmp_path,'orphan-queued')['status']=='interrupted'
