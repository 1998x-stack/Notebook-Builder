"""Regression tests: snapshot/CAS execution, spec drift and saved-output quality."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import threading

import nbformat
import pytest

import notebook_builder as b
import notebook_ops as ops
import notebook_quality as quality


def test_full_execute_releases_lock_and_preserves_concurrent_edit(monkeypatch, tmp_path):
    path = tmp_path / 'cas.ipynb'
    b.create(path, cells=[{'source': 'print(1)'}])
    edit_done = []

    def concurrent_update(_):
        worker = threading.Thread(target=lambda: edit_done.append(b.add(path, 'markdown', 'human edit')), daemon=True)
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive(), 'write lock held during notebook kernel execution'

    monkeypatch.setattr(b.NotebookClient, 'execute', concurrent_update)
    with pytest.raises(b.ConflictError, match='changed while executing'):
        b.execute(path)
    assert len(edit_done) == 1
    assert b.describe(path)['version'] == 2
    assert b._read(path).cells[1].source == 'human edit'


def test_single_cell_window_releases_lock_and_conflicts(monkeypatch, tmp_path):
    path = tmp_path / 'window.ipynb'
    b.create(path, cells=[{'source':'print(2)'}])
    class FakeClient:
        def __init__(self, document, **kwargs):
            self.document = document
        @contextmanager
        def setup_kernel(self):
            yield self
        def execute_cell(self, cell, index):
            t=threading.Thread(target=lambda:b.add(path,'markdown','edited during cell'),daemon=True)
            t.start();t.join(timeout=2)
            assert not t.is_alive(), 'lock held during cell execution'
    monkeypatch.setattr(ops, 'NotebookClient', FakeClient)
    with pytest.raises(b.ConflictError, match='changed while executing'):
        ops.execute_cells(path, start=0, count=1)
    assert b.describe(path)['version'] == 2


def test_source_byte_mutation_with_unchanged_version_conflicts(monkeypatch, tmp_path):
    path = tmp_path/'manual.ipynb'
    b.create(path, cells=[{'source':'print(2)'}])
    def edit_without_version(_):
        nb=b._read(path)
        nb.cells[0].source='print(99)'
        path.write_text(nbformat.writes(nb),encoding='utf-8')
    monkeypatch.setattr(b.NotebookClient,'execute',edit_without_version)
    with pytest.raises(b.ConflictError):
        b.execute(path)
    assert b._read(path).cells[0].source=='print(99)'


def test_quality_gate_reports_missing_prints_and_invalid_png(tmp_path):
    path=tmp_path/'qa.ipynb'
    b.create(path,cells=[{'source':'print(1)'}])
    check=quality.inspect(path)
    assert not check['passed']
    assert 'lack execution_count' in ' '.join(check['violations'])
    nb=b._read(path)
    nb.cells[0].execution_count=1
    nb.cells[0].outputs=[nbformat.v4.new_output('stream',name='stdout',text='1\n'),
                         nbformat.v4.new_output('display_data',data={'image/png':'aW52YWxpZA=='})]
    path.write_text(nbformat.writes(nb),encoding='utf-8')
    check=quality.inspect(path,min_print_cells=1,min_figure_cells=1)
    assert check['metrics']['stdout_cells']==1
    assert check['metrics']['invalid_png_cells']==[0]
    assert not check['passed']


def test_tutorial_spec_drift_is_detected_and_force_explicitly_restores(monkeypatch,tmp_path):
    import sys
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'tutorials'/'nanogpt'))
    import build_tutorials as tutorial
    monkeypatch.setattr(tutorial,'HERE',tmp_path)
    created=tutorial.run_series(build=True,only=['00'])
    assert created['count']==1
    path=Path(created['tutorials'][0]['notebook'])
    assert b._read(path).metadata[b.META_KEY]['spec_sha256'] == created['tutorials'][0]['spec_sha256']
    original=b._read(path)
    b.update(path,1,source='broken tutorial source')
    with pytest.raises(b.ConflictError,match='spec drift'):
        tutorial.run_series(build=True,only=['00'])
    assert b._read(path).cells[1].source == 'broken tutorial source'
    rebuilt=tutorial.run_series(build=True,force=True,only=['00'])
    assert rebuilt['tutorials'][0]['status']=='rebuilt'
    assert b._read(path).cells[1].source == original.cells[1].source


def test_quality_cli_exit_code(tmp_path):
    import subprocess, sys
    cli = Path(__file__).resolve().parents[1] / 'nbuild.py'
    path = tmp_path / 'not-yet-executed.ipynb'
    b.create(path, cells=[{'source': 'print(5)'}])
    result = subprocess.run([sys.executable, str(cli), 'quality', str(path)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'lack execution_count' in result.stdout


def test_run_job_creates_diagnostic_log_file(tmp_path):
    import run_jobs as jobs
    p=tmp_path/'run.ipynb'
    b.create(p, cells=[{'source':'print(12)'}])
    run_id=jobs.submit(tmp_path,'run.ipynb',b.describe(p)['cells'][0]['id'],
                       idempotency_key='diag')['run_id']
    outcome=jobs.wait(tmp_path,run_id,wait_seconds=20)
    assert outcome['status']=='succeeded',outcome
    logfile=tmp_path/outcome['log_relative_path']
    assert logfile.is_file()
    assert logfile.stat().st_mode & 0o077 == 0
