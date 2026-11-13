import json
from pathlib import Path
import nbformat
import pytest
import notebook_builder as b


def test_create_add_edit_move_delete_history(tmp_path):
    f = tmp_path / "analysis.ipynb"
    r = b.create(f, title="Test")
    assert r['version'] == 1 and r['cell_count'] == 1
    r = b.add(f, "code", "print(1)", expected_version=1)
    stable_id = r['cell_id']
    assert r['version'] == 2 and r['cell_count'] == 2
    assert b.update(f, stable_id, source="print(2)", expected_version=2)['version'] == 3
    with pytest.raises(b.ConflictError):
        b.update(f, stable_id, source="print(3)", expected_version=2)
    assert b.move(f, stable_id, 0, expected_version=3)['version'] == 4
    assert b.describe(f)['cells'][0]['id'] == stable_id
    assert b.delete(f, stable_id, expected_version=4)['version'] == 5
    assert len(b.history(f)['snapshots']) == 4
    nbformat.validate(nbformat.read(f, as_version=4))


def test_create_spec_and_html(tmp_path):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"title":"Demo", "cells":[{"cell_type":"code", "source":"x=1"}]}))
    f = tmp_path / "demo.ipynb"
    assert b.from_spec(f, spec)['cell_count'] == 2
    out = tmp_path / "demo.html"
    assert b.export_html(f, out)['bytes'] > 500
    with pytest.raises(FileExistsError):
        b.create(f)


def test_execute_kernel(tmp_path):
    f = tmp_path / "run.ipynb"
    b.create(f, cells=[{"cell_type":"code", "source":"a=21\nprint(a*2)"}])
    desc = b.execute(f, timeout=90)
    assert desc['version'] == 2
    doc = nbformat.read(f, as_version=4)
    assert '42' in doc.cells[0].outputs[0].text


def test_invalid_spec_no_partial_file(tmp_path):
    f = tmp_path / "invalid.ipynb"
    with pytest.raises(ValueError):
        b.create(f, cells=[{"cell_type":"not-real", "source":"hi"}])
    assert not f.exists()


def test_execute_error_does_not_overwrite_original(tmp_path):
    f = tmp_path / "fails.ipynb"
    b.create(f, cells=[{"cell_type":"code", "source":"raise ValueError('expected failure')"}])
    original = f.read_bytes()
    with pytest.raises(Exception):
        b.execute(f, timeout=60)
    assert f.read_bytes() == original
    assert b.describe(f)['version'] == 1


def test_update_type_resets_output_and_keeps_stable_id(tmp_path):
    f = tmp_path / 'modify.ipynb'
    b.create(f, cells=[{'cell_type':'code','source':'print("hello")'}])
    id_ = b.describe(f)['cells'][0]['id']
    b.execute(f, timeout=60)
    b.update(f, id_, cell_type='markdown', source='## Text')
    doc = nbformat.read(f, as_version=4)
    assert doc.cells[0].id == id_ and doc.cells[0].cell_type == 'markdown'
    assert 'outputs' not in doc.cells[0]
