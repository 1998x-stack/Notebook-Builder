import copy
import nbformat
import pytest
import notebook_builder as b
import notebook_ops as o


def test_receipt_idempotency_and_collision(tmp_path):
    p = tmp_path / 'test.ipynb'
    b.create(p)
    first = b.add(p, 'code', 'print(1)', expected_version=1, operation_id='retry-01')
    again = b.add(p, 'code', 'print(1)', expected_version=1, operation_id='retry-01')
    assert again['replayed'] is True
    assert first['cell_id'] == again['cell_id']
    assert b.describe(p)['version'] == 2
    with pytest.raises(b.ConflictError):
        b.add(p, 'code', 'print(2)', expected_version=1, operation_id='retry-01')


def test_versions_fork_restore_diff_and_import(tmp_path):
    p = tmp_path / 'src.ipynb'
    b.create(p, cells=[{'source':'a=1'}])
    b.update(p, 0, source='a=2')
    assert o.diff_versions(p, 1, 2)['changes'][0]['change'] == 'modified'
    fork = tmp_path / 'fork.ipynb'
    o.fork_version(p, 1, fork)
    assert o.cell_get(fork, 0)['source'] == 'a=1'
    assert b.describe(fork)['version'] == 1
    o.restore_version(p, 1, expected_version=2)
    assert o.cell_get(p, 0)['source'] == 'a=1'
    dest = tmp_path / 'imported.ipynb'
    o.import_notebook(p, dest)
    assert len(o.list_notebooks(tmp_path)) == 3
    exported = tmp_path / 'export.ipynb'
    o.export_ipynb(dest, exported)
    assert b.validate(exported)['valid']


def test_merge_independent_edits_conflict_and_choices(tmp_path):
    p = tmp_path / 'merge.ipynb'
    b.create(p, cells=[{'source': 'a=1'},{'source': 'b=2'}])
    base = nbformat.read(p, as_version=4)
    draft = copy.deepcopy(base)
    draft.cells[0].source = 'a=100'
    b.update(p, 1, source='b=200')
    preview = o.merge_preview(p, 1, dict(draft))
    assert preview['can_commit']
    result = o.merge_commit(p, 1, dict(draft), expected_version=2)
    assert result['version']==3
    assert o.cell_get(p, 0)['source']=='a=100'
    assert o.cell_get(p, 1)['source']=='b=200'
    competing = copy.deepcopy(base)
    competing.cells[0].source = 'a=-10'
    preview = o.merge_preview(p, 1, dict(competing))
    assert not preview['can_commit'] and preview['conflicts'][0]['cell_id'] == base.cells[0].id
    with pytest.raises(b.ConflictError):
        o.merge_commit(p, 1, dict(competing), expected_version=3)
    picked = o.merge_commit(p, 1, dict(competing), expected_version=3,
                            choices={base.cells[0].id: 'mine'})
    assert picked['version'] == 4 and o.cell_get(p, 0)['source'] == 'a=-10'


def test_selective_execution_warmup_and_rollback(tmp_path):
    p=tmp_path / 'run.ipynb'
    b.create(p, cells=[{'source':'base=41'},{'source':'print(base+1)'}])
    r = o.execute_cells(p,start=1,count=1,bootstrap=True,timeout=60)
    assert r['version'] == 2 and r['executed_indices'] == [1]
    doc=nbformat.read(p,as_version=4)
    assert doc.cells[0].execution_count is None
    assert '42' in doc.cells[1].outputs[0].text
    assert o.code_execute(p,'print(5*5)',persist=False)['outputs'][0]['text'].strip() == '25'
    assert b.describe(p)['version'] == 2
    out=o.code_execute(p,'print(4)',persist=True)
    assert out['persisted'] and b.describe(p)['version']==3
    b.add(p,'code',"raise RuntimeError('boom')")
    before=p.read_bytes()
    with pytest.raises(Exception):
        o.execute_cells(p,start=3,count=1,bootstrap=False,timeout=60)
    assert p.read_bytes() == before


def test_delete_is_soft_delete(tmp_path):
    p=tmp_path / 'delete.ipynb'
    b.create(p)
    result=o.delete_notebook(p,expected_version=1)
    assert not p.exists()
    assert __import__('pathlib').Path(result['recoverable_at']).is_file()
