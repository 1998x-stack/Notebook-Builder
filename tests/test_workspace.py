from pathlib import Path
import pytest
import notebook_builder as b
import workspace_ops as w


def test_workspace_roundtrip_and_list(tmp_path):
    b.create(tmp_path / 'run.ipynb')
    assert w.write(tmp_path,'run.ipynb','data/nested.csv','a,b\n1,2')['bytes'] == 7
    assert w.read(tmp_path,'run.ipynb','data/nested.csv')['text'] == 'a,b\n1,2'
    assert len(w.list_files(tmp_path, 'run.ipynb', recursive=True))==1
    assert not w.list_files(tmp_path, 'run.ipynb', recursive=False)
    with pytest.raises(FileExistsError):
        w.write(tmp_path,'run.ipynb','data/nested.csv','bad')
    assert w.delete(tmp_path,'run.ipynb','data/nested.csv')['deleted']


def test_workspace_path_guard_and_symlinks(tmp_path):
    b.create(tmp_path / 'folder' / 'a.ipynb')
    for path in ('../x.ipynb','/tmp/x.ipynb','.hidden/y.ipynb','folder/../../x.ipynb'):
        with pytest.raises(ValueError):
            w.checked_notebook(tmp_path,path)
    for path in ('../data','/etc/passwd','x/../../data','.hidden/file'):
        with pytest.raises(ValueError):
            w.checked_workspace(tmp_path,'folder/a.ipynb',path)
    real = w.checked_workspace(tmp_path,'folder/a.ipynb','data/safe.txt')
    real.parent.mkdir(parents=True)
    (real.parent/'evil').symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        w.checked_workspace(tmp_path,'folder/a.ipynb','data/evil/escape.txt')
    assert w.folder_ensure(tmp_path, 'analytics')['folder']=='analytics'
    assert 'analytics' in w.folder_list(tmp_path)


def test_workspace_caps(tmp_path):
    b.create(tmp_path/'a.ipynb')
    with pytest.raises(ValueError):
        w.write(tmp_path,'a.ipynb','huge.txt','x'*(w.MAX_FILE_BYTES+1))
    w.write(tmp_path,'a.ipynb','test.txt','a'*30)
    assert w.read(tmp_path,'a.ipynb','test.txt',max_chars=10)['truncated']
