from pathlib import Path
import pytest
import notebook_builder as b
from runtime_manager import RuntimeManager


def test_persistent_kernel_lifecycle(tmp_path):
    p = tmp_path/'runtime.ipynb'
    b.create(p)
    r=RuntimeManager()
    try:
        assert r.status(p)['status'] == 'stopped'
        assert r.start(p)['status']=='ready'
        assert r.execute(p,'x=12\nprint(x*3)')['outputs'][0]['text'].strip()=='36'
        assert '12' in r.execute(p,'print(x)')['outputs'][0]['text']
        assert any(row['name']=='x' for row in r.variables(p)['variables'])
    finally:
        assert r.stop(p)['status']=='stopped'
