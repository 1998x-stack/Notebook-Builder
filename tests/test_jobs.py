from pathlib import Path
from uuid import uuid4
import time
import pytest
import notebook_builder as b
import notebook_ops as ops
import run_jobs as runs


def test_run_submit_wait_replay(tmp_path):
    p=tmp_path/'run.ipynb'
    b.create(p,cells=[{'source':'print(7*6)'}])
    cell=b.describe(p)['cells'][0]['id']
    first=runs.submit(tmp_path,'run.ipynb',cell,idempotency_key='run1',timeout=50)
    replay=runs.submit(tmp_path,'run.ipynb',cell,idempotency_key='run1',timeout=50)
    assert replay['run_id']==first['run_id'] and replay['replayed']
    final=runs.wait(tmp_path,first['run_id'],wait_seconds=20)
    assert final['status']=='succeeded',final
    assert '42' in ops.cell_get(p,0,include_outputs=True)['outputs'][0]['text']
    with pytest.raises(b.ConflictError):
        runs.submit(tmp_path,'run.ipynb',cell,idempotency_key='run1',timeout=49)


def test_run_cancel(tmp_path):
    p=tmp_path/'slow.ipynb'
    b.create(p,cells=[{'source':'import time\ntime.sleep(30)\nprint("done")'}])
    cell=b.describe(p)['cells'][0]['id']
    job=runs.submit(tmp_path,'slow.ipynb',cell,idempotency_key='run2',timeout=50)
    runs.cancel(tmp_path,job['run_id'])
    last=runs.wait(tmp_path,job['run_id'],wait_seconds=10)
    assert last['status']=='cancelled',last
    assert b.describe(p)['version']==1
