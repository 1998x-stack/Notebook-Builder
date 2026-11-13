import hashlib
import json
from pathlib import Path

import pytest
import notebook_builder as b
import notebook_ops as ops
import workspace_ops as ws
import run_jobs as jobs
import research_ledger as r


def test_research_verified_metric_and_artifacts(tmp_path):
    p=tmp_path/'research.ipynb'
    b.create(p,cells=[
        {'source': 'x=3'},
        {'source': 'import json\nprint("NB_METRIC_V1:"+json.dumps({"name":"score","value":float(x*x)}))'},
    ])
    ids=[x['id'] for x in b.describe(p)['cells']]
    s='a'*64
    exp=r.create(tmp_path,'research.ipynb',title='E1',evaluator_cell_id=ids[1],
                 metric_name='score',direction='maximize',dataset_sha256=s,
                 environment_sha256=s,budget=3)
    expid=exp['experiment']['id']
    assert len(r.list_experiments(tmp_path,'research.ipynb'))==1
    candidate=jobs.submit(tmp_path,'research.ipynb',ids[0],idempotency_key='candidate')
    done=jobs.wait(tmp_path,candidate['run_id'],wait_seconds=20)
    assert done['status']=='succeeded',done
    trial=r.register_trial(tmp_path,'research.ipynb',expid,candidate['run_id'])
    evaluator=jobs.submit(tmp_path,'research.ipynb',ids[1],idempotency_key='evaluator')
    verified=jobs.wait(tmp_path,evaluator['run_id'],wait_seconds=20)
    assert verified['status']=='succeeded',verified
    finalized=r.finalize_trial(tmp_path,'research.ipynb',expid,trial['id'],evaluator['run_id'])
    assert finalized['score']==9.0
    assert r.finalize_trial(tmp_path,'research.ipynb',expid,trial['id'],evaluator['run_id'])['replayed']
    assert r.get(tmp_path,'research.ipynb',expid)['best_trial']['score']==9.0
    ws.write(tmp_path,'research.ipynb','results/summary.txt','score=9')
    artifact=r.artifact_upload(tmp_path,'research.ipynb',expid,'results/summary.txt')
    assert artifact['sha256']==hashlib.sha256(b'score=9').hexdigest()
    restored=r.artifact_download(tmp_path,'research.ipynb',expid,artifact['sha256'],'results/restore.txt')
    assert ws.read(tmp_path,'research.ipynb','results/restore.txt')['text']=='score=9'
    assert len(r.artifacts(tmp_path,'research.ipynb',expid))==1
    assert r.close(tmp_path,'research.ipynb',expid)['state']=='closed'
    assert r.register_trial(tmp_path,'research.ipynb',expid,candidate['run_id'])['replayed']
    with pytest.raises(FileNotFoundError):
        r.register_trial(tmp_path,'research.ipynb',expid,candidate['run_id']+'another')


def test_research_rejects_unfrozen_evaluator(tmp_path):
    p=tmp_path/'changed.ipynb'
    b.create(p,cells=[{'source':'x=1'},{'source':'print("NB_METRIC_V1:{\\"name\\":\\"accuracy\\",\\"value\\":0.5}")'}])
    ids=[x['id'] for x in b.describe(p)['cells']]
    digest='c'*64
    exp=r.create(tmp_path,'changed.ipynb',title='E2',evaluator_cell_id=ids[1],
                 metric_name='accuracy',direction='minimize',dataset_sha256=digest,
                 environment_sha256=digest,budget=2)
    job=jobs.submit(tmp_path,'changed.ipynb',ids[0],idempotency_key='c')
    assert jobs.wait(tmp_path,job['run_id'],wait_seconds=20)['status']=='succeeded'
    trial=r.register_trial(tmp_path,'changed.ipynb',exp['experiment']['id'],job['run_id'])
    b.update(p,ids[1],source='print("NB_METRIC_V1:{\\"name\\":\\"accuracy\\",\\"value\\":0.9}")')
    evaluator=jobs.submit(tmp_path,'changed.ipynb',ids[1],idempotency_key='e')
    assert jobs.wait(tmp_path,evaluator['run_id'],wait_seconds=20)['status']=='succeeded'
    with pytest.raises(ValueError,match='source changed'):
        r.finalize_trial(tmp_path,'changed.ipynb',exp['experiment']['id'],trial['id'],evaluator['run_id'])
