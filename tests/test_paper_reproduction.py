"""Reject fabricated, unfinished, or incomplete paper-reproduction claims."""
import json
from pathlib import Path
import pytest
import nbformat
import torch
import paper_reproduction as repro


def record(kind='resnet',depth=20,seed=1,origin='real',steps=64000):
    return {'family':'cifar10','kind':kind,'depth':depth,'seed':seed,
            'paper_profile':'he2015_cifar10_64k_sgd',
            'dataset':{'name':'CIFAR10','origin':origin,'classes':10,'train_count':50000,
                       'test_count':10000,'checksum_verified':True,'fingerprint_sha256':'a'*64},
            'steps_required':64000,'steps_completed':steps,'batch_size':128,
            'optimizer':'SGD(momentum=0.9,weight_decay=1e-4)',
            'scheduler':'step_32000_48000','augmentation':'pad4_crop32_hflip',
            'shortcut':'none' if kind=='plain' else 'A','accuracy_top1':.91,
            'checkpoint_sha256':'b'*64,'test_samples_evaluated':10000}


def check(tmp_path,runs,scope):
    p=tmp_path/'evidence.json';p.write_text(json.dumps({'runs':runs}))
    return repro.inspect(p,scope)


def test_empty_evidence_never_certifies_paper(tmp_path):
    r=check(tmp_path,[],'full_paper')
    assert not r['passed'] and r['stage']=='external_independent_review_required_for_full_paper'
    assert len(r['missing'])>=10


def test_reject_synthetic_and_short_runs(tmp_path):
    assert not check(tmp_path,[record(origin='synthetic')],'cifar10_single')['passed']
    assert not check(tmp_path,[record(steps=24)],'cifar10_single')['passed']
    r=record();r['test_samples_evaluated']=60
    assert not check(tmp_path,[r],'cifar10_single')['passed']


def test_one_real_experiment_is_not_paper(tmp_path):
    assert check(tmp_path,[record()],'cifar10_single')['passed']
    assert not check(tmp_path,[record()],'cifar10_paper')['passed']
    assert not check(tmp_path,[record()],'full_paper')['passed']


def test_entire_cifar_paper_requires_five_110_seeds(tmp_path):
    runs=[record('plain',d) for d in (20,32,44,56)]+[record('resnet',d) for d in (20,32,44,56,110,1202)]
    assert not check(tmp_path,runs,'cifar10_paper')['passed']
    runs.extend(record('resnet',110,seed=s) for s in (2,3,4,5))
    assert check(tmp_path,runs,'cifar10_paper')['passed']
    assert not check(tmp_path,runs,'full_paper')['passed']


def test_manifest_malformed_and_untrusted_fields(tmp_path):
    p=tmp_path/'a.json';p.write_text('{bad')
    with pytest.raises(json.JSONDecodeError):repro.inspect(p)
    r=record();r['dataset']['checksum_verified']=False
    assert not check(tmp_path,[r],'cifar10_single')['passed']
    r=record();r['shortcut']='none' # cannot call residual model a plain model
    assert not check(tmp_path,[r],'cifar10_single')['passed']


def test_notebook_real_dataset_and_zero_prepopulated_fake_results():
    here=Path(__file__).resolve().parents[1]
    for name in ['ResNet_2015_CIFAR10_FULL_PAPER_Colab.ipynb', 'ResNet_2015_ImageNet_FULL_Protocol_Colab.ipynb', 'ResNet_2015_VOC_COCO_Detection_RealData_Colab.ipynb']:
        p=here/'tutorials'/'resnet'/'reproduction'/name
        assert p.exists(), name
        nb=nbformat.read(p,as_version=4)
        assert nbformat.validate(nb) is None
        code='\n'.join(c.source for c in nb.cells if c.cell_type=='code')
        assert 'CIFAR10' in code or 'ImageFolder' in code or 'VOCDetection' in code
        assert 'FakeData(' not in code and 'random_test_accuracy=' not in code
        assert all(c.execution_count is None and not c.outputs for c in nb.cells if c.cell_type=='code')
        assert nb.metadata['notebook_builder']['reproduction']['stage']=='protocol_ready'


def test_resnet_plain_ablation_matches_resnet_parameters():
    from tutorials.resnet.resnet_educational import CIFARResNet
    with torch.device('meta'):
        residual=CIFARResNet(n=3,shortcut='A')
        plain=CIFARResNet(n=3,shortcut='none')
    assert sum(p.numel() for p in residual.parameters())==sum(p.numel() for p in plain.parameters())


def test_cli_reproduce_quality_must_fail_when_unfinished(tmp_path):
    from nbuild import main
    p=tmp_path/'evidence.json';p.write_text('{"runs":[]}')
    assert main(['reproduce-quality',str(p),'--scope','full_paper'])==2


def test_real_checkpoint_sha256_optional_validation(tmp_path):
    import hashlib
    checkpoint=tmp_path/'resnet_20_seed1.pt';checkpoint.write_bytes(b'real checkpoint content for unit test')
    row=record();row['run_id']='resnet_20_seed1';row['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    p=tmp_path/'evidence.json';p.write_text(json.dumps({'runs':[row]}))
    assert repro.inspect(p,'cifar10_single',checkpoint_root=tmp_path)['passed']
    checkpoint.write_bytes(b'tampered')
    assert not repro.inspect(p,'cifar10_single',checkpoint_root=tmp_path)['passed']


def test_cannot_assert_reproduction_on_notebook_creation(tmp_path):
    import notebook_builder as core
    with pytest.raises(ValueError):
        core.create(tmp_path/'fake.ipynb',title='fake',
                    reproduction={'stage':'protocol_complete'})


def test_cifar_checkpoint_resume_uses_real_optimizer_steps_not_claimed_result(tmp_path):
    """A tiny fixture is ONLY a software unit test; never publish its scores as CIFAR research."""
    import numpy as np
    path = Path(__file__).resolve().parents[1] / 'tutorials/resnet/reproduction/ResNet_2015_CIFAR10_FULL_PAPER_Colab.ipynb'
    nb = nbformat.read(path, as_version=4)
    ns={'__name__':'paper_unit_test'}
    for i in (2,4,10,12):
        exec(compile(nb.cells[i].source, f'unit_cell_{i}', 'exec'), ns)
    torch.set_num_threads(2)
    ns['ARTIFACTS']=tmp_path
    ns['GLOBAL_BATCH']=4
    ns['X_TRAIN']=torch.randint(0,255,(12,3,32,32),dtype=torch.uint8)
    ns['Y_TRAIN']=torch.arange(12)%10
    ns['MEAN']=torch.zeros(3,32,32)
    ns['train_ds']=range(12)
    ns['DATASET_FINGERPRINT']='c'*64
    assert ns['train_one_full_experiment']('resnet',20,1,2)==(2,None)
    assert ns['train_one_full_experiment']('resnet',20,1,2)==(2,None)
    ckpt=torch.load(tmp_path/'resnet_20_seed1.pt',map_location='cpu',weights_only=False)
    assert ckpt['step']==4
    assert ckpt['dataset_fingerprint']=='c'*64
    assert not (tmp_path/'resnet_20_seed1.json').exists(), 'A partial mocked run must never claim actual CIFAR metrics'
    ns['DATASET_FINGERPRINT']='d'*64
    with pytest.raises(ValueError,match='dataset fingerprint'):
        ns['train_one_full_experiment']('resnet',20,1,1)


def test_mcp_evidence_tool_guards_root_and_registered_readonly(tmp_path,monkeypatch):
    import mcp_tools
    assert mcp_tools.REGISTRY['paper_reproduction_quality'].permission=='read'
    with pytest.raises(ValueError):
        mcp_tools.paper_reproduction_quality('../secrets.json')
    with pytest.raises(ValueError):
        mcp_tools.paper_reproduction_quality('/tmp/private.json')
    with pytest.raises(ValueError):
        mcp_tools.paper_reproduction_quality('.hidden/a.json')
