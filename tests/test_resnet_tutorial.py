"""Offline ResNet paper-to-tutorial engineering tests (PyTorch CPU)."""
from pathlib import Path
import json
import subprocess
import sys

import nbformat
import torch

ROOT=Path(__file__).resolve().parents[1]
TUT=ROOT/'tutorials'/'resnet'
sys.path.insert(0,str(TUT))
from resnet_educational import (imagenet_resnet,CIFARResNet,OptionAShortcut,parameter_count,
                                BasicBlock,Bottleneck,topk_accuracy)
from build_tutorial import spec


def test_imagenet_model_variants_and_original_stride():
    torch.set_num_threads(min(2,torch.get_num_threads()))
    params={18:11689512,34:21797672,50:25557032,101:44549160,152:60192808}
    for depth,count in params.items():
        model=imagenet_resnet(depth)
        assert parameter_count(model)==count
        assert isinstance(model.layer1[0],BasicBlock if depth in [18,34] else Bottleneck)
        if depth>=50:
            assert model.layer2[0].conv1.stride==(2,2)
            assert model.layer2[0].conv2.stride==(1,1)
        del model


def test_imagenet_forward_and_gradients():
    torch.set_num_threads(2)
    model=imagenet_resnet(18,num_classes=7)
    scores=model(torch.randn(2,3,64,64))
    assert scores.shape==(2,7)
    F=torch.nn.functional.cross_entropy(scores,torch.tensor([1,2]));F.backward()
    assert torch.isfinite(F) and model.layer1[0].conv1.weight.grad.abs().sum()>0


def test_cifar_options_depth_and_identity_padding():
    for n in (3,5,9,18):
        model=CIFARResNet(n=n)
        assert model.depth==6*n+2
        del model
    x=torch.randn(3,16,32,32,requires_grad=True)
    y=OptionAShortcut(16,32,2)(x)
    assert y.shape==(3,32,16,16)
    assert torch.count_nonzero(y[:,:8])==0
    assert torch.equal(y[:,8:24],x[:,:,::2,::2])
    y.sum().backward()
    assert x.grad is not None and x.grad.abs().sum()>0


def test_self_contained_spec_and_paper_provenance():
    s=spec()
    assert len(s['cells'])>=40
    code='\n'.join(c['source'] for c in s['cells'] if c['cell_type']=='code')
    assert 'class ImageNetResNet' in code and 'class CIFARResNet' in code
    assert 'def fit_real_cifar10' in code and 'def fit_real_imagenet' in code
    assert "RUN_REAL_CIFAR10" in code and "RUN_IMAGENET_TRAIN" in code
    assert 'https://arxiv.org/pdf/1512.03385' in s['paper']
    assert "imagenet_resnet" in code and 'Top-5' in '\n'.join(c['source'] for c in s['cells'])


def test_runbook_and_valid_saved_output():
    from notebook_quality import inspect
    p=TUT/'outputs'/'ResNet_Deep_Residual_Learning_Complete_Tutorial.ipynb'
    assert p.exists(),f'Missing executed tutorial: {p}'
    nb=nbformat.read(p,as_version=4);nbformat.validate(nb)
    checks=inspect(p,min_print_cells=14,min_figure_cells=9)
    assert checks['passed'],checks['violations']
    assert checks['metrics']['executed_code_cells']==22
    assert checks['metrics']['valid_png_images']>=13
    assert not checks['metrics']['error_outputs']
    assert (TUT/'outputs'/'artifacts'/'run_manifest.json').exists()
    data=json.loads((TUT/'outputs'/'artifacts'/'run_manifest.json').read_text())
    assert data['steps']==24 and data['verified_reload_exact']
    assert 0<=data['validation_accuracy']<=1


def test_cli_lists_resnet_without_executing():
    proc=subprocess.run([sys.executable,str(ROOT/'nbuild.py'),'tutorial','resnet','--list'],
                        capture_output=True,text=True,timeout=30)
    assert proc.returncode==0,proc.stderr
    assert '1512.03385' in proc.stdout and 'self_contained' in proc.stdout
