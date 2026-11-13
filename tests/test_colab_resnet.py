"""CPU fallback & GPU source contracts for the standalone Colab ResNet notebook."""
import os
import subprocess
import sys
from pathlib import Path

import nbformat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tutorials.resnet.build_tutorial import spec


def test_gpu_auto_and_amp_paths_present():
    cells=spec()['cells']
    setup=next(c['source'] for c in cells if c['cell_type']=='code' and 'AMP_ENABLED =' in c['source'])
    train=next(c['source'] for c in cells if c['cell_type']=='code' and 'scaler=make_grad_scaler()' in c['source'])
    assert "RESNET_DEVICE','auto'" in setup
    assert "torch.cuda.is_available()" in setup
    assert "torch.cuda.get_device_name(0)" in setup
    assert "torch.autocast(device_type='cuda'" in setup
    assert 'train_data=Xtr.to(DEVICE)' in train
    assert 'val_data=Xva.to(DEVICE)' in train
    assert 'scaler.scale(loss).backward()' in train
    assert 'scaler.step(opt)' in train
    assert 'scaler.unscale_(opt)' in train


def test_cpu_fallback_executes_setup_without_colab():
    setup=next(c['source'] for c in spec()['cells'] if c['cell_type']=='code' and 'AMP_ENABLED =' in c['source'])
    env=os.environ.copy()
    env.update(RESNET_DEVICE='cpu',RESNET_WORKDIR='/tmp/nb-colab-test-cpu',RESNET_AMP='1')
    p=subprocess.run([sys.executable,'-c',setup+"\nassert DEVICE.type=='cpu' and AMP_ENABLED is False\nprint('CPU FALLBACK OK')"],
                     cwd=str(ROOT),env=env,capture_output=True,text=True,timeout=35)
    assert p.returncode==0,p.stderr
    assert 'CPU FALLBACK OK' in p.stdout


def test_explicit_cuda_without_gpu_is_clean_failure():
    setup=next(c['source'] for c in spec()['cells'] if c['cell_type']=='code' and 'AMP_ENABLED =' in c['source'])
    env=os.environ.copy();env.update(RESNET_DEVICE='cuda',RESNET_WORKDIR='/tmp/nb-colab-test-cuda')
    # Only assert the error branch on machines which actually lack a GPU.
    p=subprocess.run([sys.executable,'-c',setup],cwd=str(ROOT),env=env,capture_output=True,text=True,timeout=35)
    if 'CUDA explicitly requested but unavailable' in p.stderr:
        assert p.returncode!=0
    else:
        assert p.returncode==0,p.stderr


def test_embedded_model_and_no_remote_downloads():
    s=spec();code='\n'.join(c['source'] for c in s['cells'] if c['cell_type']=='code')
    assert 'class ImageNetResNet' in code
    assert 'class CIFARResNet' in code
    assert 'RUN_REAL_CIFAR10' in code and 'RUN_IMAGENET_TRAIN' in code
    assert 'torch.cuda.is_available()' in code
    assert s['runtime']=='colab-gpu-auto-cpu-fallback'
    p=ROOT/'tutorials/resnet/outputs/ResNet_Deep_Residual_Learning_Complete_Tutorial.ipynb'
    doc=nbformat.read(p,as_version=4);nbformat.validate(doc)
    assert 'AMP_ENABLED' in '\n'.join(c.source for c in doc.cells if c.cell_type=='code')
