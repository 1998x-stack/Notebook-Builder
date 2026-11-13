"""Build original-paper CIFAR-10 FULL protocol Notebook, never fabricate training outputs.

The generated notebook is intentionally UNEXECUTED. Execution requires real CIFAR-10
and tens of thousands of SGD updates for each of the published network variants.
"""
from __future__ import annotations
import json
import hashlib
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import notebook_builder as builder
MODEL_SOURCE = (Path(__file__).resolve().parents[1]/'resnet_educational.py').read_text(encoding='utf-8')
TITLE='ResNet 2015 原论文复现｜真实 CIFAR-10 · 64k Steps · Plain vs Residual'


def c(t): return {'cell_type':'code','source':t.strip()+'\n'}
def m(t): return {'cell_type':'markdown','source':t.strip()+'\n'}

CELLS=[
 m('''## 0｜论文原始协议与状态定义

**目标论文**：Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, *Deep Residual Learning for Image Recognition*, arXiv:1512.03385, Section 4.2, Table 6, Figures 6–7. 原文：https://arxiv.org/pdf/1512.03385

本文件**不是已完成复现的成绩单**：Notebook 初始执行状态是 `protocol_ready`，所有训练与测试指标必须由在 Colab 上的**真实 CIFAR-10** 运行产生，禁止使用 FakeData、合成训练集、随机准确率填充或 `SKIPPED` 冒充完成。模型源代码内嵌，可以独立运行。

- 官方数据：CIFAR-10 50,000 train + 10,000 test，10 类，torchvision 官方 MD5 校验；测试集只用于完成全部 64,000 步之后的最终评估。
- 网络：plain-20/32/44/56、ResNet-20/32/44/56/110/1202；原论文 CIFAR Option A 零填充 shortcut；ResNet-110 额外 5 种随机种子实验。
- 优化：SGD momentum 0.9、weight decay 1e-4、batch 128、初始 LR 0.1、32k/48k 除以十、64k 结束；pad 4 + random crop 32 + flip、训练集像素均值扣除，不使用 dropout。ResNet-110 使用 0.01 热身直到训练错误率低于 80%（原文大约 400 步）。
- **重要**：训练需要大量 GPU 小时，免费 Colab 单次会话通常无法完成全部模型。每个模型自动保存可恢复 checkpoint；**没有完成的模型不产生合格实验证据**。在所有模型、重复种子完成后，才能报告 CIFAR 部分协议完成。ImageNet、VOC/COCO 另有独立数据/算力要求，CIFAR 完成也不能标记“整篇论文已复现”。
- **严格一致性**：单卡 batch128 是相同全局 batch、但不同于论文两 GPU 的设备配置；AMP 默认关闭。如启用混合精度，需明确标注现代实现偏差。

**运行方式**：Colab → Runtime → Change runtime type → T4 GPU（或其他 GPU）→ 编辑下方项目目录和每会话预算 → Run all。再次启动会从 Google Drive 恢复训练。支持 CPU 但完整训练极慢。每个检查点被当作可信本地文件，不能加载不可信来源。'''),
 c('''import os, sys, time, json, hashlib, random, platform
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

IN_COLAB = 'google.colab' in sys.modules or 'COLAB_RELEASE_TAG' in os.environ
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
RUN_NAME = 'resnet2015-cifar10-paper'
# 每次运行至多执行这些 SGD 步数，分多次 Colab 会话完成所有模型；设置 None 不限制步数。
SESSION_STEP_BUDGET = 2000
CHECKPOINT_EVERY = 500
# 全论文的 CIFAR-10 实验矩阵；每次只推进未完成的下一个实验
PLAN = ([('plain', x, 1337) for x in (20,32,44,56)] +
        [('resnet', x, 1337) for x in (20,32,44,56,110,1202)] +
        [('resnet',110,s) for s in (1338,1339,1340,1341)])
# paper-faithful default; AMP 可以提速，但属于数值实现偏差
USE_AMP = False
GLOBAL_BATCH = 128
PAPER_STEPS = 64000
DATA_DIR = Path('/content/datasets/cifar10') if IN_COLAB else Path('./data/cifar10')
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    ARTIFACTS = Path('/content/drive/MyDrive/Notebook-Builder/ResNet-2015/CIFAR10')
else:
    ARTIFACTS = Path('./artifacts/cifar10-paper')
ARTIFACTS.mkdir(parents=True,exist_ok=True)
print('Mode: REAL CIFAR-10 FULL PAPER PROTOCOL (not a synthetic tutorial)')
print('device:',DEVICE,'torch:',torch.__version__,'cuda:',torch.cuda.is_available())
if DEVICE.type=='cuda':
    print('GPU:',torch.cuda.get_device_name(0),'memory_GiB:',round(torch.cuda.get_device_properties(0).total_memory/2**30,2))
print('run artifacts:',ARTIFACTS.resolve(),'planned full experiments:',len(PLAN))
print('steps/model:',PAPER_STEPS,'per-session step budget:',SESSION_STEP_BUDGET)
'''),
 m('''## 1｜原论文模型完整实现（内嵌）

直接使用当前 Notebook Builder 中的 ResNet-v1 实现。CIFAR 版 `6n+2` 层、16/32/64 通道、Option A 零填充。`plain` 消去全部残差相加，保持卷积/BN/FC 参数数量相同。此处不会调用预训练权重。'''),
 c(MODEL_SOURCE),
 c('''# 纯实现层验证，属于代码结构测试，不属于真实数据实验结果
for depth in (20,32,44,56,110,1202):
    n=(depth-2)//6
    with torch.device('meta'):
        model=CIFARResNet(n=n,shortcut='A',base_width=16)
    print(f'ResNet-{depth:4d}: {parameter_count(model):,} trainable params (paper base width=16)')
assert CIFARResNet(n=3,shortcut='none').depth==20
assert parameter_count(CIFARResNet(n=3,shortcut='none'))==parameter_count(CIFARResNet(n=3,shortcut='A'))
'''),
 m('''## 2｜真实 CIFAR-10 数据下载、来源校验、训练/测试隔离

只有 torchvision 官方 CIFAR-10 数据可以通过此门禁；严格校验官方包 MD5、50k/10k 张数、10 类标签，并计算本地数据文件 SHA-256。若下载失败或文件不完整将**直接报错**，绝不回退 FakeData。只使用训练集估计原论文像素均值。最终测试集只在模型完成 64k 步之后被读取用于评估。'''),
 c('''import torchvision
from torchvision.datasets import CIFAR10
print('torchvision:',torchvision.__version__)
DATA_DIR.mkdir(parents=True,exist_ok=True)
train_ds=CIFAR10(root=str(DATA_DIR),train=True,download=True)
test_ds=CIFAR10(root=str(DATA_DIR),train=False,download=True)
assert train_ds._check_integrity() and test_ds._check_integrity(), 'Official MD5 failed'
assert len(train_ds)==50000 and len(test_ds)==10000
assert len(set(train_ds.targets))==10 and len(set(test_ds.targets))==10
RAW_DIR=DATA_DIR/'cifar-10-batches-py'
source_files=[RAW_DIR / f'data_batch_{i}' for i in range(1,6)]+[RAW_DIR/'test_batch']
assert all(p.is_file() for p in source_files)
digest=hashlib.sha256()
for file in source_files:
    with file.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024), b''):
            digest.update(block)
DATASET_FINGERPRINT=digest.hexdigest()
print('Verified dataset:',type(train_ds).__name__,'official CIFAR-10 MD5=OK')
print('train:',len(train_ds),'test:',len(test_ds),'label classes:',train_ds.classes)
print('dataset raw content sha256:',DATASET_FINGERPRINT)
# 根据 50k 真正的训练图像计算每像素均值，分块避免额外分配 >600MB 的 float 数组
mean=np.zeros((32,32,3),dtype=np.float64)
for start in range(0,len(train_ds),500):
    mean += train_ds.data[start:start+500].sum(axis=0)
mean=(mean/len(train_ds)/255.0).transpose(2,0,1).astype('float32')
MEAN=torch.from_numpy(mean).to(DEVICE)
X_TRAIN=torch.from_numpy(np.ascontiguousarray(train_ds.data.transpose(0,3,1,2))).to(DEVICE)
Y_TRAIN=torch.tensor(train_ds.targets,device=DEVICE,dtype=torch.long)
X_TEST=torch.from_numpy(np.ascontiguousarray(test_ds.data.transpose(0,3,1,2))).to(DEVICE)
Y_TEST=torch.tensor(test_ds.targets,device=DEVICE,dtype=torch.long)
print('raw tensor memory MiB:',round((X_TRAIN.numel()+X_TEST.numel())/2**20,1))
'''),
 c('''fig,axes=plt.subplots(2,5,figsize=(11,5))
for ax,i in zip(axes.flat,range(10)):
    ax.imshow(train_ds.data[i]);ax.set_title(train_ds.classes[train_ds.targets[i]],fontsize=9);ax.axis('off')
fig.suptitle('Official, verified CIFAR-10 training images (not synthesized)')
plt.tight_layout();plt.show()
'''),
 m('''## 3｜严格的真实数据增广、全局 Batch 与 LR Schedule

按论文使用 Pad(4) → 32×32 随机裁剪 → 水平翻转，并从裁剪图像扣除训练集估计的逐像素均值。随机数生成器状态被保存进 Checkpoint，方便跨 Colab 会话恢复。默认使用现代 PyTorch CUDA 操作实现批量增广；数值计算设备与框架版本需要在运行清单中记录。'''),
 c('''def paper_lr(step:int,depth:int=20)->float:
    # ResNet110 按论文以0.01热身 ~400步，严谨性限制：原文按训练错误率 < 80% 调回0.1，
    # 实际执行代码中按监测误差切换，相关状态保存到checkpoint。
    if depth==110 and step<400: return 0.01
    if step>=48000:return 0.001
    if step>=32000:return 0.01
    return 0.1

def augmented_batch(indices, rng):
    # CPU Generator provides device-independent deterministic draw sequence.
    b=len(indices)
    images=X_TRAIN[indices.to(DEVICE)].float().div_(255.)
    padded=F.pad(images,(4,4,4,4)) # B,C,40,40
    y0=torch.randint(0,9,(b,),generator=rng).to(DEVICE)
    x0=torch.randint(0,9,(b,),generator=rng).to(DEVICE)
    do_flip=torch.randint(0,2,(b,),generator=rng).bool().to(DEVICE)
    yy=y0[:,None,None]+torch.arange(32,device=DEVICE)[None,:,None]
    xx=x0[:,None,None]+torch.arange(32,device=DEVICE)[None,None,:]
    idx=(yy*40+xx).reshape(b,1,1024).expand(-1,3,-1)
    cropped=torch.gather(padded.reshape(b,3,1600),2,idx).reshape(b,3,32,32)
    cropped=torch.where(do_flip[:,None,None,None],torch.flip(cropped,dims=(-1,)),cropped)
    return cropped-MEAN[None],Y_TRAIN[indices.to(DEVICE)]

def test_batch(i,j):
    return X_TEST[i:j].float().div(255.)-MEAN[None],Y_TEST[i:j]

steps=np.arange(64000)
plt.figure(figsize=(9,2.8));plt.plot(steps,[paper_lr(int(s)) for s in steps]);
plt.title('PAPER CONFIGURATION: planned LR (not an observed training curve)')
plt.ylabel('Learning rate');plt.xlabel('Optimizer step');plt.grid(alpha=.25);plt.show()
'''),
 m('''## 4｜完整 SGD 训练与跨会话恢复

全部训练迭代都用真实 CIFAR-10 样本。每项需要 **64,000 次 `optimizer.step()`**，不能把 `SESSION_STEP_BUDGET` 当作论文迭代上限。下面会反复写入原子 checkpoint，支持下一次运行继续。只有完整训练结束后才运行全量 10,000 张测试集。

对 ResNet-110，原论文是在训练错误率低于 80% 后从 0.01 切换到 0.1；这里实现**按实时 mini-batch 训练错误率判断**，原文没有公布精确的判定平滑策略，因此属于记录在案的近似。必须在实验报告中说明。
'''),
 c('''import tempfile

def run_filename(kind,depth,seed):return f'{kind}_{depth}_seed{seed}'

def save_atomic(obj,path):
    tmp=path.with_name(path.name+'.tmp')
    torch.save(obj,tmp)
    os.replace(tmp,path)

def sha256_file(path):
    sha=hashlib.sha256()
    with open(path,'rb') as fh:
        for block in iter(lambda:fh.read(1024*1024),b''):sha.update(block)
    return sha.hexdigest()

def build_cifar(kind,depth):
    if kind not in ('plain','resnet') or (depth-2)%6:raise ValueError('unsupported paper model')
    return CIFARResNet(n=(depth-2)//6,num_classes=10,shortcut='none' if kind=='plain' else 'A',base_width=16)

@torch.no_grad()
def eval_on_official_test(model,batch=250):
    model.eval();hits=0;loss_sum=0.;n=0
    for begin in range(0,len(test_ds),batch):
        x,y=test_batch(begin,min(begin+batch,len(test_ds)))
        out=model(x);loss_sum+=F.cross_entropy(out,y,reduction='sum').item()
        hits+=(out.argmax(dim=-1)==y).sum().item();n+=len(y)
    assert n==10000
    return {'accuracy_top1':hits/n,'cross_entropy':loss_sum/n,'test_samples_evaluated':n}

def train_one_full_experiment(kind,depth,seed,remaining):
    label=run_filename(kind,depth,seed)
    ckpt=ARTIFACTS/(label+'.pt')
    donefile=ARTIFACTS/(label+'.json')
    if donefile.exists():
        saved=json.loads(donefile.read_text())
        if saved.get('dataset',{}).get('fingerprint_sha256') != DATASET_FINGERPRINT:
            raise ValueError('Dataset fingerprint mismatch with completed real-data run')
        if not ckpt.is_file() or sha256_file(ckpt) != saved.get('checkpoint_sha256'):
            raise ValueError('Completed checkpoint missing or checksum changed')
        print('EXPERIMENT COMPLETE previously:',label,'test accuracy',saved['accuracy_top1'])
        return 0, saved
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if DEVICE.type=='cuda':torch.cuda.manual_seed_all(seed)
    model=build_cifar(kind,depth).to(DEVICE)
    optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=USE_AMP and DEVICE.type=='cuda')
    sampler=torch.Generator(device='cpu').manual_seed(seed+319)
    step=0;warmup_finished=(depth!=110);warmup_end=None
    history=[]
    if ckpt.exists():
        state=torch.load(ckpt,map_location='cpu',weights_only=False) # our OWN trusted checkpoint only
        if (state['kind'],state['depth'],state['seed'])!=(kind,depth,seed):raise ValueError('Checkpoint config changed')
        if state['device_type']!=DEVICE.type:raise ValueError('Device type changed; use same accelerator to preserve RNG')
        if state.get('dataset_fingerprint') != DATASET_FINGERPRINT:
            raise ValueError('Attempt to resume with a different dataset fingerprint')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler']);sampler.set_state(state['sampler_state'])
        torch.set_rng_state(state['torch_cpu_rng'])
        if DEVICE.type=='cuda':torch.cuda.set_rng_state_all(state['torch_cuda_rng'])
        step=state['step'];history=state['history'];warmup_finished=state['warmup_finished'];warmup_end=state['warmup_end']
        print('RESUMED:',label,'at step',step)
    before=step;until=PAPER_STEPS if remaining is None else min(PAPER_STEPS,step+remaining)
    model.train()
    tic=time.monotonic()
    for it in range(step,until):
        # Model-specific original 110-layer warm-up, until training error < 80%.
        lr=(0.01 if depth==110 and not warmup_finished else paper_lr(it,depth=20))
        for group in optimizer.param_groups:group['lr']=lr
        idx=torch.randint(0,len(train_ds),(GLOBAL_BATCH,),generator=sampler)
        x,y=augmented_batch(idx,sampler)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda',enabled=USE_AMP and DEVICE.type=='cuda'):
            logits=model(x);loss=F.cross_entropy(logits,y)
        scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
        # No evaluation on CIFAR test until all 64k steps complete.
        if depth==110 and not warmup_finished:
            batch_error=(logits.detach().argmax(1)!=y).float().mean().item()
            if batch_error<.8:
                warmup_finished=True;warmup_end=it+1
        completed=it+1
        if completed%200==0 or completed in (1,PAPER_STEPS):
            history.append({'step':completed,'loss':float(loss.item()),'batch_train_accuracy':float((logits.argmax(1)==y).float().mean().item()),'lr':lr})
            print(f'{label} {completed:6d}/64000 loss={loss.item():.4f} batch_train_acc={history[-1]["batch_train_accuracy"]:.3f} lr={lr:.4g}',flush=True)
        if completed % CHECKPOINT_EVERY == 0 or completed==until:
            state={'kind':kind,'depth':depth,'seed':seed,'step':completed,
                   'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                   'scaler':scaler.state_dict(),'sampler_state':sampler.get_state(),
                   'torch_cpu_rng':torch.get_rng_state(),
                   'torch_cuda_rng':torch.cuda.get_rng_state_all() if DEVICE.type=='cuda' else None,
                   'device_type':DEVICE.type,'history':history,'dataset_fingerprint':DATASET_FINGERPRINT,
                   'warmup_finished':warmup_finished,'warmup_end':warmup_end}
            save_atomic(state,ckpt)
    del optimizer,scaler
    print('session progress:',label,until,'of',PAPER_STEPS,'elapsed sec:',round(time.monotonic()-tic,1))
    if until < PAPER_STEPS:
        return until-before,None
    eval_result=eval_on_official_test(model)
    paper_deviations=[]
    if USE_AMP:paper_deviations.append('mixed_precision')
    if depth==110:paper_deviations.append('warmup_switch_uses_unfiltered_minibatch_error')
    if DEVICE.type!='cuda':paper_deviations.append('CPU_not_original_two_GPU')
    else:paper_deviations.append('single_GPU_not_original_two_GPU')
    record={'family':'cifar10','kind':kind,'depth':depth,'seed':seed,
            'run_id':label,'paper_profile':'he2015_cifar10_64k_sgd',
            'dataset':{'name':'CIFAR10','origin':'real','classes':10,'train_count':50000,
                       'test_count':10000,'checksum_verified':True,'fingerprint_sha256':DATASET_FINGERPRINT},
            'steps_required':PAPER_STEPS,'steps_completed':until,
            'batch_size':GLOBAL_BATCH,'optimizer':'SGD(momentum=0.9,weight_decay=1e-4)',
            'scheduler':'step_32000_48000','augmentation':'pad4_crop32_hflip',
            'shortcut':'none' if kind=='plain' else 'A',
            'device':str(DEVICE),'torch_version':torch.__version__,
            'amp_enabled':USE_AMP,'paper_deviations':paper_deviations,
            'warmup_end':warmup_end,'checkpoint_sha256':sha256_file(ckpt),
            **eval_result}
    donefile.write_text(json.dumps(record,indent=2,ensure_ascii=False,default=str))
    print('REAL DATA EVALUATION COMPLETE:',label,'accuracy',record['accuracy_top1'], 'test N=',record['test_samples_evaluated'])
    return until-before,record
'''),
 c('''# 每次 Colab 会话只推进当前未完成的模型，避免伪称全部论文实验已执行。
remaining = SESSION_STEP_BUDGET
newly_completed = []
for kind,depth,seed in PLAN:
    if remaining is not None and remaining<=0:break
    spent, complete = train_one_full_experiment(kind,depth,seed,remaining)
    if complete is not None:newly_completed.append(complete)
    if remaining is not None:remaining-=spent
    if spent>0 and (ARTIFACTS/(run_filename(kind,depth,seed)+'.pt')).exists():
        print('Resumable progress saved. Rerun all cells next session to continue.')
print('Newly completed real experiments:',len(newly_completed), 'NOT overall paper reproduction yet.')
'''),
 m('''## 5｜严格论文指标覆盖检查与图表

以下只从**已经完成并保存的真实训练**读取指标。不会编造未完成模型的测试误差。表 6 给出的 ResNet-20/32/44/56/110/1202 原文错误率依次为 8.75、7.51、7.17、6.97、6.43、7.93%。数字用于对照，而非 Notebook 事先填充的运行结果。'''),
 c('''from collections import Counter
completed=[]
for fp in sorted(ARTIFACTS.glob('*.json')):
    try: row=json.loads(fp.read_text())
    except Exception:continue
    if row.get('family')=='cifar10' and row.get('steps_completed')==64000:
        completed.append(row)
planned={(k,d,s) for k,d,s in PLAN}
recorded={(r['kind'],r['depth'],r['seed']) for r in completed}
print('PAPER EXPERIMENT STATUS: completed',len(recorded&planned),'of',len(planned))
for k,d,s in PLAN:
    match=next((r for r in completed if (r['kind'],r['depth'],r['seed'])==(k,d,s)),None)
    print(f'{k:6} depth={d:4d} seed={s}:',
          f'test error={100*(1-match["accuracy_top1"]):.3f}%' if match else 'PENDING, no metric')
paper_error={20:8.75,32:7.51,44:7.17,56:6.97,110:6.43,1202:7.93}
res=[r for r in completed if r['kind']=='resnet' and r['seed']==1337]
if res:
    fig,ax=plt.subplots(figsize=(9,4))
    xs=[r['depth'] for r in res];actual=[(1-r['accuracy_top1'])*100 for r in res]
    ax.plot(xs,actual,'o-',label='our REAL CIFAR-10 test error (completed runs)')
    ax.plot(list(paper_error),list(paper_error.values()),'s--',label='paper Table 6 (reference)')
    ax.set(xlabel='ResNet depth',ylabel='Test error (%)',title='Only completed real-data runs are plotted');ax.legend();ax.grid(alpha=.2)
    plt.show()
else:
    print('No completed real-data results to plot. This is protocol_ready, NOT a reproduced result.')
'''),
 c('''# 保存自证性实验记录：MCP/CLI 可检查覆盖范围，但不能代替独立科学审计。
manifest_path=ARTIFACTS/'paper_evidence.json'
manifest={'paper':'arxiv:1512.03385','scope':'cifar10','runs':completed,
          'required_experiments':len(PLAN),'completed_experiments':len(recorded&planned),
          'status':'cifar10_protocol_complete' if planned.issubset(recorded) else 'real_dataset_partial_or_protocol_ready',
          'warning':'ImageNet and VOC/COCO are separate requirements. Never claim entire-paper reproduction from CIFAR alone.'}
manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str))
print('Evidence manifest:',manifest_path)
print('CURRENT STATUS:',manifest['status'])
assert manifest['status']!='paper_fully_reproduced'
'''),
 m('''## 6｜科学审计结论

- 本 Notebook 含有可在 Colab GPU/CPU 上运行的**真实 CIFAR-10** 完整训练协议；它不包含事先捏造的训练指标。
- `paper_evidence.json` 一旦具备相应真实记录，可用 `nbuild reproduce-quality <...> --scope cifar10_paper` 检验全部模型覆盖；对单一完整模型可用 `--scope cifar10_single`。
- 即使 CIFAR 全矩阵完成，论文还包含 **ImageNet-1k 的 1.28M/50k 分类实验及 VOC/COCO Faster R-CNN 对照**。只有这三个实验家族全部有符合要求的证据并经外部复核，才能讨论整篇论文复现成功。
- 实验报告必须列出偏差：单卡 vs 双卡、框架版本、CIFAR-110 热身切换判据、可能的数值非确定性、数据集/权重 SHA-256；这些条件不应被自动隐藏。''')
]


def build(out: Path, *, overwrite=False)->dict:
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    spec=out.with_suffix('.spec.json')
    spec.write_text(json.dumps({'title':TITLE,'cells':CELLS,'reproduction':{'stage':'protocol_ready','paper':'arxiv:1512.03385','scope':'cifar10_paper','dataset':'CIFAR10','fidelity':'original_cifar_protocol_with_declared_deviations'},'research_status':'protocol_ready',
                                'paper':'arxiv:1512.03385','dataset':'CIFAR10_real_official'},
                               ensure_ascii=False,indent=2),encoding='utf-8')
    return builder.from_spec(out,spec,overwrite=overwrite)

if __name__=='__main__':
    dst=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/mnt/data/resnet-paper-real-reproduction/ResNet_2015_CIFAR10_FULL_PAPER_Colab.ipynb')
    print(build(dst,overwrite='--force' in sys.argv))
