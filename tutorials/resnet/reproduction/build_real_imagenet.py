"""Generate real ImageNet 2012 paper-scale classification protocol (unexecuted)."""
from __future__ import annotations
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import notebook_builder as core
from build_real_cifar import c,m,MODEL_SOURCE

CELLS=[
m('''# ImageNet-1k / ILSVRC2012：ResNet 原论文分类实验（真实数据，尚未运行）

**论文** arXiv:1512.03385 §3.4, §4.1, Tables 1–5。这个 Notebook 旨在提供真实 ImageNet 分类实验的**可执行基础设施**，初始状态为 `protocol_ready`。没有授权数据、已完成 600k 步的模型和全部 50k 验证集评估，就**不能声称复现**。

**原始要求**：1.28M train、50k val、1000 类；Plain-18/34 对 ResNet-18/34 与 ResNet-50/101/152；SGD 全局批量 256、momentum 0.9、weight decay 1e-4、从头训练（最多 600k iterations）；224×224 随机裁剪、短边随机缩放 256–480、水平翻转、原论文 AlexNet 风格 color PCA；10-crop 与多尺度评估。论文未提供所有随机种子与触发 plateau 的精确迭代，因此**完全 bitwise 复现是不现实的**。

此代码为单机 Colab 方案（梯度累积 + 现代 PyTorch），**与论文多 GPU BatchNorm / 色彩增强 / 学习率平台判据存在可记录偏差**。普通 Colab 单 GPU 无法合理完成 ImageNet 的所有架构；可迁移至多 GPU 环境。严格科学复现审计会对这些偏差拒绝“精确协议完成”。

**用户须自行合法准备** ILSVRC2012 train/val 文件，且已按同样 1000 个类名分目录存储 `train/<class>/image.jpg` 和 `val/<class>/image.jpg`。该 Notebook **不会**抓取第三方未授权 ImageNet 镜像，且缺失数据会报错。'''),
c('''import os,sys,hashlib,time,random,json
from pathlib import Path
import numpy as np
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
IN_COLAB=('google.colab' in sys.modules or 'COLAB_RELEASE_TAG' in os.environ)
DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_ROOT=Path('/content/drive/MyDrive/Datasets/ILSVRC2012')
    ARTIFACTS=Path('/content/drive/MyDrive/Notebook-Builder/ResNet-2015/ImageNet')
else:
    DATA_ROOT=Path('./data/ILSVRC2012');ARTIFACTS=Path('./artifacts/imagenet-paper')
ARTIFACTS.mkdir(parents=True,exist_ok=True)
PLAN=[('plain',18),('plain',34),('resnet',18),('resnet',34),('resnet',50),('resnet',101),('resnet',152)]
PAPER_STEPS=600000
GLOBAL_BATCH=256
MICRO_BATCH=8 # on large GPU set 256 to avoid gradient accumulation (BN differences)
assert GLOBAL_BATCH%MICRO_BATCH==0
SESSION_STEP_BUDGET=100
CHECKPOINT_EVERY=50
USE_AMP=False  # off for original numerical comparison
LR_MILESTONES=(200000,400000) # explicit approximation; paper says decay when plateau without exact milestones
print('device',DEVICE,'gpu',torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU')
print('The original paper needs millions of images and extensive multi-GPU training. This code has NOT executed it.')
'''),
m('''## 1｜原版 ImageNet ResNet-v1 结构（Bottleneck stride 在第一个 1×1）与 Plain Network'''),
c(MODEL_SOURCE),
c('''def make_imagenet_model(kind,depth):
    if kind=='plain' and depth not in (18,34):raise ValueError('Paper uses plain 18/34 only')
    if kind not in ('plain','resnet'):raise ValueError('Unknown kind')
    return imagenet_resnet(depth,num_classes=1000,plain=kind=='plain')
for kind,depth in PLAN:
    with torch.device('meta'):
        model=make_imagenet_model(kind,depth)
    print(kind,depth,'parameters',parameter_count(model))
'''),
m('''## 2｜加载有授权的真实 ImageNet；运行前检查目录与样本总数

下例使用官方数据集解包后的 ImageFolder。目录 fingerprint 为文件名/文件大小的 SHA256，**不是**数据内容校验；如需完整性认证，必须独立校验来源于原始 ILSVRC 的 tar 文件或逐文件哈希。不要手动把 `checksum_verified` 改为 True 来骗过门禁。'''),
c('''TRAIN_ROOT=DATA_ROOT/'train';VAL_ROOT=DATA_ROOT/'val'
if not TRAIN_ROOT.is_dir() or not VAL_ROOT.is_dir():
    raise FileNotFoundError(f'Requires licensed ILSVRC2012 train/val class folders: {DATA_ROOT}')
class PaperTrainTransform:
    def __call__(self,img):
        sz=random.randint(256,480)
        img=TF.resize(img,sz,interpolation=transforms.InterpolationMode.BILINEAR)
        img=transforms.RandomCrop(224)(img)
        img=transforms.RandomHorizontalFlip(p=.5)(img)
        # Approximation; original uses AlexNet PCA color augmentation, not simple brightness jitter.
        img=transforms.ColorJitter(brightness=.2,contrast=.2,saturation=.2)(img)
        return TF.normalize(TF.to_tensor(img),[.485,.456,.406],[.229,.224,.225])
train_ds=ImageFolder(str(TRAIN_ROOT), transform=PaperTrainTransform())
val_ds=ImageFolder(str(VAL_ROOT),transform=transforms.Compose([
    transforms.Resize(256),transforms.TenCrop(224),
    transforms.Lambda(lambda crops:torch.stack([TF.normalize(TF.to_tensor(c),[.485,.456,.406],[.229,.224,.225]) for c in crops]))]))
assert len(train_ds)>=1200000,('Expected ImageNet-1k train, found',len(train_ds))
assert len(val_ds)==50000,('Expected 50k ImageNet val, found',len(val_ds))
assert len(train_ds.classes)==len(val_ds.classes)==1000
assert train_ds.class_to_idx==val_ds.class_to_idx
metadata_hash=hashlib.sha256()
for path,label in train_ds.samples+val_ds.samples:
    p=Path(path)
    metadata_hash.update((str(p.relative_to(DATA_ROOT))+'|'+str(p.stat().st_size)+'\n').encode())
DATASET_FINGERPRINT=metadata_hash.hexdigest()
print('REAL ImageNet loaded',len(train_ds),len(val_ds),'1000 classes')
print('Metadata-only fingerprint (not certified SHA256 content):',DATASET_FINGERPRINT)
'''),
m('''## 3｜数据级别的训练 / 验证 / 增广示例（必须使用真实图片）'''),
c('''fig,axes=plt.subplots(2,4,figsize=(11,5))
for i,ax in enumerate(axes.flat):
    x,y=train_ds[i]
    img=x.cpu()*torch.tensor([.229,.224,.225])[:,None,None]+torch.tensor([.485,.456,.406])[:,None,None]
    ax.imshow(img.clamp(0,1).permute(1,2,0));ax.axis('off');ax.set_title(train_ds.classes[y],fontsize=9)
plt.suptitle('Actual licensed ImageNet training images');plt.tight_layout();plt.show()
'''),
m('''## 4｜实际训练、断点恢复、50k 验证与 10-crop top-1/top-5

对单卡 T4，请只运行少量步数验证管线；完整 600k/模型需要集群与大量时间。**累积批量 256 不等价于原论文每卡实际 BN 批量，故强制标记为实现偏差。** ImageNet 源数据来源认证缺失时不会授予严格复现状态。'''),
c('''def imagenet_lr(step):
    return .1/(10**sum(step>=s for s in LR_MILESTONES))

def sha256_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

@torch.no_grad()
def evaluate_tencrop(model,batch_size=2):
    model.eval();n=0;top1=0;top5=0
    for begin in range(0,len(val_ds),batch_size):
        bx,by=zip(*(val_ds[i] for i in range(begin,min(begin+batch_size,len(val_ds)))))
        batch=torch.stack(bx).to(DEVICE);targets=torch.tensor(by,device=DEVICE)
        b,c,channels,h,w=batch.shape
        pred=model(batch.reshape(b*c,channels,h,w)).reshape(b,c,1000).mean(1)
        top1+=(pred.argmax(1)==targets).sum().item()
        top5+=(pred.topk(5,dim=-1).indices==targets[:,None]).any(1).sum().item()
        n+=b
        if (begin+batch_size)%1000==0:print('REAL val images evaluated',n,'/',len(val_ds),flush=True)
    assert n==50000
    return {'accuracy_top1':top1/n,'accuracy_top5':top5/n,'test_samples_evaluated':n}

def full_imagenet_run(kind,depth,seed=1337,budget_steps=100):
    tag=f'{kind}_{depth}_seed{seed}';ckpt=ARTIFACTS/(tag+'.pt');done=ARTIFACTS/(tag+'.json')
    if done.exists():return 0,json.loads(done.read_text())
    random.seed(seed);torch.manual_seed(seed)
    model=make_imagenet_model(kind,depth).to(DEVICE)
    optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=1e-4)
    g=torch.Generator(device='cpu').manual_seed(seed+2015)
    # Statefully shuffled epochs: allows exact sample-order continuation across sessions (workers=0).
    perm=torch.randperm(len(train_ds),generator=g);cursor=0;step=0;history=[]
    if ckpt.exists():
        state=torch.load(ckpt,map_location='cpu',weights_only=False) # trusted run artifacts only
        assert (state['kind'],state['depth'],state['seed'])==(kind,depth,seed)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        g.set_state(state['g_state']);perm=state['perm'];cursor=state['cursor'];step=state['step'];history=state['history']
        random.setstate(state['py_rng']);torch.set_rng_state(state['torch_rng'])
        if DEVICE.type=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
    def next_indices(n):
        nonlocal perm,cursor
        inds=[]
        while len(inds)<n:
            if cursor==len(perm):perm=torch.randperm(len(train_ds),generator=g);cursor=0
            take=min(n-len(inds),len(perm)-cursor)
            inds.extend(perm[cursor:cursor+take].tolist());cursor+=take
        return inds
    start=step;end=min(PAPER_STEPS,step+budget_steps)
    for step in range(start,end):
        model.train()
        for pg in optimizer.param_groups:pg['lr']=imagenet_lr(step)
        optimizer.zero_grad(set_to_none=True)
        total_loss=0.
        for micro in range(GLOBAL_BATCH//MICRO_BATCH):
            idx=next_indices(MICRO_BATCH)
            batch=torch.stack([train_ds[i][0] for i in idx]).to(DEVICE)
            targets=torch.tensor([train_ds.samples[i][1] for i in idx],dtype=torch.long,device=DEVICE)
            logits=model(batch)
            loss=F.cross_entropy(logits,targets)/(GLOBAL_BATCH//MICRO_BATCH)
            loss.backward();total_loss+=loss.detach().item()
        optimizer.step()
        if (step+1)%25==0:
            history.append({'step':step+1,'batch_loss':total_loss,'lr':imagenet_lr(step)})
            print(tag,step+1,'/600000','mini-batch loss',round(total_loss,4),flush=True)
        if (step+1)%CHECKPOINT_EVERY==0 or (step+1)==end:
            state={'kind':kind,'depth':depth,'seed':seed,'model':model.state_dict(),
                   'optimizer':optimizer.state_dict(),'g_state':g.get_state(),'perm':perm,'cursor':cursor,
                   'step':step+1,'history':history,'py_rng':random.getstate(),
                   'torch_rng':torch.get_rng_state(),
                   'cuda_rng':torch.cuda.get_rng_state_all() if DEVICE.type=='cuda' else None}
            temp=ckpt.with_suffix('.pt.tmp');torch.save(state,temp);os.replace(temp,ckpt)
    if end<PAPER_STEPS:
        print('PARTIAL TRUE IMAGENET RUN:',tag,end,'steps; cannot claim paper reproduced')
        return end-start,None
    metrics=evaluate_tencrop(model)
    record=dict(family='imagenet',kind=kind,depth=depth,seed=seed,
                paper_profile='he2015_imagenet_600k_sgd',
                dataset=dict(name='ImageNet-1k/ILSVRC2012',origin='real',classes=1000,
                             train_count=len(train_ds),test_count=len(val_ds),
                             fingerprint_sha256=DATASET_FINGERPRINT,
                             checksum_verified=False), # metadata-only cannot establish archive authenticity
                steps_required=PAPER_STEPS,steps_completed=end,batch_size=GLOBAL_BATCH,
                optimizer='SGD(momentum=0.9,weight_decay=1e-4)',scheduler='declared_plateau_approximation',
                checkpoint_sha256=sha256_file(ckpt),
                paper_deviations=['approx_color_pca','gradient_accumulation_bn' if MICRO_BATCH!=256 else 'single_machine_bn',
                                  'approx_plateau_milestones','ImageFolder_metadata_hash_is_not_official_integrity_proof'],
                **metrics)
    done.write_text(json.dumps(record,indent=2,default=str))
    return end-start,record
'''),
c('''spent_budget=SESSION_STEP_BUDGET
for kind,depth in PLAN:
    if spent_budget<=0:break
    used,result=full_imagenet_run(kind,depth,budget_steps=spent_budget)
    spent_budget-=used
print('This stage is real ImageNet execution, NOT paper result certification.')
'''),
m('''## 5｜研究证据与未解决的原论文偏差

必须在 7 个模型每个 600k 步与 50k ImageNet 验证集评估完毕后，才能说 ImageNet 部分完成。**即使完成训练，现代化近似（彩色 PCA、BN 微批次、随机调度等）也必须列入差异记录，不应自动标记严格论文复现**。ImageNet 的实际官方数据来源授权与校验由操作者负责。论文还包含 VOC/COCO 检测，不能单凭 ImageNet 和 CIFAR 认定全篇论文完成。'''),
c('''completed=[]
for kind,depth in PLAN:
    p=ARTIFACTS/(f'{kind}_{depth}_seed1337.json')
    if p.exists():completed.append(json.loads(p.read_text()))
print('REAL ImageNet complete:',len(completed),'/',len(PLAN))
manifest={'paper':'arxiv:1512.03385','scope':'imagenet','runs':completed,
          'status':'protocol_experiments_finished_with_deviations' if len(completed)==len(PLAN) else 'partial_or_protocol_ready'}
path=ARTIFACTS/'paper_evidence_imagenet.json';path.write_text(json.dumps(manifest,indent=2,default=str))
print('evidence:',path,'status:',manifest['status'])
''')
]

def build(out,force=False):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    spec=out.with_suffix('.spec.json')
    spec.write_text(json.dumps({'title':'ResNet2015: Real ImageNet ILSVRC2012 Classification Protocol','cells':CELLS,
                                'reproduction':{'stage':'protocol_ready','paper':'arxiv:1512.03385','scope':'imagenet_paper','dataset':'ImageNet-1k/ILSVRC2012','fidelity':'modern_single_machine_approximation'},'research_status':'protocol_ready','paper':'arxiv:1512.03385'},indent=2,ensure_ascii=False))
    return core.from_spec(out,spec,overwrite=force)

if __name__=='__main__':
    out=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/mnt/data/resnet-paper-real-reproduction/ResNet_2015_ImageNet_FULL_Protocol_Colab.ipynb')
    print(build(out,force='--force' in sys.argv))
