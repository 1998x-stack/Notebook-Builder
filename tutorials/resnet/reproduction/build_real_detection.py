"""Generate real VOC/COCO detection pipeline, explicitly modern approximation, not 2015 replication."""
from __future__ import annotations
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import notebook_builder as core
from build_real_cifar import c,m,MODEL_SOURCE
CELLS=[
m('''# ResNet 2015 真实 PASCAL VOC / COCO Faster R-CNN 研究流程

**论文** arXiv:1512.03385 §4.3, Appendix A, Tables 7–11。**本 Notebook 初始为未执行协议**，且使用 `torchvision` 的现代 Faster R-CNN 实现，和 2015 年的原始 RPN → Fast R-CNN 分阶段、RoI-pool-before-conv5 ResNet 头、冻结 BN、8GPU COCO 训练存在实质区别，因此**无论跑到什么指标，都不能据此宣称“完全复现原论文检测实验”**。

其价值：使用**真实** PASCAL VOC 2007+2012、COCO2014 数据，比较 ImageNet 预训练 VGG16 与 ResNet101 Faster R-CNN，真实地训练、产生 bounding boxes、在完整测试集计算 VOC07 AP@0.5 和官方 COCO mAP@[.5:.95]。不能用假数据、模拟指标、预加载检测器 checkpoint 冒充由本 Notebook 训练。

你必须准备以下已授权数据，并设置目录：
- `VOCdevkit/VOC2007` 和 `VOCdevkit/VOC2012`，含 JPEGImages、Annotations、ImageSets；VOC 2007 trainval + VOC 2012 trainval 用作训练，2007 test 用于评估。
- `COCO2014/train2014`、`val2014` 和 `annotations/instances_{train,val}2014.json`。
- ResNet101 自己训练出的 ImageNet 分类 checkpoint；不能用随机权重冒充预训练。VGG16 使用 torchvision 官方 ImageNet1K 权重，但须记录来源差异。

**特别提示**：原论文 COCO 使用 80k train/40k val，8 GPU，RPN 与 Fast R-CNN 各 240k + 80k 迭代。Colab 单卡的现代端到端 Detector 是有价值的真实实验，但不是 2015 完整原配方。'''),
c('''import os,sys,json,time,hashlib,random,collections
from pathlib import Path
import torch,torchvision
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.models import vgg16,VGG16_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.datasets import VOCDetection,CocoDetection
import matplotlib.pyplot as plt
DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IN_COLAB=('google.colab' in sys.modules or 'COLAB_RELEASE_TAG' in os.environ)
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_ROOT=Path('/content/drive/MyDrive/Datasets')
    ARTIFACTS=Path('/content/drive/MyDrive/Notebook-Builder/ResNet-2015/Detection')
    RESNET101_IMAGENET_CHECKPOINT=Path('/content/drive/MyDrive/Notebook-Builder/ResNet-2015/ImageNet/resnet_101_seed1337.pt')
else:
    DATA_ROOT=Path('./data');ARTIFACTS=Path('./artifacts/detection-real')
    RESNET101_IMAGENET_CHECKPOINT=Path('./artifacts/imagenet-paper/resnet_101_seed1337.pt')
ARTIFACTS.mkdir(parents=True,exist_ok=True)
TASK='voc' # change to 'coco' for COCO2014
TRAIN_STEPS_PER_SESSION=100
MAX_STEPS=320000 # modern end-to-end schedule, NOT the original separate 320k RPN+320k Fast R-CNN
print('device',DEVICE,'dataset',TASK,'modern detector = research extension, not exact 2015 paper')
'''),
m('''## 1｜严格加载真实 VOC/COCO、真实 boxes 和 labels

无法找到标注和图片时直接终止，绝不回退到生成图像或假框。VOC 官方评测使用 2007 test；COCO 评测使用 2014 val。'''),
c('''VOC_CLASSES=('aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable',
             'dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor')
class RealVOC(torch.utils.data.Dataset):
    def __init__(self,root,year,split):
        self.ds=VOCDetection(root=str(root),year=year,image_set=split,download=False)
    def __len__(self):return len(self.ds)
    def __getitem__(self,i):
        pil,raw=self.ds[i];ann=raw['annotation'];objects=ann.get('object',[])
        if isinstance(objects,dict):objects=[objects]
        boxes=[];labels=[];difficult=[]
        for obj in objects:
            b=obj['bndbox'];xy=[float(b[t]) for t in ('xmin','ymin','xmax','ymax')]
            if xy[2]<=xy[0] or xy[3]<=xy[1]:continue
            boxes.append(xy);labels.append(VOC_CLASSES.index(obj['name'])+1)
            difficult.append(int(obj.get('difficult',0)))
        target={'boxes':torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),
                'labels':torch.tensor(labels,dtype=torch.int64),
                'difficult':torch.tensor(difficult,dtype=torch.uint8),
                'image_id':torch.tensor([i],dtype=torch.int64)}
        return TF.to_tensor(pil),target

class RealCOCO(torch.utils.data.Dataset):
    def __init__(self,images,annotations):
        self.ds=CocoDetection(root=str(images),annFile=str(annotations))
        cat_ids=sorted(self.ds.coco.getCatIds())
        assert len(cat_ids)==80
        self.to_contiguous={v:i+1 for i,v in enumerate(cat_ids)}
        self.to_coco={i+1:v for i,v in enumerate(cat_ids)}
    def __len__(self):return len(self.ds)
    def __getitem__(self,i):
        pil,anns=self.ds[i];w,h=pil.size;boxes=[];labels=[];areas=[]
        for a in anns:
            if a.get('iscrowd',0):continue
            x,y,bw,bh=a['bbox'];x2=min(w,x+bw);y2=min(h,y+bh)
            if x2<=x or y2<=y:continue
            boxes.append((x,y,x2,y2));labels.append(self.to_contiguous[a['category_id']]);areas.append(a.get('area',bw*bh))
        return TF.to_tensor(pil),{'boxes':torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),
            'labels':torch.tensor(labels,dtype=torch.int64),'area':torch.tensor(areas,dtype=torch.float32),
            'image_id':torch.tensor([self.ds.ids[i]],dtype=torch.int64)}

if TASK=='voc':
    DATASET_TRAIN=torch.utils.data.ConcatDataset([RealVOC(DATA_ROOT,'2007','trainval'), RealVOC(DATA_ROOT,'2012','trainval')])
    DATASET_TEST=RealVOC(DATA_ROOT,'2007','test')
    assert len(DATASET_TRAIN)>20000 and len(DATASET_TEST)>4000
    NUM_CLASSES=21
elif TASK=='coco':
    folder=DATA_ROOT/'COCO2014'
    DATASET_TRAIN=RealCOCO(folder/'train2014',folder/'annotations/instances_train2014.json')
    DATASET_TEST=RealCOCO(folder/'val2014',folder/'annotations/instances_val2014.json')
    assert len(DATASET_TRAIN)>70000 and len(DATASET_TEST)>35000
    NUM_CLASSES=81
else:raise ValueError('Choose voc or coco; FakeData is not an option')
print('Verified REAL annotated datasets',TASK,'train',len(DATASET_TRAIN),'eval',len(DATASET_TEST))
'''),
m('''## 2｜原论文 ImageNet 预训练 ResNet101 与 VGG16 对照；现代 Faster R-CNN 实现差异'''),
c(MODEL_SOURCE),
c('''class VGGBackbone(nn.Module):
    out_channels=512
    def __init__(self):
        super().__init__()
        base=vgg16(weights=VGG16_Weights.IMAGENET1K_V1) # official pretrained weights, may download
        self.features=base.features[:-1] # keep output stride16
    def forward(self,x):return {'0':self.features(x)}

def make_detector(kind):
    if kind=='vgg16':
        backbone=VGGBackbone()
        anchors=AnchorGenerator(sizes=((128,256,512),),aspect_ratios=((.5,1.,2.),))
        pool=MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)
        model=FasterRCNN(backbone,num_classes=NUM_CLASSES,
                         rpn_anchor_generator=anchors,box_roi_pool=pool,min_size=600,max_size=1000)
    elif kind=='resnet101':
        if not RESNET101_IMAGENET_CHECKPOINT.is_file():
            raise FileNotFoundError('Requires your OWN trained ImageNet ResNet-101 checkpoint, not random pretraining')
        base=imagenet_resnet(101,num_classes=1000)
        state=torch.load(RESNET101_IMAGENET_CHECKPOINT,map_location='cpu',weights_only=False)
        base.load_state_dict(state['model'],strict=True)
        backbone=BackboneWithFPN(base,return_layers={'layer1':'0','layer2':'1','layer3':'2','layer4':'3'},
                                 in_channels_list=[256,512,1024,2048],out_channels=256)
        model=FasterRCNN(backbone,num_classes=NUM_CLASSES,min_size=600,max_size=1000)
    else:raise ValueError('kind must be vgg16 or resnet101')
    return model.to(DEVICE)

print('Available detector backbones: VGG16 and custom ImageNet-pretrained ResNet101')
print('Protocol differences: FPN, ROIAlign, modern unified detector training; not original paper exact detection')
'''),
m('''## 3｜真实框训练、可恢复 SGD、预训练证据与本地实验日志

不使用预先训练好的“检测”模型来冒充训练。与 2015 原文不同，这里采用现代端到端优化；记录 `paper_deviations` 并禁止严格原论文认证。'''),
c('''def collate(items):return tuple(zip(*items))
loader=torch.utils.data.DataLoader(DATASET_TRAIN,batch_size=1,shuffle=True,num_workers=2,collate_fn=collate)

def train_session(kind='resnet101'):
    model=make_detector(kind)
    optim=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.001,momentum=.9,weight_decay=.0001)
    path=ARTIFACTS/f'{TASK}_{kind}_modern_detector.pt'
    step=0;history=[]
    if path.exists():
        state=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);optim.load_state_dict(state['optimizer']);step=state['step'];history=state['history']
    iterator=iter(loader)
    for it in range(step,min(MAX_STEPS,step+TRAIN_STEPS_PER_SESSION)):
        try:imgs,targets=next(iterator)
        except StopIteration:iterator=iter(loader);imgs,targets=next(iterator)
        images=[x.to(DEVICE) for x in imgs]
        annotations=[{k:v.to(DEVICE) for k,v in t.items() if k!='difficult'} for t in targets]
        model.train();loss_dict=model(images,annotations)
        loss=sum(loss_dict.values());optim.zero_grad(set_to_none=True);loss.backward();optim.step()
        history.append({'step':it+1,'loss':float(loss.item())})
        if (it+1)%10==0:print(kind,TASK,'REAL annotated training steps',it+1,'loss',round(loss.item(),4),flush=True)
        if (it+1)%50==0 or (it+1)==min(MAX_STEPS,step+TRAIN_STEPS_PER_SESSION):
            tmp=path.with_suffix('.pt.tmp')
            torch.save({'step':it+1,'model':model.state_dict(),'optimizer':optim.state_dict(),'history':history,
                        'dataset':TASK,'kind':kind},tmp)
            os.replace(tmp,path)
    return model,history

# Choose one at a time to avoid 2 huge detectors concurrently in a free Colab session.
DETECTOR_KIND='resnet101' # or 'vgg16'
model,history=train_session(DETECTOR_KIND)
print('PARTIAL REAL DATA training is NOT a completed paper detection comparison')
'''),
m('''## 4｜实际数据预测与 VOC AP@0.5 / COCO 官方 mAP 计算

在正式评估前须确保每个候选检测器完成自己约定的完整训练预算。VOC2007 采用 11 点插值 AP；COCO 使用 pycocotools 的官方 COCOeval。**在未完成模型训练前不要把评估结果当作论文结论。**'''),
c('''@torch.no_grad()
def predict_entire_test(model):
    model.eval();records=[]
    for i in range(len(DATASET_TEST)):
        img,target=DATASET_TEST[i]
        pred=model([img.to(DEVICE)])[0]
        records.append({'image_id':int(target['image_id'].item()),
                        'boxes':pred['boxes'].cpu().numpy().tolist(),
                        'scores':pred['scores'].cpu().numpy().tolist(),
                        'labels':pred['labels'].cpu().numpy().tolist(),
                        'gt_boxes':target['boxes'].numpy().tolist(),
                        'gt_labels':target['labels'].numpy().tolist(),
                        'difficult':target.get('difficult',torch.zeros(len(target['labels']))).numpy().tolist()})
        if i%250==0:print('processed real annotated images',i,'/',len(DATASET_TEST),flush=True)
    assert len(records)==len(DATASET_TEST)
    return records

def voc07_map_at_50(rows):
    import numpy as np
    aps=[]
    for label in range(1,21):
        gt={};npos=0
        for row in rows:
            objs=[]
            for box,lab,diff in zip(row['gt_boxes'],row['gt_labels'],row['difficult']):
                if lab==label:
                    objs.append((np.array(box,dtype=float),bool(diff),False))
                    if not diff:npos+=1
            gt[row['image_id']]=objs
        predictions=[]
        for row in rows:
            predictions.extend((score,row['image_id'],np.array(box,dtype=float))
              for score,lab,box in zip(row['scores'],row['labels'],row['boxes']) if lab==label)
        predictions.sort(key=lambda z:z[0],reverse=True)
        tp=[];fp=[]
        for score,img_id,box in predictions:
            objs=gt[img_id];ov=[]
            for j,(ref,diff,used) in enumerate(objs):
                lt=np.maximum(ref[:2],box[:2]);rb=np.minimum(ref[2:],box[2:]);w=np.maximum(rb-lt,0)
                intersection=w[0]*w[1];area=lambda b:max(0,b[2]-b[0])*max(0,b[3]-b[1])
                ov.append(intersection/(area(ref)+area(box)-intersection+1e-12))
            if ov and max(ov)>=.5:
                j=int(np.argmax(ov));ref,diff,used=objs[j]
                if diff:continue
                if not used:tp.append(1);fp.append(0);objs[j]=(ref,diff,True)
                else:tp.append(0);fp.append(1)
            else:tp.append(0);fp.append(1)
        if npos==0:raise ValueError('Ground truth class missing')
        tp=np.cumsum(tp);fp=np.cumsum(fp)
        recall=tp/npos;precision=tp/np.maximum(1,tp+fp)
        aps.append(np.mean([max(precision[recall>=t],default=0) for t in np.arange(0,1.01,.1)]))
    return float(np.mean(aps)),aps

def coco_map(rows,dataset):
    from pycocotools.cocoeval import COCOeval
    predictions=[]
    for row in rows:
        for box,score,label in zip(row['boxes'],row['scores'],row['labels']):
            if score<.001:continue
            predictions.append({'image_id':row['image_id'],'category_id':dataset.to_coco[label],
                                'bbox':[box[0],box[1],box[2]-box[0],box[3]-box[1]],'score':score})
    if not predictions:raise ValueError('No detections returned')
    result=dataset.ds.coco.loadRes(predictions)
    ev=COCOeval(dataset.ds.coco,result,'bbox')
    ev.params.imgIds=list(dataset.ds.ids)
    ev.evaluate();ev.accumulate();ev.summarize()
    return {'map50_95':float(ev.stats[0]),'map50':float(ev.stats[1])}

# Final evaluation is opt-in, after training budget actually completed.
RUN_FULL_DETECTION_EVAL=False
if RUN_FULL_DETECTION_EVAL:
    saved=torch.load(ARTIFACTS/f'{TASK}_{DETECTOR_KIND}_modern_detector.pt',map_location='cpu',weights_only=False)
    assert saved['step']>=MAX_STEPS,'Train before evaluating complete dataset'
    rows=predict_entire_test(model)
    result={'map_voc07_50':voc07_map_at_50(rows)[0]} if TASK=='voc' else coco_map(rows,DATASET_TEST)
    print('REAL evaluation',TASK,DETECTOR_KIND,result)
else:
    print('Evaluation NOT RUN. Do not report fabricated mAP.')
'''),
m('''## 5｜不能由这份 Notebook 直接证明的内容

1. 原版 2015 Faster R-CNN 双阶段交替训练、RPN 300 proposals、RoI-pool-before-conv5、冻结 BN 与特殊检测头；2. 8 卡 COCO 2014 原论文 240k+80k RPN/FRCNN 训练；3. VGG16 和 ResNet101 相同完整设定下的公平对照；4. Box refinement、global context、多尺度、竞赛 ensemble；5. ImageNet 预训练来源与收敛。

如需要“**整篇论文已复现**”等级，必须分别复现并审计这些模块，单卡 Colab 下的现代替代实验只能标注 **real_data_modern_approximation**。''')
]

def build(out,force=False):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    spec=out.with_suffix('.spec.json')
    spec.write_text(json.dumps({'title':'ResNet2015 Detection: Real VOC/COCO Modern Approximation — Not Exact Reproduction',
                                'cells':CELLS,'reproduction':{'stage':'protocol_ready','paper':'arxiv:1512.03385','scope':'detection_paper','dataset':'VOC2007+2012_COCO2014','fidelity':'modern_approximation_not_paper_exact'},'research_status':'protocol_ready','fidelity':'modern_approximation'},indent=2,ensure_ascii=False))
    return core.from_spec(out,spec,overwrite=force)

if __name__=='__main__':
    out=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/mnt/data/resnet-paper-real-reproduction/ResNet_2015_VOC_COCO_Detection_RealData_Colab.ipynb')
    print(build(out,force='--force' in sys.argv))
