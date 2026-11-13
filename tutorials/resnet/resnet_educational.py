"""ResNet-v1 reference architecture for the executable arXiv:1512.03385 tutorial.

Original-paper behavior: post-add ReLU and bottleneck stride in the first 1x1 conv.
Supports ImageNet 18/34/50/101/152 and CIFAR-10 6n+2, with shortcut A/B.
No trained weights, downloads, or pretrained-performance claims.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def conv3x3(inp:int, out:int, stride:int=1):
    return nn.Conv2d(inp, out, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(inp:int, out:int, stride:int=1):
    return nn.Conv2d(inp, out, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes:int, planes:int, stride:int=1, downsample=None, use_shortcut:bool=True):
        super().__init__()
        self.conv1=conv3x3(inplanes,planes,stride)
        self.bn1=nn.BatchNorm2d(planes)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=conv3x3(planes,planes)
        self.bn2=nn.BatchNorm2d(planes)
        self.downsample=downsample
        self.use_shortcut=use_shortcut
    def forward(self,x):
        identity=x
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        if self.downsample is not None:identity=self.downsample(x)
        out=out+identity if self.use_shortcut else out
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,inplanes:int,planes:int,stride:int=1,downsample=None,use_shortcut:bool=True):
        super().__init__()
        # Original ResNet-v1 stride is on the first 1x1 (not torchvision's v1.5 3x3).
        self.conv1=conv1x1(inplanes,planes,stride)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2=conv3x3(planes,planes)
        self.bn2=nn.BatchNorm2d(planes)
        self.conv3=conv1x1(planes,planes*self.expansion)
        self.bn3=nn.BatchNorm2d(planes*self.expansion)
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
        self.use_shortcut=use_shortcut
    def forward(self,x):
        identity=x
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.relu(self.bn2(self.conv2(out)))
        out=self.bn3(self.conv3(out))
        if self.downsample is not None:identity=self.downsample(x)
        return self.relu(out+identity if self.use_shortcut else out)


class ImageNetResNet(nn.Module):
    def __init__(self,block,layers, num_classes:int=1000, zero_init_residual:bool=False, plain:bool=False):
        super().__init__()
        self.inplanes=64
        self.plain=plain
        self.conv1=nn.Conv2d(3,64,7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(3,stride=2,padding=1)
        self.layer1=self._make_layer(block,64,layers[0],stride=1)
        self.layer2=self._make_layer(block,128,layers[1],stride=2)
        self.layer3=self._make_layer(block,256,layers[2],stride=2)
        self.layer4=self._make_layer(block,512,layers[3],stride=2)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512*block.expansion,num_classes)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight);nn.init.zeros_(m.bias)
        if zero_init_residual:
            # Optional modern training enhancement, NOT claimed to be original 2015 recipe.
            for m in self.modules():
                if isinstance(m,Bottleneck):nn.init.zeros_(m.bn3.weight)
                elif isinstance(m,BasicBlock):nn.init.zeros_(m.bn2.weight)
    def _make_layer(self,block,planes,blocks,stride=1):
        outplanes=planes*block.expansion
        downsample=None
        if not self.plain and (stride!=1 or self.inplanes!=outplanes):
            downsample=nn.Sequential(conv1x1(self.inplanes,outplanes,stride),nn.BatchNorm2d(outplanes))
        seq=[block(self.inplanes,planes,stride,downsample,use_shortcut=not self.plain)]
        self.inplanes=outplanes
        seq.extend(block(self.inplanes,planes,use_shortcut=not self.plain) for _ in range(1,blocks))
        return nn.Sequential(*seq)
    def forward_features(self,x):
        x=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for layer in [self.layer1,self.layer2,self.layer3,self.layer4]:x=layer(x)
        return x
    def forward(self,x):
        x=self.forward_features(x)
        return self.fc(torch.flatten(self.avgpool(x),1))

RESNET_SPECS = {'18':(BasicBlock,[2,2,2,2]),'34':(BasicBlock,[3,4,6,3]),
                '50':(Bottleneck,[3,4,6,3]),'101':(Bottleneck,[3,4,23,3]),
                '152':(Bottleneck,[3,8,36,3])}

def imagenet_resnet(depth:int=18,num_classes:int=1000,**kwargs):
    if str(depth) not in RESNET_SPECS:raise ValueError(f'Unsupported depth {depth}')
    block, layers=RESNET_SPECS[str(depth)]
    return ImageNetResNet(block,layers,num_classes=num_classes,**kwargs)


class OptionAShortcut(nn.Module):
    """Original CIFAR Option A: stride by spatial subsampling and zero-pad channels."""
    def __init__(self,inplanes:int,outplanes:int,stride:int):
        super().__init__()
        self.inplanes,self.outplanes,self.stride=inplanes,outplanes,stride
        if outplanes<inplanes:raise ValueError('Option A shortcut does not reduce channels')
    def forward(self,x):
        if self.stride!=1:x=x[:,:,::self.stride,::self.stride]
        delta=self.outplanes-x.shape[1]
        if delta:
            before=delta//2
            # Padding the channel dimension by zero tensors. F.pad's channel axis is -3.
            x=F.pad(x,(0,0,0,0,before,delta-before))
        return x


class CIFARBasicBlock(nn.Module):
    expansion=1
    def __init__(self,inplanes,planes,stride=1,shortcut='A'):
        super().__init__()
        self.conv1=conv3x3(inplanes,planes,stride)
        self.bn1=nn.BatchNorm2d(planes)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=conv3x3(planes,planes)
        self.bn2=nn.BatchNorm2d(planes)
        if shortcut=='none':self.shortcut=None
        elif stride==1 and inplanes==planes:self.shortcut=nn.Identity()
        elif shortcut=='A':self.shortcut=OptionAShortcut(inplanes,planes,stride)
        elif shortcut=='B':self.shortcut=nn.Sequential(conv1x1(inplanes,planes,stride),nn.BatchNorm2d(planes))
        elif shortcut=='none':self.shortcut=None  # paper plain-network ablation
        else:raise ValueError('shortcut must be A, B or none')
    def forward(self,x):
        y=self.relu(self.bn1(self.conv1(x)))
        y=self.bn2(self.conv2(y))
        return self.relu(y if self.shortcut is None else y+self.shortcut(x))


class CIFARResNet(nn.Module):
    """Paper CIFAR 6n+2: 3x3 stem and three stage widths 16/32/64.
    base_width=16 reproduces nominal paper network; smaller width is educational.
    """
    def __init__(self,n:int=3,num_classes:int=10,shortcut='A',base_width:int=16):
        super().__init__()
        if n<1:raise ValueError('n must be positive')
        self.n=n;self.depth=6*n+2
        self.stem=nn.Sequential(conv3x3(3,base_width),nn.BatchNorm2d(base_width),nn.ReLU(inplace=True))
        self.stages=nn.Sequential(*self._make_stages(n,shortcut,base_width))
        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(base_width*4,num_classes)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):nn.init.ones_(m.weight);nn.init.zeros_(m.bias)
    def _make_stages(self,n,shortcut,base_width):
        previous=base_width; stages=[]
        for stage_idx in range(3):
            width=base_width*2**stage_idx
            units=[]
            for block_idx in range(n):
                stride=2 if stage_idx>0 and block_idx==0 else 1
                units.append(CIFARBasicBlock(previous,width,stride,shortcut))
                previous=width
            stages.append(nn.Sequential(*units))
        return stages
    def forward(self,x):
        y=self.stages(self.stem(x))
        return self.fc(torch.flatten(self.pool(y),1))


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def topk_accuracy(logits, targets, k:int=1):
    if not 1<=k<=logits.shape[1]:raise ValueError('Invalid k')
    best=logits.topk(k,dim=1).indices
    return (best==targets[:,None]).any(dim=1).float().mean().item()
