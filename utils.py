import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import functools
import numpy as np
import cv2

#MIoU
def mIoU(p1,p2):
    if len(p1.shape)>2:
        p1=torch.argmax(p1,1)[0]
    intersection=((p1+p2)==2).sum()
    unionsection=torch.logical_or(p1,p2).sum()
    #if unionsection<1:return 1
    return (intersection/unionsection).item()

#Boundary
def Boundary(p):
    if len(p.shape)>2:
        p=torch.argmax(p,1)[0].cpu().numpy().astype('float64')
    else:p=p.cpu().numpy()
    #print(p.shape)
    h,w=p.shape
    dil=int(round(0.02*np.sqrt(h ** 2 + w ** 2)))
    if dil<1:dil=1
    p_=cv2.copyMakeBorder(p, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel=np.ones((3, 3), dtype=np.uint8)
    p_erode=cv2.erode(p_,kernel,iterations=dil)
    return torch.from_numpy(p-p_erode[1 : h + 1, 1 : w + 1])

#Boundary mIoU
def BIoU(p1,p2):
    p1,p2=Boundary(p1),Boundary(p2)
    intersection=((p1+p2)==2).sum()
    unionsection=torch.logical_or(p1,p2).sum()
    if unionsection<1:return 1
    return (intersection/unionsection).item()

#Assimilate tensors
def assim(t1,t2):
    shape=t1.shape[-2:]
    t2=F.interpolate(t2,shape)
    # shape=np.array(t1.shape)-np.array(t2.shape)
    # if shape.sum()>0:t2=F.pad(t2,list(shape),"constant",0)
    # elif shape.sum()<0:t1=F.pad(t1,list(-shape),"constant",0)
    return t1,t2

#Swish from Google
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

#MaxPooling with Same Padding
class PSP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size
        #print(self.stride,self.pool);input()
        if isinstance(self.stride, int):
            self.stride = [self.stride]*2
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
    def forward(self, x):
        h,w=x.shape[-2:]
        pad_w= (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        pad_h= (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        return self.pool(F.pad(x,[pad_w//2,pad_w-pad_w//2,pad_h//2,pad_h-pad_h//2]))

#Convolution with Same Padding
class CSP(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, bias=True,
                 groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,
                              kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride=self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation
    def forward(self, x):
        h,w=x.shape[-2:]
        pad_w= (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        pad_h= (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        return self.conv(F.pad(x,[pad_w//2,pad_w-pad_w//2,pad_h//2,pad_h-pad_h//2]))

#Depthwise Seperable Convolution Block
class DSCB(nn.Module):
    def __init__(self, in_channels,out_channels=None):
        """
        in_channels: number of input channels
        out_channels: number of output channels
        """
        super(DSCB, self).__init__()
        if out_channels is None:out_channels = in_channels
        # groups=in_channels
        self.depthwise=CSP(in_channels, in_channels,kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise=CSP(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)
    def forward(self, x):
        x = self.pointwise(self.depthwise(x))
        return self.bn(x)
class DSCB_(nn.Module):
    def __init__(self, in_channels,out_channels=None):
        """
        in_channels: number of input channels
        out_channels: number of output channels
        """
        super(DSCB_, self).__init__()
        if out_channels is None:out_channels = in_channels
        # groups=in_channels
        self.depthwise=CSP(in_channels, in_channels,kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise=CSP(in_channels, out_channels, kernel_size=1, stride=1)
        #self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

#Dynamic or static padding
def SP(image_size=None):
    if image_size is None:return CDSP
    else:
        return functools.partial(CSP, image_size=image_size)

#Dynamic Convolution with Same Padding
class CDSP(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

#SEModule
class SEM(nn.Module):
    def __init__(self, in_channels, squeeze_ch):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, squeeze_ch, kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv2d(squeeze_ch,in_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
    def forward(self, x):
        return x * torch.sigmoid(self.se(x))

#MBconvolution layer
class MBConv(nn.Module):
    def __init__(self,in_channels, out_channels,expand,
                 kernel_size, stride,
                 se_ratio, dc_ratio=0.2,resi=True):
        super().__init__()
        mid=in_channels*expand
        self.expand=nn.Sequential(
            CSP(in_channels,mid,kernel_size),
            nn.BatchNorm2d(mid,eps=1e-3, momentum=0.01),
            Swish())
        self.dwc=DSCB(mid,mid)
        self.se=SEM(mid, int(in_channels * se_ratio)) if se_ratio > 0 else nn.Identity()
        self.proj=nn.Sequential(
            CSP(mid,out_channels,kernel_size),
            nn.BatchNorm2d(out_channels, 1e-3, 0.01))
        self.resi=resi and (stride == 1) and (in_channels==out_channels)
    def forward(self, inputs):
        x = self.expand(inputs)
        x = self.dwc(x)
        x = self.proj(self.se(x))
        if self.resi:
            x = x + inputs
        return x

#MBconvolution Block
class MBBlock(nn.Module):
    def __init__(self,in_channels, out_channels,
                 expand, kernel, stride,
                 num_repeat, skip, se_ratio,
                 drop_connect_ratio=0.2):
        super().__init__()
        layers = [MBConv(in_channels, out_channels,
                         expand, kernel,stride,
                         skip, se_ratio,
                         drop_connect_ratio)]
        for i in range(1, num_repeat):
            layers.append(MBConv(out_channels,out_channels, expand, kernel, 1, skip, se_ratio, drop_connect_ratio))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

# keep the batch shape
class flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0],-1)
