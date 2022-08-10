import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import math

# EfficientNet
class EN(nn.Module):
    def __init__(self,width_coeff,
                 depth_div=8,min_depth=None,
                 dropout_rate=0.2,drop_connect_rate=0.2,
                 num_classes=1000):
        super().__init__()
        self.drop_connect_rate=drop_connect_rate
        min_depth=min_depth or depth_div
        depth_coeff=width_coeff
        def upd_chnl(x):
            if not width_coeff:return x
            x *= width_coeff
            new_x=max(min_depth,int(x+depth_div/2)//depth_div*depth_div)
            if new_x < 0.9 * x:new_x+=depth_div
            return int(new_x)
        def upd_depth(x):
            return int(math.ceil(x*depth_coeff))
        self.stem=nn.Sequential(
            CSP(3,upd_chnl(32),stride=2,kernel_size=3),
            nn.BatchNorm2d(upd_chnl(32),eps=1e-3,momentum=0.01),
            Swish())
        self.blocks=nn.Sequential(
            MBBlock(upd_chnl(32),upd_chnl(16),1,3,1,upd_depth(1),True,0.25,drop_connect_rate),
            MBBlock(upd_chnl(16),upd_chnl(24),6,3,2,upd_depth(2),True,0.25,drop_connect_rate),
            MBBlock(upd_chnl(24),upd_chnl(40),6,5,2,upd_depth(2),True,0.25,drop_connect_rate),
            MBBlock(upd_chnl(40),upd_chnl(80),6,3,2,upd_depth(3),True,0.25,drop_connect_rate),
            MBBlock(upd_chnl(80),upd_chnl(112),6,5,1,upd_depth(3),True,0.25,drop_connect_rate),
            MBBlock(upd_chnl(112),upd_chnl(192),6,5,2,upd_depth(4),True,0.25,drop_connect_rate),
            MBBlock(upd_chnl(192),upd_chnl(320),6,3,1,upd_depth(1),True,0.25,drop_connect_rate)
        )
        self.pooling=PSP(2,2)
        return
    def forward(self,x):
        x=self.stem(x)
        feature_maps=[]
        last_x=None
        for idx, block in enumerate(self.blocks):
            x=block(x)
            if idx in [3,5]:x=self.pooling(x)
            if block.layers[0].dwc.depthwise.stride == [2, 2]:
                feature_maps.append(last_x)
            else:
                feature_maps.append(x)
            last_x=x
        return feature_maps[1:]
    def forward_(self,inputs):
        return self.head(self.blocks(self.stem(inputs)))

# BiFPN
class BiFPN(nn.Module):
    def __init__(self,num_channels,conv_channels,
                 if_first=False,attention=True):
        """
        num_channels:number of channels
        attention: if attention
        conv_channels:numbers of convolution channels
        if_first: whether the input comes directly from the efficientnet
        """
        super(BiFPN,self).__init__()
        self.epsilon =1e-4
        for i in range(3,7):
            # convolution
            exec("self.conv"+str(i)+"_up=DSCB(num_channels)")
            exec("self.conv"+str(i+1)+"_down=DSCB(num_channels)")
            # feature scaling
            exec("self.p"+str(i)+"_upsample=nn.Upsample(scale_factor=2,mode='nearest')")
            exec("self.p"+str(i+1)+"_downsample=PSP(3,2)")
        self.swish=Swish()
        self.if_first=if_first
        if self.if_first:
            self.p5_down_channel=nn.Sequential(
                CSP(conv_channels[2],num_channels,1),
                nn.BatchNorm2d(num_channels,momentum=0.01,eps=1e-3),)
            self.p4_down_channel=nn.Sequential(
                CSP(conv_channels[1],num_channels,1),
                nn.BatchNorm2d(num_channels,momentum=0.01,eps=1e-3),)
            self.p3_down_channel=nn.Sequential(
                CSP(conv_channels[0],num_channels,1),
                nn.BatchNorm2d(num_channels,momentum=0.01,eps=1e-3),)
            self.p5_to_p6=nn.Sequential(
                CSP(conv_channels[2],num_channels,1),
                nn.BatchNorm2d(num_channels,momentum=0.01,eps=1e-3),
                PSP(3,2))
            self.p6_to_p7=nn.Sequential(PSP(3,2))
            self.p4_down_channel_2=nn.Sequential(
                CSP(conv_channels[1],num_channels,1),
                nn.BatchNorm2d(num_channels,momentum=0.01,eps=1e-3))
            self.p5_down_channel_2=nn.Sequential(
                CSP(conv_channels[2],num_channels,1),
                nn.BatchNorm2d(num_channels,momentum=0.01,eps=1e-3))
        # weight initialization
        for i in range(3,7):
            exec("self.p"+str(i)+"_w1=nn.Parameter(torch.ones(2,dtype=torch.float32),requires_grad=True)")
            exec("self.p"+str(i)+"_w1_relu=nn.ReLU()")
            exec("self.p"+str(i+1)+"_w2=nn.Parameter(torch.ones(3,dtype=torch.float32),requires_grad=True)")
            exec("self.p"+str(i+1)+"_w2_relu=nn.ReLU()")
        self.p7_w2=nn.Parameter(torch.ones(2,dtype=torch.float32),requires_grad=True)
        self.p7_w2_relu=nn.ReLU()
        self.attention =attention
    def forward(self,inputs):
        if self.attention:
            if self.if_first:
                p3,p4,p5=inputs
                p6_in=self.p5_to_p6(p5)
                p7_in=self.p6_to_p7(p6_in)
                p3_in=self.p3_down_channel(p3)
                p4_in=self.p4_down_channel(p4)
                p5_in=self.p5_down_channel(p5)
            else:
                p3_in,p4_in,p5_in,p6_in,p7_in=inputs
            #for i in range(3,8):exec("print(p"+str(i)+"_in.shape)")
            # Weights for P6_0 and P7_0 to P6_1
            p6_w1=self.p6_w1_relu(self.p6_w1)
            weight=p6_w1 / (torch.sum(p6_w1,dim=0) + self.epsilon)
            # Connections for P6_0 and P7_0 to P6_1 respectively
            t1,t2=assim(p6_in,self.p6_upsample(p7_in))
            p6_up=self.conv6_up(self.swish(weight[0] *t1 + weight[1] *t2))
            # Weights for P5_0 and P6_1 to P5_1
            p5_w1=self.p5_w1_relu(self.p5_w1)
            weight=p5_w1 / (torch.sum(p5_w1,dim=0) + self.epsilon)
            # Connections for P5_0 and P6_1 to P5_1 respectively
            t1,t2=assim(p5_in,self.p5_upsample(p6_up))
            p5_up=self.conv5_up(self.swish(weight[0] *t1 + weight[1] *t2))
            # Weights for P4_0 and P5_1 to P4_1
            p4_w1=self.p4_w1_relu(self.p4_w1)
            weight=p4_w1 / (torch.sum(p4_w1,dim=0) + self.epsilon)
            # Connections for P4_0 and P5_1 to P4_1 respectively
            #print(p4_in.shape,self.p4_upsample(p5_up).shape,p5_up.shape)
            t1,t2=assim(p4_in,self.p4_upsample(p5_up))
            #print(t1.shape,t2.shape)
            p4_up=self.conv4_up(self.swish(weight[0] *t1 + weight[1] *t2))
            # Weights for P3_0 and P4_1 to P3_2
            p3_w1=self.p3_w1_relu(self.p3_w1)
            weight=p3_w1 / (torch.sum(p3_w1,dim=0) + self.epsilon)
            # Connections for P3_0 and P4_1 to P3_2 respectively
            t1,t2=assim(p3_in, self.p3_upsample(p4_up))
            p3_out=self.conv3_up(self.swish(weight[0] *t1 + weight[1] *t2))
            if self.if_first:
                p4_in=self.p4_down_channel_2(p4)
                p5_in=self.p5_down_channel_2(p5)
            # Weights for P4_0,P4_1 and P3_2 to P4_2
            p4_w2=self.p4_w2_relu(self.p4_w2)
            weight=p4_w2 / (torch.sum(p4_w2,dim=0) + self.epsilon)
            # Connections for P4_0,P4_1 and P3_2 to P4_2 respectively
            #print(p4_in.shape,p4_up.shape,self.p4_downsample(p3_out).shape)
            t2,t3=assim(p4_in,self.p4_downsample(p3_out))
            t1,t3=assim(p4_up,t3)
            p4_out=self.conv4_down(
                self.swish(weight[0] *t2 + weight[1] * t1 + weight[2] *t3))
            # Weights for P5_0,P5_1 and P4_2 to P5_2
            p5_w2=self.p5_w2_relu(self.p5_w2)
            weight=p5_w2 / (torch.sum(p5_w2,dim=0) + self.epsilon)
            # Connections for P5_0,P5_1 and P4_2 to P5_2 respectively
            t2,t3=assim(p5_in, self.p5_downsample(p4_out))
            t1,t3=assim(p5_up,t3)
            p5_out=self.conv5_down(
                self.swish(weight[0] *t2 + weight[1] * t1+ weight[2] *t3))
            # Weights for P6_0,P6_1 and P5_2 to P6_2
            p6_w2=self.p6_w2_relu(self.p6_w2)
            weight=p6_w2 / (torch.sum(p6_w2,dim=0) + self.epsilon)
            # Connections for P6_0,P6_1 and P5_2 to P6_2 respectively
            t2,t3=assim(p6_in, self.p6_downsample(p5_out))
            t1,t3=assim(p6_up,t3)
            p6_out=self.conv6_down(
                self.swish(weight[0] *t2 + weight[1] *t1 + weight[2] *t3))
            # Weights for P7_0 and P6_2 to P7_2
            p7_w2=self.p7_w2_relu(self.p7_w2)
            weight=p7_w2 / (torch.sum(p7_w2,dim=0) + self.epsilon)
            # Connections for P7_0 and P6_2 to P7_2
            p7_out=self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))
            return p3_out,p4_out,p5_out,p6_out,p7_out
        else:
            if self.if_first:
                p3,p4,p5=inputs
                p6_in=self.p5_to_p6(p5)
                p7_in=self.p6_to_p7(p6_in)
                p3_in=self.p3_down_channel(p3)
                p4_in=self.p4_down_channel(p4)
                p5_in=self.p5_down_channel(p5)
            else:
                # P3_0,P4_0,P5_0,P6_0 and P7_0
                p3_in,p4_in,p5_in,p6_in,p7_in=inputs
                # P7_0 to P7_2
            # Connections for P6_0 and P7_0 to P6_1 respectively
            p6_up=self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

            # Connections for P5_0 and P6_1 to P5_1 respectively
            p5_up=self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))

            # Connections for P4_0 and P5_1 to P4_1 respectively
            p4_up=self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

            # Connections for P3_0 and P4_1 to P3_2 respectively
            p3_out=self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

            if self.if_first:
                p4_in=self.p4_down_channel_2(p4)
                p5_in=self.p5_down_channel_2(p5)
            # Connections for P4_0,P4_1 and P3_2 to P4_2 respectively
            p4_out=self.conv4_down(
                self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

            # Connections for P5_0,P5_1 and P4_2 to P5_2 respectively
            p5_out=self.conv5_down(
                self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

            # Connections for P6_0,P6_1 and P5_2 to P6_2 respectively
            p6_out=self.conv6_down(
                self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

                # Connections for P7_0 and P6_2 to P7_2
            p7_out=self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

            return p3_out,p4_out,p5_out,p6_out,p7_out

# Classificaiton
class Classifier(nn.Module):
    def __init__(self, in_channels,num_classes, num_layers, pyramid_levels=5, onnx_export=False):
        super(Classifier, self).__init__()
        self.num_classes=num_classes
        self.num_layers=num_layers
        self.conv_list=nn.ModuleList(
            [DSCB_(in_channels, in_channels) for i in range(num_layers)])
        self.bn_list=nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(pyramid_levels)])
        self.header =DSCB_(in_channels, num_classes)
        self.swish=Swish()

    def forward(self, inputs,shape):
        feats=[]
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat=self.swish(bn(conv(feat)))
            #print(feat.shape)
            feat=F.interpolate(feat,shape)
            feat=self.header(feat)
            #feat=feat.permute(0, 2, 3, 1)
            #feat=feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2],self.num_classes)
            #feat=feat.contiguous().view(-1, self.num_classes)
            feats.append(feat)
        return sum(feats).sigmoid()

# Class Semantic Segmentation
class SSNet(nn.Module):
    def __init__(self,class_num=2,compound_coef=0):
        super(SSNet, self).__init__()
        self.compound_coef=compound_coef
        self.class_num=class_num
        self.backbone_compound_coef=[0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters=[64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats=[3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes=[512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats=[3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels=[5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale=[4., 4., 4., 4., 4., 4., 4., 5., 4.]
        self.conv_channel_coef={
            0: [40, 80, 112],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],}
        self.BiFPN= nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    self.conv_channel_coef[compound_coef],
                    True if i == 0 else False,
                    attention=True)
              for i in range(self.fpn_cell_repeats[compound_coef])])
        self.classifier=Classifier(in_channels=self.fpn_num_filters[self.compound_coef],
                                     num_classes=self.class_num,
                                     num_layers=self.box_class_repeats[self.compound_coef],
                                     pyramid_levels=self.pyramid_levels[self.compound_coef])
        self.efficient=EN(self.backbone_compound_coef[compound_coef])
        return
    def forward(self,x):
        shape=x.shape[-2:]
        x=self.efficient(x)
        #for ele in x:print(ele.shape)
        #x=[x[i] for i in [1,3,5]]
        x=[x[i] for i in [1,2,3]]
        x=self.BiFPN(x)
        x=self.classifier(x,shape)
        return x

if __name__=="__main__":
    tmp=torch.rand(1,3,229,123)
    layer=SSNet(2,0)
    tmp=layer(tmp)
    print(tmp.shape)