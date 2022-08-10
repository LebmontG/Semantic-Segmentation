# -*- coding: utf-8 -*-
"""
Object detection and semantic segmentation model BiFPN
Created on Sat Jul 23 16:43:12 2022 by Z.Gan
"""

import numpy as np
import os,sys,gc
import cv2
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from utils import mIoU,BIoU
from model import SSNet
from tqdm import tqdm
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

labelpath="dataset/mask/"
datapath="dataset/horse/"
prop=0.85
class_num=2
compound_coef=0
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

names=os.walk(labelpath)
for name in names:
    names=name[-1]
    break
loc,l=1,len(names)
tx,ty,vx,vy=[],[],[],[]
for name in names:
    if loc<l*prop:
        tmp=cv2.imread(labelpath+name)
        tmp=np.mean(tmp,2)
        ty.append(torch.tensor(tmp).contiguous().view(1,-1))
        #ty.append(torch.tensor(tmp))
        tmp=cv2.imread(datapath+name)
        tx.append(torch.FloatTensor(tmp).unsqueeze(0).permute(0,3,1,2))
    else:
        tmp=cv2.imread(labelpath+name)
        tmp=np.mean(tmp,2)
        #vy.append(torch.tensor(tmp).contiguous().view(1,-1))
        vy.append(torch.tensor(tmp))
        tmp=cv2.imread(datapath+name)
        vx.append(torch.FloatTensor(tmp).unsqueeze(0).permute(0,3,1,2))
    loc+=1
#tx,ty=np.array(tx),np.array(ty)
ind=np.random.permutation(np.arange(len(tx)))
tx,ty=[tx[i] for i in ind],[ty[i] for i in ind]
gc.collect()

model=SSNet(class_num,compound_coef).to(device)
#model.load_state_dict(torch.load('m.pth'))
loss=nn.CrossEntropyLoss()
opt=optim.Adam(model.parameters())
for image,label in zip(tx,ty):
    torch.cuda.empty_cache()
    image,label=image.to(device),label.to(device)
    opt.zero_grad()
    y=model(image)
    y=y.permute(0, 2, 3, 1)
    y=y.contiguous().view(-1,class_num)
    l=loss(y,torch.tensor(label,dtype=torch.long)[0])
    l.backward()
    opt.step()
    torch.save(model.state_dict(),'m.pth')
    gc.collect()

miou,biou=[],[]
with torch.no_grad():
    for image,label in tqdm(zip(vx,vy)):
        image,label=image.to(device),label.to(device)
        y=model(image)
        miou.append(mIoU(y,label))
        biou.append(BIoU(y,label))
plt.plot(miou)
plt.plot(biou)

answer=miou
plt.plot(answer,color='blue', linewidth=2.5)
plt.ylabel("MIoU")
plt.xlabel("Episodes")
plt.title("MIoU Curve")
#plt.xlim(0,750)
#if i==0:plt.xlabel('t/s')
plt.gcf().savefig('1.eps',format='eps')