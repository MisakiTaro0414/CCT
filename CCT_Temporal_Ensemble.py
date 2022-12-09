#!/usr/bin/env python
# coding: utf-8

# In[ ]:




# In[ ]:


import os
import json
import argparse
import torch
import dataloaders
import models
import math
from utils import Logger
from trainer import Trainer
import torch.nn.functional as F
from utils.losses import abCE_loss, CE_loss, consistency_weight, FocalLoss, softmax_helper, get_alpha
from itertools import cycle
from tqdm import tqdm
import numpy as np
from utils.metrics import eval_metrics, AverageMeter
from math import ceil


# In[ ]:


torch.manual_seed(42)


# In[ ]:


config = json.load(open('./configs/config_temporal.json'))


# In[ ]:


# DATA LOADERS
config['train_supervised']['n_labeled_examples'] = config['n_labeled_examples']
supervised_loader = dataloaders.VOC(config['train_supervised'])
val_loader = dataloaders.VOC(config['val_loader'])
iter_per_epoch = len(supervised_loader)


# In[ ]:


# Utils
def temporal_loss(out1, out2, w, labels, mask_T):
    sup_loss = F.cross_entropy(out1, labels, ignore_index=255)
    
    # out2: (B,C,H,W)
    if w == 0:
        return sup_loss, sup_loss, torch.tensor(0)
    
    pseudo_label = F.softmax(out2, dim=1) # (B,C,H,W): compute the probability distribution over C classes
    low_confident_mask = torch.amax(pseudo_label, dim=1) < mask_T # (B,H,W) # Get the highest prob, if prob < threshold, ignore that position
    pseudo_label = pseudo_label.argmax(dim=1) # (B,H,W) # turn prob distribution into class index
    pseudo_label[low_confident_mask] = 255 # Ignore position with low condifent
    unsup_loss = F.cross_entropy(out1, pseudo_label, ignore_index=255)
    
    return sup_loss + w * unsup_loss, sup_loss, w * unsup_loss

def ramp_up(epoch, max_epochs, max_val, mult):
    if epoch == 0:
        return 0.
    elif epoch >= max_epochs:
        return max_val
    return max_val * np.exp(mult * (1. - float(epoch) / max_epochs) ** 2)


def weight_schedule(epoch, max_epochs, max_val, mult, n_samples):
    return ramp_up(epoch, max_epochs, max_val, mult)


# In[ ]:


# Model
model = models.CCT(num_classes=21, conf=config['model'], testing=True)
CCT_checkpoint = torch.load('./saved/CCT_wssl/best_model.pth')
model = torch.nn.DataParallel(model)
try:
    model.load_state_dict(CCT_checkpoint['state_dict'], strict=True)
except Exception as e:
    print(f'Some modules are missing: {e}')
    model.load_state_dict(CCT_checkpoint['state_dict'], strict=False)
model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=2.5e-4, weight_decay=1e-4, momentum=0.9)


# In[ ]:


ntrain, n_classes, H, W = config['n_labeled_examples'], 21, 320, 320
losses = []
sup_losses = []
unsup_losses = []
best_loss = 20.
batch_size = 10
num_epochs = 10
alpha = 0.6
max_val=30.
ramp_up_mult=-5.
mask_T = 0.2

Z = torch.zeros(ntrain, n_classes, H, W).float().cuda()  # intermediate values
z = torch.zeros(ntrain, n_classes, H, W).float().cuda()  # temporal outputs


# In[ ]:


best_val_mIoU = 0.735
mIoU_hist = []
for epoch in range(num_epochs):
    model.train()
    tbar = tqdm(supervised_loader, ncols=135)
    
#     w = weight_schedule(epoch, num_epochs, max_val, ramp_up_mult, ntrain)
     
#     print('unsupervised loss weight : {}'.format(w))

#     # turn it into a usable pytorch object
#     w = torch.tensor([w], dtype=float, requires_grad=False).cuda()

    l = []
    supl = []
    unsupl = []
    
    for i, (images, labels) in enumerate(tbar):
        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        
        out = model(images)
        zcomp = z[i * batch_size: (i + 1) * batch_size].detach()
#         zcomp.requires_grad = False
        if epoch == 0:
            loss, suploss, unsuploss = temporal_loss(out, zcomp, 0, labels, mask_T)
        else:
            loss, suploss, unsuploss = temporal_loss(out, zcomp, 0.01, labels, mask_T)
        
        # update temporal ensemble
        Z[i * batch_size: (i + 1) * batch_size] = alpha * Z[i * batch_size: (i + 1) * batch_size] + (1. - alpha) * out
        z[i * batch_size: (i + 1) * batch_size] = Z[i * batch_size: (i + 1) * batch_size] * (1. / (1. - alpha ** (epoch + 1)))

        # save outputs and losses
#         outputs[i * batch_size: (i + 1) * batch_size] = out.data.clone()
        l.append(loss.item())
        supl.append(suploss.item())
        unsupl.append(unsuploss.item())

        # backprop
        loss.backward()
        optimizer.step()
        
        # print loss
        tbar.set_description('Epoch [%d/%d], Step [%d/%d], suploss: %.3f, unsuploss: %.3f' 
                               %(epoch + 1, num_epochs, i + 1, iter_per_epoch, np.mean(supl), np.mean(unsupl)))

    # Eval epoch
    model.eval()
    total_loss_val = AverageMeter()
    total_inter, total_union = 0, 0
    total_correct, total_label = 0, 0
    mIoU= None
    
    tbar = tqdm(val_loader, ncols=135)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tbar):
            target, data = target.cuda(), data.cuda()

            H, W = target.size(1), target.size(2)
            up_sizes = (ceil(H / 8) * 8, ceil(W / 8) * 8)
            pad_h, pad_w = up_sizes[0] - data.size(2), up_sizes[1] - data.size(3)
            data = F.pad(data, pad=(0, pad_w, 0, pad_h), mode='reflect')
            output = model(data)
            output = output[:, :, :H, :W]
            
            loss = F.cross_entropy(output, target, ignore_index=255)
            total_loss_val.update(loss.item())
            correct, labeled, inter, union = eval_metrics(output, target, 21, 255)
            total_inter, total_union = total_inter+inter, total_union+union
            total_correct, total_label = total_correct+correct, total_label+labeled
            
            # PRINT INFO
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            seg_metrics = {"Pixel_Accuracy": np.round(pixAcc, 3), "Mean_IoU": np.round(mIoU, 3),
                            "Class_IoU": dict(zip(range(21), np.round(IoU, 3)))}

            tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.3f} |'.format( epoch,
                                            total_loss_val.average, pixAcc, mIoU))
        
    mIoU_hist.append(mIoU)    
    if mIoU > best_val_mIoU:
        best_val_mIoU = mIoU
        state = {
            'CCT_state_dict': model.state_dict(),
            'discriminator_state_dict': model_D.state_dict()
        }
        torch.save(state, f'./saved/Temporal_Ensemble/{best_val_mIoU}/best_model.pth')
        print('Save best checkpoint, mIoU: ', mIoU)
    
    # update temporal ensemble
#     Z = alpha * Z + (1. - alpha) * outputs
#     z = Z * (1. / (1. - alpha ** (epoch + 1)))

    # handle metrics, losses, etc.
    eloss = np.mean(l)
    losses.append(eloss)
    sup_losses.append(np.mean(supl))
    unsup_losses.append(np.mean(unsupl))

