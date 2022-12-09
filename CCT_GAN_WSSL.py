#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[2]:


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


# In[3]:


torch.manual_seed(42)


# In[4]:


config = json.load(open('./configs/config_wssl.json'))


# In[5]:


# DATA LOADERS
config['train_supervised']['n_labeled_examples'] = config['n_labeled_examples']
config['train_unsupervised']['n_labeled_examples'] = config['n_labeled_examples']
config['train_unsupervised']['use_weak_lables'] = config['use_weak_lables']
supervised_loader = dataloaders.VOC(config['train_supervised'])
unsupervised_loader = dataloaders.VOC(config['train_unsupervised'])
val_loader = dataloaders.VOC(config['val_loader'])
iter_per_epoch = len(unsupervised_loader)


# In[6]:


import torch.nn as nn
import torch.nn.functional as F

class FCDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf=64):
        super(FCDiscriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.classifier(x)
        
        return x


# In[7]:


# Loss
class CrossEntropy2d(nn.Module):

    def __init__(self, reduction='mean', ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.reduction = reduction
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, reduction=self.reduction)
        return loss
    
class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, reduction='mean', ignore_label=255):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.reduction = reduction
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict[target_mask]
        loss = F.binary_cross_entropy_with_logits(predict, target, weight=weight, reduction=self.reduction)
        return loss


# In[8]:


def one_hot(label):
    label = label.cpu().numpy()
    one_hot = np.zeros((label.shape[0], 21, label.shape[1], label.shape[2]), dtype=label.dtype)
    for i in range(21):
        one_hot[:,i,...] = (label==i)
    #handle ignore labels
    return torch.FloatTensor(one_hot).cuda()

def make_D_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)
    D_label = np.ones(ignore_mask.shape)*label
    D_label[ignore_mask] = 255
    D_label = torch.FloatTensor(D_label)

    return D_label.cuda()

def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label.long().cuda()
    criterion = CrossEntropy2d().cuda()

    return criterion(pred, label)

def get_seg_metrics(correct, label, inter, union):
    pixAcc = 1.0 * correct / (np.spacing(1) + label)
    IoU = 1.0 * inter / (np.spacing(1) + union)
    mIoU = IoU.mean()
    return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(21), np.round(IoU, 3)))
        }


# In[9]:


model = models.CCT(num_classes=21, conf=config['model'], testing=True)
model_D = FCDiscriminator(num_classes=21).cuda()

CCT_checkpoint = torch.load('./saved/CCT_wssl/best_model.pth')
model = torch.nn.DataParallel(model)
try:
    model.load_state_dict(CCT_checkpoint['state_dict'], strict=True)
except Exception as e:
    print(f'Some modules are missing: {e}')
    model.load_state_dict(CCT_checkpoint['state_dict'], strict=False)
model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=2.5e-4, momentum=0.9, weight_decay=1e-4)
optimizer_D = torch.optim.Adam(model_D.parameters(), lr=1e-4 ,betas=(0.9, 0.99))

scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
scheduler_D = torch.optim.lr_scheduler.ConstantLR(optimizer_D)


# In[10]:


bce_loss = BCEWithLogitsLoss2d()
interp = nn.Upsample(size=(320, 320), mode='bilinear')

epochs = 10

pred_label = 0
gt_label = 1

lambda_semi_adv = 0.001
lambda_adv_pred = 0.01
lambda_semi = 0.1
mask_T = 0.2
weakly_loss_w = 0.01


# In[11]:


# %%capture output
best_val_mIoU = 0.74
mIoU_hist = []

for epoch in range(epochs):
    model.train()
    dataloader = iter(zip(cycle(supervised_loader), unsupervised_loader))
    tbar = tqdm(range(len(unsupervised_loader)), ncols=135)
    for batch_idx in tbar:
        (input_l, target_l), (input_ul, target_ul) = next(dataloader)
        input_l, target_l = input_l.cuda(), target_l.cuda()
        input_ul, target_ul = input_ul.cuda(), target_ul.cuda()
        
        optimizer.zero_grad()
        
        ##########################
        # Train G ################
        ##########################
        
        # Train in semi setting
        pred = interp(model(input_ul)) # S(X_n): (B,C,H,W)

        D_out = interp(model_D(F.softmax(pred, dim=1))) # D(S(X_n)): (B,1,H,W)
        D_out_sigmoid = torch.sigmoid(D_out).data.cpu().numpy().squeeze(axis=1) # D(S(X_n)): (B,1,H,W)

        ignore_mask_remain = np.zeros(D_out_sigmoid.shape).astype(bool) # remain all value, no ignore in D_out_sigmoid
        
        loss_semi_adv = lambda_semi_adv * bce_loss(D_out, make_D_label(gt_label, ignore_mask_remain))
        
        # produce ignore mask
        semi_ignore_mask = (D_out_sigmoid < mask_T)

        semi_gt = pred.data.cpu().numpy().argmax(axis=1) # Pred with high confidence is self-taught label
        semi_gt[semi_ignore_mask] = 255 

        semi_ratio = 1.0 - float(semi_ignore_mask.sum())/semi_ignore_mask.size
        # print('semi ratio: {:.4f}'.format(semi_ratio))

        semi_gt = torch.FloatTensor(semi_gt)

        loss_semi = lambda_semi * loss_calc(pred, semi_gt) # Seft-taught
        loss_semi += loss_semi_adv 
#             loss_semi.backward()
        
        # Train with weak label
        loss_weakly = weakly_loss_w * loss_calc(pred, target_ul)
#         loss_weakly.backward()
        (loss_semi + loss_weakly).backward()
        
        pred_remain = pred.detach()
        
        # Train with labeled data
        ignore_mask = (target_l.cpu().numpy() == 255)
        pred = interp(model(input_l)) # pred change value
        
        loss_seg = loss_calc(pred, target_l) # L_ce - equation (3)
        
        D_out = interp(model_D(F.softmax(pred, dim=1)))
        
        loss_adv_pred = bce_loss(D_out, make_D_label(gt_label, ignore_mask)) # L_adv - equation (4)
        
        loss = loss_seg + lambda_adv_pred * loss_adv_pred # Equation (2) without semi loss, semi loss is computed above
        loss.backward()
        
        optimizer.step()
        
        ##########################
        # Train D ################
        ##########################
        
        optimizer_D.zero_grad()
            
        pred = pred.detach() # No backprop to G
        pred = torch.cat((pred, pred_remain), 0) # pred - label data, pred_remain: unlabel data
        ignore_mask = np.concatenate((ignore_mask,ignore_mask_remain), axis = 0)
        D_out = interp(model_D(F.softmax(pred, dim=1)))
        loss_D = bce_loss(D_out, make_D_label(pred_label, ignore_mask)) # Part 1 Equation (1)
        
        D_gt_v = one_hot(target_l).cuda()
        ignore_mask_gt = (target_l.cpu().numpy() == 255)
        D_out = interp(model_D(D_gt_v))
        
        loss_D += bce_loss(D_out, make_D_label(gt_label, ignore_mask_gt)) # Part 2 Equation (1)
        loss_D.backward()
                          
        optimizer_D.step()
        
        
        tbar.set_description('loss_seg = {0:.3f}, loss_adv_p = {1:.3f}, loss_D = {2:.3f}, loss_semi = {3:.3f}, loss_semi_adv = {4:.3f}, loss_weakly = {5:.3f}'.format(loss_seg.item(), loss_adv_pred.item(), loss_D.item(), loss_semi.item(), loss_semi_adv.item(), loss_weakly.item()))
    
    scheduler.step()
    scheduler_D.step()
    
    # Eval epoch
    model.eval()
    total_loss_val = AverageMeter()
    total_inter, total_union = 0, 0
    total_correct, total_label = 0, 0
    mIoU= None
    
    tbar = tqdm(val_loader, ncols=130)
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
        os.makedirs(f'./saved/GAN_WSSL/{best_val_mIoU}')
        torch.save(state, f'./saved/GAN_WSSL/{best_val_mIoU}/best_model.pth')
        print('Save best checkpoint')
            


# In[ ]:


print('best validation mIoU: ', best_val_mIoU)

