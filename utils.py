#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Utilities file
This file contains utility functions for bookkeeping, logging, and data loading.
Methods which directly affect training should either go in layers, the model,
or train_fns.py.
'''

from __future__ import print_function
import sys
import os
import numpy as np
import time
import datetime
import json
import pickle
import pywt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


norm_dict = {}

# WT filter creating function
def create_filters(device, wt_fn='bior2.2'):
    w = pywt.Wavelet(wt_fn)

    dec_hi = torch.Tensor(w.dec_hi[::-1]).to(device)
    dec_lo = torch.Tensor(w.dec_lo[::-1]).to(device)

    filters = torch.stack([dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1),
                          dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1),
                          dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1),
                          dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)], dim=0)

    return filters

# IWT filter creating function
def create_inv_filters(device, wt_fn='bior2.2'):
    w = pywt.Wavelet(wt_fn)

    rec_hi = torch.Tensor(w.rec_hi).to(device)
    rec_lo = torch.Tensor(w.rec_lo).to(device)

    inv_filters = torch.stack([rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1),
                              rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1),
                              rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1),
                              rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)], dim=0)

    return inv_filters

# WT function
def wt(vimg, filters, levels=1):
    bs = vimg.shape[0]
    h = vimg.size(2)
    w = vimg.size(3)
    vimg = vimg.reshape(-1, 1, h, w)
    padded = torch.nn.functional.pad(vimg,(2,2,2,2))
    res = torch.nn.functional.conv2d(padded, Variable(filters[:,None]),stride=2)
    if levels>1:
        res[:,:1] = wt(res[:,:1], filters, levels-1)
        res[:,:1,32:,:] = res[:,:1,32:,:]*1.
        res[:,:1,:,32:] = res[:,:1,:,32:]*1.
        res[:,1:] = res[:,1:]*1.
    res = res.view(-1,2,h//2,w//2).transpose(1,2).contiguous().view(-1,1,h,w)
    return res.reshape(bs, -1, h, w)


# IWT function
def iwt(vres, inv_filters, levels=1):
    bs = vres.shape[0]
    h = vres.size(2)
    w = vres.size(3)
    vres = vres.reshape(-1, 1, h, w)
    res = vres.contiguous().view(-1, h//2, 2, w//2).transpose(1, 2).contiguous().view(-1, 4, h//2, w//2).clone()
    if levels > 1:
        res[:,:1] = iwt(res[:,:1], inv_filters, levels=levels-1)
    res = torch.nn.functional.conv_transpose2d(res, Variable(inv_filters[:,None]),stride=2)
    res = res[:,:,2:-2,2:-2] #removing padding

    return res.reshape(bs, -1, h, w)

def denormalize_pixel(x):
    return (x * 0.5) + 0.5

def denormalize_wt(x, shift, scale):
    return ((x * 0.5) + 0.5) * scale - shift
    
def normalize_wt(x, shift, scale):
    return (((x + shift) / scale) - 0.5) / 0.5
  
def load_norm_dict(path):
    loaded = np.load(path)
    global norm_dict
    norm_dict = loaded
    return norm_dict

def get_norm_dict():
    assert (len(norm_dict) > 0)
    return norm_dict

# Create padding on patch so that this patch is formed into a square image with other patches as 0
# 3 x 128 x 128 => 3 x target_dim x target_dim
def zero_pad(img, target_dim, device='cpu'):
    batch_size = img.shape[0]
    num_channels = img.shape[1]
    padded_img = torch.zeros((batch_size, num_channels, target_dim, target_dim), device=device)
    padded_img[:, :, :img.shape[2], :img.shape[3]] = img.to(device)

    return padded_img
