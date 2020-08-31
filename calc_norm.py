''' Calculate normalization values for dataset (WT)
 This script iterates over the dataset and calculates the normalization values
 (minimum, maximum, shift, scale).
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import utils
from tqdm import tqdm, trange

train_transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()])
train_dataset = ImageFolder(root='/disk_c/data/church/', transform=train_transform)
train_loader = DataLoader(train_dataset,
                            batch_size=64,
                            shuffle=False)

def run():
    device = 'cuda'
    filters = utils.create_filters(device=device)
    min_val = float('inf')
    max_val = float('-inf')
    for i, (x, y) in enumerate(tqdm(train_loader)):
        x = x.to(device)
        x = utils.wt(x, filters, levels=2)[:, :, :64, :64]
        min_val = min(min_val, torch.min(x))
        max_val = max(max_val, torch.max(x))

    shift = torch.ceil(torch.abs(min_val))
    scale = shift + torch.ceil(max_val)

    print('Minimum: {} \t Maximum: {}\nShift: {} \t Scale: {}'.format(min_val, max_val, shift, scale))

    print('Saving normalization values to disk...')
    np.savez('lsun_church_norm_values.npz', **{'min' : min_val.cpu(), 'max' : max_val.cpu(), 'shift': shift.cpu(), 'scale': scale.cpu()})


if __name__ == '__main__':    
    run()
