import numpy as np
import torch
from torch import nn

def gauss_kernel(size,sigma):
    '''
    Reduce noise using Gaussian blur
    '''
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1

    if size % 2 == 0:
        offset = 0.5
        stop -= 1

    x, y = torch.meshgrid(torch.arange(offset + start, stop), torch.arange(offset + start, stop))
    g = torch.exp(-((x**2 + y**2) / (2.0 * sigma**2)))

    return g / g.sum()

def ssim(img1,img2,max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    '''
    Compute basic SSIM value
    '''
    pass
    