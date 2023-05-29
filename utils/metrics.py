import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def gauss_kernel(size, sigma):
    """
    Reduce noise using Gaussian blur
    """
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1

    if size % 2 == 0:
        offset = 0.5
        stop -= 1

    # Create a 2D grid of x and y coordinates
    x, y = torch.meshgrid(
        torch.arange(offset + start, stop), torch.arange(offset + start, stop)
    )
    # Calculate the Gaussian values based on the grid and sigma
    g = torch.exp(-((x**2 + y**2) / (2.0 * sigma**2)))

     # Normalize the kernel values to sum up to 1
    return g / g.sum()


def ssim(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    """
    Compute basic SSIM value for each scale
    """
    img1 = img1.float()
    img2 = img2.float()
    _, height, width, _ = img1.size()

    size = min(filter_size, height, width)
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        # Calculate the Gaussian window
        window = gauss_kernel(size, sigma).unsqueeze(0).unsqueeze(3)

        # Convolve the images with the Gaussian window
        mu1 = F.conv2d(img1, window, padding=0)
        mu2 = F.conv2d(img2, window, padding=0)
        sigma11 = F.conv2d(img1 * img1, window, padding=0)
        sigma22 = F.conv2d(img2 * img2, window, padding=0)
        sigma12 = F.conv2d(img1 * img2, window, padding=0)
    else:
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2

    # Calculate the SSIM and contrast sensitivity
    ssim = torch.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
    cs = torch.mean(v1 / v2)

    return ssim, cs

def MultiScaleSSIM(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, weights=None):
    # Convert the weights to a tensor if provided, otherwise use default weights
    weights = torch.tensor(weights) if weights else torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size(0)

    # Define the downsample filter
    downsample_filter = torch.tensor([[1, 1], [1, 1]], dtype=torch.float) / 4.0

    # Convert the input images to float tensors
    im1, im2 = img1.float(), img2.float()

    # Initialize empty tensors to store SSIM and CS values for each scale
    mssim = torch.tensor([])
    mcs = torch.tensor([])

    for _ in range(levels):
        # Calculate SSIM and CS at the current scale
        ssim, cs = ssim(im1, im2, max_val=max_val, filter_size=filter_size, filter_sigma=filter_sigma, k1=k1, k2=k2)

        # Append the calculated values to the respective tensors
        mssim = torch.cat((mssim, ssim.unsqueeze(0)))
        mcs = torch.cat((mcs, cs.unsqueeze(0)))

        # Downsample the images using the downsample filter
        filtered = [F.conv2d(im, downsample_filter.unsqueeze(0).unsqueeze(0)) for im in [im1, im2]]
        im1, im2 = [x[:, :, ::2, ::2] for x in filtered]

    # Calculate the final MS-SSIM score using the computed values and weights
    return torch.prod(mcs[:-1] ** weights[:-1]) * (mssim[-1] ** weights[-1])
