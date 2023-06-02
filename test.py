import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm
from torch import nn


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.0) / (kernlen)
    x = np.linspace(-nsig - interval / 2.0, nsig + interval / 2.0, kernlen + 1)
    kern1d = np.diff(norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    gaussian_kernel = np.array(kernel, dtype=np.float32)
    gaussian_kernel = torch.from_numpy(gaussian_kernel).view(1, 1, kernlen, kernlen)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    return gaussian_kernel


def blur(x):
    gaussian_filter = nn.Conv2d(
        in_channels=3, out_channels=3, kernel_size=21, groups=3, bias=False
    )

    gaussian_filter.weight.data = gauss_kernel(21, 3, 3)
    return gaussian_filter(x)


# # Assuming you have an input image tensor 'input_image' with shape (batch, channel, height, width)
img = cv2.imread("ref/dped/iphone/test_data/patches/canon/0.jpg")
cv2.imwrite("blur_src.png", img)
img = torch.from_numpy(np.expand_dims(img, axis=0))
img = img.permute(0, 3, 1, 2)
print(img.shape)
print(img.dtype)
# Apply Gaussian blur to the input image
blurred_image = blur(img.float())
img = blurred_image
# img=img.to(torch.uint8)
img = img.permute(0, 2, 3, 1)
print(img.shape)
print(img)
cv2.imwrite("blur_lib.png", img.detach().numpy()[0])
print("Done")
