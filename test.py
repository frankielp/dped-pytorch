import torch
import cv2
import numpy as np
import torch.nn.functional as F


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = torch.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = torch.diff(torch.distributions.Normal(0, nsig).cdf(x))
    kernel_raw = torch.sqrt(torch.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = kernel.view(1, 1,kernlen, kernlen)
    out_filter = out_filter.repeat(1, channels, 1, 1)
    return out_filter

def blur(x):
    kernel_var = gauss_kernel(21, 3, x.size(1))
    return F.conv2d(x, kernel_var, padding=10)

# Assuming you have an input image tensor 'input_image' with shape (batch, channel, height, width)
img=cv2.imread('ref/test_img/test00.png')
img = torch.from_numpy(np.expand_dims(img, axis=0))
img=img.permute(0,3,1,2)
print(img.shape)
# Apply Gaussian blur to the input image
blurred_image = blur(img)

cv2.imwrite('blur.jpg', blurred_image)

