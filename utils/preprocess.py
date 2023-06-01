import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm

def get_specified_res(phone, resolution):
    default_res = default_resolutions()
    if resolution == "orig":
        IMAGE_HEIGHT = default_res[phone][0]
        IMAGE_WIDTH = default_res[phone][1]
    else:
        IMAGE_HEIGHT = default_res[resolution][0]
        IMAGE_WIDTH = default_res[resolution][1]

    IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3

    return IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE


# Support function
def default_resolutions():
    # IMAGE_HEIGHT, IMAGE_WIDTH

    default_res = {}
    # Based on phone
    default_res["iphone"] = [1536, 2048]
    default_res["iphone_orig"] = [1536, 2048]
    default_res["blackberry"] = [1560, 2080]
    default_res["blackberry_orig"] = [1560, 2080]
    default_res["sony"] = [1944, 2592]
    default_res["sony_orig"] = [1944, 2592]

    # Based on resolution
    default_res["high"] = [1260, 1680]
    default_res["medium"] = [1024, 1366]
    default_res["small"] = [768, 1024]
    default_res["tiny"] = [600, 800]

    return default_res


def extract_crop(image, resolution, phone):
    default_res = default_resolutions()

    if resolution == "orig":
        return image

    else:
        x_up = int((default_res[phone][1] - default_res[resolution][1]) / 2)
        y_up = int((default_res[phone][0] - default_res[resolution][0]) / 2)

        x_down = x_up + default_res[resolution][1]
        y_down = y_up + default_res[resolution][0]

        return image[y_up:y_down, x_up:x_down, :]

def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((1, 1, kernlen, kernlen))
    out_filter = np.repeat(out_filter, channels, axis=1)
    return torch.from_numpy(out_filter)

def blur(x):
    kernel_var = gauss_kernel(21, 3, x.size(1))
    kernel_var = kernel_var.to(x.device)  # Move the kernel to the same device as input tensor
    return F.conv2d(x, kernel_var, padding=(21 // 2), groups=x.size(1))

def gaussian_blur(image, kernel_size=21, sigma=3):
    # Create a Gaussian kernel
    kernel = torch.ones(1, 1, kernel_size, kernel_size)
    kernel = kernel.to(image.device)  # Move the kernel to the same device as the input tensor
    kernel = kernel.float()
    kernel = kernel / (2 * sigma**2)
    kernel = kernel * (-1 / (2 * sigma**2))
    kernel = torch.exp(kernel)
    
    # Normalize the kernel
    kernel_sum = torch.sum(kernel)
    kernel = kernel / kernel_sum

    # Apply convolution with the Gaussian kernel
    blurred_image = F.conv2d(image, kernel, padding=kernel_size // 2, groups=image.size(1))
    
    return blurred_image