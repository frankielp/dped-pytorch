import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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
    gaussian_kernel = np.array(kernel, dtype=np.float32)
    gaussian_kernel = torch.from_numpy(gaussian_kernel).view(1, 1, kernlen, kernlen)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    return gaussian_kernel


def blur(x):
    gaussian_filter = nn.Conv2d(in_channels=3, out_channels=3,
                                    kernel_size=21, groups=3, bias=False)
    gaussian_filter=gaussian_filter.to(x.device)
    gaussian_filter.weight.data = gauss_kernel(21,3,3)
    return gaussian_filter(x)
