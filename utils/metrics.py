import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from scipy.ndimage import convolve


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


def SSIM(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    """
    Compute basic SSIM value for each scale
    """
    # Calculate SSIM and CS for multi-scale images
    img1 = img1.float()
    img2 = img2.float()
    _, height, width, _ = img1.shape

    size = min(filter_size, height, width)
    sigma = size * filter_sigma / filter_size if filter_size else 0

    # applying fast fourier transform convolve
    if filter_size:
        window = torch.reshape(gauss_kernel(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode="valid")
        mu2 = signal.fftconvolve(img2, window, mode="valid")
        sigma11 = signal.fftconvolve(img1 * img1, window, mode="valid")
        sigma22 = signal.fftconvolve(img2 * img2, window, mode="valid")
        sigma12 = signal.fftconvolve(img1 * img2, window, mode="valid")
    else:
        # If filter_size is 0, skip the filtering step
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    # Calculate the squared means
    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2

    # Calculate the variances and covariance
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Constants for stability
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    # Calculate intermediate terms
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2

    # Calculate SSIM and CS
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
    cs = np.mean(v1 / v2)

    return ssim, cs


def MultiScaleSSIM(
    img1,
    img2,
    max_val=255,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    weights=None,
):
    # Calculate Multi-Scale SSIM
    weights = np.array(weights if weights else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size

    # Define the downsample filter
    downsample_filter = np.ones((1, 2, 2, 1)) / 4.0

    # Convert input images to float
    im1, im2 = [x.float() for x in [img1, img2]]

    # Initialize arrays to store SSIM and CS values at different scales
    mssim = np.array([])
    mcs = np.array([])

    for _ in range(levels):
        # Calculate SSIM and CS at the current scale
        ssim, cs = SSIM(
            im1,
            im2,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2,
        )

        # Store the SSIM and CS values
        mssim = np.append(mssim, ssim)
        mcs = np.append(mcs, cs)

        # Downsample the images for the next scale
        filtered = [
            convolve(im, downsample_filter, mode="reflect") for im in [im1, im2]
        ]
        im1, im2 = [torch.tensor(x[:, ::2, ::2, :]) for x in filtered]

    # Calculate the final MS-SSIM value
    ms_ssim = np.prod(mcs[0 : levels - 1] ** weights[0 : levels - 1]) * (
        mssim[levels - 1] ** weights[levels - 1]
    )

    return ms_ssim


if __name__ == "__main__":
    print("Testing ssim.py")
    # Test case 1
    img1 = cv2.imread(
        "../ref/test_img/test00.png"
    )  # Replace 'image1.jpg' with the path to your first image
    img2 = cv2.imread(
        "../ref/test_img/test01.png"
    )  # Replace 'image2.jpg' with the path to your second image
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    img1 = torch.from_numpy(np.expand_dims(img1, axis=0))
    img2 = torch.from_numpy(np.expand_dims(img2, axis=0))
    max_val = 255
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    ms_ssim = MultiScaleSSIM(
        img1,
        img2,
        max_val=max_val,
        filter_size=filter_size,
        filter_sigma=filter_sigma,
        k1=k1,
        k2=k2,
        weights=weights,
    )

    # Test case 2 (additional test case)
    img3 = cv2.imread(
        "../ref/test_img/test02.png"
    )  # Replace 'image3.jpg' with the path to your third image
    img4 = cv2.imread(
        "../ref/test_img/test03.png"
    )  # Replace 'image4.jpg' with the path to your fourth image
    img3 = cv2.resize(img3, (img4.shape[1], img4.shape[0]))
    img3 = torch.from_numpy(np.expand_dims(img3, axis=0))
    img4 = torch.from_numpy(np.expand_dims(img4, axis=0))

    max_val = 255
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    ms_ssim2 = MultiScaleSSIM(
        img3,
        img4,
        max_val=max_val,
        filter_size=filter_size,
        filter_sigma=filter_sigma,
        k1=k1,
        k2=k2,
        weights=weights,
    )

    # Assert statements
    assert ms_ssim >= 0.0 and ms_ssim <= 1.0, "Invalid MS-SSIM value for test case 1"
    assert ms_ssim2 >= 0.0 and ms_ssim2 <= 1.0, "Invalid MS-SSIM value for test case 2"

    print(f"ssim for test case 1 {ms_ssim}")
    print(f"ssim for test case 2 {ms_ssim2}")
    print("All test cases passed!")
