import sys
from functools import reduce

import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt


def tensor_size(tensor):
    """
    Compute size of feature map at CONTENT_LAYER
    """
    from operator import mul

    return reduce(mul, (d for d in tensor.shape[1:]), 1)


def plot_losses(train_loss_gen, train_loss_discrim, train_acc_discrim, test_loss_gen, test_loss_discrim, test_psnr, test_ssim, test_content, test_color, test_texture, test_tv, test_acc_discrim, file_path):
    fig, axs = plt.subplots(2, 6, figsize=(12, 10))
    
    # Plot train loss
    axs[0, 0].plot(train_loss_gen, label='Train Generator Loss')
    axs[0, 0].set_title('Train Generator Loss')
    
    
    axs[0, 1].plot(train_loss_discrim, label='Train Discriminator Loss')
    axs[0, 1].set_title('Train Discriminator Loss')
    
    axs[0, 2].plot(train_acc_discrim, label='Train Discriminator Accuracy')
    axs[0, 2].set_title('Train Discriminator Accuracy')

    # Plot test loss
    axs[0, 3].plot(test_loss_gen, label='Test Generator Loss')
    axs[0, 3].set_title('Test Generator Loss')
    
    axs[0, 4].plot(test_loss_discrim, label='Test Discriminator Loss')
    axs[0, 4].set_title('Test Discriminator Loss')

    # Plot test metrics
    axs[0, 5].plot(test_psnr, label='Test PSNR')
    axs[0, 5].set_title('Test PSNR')
    
    axs[1, 0].plot(test_ssim, label='Test SSIM')
    axs[1, 0].set_title('Test SSIM')
    
    axs[1, 1].plot(test_content, label='Test Content Loss')
    axs[1, 1].set_title('Test Content Loss')
    
    axs[1, 2].plot(test_color, label='Test Color Loss')
    axs[1, 2].set_title('Test Color Loss')
    
    axs[1, 3].plot(test_texture, label='Test Texture Loss')
    axs[1, 3].set_title('Test Texture Loss')
    
    axs[1, 4].plot(test_tv, label='Test TV Loss')
    axs[1, 4].set_title('Test TV Loss')
    
    axs[1, 5].plot(test_acc_discrim, label='Test Discriminator Accuracy')
    axs[1, 5].set_title('Test Discriminator Accuracy')

    # Adjust subplot spacing
    plt.tight_layout()

    # Save figure
    plt.savefig(file_path)


