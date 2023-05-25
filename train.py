import os
import sys

import imageio
import numpy as np
import torch
from torch import nn

import utils.parse_arg


def run(
    batch_size,
    train_size,
    lr,
    epochs,
    w_content,
    w_color,
    w_texture,
    w_tv,
    dataset,
    vgg_pretrained,
    eval_step,
    phone,
):
    np.random.seed(0)
    # defining system architecture


if __name__ == "__main__":
    run(**utils.parse_arg.train_args(sys.argv))
