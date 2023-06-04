# python predict.py model=iphone datadir=ref/test_img/ test_subset=full resolution=orig weight_path=models/100trainsize/weights/generator_epoches_209.pth
import os
import sys

from tqdm import tqdm
import imageio
import numpy as np
import cv2
import torch
from torch import nn
from utils.dataset import DPEDTestDataset

import utils.parse_arg
from utils.preprocess import get_specified_res
from utils.models import ResNet


def run(phone, datadir, test_subset, weight_path, resolution):
    # Set the device (GPU or CPU)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device=torch.device("cpu")

    # Load the model and set it to evaluation mode
    generator = ResNet() 
    generator.load_state_dict(torch.load(weight_path,map_location=device))
    generator=generator.to(device) 
    generator.eval()
    # Get image size
    IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE = get_specified_res(phone, resolution)

    # Output folder
    output_path = 'output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    folders = os.listdir(output_path)
    new_id = 0
    if len(folders) > 0:
        for folder in folders:
            if not folder.startswith('exp_'):
                continue
            new_id = max(new_id, int(folder.split('exp_')[-1]))
        new_id += 1
    output_path = os.path.join(output_path, f'exp_{new_id}')
    os.makedirs(output_path)
    output_path=output_path+'/'


    test_ds=DPEDTestDataset(phone, datadir, resolution, test_subset)
    idx=0
    print('Processing image')
    with torch.no_grad():
        for image in tqdm(test_ds):
            idx+=1
            phone_img=image.float().to(device)
            # Reshape to fit into the net
            phone_img = (
                phone_img.view(-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
                .permute(0, 3, 1, 2))
            
            enhanced_img=generator(phone_img)
            before_after = np.hstack(
                        (
                            phone_img[0].permute(1, 2, 0).cpu().numpy(),
                            enhanced_img[0].permute(1, 2, 0).cpu().numpy(),
                        )
                    )
            imageio.imwrite(output_path+f"before_after_{idx}.jpg",
                        cv2.normalize(
                            before_after,
                            None,
                            alpha=0,
                            beta=255,
                            norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_8U,
                        ).astype(np.uint8),
                    )
            imageio.imwrite(output_path+f"enhanced_{idx}.jpg",
                        cv2.normalize(
                            enhanced_img[0].permute(1, 2, 0).cpu().numpy(),
                            None,
                            alpha=0,
                            beta=255,
                            norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_8U,
                        ).astype(np.uint8),
            )
            print(f'Saved enhanced images to {output_path}'+f"enhanced_{idx}.jpg")
            
    return {output_path}+f"enhanced_{idx}.jpg"

if __name__ == "__main__":
    run(**utils.parse_arg.test_args(sys.argv))
