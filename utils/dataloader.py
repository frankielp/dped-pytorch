import os
# import torch
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from preprocess import *


class DPEDTestDataset(Dataset):
    def __init__(self, phone, dataset, resolution, test_subset):
        super().__init__()
        self.test_image=[]
        self.transform = transforms.Compose([
            transforms.ToTensor()
            # transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.load(phone, dataset, resolution, test_subset)


    def __len__(self):
        return len(self.test_image)

    def __getitem__(self, idx):
        phone_image=self.test_image[idx]
        return phone_image
    
        

    def load(self, phone, dataset, resolution, test_subset):
        test_image = sorted([name for name in os.listdir(dataset)
                                   if os.path.isfile(os.path.join(dataset, name))])
        
        # use 5 images only, if "full" then pass
        if test_subset == "small":
            test_image = test_image[0:1]
        
        # Preprocess and transform to tensor
        IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE =get_specified_res(phone, resolution)
        for image_name in test_image:
            image_path = os.path.join(dataset, image_name)
            image = np.float16(np.array(Image.open(image_path).resize([IMAGE_WIDTH, IMAGE_HEIGHT]))) / 255  # 0-255 -> 0-1
            image_crop = extract_crop(image, resolution, phone)
            image_crop = np.reshape(image_crop, [1, IMAGE_SIZE])
            image_crop_tensor = self.transform(image_crop)
            self.test_image.append(image_crop_tensor)


        
       
    
    






if __name__ == '__main__':
    # Set your desired parameters
    phone = 'iphone'
    dataset = 'dped/iphone/test_data/full_size_test_images/'
    image_size = 256
    batch_size = 32
    num_workers = 4
    resolution='orig'
    test_subset='small'
    iteration=20000

    # Create the test dataset
    test_dataset = DPEDTestDataset(phone, dataset, resolution, test_subset)
    print('Number of test size', len(test_dataset))

    # Create the test data loader
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

