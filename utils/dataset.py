import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from utils.preprocess import *

PATCH_WIDTH = 100
PATCH_HEIGHT = 100
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3


class DPEDTestDataset(Dataset):
    def __init__(self, phone, datadir, resolution, test_subset):
        super().__init__()
        self.test_image = []
        self.phone = phone
        self.resolution = resolution
        self.transform = transforms.Compose(
            [
                transforms.ToTensor()
                # transforms.Normalize(mean=[0.5], std=[0.5])
            ]
        )

        # Load image tensor
        self.load(datadir, test_subset)

    def __len__(self):
        return len(self.test_image)

    def __getitem__(self, idx):
        return self.test_image[idx]

    def load(self, datadir, test_subset):
        test_image = sorted(
            [
                os.path.join(datadir, name)
                for name in os.listdir(datadir)
                if os.path.isfile(os.path.join(datadir, name))
            ]
        )

        # use 5 images only, if "full" then pass
        if test_subset == "small":
            test_image = test_image[0:5]

        # Preprocess and transform to tensor
        IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE = get_specified_res(
            self.phone, self.resolution
        )
        print("Loading test data:")
        for image_path in tqdm(test_image):
            image = (
                np.float16(
                    np.array(Image.open(image_path).resize([IMAGE_WIDTH, IMAGE_HEIGHT]))
                )
                / 255
            )  # 0-255 -> 0-1
            image_crop = extract_crop(image, self.resolution, self.phone)
            image_crop = np.reshape(image_crop, [1, IMAGE_SIZE])
            image_tensor = self.transform(image_crop)
            self.test_image.append(image_tensor)

    def collate_fn(self, batch):
        raise NotImplementedError


class DPEDTrainDataset(Dataset):
    def __init__(
        self, phone, datadir, train_size=-1, is_train=True
    ):  # train_size=-1 (full)
        super().__init__()
        self.train_image = []
        self.phone = phone
        self.train_size = train_size
        self.is_train = is_train
        self.transform = transforms.Compose(
            [
                transforms.ToTensor()
                # transforms.Normalize(mean=[0.5], std=[0.5])
            ]
        )

        # Load image tensor
        self.load(datadir)

    def __len__(self):
        return len(self.train_image)

    def __getitem__(self, idx):
        phone_image = self.train_image[idx]["phone_image"]
        dslr_image = self.train_image[idx]["dslr_image"]
        return {"phone_image": phone_image, "dslr_image": dslr_image}

    def load(self, datadir):
        if self.is_train:
            phone_dir = (
                datadir + str(self.phone) + "/training_data/" + str(self.phone) + "/"
            )
            dslr_dir = datadir + str(self.phone) + "/training_data/canon/"

            NUM_TRAINING_IMAGES = len(
                [
                    name
                    for name in os.listdir(phone_dir)
                    if os.path.isfile(os.path.join(dslr_dir, name))
                ]
            )

            if self.train_size == -1:
                self.train_size = NUM_TRAINING_IMAGES
                train_image = np.arange(0, self.train_size)
            else:
                train_image = np.random.choice(
                    np.arange(0, NUM_TRAINING_IMAGES), self.train_size, replace=False
                )

            # Reshape and transform to tensor
            print("Loading train data")
            for img_path in tqdm(train_image):
                phone_image_path = os.path.join(phone_dir, str(img_path) + ".jpg")
                dslr_image_path = os.path.join(dslr_dir, str(img_path) + ".jpg")

                phone_image = (
                    np.float16(np.array(Image.open(phone_image_path)))
                    / 255  # (100,100,3)
                )  # 0-255 -> 0-1
                phone_image_2d = np.reshape(phone_image, [1, PATCH_SIZE])
                phone_image_tensor = self.transform(phone_image_2d)

                dslr_image = (
                    np.float16(np.array(Image.open(dslr_image_path))) / 255
                )  # 0-255 -> 0-1
                dslr_image_2d = np.reshape(dslr_image, [1, PATCH_SIZE])
                dslr_image_tensor = self.transform(dslr_image_2d)

                self.train_image.append(
                    {"phone_image": phone_image_tensor, "dslr_image": dslr_image_tensor}
                )

        else:
            phone_dir = (
                datadir
                + str(self.phone)
                + "/test_data/patches/"
                + str(self.phone)
                + "/"
            )
            dslr_dir = datadir + str(self.phone) + "/test_data/patches/canon/"

            NUM_TEST_IMAGES = len(
                [
                    name
                    for name in os.listdir(phone_dir)
                    if os.path.isfile(os.path.join(phone_dir, name))
                ]
            )

            # for debug
            # NUM_TEST_IMAGES = 10

            # Reshape and transform to tensor
            print("Loading eval data")
            for img_path in tqdm(range(NUM_TEST_IMAGES)):
                phone_image_path = os.path.join(phone_dir, str(img_path) + ".jpg")
                dslr_image_path = os.path.join(dslr_dir, str(img_path) + ".jpg")

                phone_image = (
                    np.float16(np.array(Image.open(phone_image_path))) / 255
                )  # 0-255 -> 0-1
                phone_image_2d = np.reshape(phone_image, [1, PATCH_SIZE])
                phone_image_tensor = self.transform(phone_image_2d)

                dslr_image = (
                    np.float16(np.array(Image.open(dslr_image_path))) / 255
                )  # 0-255 -> 0-1
                dslr_image_2d = np.reshape(dslr_image, [1, PATCH_SIZE])
                dslr_image_tensor = self.transform(dslr_image_2d)

                self.train_image.append(
                    {"phone_image": phone_image_tensor, "dslr_image": dslr_image_tensor}
                )

    def collate_fn(self, batch):
        batch_dict = {
            "phone_image": torch.stack([item["phone_image"] for item in batch], dim=0),
            "dslr_image": torch.stack([item["dslr_image"] for item in batch], dim=0),
        }

        return batch_dict


if __name__ == "__main__":
    # Set your desired parameters
    phone = "iphone"
    traindir = "dped/"
    testdir = "dped/iphone/test_data/full_size_test_images/"
    image_size = 256
    batch_size = 32
    num_workers = 4
    resolution = "orig"
    test_subset = "small"
    iteration = 20000

    # Create the train dataset
    train_dataset = DPEDTrainDataset(phone, traindir, train_size=100, is_train=True)
    print("Number of train size:", len(train_dataset))
    print("Sample:")
    print(train_dataset[0])

    # Create the train data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn,
    )

    # Create the test dataset
    test_dataset = DPEDTestDataset(phone, testdir, resolution, test_subset)
    print("Number of test size", len(test_dataset))
    print("Sample:")
    print(test_dataset[0].shape)

    # Create the test data loader
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print("Dataloader status: Done!")
