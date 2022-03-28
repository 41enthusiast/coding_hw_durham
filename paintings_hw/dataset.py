from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class PaintingDataset(Dataset):
    def __init__(self, root_trainA, root_trainB, transform=None):
        self.root_trainA = root_trainA
        self.root_trainB = root_trainB
        self.transform = transform

        self.trainA_images = os.listdir(root_trainA)
        self.trainB_images = os.listdir(root_trainB)
        self.length_dataset = max(len(self.trainA_images), len(self.trainB_images)) # 1000, 1500
        self.trainA_len = len(self.trainA_images)
        self.trainB_len = len(self.trainB_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        trainA_img = self.trainA_images[index % self.trainA_len]
        trainB_img = self.trainB_images[index % self.trainB_len]

        trainA_path = os.path.join(self.root_trainA, trainA_img)
        trainB_path = os.path.join(self.root_trainB, trainB_img)

        trainA_img = np.array(Image.open(trainA_path).convert("RGB"))
        trainB_img = np.array(Image.open(trainB_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=trainA_img, image0=trainB_img)
            trainA_img = augmentations["image"]
            trainB_img = augmentations["image0"]

        return trainA_img, trainB_img


