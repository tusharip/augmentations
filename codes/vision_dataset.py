import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from vision import show

class Custom_dataset(Dataset):
    def __init__(self, dir_path):
        super().__init__()
        self.all_paths = glob.glob(dir_path+"/*.jpg")
        self.size      = [500, 500]
        self.transforms = T.Compose([
                        T.ToTensor(),
                        T.Resize(self.size),
                        T.CenterCrop(500),
                        # T.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
                        T.RandomApply([T.Grayscale(num_output_channels=3),
                                        T.RandomPerspective(distortion_scale=0.6, p=1.0),
                                        ],p=0.3),
                        # T.Pad(padding=100, fill=0, padding_mode='constant'),
                        T.RandomResizedCrop(size=self.size, scale=(0.08, 1.0) ),
                        T.RandomHorizontalFlip(p=0.3),
                        T.RandomVerticalFlip(p=0.3),
                        T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                        T.RandomRotation(degrees=(0, 180)),
                        T.Normalize(mean=[0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
                                ])

    def __len__(self, ):
        return len(self.all_paths)

    def __getitem__(self, ind):
        img = cv2.imread(self.all_paths[ind])[..., ::-1 ].copy()
        if self.transforms:
            img = self.transforms(img)
        return img



if __name__ == "__main__":
    dataset = Custom_dataset("../data")
    print(len(dataset))
    show("hii", dataset[1])


