from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import os

from models import *

# -----------------
# image pre-process
# -----------------
"""
10-crop:

    Crop the given PIL Image into four corners and the central crop 
    plus the flipped version of these (horizontal flipping is used by default)

https://pytorch.org/docs/stable/torchvision/transforms.html
"""

class ImageDataset(Dataset):
    def __init__(self, image_dir):
        data_list = []
        for x in os.listdir(image_dir):
            check_suffix = [x.lower().endswith(s) for s in ['.png', '.jpg', '.jpeg']]
            data_list.append(os.path.join(image_dir, x))

        self.data_list = data_list
        self.pipeline = transforms.Compose([
            # transforms.Grayscale(),  # not need ?
            transforms.Resize(48),
            transforms.TenCrop(44),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])
        
    def __getitem__(self, i):
        # get file name and load
        p = self.data_list[i]
        x = Image.open(p)
        x = self.pipeline(x)
        
        # to 3 channel
        if x.shape[1] == 1:
            x = torch.cat([x, x, x], dim=1)
     
        return p.split('/')[-1], x

    def __len__(self):
        return len(self.data_list)
