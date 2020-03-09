"""
This file contains a PyTorch dataloader for CMP Facade Dataset.
You can learn how to write your own custom dataloader on:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

import os
import torch
import numpy as np
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, transform
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

"""
Note: In order to use this file, you need to organise your data in the following structure:
    CMP_Facades:
        rgb
            cmp_b0001
            cmp_b0002
            cmp_b0003
        semantics
            cmp_b0001
            cmp_b0002
            cmp_b0003
"""

class facades(Dataset):
    def __init__(self,
                 root_dir = "C:\\Users\\akbar\\PycharmProjects\\CMP_Facades",
                 img_height = 256,
                 img_width = 256,
                 transform = transforms.ToTensor()
                 ):
        self.root_dir = root_dir
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform
        self.load_data()

    def load_data(self):
        self.rgb_files = []
        self.semantics_files = []
        "Join path of rgb directory and semantic, \\ to complete the path for loop below."
        rgb_path = os.path.join(self.root_dir,'rgb\\')
        semantics_path = os.path.join(self.root_dir,'semantics\\')

        "Collect Files"

        for rgb_file in os.listdir(rgb_path):
            self.rgb_files.append(rgb_path + rgb_file)
        self.rgb_files = sorted(self.rgb_files)

        for semantic_file in os.listdir(semantics_path):
            self.semantics_files.append(semantics_path + semantic_file)
        self.semantics_files = sorted(self.semantics_files)

    def resize_randomcrop(self, rgb, semantic):
        torchvision.transforms.Resize(self.img_width,interpolation=2)(rgb)
        torchvision.transforms.Resize(self.img_width, interpolation=2)(semantic)
        return rgb, semantic

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, item):
        rgb_item = self.rgb_files[item]
        semantic_item = self.semantics_files[item]
        sample = {}

        rgb = Image.open(rgb_item)
        semantic = Image.open(semantic_item)
        sample['rgb'], sample['semantic'] = self.resize_randomcrop(rgb, semantic)
        sample['rgb'] = self.transform(sample['rgb'])
        sample['semantic'] = self.transform(sample['semantic'])

        return sample


"Here I test the dataset to make sure everything is loading correctly"
if __name__ == '__main__':
    dataset = facades()
    trainloader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False)
    for i,batch in enumerate(trainloader):
        rgb = torchvision.utils.make_grid(batch['rgb'])
        rgb = rgb.numpy().transpose((1, 2, 0))
        plt.imshow(rgb.astype('uint8'))
        plt.pause(0.001)
