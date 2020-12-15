"""
This file contains a PyTorch dataloader for CMP Facade Dataset.
It only supports the datasets from the link provided.
You can learn how to write your own custom dataloader on:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

To Download Datasets : https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/


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
    facades:
        train
            image1
            image2
            image3
        val
            image1
            image2
            image3
        test
            image1
            image2
            image3
"""

class dataset(Dataset):
    def __init__(self,
                 root_dir = "",
                 dataset ="facades",
                 mode ='train',
                 direction='BtoA',
                 transform = None
                 ):
        self.root_dir = root_dir
        self.mode = mode
        self.direction = direction
        self.dataset = dataset
        self.transform = transforms.Compose(transform)

        self.load_data()

    def load_data(self):
        self._files = []
        "Join path of dataset and train/val/test into root"
        _file_path = os.path.join(self.root_dir, self.dataset)
        _file_path_mode = os.path.join(_file_path, self.mode)

        "Collect Files"

        for _file in os.listdir(_file_path_mode):
            self._files.append(_file_path_mode+"\\"+_file)
        self._files = sorted(self._files)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, item):
        file = self._files[item]
        sample = {}
        image = Image.open(file)
        w, h = image.size

        ".crop -> [left, upper, right, lower] tuple"
        "AtoB -> RGB to Semantic :::: BtoA -> Semantic to RGB"
        if self.direction == 'AtoB':
            left = image.crop((0, 0, w/2, h))
            right = image.crop((w/2, 0, w, h))
            sample['input'] = self.transform(left)
            sample['target'] = self.transform(right)
        elif self.direction == 'BtoA':
            left = image.crop((0, 0, w / 2, h))
            right = image.crop((w / 2, 0, w, h))
            sample['input'] = self.transform(right)
            sample['target'] = self.transform(left)

        return sample


# "Here I test the dataset to make sure everything is loading correctly"
# if __name__ == '__main__':
#     dataset = dataset(transform = [
#     transforms.Resize((256, 256), Image.BICUBIC),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ]
# )
#     trainloader = DataLoader(dataset,
#                              batch_size=1,
#                              shuffle=False)
#     for i,batch in enumerate(trainloader):
#         input = batch['input'].cpu().numpy()
#         input = input.squeeze().transpose((1,2,0))
#         fig = plt.figure()
#         plt.imshow(input)
#         plt.pause(0.001)
