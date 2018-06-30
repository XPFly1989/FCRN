import os
import numpy as np
import h5py
from PIL import Image
import torch
import torch.utils.data as data
import flow_transforms
import torchvision.transforms as transforms


class NyuDepthLoader(data.Dataset):
    def __init__(self, data_path, lists):
        self.data_path = data_path
        self.lists = lists

        self.nyu = h5py.File(self.data_path)

        self.imgs = self.nyu['images']
        self.dpts = self.nyu['depths']

    def __getitem__(self, index):
        img_idx = self.lists[index]
        img = self.imgs[img_idx].transpose(2, 1, 0)
        #img = self.imgs[img_idx]
        dpt = self.dpts[img_idx].transpose(1, 0)
        #dpt = self.dpts[img_idx]

        #image = Image.fromarray(np.uint8(img))
        #depth = Image.fromarray(np.uint8(dpt))

        #image.save('img1.png')

        #input_transform = transforms.Compose([flow_transforms.Scale(228)])
        #input_transform = transforms.Compose([flow_transforms.ArrayToTensor()])
        input_transform = transforms.Compose([flow_transforms.Scale(228),
                                              flow_transforms.ArrayToTensor()])
        #target_depth_transform = transforms.Compose([flow_transforms.Scale(228)])
        #target_depth_transform = transforms.Compose([flow_transforms.ArrayToTensor()])
        target_depth_transform = transforms.Compose([flow_transforms.Scale_Single(228),
                                                     flow_transforms.ArrayToTensor()])

        img = input_transform(img)
        dpt = target_depth_transform(dpt)

        #image = Image.fromarray(np.uint8(img))
        #image.save('img2.png')

        return img, dpt

    def __len__(self):
        return len(self.lists)
