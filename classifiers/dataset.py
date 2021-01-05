import h5py
import json
import numpy as np
import os
import torch
import pdb
from torch.utils.data import Dataset


class FICDataset(Dataset):
    """
    A Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES' + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Load categories
        with open(os.path.join(data_folder, self.split + '_CATES' + '.json'), 'r') as f:
            self.cates = json.load(f)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.cates)

    def __getitem__(self, i):
        # the Nth caption
        img = torch.FloatTensor(self.imgs[i] / 255.)
        if self.transform is not None:
            img = self.transform(img)
        if type(self.cates[i]) == list:
            cate = torch.tensor(self.cates[i][0], dtype=torch.long)
        else:
            cate = torch.tensor(self.cates[i], dtype=torch.long)

        return img, cate

    def __len__(self):
        return self.dataset_size
