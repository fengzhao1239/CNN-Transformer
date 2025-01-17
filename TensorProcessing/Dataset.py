import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import os
import h5py
from pathlib import Path


class FourWellDataset(Dataset):

    def __init__(self, train_or_val, label_name):
        self.dataset_dir = Path('G:/optim_code/tensordata/dataset_4_wells_new.hdf5')

        if train_or_val == 'train':
            set_name = 'training_set'
        elif train_or_val == 'val':
            set_name = 'validation_set'
        elif train_or_val == 'test':
            set_name = 'test_set'
        else:
            raise ValueError(f"Invalid string: {train_or_val}.")

        f = h5py.File(self.dataset_dir, 'r')
        self.X1 = np.array(f[set_name]['input_features']['static'])
        self.X2 = np.array(f[set_name]['input_features']['dynamic'])
        self.Y = np.array(f[set_name]['output_labels'][label_name])

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, item):
        return (torch.from_numpy(self.X1[item]).float(),
                torch.from_numpy(self.X2[item]).float(),
                torch.from_numpy(self.Y[item]).float())

