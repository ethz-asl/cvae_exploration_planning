import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        'Initialization'
        self.len = len(data)
        self.list_IDs = np.arange(0,len(data))
        self.data = data  # npy data

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, item):
        'Generate one sample of data'
        ID = self.list_IDs[item]
        x = self.data[ID]
        # sample = torch.tensor(x, dtype=config.dtype, device=config.device)
        return x
