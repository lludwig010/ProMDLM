import torch
import h5py
import pandas as pd
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pickle

class CustomDataset(Dataset):
    def __init__(self, dataset_file):

        with open(dataset_file, 'rb') as f:
            self.data = pickle.load(f)

        self.length = len(self.data)
    
    def __len__(self):

        return self.length
    
    def __getitem__(self, idx):

        #convert to tensor
        tensor_tokenized = torch.from_numpy(self.data[idx])

        return tensor_tokenized






