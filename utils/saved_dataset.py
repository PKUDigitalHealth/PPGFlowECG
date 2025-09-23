import numpy as np
import torch
from torch.utils.data import Dataset

class SavedDataset(Dataset):
    def __init__(self, data_dir, split='test'):
        self.ppg_data = np.load(f"{data_dir}/{split}_ppg.npy")
        self.ecg_data = np.load(f"{data_dir}/{split}_ecg.npy")
        self.labels = np.load(f"{data_dir}/{split}_labels.npy")
        
    def __len__(self):
        return len(self.ecg_data)
    
    def __getitem__(self, idx):
        ppg = torch.FloatTensor(self.ppg_data[idx])
        ecg = torch.FloatTensor(self.ecg_data[idx])
        label = torch.FloatTensor(self.labels[idx])
        
        return ppg, ecg, label
