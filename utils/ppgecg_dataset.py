import os
import glob
import lmdb
import torch
import pickle
import numpy as np
import neurokit2 as nk
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class ECGPPGLMDBDataset(Dataset):
    def __init__(self, dataset_dir: str, split='train', dataset='MCMED'):
        self.dataset = dataset
        self.all_subject_data_dirs = glob.glob(f'{os.path.join(dataset_dir, split)}/*.lmdb')
        

    def __getitem__(self, index):
        files, ppg_data, ecg_data = self.load_lmdb_data(self.all_subject_data_dirs[index])
        return files, torch.tensor(ppg_data, dtype=torch.float32), torch.tensor(ecg_data, dtype=torch.float32)
    

    def __len__(self):
        return len(self.all_subject_data_dirs)

    @staticmethod
    def load_lmdb_data(lmdb_path: str, max_readers: int = 16):
        env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            max_readers=max_readers,
        )
        with env.begin() as txn:
            data = [pickle.loads(value) for _, value in txn.cursor()]

        file = np.stack([row[0] for row in data], axis=0)
        ppg_data = np.stack([row[1] for row in data], axis=0)
        ecg_data = np.stack([row[2] for row in data], axis=0)
        return file, ppg_data, ecg_data

    @staticmethod
    def collate_fn(batch):
        files = [item[0] for item in batch]
        ppg_data = [item[1] for item in batch]
        ecg_data = [item[2] for item in batch]
        ppg_data = torch.cat(ppg_data, dim=0)
        ecg_data = torch.cat(ecg_data, dim=0)
        return files, ppg_data, ecg_data