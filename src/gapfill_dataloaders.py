import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

DEFAULT_X_PATH = "data/X_gap_v2.npy"
DEFAULT_Y_PATH = "data/Y_gap_v2.npy"

class GapDataset(Dataset):
    def __init__(self, x_path=DEFAULT_X_PATH, y_path=DEFAULT_Y_PATH):
        self.X = np.load(x_path)
        self.Y = np.load(y_path)
        assert self.X.shape[0] == self.Y.shape[0]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.Y[idx], dtype=torch.float32)
        return x, y


def create_dataloaders(x_path=DEFAULT_X_PATH, y_path=DEFAULT_Y_PATH, batch_size=256, val_ratio=0.05):
    dataset = GapDataset(x_path, y_path)

    N = len(dataset)
    val_size = int(N * val_ratio)
    train_size = N - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
