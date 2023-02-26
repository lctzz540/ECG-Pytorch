import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform

        # Load data from ARFF file
        with open(self.file_path, "r") as f:
            lines = f.readlines()

        # Extract signal and label data from file
        signal_data = []
        label_data = []
        for line in lines:
            if line.startswith("@attribute signal"):
                signal_names = line.split()[2:]
            elif line.startswith("@attribute target"):
                label_names = line.split()[2:]
            elif line.startswith("@data"):
                break
        for line in lines:
            if line.startswith("{") and line.endswith("}\n"):
                values = line[1:-2].split(",")
                signal = [float(value.strip()) for value in values[:-1]]
                label = [int(value.strip()) for value in values[-1].split()]
                signal_data.append(signal)
                label_data.append(label)

        # Convert data to numpy arrays
        self.signal_data = np.array(signal_data)
        self.label_data = np.array(label_data)

    def __len__(self):
        return len(self.signal_data)

    def __getitem__(self, idx):
        signal = self.signal_data[idx]
        label = self.label_data[idx]
        if self.transform:
            signal = self.transform(signal)
        return signal, label
