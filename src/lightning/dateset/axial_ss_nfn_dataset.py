# datasets/axial_ss_nfn_dataset.py

import torch
from torch.utils.data import Dataset
import cv2

class AxialSSNFNDataset(Dataset):
    def __init__(self, df, transforms=None, cfg=None, phase="train", **kwargs):
        self.df = df
        self.transforms = transforms
        self.cfg = cfg
        self.phase = phase

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        label = torch.tensor([
            int(row["nfn_class"]),
            int(row["ss_class"])
        ], dtype=torch.long)

        return image, label

    def __len__(self):
        return len(self.df)
