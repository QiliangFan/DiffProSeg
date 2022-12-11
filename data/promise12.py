"""
The Processed Promise12 datasets utilities.
""" 
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from typing import List
from glob import glob
import os
import SimpleITK as sitk
import torch


class Promise12Dataset(LightningDataModule):

    def __init__(self, data_root: str, fold_idx: int):
        super().__init__()

        idxs = range(50)
        self.imgs = [os.path.join(data_root, f"{idx}.mhd") for idx in idxs]
        self.labels = [os.path.join(data_root, f"{idx}_seg.mhd") for idx in idxs]
        
        fold_lower_bound = fold_idx * 10
        fold_upper_bound = (fold_idx + 1) * 10

        self.train_imgs = [img for i, img in enumerate(self.imgs) if not fold_lower_bound <= i < fold_upper_bound]
        self.train_labels = [label for i, label in enumerate(self.labels) if not fold_lower_bound <= i < fold_upper_bound]

        self.test_imgs = [img for i, img in enumerate(self.imgs) if fold_lower_bound <= i < fold_upper_bound]
        self.test_labels = [label for i, label in enumerate(self.labels) if fold_lower_bound <= i < fold_upper_bound]

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        print(f"Current Stage: {stage}")
        if stage == "fit":
            self.train_data = Promise12(self.train_imgs, self.train_labels)
        else:
            self.test_data = Promise12(self.test_imgs, self.test_labels)

    def train_dataloader(self):
        train_data = DataLoader(self.train_data, batch_size=1, shuffle=True, num_workers=4)
        return train_data

    def test_dataloader(self):
        test_data = DataLoader(self.test_data, batch_size=1, shuffle=False, num_workers=4)
        return test_data

    # def val_dataloader(self, stage: str):
    #     pass

class Promise12(Dataset):

    def __init__(self, imgs: List[str], labels: List[str] = None):
        super().__init__()
        self.imgs = imgs
        self.labels = labels

    def __getitem__(self, idx: int):
        img = sitk.GetArrayFromImage(sitk.ReadImage(self.imgs[idx]))
        img = torch.as_tensor(img)[None, ...]

        if self.labels is not None:
            label = sitk.GetArrayFromImage(sitk.ReadImage(self.labels[idx]))
            label = torch.as_tensor(label)[None, ...]
        else:
            label = None

        return img, label

    def __len__(self):
        return len(self.imgs)