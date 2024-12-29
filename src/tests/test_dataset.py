import unittest

import torch.utils
import torch.utils.data
import torch

import classify.dataset as ds


class TestDataset(unittest.TestCase):
    def test_dataset(self):
        root_dir = r"D:\datasets\classify-leaves"
        fp = r"D:\datasets\classify-leaves\train_splited.csv"
        train_set = ds.LeafDataset(root_dir=root_dir, fp=fp)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=False, drop_last=True)
        for img, label in train_loader:
            print(img, label)
