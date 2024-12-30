import unittest

import torch.utils
import torch.utils.data
import torch

from tqdm import tqdm

import classify.dataset as ds


class TestDataset(unittest.TestCase):
    def test_dataset(self):
        root_dir = r"D:\datasets\classify-leaves"
        fp = "tmp_test.csv"
        train_set = ds.LeafDataset(root_dir=root_dir, filename=fp)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=False, drop_last=True)
        for img, label in tqdm(train_loader, ):
            pass
