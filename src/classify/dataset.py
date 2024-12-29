import os

import torch
from torch.utils.data import Dataset

import pandas
from PIL import Image

from classify.utils import train_augs


class LeafDataset(Dataset):
    def __init__(self, root_dir, fp, mode='train_img'):
        self.root_dir = root_dir
        self.mode = mode
        self.fp = fp
        self.data = pandas.DataFrame(pandas.read_csv(fp))
        self.data_fetcher = None
        labels = self.data['label'].drop_duplicates().values.tolist()
        self.labels = {label:idx for idx, label in enumerate(labels)}
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 确保多次epoch时，能循环使用data_fetcher生成器
        if idx == 0:
            self.data_fetcher = zip(self.data['image'], self.data['label'])
        img, label = next(self.data_fetcher)
        fp = os.path.join(self.root_dir, self.mode, label, f"{img}.jpg")
        return train_augs(Image.open(fp)), torch.as_tensor(self.labels[label])
