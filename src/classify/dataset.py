import os

import torch
from torch.utils.data import Dataset

import pandas
from PIL import Image

from classify.utils import train_augs, vaild_augs


class LeafDataset(Dataset):
    def __init__(self, root_dir, filename, label_file="label.txt"):
        self.root_dir = root_dir
        self.filename = filename
        self.data = pandas.DataFrame(pandas.read_csv(os.path.join(root_dir, filename)))
        self.data_fetcher = None
        with open(os.path.join(root_dir, label_file), 'r') as f:
            label_list = f.readlines()
        self.idx_2_label = label_list
        self.label_2_idx = {label.replace('\n', '').replace('\r', ''):i for i, label in enumerate(label_list)}
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 确保多次epoch时，能循环使用data_fetcher生成器
        if idx == 0:
            self.data_fetcher = zip(self.data['image'], self.data['label'])
        img, label = next(self.data_fetcher)
        fp = os.path.join(self.root_dir, img)
        return train_augs(Image.open(fp)), torch.as_tensor(self.labels[label])
