from torchvision import transforms

import torch
import numpy as np
from PIL import Image


# 数据增强和预处理
train_augs = transforms.Compose([
        transforms.RandomRotation((-180, 180)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        
])

vaild_augs = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def pre_process(image_path, form='RGB', classify='img'):
        img = Image.open(image_path).convert(form)
        np_img = np.array(img)
        if classify == 'img':
            np_img = np_img.transpose((2, 0, 1))
            if (np_img > 1).any():
                np_img = (np_img / 255.0).astype(np.float32)
        else:
            np_img = np_img[np.newaxis, :].astype(np.int64)
        tensor_np_img = torch.as_tensor(np_img)
        return tensor_np_img
