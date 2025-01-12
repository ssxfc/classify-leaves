import os

import torch
from torchvision import transforms

import uuid
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


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


def axis_plot(title, x: dict, y: dict, save=True, save_dir='tmp'):
    r"""二维坐标图，可用户画dice,loss,precision,recall等随epoch的变化曲线.

    Args:
        title: 二维坐标图名称
        x: x坐标集，字典类型，{'name':'axis name', 'list': [...]}
        y: y坐标集，字典类型，{'name':'axis name', 'list': [...]}
        save: 是否保存图像
    """
    fig, axis = plt.subplots()
    axis.set_title(title)
    axis.set_xlabel(x['name'])
    axis.set_ylabel(y['name'])
    axis.plot(x['list'], y['list'])
    # 重新计算坐标轴范围限制
    axis.relim()
    # 根据新的范围更新视图
    axis.autoscale_view()
    if save:
        plt.savefig(os.path.join(save_dir, f'{title.replace(" ", "_")}.png'))
    else:
        plt.show()


def draw(images, gt_label_list, pred_label_list, rows=3, cols=3, dest="./tmp"):
    r"""图像分类画图"""
    batch_size = images.size(0)
    scale = rows * cols
    fig, ax = plt.subplots(nrows=rows, ncols=cols)
    for i in range(batch_size):
        if batch_size == 1:
            ax = [ax,]
        if i == scale:
            break
        ax[i // cols, i % 3].set_title(f'GT:{gt_label_list[i]}\PR:{pred_label_list[i]}', fontsize=8)
        ax[i // cols, i % 3].imshow(images[i].permute(1, 2, 0).numpy())
        ax[i // cols, i % 3].set_xticks([]), ax[i // cols, i % 3].set_yticks([])
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(os.path.join(dest, f'{uuid.uuid1()}.png'))
    plt.close()
