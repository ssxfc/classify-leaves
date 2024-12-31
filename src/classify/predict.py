import sys
import logging
sys.path.append(r"/home/dcd/zww/repos/classify-leaves/src")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from tqdm import tqdm

import classify.dataset as ds

logger = logging.getLogger(__name__)


def infer(model: nn.Module, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, total=len(test_loader), desc='MODEL-TEST', unit='batch', leave=False):
            images, labels = x, y
            # move images and labels to correct device and type
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            # predict the label
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=True):
                logits = model(images)
            # 推理
            pred = F.softmax(logits, dim=1).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.shape[0]
    accuracy = correct / total
    logger.info(f"Test Accuracy: {accuracy}")


if __name__ == "__main__":
    root_dir = r"/home/dcd/zww/data/classify-leaves"
    fp = r"tmp_test.csv"
    test_set = ds.LeafDataset(root_dir=root_dir, filename=fp)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=2, shuffle=False, drop_last=True)


    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = models.resnext50_32x4d(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 176)
    state_dict = torch.load("./mdl/epoch_99_model.pth")
    model.load_state_dict(state_dict)
    model = model.to(device=device)

    infer(model, test_loader, device)
