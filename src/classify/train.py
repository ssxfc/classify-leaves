import sys
sys.path.append(r"/home/dcd/zww/repos/classify-leaves/src")
import torch
import torch.nn as nn
from torchvision import models

from tqdm import tqdm

from classify.dataset import LeafDataset
from classify.utils import axis_plot
from val import val


def train(model, train_dataloader, val_dataloader, device, epochs=200):
    loss_fn = nn.CrossEntropyLoss().to(device)
    model = model.to(device)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    train_loss_list = []
    val_loss_list = []
    acc_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.1)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        with tqdm(total=len(train_dataloader.dataset), desc=f'MODEL-TRAIN {epoch}/{epochs}', unit='img') as pbar:
            for image, mask in train_dataloader:
                image, mask = image.to(device, dtype=torch.float32), mask.to(device, dtype=torch.long)
                optimizer.zero_grad()
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=True):
                    logits = model(image)
                    loss = loss_fn(logits, mask)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                train_loss += loss
                pbar.update(image.shape[0])
        scheduler.step()
        torch.save(model.state_dict(), f'mdl/epoch_{epoch}_model.pth')
        # 训练信息统计
        train_loss_list.append(train_loss.item())
        axis_plot("train loss",
                    {"name": "epoch",
                        "list":[i for i in range(epoch)]},
                    {"name": "loss",
                        "list":train_loss_list},
                    True)
        # 验证信息统计
        val_loss, acc = val(model, val_dataloader, device)
        val_loss_list.append(val_loss)
        acc_list.append(acc)
        axis_plot("val loss",
                    {"name": "epoch",
                        "list":[i for i in range(epoch)]},
                    {"name": "loss",
                        "list":val_loss_list},
                    True)
        axis_plot("val acc",
                    {"name": "epoch",
                        "list":[i for i in range(epoch)]},
                    {"name": "acc",
                        "list":acc_list},
                    True)
    with open("loss.txt", 'w') as f:
        for j in train_loss_list:
            f.write(str(j) + "\n")


if __name__ == "__main__":
    device = torch.device(f"cuda:2" if torch.cuda.is_available() else "cpu")
    root_dir = r"/home/dcd/zww/data/classify-leaves"
    train_file = r"tmp_train.csv"
    val_file = "tmp_val.csv"
    train_set = LeafDataset(root_dir=root_dir, filename=train_file)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=False, drop_last=True)

    val_set = LeafDataset(root_dir=root_dir, filename=val_file)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=False, drop_last=True)

    model = models.resnext50_32x4d(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 176)
    nn.init.xavier_uniform_(model.fc.weight)

    train(model, train_loader, val_loader, device)
