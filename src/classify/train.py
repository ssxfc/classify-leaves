import sys
sys.path.append(r"/home/dcd/zww/repos/classify-leaves/src")
import torch
import torch.nn as nn

from tqdm import tqdm

import classify.dataset as ds
from classify.model import ResNet50


root_dir = r"/home/dcd/zww/data/classify-leaves"
fp = r"/home/dcd/zww/data/classify-leaves/train_splited.csv"
train_set = ds.LeafDataset(root_dir=root_dir, fp=fp)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=False, drop_last=True)


import torch
# define some core components
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
loss_fn = nn.CrossEntropyLoss().to(device)

model = ResNet50().to(device)

epochs = 200
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
for epoch in range(1, epochs + 1):
    epoch_loss = 0
    model.train()  # 该语句用来启动网络模型中的BN操作
    with tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
        for image, mask in train_loader:
            image, mask = image.to(device, dtype=torch.float32), mask.to(device, dtype=torch.long)
            optimizer.zero_grad()
            logits = model(image)
            loss = loss_fn(logits, mask)
            loss.backward()
            optimizer.step()
            # logit
            epoch_loss += loss
            pbar.update(image.shape[0])
    scheduler.step()
    print(epoch_loss)
    torch.save(model.state_dict(), f'mdl/epoch_{epoch}_model.pth')
