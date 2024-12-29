import sys
sys.path.append(r"/home/dcd/zww/repos/classify-leaves/src")
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import classify.dataset as ds
from classify.model import ResNet50


def val(model: nn.Module, val_loader, device):
    model.eval()
    num_val_batches = len(val_loader)
    # iterate over the validation set
    cnt = 0
    for x, y in tqdm(val_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, true_mask = x, y
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        true_mask = true_mask.to(device=device, dtype=torch.long)
        # predict the mask
        logits = model(image)
        soft = nn.Softmax(dim=1)
        output = soft(logits)
        x = torch.argmax(output, dim=1)
        true_num = (x == true_mask).sum()
        cnt += true_num
    print(cnt / num_val_batches)

if __name__ == "__main__":
    root_dir = r"/home/dcd/zww/data/classify-leaves"
    fp = r"/home/dcd/zww/data/classify-leaves/vaild_splited.csv"
    val_set = ds.LeafDataset(root_dir=root_dir, fp=fp, mode='valid_img')
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, drop_last=True)


    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet50()
    state_dict = torch.load(r"mdl/epoch_99_model.pth")
    model.load_state_dict(state_dict)
    model = model.to(device=device)

    val(model, val_loader, device)
