import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


def val(model: nn.Module, val_loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss().to(device)
    num_val_batches = len(val_loader)
    # iterate over the validation set
    total_loss = 0.0
    total_ok = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(val_loader, total=num_val_batches, desc='MODEL-VAL', unit='batch', leave=False):
            image, true_mask = x, y
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            true_mask = true_mask.to(device=device, dtype=torch.long)
            # predict the mask
            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=True):
                logits = model(image)
                loss = loss_fn(logits, true_mask)
            total_loss += loss.item()
            # 正确率
            pred = F.softmax(logits, dim=1).argmax(dim=1)
            ok_pred = (pred == true_mask).sum().item()
            total_ok += ok_pred
            total += pred.shape[0]
    return total_loss, total_ok * 1.0 / total
